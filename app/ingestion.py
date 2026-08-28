import ast
import re
from collections import Counter
from pathlib import Path
from hashlib import sha256
import uuid

from pypdf import PdfReader

try:
    import pdfplumber
except ImportError:  # PDF 表格能力是可选依赖，缺失时仍保留普通 PDF 文本导入。
    pdfplumber = None

from .db import ensure_document, find_reusable_document, insert_chunks
from .config import settings
from .llm import embed

TEXT_EXTENSIONS = {
    ".md", ".txt", ".rst", ".pdf", ".py", ".js", ".jsx", ".ts", ".tsx", ".go",
    ".java", ".kt", ".json", ".yaml", ".yml", ".toml", ".xml", ".sql",
    ".dockerfile", ".gradle", ".properties",
}


def split_text(text: str, size: int = 1400, overlap: int = 200) -> list[str]:
    """按长度兜底切分，并优先在段落、换行和空格边界结束。"""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            # 在窗口末尾向前寻找自然边界，减少段落和语句被硬切开的概率。
            boundary = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end), text.rfind(" ", start, end))
            if boundary > start + size // 2:
                end = boundary
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _heading_path(stack: list[tuple[int, str]]) -> str:
    """把 Markdown 标题栈转换为可读的来源定位。"""
    return " > ".join(title for _, title in stack)


def _is_markdown_table_separator(line: str) -> bool:
    """判断一行是否为 Markdown 表格的分隔行，例如 `| --- | --- |`。"""
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return len(cells) >= 2 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _markdown_cells(line: str) -> list[str]:
    """拆分 Markdown 表格单元格；当前 MVP 约定不处理单元格内未转义的竖线。"""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _normalize_markdown_table(lines: list[str], size: int, overlap: int) -> list[str]:
    """把表格行转换为带列名的检索文本，并在分块时重复表头。"""
    if len(lines) < 3:
        return []
    headers = _markdown_cells(lines[0])
    data_lines = lines[2:]
    header_text = "表格列：" + "；".join(headers)
    rows: list[str] = []
    for line in data_lines:
        values = _markdown_cells(line)
        pairs = []
        for index, value in enumerate(values):
            column = headers[index] if index < len(headers) else f"列{index + 1}"
            pairs.append(f"{column}：{value}")
        if pairs:
            rows.append("表格行：" + "；".join(pairs))

    chunks: list[str] = []
    current = header_text
    for row in rows:
        candidate = f"{current}\n{row}"
        if len(candidate) > size and current != header_text:
            chunks.append(current)
            current = f"{header_text}\n{row}"
        else:
            current = candidate
    if current != header_text or not rows:
        chunks.append(current)
    return chunks


def _split_markdown_section(section: str, ref: str, size: int, overlap: int) -> list[tuple[str, str]]:
    """在一个 Markdown 章节内部隔离表格，避免表头和数据行被普通切分打散。"""
    lines = section.splitlines()
    result: list[tuple[str, str]] = []
    prose: list[str] = []
    table_number = 0

    def flush_prose() -> None:
        nonlocal prose
        body = "\n".join(prose).strip()
        if body:
            result.extend((part, ref) for part in split_text(body, size, overlap))
        prose = []

    index = 0
    while index < len(lines):
        if index + 1 < len(lines) and "|" in lines[index] and _is_markdown_table_separator(lines[index + 1]):
            flush_prose()
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            table_number += 1
            table_ref = f"{ref}#table-{table_number}"
            result.extend((chunk, table_ref) for chunk in _normalize_markdown_table(table_lines, size, overlap))
        else:
            prose.append(lines[index])
            index += 1
    flush_prose()
    return result


def split_markdown(text: str, size: int = 1400, overlap: int = 200) -> list[tuple[str, str]]:
    """按 Markdown 标题聚合章节，再对超长章节做长度兜底。"""
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    current: list[str] = []
    current_ref = "document"
    in_fence = False

    def flush() -> None:
        nonlocal current
        body = "\n".join(current).strip()
        if body:
            sections.append((body, current_ref))
        current = []

    for line in lines:
        if line.strip().startswith("```") or line.strip().startswith("~~~"):
            in_fence = not in_fence
        heading = None if in_fence else re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip().rstrip("#").strip()
            stack[:] = [(item_level, item_title) for item_level, item_title in stack if item_level < level]
            stack.append((level, title))
            current_ref = _heading_path(stack)
            current.append(line)
        else:
            current.append(line)
    flush()

    chunks: list[tuple[str, str]] = []
    for section, ref in sections:
        chunks.extend(_split_markdown_section(section, ref, size, overlap))
    return chunks


def _code_chunks_from_ranges(text: str, ranges: list[tuple[str, int, int]], size: int, overlap: int) -> list[tuple[str, str]]:
    """按代码定义的行号切分；超长定义继续使用通用边界切分。"""
    lines = text.splitlines()
    chunks: list[tuple[str, str]] = []
    for name, start_line, end_line in ranges:
        body = "\n".join(lines[start_line - 1:end_line]).strip()
        if not body:
            continue
        for part in split_text(body, size, overlap):
            chunks.append((part, name))
    return chunks


def split_python_code(text: str, size: int = 1400, overlap: int = 200) -> list[tuple[str, str]]:
    """使用 Python AST 按类、方法和函数切分，解析失败时回退到通用切分。"""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [(part, "code") for part in split_text(text, size, overlap)]

    lines = text.splitlines()
    ranges: list[tuple[str, int, int]] = []
    expanded: list[tuple[str, str]] = []
    first_definition_line = len(lines)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", start_line)
        first_definition_line = min(first_definition_line, start_line)
        if isinstance(node, ast.ClassDef) and node.body:
            class_header = "\n".join(lines[start_line - 1:node.body[0].lineno - 1]).strip()
            method_found = False
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_found = True
                    child_end = getattr(child, "end_lineno", child.lineno)
                    prefix = class_header + "\n" if class_header else ""
                    child_body = prefix + "\n".join(lines[child.lineno - 1:child_end]).strip()
                    expanded.extend((part, f"class {node.name}.{child.name}") for part in split_text(child_body, size, overlap))
            if not method_found:
                ranges.append((f"class {node.name}", start_line, end_line))
        else:
            ranges.append((f"function {getattr(node, 'name', 'anonymous')}", start_line, end_line))

    result: list[tuple[str, str]] = []
    result.extend(expanded)
    result.extend(_code_chunks_from_ranges(text, ranges, size, overlap))

    # Imports, constants and top-level statements remain searchable as a preamble chunk.
    preamble = "\n".join(lines[:first_definition_line - 1]).strip()
    if preamble:
        result[0:0] = [(part, "module preamble") for part in split_text(preamble, size, overlap)]
    return result or [(part, "module") for part in split_text(text, size, overlap)]


CODE_DEFINITION_PATTERNS = {
    ".js": r"^\s*(?:export\s+)?(?:async\s+)?function\s+([\w$]+)|^\s*(?:export\s+)?class\s+([\w$]+)",
    ".jsx": r"^\s*(?:export\s+)?(?:async\s+)?function\s+([\w$]+)|^\s*(?:export\s+)?class\s+([\w$]+)",
    ".ts": r"^\s*(?:export\s+)?(?:async\s+)?function\s+([\w$]+)|^\s*(?:export\s+)?class\s+([\w$]+)",
    ".tsx": r"^\s*(?:export\s+)?(?:async\s+)?function\s+([\w$]+)|^\s*(?:export\s+)?class\s+([\w$]+)",
    ".java": r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:class|interface|enum)\s+(\w+)",
    ".go": r"^\s*func\s+(?:\([^)]*\)\s*)?(\w+)",
}


def split_code(text: str, suffix: str, size: int = 1400, overlap: int = 200) -> list[tuple[str, str]]:
    """代码优先按结构定义切分；当前先对 Python 使用 AST，其他语言使用定义边界兜底。"""
    if suffix == ".py":
        return split_python_code(text, size, overlap)
    pattern = CODE_DEFINITION_PATTERNS.get(suffix)
    if not pattern:
        return [(part, "code") for part in split_text(text, size, overlap)]
    lines = text.splitlines()
    starts: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        match = re.search(pattern, line)
        if match:
            name = next((group for group in match.groups() if group), "anonymous")
            starts.append((name, index))
    if not starts:
        return [(part, "code") for part in split_text(text, size, overlap)]
    ranges: list[tuple[str, int, int]] = []
    if starts[0][1] > 0:
        ranges.append(("module preamble", 1, starts[0][1]))
    for index, (name, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(lines)
        ranges.append((name, start + 1, end))
    return _code_chunks_from_ranges(text, ranges, size, overlap)


def read_file(path: Path) -> str:
    """读取 PDF 或普通文本类文件，统一转换成字符串。"""
    if path.suffix.lower() == ".pdf":
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _normalize_pdf_table(rows: list[list[str | None]], size: int, overlap: int) -> list[str]:
    """把 PDF 表格转换为带列名的文本；每个长度子块都重复表头。"""
    cleaned = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if len(cleaned) < 2:
        return []
    headers = cleaned[0]
    header_text = "表格列：" + "；".join(value or f"列{index + 1}" for index, value in enumerate(headers))
    chunks: list[str] = []
    current = header_text
    for values in cleaned[1:]:
        pairs = []
        for index, value in enumerate(values):
            column = headers[index] if index < len(headers) and headers[index] else f"列{index + 1}"
            pairs.append(f"{column}：{value}")
        row_text = "表格行：" + "；".join(pairs)
        candidate = f"{current}\n{row_text}"
        if len(candidate) > size and current != header_text:
            chunks.append(current)
            current = f"{header_text}\n{row_text}"
        else:
            current = candidate
    if current != header_text:
        chunks.append(current)
    return chunks


def _normalize_pdf_line(text: str) -> str:
    """把 PDF 行归一化为可跨页比较的指纹，降低空格、大小写和页码差异的影响。"""
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    normalized = re.sub(r"(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?", "#", normalized)
    normalized = re.sub(r"第\s*#\s*页(?:\s*/\s*#)?", "#", normalized)
    return normalized


def _is_standalone_page_number(text: str) -> bool:
    """识别独立页码，不删除正文中的版本号、年份或带数字的业务内容。"""
    compact = re.sub(r"\s+", "", text).casefold()
    return bool(
        re.fullmatch(r"\d+", compact)
        or re.fullmatch(r"page\d+(?:of\d+|/\d+)?", compact)
        or re.fullmatch(r"第\d+页(?:/\d+)?", compact)
    )


def _pdf_lines(page) -> list[dict]:
    """将 pdfplumber 的 words 聚合成带坐标的文本行，供清洗和双栏排序使用。"""
    page_width = float(page.width)
    words = page.extract_words(
        x_tolerance=2,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=False,
    ) or []
    lines: list[dict] = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        # 同一基线上的左右栏文字可能被 PDF 提取器交错返回；横向间隔明显大于
        # 普通词间距时拆成两个 line，后续才能判断页面是否为双栏。
        same_line = lines and abs(word["top"] - lines[-1]["top"]) <= 3
        large_horizontal_gap = lines and word["x0"] - lines[-1]["x1"] > page_width * 0.12
        if not same_line or large_horizontal_gap:
            lines.append(
                {
                    "top": word["top"],
                    "bottom": word["bottom"],
                    "x0": word["x0"],
                    "x1": word["x1"],
                    "words": [word],
                }
            )
        else:
            line = lines[-1]
            line["bottom"] = max(line["bottom"], word["bottom"])
            line["x0"] = min(line["x0"], word["x0"])
            line["x1"] = max(line["x1"], word["x1"])
            line["words"].append(word)

    for line in lines:
        line["words"].sort(key=lambda item: item["x0"])
        line["text"] = " ".join(item["text"] for item in line["words"]).strip()
        line.pop("words")
    return lines


def _repeated_pdf_decoration_fingerprints(page_lines: list[list[dict]], page_height: float) -> set[str]:
    """从多页的顶部/底部区域找重复页眉页脚，避免误删正文中的同名短语。"""
    fingerprints_by_page: list[set[str]] = []
    occurrences: Counter[str] = Counter()
    for lines in page_lines:
        page_fingerprints: set[str] = set()
        for line in lines:
            in_margin = line["top"] <= page_height * 0.14 or line["bottom"] >= page_height * 0.86
            fingerprint = _normalize_pdf_line(line["text"])
            if in_margin and len(fingerprint) >= 3:
                page_fingerprints.add(fingerprint)
        fingerprints_by_page.append(page_fingerprints)
        occurrences.update(page_fingerprints)

    # 至少跨两页重复，并且覆盖半数页面，才认为是模板页眉/页脚。
    minimum_pages = max(2, (len(page_lines) + 1) // 2)
    return {fingerprint for fingerprint, count in occurrences.items() if count >= minimum_pages}


def _looks_like_two_columns(lines: list[dict], page_width: float) -> bool:
    """用文字块的中线分布做保守双栏判断，单栏页面不满足时保持原始顺序。"""
    if len(lines) < 8:
        return False
    midpoint = page_width / 2
    left = [line for line in lines if line["x1"] < midpoint - page_width * 0.04]
    right = [line for line in lines if line["x0"] > midpoint + page_width * 0.04]
    spanning = [line for line in lines if line["x0"] < midpoint < line["x1"]]
    return len(left) >= 3 and len(right) >= 3 and len(spanning) <= max(1, len(lines) // 8)


def _extract_clean_pdf_page_text(page, removable_fingerprints: set[str], page_number: int) -> str:
    """清洗单页文本并恢复基础双栏顺序；页码和装饰文本只影响正文，不影响 page-N 引用。"""
    lines = _pdf_lines(page)
    kept: list[dict] = []
    for line in lines:
        text = line["text"]
        fingerprint = _normalize_pdf_line(text)
        if _is_standalone_page_number(text) or fingerprint in removable_fingerprints:
            continue
        kept.append(line)

    if _looks_like_two_columns(kept, float(page.width)):
        midpoint = float(page.width) / 2
        left = [line for line in kept if line["x1"] < midpoint]
        right = [line for line in kept if line["x0"] >= midpoint]
        ordered = sorted(left, key=lambda line: (line["top"], line["x0"]))
        ordered.extend(sorted(right, key=lambda line: (line["top"], line["x0"])))
    else:
        ordered = sorted(kept, key=lambda line: (line["top"], line["x0"]))

    # 保留页面边界，便于后续 source_ref 继续显示 page-N；这里不把页码写回证据文本。
    return "\n".join(line["text"] for line in ordered if line["text"]).strip()


def split_pdf(path: Path, size: int = 1400, overlap: int = 200) -> list[tuple[str, str]]:
    """按页清洗 PDF 文本、恢复基础双栏顺序，并额外提取可识别的表格块。"""
    chunks: list[tuple[str, str]] = []
    if pdfplumber is None:
        # 可选依赖不可用时，保持原有 PDF 导入能力。
        for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
            page_text = page.extract_text(extraction_mode="layout") or ""
            chunks.extend((part, f"page-{page_number}") for part in split_text(page_text, size, overlap))
        return chunks

    with pdfplumber.open(path) as pdf:
        # 先读取所有页面的坐标行，再识别跨页重复装饰，避免只看单页导致误删正文。
        page_lines = [_pdf_lines(page) for page in pdf.pages]
        page_height = max((float(page.height) for page in pdf.pages), default=0.0)
        removable_fingerprints = _repeated_pdf_decoration_fingerprints(page_lines, page_height)
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = _extract_clean_pdf_page_text(page, removable_fingerprints, page_number)
            chunks.extend((part, f"page-{page_number}") for part in split_text(page_text, size, overlap))
            for table_number, table in enumerate(page.extract_tables() or [], start=1):
                table_chunks = _normalize_pdf_table(table, size, overlap)
                table_ref = f"page-{page_number}#table-{table_number}"
                chunks.extend((part, table_ref) for part in table_chunks)
    return chunks


def split_document(path: Path, content: str, size: int = 1400, overlap: int = 200) -> list[tuple[str, str]]:
    """根据文件类型选择结构化切分器，并统一返回文本和来源定位。"""
    suffix = path.suffix.lower()
    if suffix == ".md":
        return split_markdown(content, size, overlap)
    if suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java"}:
        return split_code(content, suffix, size, overlap)
    if suffix == ".pdf":
        return split_pdf(path, size, overlap)
    return [(part, "document") for part in split_text(content, size, overlap)]


def document_metadata(
    path: Path,
    source_type: str,
    scope_type: str = "personal",
    department_id: str | None = None,
    project_name: str | None = None,
    workspace_name: str | None = None,
) -> dict:
    """生成文档级元数据；标题取文件名，作者暂时固定为 User。"""
    digest = sha256(path.read_bytes()).hexdigest()
    return {
        "source_type": source_type,
        "title": path.name,
        "source_name": path.name,
        "source_path": str(path),
        "project_name": project_name if source_type == "wiki" else None,
        "workspace_name": (workspace_name or settings.current_workspace_name) if scope_type == "workspace" else None,
        "author": "User",
        "scope_type": scope_type,
        "owner_user_id": settings.current_user_id if scope_type == "personal" else None,
        "owner_department_id": department_id if scope_type == "department" else None,
        "content_hash": digest,
    }


def build_rows(path: Path, source_type: str) -> list[dict]:
    """先生成结构化父块，再切出子块；只有子块调用 Embedding API。"""
    content = read_file(path)
    # 父块按较大的结构单元生成；子块使用更短窗口承担向量召回。
    structured_units = split_document(path, content, settings.parent_chunk_size, 0)
    if not structured_units:
        return []

    suffix = path.suffix.lower()
    language = {
        ".md": "markdown", ".pdf": "pdf", ".py": "python", ".js": "javascript",
        ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript", ".go": "go",
        ".java": "java",
    }.get(suffix)
    rows: list[dict] = []
    for parent_index, (parent_text, ref) in enumerate(structured_units):
        # split_document 已按标题、页码、表格或代码定义形成结构单元；每个单元是一个可解释父块。
        parent_id = str(uuid.uuid4())
        element_type = "table" if "#table-" in ref else "page_text" if ref.startswith("page-") else "code" if suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java"} else "section"
        page_match = re.search(r"(?:^|#)page-(\d+)", ref)
        symbol = ref if element_type == "code" and ref not in {"code", "module", "module preamble"} else None
        section_path = ref if suffix == ".md" and not ref.startswith("document") else None
        metadata = {
            "parser": "structure-first",
            "parser_version": "1",
            "source_ref": ref,
        }
        rows.append(
            {
                "chunk_id": parent_id,
                "parent_key": parent_id,
                "chunk_level": "parent",
                "source_type": source_type,
                "source_name": path.name,
                "source_path": str(path),
                "source_ref": f"{path.name}#{ref}#parent",
                "chunk_index": parent_index,
                "content": parent_text,
                "element_type": element_type,
                "section_path": section_path,
                "page_number": int(page_match.group(1)) if page_match else None,
                "language": language,
                "code_symbol": symbol,
                "metadata": metadata,
            }
        )
        child_texts = split_text(parent_text, settings.child_chunk_size, settings.child_chunk_overlap)
        for child_index, child_text in enumerate(child_texts):
            rows.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "parent_key": parent_id,
                    "chunk_level": "child",
                    "source_type": source_type,
                    "source_name": path.name,
                    "source_path": str(path),
                    "source_ref": f"{path.name}#{ref}#child-{child_index + 1}",
                    "chunk_index": parent_index,
                    "child_index": child_index,
                    "content": child_text,
                    "embedding": embed(child_text),
                    "element_type": element_type,
                    "section_path": section_path,
                    "page_number": int(page_match.group(1)) if page_match else None,
                    "language": language,
                    "code_symbol": symbol,
                    "metadata": metadata,
                }
            )
    return rows


def ingest_file(
    path: Path,
    source_type: str,
    scope_type: str = "personal",
    department_id: str | None = None,
    project_name: str | None = None,
    workspace_name: str | None = None,
    **metadata_overrides: str | None,
) -> tuple[str, int]:
    """完成单文件导入：先向量化，成功后再提交文档元数据和分块。"""
    metadata = document_metadata(path, source_type, scope_type, department_id, project_name, workspace_name)
    reusable_id = find_reusable_document(metadata)
    if reusable_id:
        return reusable_id, 0

    # 外部 Embedding 失败时，此时数据库还没有新增或更新文档，避免出现半成功记录。
    rows = build_rows(path, source_type)
    document_id, created, revision_no = ensure_document(metadata)
    if not created:
        return document_id, 0
    count = insert_chunks(rows, document_id)
    return document_id, count


def ingest_path(path: Path, source_type: str) -> int:
    """扫描单个文件或目录中的支持类型文件，并返回新增分块数量。"""
    if path.is_file():
        return ingest_file(path, source_type, "workspace")[1]
    count = 0
    for file_path in path.rglob("*"):
        if file_path.is_file() and (file_path.suffix.lower() in TEXT_EXTENSIONS or file_path.name == "Dockerfile"):
            count += ingest_file(file_path, source_type, "workspace")[1]
    return count
