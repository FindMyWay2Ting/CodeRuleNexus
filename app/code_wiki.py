"""Code Intelligence 的第一阶段实现。

本模块只负责建立“代码事实层”，不让 LLM 直接猜测或修改代码知识：

1. 接收本地项目，或将公开 GitHub 仓库拉取到受控目录后扫描。
2. 扫描项目文件并记录项目、Commit、文件和内容哈希。
3. Python/Go 使用 Tree-sitter 提取结构，Python 在 grammar 缺失时回退到标准库 AST。
4. Go module 使用 SCIP 补充精确定义、引用和实现关系，失败时保留结构扫描结果。
5. 其他语言使用保守的定义边界回退，保证扫描链路可以先跑起来。
6. 从依赖文件、初始化代码和配置文件中识别组件，并保存证据。
7. 将结果写入独立的 Code Wiki 表，和 RAG 的 knowledge_chunks 分开。

Tree-sitter 已作为 Python/Go 的实际解析主路径；SCIP indexer 作为精确语义索引增强路径，通过扫描结果报告可用性，避免外部构建环境阻塞基础事实扫描。
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from .db import connection
from . import scip_pb2

try:
    # grammar 包独立发布，缺包时仍允许旧环境使用 AST/正则回退。
    from tree_sitter import Language, Parser
    import tree_sitter_go
    import tree_sitter_python
except ImportError:  # pragma: no cover - 仅覆盖未安装可选依赖的旧环境
    Language = None
    Parser = None
    tree_sitter_go = None
    tree_sitter_python = None


CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
}

IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".venv", ".codex_tmp", "venv", "env", "node_modules",
    "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build",
    "target", "uploads", ".idea", ".vscode",
}

ID_NAMESPACE = uuid.UUID("2c6b3b0d-0d8b-4a68-b5e5-b0a2f0f48f7e")
DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent / "data" / "code_repositories"

# Wiki 事实层保留可用于架构理解和导航的定义；局部变量仍可作为引用 target_ref，
# 但不单独生成数据库符号，避免大型项目被参数和临时变量淹没。
SCIP_WIKI_KINDS = {
    "class", "constructor", "enum", "field", "function", "interface", "method",
    "module", "namespace", "package", "struct", "trait", "type", "typealias",
}


@dataclass
class SymbolFact:
    """一个可被引用的代码符号，位置始终相对于项目根目录。"""

    path: str
    name: str
    qualified_name: str
    kind: str
    signature: str | None
    start_line: int
    end_line: int
    parent_qualified_name: str | None = None
    docstring: str | None = None
    metadata: dict = field(default_factory=dict)
    symbol_id: str | None = None


@dataclass
class RelationFact:
    """代码事实之间的有向关系；target_ref 保留无法静态解析的目标。"""

    source_symbol_key: str
    relation_type: str
    target_symbol_key: str | None = None
    target_ref: str | None = None
    evidence: dict = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class FileFact:
    path: str
    language: str
    content_hash: str
    line_count: int
    symbols: list[SymbolFact] = field(default_factory=list)
    relations: list[RelationFact] = field(default_factory=list)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _git_commit(root: Path) -> tuple[str, str]:
    """优先记录 Git HEAD；非 Git 目录使用文件哈希生成可复现扫描版本。"""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        commit = result.stdout.strip()
        if commit:
            return commit, "git"
    except (OSError, subprocess.SubprocessError):
        pass
    return "scan-" + _hash_text(str(root.resolve()))[:16], "content_scan"


def normalize_github_url(repository_url: str) -> tuple[str, str, str]:
    """校验公开 GitHub HTTPS 地址，并返回规范 URL、owner 和仓库名。"""
    parsed = urlparse(repository_url.strip())
    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() != "github.com"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("请输入标准 GitHub HTTPS 地址，例如 https://github.com/owner/repository")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) != 2:
        raise ValueError("GitHub 地址必须指向一个仓库，不能是用户页、文件页或分支页")
    owner, repository = parts
    if repository.lower().endswith(".git"):
        repository = repository[:-4]
    if (
        not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", owner)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", repository)
        or repository in {".", ".."}
    ):
        raise ValueError("GitHub owner 或仓库名称格式不正确")
    return f"https://github.com/{owner}/{repository}.git", owner, repository


def normalize_uploaded_path(filename: str) -> tuple[str, PurePosixPath]:
    """校验浏览器目录上传携带的相对路径，禁止绝对路径和目录穿越。"""
    raw = (filename or "").replace("\\", "/").strip()
    path = PurePosixPath(raw)
    parts = path.parts
    if path.is_absolute() or len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("本地项目文件必须包含文件夹相对路径，请重新选择整个项目目录")
    if any(re.search(r'[<>:"|?*\x00-\x1f]', part) for part in parts):
        raise ValueError("本地项目中包含 Windows 不支持的文件名")
    project_name = parts[0]
    if len(project_name) > 100:
        raise ValueError("本地项目文件夹名称不能超过 100 个字符")
    return project_name, PurePosixPath(*parts[1:])


def managed_local_repository_path(project_name: str, repository_root: Path | None = None) -> Path:
    """为浏览器上传项目生成稳定托管目录；同名文件夹会更新同一个项目。"""
    root = (repository_root or DEFAULT_REPOSITORY_ROOT).resolve() / "local"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", project_name).strip("-.")[:48] or "project"
    digest = _hash_text(project_name)[:8]
    return root / f"{slug}-{digest}"


def _run_git(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """运行无交互 Git 命令；避免服务端扫描期间等待凭据输入。"""
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("GitHub 仓库拉取超时，请检查网络或仓库大小") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("未找到 Git，请先安装 Git for Windows") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "Git 命令执行失败").strip().splitlines()[-1]
        raise RuntimeError(f"GitHub 仓库拉取失败：{detail}") from exc


def import_github_repository(repository_url: str, repository_root: Path | None = None) -> dict:
    """将公开 GitHub 仓库克隆到托管目录；已有仓库仅允许 fast-forward 更新。"""
    canonical_url, owner, repository = normalize_github_url(repository_url)
    managed_root = (repository_root or DEFAULT_REPOSITORY_ROOT).resolve()
    managed_root.mkdir(parents=True, exist_ok=True)
    target = managed_root / f"{owner}__{repository}"

    if target.exists():
        if not (target / ".git").is_dir():
            raise RuntimeError(f"托管目录已存在但不是 Git 仓库：{target}")
        remote = _run_git(["git", "-C", str(target), "remote", "get-url", "origin"], timeout=10)
        try:
            existing_url = normalize_github_url(remote.stdout.strip())[0]
        except ValueError as exc:
            raise RuntimeError("已有托管仓库的 origin 不是受支持的 GitHub HTTPS 地址") from exc
        if existing_url.lower() != canonical_url.lower():
            raise RuntimeError("托管目录对应另一个远程仓库，已停止更新")
        _run_git(["git", "-C", str(target), "pull", "--ff-only", "--depth", "1"])
        action = "updated"
    else:
        temporary_root = Path(tempfile.mkdtemp(prefix=".clone-", dir=managed_root))
        temporary_target = temporary_root / "repository"
        try:
            _run_git(["git", "clone", "--depth", "1", canonical_url, str(temporary_target)])
            os.replace(temporary_target, target)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
        action = "cloned"

    return {
        "path": str(target.resolve()),
        "repository_url": canonical_url.removesuffix(".git"),
        "owner": owner,
        "repository": repository,
        "action": action,
    }


def _scip_indexer_status() -> dict:
    """报告 SCIP CLI 是否真正可执行；不在普通扫描中生成索引，避免阻塞语法扫描。"""

    def probe(executable: str | None) -> str:
        """用版本命令做轻量冒烟测试，区分“文件存在”和“CLI 能启动”。"""
        if not executable:
            return "not_installed"
        try:
            result = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return "unavailable"
        return "ready" if result.returncode == 0 else "installed_but_unavailable"

    python_path = shutil.which("scip-python")
    go_path = shutil.which("scip-go")
    if not go_path:
        go_bin = Path(subprocess.run(["go", "env", "GOPATH"], capture_output=True, text=True).stdout.strip()) / "bin"
        candidate = go_bin / ("scip-go.exe" if os.name == "nt" else "scip-go")
        if candidate.exists():
            go_path = str(candidate)
    node_version = None
    node_path = shutil.which("node")
    if node_path:
        try:
            node_version = subprocess.run([node_path, "--version"], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            node_version = None
    python_status = probe(python_path)
    if python_path and not node_version:
        python_status = "requires_node_20_lts"
    elif python_path and node_version and not node_version.startswith("v20."):
        python_status = "requires_node_20_lts"
    return {
        "python": {"available": bool(python_path), "executable": python_path, "node_version": node_version, "status": python_status},
        "go": {"available": bool(go_path), "executable": go_path, "status": probe(go_path)},
    }


def _iter_source_files(root: Path) -> list[Path]:
    """过滤构建产物、虚拟环境和上传目录，避免把无关内容写入代码 Wiki。"""
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        if any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
            continue
        lower_name = path.name.lower()
        if (
            lower_name.endswith(("_pb2.py", "_pb2_grpc.py", ".pb.go", ".gen.go", "_generated.go"))
            or ".generated." in lower_name
        ):
            continue
        files.append(path)
    return sorted(files)


def _node_signature(node: ast.AST, source_lines: list[str]) -> str | None:
    """提取 Python 函数/类的首行签名，避免把完整函数体塞进事实索引。"""
    line = getattr(node, "lineno", 0)
    if not line or line > len(source_lines):
        return None
    return source_lines[line - 1].strip()


class _PythonExtractor(ast.NodeVisitor):
    """从 Python AST 提取符号和局部调用/导入关系。"""

    def __init__(self, path: str, source: str):
        self.path = path
        self.lines = source.splitlines()
        self.symbols: list[SymbolFact] = []
        self.relations: list[RelationFact] = []
        self.scope: list[str] = []
        self.file_key = f"{path}::file"

    def _add_symbol(self, node: ast.AST, name: str, kind: str) -> SymbolFact:
        qualified = ".".join([*self.scope, name]) if self.scope else name
        end_line = getattr(node, "end_lineno", getattr(node, "lineno", 1))
        symbol = SymbolFact(
            path=self.path,
            name=name,
            qualified_name=qualified,
            kind=kind,
            signature=_node_signature(node, self.lines),
            start_line=getattr(node, "lineno", 1),
            end_line=end_line,
            parent_qualified_name=".".join(self.scope) if self.scope else None,
            docstring=ast.get_docstring(node),
        )
        self.symbols.append(symbol)
        return symbol

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_symbol(node, node.name, "class")
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, "async_function")

    def _visit_function(self, node: ast.AST, kind: str) -> None:
        name = getattr(node, "name", "unknown")
        self._add_symbol(node, name, kind)
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.relations.append(RelationFact(
                source_symbol_key=self.file_key,
                relation_type="imports",
                target_ref=alias.name,
                evidence={"line": node.lineno, "asname": alias.asname},
            ))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            self.relations.append(RelationFact(
                source_symbol_key=self.file_key,
                relation_type="imports",
                target_ref=f"{module}:{alias.name}",
                evidence={"line": node.lineno, "asname": alias.asname},
            ))

    def visit_Call(self, node: ast.Call) -> None:
        target = "unknown"
        if isinstance(node.func, ast.Name):
            target = node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts: list[str] = []
            current: ast.AST | None = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            target = ".".join(reversed(parts))
        source = self.scope[-1] if self.scope else self.file_key
        source_key = f"{self.path}::{'.'.join(self.scope)}" if self.scope else source
        self.relations.append(RelationFact(
            source_symbol_key=source_key,
            relation_type="calls",
            target_ref=target,
            evidence={"line": getattr(node, "lineno", None)},
            confidence=0.85,
        ))
        self.generic_visit(node)


def _extract_python(path: str, source: str) -> tuple[list[SymbolFact], list[RelationFact]]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return _extract_generic(path, source, parse_error=str(exc))
    extractor = _PythonExtractor(path, source)
    extractor.visit(tree)
    return extractor.symbols, extractor.relations


def _tree_sitter_parser(language: str):
    """创建 Python/Go Tree-sitter parser；接口变化或 grammar 缺失时返回 None。"""
    if Parser is None or Language is None:
        return None
    grammar = tree_sitter_python if language == "python" else tree_sitter_go
    if grammar is None:
        return None
    try:
        return Parser(Language(grammar.language()))
    except (TypeError, RuntimeError):
        return None


def _tree_text(source_bytes: bytes, node) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_tree_sitter(path: str, language: str, source: str) -> tuple[list[SymbolFact], list[RelationFact]] | None:
    """使用 Tree-sitter 提取语法结构和基础关系；语义精确关系由 SCIP 后续补齐。"""
    parser = _tree_sitter_parser(language)
    if parser is None:
        return None
    source_bytes = source.encode("utf-8", errors="replace")
    tree = parser.parse(source_bytes)
    lines = source.splitlines()
    symbols: list[SymbolFact] = []
    relations: list[RelationFact] = []
    definition_types = {
        "python": {"class_definition": ("class", "name"), "function_definition": ("function", "name")},
        "go": {"function_declaration": ("function", "name"), "method_declaration": ("method", "name"), "type_spec": ("type", "name")},
    }[language]
    scope: list[str] = []

    def walk(node) -> None:
        nonlocal scope
        node_type = node.type
        definition = definition_types.get(node_type)
        pushed = False
        if definition:
            kind, name_field = definition
            name_node = node.child_by_field_name(name_field)
            if name_node is not None:
                name = _tree_text(source_bytes, name_node)
                qualified = ".".join([*scope, name]) if scope else name
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                symbols.append(SymbolFact(
                    path=path, name=name, qualified_name=qualified, kind=kind,
                    signature=lines[start_line - 1].strip() if start_line <= len(lines) else None,
                    start_line=start_line, end_line=end_line,
                    parent_qualified_name=".".join(scope) if scope else None,
                ))
                # Python class/function和 Go type/function均可作为后续调用关系的范围节点。
                if node_type in {"class_definition", "function_definition", "type_spec", "function_declaration", "method_declaration"}:
                    scope.append(name)
                    pushed = True
        if node_type in {"import_statement", "import_declaration"}:
            target = _tree_text(source_bytes, node).strip()
            source_key = f"{path}::{'.'.join(scope)}" if scope else f"{path}::file"
            relations.append(RelationFact(
                source_symbol_key=source_key, relation_type="imports", target_ref=target,
                evidence={"line": node.start_point[0] + 1, "parser": "tree-sitter"}, confidence=0.9,
            ))
        if node_type in {"call", "call_expression"}:
            function_node = node.child_by_field_name("function") or node.child_by_field_name("function_name")
            target = _tree_text(source_bytes, function_node or node).strip()
            source_key = f"{path}::{'.'.join(scope)}" if scope else f"{path}::file"
            relations.append(RelationFact(
                source_symbol_key=source_key, relation_type="calls", target_ref=target,
                evidence={"line": node.start_point[0] + 1, "parser": "tree-sitter"}, confidence=0.8,
            ))
        for child in node.children:
            walk(child)
        if pushed:
            scope.pop()

    walk(tree.root_node)
    return symbols, relations


def _extract_generic(path: str, source: str, parse_error: str | None = None) -> tuple[list[SymbolFact], list[RelationFact]]:
    """多语言临时回退；后续由 SCIP/Tree-sitter 适配器替换。"""
    patterns = [
        (r"\b(class|interface|struct)\s+([A-Za-z_$][\w$]*)", "class"),
        (r"\b(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", "function"),
        (r"\b(?:public|private|protected|static|func|def|void|int|string|boolean|\w+)\s+([A-Za-z_$][\w$]*)\s*\([^\n]*\)", "function"),
    ]
    symbols: list[SymbolFact] = []
    lines = source.splitlines()
    for pattern, kind in patterns:
        for match in re.finditer(pattern, source):
            line = source.count("\n", 0, match.start()) + 1
            name = match.group(match.lastindex or 1)
            symbols.append(SymbolFact(
                path=path, name=name, qualified_name=name, kind=kind,
                signature=lines[line - 1].strip() if line <= len(lines) else None,
                start_line=line, end_line=line,
            ))
    relations: list[RelationFact] = []
    for line_no, line in enumerate(lines, 1):
        import_match = re.search(r"^\s*(?:import|from)\s+([^;]+)", line)
        if import_match:
            relations.append(RelationFact(
                source_symbol_key=f"{path}::file", relation_type="imports",
                target_ref=import_match.group(1).strip(), evidence={"line": line_no},
                confidence=0.65,
            ))
    if parse_error:
        relations.append(RelationFact(
            source_symbol_key=f"{path}::file", relation_type="parse_warning",
            target_ref=parse_error, evidence={"line": 1}, confidence=1.0,
        ))
    return symbols, relations


def _extract_file(root: Path, path: Path) -> FileFact:
    relative = path.relative_to(root).as_posix()
    language = CODE_EXTENSIONS[path.suffix.lower()]
    source = _read_text(path)
    if language in {"python", "go"}:
        parsed = _extract_tree_sitter(relative, language, source)
        if parsed is not None:
            symbols, relations = parsed
        elif language == "python":
            symbols, relations = _extract_python(relative, source)
        else:
            symbols, relations = _extract_generic(relative, source)
    else:
        symbols, relations = _extract_generic(relative, source)
    # 数据库事实键是“文件 + 限定名 + 起始行”；压缩源码可能在同一行重复声明，
    # 这里按同一键保留首个定义，避免一个噪声文件让整次项目扫描回滚。
    unique_symbols: list[SymbolFact] = []
    seen_symbol_keys: set[tuple[str, int]] = set()
    for symbol in symbols:
        key = (symbol.qualified_name, symbol.start_line)
        if key not in seen_symbol_keys:
            unique_symbols.append(symbol)
            seen_symbol_keys.add(key)
    symbols = unique_symbols
    # 每个文件都有一个稳定的 file 节点，便于挂载 imports 和组件证据。
    symbols.insert(0, SymbolFact(
        path=relative, name=relative, qualified_name="file", kind="file",
        signature=None, start_line=1, end_line=max(1, len(source.splitlines())),
    ))
    return FileFact(relative, language, _hash_text(source), len(source.splitlines()), symbols, relations)


def _scip_lines(occurrence, *, enclosing: bool = False) -> tuple[int, int]:
    """把 SCIP 的 0-based 半开区间转换成界面和数据库使用的 1-based 行号。"""
    oneof = "typed_enclosing_range" if enclosing else "typed_range"
    selected = occurrence.WhichOneof(oneof)
    if selected:
        value = getattr(occurrence, selected)
        if selected.startswith("single_line"):
            return value.line + 1, value.line + 1
        # SCIP 是半开区间；结束字符为 0 时，end_line 指向下一行开头，不应计入展示范围。
        end_line = value.end_line if value.end_character == 0 and value.end_line > value.start_line else value.end_line + 1
        return value.start_line + 1, end_line
    values = list(occurrence.enclosing_range if enclosing else occurrence.range)
    if len(values) == 3:
        return values[0] + 1, values[0] + 1
    if len(values) == 4:
        end_line = values[2] if values[3] == 0 and values[2] > values[0] else values[2] + 1
        return values[0] + 1, end_line
    return 1, 1


def _scip_symbol_name(info) -> str:
    """优先使用 indexer 提供的 display_name，避免自行猜测 SCIP descriptor 转义。"""
    if info and info.display_name:
        return info.display_name
    if info and info.symbol.startswith("local "):
        return info.symbol.split(" ", 1)[1]
    symbol = info.symbol if info else "unknown"
    tail = re.split(r"[/#.]", symbol.rstrip("/#."))[-1]
    return tail.split("(", 1)[0] or "unknown"


def _scip_symbol_kind(info) -> str:
    if not info or not info.kind:
        return "symbol"
    return scip_pb2.SymbolInformation.Kind.Name(info.kind).removesuffix("Kind").lower()


def _source_symbol_at(file_fact: FileFact, line: int) -> SymbolFact:
    """把 SCIP 引用位置归属到最小的 Tree-sitter 结构范围，用于建立调用方/引用方。"""
    candidates = [
        symbol for symbol in file_fact.symbols
        if symbol.kind != "file" and symbol.start_line <= line <= symbol.end_line
    ]
    if candidates:
        return min(candidates, key=lambda item: (item.end_line - item.start_line, -item.start_line))
    return next(symbol for symbol in file_fact.symbols if symbol.kind == "file")


def _merge_scip_index(index_path: Path, module_root: Path, project_root: Path, files: list[FileFact]) -> dict:
    """解析官方 SCIP Protobuf，并把定义/引用合并进 Tree-sitter 事实。"""
    index = scip_pb2.Index()
    index.ParseFromString(index_path.read_bytes())
    file_by_path = {item.path: item for item in files}
    module_prefix = module_root.relative_to(project_root).as_posix()
    module_prefix = "" if module_prefix == "." else module_prefix
    info_by_symbol = {
        info.symbol: info
        for document in index.documents
        for info in document.symbols
    }
    info_by_symbol.update({info.symbol: info for info in index.external_symbols})
    local_keys: dict[str, str] = {}
    definition_count = 0
    reference_count = 0
    relationship_count = 0

    # 第一遍先登记定义，确保第二遍处理引用时可以解析本项目目标。
    for document in index.documents:
        relative = "/".join(part for part in (module_prefix, document.relative_path.replace("\\", "/")) if part)
        file_fact = file_by_path.get(relative)
        if not file_fact:
            continue
        for occurrence in document.occurrences:
            if not occurrence.symbol or not occurrence.symbol_roles & scip_pb2.Definition:
                continue
            info = info_by_symbol.get(occurrence.symbol)
            if not info or _scip_symbol_kind(info) not in SCIP_WIKI_KINDS:
                continue
            name = _scip_symbol_name(info)
            occurrence_start, _ = _scip_lines(occurrence)
            start_line, end_line = _scip_lines(occurrence, enclosing=True)
            existing = next((
                symbol for symbol in file_fact.symbols
                if symbol.kind != "file" and symbol.name == name
                and symbol.start_line <= occurrence_start <= symbol.end_line
            ), None)
            if existing is None:
                existing = SymbolFact(
                    path=relative,
                    name=name,
                    qualified_name=f"{name}@{occurrence_start}",
                    kind=_scip_symbol_kind(info),
                    signature=info.signature_documentation.text if info and info.HasField("signature_documentation") else None,
                    start_line=start_line,
                    end_line=max(start_line, end_line),
                    docstring="\n\n".join(info.documentation) if info and info.documentation else None,
                )
                file_fact.symbols.append(existing)
            existing.metadata.update({
                "parser": "scip",
                "scip_symbol": occurrence.symbol,
                "scip_kind": _scip_symbol_kind(info),
            })
            if info and info.HasField("signature_documentation") and not existing.signature:
                existing.signature = info.signature_documentation.text
            if info and info.documentation and not existing.docstring:
                existing.docstring = "\n\n".join(info.documentation)
            local_keys[occurrence.symbol] = f"{relative}::{existing.qualified_name}"
            definition_count += 1

    # 第二遍记录精确引用。调用关系仍由 Tree-sitter 保留；SCIP references 表示更广义的语义使用。
    for document in index.documents:
        relative = "/".join(part for part in (module_prefix, document.relative_path.replace("\\", "/")) if part)
        file_fact = file_by_path.get(relative)
        if not file_fact:
            continue
        existing_relations = {
            (item.source_symbol_key, item.relation_type, item.target_ref, item.evidence.get("line"))
            for item in file_fact.relations
        }
        for occurrence in document.occurrences:
            if not occurrence.symbol or occurrence.symbol_roles & scip_pb2.Definition:
                continue
            line, _ = _scip_lines(occurrence)
            source = _source_symbol_at(file_fact, line)
            source_key = f"{relative}::{source.qualified_name}"
            relation_type = "imports" if occurrence.symbol_roles & scip_pb2.Import else "references"
            dedupe_key = (source_key, relation_type, occurrence.symbol, line)
            if dedupe_key in existing_relations:
                continue
            file_fact.relations.append(RelationFact(
                source_symbol_key=source_key,
                relation_type=relation_type,
                target_symbol_key=local_keys.get(occurrence.symbol),
                target_ref=occurrence.symbol,
                evidence={"line": line, "parser": "scip", "symbol_roles": occurrence.symbol_roles},
                confidence=1.0,
            ))
            existing_relations.add(dedupe_key)
            reference_count += 1

        for info in document.symbols:
            source_key = local_keys.get(info.symbol)
            if not source_key:
                continue
            for relationship in info.relationships:
                relation_types = []
                if relationship.is_implementation:
                    relation_types.append("implements")
                if relationship.is_type_definition:
                    relation_types.append("type_definition")
                if relationship.is_reference:
                    relation_types.append("related_reference")
                if relationship.is_definition:
                    relation_types.append("related_definition")
                for relation_type in relation_types:
                    file_fact.relations.append(RelationFact(
                        source_symbol_key=source_key,
                        relation_type=relation_type,
                        target_symbol_key=local_keys.get(relationship.symbol),
                        target_ref=relationship.symbol,
                        evidence={"parser": "scip", "relationship": True},
                        confidence=1.0,
                    ))
                    relationship_count += 1
    return {
        "documents": len(index.documents),
        "definitions": definition_count,
        "references": reference_count,
        "relationships": relationship_count,
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }


def _run_go_scip(root: Path, files: list[FileFact], indexers: dict) -> dict:
    """为项目中的每个 Go module 生成并消费 SCIP；单模块失败不会丢失 Tree-sitter 结果。"""
    go_status = indexers.get("go", {})
    module_files = [
        path for path in root.rglob("go.mod")
        if not any(part in IGNORED_DIRS for part in path.relative_to(root).parts)
    ]
    report = {"status": "skipped", "modules": [], "summary": {"succeeded": 0, "failed": 0}}
    if not module_files:
        report["reason"] = "no_go_mod"
        return report
    if go_status.get("status") != "ready":
        report["reason"] = "scip_go_unavailable"
        return report

    with tempfile.TemporaryDirectory(prefix="knowledge-scip-") as temp_dir:
        for number, module_file in enumerate(sorted(module_files)):
            module_root = module_file.parent
            output = Path(temp_dir) / f"go-{number}.scip"
            command = [
                go_status["executable"], "index", "./...",
                "--module-root=.", f"--output={output}", "--quiet",
            ]
            try:
                result = subprocess.run(
                    command,
                    cwd=module_root,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                )
                if result.returncode != 0 or not output.is_file():
                    raise RuntimeError((result.stderr or result.stdout or "index file was not generated").strip()[-2000:])
                stats = _merge_scip_index(output, module_root, root, files)
                report["modules"].append({
                    "module_root": module_root.relative_to(root).as_posix(),
                    "status": "succeeded",
                    **stats,
                })
                report["summary"]["succeeded"] += 1
            except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
                report["modules"].append({
                    "module_root": module_root.relative_to(root).as_posix(),
                    "status": "failed",
                    "error": str(exc),
                })
                report["summary"]["failed"] += 1
    report["status"] = "succeeded" if not report["summary"]["failed"] else (
        "partial" if report["summary"]["succeeded"] else "failed"
    )
    return report


def _component_facts(root: Path, files: list[FileFact]) -> list[dict]:
    """根据依赖声明、初始化代码和配置证据识别组件，不让 LLM 猜框架。"""
    evidence: dict[tuple[str, str], list[dict]] = {}
    all_text: dict[str, str] = {}
    manifest_names = {
        "requirements.txt", "pyproject.toml", "package.json", "pom.xml", "build.gradle",
        "build.gradle.kts", "go.mod", "cargo.toml", "dockerfile", "compose.yml",
        "compose.yaml", "docker-compose.yml", "docker-compose.yaml", ".env.example",
    }
    for path in root.rglob("*"):
        if not path.is_file() or any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
            continue
        # 扫描器源码包含组件识别规则本身，不能把规则字符串当成被扫描项目的使用证据。
        if path.name == "code_wiki.py" and path.parent.name == "app":
            continue
        # 组件识别只读取源码和真实依赖/部署配置，不从设计文档中的示例文字推断组件。
        if path.suffix.lower() not in CODE_EXTENSIONS and path.name.lower() not in manifest_names:
            continue
        try:
            all_text[path.relative_to(root).as_posix()] = _read_text(path)
        except OSError:
            continue

    signatures = {
        "FastAPI": ("web_framework", [r"fastapi", r"FastAPI\s*\(", r"@(?:app|router)\.(?:get|post|put|delete)"]),
        "PostgreSQL": ("database", [r"postgresql", r"psycopg", r"DATABASE_URL"]),
        "pgvector": ("vector_store", [r"pgvector", r"vector\(", r"CREATE EXTENSION IF NOT EXISTS vector"]),
        "SSE": ("transport", [r"text/event-stream", r"StreamingResponse", r"event:"]),
        "OpenAI-compatible API": ("model_api", [r"chat\.completions", r"embeddings\.create", r"LLM_API_BASE"]),
    }
    for relative, text in all_text.items():
        for name, (category, patterns) in signatures.items():
            matched = [pattern for pattern in patterns if re.search(pattern, text, re.IGNORECASE)]
            if matched:
                evidence.setdefault((name, category), []).append({"file": relative, "patterns": matched})
    docker_names = {"dockerfile", "compose.yml", "compose.yaml", "docker-compose.yml", "docker-compose.yaml"}
    for relative in all_text:
        if Path(relative).name.lower() in docker_names:
            evidence.setdefault(("Docker", "deployment"), []).append(
                {"file": relative, "patterns": ["deployment_file"]}
            )
    return [
        {"name": name, "category": category, "confidence": min(0.99, 0.65 + 0.1 * len(items)), "evidence": items}
        for (name, category), items in sorted(evidence.items())
    ]


def scan_project(root_path: str) -> dict:
    """扫描项目并返回可入库的事实；不调用模型，便于重复运行和自动测试。"""
    root = Path(root_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("project path must be an existing directory")
    commit_hash, commit_source = _git_commit(root)
    files = [_extract_file(root, path) for path in _iter_source_files(root)]
    # 浏览器上传目录没有 Git HEAD，以源码路径和内容哈希生成版本，确保内容变化会产生新快照。
    if commit_source == "content_scan":
        snapshot = "\n".join(f"{item.path}:{item.content_hash}" for item in sorted(files, key=lambda item: item.path))
        commit_hash = "scan-" + _hash_text(snapshot)[:16]
    indexers = _scip_indexer_status()
    scip_report = {"go": _run_go_scip(root, files, indexers)}
    components = _component_facts(root, files)
    return {
        "project_id": str(uuid.uuid5(ID_NAMESPACE, str(root))),
        "project_name": root.name,
        "root_path": str(root),
        "commit_hash": commit_hash,
        "commit_source": commit_source,
        "scip_indexers": indexers,
        "scip": scip_report,
        "files": files,
        "components": components,
    }


def initialize_code_wiki() -> None:
    """创建独立 Code Wiki 表；不复用 knowledge_chunks，保持 Wiki/RAG 边界。"""
    with connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_projects (
                project_id UUID PRIMARY KEY,
                project_name TEXT NOT NULL,
                root_path TEXT NOT NULL,
                current_commit TEXT NOT NULL,
                current_commit_source TEXT NOT NULL,
                scan_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        conn.execute("ALTER TABLE code_projects ADD COLUMN IF NOT EXISTS scan_metadata JSONB NOT NULL DEFAULT '{}'::jsonb")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_files (
                file_id UUID PRIMARY KEY,
                project_id UUID NOT NULL REFERENCES code_projects(project_id) ON DELETE CASCADE,
                commit_hash TEXT NOT NULL,
                path TEXT NOT NULL,
                language TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                line_count INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(project_id, commit_hash, path)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_symbols (
                symbol_id UUID PRIMARY KEY,
                project_id UUID NOT NULL REFERENCES code_projects(project_id) ON DELETE CASCADE,
                file_id UUID NOT NULL REFERENCES code_files(file_id) ON DELETE CASCADE,
                commit_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                symbol_kind TEXT NOT NULL,
                signature TEXT,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                parent_qualified_name TEXT,
                docstring TEXT,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                UNIQUE(project_id, commit_hash, file_id, qualified_name, start_line)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_relations (
                relation_id UUID PRIMARY KEY,
                project_id UUID NOT NULL REFERENCES code_projects(project_id) ON DELETE CASCADE,
                commit_hash TEXT NOT NULL,
                source_symbol_id UUID NOT NULL REFERENCES code_symbols(symbol_id) ON DELETE CASCADE,
                target_symbol_id UUID REFERENCES code_symbols(symbol_id) ON DELETE SET NULL,
                relation_type TEXT NOT NULL,
                target_ref TEXT,
                evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
                confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_components (
                component_id UUID PRIMARY KEY,
                project_id UUID NOT NULL REFERENCES code_projects(project_id) ON DELETE CASCADE,
                commit_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
                UNIQUE(project_id, commit_hash, name, category)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_code_files_project ON code_files(project_id, commit_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_code_symbols_name ON code_symbols(project_id, name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_code_relations_source ON code_relations(source_symbol_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_code_relations_target ON code_relations(target_symbol_id)")
        conn.execute("COMMENT ON TABLE code_projects IS '代码 Wiki 项目事实层：项目和当前扫描版本'")
        conn.execute("COMMENT ON COLUMN code_projects.scan_metadata IS '最近一次扫描的来源、SCIP CLI 状态、模块结果、索引哈希和错误摘要'")
        conn.execute("COMMENT ON TABLE code_files IS '代码 Wiki 文件事实：路径、语言、内容哈希和 Commit'")
        conn.execute("COMMENT ON TABLE code_symbols IS '代码 Wiki 符号事实：类、函数、方法和精确源代码范围'")
        conn.execute("COMMENT ON TABLE code_relations IS '代码 Wiki 关系事实：导入、调用和后续跨项目关系'")
        conn.execute("COMMENT ON TABLE code_components IS '代码 Wiki 组件识别结果及其依赖文件/初始化代码证据'")


def _symbol_id(project_id: str, commit_hash: str, path: str, qualified: str, line: int) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"{project_id}:{commit_hash}:{path}:{qualified}:{line}"))


def persist_scan(scan: dict) -> dict:
    """以 Commit 为边界幂等写入扫描结果；同一版本重复扫描不会叠加关系。"""
    project_id = scan["project_id"]
    commit_hash = scan["commit_hash"]
    with connection() as conn:
        conn.execute(
            """INSERT INTO code_projects
            (project_id, project_name, root_path, current_commit, current_commit_source, scan_metadata)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (project_id) DO UPDATE SET
              project_name = EXCLUDED.project_name, root_path = EXCLUDED.root_path,
              current_commit = EXCLUDED.current_commit, current_commit_source = EXCLUDED.current_commit_source,
              scan_metadata = EXCLUDED.scan_metadata,
              updated_at = NOW()""",
            (project_id, scan["project_name"], scan["root_path"], commit_hash, scan["commit_source"],
             json.dumps({
                 "scip_indexers": scan.get("scip_indexers", {}),
                 "scip": scan.get("scip", {}),
                 "source": scan.get("source", {"type": "local", "path": scan["root_path"]}),
             }, ensure_ascii=False)),
        )
        # 重新扫描同一 Commit 时先替换该版本事实，避免旧的符号/关系残留。
        conn.execute("DELETE FROM code_relations WHERE project_id = %s AND commit_hash = %s", (project_id, commit_hash))
        conn.execute("DELETE FROM code_symbols WHERE project_id = %s AND commit_hash = %s", (project_id, commit_hash))
        conn.execute("DELETE FROM code_files WHERE project_id = %s AND commit_hash = %s", (project_id, commit_hash))
        conn.execute("DELETE FROM code_components WHERE project_id = %s AND commit_hash = %s", (project_id, commit_hash))

        symbol_ids: dict[str, str] = {}
        file_ids: dict[str, str] = {}
        for file_fact in scan["files"]:
            file_id = str(uuid.uuid5(ID_NAMESPACE, f"file:{project_id}:{commit_hash}:{file_fact.path}"))
            file_ids[file_fact.path] = file_id
            conn.execute(
                """INSERT INTO code_files
                (file_id, project_id, commit_hash, path, language, content_hash, line_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (file_id, project_id, commit_hash, file_fact.path, file_fact.language, file_fact.content_hash, file_fact.line_count),
            )
            for symbol in file_fact.symbols:
                symbol.symbol_id = _symbol_id(project_id, commit_hash, symbol.path, symbol.qualified_name, symbol.start_line)
                symbol_ids[f"{symbol.path}::{symbol.qualified_name}"] = symbol.symbol_id
                conn.execute(
                    """INSERT INTO code_symbols
                    (symbol_id, project_id, file_id, commit_hash, name, qualified_name, symbol_kind,
                     signature, start_line, end_line, parent_qualified_name, docstring, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)""",
                    (symbol.symbol_id, project_id, file_id, commit_hash, symbol.name, symbol.qualified_name,
                     symbol.kind, symbol.signature, symbol.start_line, symbol.end_line,
                     symbol.parent_qualified_name, symbol.docstring, json.dumps(symbol.metadata, ensure_ascii=False)),
                )
        # 先建立文件内和同名符号索引，再尝试把调用目标解析到本项目符号。
        by_name: dict[str, str] = {}
        for key, symbol_id in symbol_ids.items():
            by_name.setdefault(key.rsplit("::", 1)[-1].split(".")[-1], symbol_id)
            by_name.setdefault(key.rsplit("::", 1)[-1], symbol_id)
        for file_fact in scan["files"]:
            for relation in file_fact.relations:
                source_id = symbol_ids.get(relation.source_symbol_key) or symbol_ids.get(f"{file_fact.path}::file")
                if not source_id:
                    continue
                target_id = symbol_ids.get(relation.target_symbol_key or "") or by_name.get(relation.target_ref or "")
                conn.execute(
                    """INSERT INTO code_relations
                    (relation_id, project_id, commit_hash, source_symbol_id, target_symbol_id,
                     relation_type, target_ref, evidence, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)""",
                    (str(uuid.uuid4()), project_id, commit_hash, source_id, target_id,
                     relation.relation_type, relation.target_ref, json.dumps(relation.evidence, ensure_ascii=False), relation.confidence),
                )
        for component in scan["components"]:
            conn.execute(
                """INSERT INTO code_components
                (component_id, project_id, commit_hash, name, category, confidence, evidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)""",
                (str(uuid.uuid4()), project_id, commit_hash, component["name"], component["category"],
                 component["confidence"], json.dumps(component["evidence"], ensure_ascii=False)),
            )
    return {
        "project_id": project_id, "project_name": scan["project_name"], "commit_hash": commit_hash,
        "file_count": len(scan["files"]),
        "symbol_count": sum(len(file_fact.symbols) for file_fact in scan["files"]),
        "relation_count": sum(len(file_fact.relations) for file_fact in scan["files"]),
        "component_count": len(scan["components"]),
        "scip_indexers": scan.get("scip_indexers", {}),
        "scip": scan.get("scip", {}),
    }


def list_code_projects() -> list[dict]:
    with connection() as conn:
        result = conn.execute(
            """SELECT project_id, project_name, root_path, current_commit, current_commit_source,
                      status, scan_metadata, created_at, updated_at
               FROM code_projects ORDER BY updated_at DESC"""
        )
        columns = [desc.name for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]


def get_code_overview(project_id: str) -> dict | None:
    with connection() as conn:
        project = conn.execute(
            """SELECT project_id, project_name, root_path, current_commit, current_commit_source, status, scan_metadata
               FROM code_projects WHERE project_id = %s""", (project_id,)
        ).fetchone()
        if not project:
            return None
        components = conn.execute(
            """SELECT name, category, confidence, evidence FROM code_components
               WHERE project_id = %s AND commit_hash = %s ORDER BY category, name""", (project_id, project[3])
        ).fetchall()
        counts = conn.execute(
            """SELECT (SELECT COUNT(*) FROM code_files WHERE project_id = %s AND commit_hash = %s),
                      (SELECT COUNT(*) FROM code_symbols WHERE project_id = %s AND commit_hash = %s),
                      (SELECT COUNT(*) FROM code_relations WHERE project_id = %s AND commit_hash = %s)""",
            (project_id, project[3], project_id, project[3], project_id, project[3]),
        ).fetchone()
        return {
            "project_id": str(project[0]), "project_name": project[1], "root_path": project[2],
            "current_commit": project[3], "current_commit_source": project[4], "status": project[5],
            "scan_metadata": project[6],
            "counts": {"files": counts[0], "symbols": counts[1], "relations": counts[2]},
            "components": [
                {"name": item[0], "category": item[1], "confidence": item[2], "evidence": item[3]}
                for item in components
            ],
        }


def list_code_files(project_id: str) -> list[dict]:
    """列出项目当前 Commit 的文件，并附带符号和出站关系数量。"""
    with connection() as conn:
        result = conn.execute(
            """SELECT f.file_id, f.path, f.language, f.line_count, f.content_hash,
                      COUNT(DISTINCT s.symbol_id) AS symbol_count,
                      COUNT(DISTINCT r.relation_id) AS relation_count
               FROM code_projects p
               JOIN code_files f ON f.project_id = p.project_id AND f.commit_hash = p.current_commit
               LEFT JOIN code_symbols s ON s.file_id = f.file_id
               LEFT JOIN code_relations r ON r.source_symbol_id = s.symbol_id
               WHERE p.project_id = %s
               GROUP BY f.file_id, f.path, f.language, f.line_count, f.content_hash
               ORDER BY f.path""",
            (project_id,),
        )
        columns = [desc.name for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]


def search_code_symbols(
    project_id: str,
    query: str = "",
    limit: int = 100,
    file_path: str | None = None,
    symbol_kind: str | None = None,
) -> list[dict]:
    """搜索当前 Commit 的符号；空 query 用于页面按文件浏览，不跨历史版本混排。"""
    with connection() as conn:
        result = conn.execute(
            """SELECT s.symbol_id, s.name, s.qualified_name, s.symbol_kind, f.path,
                      s.start_line, s.end_line, s.signature, s.metadata
               FROM code_projects p
               JOIN code_symbols s ON s.project_id = p.project_id AND s.commit_hash = p.current_commit
               JOIN code_files f ON f.file_id = s.file_id
               WHERE p.project_id = %s
                 AND (%s = '' OR s.name ILIKE %s OR s.qualified_name ILIKE %s OR f.path ILIKE %s)
                 AND (%s::text IS NULL OR f.path = %s::text)
                 AND (%s::text IS NULL OR s.symbol_kind = %s::text)
               ORDER BY CASE WHEN lower(s.name) = lower(%s) THEN 0 ELSE 1 END,
                        f.path, s.start_line, length(s.qualified_name) LIMIT %s""",
            (
                project_id, query, f"%{query}%", f"%{query}%", f"%{query}%",
                file_path, file_path, symbol_kind, symbol_kind, query, limit,
            ),
        )
        columns = [desc.name for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]


def get_code_symbol(symbol_id: str) -> dict | None:
    with connection() as conn:
        symbol = conn.execute(
            """SELECT s.symbol_id, s.project_id, s.commit_hash, s.name, s.qualified_name,
                      s.symbol_kind, f.path, s.signature, s.start_line, s.end_line,
                      s.parent_qualified_name, s.docstring, s.metadata
               FROM code_symbols s JOIN code_files f ON f.file_id = s.file_id
               WHERE s.symbol_id = %s""", (symbol_id,)
        ).fetchone()
        if not symbol:
            return None
        relations = conn.execute(
            """SELECT r.relation_type, r.target_ref, r.confidence,
                      ts.name AS target_name, tf.path AS target_path,
                      r.evidence, ts.symbol_id AS target_symbol_id
               FROM code_relations r
               LEFT JOIN code_symbols ts ON ts.symbol_id = r.target_symbol_id
               LEFT JOIN code_files tf ON tf.file_id = ts.file_id
               WHERE r.source_symbol_id = %s ORDER BY r.relation_type, r.target_ref""", (symbol_id,)
        ).fetchall()
        incoming_relations = conn.execute(
            """SELECT r.relation_type, r.target_ref, r.confidence,
                      ss.name AS source_name, sf.path AS source_path,
                      ss.start_line AS source_line, ss.symbol_id AS source_symbol_id,
                      r.evidence
               FROM code_relations r
               JOIN code_symbols ss ON ss.symbol_id = r.source_symbol_id
               JOIN code_files sf ON sf.file_id = ss.file_id
               WHERE r.target_symbol_id = %s
               ORDER BY r.relation_type, sf.path, ss.start_line""",
            (symbol_id,),
        ).fetchall()
        return {
            "symbol_id": str(symbol[0]), "project_id": str(symbol[1]), "commit_hash": symbol[2],
            "name": symbol[3], "qualified_name": symbol[4], "symbol_kind": symbol[5],
            "path": symbol[6], "signature": symbol[7], "start_line": symbol[8], "end_line": symbol[9],
            "parent_qualified_name": symbol[10], "docstring": symbol[11],
            "metadata": symbol[12],
            "relations": [
                {"relation_type": row[0], "target_ref": row[1], "confidence": row[2],
                 "target_name": row[3], "target_path": row[4], "evidence": row[5],
                 "target_symbol_id": str(row[6]) if row[6] else None}
                for row in relations
            ],
            "incoming_relations": [
                {"relation_type": row[0], "target_ref": row[1], "confidence": row[2],
                 "source_name": row[3], "source_path": row[4], "source_line": row[5],
                 "source_symbol_id": str(row[6]), "evidence": row[7]}
                for row in incoming_relations
            ],
        }
