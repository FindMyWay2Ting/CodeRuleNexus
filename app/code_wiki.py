"""Code Intelligence 的第一阶段实现。

本模块只负责建立“代码事实层”，不让 LLM 直接猜测或修改代码知识：

1. 接收本地项目，或将公开 GitHub 仓库拉取到受控目录后扫描。
2. 扫描项目文件并记录项目、Commit、文件和内容哈希。
3. Python/Go 使用 Tree-sitter 提取结构，Python 在 grammar 缺失时回退到标准库 AST。
4. Go module 使用本机 SCIP，Python 项目使用 Docker 中的 SCIP 补充精确定义、引用和实现关系，失败时保留结构扫描结果。
5. 其他语言使用保守的定义边界回退，保证扫描链路可以先跑起来。
6. 从依赖、初始化代码和配置中识别组件、模块、入口、API、资源和下游，并保存证据。
7. 将结果写入独立的 Code Wiki 表，和 RAG 的 knowledge_chunks 分开。

Tree-sitter 已作为 Python/Go 的实际解析主路径；SCIP indexer 作为精确语义索引增强路径，通过扫描结果报告可用性，避免外部构建环境阻塞基础事实扫描。
"""

from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import psycopg

from .db import connection
from .config import settings
from . import scip_pb2
from .config_parsers import parse_project_configs

logger = logging.getLogger("knowledge.code_wiki")


class RepositoryImportBusy(RuntimeError):
    """同一工作空间中的同一远程仓库已有导入任务。"""


class CodeImportValidationError(ValueError):
    """可以安全返回给客户端的代码导入参数错误。"""

try:
    import yaml
except ImportError:  # pragma: no cover - requirements 已声明，保留旧环境降级能力
    yaml = None

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
    "module", "namespace", "package", "struct", "trait", "type", "typealias", "symbol",
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


@dataclass
class ArchitectureFact:
    """架构理解层的一条确定性事实，必须可以回溯到文件和代码行。"""

    fact_type: str
    name: str
    value: str | None
    path: str
    line: int
    evidence: str
    confidence: float = 1.0


@dataclass
class ArchitectureLink:
    """连接配置、客户端初始化和实际使用点的可解释架构边。"""

    source_fact_key: tuple[str, str, str | None, str, int]
    target_fact_key: tuple[str, str, str | None, str, int]
    relation_type: str
    evidence: str
    confidence: float = 1.0


def _architecture_fact_key(fact: ArchitectureFact) -> tuple[str, str, str | None, str, int]:
    """架构事实的扫描期稳定键，与数据库唯一约束字段保持一致。"""
    return (fact.fact_type, fact.name, fact.value, fact.path, fact.line)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _hash_file_bytes(path: Path) -> str:
    """流式计算大文件哈希，避免为生成内容版本一次读入整个文件。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_architecture_value(value: str) -> str:
    """脱敏密码、Token 和连接串凭据，只保留架构识别需要的名称与地址。"""
    clean = value.strip().strip("\"'`")
    clean = re.sub(
        r"(?i)(password|passwd|pwd|token|secret|api[_-]?key)\s*[:=]\s*([^;,&\s]+)",
        r"\1=[REDACTED]",
        clean,
    )
    clean = re.sub(r"(?i)(://[^:/\s]+:)[^@/\s]+@", r"\1[REDACTED]@", clean)
    clean = re.sub(r"(?i)([?&](?:token|key|signature|sig|credential)=)[^&#\s]+", r"\1[REDACTED]", clean)
    return clean[:177] + "..." if len(clean) > 180 else clean


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _yaml_scalar_entries(text: str) -> list[tuple[str, str, int, str]]:
    """读取 YAML 标量并保留完整键路径与行号；解析失败时由普通行规则继续兜底。"""
    if yaml is None:
        return []
    try:
        root = yaml.compose(text)
    except yaml.YAMLError:
        return []
    entries: list[tuple[str, str, int, str]] = []

    def walk(node, path: list[str]) -> None:
        if isinstance(node, yaml.MappingNode):
            for key_node, value_node in node.value:
                key = str(getattr(key_node, "value", ""))
                walk(value_node, [*path, key])
        elif isinstance(node, yaml.SequenceNode):
            for value_node in node.value:
                walk(value_node, path)
        elif isinstance(node, yaml.ScalarNode) and path:
            line = node.start_mark.line + 1
            value = str(node.value)
            full_key = ".".join(path)
            entries.append((full_key, value, line, f"{full_key}: {value}"))

    if root is not None:
        walk(root, [])
    return entries


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
        raise CodeImportValidationError("请输入标准 GitHub HTTPS 地址，例如 https://github.com/owner/repository")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) != 2:
        raise CodeImportValidationError("GitHub 地址必须指向一个仓库，不能是用户页、文件页或分支页")
    owner, repository = parts
    if repository.lower().endswith(".git"):
        repository = repository[:-4]
    if (
        not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", owner)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", repository)
        or repository in {".", ".."}
    ):
        raise CodeImportValidationError("GitHub owner 或仓库名称格式不正确")
    return f"https://github.com/{owner}/{repository}.git", owner, repository


def github_repository_key(owner: str, repository: str) -> str:
    """生成 GitHub 仓库的大小写无关身份键。"""
    return f"{owner.casefold()}__{repository.casefold()}"


def normalize_uploaded_path(filename: str) -> tuple[str, PurePosixPath]:
    """校验浏览器目录上传携带的相对路径，禁止绝对路径和目录穿越。"""
    raw = (filename or "").replace("\\", "/").strip()
    path = PurePosixPath(raw)
    parts = path.parts
    if path.is_absolute() or len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        raise CodeImportValidationError("本地项目文件必须包含文件夹相对路径，请重新选择整个项目目录")
    if any(re.search(r'[<>:"|?*\x00-\x1f]', part) for part in parts):
        raise CodeImportValidationError("本地项目中包含 Windows 不支持的文件名")
    project_name = parts[0]
    if len(project_name) > 100:
        raise CodeImportValidationError("本地项目文件夹名称不能超过 100 个字符")
    return project_name, PurePosixPath(*parts[1:])


def managed_local_repository_path(project_name: str, repository_root: Path | None = None, workspace_id: str | None = None) -> Path:
    """为浏览器上传项目生成稳定托管目录；同名文件夹会更新同一个项目。"""
    root = (repository_root or DEFAULT_REPOSITORY_ROOT).resolve() / "local"
    if workspace_id and workspace_id != settings.workspace_id:
        root = root / workspace_id
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", project_name).strip("-.")[:48] or "project"
    digest = _hash_text(project_name)[:8]
    return root / f"{slug}-{digest}"


def managed_code_project_id(identity_path: Path, workspace_id: str) -> str:
    """由稳定托管身份生成项目 ID；随机 staging 和 Commit 目录都不参与。"""
    identity = str(identity_path.resolve())
    if workspace_id != settings.workspace_id:
        identity = f"{workspace_id}:{identity}"
    return str(uuid.uuid5(ID_NAMESPACE, identity))


def _remove_managed_tree(path: Path) -> None:
    """删除系统托管目录，并处理 Git pack 文件在 Windows 上的只读属性。"""
    def remove_readonly(func, filename, _exc_info):
        os.chmod(filename, stat.S_IWRITE)
        func(filename)

    shutil.rmtree(path, onerror=remove_readonly)


def _run_git(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """运行无交互 Git 命令；避免服务端扫描期间等待凭据输入。"""
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        return subprocess.run(
            command,
            capture_output=True,
            # Windows 默认代码页可能无法解码 Git 输出中的中文路径，导致
            # subprocess reader 线程异常并返回 stdout=None；统一用 UTF-8 容错。
            text=True,
            encoding="utf-8",
            errors="replace",
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


@lru_cache(maxsize=1)
def _scip_indexer_status() -> dict:
    """报告 SCIP CLI 是否真正可执行；索引器探测失败不能阻塞基础语法扫描。"""

    def probe(executable: str | None) -> str:
        """用版本命令做轻量冒烟测试，区分“文件存在”和“CLI 能启动”。"""
        if not executable:
            return "not_installed"
        try:
            result = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=3,
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
    docker_path = shutil.which("docker")
    python_container_status = "not_installed"
    if docker_path:
        try:
            image_result = subprocess.run(
                [docker_path, "image", "inspect", settings.scip_python_image],
                capture_output=True,
                text=True,
                    timeout=3,
                check=False,
            )
            python_container_status = "ready" if image_result.returncode == 0 else "image_not_found"
        except (OSError, subprocess.SubprocessError):
            python_container_status = "unavailable"
    return {
        "python": {"available": bool(python_path), "executable": python_path, "node_version": node_version, "status": python_status},
        "python_container": {"available": python_container_status == "ready", "executable": docker_path, "image": settings.scip_python_image, "status": python_container_status},
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
    except SyntaxError:
        # 解析诊断可能包含源码片段和本机路径；事实层只记录稳定错误类型。
        return _extract_generic(path, source, parse_error="python_syntax_error")
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
    # 部分 SCIP 索引器（包括 scip-python）会把合法符号的 kind 写成
    # Unspecified(0)，但 definition occurrence 和符号描述仍然完整；保留为
    # 通用 symbol，避免因为枚举值缺省而丢掉整个语言的定义层。
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
            # scip-python 偶尔会为缺少有效 SymbolInformation 的定义生成 unknown；
            # 这类节点既不能导航，又可能在同一位置重复出现，不写入 Wiki 事实层。
            if name == "unknown":
                continue
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
                logger.warning(
                    "go_scip_module_failed module=%s error_type=%s",
                    module_root.relative_to(root).as_posix(), type(exc).__name__, exc_info=True,
                )
                report["modules"].append({
                    "module_root": module_root.relative_to(root).as_posix(),
                    "status": "failed",
                    "error": "scip_go_execution_failed",
                })
                report["summary"]["failed"] += 1
    report["status"] = "succeeded" if not report["summary"]["failed"] else (
        "partial" if report["summary"]["succeeded"] else "failed"
    )
    return report


def _run_python_scip(root: Path, files: list[FileFact], indexers: dict) -> dict:
    """在隔离 Linux 容器中运行 Python SCIP，并将 index.scip 合并到统一事实层。

    源码目录只读挂载，输出目录单独读写挂载；容器使用 network none，避免索引阶段
    访问外部网络。任何 Docker、依赖或 SCIP 错误都只影响语义增强，不丢失 Tree-sitter 结果。
    """
    python_files = [item for item in files if item.language == "python"]
    report = {"status": "skipped", "reason": "no_python_files", "documents": 0, "definitions": 0, "references": 0, "relationships": 0}
    if not python_files:
        return report
    # 没有项目级环境描述时不启动昂贵的语义索引；Tree-sitter 仍会提供结构事实。
    environment_files = {"pyproject.toml", "requirements.txt", "requirements-dev.txt", "setup.py", "setup.cfg", "Pipfile", "poetry.lock"}
    has_environment = any(
        path.is_file()
        and path.name in environment_files
        and not any(part in IGNORED_DIRS for part in path.relative_to(root).parts)
        for path in root.rglob("*")
    )
    if not has_environment:
        report["reason"] = "no_python_environment_manifest"
        return report
    container_status = indexers.get("python_container", {})
    docker_path = container_status.get("executable")
    if container_status.get("status") != "ready" or not docker_path:
        report.update(status="unavailable", reason=container_status.get("status", "docker_unavailable"))
        return report

    with tempfile.TemporaryDirectory(prefix="knowledge-python-scip-", dir=str(root.parent)) as output_dir:
        output = Path(output_dir) / "index.scip"
        # Docker Desktop 接受 Windows 主机路径；统一使用绝对路径，避免工作目录漂移。
        command = [
            docker_path, "run", "--rm", "--network", "none",
            "--read-only", "--cpus", "2", "--memory", "2g",
            # scip-python 会在 /tmp 创建临时虚拟环境探测目录；仅开放这块临时空间。
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=512m",
            "-v", f"{root.resolve()}:/work:ro",
            "-v", f"{Path(output_dir).resolve()}:/output:rw",
            "--entrypoint", "scip-python", settings.scip_python_image,
            "index", "--cwd", "/work", "--project-name", root.name,
            "--output", "/output/index.scip",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=settings.scip_python_timeout,
                check=False,
            )
            if result.returncode != 0 or not output.is_file():
                detail = (result.stderr or result.stdout or "index file was not generated").strip()
                raise RuntimeError(detail[-2000:])
            stats = _merge_scip_index(output, root, root, files)
            report.update(status="succeeded", reason=None, **stats)
        except subprocess.TimeoutExpired:
            report.update(status="failed", reason=f"timeout_{settings.scip_python_timeout}s")
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            # 第三方索引器 stderr 可能包含主机路径、环境变量或依赖细节，只写服务日志。
            logger.warning("python_scip_failed error_type=%s", type(exc).__name__, exc_info=True)
            report.update(status="failed", reason="python_scip_execution_failed")
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


def _architecture_facts(root: Path) -> list[ArchitectureFact]:
    """提取模块、执行入口、API、组件和资源，结果只来自可定位的静态证据。"""
    manifest_names = {
        "requirements.txt", "pyproject.toml", "package.json", "pom.xml", "build.gradle",
        "build.gradle.kts", "go.mod", "cargo.toml", "dockerfile", "compose.yml",
        "compose.yaml", "docker-compose.yml", "docker-compose.yaml", ".env", ".env.example",
    }
    config_suffixes = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf", ".properties", ".proto"}
    lockfile_names = {"package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml", "go.sum", "poetry.lock", "cargo.lock"}
    ignored = {item.lower() for item in IGNORED_DIRS}
    architecture_ignored = ignored | {"test", "tests", "__tests__", "fixtures", "examples", "example"}
    facts: dict[tuple[str, str, str | None, str, int], ArchitectureFact] = {}
    declared_modules: set[tuple[str, str]] = set()
    component_patterns = {
        "Kafka": ("messaging", re.compile(r"(?im)((?:from|import)\s+[^\n]*kafka|kafka-python|confluent[-_]kafka|aiokafka|sarama|KafkaProducer|KafkaConsumer|^\s*KAFKA[A-Z0-9_.-]*\s*[:=])")),
        "MongoDB": ("database", re.compile(r"(?im)((?:from|import)\s+[^\n]*(?:pymongo|motor|mongoengine)|go\.mongodb\.org|MongoClient|^\s*MONGO[A-Z0-9_.-]*\s*[:=])")),
        "Redis": ("cache", re.compile(r"(?im)((?:from|import)\s+[^\n]*(?:redis|aioredis)|github\.com/redis|redis-py|Redis\s*\(|^\s*REDIS[A-Z0-9_.-]*\s*[:=])")),
        "gRPC": ("rpc", re.compile(r"(?im)((?:from|import)\s+[^\n]*grpc|google\.golang\.org/grpc|grpcio|grpc\.(?:Dial|NewClient|NewServer)|grpc\.insecure_channel|^\s*GRPC[A-Z0-9_.-]*\s*[:=])")),
        "HTTP client": ("downstream_transport", re.compile(r"(?i)(requests\.|httpx\.|aiohttp|resty\.New|axios\.|fetch\s*\()")),
    }

    def add(
        fact_type: str,
        name: str,
        value: str | None,
        path: str,
        line: int,
        evidence: str,
        confidence: float = 0.9,
    ) -> None:
        safe_name = _safe_architecture_value(name)
        safe_value = _safe_architecture_value(value) if value else None
        safe_evidence = _safe_architecture_value(evidence)
        key = (fact_type, safe_name, safe_value, path, line)
        facts.setdefault(
            key,
            ArchitectureFact(fact_type, safe_name, safe_value, path, line, safe_evidence, confidence),
        )

    def downstream_name(key: str, value: str) -> str:
        service_match = re.match(r"(?i)^(.+)_SERVICE_(?:BASE_)?(?:URL|URI|ENDPOINT|TARGET)$", key)
        if service_match:
            return service_match.group(1).lower().replace("_", "-") + "-service"
        if "." in key:
            segments = [segment for segment in re.split(r"[._-]+", key.lower()) if segment]
            ignored_segments = {"url", "uri", "endpoint", "target", "base", "service", "services", "downstream", "client"}
            for segment in reversed(segments):
                if segment not in ignored_segments:
                    return segment
        stem = re.sub(r"(?i)(?:_BASE)?_(?:URL|URI|ENDPOINT|TARGET)$", "", key)
        if stem and stem.lower() not in {"url", "uri", "endpoint", "target", "base", "baseurl", "serviceurl"}:
            return stem.lower().replace("_", "-").replace(".", "-")
        parsed = urlparse(value)
        return parsed.hostname or value.split(":", 1)[0]

    def add_module(name: str, module_kind: str, path: str, line: int, evidence: str) -> None:
        """同一个逻辑模块只保留一条边界证据，避免每个源码文件重复展示。"""
        key = (module_kind, name)
        if key in declared_modules:
            return
        declared_modules.add(key)
        add("module", name, module_kind, path, line, evidence, 0.95)

    for path in sorted(root.rglob("*")):
        relative_parts = path.relative_to(root).parts
        if not path.is_file() or any(part.lower() in architecture_ignored for part in relative_parts):
            continue
        relative = path.relative_to(root).as_posix()
        lower_name = path.name.lower()
        if lower_name in lockfile_names:
            continue
        if (
            lower_name.endswith(("_pb2.py", "_pb2_grpc.py", ".pb.go", ".gen.go", "_generated.go"))
            or ".generated." in lower_name
        ):
            continue
        if path.suffix.lower() not in set(CODE_EXTENSIONS) | config_suffixes and path.name.lower() not in manifest_names:
            continue
        if path.name == "code_wiki.py" and path.parent.name == "app":
            continue
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
            text = _read_text(path)
        except OSError:
            continue

        # 模块边界优先使用语言原生声明。目录只是定位，不直接猜测业务职责。
        if lower_name == "go.mod":
            module_match = re.search(r"(?m)^\s*module\s+([^\s]+)", text)
            if module_match:
                add_module(module_match.group(1), "go_module", relative, _line_number(text, module_match.start()), module_match.group(0))
        elif lower_name == "package.json":
            try:
                package_data = json.loads(text)
            except json.JSONDecodeError:
                package_data = {}
            package_name = package_data.get("name") if isinstance(package_data, dict) else None
            if isinstance(package_name, str) and package_name:
                add_module(package_name, "javascript_package", relative, 1, f'package name: {package_name}')
            scripts = package_data.get("scripts", {}) if isinstance(package_data, dict) else {}
            if isinstance(scripts, dict):
                for script_name in ("start", "dev", "serve"):
                    command = scripts.get(script_name)
                    if isinstance(command, str) and command:
                        line = next((number for number, value in enumerate(text.splitlines(), 1) if f'"{script_name}"' in value), 1)
                        add("entrypoint", script_name, command, relative, line, f'{script_name}: {command}', 0.9)

        if path.suffix.lower() == ".py":
            package_marker = path.parent / "__init__.py"
            package_path = PurePosixPath(relative).parent
            if package_path.parts and package_marker.is_file():
                add_module(".".join(package_path.parts), "python_package", relative, 1, f"Python package: {package_path.as_posix()}")

            # 将事实行定位在处理函数上，持久化时可以直接关联 source_symbol_id。
            route_pattern = re.compile(
                r"(?ms)^[ \t]*@(?P<router>[A-Za-z_][\w.]*)\.(?P<method>get|post|put|patch|delete|options|head|route)"
                r"\(\s*[\"'](?P<route>[^\"']+)[\"'][^)]*\)\s*\n[ \t]*(?:async\s+)?def\s+(?P<handler>[A-Za-z_]\w*)"
            )
            for match in route_pattern.finditer(text):
                method = match.group("method").upper()
                method = "ROUTE" if method == "ROUTE" else method
                handler_offset = match.start("handler")
                add("http_api", match.group("handler"), f"{method} {match.group('route')}", relative,
                    _line_number(text, handler_offset), match.group(0).splitlines()[0].strip(), 0.98)

            task_pattern = re.compile(
                r"(?ms)^[ \t]*@(?P<decorator>[A-Za-z_][\w.]*(?:task|scheduled_job|cron))\b[^\n]*\n"
                r"[ \t]*(?:async\s+)?def\s+(?P<handler>[A-Za-z_]\w*)"
            )
            for match in task_pattern.finditer(text):
                add("background_job", match.group("handler"), match.group("decorator"), relative,
                    _line_number(text, match.start("handler")), match.group(0).splitlines()[0].strip(), 0.95)

            lifecycle_pattern = re.compile(
                r"(?ms)^[ \t]*@(?P<app>[A-Za-z_]\w*)\.on_event\(\s*[\"'](?P<event>startup|shutdown)[\"']\s*\)\s*\n"
                r"[ \t]*(?:async\s+)?def\s+(?P<handler>[A-Za-z_]\w*)"
            )
            for match in lifecycle_pattern.finditer(text):
                add("lifecycle_hook", match.group("handler"), match.group("event"), relative,
                    _line_number(text, match.start("handler")), match.group(0).splitlines()[0].strip(), 0.98)

            for match in re.finditer(r"(?m)^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<framework>FastAPI|Flask)\s*\(", text):
                add("entrypoint", match.group("name"), f"{match.group('framework')} application", relative,
                    _line_number(text, match.start()), match.group(0).strip(), 0.98)
            for match in re.finditer(r"(?m)^\s*if\s+__name__\s*==\s*[\"']__main__[\"']\s*:", text):
                add("entrypoint", relative, "python_main", relative, _line_number(text, match.start()), match.group(0).strip(), 1.0)

        elif path.suffix.lower() == ".go":
            package_match = re.search(r"(?m)^\s*package\s+([A-Za-z_]\w*)", text)
            package_name = package_match.group(1) if package_match else None
            if package_match:
                package_dir = PurePosixPath(relative).parent.as_posix()
                add_module(package_dir if package_dir != "." else package_name, f"go_package:{package_name}", relative,
                           _line_number(text, package_match.start()), package_match.group(0).strip())
            if package_name == "main":
                main_match = re.search(r"(?m)^\s*func\s+main\s*\(", text)
                if main_match:
                    add("entrypoint", "main", "go_main", relative, _line_number(text, main_match.start()), main_match.group(0).strip(), 1.0)

            go_route_patterns = [
                re.compile(r"(?m)\b[A-Za-z_]\w*\.(?P<method>GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD|Get|Post|Put|Patch|Delete)\(\s*[\"'](?P<route>[^\"']+)[\"']\s*,\s*(?P<handler>[A-Za-z_][\w.]*)"),
                re.compile(r"(?m)\bhttp\.HandleFunc\(\s*[\"'](?P<route>[^\"']+)[\"']\s*,\s*(?P<handler>[A-Za-z_][\w.]*)"),
            ]
            for pattern in go_route_patterns:
                for match in pattern.finditer(text):
                    method = match.groupdict().get("method") or "ANY"
                    route_label = f"{method.upper()} {match.group('route')}"
                    handler = match.group("handler")
                    # Go 匿名函数以 func 开头，它不是可跳转的符号名，改用路由标识展示。
                    name = route_label if handler == "func" else handler
                    value = "inline_handler" if handler == "func" else route_label
                    add("http_api", name, value, relative,
                        _line_number(text, match.start()), match.group(0).strip(), 0.95)

        elif path.suffix.lower() == ".proto":
            for service_match in re.finditer(r"(?m)^\s*service\s+([A-Za-z_]\w*)\s*\{", text):
                service_name = service_match.group(1)
                add("rpc_service", service_name, "grpc", relative, _line_number(text, service_match.start()), service_match.group(0).strip(), 1.0)
            for rpc_match in re.finditer(r"(?m)^\s*rpc\s+([A-Za-z_]\w*)\s*\(\s*([^)]*)\)\s+returns\s+\(\s*([^)]*)\)", text):
                add("rpc_method", rpc_match.group(1), f"{rpc_match.group(2)} -> {rpc_match.group(3)}", relative,
                    _line_number(text, rpc_match.start()), rpc_match.group(0).strip(), 1.0)

        detected_components: set[str] = set()
        for name, (category, pattern) in component_patterns.items():
            match = pattern.search(text)
            if not match:
                continue
            detected_components.add(name)
            confidence = 0.95 if path.name.lower() in manifest_names else 0.85
            add("component", name, category, relative, _line_number(text, match.start()), match.group(0), confidence)

        has_kafka = "Kafka" in detected_components
        has_mongo = "MongoDB" in detected_components
        is_config_file = path.suffix.lower() in config_suffixes or path.name.lower() in manifest_names
        is_example_config = lower_name.endswith(".example") or ".example." in lower_name

        def record_config(key: str, value: str, line: int, evidence: str) -> None:
            lower_key = key.lower()
            if re.search(r"(?i)(password|passwd|pwd|token|secret|api[_-]?key)", key) or is_example_config:
                return
            if has_kafka and "topic" in lower_key:
                add("kafka_topic_config", value, value, relative, line, evidence, 0.95)
            elif has_kafka and ("group" in lower_key or "consumer" in lower_key):
                add("kafka_consumer_group", value, value, relative, line, evidence, 0.9)
            elif has_kafka and any(term in lower_key for term in ("broker", "bootstrap", "kafka_url")):
                add("kafka_cluster", key, value, relative, line, evidence, 0.9)
            if has_mongo and "collection" in lower_key:
                add("mongo_collection", value, value, relative, line, evidence, 0.95)
            elif has_mongo and (lower_key.endswith("database") or lower_key.endswith("db") or "mongo_db" in lower_key):
                add("mongo_database", value, value, relative, line, evidence, 0.9)
            elif has_mongo and any(term in lower_key for term in ("mongo_url", "mongo_uri", "mongodb_url", "mongodb_uri")):
                add("mongo_cluster", key, value, relative, line, evidence, 0.9)
            if value.startswith(("http://", "https://")) and any(term in lower_key for term in ("url", "uri", "endpoint", "service")):
                add("downstream_http", downstream_name(key, value), value, relative, line, evidence, 0.9)
            elif "grpc" in lower_key and any(term in lower_key for term in ("target", "endpoint", "service", "addr", "host")):
                add("downstream_grpc", downstream_name(key, value), value, relative, line, evidence, 0.9)
            # ADDR/HOST/TARGET 本身不能证明传输协议，只记录为中性端点；只有后续
            # 与具体 Client/Stub 类型名称一致时，才会生成 configures_client 关联。
            if re.search(r"(?i)(?:^|_)(?:ADDR|HOST|ENDPOINT|TARGET)$", key) and value:
                add("endpoint_config", key, value, relative, line, evidence, 0.75)

        config_pattern = re.compile(r"(?im)^\s*[\"']?([A-Za-z_][A-Za-z0-9_.-]*)[\"']?\s*[:=]\s*[\"']?([^\"'\r\n#]+)")
        yaml_entries = _yaml_scalar_entries(text) if path.suffix.lower() in {".yaml", ".yml"} else []
        if yaml_entries:
            for key, value, line, evidence in yaml_entries:
                record_config(key, value, line, evidence)
        else:
            for match in config_pattern.finditer(text):
                key, value = match.group(1), match.group(2).strip().rstrip(",")
                if not is_config_file and key.upper() != key:
                    continue
                record_config(key, value, _line_number(text, match.start()), match.group(0))

        if has_kafka:
            for pattern, fact_type in [
                (re.compile(r"(?i)(?:send|publish|produce)\s*\(\s*[\"']([^\"']+)[\"']"), "kafka_topic_producer"),
                (re.compile(r"(?i)(?:subscribe|consume)\s*\(\s*(?:\[\s*)?[\"']([^\"']+)[\"']"), "kafka_topic_consumer"),
            ]:
                for match in pattern.finditer(text):
                    add(fact_type, match.group(1), match.group(1), relative, _line_number(text, match.start()), match.group(0), 0.9)
        if has_mongo:
            for pattern in [
                re.compile(r"(?i)(?:get_collection|collection)\s*\(\s*[\"']([^\"']+)[\"']"),
                re.compile(r"(?i)(?:db|database)\s*\[\s*[\"']([^\"']+)[\"']\s*\]"),
            ]:
                for match in pattern.finditer(text):
                    add("mongo_collection", match.group(1), match.group(1), relative, _line_number(text, match.start()), match.group(0), 0.9)

        if "gRPC" in detected_components:
            for pattern in [
                re.compile(r"(?i)New([A-Za-z0-9_]+)Client\s*\("),
                re.compile(r"(?i)([A-Za-z0-9_]+)Stub\s*\("),
            ]:
                for match in pattern.finditer(text):
                    add("downstream_grpc", match.group(1), None, relative, _line_number(text, match.start()), match.group(0), 0.9)

            # 只做文件内、可由赋值语句证明的轻量数据流：先记录客户端变量，
            # 再把该变量后续的方法调用关联到初始化。跨函数注入留给后续 Agent。
            client_patterns = [
                re.compile(r"(?m)\b(?P<variable>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*:?=\s*(?:[A-Za-z_]\w*\.)?New(?P<service>[A-Za-z_]\w*)Client\s*\("),
                re.compile(r"(?m)\b(?P<variable>(?:self\.)?[A-Za-z_]\w*)\s*=\s*(?:[A-Za-z_]\w*\.)?(?P<service>[A-Za-z_]\w*)Stub\s*\("),
                re.compile(r"(?m)\b(?:const|let|var)\s+(?P<variable>[A-Za-z_$][\w$]*)\s*=\s*new\s+(?:[A-Za-z_$][\w$]*\.)?(?P<service>[A-Za-z_$][\w$]*)Client\s*\("),
            ]
            client_bindings: list[tuple[str, str]] = []
            for pattern in client_patterns:
                for match in pattern.finditer(text):
                    variable = match.group("variable")
                    service = match.group("service")
                    line = _line_number(text, match.start())
                    client_bindings.append((variable, service))
                    source_line = text.splitlines()[line - 1].strip() if line <= len(text.splitlines()) else match.group(0)
                    add("client_initialization", service, variable, relative, line, source_line, 0.96)
            for variable, service in client_bindings:
                field_name = variable.rsplit(".", 1)[-1]
                receiver = rf"(?:[A-Za-z_$][\w$]*\.)?{re.escape(field_name)}" if "." in variable else re.escape(variable)
                call_pattern = re.compile(rf"\b{receiver}\.(?P<method>[A-Za-z_$][\w$]*)\s*\(")
                for match in call_pattern.finditer(text):
                    method = match.group("method")
                    if method.lower() in {"close", "connect", "waitforready"}:
                        continue
                    add(
                        "downstream_call", f"{service}.{method}", variable, relative,
                        _line_number(text, match.start()), match.group(0), 0.95,
                    )

        if not is_config_file:
            http_target = re.compile(r"(?i)(?:base_url|service_url|endpoint|url)\s*[:=(]\s*[\"'](https?://[^\"']+)[\"']")
            for match in http_target.finditer(text):
                value = match.group(1)
                add("downstream_http", downstream_name("url", value), value, relative, _line_number(text, match.start()), match.group(0), 0.9)

    return sorted(facts.values(), key=lambda item: (item.fact_type, item.name, item.path, item.line))


def _architecture_identity(fact: ArchitectureFact) -> str:
    """生成仅用于同类架构事实匹配的保守标识，不作为展示名称。"""
    candidate = fact.name
    if fact.fact_type.startswith("downstream_") and fact.value:
        parsed = urlparse(fact.value)
        if parsed.hostname:
            candidate = fact.name or parsed.hostname
    normalized = re.sub(r"[^a-z0-9]", "", candidate.lower())
    if fact.fact_type in {"downstream_grpc", "client_initialization"}:
        normalized = normalized.replace("grpc", "")
    if fact.fact_type == "endpoint_config":
        for suffix in ("endpoint", "target", "address", "addr", "host", "port"):
            if normalized.endswith(suffix) and len(normalized) > len(suffix):
                normalized = normalized[:-len(suffix)]
                break
    # Service/Client/Stub 是代码生成器常见后缀，配置名通常不会携带这些词。
    for suffix in ("serviceclient", "servicestub", "service", "client", "stub"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            normalized = normalized[:-len(suffix)]
            break
    return normalized


def _is_config_fact(fact: ArchitectureFact) -> bool:
    suffix = PurePosixPath(fact.path).suffix.lower()
    name = PurePosixPath(fact.path).name.lower()
    return suffix in {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf", ".properties"} or name.startswith(".env")


def _architecture_links(facts: list[ArchitectureFact]) -> list[ArchitectureLink]:
    """用精确资源名和局部客户端变量构建可验证关联，不猜测动态依赖。"""
    links: dict[tuple, ArchitectureLink] = {}

    def add(source: ArchitectureFact, target: ArchitectureFact, relation_type: str, evidence: str, confidence: float) -> None:
        if _architecture_fact_key(source) == _architecture_fact_key(target):
            return
        key = (_architecture_fact_key(source), _architecture_fact_key(target), relation_type)
        links.setdefault(key, ArchitectureLink(key[0], key[1], relation_type, evidence, confidence))

    initializations = [fact for fact in facts if fact.fact_type == "client_initialization"]
    calls = [fact for fact in facts if fact.fact_type == "downstream_call"]
    for initialization in initializations:
        # 同一文件、同一客户端绑定才连边。模块级客户端常在 main 中初始化，
        # 其调用函数可能写在初始化语句之前，因此不能用源码行号代表执行先后。
        for call in calls:
            if (
                call.path == initialization.path
                and call.value == initialization.value
            ):
                add(
                    initialization, call, "client_invokes",
                    f"变量 {initialization.value} 从初始化流向 {call.name}", 0.96,
                )
        init_identity = _architecture_identity(initialization)
        if not init_identity:
            continue
        for configured in facts:
            if (
                configured.fact_type in {"downstream_grpc", "endpoint_config"}
                and _is_config_fact(configured)
                and _architecture_identity(configured) == init_identity
            ):
                add(
                    configured, initialization, "configures_client",
                    f"配置目标 {configured.name} 与客户端 {initialization.name} 名称一致", 0.9,
                )

    exact_resource_pairs = {
        "kafka_topic_config": {"kafka_topic_producer", "kafka_topic_consumer"},
        "mongo_collection": {"mongo_collection"},
    }
    for configured in facts:
        target_types = exact_resource_pairs.get(configured.fact_type)
        if not target_types or not _is_config_fact(configured):
            continue
        configured_identity = (configured.value or configured.name).strip().casefold()
        if not configured_identity:
            continue
        for usage in facts:
            if usage.fact_type not in target_types or _is_config_fact(usage):
                continue
            usage_identity = (usage.value or usage.name).strip().casefold()
            if configured_identity == usage_identity:
                add(
                    configured, usage, "configures_usage",
                    f"配置值 {configured.value or configured.name} 与代码使用值完全一致", 0.98,
                )

    return sorted(links.values(), key=lambda item: (item.relation_type, item.source_fact_key, item.target_fact_key))


def _architecture_source_symbol(fact: ArchitectureFact, files: list[FileFact]) -> SymbolFact | None:
    """把架构证据关联到包含该行的最具体符号，无法确定时回退到文件节点。"""
    file_fact = next((item for item in files if item.path == fact.path), None)
    if not file_fact:
        return None
    business_symbols = [item for item in file_fact.symbols if item.kind != "file"]
    containing = [item for item in business_symbols if item.start_line <= fact.line <= item.end_line]
    if containing:
        return min(containing, key=lambda item: (item.end_line - item.start_line, -item.start_line))
    return next((item for item in file_fact.symbols if item.kind == "file"), None)


def _file_inventory(root: Path) -> list[dict]:
    """记录项目文件资产，供 Agent 先发现文件类型和解析覆盖率，不把配置误当成代码符号。"""
    config_names = {"dockerfile", "makefile", "procfile", "jenkinsfile", "compose.yml", "compose.yaml", "docker-compose.yml", "docker-compose.yaml"}
    config_suffixes = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf", ".properties", ".env"}
    language_names = {".py": "python", ".go": "go", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript", ".java": "java"}
    inventory = []
    ignored = {item.lower() for item in IGNORED_DIRS}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part.lower() in ignored for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root).as_posix()
        name = path.name.lower()
        suffix = path.suffix.lower()
        if suffix not in language_names and suffix not in config_suffixes and name not in config_names:
            role, parser_status, parser_name = "other", "unclassified", None
        elif suffix in language_names:
            role, parser_status, parser_name = "source_code", "parsed", "tree-sitter/scip"
        elif name in config_names:
            role, parser_status, parser_name = "deployment_config", "partial", "format-specific config parser"
        else:
            role, parser_status, parser_name = "config", "partial", "yaml/json config parser" if suffix in {".yaml", ".yml", ".json"} else "text config parser"
        try:
            size = path.stat().st_size
            line_count = len(_read_text(path).splitlines()) if size <= 2 * 1024 * 1024 else None
            # 小文本沿用解码后的哈希，便于读取时复核；大文件用流式原始字节哈希参与版本号。
            content_hash = _hash_text(_read_text(path)) if size <= 2 * 1024 * 1024 else _hash_file_bytes(path)
        except OSError:
            size, line_count, content_hash = 0, None, None
        detected_format = None
        if name in {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}:
            detected_format = "docker_compose"
        elif name == "values.yaml" or name.endswith("-values.yaml"):
            detected_format = "helm_values"
        elif "otel" in name or "opentelemetry" in name:
            detected_format = "possible_observability_config"
        inventory.append({"path": relative, "file_name": path.name, "extension": suffix, "file_role": role, "detected_format": detected_format, "language": language_names.get(suffix), "size_bytes": size, "line_count": line_count, "content_hash": content_hash, "parser_status": parser_status, "parser_name": parser_name})
    return inventory


def scan_project(root_path: str, workspace_id: str | None = None) -> dict:
    """扫描项目并返回可入库的事实；不调用模型，便于重复运行和自动测试。"""
    root = Path(root_path).expanduser().resolve()
    workspace_id = workspace_id or settings.workspace_id
    if not root.is_dir():
        raise ValueError("project path must be an existing directory")
    commit_hash, commit_source = _git_commit(root)
    files = [_extract_file(root, path) for path in _iter_source_files(root)]
    file_inventory = _file_inventory(root)
    # 非 Git 项目必须覆盖全部可读取资产，而不只是进入符号索引的源码；否则只修改
    # YAML/Compose/Helm 时版本号不变，会错误复用旧快照目录。
    if commit_source == "content_scan":
        snapshot = "\n".join(
            f"{item['path']}:{item.get('content_hash') or 'unreadable'}:{item.get('size_bytes', 0)}"
            for item in sorted(file_inventory, key=lambda item: item["path"])
        )
        commit_hash = "scan-" + _hash_text(snapshot)[:16]
    indexers = _scip_indexer_status()
    scip_report = {
        "go": _run_go_scip(root, files, indexers),
        "python": _run_python_scip(root, files, indexers),
    }
    components = _component_facts(root, files)
    architecture_facts = _architecture_facts(root)
    architecture_links = _architecture_links(architecture_facts)
    config_facts = parse_project_configs(root)
    return {
        # 旧的默认空间沿用历史 ID；其他空间把 workspace 纳入身份，避免同仓库相互覆盖。
        "project_id": str(uuid.uuid5(ID_NAMESPACE, str(root) if workspace_id == settings.workspace_id else f"{workspace_id}:{root}")),
        "workspace_id": workspace_id,
        "project_name": root.name,
        "root_path": str(root),
        "commit_hash": commit_hash,
        "commit_source": commit_source,
        "scip_indexers": indexers,
        "scip": scip_report,
        "files": files,
        "file_inventory": file_inventory,
        "components": components,
        "architecture_facts": architecture_facts,
        "architecture_links": architecture_links,
        "config_facts": config_facts,
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
        # 项目也必须和 RAG 文档一样绑定工作空间；旧项目统一迁移到当前 MVP 空间。
        conn.execute("ALTER TABLE code_projects ADD COLUMN IF NOT EXISTS workspace_id TEXT")
        conn.execute("ALTER TABLE code_projects ADD COLUMN IF NOT EXISTS owner_user_id TEXT")
        conn.execute("ALTER TABLE code_projects ADD COLUMN IF NOT EXISTS created_by_user_id TEXT")
        conn.execute("ALTER TABLE code_projects ADD COLUMN IF NOT EXISTS access_scope TEXT NOT NULL DEFAULT 'workspace'")
        conn.execute("ALTER TABLE code_projects ADD COLUMN IF NOT EXISTS repository_key TEXT")
        conn.execute("UPDATE code_projects SET workspace_id = %s WHERE workspace_id IS NULL", (settings.workspace_id,))
        conn.execute("UPDATE code_projects SET owner_user_id = %s WHERE owner_user_id IS NULL", (settings.current_user_id,))
        conn.execute("UPDATE code_projects SET created_by_user_id = owner_user_id WHERE created_by_user_id IS NULL")
        conn.execute("ALTER TABLE code_projects ALTER COLUMN workspace_id SET NOT NULL")
        conn.execute("ALTER TABLE code_projects ALTER COLUMN owner_user_id SET NOT NULL")
        conn.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_code_projects_access_scope') THEN
                    ALTER TABLE code_projects ADD CONSTRAINT ck_code_projects_access_scope
                    CHECK (access_scope IN ('private', 'workspace'));
                END IF;
            END $$
            """
        )
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
            CREATE TABLE IF NOT EXISTS code_project_snapshots (
                -- 项目与 Commit 共同标识一个不可变源码/索引快照。
                project_id UUID NOT NULL REFERENCES code_projects(project_id) ON DELETE CASCADE,
                commit_hash TEXT NOT NULL,
                -- 该 Commit 对应的不可变托管源码目录；Agent 按此路径读取原文。
                root_path TEXT NOT NULL,
                -- 文件资产、配置事实和 SCIP 诊断均绑定到 Commit，不能只保存在项目当前态。
                scan_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (project_id, commit_hash)
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_architecture_facts (
                fact_id UUID PRIMARY KEY,
                project_id UUID NOT NULL REFERENCES code_projects(project_id) ON DELETE CASCADE,
                commit_hash TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                name TEXT NOT NULL,
                value TEXT,
                source_path TEXT NOT NULL,
                source_line INTEGER NOT NULL,
                evidence TEXT NOT NULL,
                source_symbol_id UUID REFERENCES code_symbols(symbol_id) ON DELETE SET NULL,
                confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                UNIQUE(project_id, commit_hash, fact_type, name, value, source_path, source_line)
            )
        """)
        conn.execute("ALTER TABLE code_architecture_facts ADD COLUMN IF NOT EXISTS source_symbol_id UUID REFERENCES code_symbols(symbol_id) ON DELETE SET NULL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_architecture_links (
                link_id UUID PRIMARY KEY,
                project_id UUID NOT NULL REFERENCES code_projects(project_id) ON DELETE CASCADE,
                commit_hash TEXT NOT NULL,
                source_fact_id UUID NOT NULL REFERENCES code_architecture_facts(fact_id) ON DELETE CASCADE,
                target_fact_id UUID NOT NULL REFERENCES code_architecture_facts(fact_id) ON DELETE CASCADE,
                relation_type TEXT NOT NULL,
                evidence TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                UNIQUE(project_id, commit_hash, source_fact_id, target_fact_id, relation_type)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_code_files_project ON code_files(project_id, commit_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_code_symbols_name ON code_symbols(project_id, name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_code_relations_source ON code_relations(source_symbol_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_code_relations_target ON code_relations(target_symbol_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_code_architecture_facts_project ON code_architecture_facts(project_id, commit_hash, fact_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_code_architecture_links_project ON code_architecture_links(project_id, commit_hash, relation_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_code_project_snapshots_root ON code_project_snapshots(root_path)")
        # 规范化仓库键由应用生成，唯一索引防止并发或大小写变体建立重复项目。
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_code_projects_workspace_repository_key "
            "ON code_projects(workspace_id, repository_key) WHERE repository_key IS NOT NULL"
        )
        # 旧项目只有当前 root_path；先把它登记为第一个快照，保持升级中的 Agent 可读。
        conn.execute(
            """INSERT INTO code_project_snapshots (project_id, commit_hash, root_path, scan_metadata)
               SELECT project_id, current_commit, root_path, scan_metadata FROM code_projects
               ON CONFLICT (project_id, commit_hash) DO NOTHING"""
        )
        # 旧实现可能保留非当前 Commit 的事实，却没有对应源码目录。把它们指向当前
        # root_path 会制造假快照，因此升级时只删除这些已无法复核的孤立历史事实。
        for table in (
            "code_architecture_links", "code_relations", "code_symbols",
            "code_files", "code_components", "code_architecture_facts",
        ):
            conn.execute(f"""
                DELETE FROM {table} item
                WHERE NOT EXISTS (
                    SELECT 1 FROM code_project_snapshots snapshot
                    WHERE snapshot.project_id = item.project_id
                      AND snapshot.commit_hash = item.commit_hash
                )
            """)
        # 所有版本化事实必须属于一个已登记快照；NOT VALID 先兼容升级，再立即验证现有数据。
        for constraint, table in (
            ("fk_code_files_snapshot", "code_files"),
            ("fk_code_symbols_snapshot", "code_symbols"),
            ("fk_code_relations_snapshot", "code_relations"),
            ("fk_code_components_snapshot", "code_components"),
            ("fk_code_architecture_facts_snapshot", "code_architecture_facts"),
            ("fk_code_architecture_links_snapshot", "code_architecture_links"),
        ):
            conn.execute(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = '{constraint}') THEN
                        ALTER TABLE {table} ADD CONSTRAINT {constraint}
                        FOREIGN KEY (project_id, commit_hash)
                        REFERENCES code_project_snapshots(project_id, commit_hash)
                        ON DELETE CASCADE NOT VALID;
                    END IF;
                END $$
            """)
            conn.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {constraint}")
        conn.execute("COMMENT ON TABLE code_projects IS '代码 Wiki 项目事实层：项目和当前扫描版本'")
        conn.execute("COMMENT ON COLUMN code_projects.scan_metadata IS '最近一次扫描的来源、SCIP CLI 状态、模块结果、索引哈希和错误摘要'")
        conn.execute("COMMENT ON COLUMN code_projects.workspace_id IS '代码项目所属工作空间；跨空间查询永远不允许'")
        conn.execute("COMMENT ON COLUMN code_projects.owner_user_id IS '代码项目所有者；重新扫描不会隐式转移所有权'")
        conn.execute("COMMENT ON COLUMN code_projects.created_by_user_id IS '首次扫描或导入代码项目的用户'")
        conn.execute("COMMENT ON COLUMN code_projects.access_scope IS '项目访问范围：private 仅所有者/ACL，workspace 对空间成员开放'")
        conn.execute("COMMENT ON COLUMN code_projects.repository_key IS '工作空间内大小写无关的稳定仓库身份键，用于幂等导入和并发约束'")
        conn.execute("COMMENT ON TABLE code_files IS '代码 Wiki 文件事实：路径、语言、内容哈希和 Commit'")
        conn.execute("COMMENT ON TABLE code_symbols IS '代码 Wiki 符号事实：类、函数、方法和精确源代码范围'")
        conn.execute("COMMENT ON TABLE code_relations IS '代码 Wiki 关系事实：导入、调用和后续跨项目关系'")
        conn.execute("COMMENT ON TABLE code_components IS '代码 Wiki 组件识别结果及其依赖文件/初始化代码证据'")
        conn.execute("COMMENT ON TABLE code_architecture_facts IS '架构理解层确定性事实：组件、消息资源、数据库资源和下游服务及其代码证据'")
        conn.execute("COMMENT ON COLUMN code_architecture_facts.fact_id IS '架构事实唯一 ID'")
        conn.execute("COMMENT ON COLUMN code_architecture_facts.project_id IS '事实所属代码项目 ID'")
        conn.execute("COMMENT ON COLUMN code_architecture_facts.commit_hash IS '事实对应的 Git Commit 或内容扫描版本'")
        conn.execute("COMMENT ON COLUMN code_architecture_facts.fact_type IS '事实类型，例如 component、kafka_topic_producer、mongo_collection 或 downstream_http'")
        conn.execute("COMMENT ON COLUMN code_architecture_facts.name IS '组件、资源或下游服务的可读名称'")
        conn.execute("COMMENT ON COLUMN code_architecture_facts.value IS '脱敏后的配置值、资源名或目标地址'")
        conn.execute("COMMENT ON COLUMN code_architecture_facts.source_path IS '产生该事实的项目内相对文件路径'")
        conn.execute("COMMENT ON COLUMN code_architecture_facts.source_line IS '产生该事实的源码或配置行号，从 1 开始'")
        conn.execute("COMMENT ON COLUMN code_architecture_facts.evidence IS '经过脱敏和截断的原始证据片段'")
        conn.execute("COMMENT ON COLUMN code_architecture_facts.source_symbol_id IS '证据所在或紧邻的代码符号，用于从架构事实跳转到定义与关系'")
        conn.execute("COMMENT ON COLUMN code_architecture_facts.confidence IS '确定性规则给出的证据置信度，范围 0 到 1'")
        conn.execute("COMMENT ON TABLE code_architecture_links IS '配置、客户端初始化和实际资源调用之间的可验证架构关联'")
        conn.execute("COMMENT ON TABLE code_project_snapshots IS '代码项目不可变 Commit 快照；固定 Agent 请求期间的源码与扫描元数据'")
        conn.execute("COMMENT ON COLUMN code_architecture_links.relation_type IS '关联类型，例如 configures_client、client_invokes 或 configures_usage'")
        conn.execute("COMMENT ON COLUMN code_architecture_links.evidence IS '建立关联所依据的变量、服务名或精确资源值证据'")


def _symbol_id(project_id: str, commit_hash: str, path: str, qualified: str, line: int) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"{project_id}:{commit_hash}:{path}:{qualified}:{line}"))


def _architecture_fact_id(project_id: str, commit_hash: str, fact: ArchitectureFact) -> str:
    """事实 ID 由内容与位置稳定生成，供架构关联边幂等引用。"""
    return str(uuid.uuid5(ID_NAMESPACE, f"arch:{project_id}:{commit_hash}:{_architecture_fact_key(fact)!r}"))


def persist_scan(scan: dict) -> dict:
    """以 Commit 为边界幂等写入扫描结果；同一版本重复扫描不会叠加关系。"""
    project_id = scan["project_id"]
    commit_hash = scan["commit_hash"]
    scan_metadata = {
        "scip_indexers": scan.get("scip_indexers", {}),
        "scip": scan.get("scip", {}),
        "source": scan.get("source", {"type": "local", "path": scan["root_path"]}),
        "file_inventory": scan.get("file_inventory", [])[:10000],
        "config_facts": scan.get("config_facts", [])[:20000],
    }
    with connection() as conn:
        conn.execute(
            """INSERT INTO code_projects
            (project_id, project_name, root_path, current_commit, current_commit_source,
             workspace_id, owner_user_id, created_by_user_id, access_scope, repository_key, scan_metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'workspace', %s, %s::jsonb)
            ON CONFLICT (project_id) DO UPDATE SET
              project_name = EXCLUDED.project_name, root_path = EXCLUDED.root_path,
              current_commit = EXCLUDED.current_commit, current_commit_source = EXCLUDED.current_commit_source,
              workspace_id = EXCLUDED.workspace_id,
              owner_user_id = code_projects.owner_user_id,
              created_by_user_id = COALESCE(code_projects.created_by_user_id, EXCLUDED.created_by_user_id),
              repository_key = COALESCE(code_projects.repository_key, EXCLUDED.repository_key),
              scan_metadata = EXCLUDED.scan_metadata,
              updated_at = NOW()""",
            (project_id, scan["project_name"], scan["root_path"], commit_hash, scan["commit_source"],
             scan.get("workspace_id", settings.workspace_id), scan.get("owner_user_id", settings.current_user_id),
             scan.get("created_by_user_id", scan.get("owner_user_id", settings.current_user_id)),
             scan.get("source", {}).get("repository_key"),
             json.dumps(scan_metadata, ensure_ascii=False)),
        )
        conn.execute(
            """INSERT INTO code_project_snapshots (project_id, commit_hash, root_path, scan_metadata)
               VALUES (%s, %s, %s, %s::jsonb)
               ON CONFLICT (project_id, commit_hash) DO UPDATE SET
                 root_path = EXCLUDED.root_path,
                 scan_metadata = EXCLUDED.scan_metadata""",
            (project_id, commit_hash, scan["root_path"], json.dumps(scan_metadata, ensure_ascii=False)),
        )
        # 重新扫描同一 Commit 时先替换该版本事实，避免旧的符号/关系残留。
        conn.execute("DELETE FROM code_relations WHERE project_id = %s AND commit_hash = %s", (project_id, commit_hash))
        conn.execute("DELETE FROM code_symbols WHERE project_id = %s AND commit_hash = %s", (project_id, commit_hash))
        conn.execute("DELETE FROM code_files WHERE project_id = %s AND commit_hash = %s", (project_id, commit_hash))
        conn.execute("DELETE FROM code_components WHERE project_id = %s AND commit_hash = %s", (project_id, commit_hash))
        conn.execute("DELETE FROM code_architecture_facts WHERE project_id = %s AND commit_hash = %s", (project_id, commit_hash))

        symbol_ids: dict[str, str] = {}
        file_ids: dict[str, str] = {}
        inserted_symbol_ids: set[str] = set()
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
                # Tree-sitter 与 SCIP 可能对同一事实各产出一次；稳定 ID 相同时只写一条，
                # 防止单个索引器的重复 occurrence 让整个项目事务回滚。
                if symbol.symbol_id in inserted_symbol_ids:
                    continue
                inserted_symbol_ids.add(symbol.symbol_id)
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
        architecture_fact_ids: dict[tuple[str, str, str | None, str, int], str] = {}
        for fact in scan.get("architecture_facts", []):
            source_symbol = _architecture_source_symbol(fact, scan["files"])
            fact_id = _architecture_fact_id(project_id, commit_hash, fact)
            architecture_fact_ids[_architecture_fact_key(fact)] = fact_id
            conn.execute(
                """INSERT INTO code_architecture_facts
                (fact_id, project_id, commit_hash, fact_type, name, value, source_path, source_line, evidence, source_symbol_id, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING""",
                (fact_id, project_id, commit_hash, fact.fact_type, fact.name, fact.value,
                 fact.path, fact.line, fact.evidence, source_symbol.symbol_id if source_symbol else None, fact.confidence),
            )
        for link in scan.get("architecture_links", []):
            source_fact_id = architecture_fact_ids.get(link.source_fact_key)
            target_fact_id = architecture_fact_ids.get(link.target_fact_key)
            if not source_fact_id or not target_fact_id:
                continue
            link_id = str(uuid.uuid5(
                ID_NAMESPACE,
                f"arch-link:{project_id}:{commit_hash}:{source_fact_id}:{target_fact_id}:{link.relation_type}",
            ))
            conn.execute(
                """INSERT INTO code_architecture_links
                (link_id, project_id, commit_hash, source_fact_id, target_fact_id, relation_type, evidence, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING""",
                (link_id, project_id, commit_hash, source_fact_id, target_fact_id,
                 link.relation_type, link.evidence, link.confidence),
            )
    return {
        "project_id": project_id, "project_name": scan["project_name"], "commit_hash": commit_hash,
        "file_count": len(scan["files"]),
        "symbol_count": sum(len(file_fact.symbols) for file_fact in scan["files"]),
        "relation_count": sum(len(file_fact.relations) for file_fact in scan["files"]),
        "component_count": len(scan["components"]),
        "architecture_fact_count": len(scan.get("architecture_facts", [])),
        "architecture_link_count": len(scan.get("architecture_links", [])),
        "scip_indexers": scan.get("scip_indexers", {}),
        "scip": scan.get("scip", {}),
        "file_inventory_count": len(scan.get("file_inventory", [])),
        "config_fact_count": len(scan.get("config_facts", [])),
    }


@contextmanager
def repository_import_lock(workspace_id: str, resource_key: str):
    """使用 PostgreSQL session advisory lock 串行化同一代码项目的导入。"""
    lock_key = int.from_bytes(
        hashlib.sha256(f"code-import:{workspace_id}:{resource_key.casefold()}".encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=True,
    )
    lock_connection = psycopg.connect(settings.database_url, autocommit=True)
    acquired = False
    try:
        acquired = lock_connection.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,)).fetchone()[0]
        if not acquired:
            raise RepositoryImportBusy("repository import already running")
        yield
    finally:
        try:
            if acquired:
                lock_connection.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
        finally:
            lock_connection.close()


def import_and_scan_github_repository(
    repository_url: str,
    workspace_id: str,
    owner_user_id: str,
) -> dict:
    """以不可变 Commit 快照导入 GitHub；失败时不改变当前 DB 指针和旧源码。"""
    canonical_url, owner, repository = normalize_github_url(repository_url)
    managed_root = DEFAULT_REPOSITORY_ROOT.resolve()
    if workspace_id != settings.workspace_id:
        managed_root = managed_root / workspace_id
    managed_root.mkdir(parents=True, exist_ok=True)
    # GitHub 仓库身份大小写无关，避免 URL 大小写变体绕过项目锁和幂等 ID。
    repository_key = github_repository_key(owner, repository)
    stable_identity_path = managed_root / repository_key
    project_id = managed_code_project_id(stable_identity_path, workspace_id)

    with repository_import_lock(workspace_id, f"project:{project_id}"):
        with connection() as conn:
            existed = conn.execute(
                "SELECT 1 FROM code_projects WHERE project_id = %s AND workspace_id = %s",
                (project_id, workspace_id),
            ).fetchone() is not None

        staging_root = Path(tempfile.mkdtemp(prefix=f".{repository_key}-", dir=managed_root))
        staging_repository = staging_root / "repository"
        promoted_path: Path | None = None
        promoted_here = False
        try:
            _run_git(["git", "clone", "--depth", "1", canonical_url, str(staging_repository)])
            scan = scan_project(str(staging_repository), workspace_id)
            commit_hash = str(scan["commit_hash"])
            safe_commit = re.sub(r"[^A-Za-z0-9._-]", "-", commit_hash)[:80]
            snapshots_root = managed_root / ".snapshots" / repository_key
            snapshots_root.mkdir(parents=True, exist_ok=True)
            promoted_path = snapshots_root / safe_commit
            if promoted_path.exists():
                existing_commit, _ = _git_commit(promoted_path)
                if existing_commit != commit_hash:
                    raise RuntimeError("existing repository snapshot does not match its commit")
                _remove_managed_tree(staging_repository)
            else:
                os.replace(staging_repository, promoted_path)
                promoted_here = True

            # project_id 来自稳定仓库身份，不受随机 staging 或 Commit 目录影响。
            scan.update({
                "project_id": project_id,
                "project_name": repository,
                "root_path": str(promoted_path.resolve()),
                "workspace_id": workspace_id,
                "owner_user_id": owner_user_id,
                "created_by_user_id": owner_user_id,
                "source": {
                    "type": "github",
                    "repository_url": canonical_url.removesuffix(".git"),
                    "owner": owner,
                    "repository": repository,
                    "repository_key": repository_key,
                },
            })
            result = persist_scan(scan)
            return {
                "import_action": "updated" if existed else "cloned",
                "repository_url": canonical_url.removesuffix(".git"),
                **result,
            }
        except Exception:
            # 只有本次新建且尚未被 DB 激活的快照才回收；既有同 Commit 快照不动。
            if promoted_here and promoted_path and promoted_path.exists():
                _remove_managed_tree(promoted_path)
            raise
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)


def list_code_projects(workspace_id: str | None = None, user_id: str | None = None) -> list[dict]:
    """只列出服务端已授权空间中的项目，避免前端看到其他空间项目。"""
    workspace_id = workspace_id or settings.workspace_id
    user_id = user_id or settings.current_user_id
    with connection() as conn:
        result = conn.execute(
            """SELECT project_id, project_name, root_path, current_commit, current_commit_source,
                      status, scan_metadata, workspace_id, owner_user_id, access_scope, created_at, updated_at
               FROM code_projects
               WHERE workspace_id = %s AND status = 'active'
                 AND (access_scope = 'workspace' OR owner_user_id = %s
                      OR EXISTS (SELECT 1 FROM code_project_access a WHERE a.project_id = code_projects.project_id AND a.user_id = %s AND a.status = 'active'))
               ORDER BY updated_at DESC""", (workspace_id, user_id, user_id)
        )
        columns = [desc.name for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]


def delete_code_projects(
    project_ids: list[str],
    workspace_id: str | None = None,
    user_id: str | None = None,
    workspace_role: str | None = None,
) -> dict:
    """批量删除代码项目及其托管副本；外部源码目录永远不会被删除。"""
    unique_ids = list(dict.fromkeys(project_ids))
    if not unique_ids:
        return {"deleted": [], "missing": [], "filesystem_errors": []}
    managed_root = DEFAULT_REPOSITORY_ROOT.resolve()
    deleted: list[str] = []
    missing: list[str] = []
    filesystem_errors: list[dict] = []
    workspace_id = workspace_id or settings.workspace_id
    user_id = user_id or settings.current_user_id
    for project_id in unique_ids:
        # 删除与导入竞争同一把项目锁，避免旧目录清理误删刚完成的新快照。
        with repository_import_lock(workspace_id, f"project:{project_id}"):
            repository_paths: list[Path] = []
            with connection() as conn:
                row = conn.execute(
                    """SELECT project_id, root_path FROM code_projects
                       WHERE project_id = %s AND workspace_id = %s
                         AND (%s IN ('owner', 'admin') OR (%s = 'editor' AND owner_user_id = %s) OR EXISTS (
                             SELECT 1 FROM code_project_access a
                             WHERE a.project_id = code_projects.project_id AND a.user_id = %s
                               AND a.permission = 'admin' AND a.status = 'active'
                         ))""",
                    (project_id, workspace_id, workspace_role or '', workspace_role or '', user_id, user_id),
                ).fetchone()
                if not row:
                    missing.append(project_id)
                    continue
                snapshot_paths = conn.execute(
                    "SELECT root_path FROM code_project_snapshots WHERE project_id = %s",
                    (project_id,),
                ).fetchall()
                for raw_path in {row[1], *(item[0] for item in snapshot_paths)}:
                    candidate = Path(raw_path).expanduser().resolve()
                    if candidate != managed_root and managed_root in candidate.parents:
                        repository_paths.append(candidate)
                # ON DELETE CASCADE 会同步删除项目下的文件、符号、关系和架构事实。
                conn.execute("DELETE FROM code_projects WHERE project_id = %s", (project_id,))
                deleted.append(str(row[0]))

            # connection 上下文已提交删除，但仍持有项目锁，因此同项目不能在清理期间重建。
            for repository_path in repository_paths:
                try:
                    if repository_path.exists():
                        _remove_managed_tree(repository_path)
                except OSError:
                    # 前端只需要知道哪个托管快照清理失败；绝对路径和 OS 错误保留在日志。
                    logger.exception("managed_repository_cleanup_failed path=%s", repository_path)
                    filesystem_errors.append({
                        "repository": repository_path.name,
                        "error_code": "MANAGED_REPOSITORY_CLEANUP_FAILED",
                    })
    return {"deleted": deleted, "missing": missing, "filesystem_errors": filesystem_errors}


def get_code_overview(project_id: str, workspace_id: str | None = None) -> dict | None:
    """读取项目详情时再次校验空间，不能只依赖上游列表接口。"""
    workspace_id = workspace_id or settings.workspace_id
    with connection() as conn:
        project = conn.execute(
            """SELECT project_id, project_name, root_path, current_commit, current_commit_source, status, scan_metadata
               FROM code_projects WHERE project_id = %s AND workspace_id = %s""", (project_id, workspace_id)
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
            "architecture_facts": _list_architecture_facts(conn, project_id, project[3]),
            "architecture_links": _list_architecture_links(conn, project_id, project[3]),
        }


def _list_architecture_facts(conn, project_id: str, commit_hash: str) -> list[dict]:
    """读取当前 Commit 的架构事实，供项目概览和后续 Agent 查询工具复用。"""
    result = conn.execute(
        """SELECT fact_id, fact_type, name, value, source_path, source_line, evidence, source_symbol_id, confidence
           FROM code_architecture_facts
           WHERE project_id = %s AND commit_hash = %s
           ORDER BY fact_type, name, source_path, source_line""",
        (project_id, commit_hash),
    )
    columns = [desc.name for desc in result.description]
    return [dict(zip(columns, row)) for row in result.fetchall()]


def _list_architecture_links(conn, project_id: str, commit_hash: str) -> list[dict]:
    """返回关联边及两端事实，Agent 不需要再用名称进行二次猜测。"""
    result = conn.execute(
        """SELECT l.link_id, l.relation_type, l.evidence, l.confidence,
                  sf.fact_id AS source_fact_id, sf.fact_type AS source_fact_type,
                  sf.name AS source_name, sf.value AS source_value,
                  sf.source_path AS source_path, sf.source_line AS source_line,
                  sf.source_symbol_id AS source_symbol_id,
                  tf.fact_id AS target_fact_id, tf.fact_type AS target_fact_type,
                  tf.name AS target_name, tf.value AS target_value,
                  tf.source_path AS target_path, tf.source_line AS target_line,
                  tf.source_symbol_id AS target_symbol_id
           FROM code_architecture_links l
           JOIN code_architecture_facts sf ON sf.fact_id = l.source_fact_id
           JOIN code_architecture_facts tf ON tf.fact_id = l.target_fact_id
           WHERE l.project_id = %s AND l.commit_hash = %s
           ORDER BY l.relation_type, sf.name, tf.name""",
        (project_id, commit_hash),
    )
    columns = [desc.name for desc in result.description]
    return [dict(zip(columns, row)) for row in result.fetchall()]


def list_code_architecture(project_id: str, commit_hash: str | None = None) -> list[dict]:
    """返回指定 Commit 的架构事实；未指定时仅供页面读取当前版本。"""
    with connection() as conn:
        if not commit_hash:
            project = conn.execute(
                "SELECT current_commit FROM code_projects WHERE project_id = %s",
                (project_id,),
            ).fetchone()
            commit_hash = project[0] if project else None
        return _list_architecture_facts(conn, project_id, commit_hash) if commit_hash else []


def list_code_architecture_links(project_id: str, commit_hash: str | None = None) -> list[dict]:
    """返回指定 Commit 的配置到调用关联；页面可省略 Commit。"""
    with connection() as conn:
        if not commit_hash:
            project = conn.execute(
                "SELECT current_commit FROM code_projects WHERE project_id = %s",
                (project_id,),
            ).fetchone()
            commit_hash = project[0] if project else None
        return _list_architecture_links(conn, project_id, commit_hash) if commit_hash else []


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


def list_code_file_inventory(
    project_id: str, query: str = "", limit: int = 200, offset: int = 0,
    commit_hash: str | None = None,
) -> dict:
    """返回指定 Commit 的文件资产；Agent 必须传入会话锁定的 Commit。"""
    with connection() as conn:
        if commit_hash:
            row = conn.execute(
                "SELECT commit_hash, scan_metadata FROM code_project_snapshots WHERE project_id = %s AND commit_hash = %s",
                (project_id, commit_hash),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT current_commit, scan_metadata FROM code_projects WHERE project_id = %s", (project_id,)
            ).fetchone()
    if not row:
        return {"project_id": project_id, "items": [], "coverage": {}}
    items = row[1].get("file_inventory", []) if isinstance(row[1], dict) else []
    term = query.strip().casefold()
    aliases = {
        "配置": ("config", "yaml", "yml", "compose", "helm", "env", "toml", "json", "properties", "cfg", "conf"),
        "配置文件": ("config", "yaml", "yml", "compose", "helm", "env", "toml", "json", "properties", "cfg", "conf"),
        "部署": ("deployment", "compose", "helm", "docker", "kubernetes", "yaml", "yml"),
        "yaml": ("yaml", "yml"), "yml": ("yaml", "yml"), "compose": ("compose",),
        "helm": ("helm", "values", "template"), "env": (".env", "env"),
    }
    terms = aliases.get(term, (term,)) if term else ()
    if terms:
        items = [item for item in items if any(token in " ".join(str(value or "") for value in item.values()).casefold() for token in terms)]
    counts = {"total": len(row[1].get("file_inventory", [])) if isinstance(row[1], dict) else 0}
    counts.update({status: sum(1 for item in (row[1].get("file_inventory", []) if isinstance(row[1], dict) else []) if item.get("parser_status") == status) for status in ("parsed", "partial", "unsupported", "unclassified")})
    safe_offset = max(0, offset)
    safe_limit = max(1, min(limit, 200))
    return {"project_id": project_id, "commit_hash": row[0], "items": items[safe_offset:safe_offset + safe_limit], "offset": safe_offset, "limit": safe_limit, "has_more": safe_offset + safe_limit < len(items), "coverage": counts}


def list_code_config_facts(
    project_id: str,
    query: str = "",
    limit: int = 200,
    offset: int = 0,
    path_prefix: str = "",
    fact_types: list[str] | None = None,
    formats: list[str] | None = None,
    citations=None,
    commit_hash: str | None = None,
) -> dict:
    """返回指定 Commit 的通用配置事实，避免导入并发时跨版本取证。"""
    with connection() as conn:
        if commit_hash:
            row = conn.execute(
                "SELECT root_path, commit_hash, scan_metadata FROM code_project_snapshots WHERE project_id = %s AND commit_hash = %s",
                (project_id, commit_hash),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT root_path, current_commit, scan_metadata FROM code_projects WHERE project_id = %s", (project_id,)
            ).fetchone()
    if not row:
        return {"project_id": project_id, "items": [], "count": 0}
    metadata = row[2] if isinstance(row[2], dict) else {}
    items = metadata.get("config_facts", [])
    term = query.strip().casefold()
    # 中文泛化查询表示“配置这一层”，不应被当成英文路径关键字而过滤为空。
    config_aliases = {
        "配置": (), "配置文件": (), "部署": (),
        "yaml": ("yaml", "yml"), "yml": ("yaml", "yml"),
        "compose": ("compose",), "docker compose": ("compose",),
        "helm": ("helm", "values", "template"),
    }
    terms = config_aliases.get(term, (term,)) if term else ()
    if terms:
        items = [item for item in items if any(token in " ".join(str(value or "") for value in item.values()).casefold() for token in terms)]
    prefix = path_prefix.strip().replace("\\", "/").strip("/").casefold()
    if prefix:
        items = [item for item in items if str(item.get("path", "")).replace("\\", "/").casefold().startswith(prefix)]
    wanted_types = {str(value).strip().casefold() for value in (fact_types or []) if str(value).strip()}
    if wanted_types:
        items = [item for item in items if str(item.get("fact_type", "")).casefold() in wanted_types]
    wanted_formats = {str(value).strip().casefold() for value in (formats or []) if str(value).strip()}
    if wanted_formats:
        items = [item for item in items if str(item.get("config_format", "")).casefold() in wanted_formats]
    if citations is not None:
        for item in items[:500]:
            item["citation"] = f"[{citations.add(item.get('path', ''), item.get('line', 1), item.get('key_path', item.get('fact_type', 'config')))}]"
    safe_offset = max(0, offset)
    safe_limit = max(1, min(limit, 100))
    return {"project_id": project_id, "commit_hash": row[1], "items": items[safe_offset:safe_offset + safe_limit], "count": len(items), "offset": safe_offset, "limit": safe_limit, "has_more": safe_offset + safe_limit < len(items), "source": "stored_scan"}


def read_code_inventory_source(
    project_id: str, relative_path: str, start_line: int = 1,
    end_line: int | None = None, commit_hash: str | None = None,
) -> dict | None:
    """从指定 Commit 快照读取未入符号索引的文本文件。"""
    with connection() as conn:
        if commit_hash:
            row = conn.execute(
                "SELECT root_path, commit_hash, scan_metadata FROM code_project_snapshots WHERE project_id = %s AND commit_hash = %s",
                (project_id, commit_hash),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT root_path, current_commit, scan_metadata FROM code_projects WHERE project_id = %s", (project_id,)
            ).fetchone()
    if not row:
        return None
    inventory = row[2].get("file_inventory", []) if isinstance(row[2], dict) else []
    inventory_item = next((item for item in inventory if item.get("path") == relative_path), None)
    if not inventory_item:
        return None
    root = Path(row[0]).expanduser().resolve()
    target = (root / Path(relative_path)).resolve()
    if target != root and root not in target.parents or not target.is_file():
        return None
    if target.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("source file is larger than 2 MB")
    text = target.read_text(encoding="utf-8", errors="replace")
    expected_hash = inventory_item.get("content_hash")
    if not expected_hash or _hash_text(text) != expected_hash:
        raise RuntimeError("source snapshot integrity check failed")
    lines = text.splitlines()
    first = max(1, start_line)
    last = min(len(lines), end_line if end_line is not None else first + 119, first + 199)
    if first > max(1, len(lines)) or last < first:
        raise ValueError("source line range is outside the file")
    return {"project_id": project_id, "commit_hash": row[1], "path": relative_path, "start_line": first, "end_line": last, "numbered_content": "\n".join(f"{line_no:>6}  {line}" for line_no, line in enumerate(lines[first - 1:last], first))}


def search_code_symbols(
    project_id: str,
    query: str = "",
    limit: int = 100,
    file_path: str | None = None,
    symbol_kind: str | None = None,
    commit_hash: str | None = None,
) -> list[dict]:
    """搜索指定 Commit 的符号；未指定时供页面浏览当前版本。"""
    with connection() as conn:
        result = conn.execute(
            """SELECT s.symbol_id, s.name, s.qualified_name, s.symbol_kind, f.path,
                      s.start_line, s.end_line, s.signature, s.metadata
               FROM code_projects p
               JOIN code_symbols s ON s.project_id = p.project_id
                    AND s.commit_hash = COALESCE(%s::text, p.current_commit)
               JOIN code_files f ON f.file_id = s.file_id
               WHERE p.project_id = %s
                 AND (%s = '' OR s.name ILIKE %s OR s.qualified_name ILIKE %s OR f.path ILIKE %s)
                 AND (%s::text IS NULL OR f.path = %s::text)
                 AND (%s::text IS NULL OR s.symbol_kind = %s::text)
               ORDER BY CASE WHEN lower(s.name) = lower(%s) THEN 0 ELSE 1 END,
                        f.path, s.start_line, length(s.qualified_name) LIMIT %s""",
            (
                commit_hash, project_id, query, f"%{query}%", f"%{query}%", f"%{query}%",
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


def read_code_source(
    project_id: str,
    relative_path: str,
    start_line: int = 1,
    end_line: int | None = None,
    commit_hash: str | None = None,
) -> dict | None:
    """读取指定 Commit 的受限源码，并校验快照内容哈希。"""
    with connection() as conn:
        if commit_hash:
            row = conn.execute(
                """SELECT s.root_path, s.commit_hash, f.path, f.content_hash, f.line_count
                   FROM code_project_snapshots s
                   JOIN code_files f ON f.project_id = s.project_id AND f.commit_hash = s.commit_hash
                   WHERE s.project_id = %s AND s.commit_hash = %s AND f.path = %s""",
                (project_id, commit_hash, relative_path),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT p.root_path, p.current_commit, f.path, f.content_hash, f.line_count
                   FROM code_projects p
                   JOIN code_files f ON f.project_id = p.project_id AND f.commit_hash = p.current_commit
                   WHERE p.project_id = %s AND f.path = %s""",
                (project_id, relative_path),
            ).fetchone()
    if not row:
        return None

    root = Path(row[0]).expanduser().resolve()
    target = (root / Path(row[2])).resolve()
    if target != root and root not in target.parents:
        raise ValueError("source path escapes project root")
    if not target.is_file():
        raise FileNotFoundError("source file no longer exists")
    if target.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("source file is larger than 2 MB")

    text = target.read_text(encoding="utf-8", errors="replace")
    if _hash_text(text) != row[3]:
        # 返回内容和索引哈希不一致会制造无法复核的引用，因此直接中止本次取证。
        raise RuntimeError("source snapshot integrity check failed")
    lines = text.splitlines()
    first = max(1, start_line)
    last = min(len(lines), end_line if end_line is not None else first + 119, first + 199)
    if first > max(1, len(lines)) or last < first:
        raise ValueError("source line range is outside the file")
    selected = lines[first - 1:last]
    return {
        "project_id": project_id,
        "commit_hash": row[1],
        "path": row[2],
        "start_line": first,
        "end_line": last,
        "line_count": row[4],
        "content": "\n".join(selected),
        "numbered_content": "\n".join(f"{line_no:>6}  {line}" for line_no, line in enumerate(selected, first)),
        "stale": False,
    }


def trace_code_call_chain(
    symbol_id: str, max_depth: int = 4, max_nodes: int = 80,
    commit_hash: str | None = None,
) -> dict | None:
    """从指定 Commit 的符号沿 calls 边做有界 BFS。"""
    with connection() as conn:
        start = conn.execute(
            """SELECT s.symbol_id, s.project_id, s.commit_hash, s.name, s.qualified_name,
                      s.symbol_kind, f.path, s.start_line, s.end_line
               FROM code_symbols s
               JOIN code_files f ON f.file_id = s.file_id
               WHERE s.symbol_id = %s AND (%s::text IS NULL OR s.commit_hash = %s::text)""",
            (symbol_id, commit_hash, commit_hash),
        ).fetchone()
        if not start:
            return None

        project_id, commit_hash = str(start[1]), start[2]

        def node_from_row(row, depth: int) -> dict:
            return {
                "symbol_id": str(row[0]), "name": row[3], "qualified_name": row[4],
                "symbol_kind": row[5], "path": row[6], "start_line": row[7],
                "end_line": row[8], "depth": depth, "architecture_facts": [],
            }

        nodes = {str(start[0]): node_from_row(start, 0)}
        queue: list[tuple[str, int, frozenset[str]]] = [
            (str(start[0]), 0, frozenset({str(start[0])}))
        ]
        edges: list[dict] = []
        unresolved_edges: list[dict] = []
        cycles: list[dict] = []
        seen_edges: set[str] = set()
        truncated = False

        while queue:
            source_id, depth, ancestors = queue.pop(0)
            if depth >= max_depth:
                continue
            relations = conn.execute(
                """SELECT r.relation_id, r.target_ref, r.confidence, r.evidence,
                          ts.symbol_id, ts.project_id, ts.commit_hash, ts.name, ts.qualified_name,
                          ts.symbol_kind, tf.path, ts.start_line, ts.end_line
                   FROM code_relations r
                   LEFT JOIN code_symbols ts ON ts.symbol_id = r.target_symbol_id
                   LEFT JOIN code_files tf ON tf.file_id = ts.file_id
                   WHERE r.source_symbol_id = %s AND r.relation_type = 'calls'
                   ORDER BY r.confidence DESC, r.target_ref""",
                (source_id,),
            ).fetchall()
            for relation in relations:
                relation_id = str(relation[0])
                if relation_id in seen_edges:
                    continue
                seen_edges.add(relation_id)
                edge = {
                    "relation_id": relation_id, "source_symbol_id": source_id,
                    "target_symbol_id": str(relation[4]) if relation[4] else None,
                    "target_ref": relation[1], "confidence": relation[2],
                    "evidence": relation[3], "depth": depth + 1,
                }
                if not relation[4]:
                    unresolved_edges.append(edge)
                    continue
                target_id = str(relation[4])
                edges.append(edge)
                if target_id in ancestors:
                    cycles.append(edge)
                    continue
                if target_id in nodes:
                    # 多条分支汇聚到同一函数不是环路；节点只展开一次，但边仍保留。
                    continue
                if len(nodes) >= max_nodes:
                    truncated = True
                    continue
                target_row = (
                    relation[4], relation[5], relation[6], relation[7], relation[8],
                    relation[9], relation[10], relation[11], relation[12],
                )
                nodes[target_id] = node_from_row(target_row, depth + 1)
                queue.append((target_id, depth + 1, ancestors | {target_id}))

        if nodes:
            placeholders = ",".join(["%s"] * len(nodes))
            facts = conn.execute(
                f"""SELECT fact_id, fact_type, name, value, source_path, source_line,
                            evidence, source_symbol_id, confidence
                     FROM code_architecture_facts
                     WHERE project_id = %s AND commit_hash = %s
                       AND source_symbol_id IN ({placeholders})
                     ORDER BY source_line, fact_type""",
                (project_id, commit_hash, *nodes.keys()),
            ).fetchall()
            for fact in facts:
                owner_id = str(fact[7])
                if owner_id in nodes:
                    nodes[owner_id]["architecture_facts"].append({
                        "fact_id": str(fact[0]), "fact_type": fact[1], "name": fact[2],
                        "value": fact[3], "source_path": fact[4], "source_line": fact[5],
                        "evidence": fact[6], "confidence": fact[8],
                    })

    return {
        "project_id": project_id, "commit_hash": commit_hash, "root_symbol_id": str(start[0]),
        "max_depth": max_depth, "max_nodes": max_nodes, "truncated": truncated,
        "nodes": sorted(nodes.values(), key=lambda item: (item["depth"], item["path"], item["start_line"])),
        "edges": edges, "unresolved_edges": unresolved_edges, "cycles": cycles,
    }
