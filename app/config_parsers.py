"""通用配置文件解析器。

解析器只提取可定位的配置事实，不根据某个具体项目名称编写规则。
复杂模板无法完全展开时保留原始表达式，并明确标记 partial。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


SECRET_KEY = re.compile(r"(?i)(password|passwd|pwd|token|secret|api[_-]?key|credential)")
COMPOSE_NAMES = {"compose.yml", "compose.yaml", "docker-compose.yml", "docker-compose.yaml"}


def _safe_value(key: str, value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if SECRET_KEY.search(key):
        return "[REDACTED]"
    return text[:500]


def _fact(path: str, line: int, parser: str, fmt: str, fact_type: str, key_path: str, value: Any, value_kind: str = "scalar", status: str = "parsed") -> dict:
    return {
        "path": path, "line": line, "parser_name": parser, "config_format": fmt,
        "fact_type": fact_type, "key_path": key_path, "value": _safe_value(key_path, value),
        "value_kind": value_kind, "parse_status": status,
    }


def _yaml_tree(text: str) -> tuple[Any, list[tuple[str, Any, int]]]:
    """读取 YAML，同时保留标量键路径和起始行号。"""
    root = yaml.compose(text)
    entries: list[tuple[str, Any, int]] = []

    def walk(node: Any, path: list[str]) -> None:
        if isinstance(node, yaml.MappingNode):
            for key_node, value_node in node.value:
                walk(value_node, [*path, str(key_node.value)])
        elif isinstance(node, yaml.SequenceNode):
            for index, value_node in enumerate(node.value):
                walk(value_node, [*path, str(index)])
        elif isinstance(node, yaml.ScalarNode) and path:
            entries.append((".".join(path), node.value, node.start_mark.line + 1))

    if root is not None:
        walk(root, [])
    return root, entries


def parse_yaml(path: str, text: str) -> list[dict]:
    """通用 YAML 解析：保存键路径、标量类型和源码行号。"""
    try:
        _, entries = _yaml_tree(text)
    except yaml.YAMLError:
        return [_fact(path, 1, "yaml", "yaml", "parse_error", "$", "YAML parse failed", "error", "partial")]
    return [_fact(path, line, "yaml", "yaml", "config_key", key, value) for key, value, line in entries]


def parse_compose(path: str, text: str) -> list[dict]:
    """解析 Compose 的通用服务语义，并保留 YAML 键事实作为兜底。"""
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return [_fact(path, 1, "docker-compose", "docker_compose", "parse_error", "$", "Compose parse failed", "error", "partial")]
    facts = parse_yaml(path, text)
    services = data.get("services", {}) if isinstance(data, dict) else {}
    if not isinstance(services, dict):
        return facts
    lines = text.splitlines()
    for service, config in services.items():
        base = f"services.{service}"
        line = next((i for i, value in enumerate(lines, 1) if re.match(rf"^\s{{2,}}{re.escape(str(service))}\s*:", value)), 1)
        facts.append(_fact(path, line, "docker-compose", "docker_compose", "service", base, service))
        if not isinstance(config, dict):
            continue
        for field in ("image", "build", "ports", "environment", "depends_on", "networks", "volumes"):
            if field not in config:
                continue
            value = config[field]
            kind = "list" if isinstance(value, list) else "mapping" if isinstance(value, dict) else "scalar"
            field_line = next((i for i, item in enumerate(lines, 1) if re.match(rf"^\s+{re.escape(field)}\s*:", item) and i >= line), line)
            facts.append(_fact(path, field_line, "docker-compose", "docker_compose", f"compose_{field}", f"{base}.{field}", value, kind))
            if field == "depends_on" and isinstance(value, (list, dict)):
                dependencies = value if isinstance(value, list) else list(value)
                for dependency in dependencies:
                    facts.append(_fact(path, field_line, "docker-compose", "docker_compose", "dependency", base, dependency))
    return facts


def parse_helm(path: str, text: str) -> list[dict]:
    """解析 Helm values 和模板中的资源/Values 引用；模板展开不足时标记 partial。"""
    facts = parse_yaml(path, text)
    if Path(path).name.lower() == "values.yaml":
        for item in facts:
            item["parser_name"] = "helm-values"
            item["config_format"] = "helm_values"
            item["fact_type"] = "helm_value"
    for match in re.finditer(r"\{\{\s*\.Values\.([A-Za-z0-9_.-]+)", text):
        facts.append(_fact(path, text.count("\n", 0, match.start()) + 1, "helm-template", "helm_template", "helm_template", "helm.Values", match.group(1), "template", "partial"))
    for match in re.finditer(r"(?m)^\s*kind:\s*([A-Za-z0-9]+)", text):
        facts.append(_fact(path, text.count("\n", 0, match.start()) + 1, "helm-template", "helm_template", "helm_template", "kubernetes.kind", match.group(1), "scalar"))
    return facts


def parse_config_file(path: str, text: str) -> list[dict]:
    """按文件格式选择解析器；返回空列表表示该文件不是当前支持的配置格式。"""
    name = Path(path).name.lower()
    if name in COMPOSE_NAMES:
        return parse_compose(path, text)
    if name == "values.yaml" or name.endswith("-values.yaml") or ("/templates/" in path.replace("\\", "/") and name.endswith((".yaml", ".yml"))):
        return parse_helm(path, text)
    if name.endswith((".yaml", ".yml")):
        return parse_yaml(path, text)
    return []


def parse_project_configs(root: Path) -> list[dict]:
    """遍历项目配置文件，统一输出配置事实。"""
    facts: list[dict] = []
    ignored = {".git", "node_modules", ".venv", "venv", "dist", "build", "target"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part.lower() in ignored for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root).as_posix()
        if path.stat().st_size > 2 * 1024 * 1024:
            continue
        if not path.name.lower().endswith((".yaml", ".yml")):
            continue
        try:
            facts.extend(parse_config_file(relative, path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return facts[:20000]
