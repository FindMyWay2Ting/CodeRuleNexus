"""检查演示环境是否具备运行条件，不发起任何模型调用。"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许从仓库根目录直接执行 `python scripts/check_readiness.py`。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg

from app.config import settings
from app.bootstrap import REQUIRED_SCHEMA_VERSION


REQUIRED_TABLES = {
    "documents",
    "knowledge_chunks",
    "users",
    "user_sessions",
    "workspaces",
    "workspace_members",
    "code_projects",
    "code_files",
    "code_symbols",
    "code_relations",
    "document_revisions",
    "workspace_invitations",
    "user_data_handovers",
    "audit_events",
    "auth_rate_limits",
    "schema_migrations",
    "code_project_access",
    "code_components",
    "code_architecture_facts",
    "code_architecture_links",
    "code_project_snapshots",
}

REQUIRED_CONSTRAINTS = {
    "fk_knowledge_chunks_document_workspace",
    "fk_knowledge_chunks_parent_scope",
    "fk_code_files_snapshot",
    "fk_code_symbols_snapshot",
    "fk_code_relations_snapshot",
    "fk_code_components_snapshot",
    "fk_code_architecture_facts_snapshot",
    "fk_code_architecture_links_snapshot",
}


def configured(value: str) -> bool:
    """占位符不算有效配置，避免演示时到第一次请求才暴露问题。"""
    normalized = value.strip().lower()
    return bool(normalized) and not normalized.startswith("your-")


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.extend([
        ("Chat LLM", all(map(configured, [settings.llm_api_base, settings.llm_api_key, settings.chat_model])), "check LLM_API_BASE / LLM_API_KEY / CHAT_MODEL"),
        ("Embedding", all(map(configured, [settings.embedding_api_base, settings.embedding_api_key, settings.embedding_model])), "check EMBEDDING_API_BASE / EMBEDDING_API_KEY / EMBEDDING_MODEL"),
        ("Reranker", all(map(configured, [settings.reranker_api_url, settings.reranker_api_key, settings.reranker_model])), "check RERANKER_API_URL / RERANKER_API_KEY / RERANKER_MODEL"),
    ])

    try:
        with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
            extensions = {row[0] for row in conn.execute("SELECT extname FROM pg_extension").fetchall()}
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = current_schema()"
                ).fetchall()
            }
            constraints = {
                row[0]
                for row in conn.execute(
                    """
                    SELECT conname
                    FROM pg_constraint
                    WHERE connamespace = current_schema()::regnamespace
                      AND convalidated = TRUE
                    """
                ).fetchall()
            }
            migrations = {
                row[0] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
            }
            embedding_type = conn.execute(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = 'knowledge_chunks' AND a.attname = 'embedding'
                """
            ).fetchone()
        checks.append(("PostgreSQL", True, "connection available"))
        missing_extensions = {"vector", "pg_trgm"} - extensions
        checks.append(("Database extensions", not missing_extensions, f"missing: {', '.join(sorted(missing_extensions))}" if missing_extensions else "vector and pg_trgm enabled"))
        missing_tables = REQUIRED_TABLES - tables
        checks.append(("Core tables", not missing_tables, f"missing: {', '.join(sorted(missing_tables))}" if missing_tables else f"verified {len(REQUIRED_TABLES)} tables"))
        missing_constraints = REQUIRED_CONSTRAINTS - constraints
        checks.append(("Isolation constraints", not missing_constraints, f"missing: {', '.join(sorted(missing_constraints))}" if missing_constraints else "chunk/document workspace boundary enforced"))
        checks.append(("Schema version", REQUIRED_SCHEMA_VERSION in migrations, REQUIRED_SCHEMA_VERSION))
        expected_vector = f"vector({settings.embedding_dimensions})"
        actual_vector = embedding_type[0] if embedding_type else "missing"
        checks.append(("Embedding dimension", actual_vector == expected_vector, f"expected {expected_vector}, found {actual_vector}"))
    except Exception as exc:
        checks.append(("PostgreSQL", False, f"connection failed: {type(exc).__name__}"))

    print("Knowledge Base Readiness Check")
    print("=" * 32)
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'FAIL'}] {name}: {detail}")
    failures = sum(not passed for _, passed, _ in checks)
    print("=" * 32)
    print("All checks passed" if failures == 0 else f"{failures} check(s) failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
