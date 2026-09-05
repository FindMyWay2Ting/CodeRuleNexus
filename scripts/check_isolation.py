"""只读检查工作空间数据边界和 Agent 授权项目目录。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.access_control import resolve_scope
from app.code_wiki import DEFAULT_REPOSITORY_ROOT, _git_commit
from app.db import connection


def main() -> int:
    """返回非零状态表示数据库中存在跨空间脏数据或授权目录不一致。"""
    checks: list[tuple[str, bool, str]] = []
    with connection() as conn:
        chunk_mismatches = conn.execute(
            """
            SELECT COUNT(*)
            FROM knowledge_chunks c
            JOIN documents d ON d.document_id = c.document_id
            WHERE c.workspace_id <> d.workspace_id
            """
        ).fetchone()[0]
        parent_mismatches = conn.execute(
            """
            SELECT COUNT(*)
            FROM knowledge_chunks c
            JOIN knowledge_chunks p ON p.chunk_id = c.parent_chunk_id
            WHERE c.parent_chunk_id IS NOT NULL
              AND (c.document_id <> p.document_id OR c.workspace_id <> p.workspace_id)
            """
        ).fetchone()[0]
        project_owner_gaps = conn.execute(
            """
            SELECT COUNT(*) FROM code_projects
            WHERE workspace_id IS NULL OR owner_user_id IS NULL OR created_by_user_id IS NULL
            """
        ).fetchone()[0]
        snapshot_pointer_gaps = conn.execute(
            """
            SELECT COUNT(*)
            FROM code_projects p
            LEFT JOIN code_project_snapshots s
              ON s.project_id = p.project_id AND s.commit_hash = p.current_commit
            WHERE s.project_id IS NULL OR s.root_path <> p.root_path
            """
        ).fetchone()[0]
        memberships = conn.execute(
            """
            SELECT wm.user_id, wm.workspace_id
            FROM workspace_members wm
            JOIN users u ON u.user_id = wm.user_id
            WHERE wm.status = 'active' AND u.status = 'active'
            """
        ).fetchall()
        active_snapshots = conn.execute(
            """
            SELECT p.project_id, p.root_path, p.current_commit, p.current_commit_source,
                   p.scan_metadata
            FROM code_projects p
            WHERE p.status = 'active'
            """
        ).fetchall()

    checks.append(("Chunk/document workspace", chunk_mismatches == 0, f"mismatches={chunk_mismatches}"))
    checks.append(("Parent/child scope", parent_mismatches == 0, f"mismatches={parent_mismatches}"))
    checks.append(("Code project ownership", project_owner_gaps == 0, f"missing_scope_or_owner={project_owner_gaps}"))
    checks.append(("Code snapshot pointer", snapshot_pointer_gaps == 0, f"missing_or_mismatched={snapshot_pointer_gaps}"))

    managed_root = DEFAULT_REPOSITORY_ROOT.resolve()
    snapshot_disk_errors = 0
    for _project_id, root_path, commit_hash, commit_source, metadata in active_snapshots:
        root = Path(root_path).expanduser().resolve()
        source_type = (metadata or {}).get("source", {}).get("type") if isinstance(metadata, dict) else None
        if not root.is_dir():
            snapshot_disk_errors += 1
            continue
        if source_type in {"github", "local_upload"} and managed_root not in root.parents:
            snapshot_disk_errors += 1
            continue
        if commit_source == "git" and _git_commit(root)[0] != commit_hash:
            snapshot_disk_errors += 1
    checks.append((
        "Active snapshot disk",
        snapshot_disk_errors == 0,
        f"projects={len(active_snapshots)}, missing_outside_or_commit_mismatch={snapshot_disk_errors}",
    ))

    catalog_errors = 0
    for user_id, workspace_id in memberships:
        scope = resolve_scope(str(workspace_id), str(user_id))
        catalog_ids = {str(item["project_id"]) for item in scope.authorized_projects}
        if catalog_ids != set(scope.allowed_project_ids):
            catalog_errors += 1
    checks.append(("Agent project catalog", catalog_errors == 0, f"memberships={len(memberships)}, mismatches={catalog_errors}"))

    print("Workspace Isolation Check")
    print("=" * 32)
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'FAIL'}] {name}: {detail}")
    failures = sum(not passed for _, passed, _ in checks)
    print("=" * 32)
    print("All checks passed" if failures == 0 else f"{failures} check(s) failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
