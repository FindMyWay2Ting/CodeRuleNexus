"""修复曾继承父 Git 仓库 Commit 的本地上传项目快照。"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.code_wiki import (
    DEFAULT_REPOSITORY_ROOT,
    managed_local_repository_path,
    persist_scan,
    repository_import_lock,
    repository_lock_resource,
    scan_project,
)
from app.db import connection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="建立新快照并切换数据库；省略时只预览")
    args = parser.parse_args()

    with connection() as conn:
        rows = conn.execute(
            """SELECT project_id, project_name, root_path, current_commit, current_commit_source,
                      workspace_id, owner_user_id, created_by_user_id, scan_metadata, repository_key
               FROM code_projects
               WHERE status = 'active' AND scan_metadata->'source'->>'type' = 'local_upload'
               ORDER BY project_name"""
        ).fetchall()

    candidate_count = 0
    for initial_row in rows:
        initial_project_id, _, initial_root_path, _, _, initial_workspace_id = initial_row[:6]
        initial_repository_key = str(
            initial_row[9] or f"local:{Path(initial_root_path).name.casefold()}"
        )
        # 状态读取、扫描和写回必须全部位于同一仓库锁内。否则并发导入可能先
        # 激活新版本，修复脚本随后却把锁外扫描的旧版本重新写成 current。
        with repository_import_lock(
            str(initial_workspace_id),
            repository_lock_resource(initial_repository_key, str(initial_project_id)),
        ):
            with connection() as conn:
                row = conn.execute(
                    """SELECT project_id, project_name, root_path, current_commit, current_commit_source,
                              workspace_id, owner_user_id, created_by_user_id, scan_metadata, repository_key
                       FROM code_projects
                       WHERE project_id = %s AND workspace_id = %s AND status = 'active'
                         AND scan_metadata->'source'->>'type' = 'local_upload'""",
                    (initial_project_id, initial_workspace_id),
                ).fetchone()
            if not row:
                continue
            (
                project_id, project_name, root_path, old_commit, old_source,
                workspace_id, owner_user_id, created_by_user_id, metadata, repository_key,
            ) = row
            repository_key = str(repository_key or f"local:{Path(root_path).name.casefold()}")
            if repository_lock_resource(repository_key, str(project_id)) != repository_lock_resource(
                initial_repository_key, str(initial_project_id),
            ):
                raise RuntimeError(f"项目仓库身份在加锁期间发生变化：{project_name}")
            scan = scan_project(str(Path(root_path).expanduser().resolve()), str(workspace_id))
            if old_source == "content_scan" and old_commit == scan["commit_hash"]:
                continue
            candidate_count += 1
            print(f"{project_name}: {old_commit} ({old_source}) -> {scan['commit_hash']} (content_scan)")
            if not args.apply:
                continue

            identity_path = managed_local_repository_path(
                str(project_name), DEFAULT_REPOSITORY_ROOT, str(workspace_id),
            )
            safe_commit = re.sub(r"[^A-Za-z0-9._-]", "-", str(scan["commit_hash"]))[:80]
            snapshot_path = identity_path.parent / ".snapshots" / identity_path.name / safe_commit
            copied_here = False
            if not snapshot_path.exists():
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(Path(root_path), snapshot_path)
                copied_here = True
            repaired = scan_project(str(snapshot_path), str(workspace_id))
            if repaired["commit_hash"] != scan["commit_hash"]:
                if copied_here:
                    shutil.rmtree(snapshot_path, ignore_errors=True)
                raise RuntimeError(f"复制后内容哈希变化：{project_name}")
            repaired.update({
                "project_id": str(project_id),
                "project_name": str(project_name),
                "root_path": str(snapshot_path.resolve()),
                "workspace_id": str(workspace_id),
                "owner_user_id": str(owner_user_id),
                "created_by_user_id": str(created_by_user_id),
                "source": {**((metadata or {}).get("source") or {}), "repository_key": repository_key},
            })
            try:
                persist_scan(repaired)
            except Exception:
                if copied_here:
                    shutil.rmtree(snapshot_path, ignore_errors=True)
                raise
            print(f"Repaired: {project_name} -> {repaired['commit_hash']}")

    if candidate_count == 0:
        print("No local snapshots require repair.")
    elif not args.apply:
        print(f"Dry run only. {candidate_count} project(s) require repair; re-run with --apply.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
