"""将 Code Wiki 快照迁移到独立持久化目录，并原子更新数据库路径。"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.code_wiki import DEFAULT_REPOSITORY_ROOT, github_repository_key
from app.db import connection


def _relative_to(path_value: str, source_root: Path) -> Path:
    path = Path(path_value).expanduser().resolve()
    try:
        return path.relative_to(source_root)
    except ValueError as exc:
        raise RuntimeError(f"数据库路径不属于源目录：{path}") from exc


def _local_identity_name(root_path: str) -> str:
    path = Path(root_path)
    if path.parent.parent.name == ".snapshots":
        return path.parent.name
    return path.name


def _repository_key(project_name: str, root_path: str, metadata: dict) -> str:
    source = metadata.get("source", {}) if isinstance(metadata, dict) else {}
    source_type = source.get("type")
    if source_type == "github" and source.get("owner") and source.get("repository"):
        return github_repository_key(str(source["owner"]), str(source["repository"]))
    if source_type == "local_upload":
        return f"local:{_local_identity_name(root_path).casefold()}"
    raise RuntimeError(f"项目 {project_name} 缺少可迁移的托管来源身份")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="当前数据库所引用的旧代码快照根目录")
    parser.add_argument("--apply", action="store_true", help="执行复制和数据库更新；省略时只预览")
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    target_root = DEFAULT_REPOSITORY_ROOT.resolve()
    if source_root == target_root:
        raise RuntimeError("源目录和 CODE_REPOSITORY_ROOT 相同，无需迁移")
    if not source_root.is_dir():
        raise RuntimeError(f"源目录不存在：{source_root}")

    with connection() as conn:
        projects = conn.execute(
            """SELECT project_id, project_name, root_path, scan_metadata, repository_key, workspace_id
               FROM code_projects WHERE status = 'active'"""
        ).fetchall()
        snapshots = conn.execute(
            "SELECT project_id, commit_hash, root_path FROM code_project_snapshots"
        ).fetchall()

    project_updates = []
    keys_seen: set[tuple[str, str]] = set()
    for project_id, project_name, root_path, metadata, existing_key, workspace_id in projects:
        relative = _relative_to(root_path, source_root)
        repository_key = str(existing_key or _repository_key(project_name, root_path, metadata))
        identity = (str(workspace_id), repository_key)
        if identity in keys_seen:
            raise RuntimeError(f"迁移后仓库键重复：{repository_key}")
        keys_seen.add(identity)
        project_updates.append((str(target_root / relative), repository_key, project_id))

    snapshot_updates = [
        (str(target_root / _relative_to(root_path, source_root)), project_id, commit_hash)
        for project_id, commit_hash, root_path in snapshots
    ]

    print(f"Source: {source_root}")
    print(f"Target: {target_root}")
    print(f"Projects: {len(project_updates)}; snapshots: {len(snapshot_updates)}")
    if not args.apply:
        print("Dry run only. Re-run with --apply after stopping the web service.")
        return 0
    if target_root.exists():
        raise RuntimeError(f"目标目录已存在，拒绝覆盖：{target_root}")

    target_root.parent.mkdir(parents=True, exist_ok=True)
    staging = target_root.parent / f".{target_root.name}-migration-{uuid.uuid4().hex}"
    promoted = False
    try:
        shutil.copytree(source_root, staging)
        os.replace(staging, target_root)
        promoted = True
        with connection() as conn:
            for root_path, repository_key, project_id in project_updates:
                conn.execute(
                    "UPDATE code_projects SET root_path = %s, repository_key = %s WHERE project_id = %s",
                    (root_path, repository_key, project_id),
                )
            for root_path, project_id, commit_hash in snapshot_updates:
                conn.execute(
                    """UPDATE code_project_snapshots SET root_path = %s
                       WHERE project_id = %s AND commit_hash = %s""",
                    (root_path, project_id, commit_hash),
                )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if promoted and target_root.exists():
            shutil.rmtree(target_root, ignore_errors=True)
        raise

    print("Migration complete. The source directory was preserved for rollback.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
