"""离线迁移与修复脚本的并发安全回归测试。"""

from contextlib import contextmanager
from unittest.mock import MagicMock

from scripts import repair_local_code_snapshots as repair_script
from scripts.smoke_runtime import normalize_workspace_id


def _connection_context(connection):
    context = MagicMock()
    context.__enter__.return_value = connection
    context.__exit__.return_value = False
    return context


def test_local_snapshot_repair_rereads_current_state_inside_repository_lock(monkeypatch):
    """列表读取后的并发更新不能让修复脚本继续扫描并覆盖旧 root_path。"""
    initial_row = (
        "project-1", "demo", "C:/snapshots/old", "parent-git", "git",
        "workspace-1", "owner-1", "creator-1", {"source": {"type": "local_upload"}}, "local:demo",
    )
    current_row = (
        "project-1", "demo", "C:/snapshots/new", "scan-current", "content_scan",
        "workspace-1", "owner-1", "creator-1", {"source": {"type": "local_upload"}}, "local:demo",
    )
    initial_connection = MagicMock()
    initial_connection.execute.return_value.fetchall.return_value = [initial_row]
    locked_connection = MagicMock()
    locked_connection.execute.return_value.fetchone.return_value = current_row
    contexts = iter([
        _connection_context(initial_connection),
        _connection_context(locked_connection),
    ])
    lock_held = False

    @contextmanager
    def fake_lock(_workspace_id, _resource_key):
        nonlocal lock_held
        lock_held = True
        try:
            yield
        finally:
            lock_held = False

    scanned_paths = []

    def fake_scan(path, _workspace_id):
        assert lock_held is True
        scanned_paths.append(path.replace("\\", "/"))
        return {"commit_hash": "scan-current"}

    monkeypatch.setattr(repair_script, "connection", lambda: next(contexts))
    monkeypatch.setattr(repair_script, "repository_import_lock", fake_lock)
    monkeypatch.setattr(repair_script, "scan_project", fake_scan)
    persist = MagicMock()
    monkeypatch.setattr(repair_script, "persist_scan", persist)
    monkeypatch.setattr("sys.argv", ["repair_local_code_snapshots"])

    assert repair_script.main() == 0
    assert scanned_paths == ["C:/snapshots/new"]
    persist.assert_not_called()


def test_runtime_smoke_accepts_legacy_and_uuid_workspace_ids():
    assert normalize_workspace_id("workspace-001") == "workspace-001"
    assert normalize_workspace_id("878cefeb-6218-476a-84d1-9c312b319821") == (
        "878cefeb-6218-476a-84d1-9c312b319821"
    )


def test_runtime_smoke_rejects_unsafe_workspace_id():
    import pytest

    with pytest.raises(ValueError):
        normalize_workspace_id("../workspace-001?admin=true")
