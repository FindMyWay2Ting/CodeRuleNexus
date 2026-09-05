from contextlib import contextmanager

import pytest

from app import access_control, main


class FakeResult:
    """提供 create_workspace 所需的最小数据库结果接口。"""

    def __init__(self, row=None):
        self.row = row

    def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self, inserted=("workspace-id",), selected=None):
        self.inserted = inserted
        self.selected = selected
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))
        if "INSERT INTO workspaces" in sql:
            return FakeResult(self.inserted)
        if "FROM documents d" in sql:
            return FakeResult(self.selected)
        return FakeResult()


def fake_connection(conn):
    @contextmanager
    def open_connection():
        yield conn

    return open_connection


def test_create_workspace_adds_owner_membership_in_same_transaction(monkeypatch):
    conn = FakeConnection()
    monkeypatch.setattr(access_control, "connection", fake_connection(conn))
    monkeypatch.setattr(access_control.uuid, "uuid4", lambda: "workspace-id")

    result = access_control.create_workspace("  支付   平台组  ")

    assert result["workspace_id"] == "workspace-id"
    assert result["workspace_name"] == "支付 平台组"
    assert result["role"] == "owner"
    assert any("INSERT INTO workspaces" in sql for sql, _ in conn.calls)
    assert any("INSERT INTO workspace_members" in sql for sql, _ in conn.calls)


def test_create_workspace_rejects_duplicate_name_before_insert(monkeypatch):
    conn = FakeConnection(inserted=None)
    monkeypatch.setattr(access_control, "connection", fake_connection(conn))

    with pytest.raises(access_control.WorkspaceNameConflict, match="同名"):
        access_control.create_workspace("已有空间")

    assert not any("INSERT INTO workspace_members" in sql for sql, _ in conn.calls)


def test_document_management_uses_server_resolved_scope(monkeypatch):
    conn = FakeConnection(selected=(1,))
    monkeypatch.setattr(access_control, "connection", fake_connection(conn))
    scope = access_control.AccessScope("user-1", "workspace-1", "研发", None, frozenset())

    assert access_control.can_manage_document("document-1", scope) is True
    _, params = next((sql, params) for sql, params in conn.calls if "FROM documents d" in sql)
    assert params == ("document-1", "workspace-1", "user-1", "user-1", "user-1")


def test_cached_rag_scope_rejects_department_change(monkeypatch):
    previous = access_control.AccessScope("user-1", "workspace-1", "研发", "dept-a", frozenset())
    changed = access_control.AccessScope("user-1", "workspace-1", "研发", "dept-b", frozenset())
    monkeypatch.setattr(main, "refresh_request_scope", lambda _scope: changed)

    with pytest.raises(PermissionError, match="部门已变化"):
        main.refresh_scope_for_cached_rag(previous)
