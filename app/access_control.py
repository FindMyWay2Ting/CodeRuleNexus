"""服务端工作空间和代码项目访问边界。

环境变量身份仅用于旧数据库启动迁移，HTTP 请求必须传入认证会话中的用户。
浏览器可以请求切换空间，但最终能否使用该空间，必须由这里查询成员关系决定。
这个模块只负责身份、空间和项目授权，不参与 Agent 的规划或循环控制。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import re
import secrets
import unicodedata
import uuid

from .config import settings
from .db import connection


class WorkspaceNameConflict(ValueError):
    """同一所有者名下已经存在规范化后的同名空间。"""


@dataclass(frozen=True)
class AccessScope:
    """一次请求的服务端授权结果，供 RAG、Code Wiki 和 Agent 共同使用。"""

    user_id: str
    workspace_id: str
    workspace_name: str
    department_id: str | None
    allowed_project_ids: frozenset[str]
    workspace_role: str = "viewer"
    # 只包含已授权项目的轻量目录，供顶层 Agent 根据名称和技术事实选择项目。
    authorized_projects: tuple[dict, ...] = ()


def initialize_access_control() -> None:
    """创建空间、成员和项目 ACL 表，并兼容当前单空间 MVP 数据。"""
    with connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                -- 工作空间稳定标识；所有文档和代码项目都必须归属于一个空间。
                workspace_id TEXT PRIMARY KEY,
                -- 页面展示名称。
                workspace_name TEXT NOT NULL,
                -- 名称规范化键，仅用于同一 owner 下防重，不用于展示或授权。
                normalized_name TEXT NOT NULL,
                -- 工作空间所有者；认证接入后由用户系统提供。
                owner_user_id TEXT NOT NULL,
                -- personal 只属于创建者；team 支持邀请成员。
                workspace_type TEXT NOT NULL DEFAULT 'team'
                    CHECK (workspace_type IN ('personal', 'team')),
                -- 空间是否仍可被选择。
                status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
                -- 空间创建时间。
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                -- 空间元数据更新时间。
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute("ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS normalized_name TEXT")
        conn.execute("ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS workspace_type TEXT NOT NULL DEFAULT 'team'")
        # 旧库也使用与创建接口完全相同的 Unicode/空白规范化；重名旧记录保留展示名，
        # 但追加稳定 ID 作为迁移键，避免唯一索引导致服务无法启动。
        legacy_rows = conn.execute(
            "SELECT workspace_id, workspace_name, owner_user_id FROM workspaces ORDER BY created_at, workspace_id"
        ).fetchall()
        used_name_keys: set[tuple[str, str]] = set()
        for workspace_id, workspace_name, owner_user_id in legacy_rows:
            normalized_name = _workspace_name_key(workspace_name)
            owner_key = (str(owner_user_id), normalized_name)
            if owner_key in used_name_keys:
                normalized_name = f"{normalized_name}#{workspace_id}"
                owner_key = (str(owner_user_id), normalized_name)
            used_name_keys.add(owner_key)
            conn.execute(
                "UPDATE workspaces SET normalized_name = %s WHERE workspace_id = %s",
                (normalized_name, workspace_id),
            )
        conn.execute("ALTER TABLE workspaces ALTER COLUMN normalized_name SET NOT NULL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workspace_members (
                -- 成员所属工作空间。
                workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
                -- 当前认证用户的稳定标识。
                user_id TEXT NOT NULL,
                -- owner 管所有权，admin 管成员，editor 可写知识，viewer 只读。
                role TEXT NOT NULL DEFAULT 'viewer'
                    CHECK (role IN ('owner', 'admin', 'editor', 'viewer')),
                -- 成员是否仍可访问该空间。
                status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
                -- 加入时间。
                joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (workspace_id, user_id)
            )
            """
        )
        # 旧版只有 owner/member；member 迁移为 editor，保持原有成员的写入能力。
        conn.execute("ALTER TABLE workspace_members DROP CONSTRAINT IF EXISTS workspace_members_role_check")
        conn.execute("UPDATE workspace_members SET role = 'editor' WHERE role = 'member'")
        conn.execute(
            "ALTER TABLE workspace_members ADD CONSTRAINT workspace_members_role_check "
            "CHECK (role IN ('owner', 'admin', 'editor', 'viewer'))"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS code_project_access (
                -- 被授权的代码项目。
                project_id UUID NOT NULL REFERENCES code_projects(project_id) ON DELETE CASCADE,
                -- 被授予访问权的用户。
                user_id TEXT NOT NULL,
                -- 当前 MVP 先支持 read；写入仍由服务端扫描接口控制。
                permission TEXT NOT NULL DEFAULT 'read' CHECK (permission IN ('read', 'write', 'admin')),
                -- 授权记录是否有效。
                status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
                -- 授权时间。
                granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (project_id, user_id)
            )
            """
        )
        default_name_key = _workspace_name_key(settings.current_workspace_name)
        conflicting_default = conn.execute(
            """
            SELECT workspace_id FROM workspaces
            WHERE owner_user_id = %s AND normalized_name = %s AND workspace_id <> %s
            """,
            (settings.current_user_id, default_name_key, settings.workspace_id),
        ).fetchone()
        if conflicting_default:
            default_name_key = f"{default_name_key}#{settings.workspace_id}"
        conn.execute(
            """
            INSERT INTO workspaces (workspace_id, workspace_name, normalized_name, owner_user_id)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (workspace_id) DO UPDATE
            SET workspace_name = EXCLUDED.workspace_name,
                normalized_name = EXCLUDED.normalized_name
            """, (settings.workspace_id, settings.current_workspace_name, default_name_key, settings.current_user_id)
        )
        conn.execute(
            """
            INSERT INTO workspace_members (workspace_id, user_id, role)
            VALUES (%s, %s, 'owner')
            ON CONFLICT (workspace_id, user_id) DO NOTHING
            """, (settings.workspace_id, settings.current_user_id)
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_workspace_active_owner "
            "ON workspace_members (workspace_id) WHERE role = 'owner' AND status = 'active'"
        )
        conn.execute(
            """
            COMMENT ON TABLE workspaces IS '工作空间主表；空间是 RAG 和 Code Wiki 的第一层隔离边界';
            COMMENT ON TABLE workspace_members IS '用户加入工作空间的服务端授权关系';
            COMMENT ON TABLE code_project_access IS '代码项目的可选用户级授权；不能跨工作空间生效';
            """
        )
        conn.execute("DROP INDEX IF EXISTS uq_workspaces_name_ci")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_workspaces_owner_name ON workspaces (owner_user_id, normalized_name)"
        )


def resolve_scope(requested_workspace_id: str | None = None, actor_user_id: str | None = None) -> AccessScope:
    """解析当前用户的授权空间和可访问项目集合。

    requested_workspace_id 只能从已加入的空间中选择；不存在或无权访问时直接拒绝。
    """
    actor_user_id = actor_user_id or settings.current_user_id
    with connection() as conn:
        if requested_workspace_id:
            workspace_id = requested_workspace_id.strip()
        else:
            # 默认空间可能已经退出或被管理员移除。优先取仍有效的默认空间，
            # 否则稳定回落到最早加入的空间，避免登录后陷入永久 403。
            selected = conn.execute(
                """
                SELECT w.workspace_id
                FROM users u
                JOIN workspace_members m ON m.workspace_id = u.default_workspace_id
                    AND m.user_id = u.user_id AND m.status = 'active'
                JOIN workspaces w ON w.workspace_id = m.workspace_id AND w.status = 'active'
                WHERE u.user_id = %s
                """, (actor_user_id,),
            ).fetchone()
            if not selected:
                selected = conn.execute(
                    """
                    SELECT w.workspace_id
                    FROM workspace_members m JOIN workspaces w ON w.workspace_id = m.workspace_id
                    WHERE m.user_id = %s AND m.status = 'active' AND w.status = 'active'
                    ORDER BY m.joined_at, w.workspace_id LIMIT 1
                    """, (actor_user_id,),
                ).fetchone()
            if not selected:
                raise PermissionError("当前用户尚未加入任何工作空间")
            workspace_id = str(selected[0])
        workspace = conn.execute(
            """
            SELECT w.workspace_id, w.workspace_name, m.role,
                   (SELECT department_id FROM users WHERE user_id = %s)
            FROM workspaces w
            JOIN workspace_members m ON m.workspace_id = w.workspace_id
            WHERE w.workspace_id = %s AND w.status = 'active'
              AND m.user_id = %s AND m.status = 'active'
            """, (actor_user_id, workspace_id, actor_user_id)
        ).fetchone()
        if not workspace:
            raise PermissionError("当前用户未加入该工作空间")
        projects = conn.execute(
            """
            SELECT p.project_id, p.project_name, p.current_commit,
                   ARRAY(
                     SELECT DISTINCT f.language FROM code_files f
                     WHERE f.project_id = p.project_id AND f.commit_hash = p.current_commit
                       AND f.language IS NOT NULL
                     ORDER BY f.language LIMIT 8
                   ) AS languages,
                   ARRAY(
                     SELECT DISTINCT a.name FROM code_architecture_facts a
                     WHERE a.project_id = p.project_id AND a.commit_hash = p.current_commit
                       AND a.fact_type = 'component'
                     ORDER BY a.name LIMIT 8
                   ) AS components
            FROM code_projects p
            WHERE p.workspace_id = %s AND p.status = 'active'
              AND (
                p.access_scope = 'workspace'
                OR p.owner_user_id = %s
                OR EXISTS (
                    SELECT 1 FROM code_project_access a
                    WHERE a.project_id = p.project_id AND a.user_id = %s AND a.status = 'active'
                )
              )
            """, (workspace_id, actor_user_id, actor_user_id)
        ).fetchall()
    return AccessScope(
        user_id=actor_user_id,
        workspace_id=str(workspace[0]),
        workspace_name=str(workspace[1]),
        department_id=workspace[3],
        allowed_project_ids=frozenset(str(row[0]) for row in projects),
        workspace_role=str(workspace[2]),
        authorized_projects=tuple({
            "project_id": str(row[0]), "project_name": str(row[1]),
            "current_commit": str(row[2] or "")[:12],
            "languages": list(row[3] or []), "components": list(row[4] or []),
        } for row in projects),
    )


def list_user_workspaces(actor_user_id: str | None = None) -> list[dict]:
    """返回当前用户已加入的空间；前端只能从这份服务端候选集切换。"""
    actor_user_id = actor_user_id or settings.current_user_id
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT w.workspace_id, w.workspace_name, w.workspace_type, m.role, m.joined_at
            FROM workspaces w
            JOIN workspace_members m ON m.workspace_id = w.workspace_id
            WHERE m.user_id = %s AND m.status = 'active' AND w.status = 'active'
            ORDER BY w.workspace_name, w.workspace_id
            """, (actor_user_id,)
        ).fetchall()
    return [
        {"workspace_id": str(row[0]), "workspace_name": row[1], "workspace_type": row[2], "role": row[3], "joined_at": row[4]}
        for row in rows
    ]


def set_default_workspace(actor_user_id: str, workspace_id: str) -> dict:
    """持久化空间切换；目标必须来自当前用户有效成员关系。"""
    with connection() as conn:
        row = conn.execute(
            """
            SELECT w.workspace_id, w.workspace_name, w.workspace_type, m.role
            FROM workspaces w JOIN workspace_members m ON m.workspace_id = w.workspace_id
            WHERE w.workspace_id = %s AND w.status = 'active'
              AND m.user_id = %s AND m.status = 'active'
            """, (workspace_id, actor_user_id),
        ).fetchone()
        if not row:
            raise PermissionError("当前用户未加入该工作空间")
        conn.execute(
            "UPDATE users SET default_workspace_id = %s, updated_at = NOW() WHERE user_id = %s",
            (workspace_id, actor_user_id),
        )
    return {
        "workspace_id": str(row[0]), "workspace_name": row[1],
        "workspace_type": row[2], "role": row[3],
    }


def _repair_default_workspace(conn, user_id: str, inaccessible_workspace_id: str) -> None:
    """成员关系失效后，将默认空间改为另一个仍可访问的空间。"""
    replacement = conn.execute(
        """
        SELECT w.workspace_id
        FROM workspace_members m JOIN workspaces w ON w.workspace_id = m.workspace_id
        WHERE m.user_id = %s AND m.status = 'active' AND w.status = 'active'
          AND w.workspace_id <> %s
        ORDER BY m.joined_at, w.workspace_id LIMIT 1
        """, (user_id, inaccessible_workspace_id),
    ).fetchone()
    conn.execute(
        """
        UPDATE users SET default_workspace_id = %s, updated_at = NOW()
        WHERE user_id = %s AND default_workspace_id = %s
        """, (str(replacement[0]) if replacement else None, user_id, inaccessible_workspace_id),
    )


def can_manage_document(document_id: str, scope: AccessScope) -> bool:
    """owner/admin 可治理全空间；editor 只能治理自己拥有或创建的文档。"""
    with connection() as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM documents d
            JOIN workspace_members m ON m.workspace_id = d.workspace_id
            WHERE d.document_id = %s AND d.workspace_id = %s
              AND m.user_id = %s AND m.status = 'active'
              AND (
                m.role IN ('owner', 'admin')
                OR (m.role = 'editor' AND (d.owner_user_id = %s OR d.created_by_user_id = %s))
              )
            """,
            (document_id, scope.workspace_id, scope.user_id, scope.user_id, scope.user_id),
        ).fetchone()
    return row is not None


def can_read_document(document_id: str, scope: AccessScope) -> bool:
    """直接读取修订历史时复用与检索一致的 personal/department/workspace 边界。"""
    with connection() as conn:
        row = conn.execute(
            """
            SELECT 1 FROM documents d
            WHERE d.document_id = %s AND d.workspace_id = %s
              AND (
                d.scope_type = 'workspace'
                OR (d.scope_type = 'personal' AND d.owner_user_id = %s)
                OR (d.scope_type = 'department' AND d.owner_department_id = %s)
              )
            """,
            (document_id, scope.workspace_id, scope.user_id, scope.department_id),
        ).fetchone()
    return row is not None


def _normalize_workspace_name(workspace_name: str) -> str:
    """统一 Unicode 和空白，避免视觉同名绕过重复约束。"""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", workspace_name)).strip()


def _workspace_name_key(workspace_name: str) -> str:
    return _normalize_workspace_name(workspace_name).casefold()


def create_workspace(workspace_name: str, actor_user_id: str | None = None) -> dict:
    """原子创建空间并把当前服务端身份登记为 owner。

    UUID 是不可猜测的稳定主键；名称只用于展示，不能作为授权凭据。
    """
    display_name = _normalize_workspace_name(workspace_name)
    actor_user_id = actor_user_id or settings.current_user_id
    normalized_name = display_name.casefold()
    if not 2 <= len(display_name) <= 50:
        raise ValueError("工作空间名称长度必须为 2 到 50 个字符")
    if any(ord(char) < 32 for char in display_name) or any(char in "/\\" for char in display_name):
        raise ValueError("工作空间名称不能包含控制字符")

    workspace_id = str(uuid.uuid4())
    with connection() as conn:
        inserted = conn.execute(
            """
            INSERT INTO workspaces (workspace_id, workspace_name, normalized_name, owner_user_id, workspace_type)
            VALUES (%s, %s, %s, %s, 'team')
            ON CONFLICT (owner_user_id, normalized_name) DO NOTHING
            RETURNING workspace_id
            """,
            (workspace_id, display_name, normalized_name, actor_user_id),
        ).fetchone()
        if not inserted:
            raise WorkspaceNameConflict("当前用户已创建同名工作空间")
        # 与空间主记录处于同一个数据库事务；任何一步失败都会一起回滚。
        conn.execute(
            """
            INSERT INTO workspace_members (workspace_id, user_id, role)
            VALUES (%s, %s, 'owner')
            """,
            (workspace_id, actor_user_id),
        )
    return {
        "workspace_id": workspace_id,
        "workspace_name": display_name,
        "owner_user_id": actor_user_id,
        "role": "owner",
    }


def require_workspace_role(scope: AccessScope, *allowed_roles: str) -> None:
    """业务写入口使用统一角色门禁，不能把按钮隐藏当作权限控制。"""
    if scope.workspace_role not in allowed_roles:
        raise PermissionError("当前角色无权执行此操作")


def list_workspace_members(scope: AccessScope) -> dict:
    require_workspace_role(scope, "owner", "admin")
    with connection() as conn:
        members = conn.execute(
            """
            SELECT u.user_id, u.email, u.display_name, m.role, m.joined_at
            FROM workspace_members m JOIN users u ON u.user_id = m.user_id
            WHERE m.workspace_id = %s AND m.status = 'active'
            ORDER BY CASE m.role WHEN 'owner' THEN 0 WHEN 'admin' THEN 1 WHEN 'editor' THEN 2 ELSE 3 END,
                     u.display_name, u.user_id
            """, (scope.workspace_id,),
        ).fetchall()
        invitations = conn.execute(
            """
            SELECT invitation_id, normalized_email, role, expires_at, created_at
            FROM workspace_invitations
            WHERE workspace_id = %s AND accepted_at IS NULL AND revoked_at IS NULL AND expires_at > NOW()
            ORDER BY created_at DESC
            """, (scope.workspace_id,),
        ).fetchall()
    return {
        "members": [
            {"user_id": str(row[0]), "email": row[1], "display_name": row[2], "role": row[3], "joined_at": row[4]}
            for row in members
        ],
        "invitations": [
            {"invitation_id": str(row[0]), "email": row[1], "role": row[2], "expires_at": row[3], "created_at": row[4]}
            for row in invitations
        ],
    }


def create_workspace_invitation(scope: AccessScope, email: str, role: str) -> dict:
    """创建一次性邀请；数据库只保存 token 摘要，原 token 仅返回一次。"""
    require_workspace_role(scope, "owner", "admin")
    if role not in {"admin", "editor", "viewer"}:
        raise ValueError("邀请角色必须是 admin、editor 或 viewer")
    if scope.workspace_role == "admin" and role == "admin":
        raise PermissionError("只有 owner 可以邀请 admin")
    normalized_email = unicodedata.normalize("NFKC", email).strip().casefold()
    if len(normalized_email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", normalized_email):
        raise ValueError("请输入有效邮箱")
    raw_token = secrets.token_urlsafe(32)
    token_hash = sha256(raw_token.encode("utf-8")).hexdigest()
    invitation_id = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    with connection() as conn:
        workspace_type = conn.execute(
            "SELECT workspace_type FROM workspaces WHERE workspace_id = %s", (scope.workspace_id,)
        ).fetchone()
        if not workspace_type or workspace_type[0] != "team":
            raise ValueError("个人空间不能邀请成员")
        if conn.execute(
            """
            SELECT 1 FROM workspace_members m JOIN users u ON u.user_id = m.user_id
            WHERE m.workspace_id = %s AND m.status = 'active' AND u.normalized_email = %s
            """, (scope.workspace_id, normalized_email),
        ).fetchone():
            raise ValueError("该用户已是空间成员")
        conn.execute(
            """
            INSERT INTO workspace_invitations
                (invitation_id, workspace_id, normalized_email, role, token_hash,
                 invited_by_user_id, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (workspace_id, normalized_email)
                WHERE accepted_at IS NULL AND revoked_at IS NULL
            DO UPDATE SET role = EXCLUDED.role, token_hash = EXCLUDED.token_hash,
                          invited_by_user_id = EXCLUDED.invited_by_user_id,
                          expires_at = EXCLUDED.expires_at, created_at = NOW()
            RETURNING invitation_id
            """,
            (invitation_id, scope.workspace_id, normalized_email, role, token_hash, scope.user_id, expires_at),
        )
    return {"invitation_id": invitation_id, "email": normalized_email, "role": role, "token": raw_token, "expires_at": expires_at}


def accept_workspace_invitation(raw_token: str, user_id: str, user_email: str) -> dict:
    token_hash = sha256(raw_token.strip().encode("utf-8")).hexdigest()
    normalized_email = unicodedata.normalize("NFKC", user_email).strip().casefold()
    with connection() as conn:
        invitation = conn.execute(
            """
            SELECT invitation_id, workspace_id, normalized_email, role
            FROM workspace_invitations
            WHERE token_hash = %s AND accepted_at IS NULL AND revoked_at IS NULL
              AND expires_at > NOW() FOR UPDATE
            """, (token_hash,),
        ).fetchone()
        if not invitation or invitation[2] != normalized_email:
            raise PermissionError("邀请无效、已过期或与当前登录邮箱不匹配")
        existing = conn.execute(
            """
            SELECT role, status FROM workspace_members
            WHERE workspace_id = %s AND user_id = %s FOR UPDATE
            """, (invitation[1], user_id),
        ).fetchone()
        accepted_role = str(existing[0]) if existing and existing[1] == "active" else str(invitation[3])
        if existing:
            # 旧邀请不能降低已有有效角色；停用成员则按本次邀请重新加入。
            if existing[1] != "active":
                conn.execute(
                    "UPDATE workspace_members SET role = %s, status = 'active', joined_at = NOW() "
                    "WHERE workspace_id = %s AND user_id = %s",
                    (invitation[3], invitation[1], user_id),
                )
        else:
            conn.execute(
                "INSERT INTO workspace_members (workspace_id, user_id, role, status) VALUES (%s, %s, %s, 'active')",
                (invitation[1], user_id, invitation[3]),
            )
        conn.execute(
            """
            UPDATE workspace_invitations
            SET accepted_by_user_id = %s, accepted_at = NOW()
            WHERE invitation_id = %s
            """, (user_id, invitation[0]),
        )
        conn.execute(
            """
            UPDATE users SET default_workspace_id = COALESCE(default_workspace_id, %s), updated_at = NOW()
            WHERE user_id = %s
            """, (invitation[1], user_id),
        )
        workspace = conn.execute(
            "SELECT workspace_name FROM workspaces WHERE workspace_id = %s", (invitation[1],)
        ).fetchone()
    return {"workspace_id": str(invitation[1]), "workspace_name": workspace[0], "role": accepted_role}


def revoke_workspace_invitation(scope: AccessScope, invitation_id: str) -> bool:
    require_workspace_role(scope, "owner", "admin")
    with connection() as conn:
        row = conn.execute(
            """
            UPDATE workspace_invitations SET revoked_at = NOW()
            WHERE invitation_id = %s AND workspace_id = %s
              AND accepted_at IS NULL AND revoked_at IS NULL RETURNING invitation_id
            """, (invitation_id, scope.workspace_id),
        ).fetchone()
    return row is not None


def update_workspace_member_role(scope: AccessScope, target_user_id: str, role: str) -> bool:
    require_workspace_role(scope, "owner")
    if role not in {"admin", "editor", "viewer"}:
        raise ValueError("成员角色必须是 admin、editor 或 viewer")
    with connection() as conn:
        row = conn.execute(
            """
            UPDATE workspace_members SET role = %s
            WHERE workspace_id = %s AND user_id = %s AND role <> 'owner' AND status = 'active'
            RETURNING user_id
            """, (role, scope.workspace_id, target_user_id),
        ).fetchone()
    return row is not None


def remove_workspace_member(scope: AccessScope, target_user_id: str) -> bool:
    require_workspace_role(scope, "owner", "admin")
    if target_user_id == scope.user_id:
        raise ValueError("请使用退出空间操作")
    with connection() as conn:
        target = conn.execute(
            "SELECT role FROM workspace_members WHERE workspace_id = %s AND user_id = %s AND status = 'active'",
            (scope.workspace_id, target_user_id),
        ).fetchone()
        if not target or target[0] == "owner" or (scope.workspace_role == "admin" and target[0] == "admin"):
            return False
        conn.execute(
            "UPDATE workspace_members SET status = 'disabled' WHERE workspace_id = %s AND user_id = %s",
            (scope.workspace_id, target_user_id),
        )
        _repair_default_workspace(conn, target_user_id, scope.workspace_id)
    return True


def leave_workspace(scope: AccessScope) -> None:
    if scope.workspace_role == "owner":
        raise ValueError("owner 必须先转移所有权，不能直接退出空间")
    with connection() as conn:
        conn.execute(
            "UPDATE workspace_members SET status = 'disabled' WHERE workspace_id = %s AND user_id = %s",
            (scope.workspace_id, scope.user_id),
        )
        _repair_default_workspace(conn, scope.user_id, scope.workspace_id)


def transfer_workspace_ownership(scope: AccessScope, target_user_id: str) -> None:
    require_workspace_role(scope, "owner")
    if target_user_id == scope.user_id:
        raise ValueError("目标用户已经是 owner")
    with connection() as conn:
        # 串行化同一空间的转让，配合唯一索引保证只有一个有效 owner。
        workspace = conn.execute(
            "SELECT owner_user_id, normalized_name FROM workspaces WHERE workspace_id = %s FOR UPDATE",
            (scope.workspace_id,),
        ).fetchone()
        if not workspace or str(workspace[0]) != scope.user_id:
            raise PermissionError("当前用户不再是空间 owner")
        target = conn.execute(
            """
            SELECT 1 FROM workspace_members
            WHERE workspace_id = %s AND user_id = %s AND status = 'active' FOR UPDATE
            """, (scope.workspace_id, target_user_id),
        ).fetchone()
        if not target:
            raise ValueError("目标用户不是当前空间成员")
        if conn.execute(
            """
            SELECT 1 FROM workspaces
            WHERE owner_user_id = %s AND normalized_name = %s AND workspace_id <> %s
            """, (target_user_id, workspace[1], scope.workspace_id),
        ).fetchone():
            raise ValueError("目标用户已拥有同名工作空间，请先修改空间名称")
        conn.execute("UPDATE workspace_members SET role = 'admin' WHERE workspace_id = %s AND user_id = %s", (scope.workspace_id, scope.user_id))
        conn.execute("UPDATE workspace_members SET role = 'owner' WHERE workspace_id = %s AND user_id = %s", (scope.workspace_id, target_user_id))
        conn.execute("UPDATE workspaces SET owner_user_id = %s, updated_at = NOW() WHERE workspace_id = %s", (target_user_id, scope.workspace_id))
