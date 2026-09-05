"""用户认证、可撤销会话和请求级身份上下文。

浏览器持有随机会话令牌，PostgreSQL 只保存 SHA-256 摘要。业务模块只能读取
``AuthenticatedPrincipal``，不能从表单、查询参数或环境变量推断当前用户。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import json
import logging
import re
import secrets
import threading
import time
import unicodedata
import uuid

from pwdlib import PasswordHash
from starlette.responses import JSONResponse

from .config import settings
from .db import connection
from .errors import error_payload


logger = logging.getLogger(__name__)
password_hash = PasswordHash.recommended()
_DUMMY_PASSWORD_HASH = password_hash.hash("not-a-real-user-password")
_principal_context: ContextVar[AuthenticatedPrincipal | None] = ContextVar("principal", default=None)
_login_attempts: dict[str, list[float]] = {}
_login_attempts_lock = threading.Lock()


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """一次已认证请求的最小身份，不包含密码、Cookie 原文或业务权限。"""

    user_id: str
    session_id: str
    email: str
    display_name: str
    department_id: str | None
    default_workspace_id: str | None
    csrf_token_hash: str


@dataclass(frozen=True)
class SessionBundle:
    """仅在创建会话时短暂存在；原始令牌只返回给浏览器。"""

    principal: AuthenticatedPrincipal
    session_token: str
    csrf_token: str
    expires_at: datetime


class AuthenticationError(ValueError):
    """认证失败；接口统一返回相同信息，避免枚举账号。"""


class RegistrationConflict(ValueError):
    """规范化邮箱已被注册。"""


class HandoverError(ValueError):
    """员工数据交接不满足安全或状态约束。"""


class RateLimitError(RuntimeError):
    """认证入口触发共享或进程内限流，并告知客户端最迟重试时间。"""

    def __init__(self, message: str, retry_after: int) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _token_hash(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def normalize_email(email: str) -> str:
    """邮箱登录名使用 NFKC、去空白和大小写折叠生成唯一键。"""
    return unicodedata.normalize("NFKC", email).strip().casefold()


def _validate_registration(email: str, password: str, display_name: str) -> tuple[str, str]:
    normalized_email = normalize_email(email)
    display_name = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", display_name)).strip()
    if len(normalized_email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", normalized_email):
        raise ValueError("请输入有效邮箱")
    if not 2 <= len(display_name) <= 50:
        raise ValueError("显示名称长度必须为 2 到 50 个字符")
    if not 12 <= len(password) <= 128:
        raise ValueError("密码长度必须为 12 到 128 个字符")
    return normalized_email, display_name


def initialize_auth() -> None:
    """创建认证表，并写入可被首位注册用户接管的旧开发身份。"""
    with connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                -- 不可变用户标识；新用户使用 UUID，旧数据首次注册时沿用迁移标识。
                user_id TEXT PRIMARY KEY,
                -- 邮箱原始展示值；尚未被注册的迁移用户可以为空。
                email TEXT,
                -- NFKC + casefold 后的登录唯一键。
                normalized_email TEXT,
                -- 页面展示名，同时写入新文档的作者快照。
                display_name TEXT NOT NULL,
                -- 权威部门标识；用户不能在上传表单中自行伪造。
                department_id TEXT,
                -- 登录后优先进入的工作空间。
                default_workspace_id TEXT,
                -- pending_claim 仅用于旧数据接管，active 才允许登录。
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('pending_claim', 'active', 'disabled')),
                -- 修改密码或全量登出时递增，使旧会话立即失效。
                auth_version INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_normalized_email "
            "ON users (normalized_email) WHERE normalized_email IS NOT NULL"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_credentials (
                -- 一个用户当前只有 password 凭据，拆表为后续 OIDC/LDAP 保留边界。
                user_id TEXT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                password_hash TEXT NOT NULL,
                password_changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id UUID PRIMARY KEY,
                -- 原始 Session Token 永不入库，只保存固定长度摘要。
                token_hash TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                csrf_token_hash TEXT NOT NULL,
                auth_version INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                idle_expires_at TIMESTAMPTZ NOT NULL,
                expires_at TIMESTAMPTZ NOT NULL,
                revoked_at TIMESTAMPTZ,
                ip_address TEXT,
                user_agent TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS ix_user_sessions_user ON user_sessions (user_id, revoked_at)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workspace_invitations (
                invitation_id UUID PRIMARY KEY,
                workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
                normalized_email TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('admin', 'editor', 'viewer')),
                token_hash TEXT NOT NULL UNIQUE,
                invited_by_user_id TEXT NOT NULL REFERENCES users(user_id),
                expires_at TIMESTAMPTZ NOT NULL,
                accepted_by_user_id TEXT REFERENCES users(user_id),
                accepted_at TIMESTAMPTZ,
                revoked_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_active_workspace_invite "
            "ON workspace_invitations (workspace_id, normalized_email) "
            "WHERE accepted_at IS NULL AND revoked_at IS NULL"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_data_handovers (
                -- 一次员工离职交接的稳定标识，用于审计而不是作为接管凭据。
                handover_id UUID PRIMARY KEY,
                -- 即将离职且主动发起交接的原用户。
                source_user_id TEXT NOT NULL REFERENCES users(user_id),
                -- 被指定接收数据的员工邮箱；接管码不能由其他账号使用。
                target_normalized_email TEXT NOT NULL,
                -- 原始接管码只显示一次，数据库仅保存 SHA-256 摘要。
                token_hash TEXT NOT NULL UNIQUE,
                -- 接管码过期时间；当前固定为创建后 24 小时。
                expires_at TIMESTAMPTZ NOT NULL,
                -- 实际完成接管的用户；完成前为空。
                accepted_by_user_id TEXT REFERENCES users(user_id),
                -- 完成时间；非空表示令牌已消费，不能再次使用。
                accepted_at TIMESTAMPTZ,
                -- 发起人重新生成接管码时，旧码会被显式撤销。
                revoked_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_active_user_handover "
            "ON user_data_handovers (source_user_id) "
            "WHERE accepted_at IS NULL AND revoked_at IS NULL"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id UUID PRIMARY KEY,
                actor_user_id TEXT,
                workspace_id TEXT,
                action TEXT NOT NULL,
                object_type TEXT,
                object_id TEXT,
                outcome TEXT NOT NULL CHECK (outcome IN ('success', 'denied', 'failed')),
                details JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_rate_limits (
                rate_key TEXT PRIMARY KEY,
                window_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                attempts INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        # 旧项目先登记成不可登录的占位用户。首位注册者接管该 ID，原空间和知识无需搬迁。
        conn.execute(
            """
            INSERT INTO users (user_id, display_name, department_id, default_workspace_id, status)
            VALUES (%s, %s, %s, %s, 'pending_claim')
            ON CONFLICT (user_id) DO NOTHING
            """,
            (settings.current_user_id, "User", settings.current_department_id, settings.workspace_id),
        )
        # 认证模块上线前产生的资源归属于旧占位用户；回填后才能启用完整关系约束。
        conn.execute(
            "UPDATE documents SET created_by_user_id = %s WHERE created_by_user_id IS NULL",
            (settings.current_user_id,),
        )
        conn.execute(
            "UPDATE code_projects SET created_by_user_id = %s WHERE created_by_user_id IS NULL",
            (settings.current_user_id,),
        )
        # Chunk 同时保存 workspace_id 以提高检索过滤效率；复合唯一键让数据库能
        # 强制它与所属 Document 的空间一致，避免迁移或脚本写入跨空间脏数据。
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_id_workspace "
            "ON documents (document_id, workspace_id)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_id_document_workspace "
            "ON knowledge_chunks (chunk_id, document_id, workspace_id)"
        )
        conn.execute(
            """
            COMMENT ON TABLE users IS '用户主体；认证身份与业务资源 owner 的根节点';
            COMMENT ON TABLE user_credentials IS '密码凭据；仅保存 Argon2id 编码哈希';
            COMMENT ON TABLE user_sessions IS '服务端可撤销会话；数据库不保存浏览器原始令牌';
            COMMENT ON TABLE workspace_invitations IS '一次性工作空间邀请；只保存邀请令牌摘要';
            COMMENT ON TABLE user_data_handovers IS '员工离职数据交接单；接管码一次性、限时且绑定接收邮箱';
            COMMENT ON TABLE audit_events IS '认证与授权安全事件，不保存密码、Cookie 或源码正文';
            """
        )
        # 旧库可能含历史脏数据，因此先用 NOT VALID 建立“新写入必须合法”的边界；
        # 后续数据治理完成后可执行 VALIDATE CONSTRAINT，而不会阻塞本次升级。
        conn.execute(
            """
            DO $$ BEGIN
              IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_workspaces_owner_user') THEN
                ALTER TABLE workspaces ADD CONSTRAINT fk_workspaces_owner_user
                  FOREIGN KEY (owner_user_id) REFERENCES users(user_id) NOT VALID;
              END IF;
              IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_workspace_members_user') THEN
                ALTER TABLE workspace_members ADD CONSTRAINT fk_workspace_members_user
                  FOREIGN KEY (user_id) REFERENCES users(user_id) NOT VALID;
              END IF;
              IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_users_default_workspace') THEN
                ALTER TABLE users ADD CONSTRAINT fk_users_default_workspace
                  FOREIGN KEY (default_workspace_id) REFERENCES workspaces(workspace_id) ON DELETE SET NULL NOT VALID;
              END IF;
              IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_documents_workspace') THEN
                ALTER TABLE documents ADD CONSTRAINT fk_documents_workspace
                  FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id) NOT VALID;
              END IF;
              IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_documents_owner_user') THEN
                ALTER TABLE documents ADD CONSTRAINT fk_documents_owner_user
                  FOREIGN KEY (owner_user_id) REFERENCES users(user_id) NOT VALID;
              END IF;
              IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_documents_creator_user') THEN
                ALTER TABLE documents ADD CONSTRAINT fk_documents_creator_user
                  FOREIGN KEY (created_by_user_id) REFERENCES users(user_id) NOT VALID;
              END IF;
              IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_documents_scope_owner') THEN
                ALTER TABLE documents ADD CONSTRAINT ck_documents_scope_owner CHECK (
                  (scope_type = 'personal' AND owner_user_id IS NOT NULL)
                  OR (scope_type = 'department' AND owner_department_id IS NOT NULL)
                  OR scope_type = 'workspace'
                ) NOT VALID;
              END IF;
              IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_documents_creator') THEN
                ALTER TABLE documents ADD CONSTRAINT ck_documents_creator
                  CHECK (created_by_user_id IS NOT NULL) NOT VALID;
              END IF;
              IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_code_projects_workspace') THEN
                ALTER TABLE code_projects ADD CONSTRAINT fk_code_projects_workspace
                  FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id) NOT VALID;
              END IF;
              IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_code_projects_owner_user') THEN
                ALTER TABLE code_projects ADD CONSTRAINT fk_code_projects_owner_user
                  FOREIGN KEY (owner_user_id) REFERENCES users(user_id) NOT VALID;
              END IF;
              IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_code_projects_creator_user') THEN
                ALTER TABLE code_projects ADD CONSTRAINT fk_code_projects_creator_user
                  FOREIGN KEY (created_by_user_id) REFERENCES users(user_id) NOT VALID;
              END IF;
              IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_knowledge_chunks_document') THEN
                ALTER TABLE knowledge_chunks ADD CONSTRAINT fk_knowledge_chunks_document
                  FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE NOT VALID;
              END IF;
              IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_knowledge_chunks_document_workspace') THEN
                ALTER TABLE knowledge_chunks ADD CONSTRAINT fk_knowledge_chunks_document_workspace
                  FOREIGN KEY (document_id, workspace_id)
                  REFERENCES documents(document_id, workspace_id) ON DELETE CASCADE NOT VALID;
              END IF;
              IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_knowledge_chunks_parent_scope') THEN
                ALTER TABLE knowledge_chunks ADD CONSTRAINT fk_knowledge_chunks_parent_scope
                  FOREIGN KEY (parent_chunk_id, document_id, workspace_id)
                  REFERENCES knowledge_chunks(chunk_id, document_id, workspace_id)
                  ON DELETE CASCADE NOT VALID;
              END IF;
            END $$;
            """
        )
        # 回填完成后立即验证，避免权限字段只有“未来写入有效”而历史数据仍游离。
        for constraint, table in (
            ("fk_workspaces_owner_user", "workspaces"),
            ("fk_workspace_members_user", "workspace_members"),
            ("fk_users_default_workspace", "users"),
            ("fk_documents_workspace", "documents"),
            ("fk_documents_owner_user", "documents"),
            ("fk_documents_creator_user", "documents"),
            ("ck_documents_scope_owner", "documents"),
            ("ck_documents_creator", "documents"),
            ("fk_code_projects_workspace", "code_projects"),
            ("fk_code_projects_owner_user", "code_projects"),
            ("fk_code_projects_creator_user", "code_projects"),
            ("fk_knowledge_chunks_document", "knowledge_chunks"),
            ("fk_knowledge_chunks_document_workspace", "knowledge_chunks"),
            ("fk_knowledge_chunks_parent_scope", "knowledge_chunks"),
        ):
            conn.execute(f'ALTER TABLE "{table}" VALIDATE CONSTRAINT "{constraint}"')
        conn.execute("ALTER TABLE knowledge_chunks ALTER COLUMN document_id SET NOT NULL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            "INSERT INTO schema_migrations (version) VALUES ('2026-09-chunk-workspace-fk-v1') ON CONFLICT DO NOTHING"
        )
        conn.execute(
            "INSERT INTO schema_migrations (version) VALUES ('2026-09-parent-chunk-scope-fk-v1') ON CONFLICT DO NOTHING"
        )
        conn.execute(
            "INSERT INTO schema_migrations (version) VALUES ('2026-09-code-snapshots-v1') ON CONFLICT DO NOTHING"
        )
        conn.execute(
            "INSERT INTO schema_migrations (version) VALUES ('2026-09-code-snapshots-v2') ON CONFLICT DO NOTHING"
        )
        if settings.legacy_bootstrap_token and len(settings.legacy_bootstrap_token.encode("utf-8")) < 32:
            raise RuntimeError("LEGACY_BOOTSTRAP_TOKEN 启用时至少需要 32 字节")


def _consume_registration_limit(ip_address: str | None, bootstrap_attempt: bool) -> None:
    """使用 PostgreSQL 固定窗口限流，使多进程部署共享注册与接管尝试次数。"""
    identity = ip_address or "unknown"
    purpose = "bootstrap" if bootstrap_attempt else "register"
    rate_key = _token_hash(f"{purpose}:{identity}")
    limit = 5 if bootstrap_attempt else 20
    with connection() as conn:
        row = conn.execute(
            """
            INSERT INTO auth_rate_limits (rate_key, attempts)
            VALUES (%s, 1)
            ON CONFLICT (rate_key) DO UPDATE SET
              attempts = CASE
                WHEN auth_rate_limits.window_started_at < NOW() - INTERVAL '1 hour' THEN 1
                ELSE auth_rate_limits.attempts + 1
              END,
              window_started_at = CASE
                WHEN auth_rate_limits.window_started_at < NOW() - INTERVAL '1 hour' THEN NOW()
                ELSE auth_rate_limits.window_started_at
              END
            RETURNING attempts
            """,
            (rate_key,),
        ).fetchone()
    if row[0] > limit:
        raise RateLimitError("注册尝试过于频繁，请稍后再试", 3600)


def _audit(conn, actor_user_id: str | None, action: str, outcome: str, *, workspace_id: str | None = None, object_type: str | None = None, object_id: str | None = None, details: str = "{}") -> None:
    conn.execute(
        """
        INSERT INTO audit_events
            (event_id, actor_user_id, workspace_id, action, object_type, object_id, outcome, details)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        """,
        (uuid.uuid4(), actor_user_id, workspace_id, action, object_type, object_id, outcome, details),
    )


def record_audit_event(
    actor_user_id: str | None,
    action: str,
    *,
    outcome: str = "success",
    workspace_id: str | None = None,
    object_type: str | None = None,
    object_id: str | None = None,
    details: dict | None = None,
) -> None:
    """尽力记录审计；审计故障不能把已经提交的业务操作伪装成失败。"""
    try:
        with connection() as conn:
            _audit(
                conn, actor_user_id, action, outcome, workspace_id=workspace_id,
                object_type=object_type, object_id=object_id,
                details=json.dumps(details or {}, ensure_ascii=False),
            )
    except Exception:
        logger.exception("audit_event_write_failed action=%s outcome=%s", action, outcome)


def _create_session(conn, user_row, ip_address: str | None, user_agent: str | None) -> SessionBundle:
    session_id = str(uuid.uuid4())
    session_token = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=settings.session_absolute_hours)
    idle_expires_at = min(expires_at, now + timedelta(minutes=settings.session_idle_minutes))
    conn.execute(
        """
        INSERT INTO user_sessions
            (session_id, token_hash, user_id, csrf_token_hash, auth_version,
             idle_expires_at, expires_at, ip_address, user_agent)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            session_id, _token_hash(session_token), user_row[0], _token_hash(csrf_token),
            user_row[6], idle_expires_at, expires_at, ip_address, (user_agent or "")[:500],
        ),
    )
    principal = AuthenticatedPrincipal(
        user_id=str(user_row[0]), email=str(user_row[1]), display_name=str(user_row[3]),
        department_id=user_row[4], default_workspace_id=user_row[5],
        session_id=session_id, csrf_token_hash=_token_hash(csrf_token),
    )
    return SessionBundle(principal, session_token, csrf_token, expires_at)


def register_user(
    email: str,
    password: str,
    display_name: str,
    ip_address: str | None = None,
    user_agent: str | None = None,
    bootstrap_token: str | None = None,
    handover_token: str | None = None,
) -> SessionBundle:
    """注册用户并创建个人空间；只有显式接管码可以认领旧数据。"""
    if not settings.registration_enabled:
        raise PermissionError("当前系统已关闭自主注册")
    normalized_email, display_name = _validate_registration(email, password, display_name)
    if bootstrap_token and handover_token:
        raise PermissionError("系统迁移码和员工交接码不能同时使用")
    # 在执行昂贵的 Argon2id 前先使用数据库共享限流，降低匿名 CPU/存储耗尽风险。
    _consume_registration_limit(ip_address, bool(bootstrap_token))
    encoded_password = password_hash.hash(password)
    with connection() as conn:
        if conn.execute("SELECT 1 FROM users WHERE normalized_email = %s", (normalized_email,)).fetchone():
            raise RegistrationConflict("该邮箱已注册")
        legacy = None
        # loopback、代理来源地址都不是授权凭据。接管码为空或不匹配时只创建普通账号。
        valid_bootstrap = bool(
            settings.legacy_bootstrap_token
            and bootstrap_token
            and hmac.compare_digest(bootstrap_token, settings.legacy_bootstrap_token)
        )
        if bootstrap_token and not valid_bootstrap:
            raise PermissionError("旧数据接管码无效")
        if valid_bootstrap:
            legacy = conn.execute(
                "SELECT user_id FROM users WHERE status = 'pending_claim' ORDER BY created_at LIMIT 1 FOR UPDATE"
            ).fetchone()
            if not legacy:
                raise PermissionError("旧数据已被接管或当前没有可接管数据")
        if legacy:
            user_id = str(legacy[0])
            conn.execute(
                """
                UPDATE users SET email = %s, normalized_email = %s, display_name = %s,
                    status = 'active', updated_at = NOW()
                WHERE user_id = %s
                """, (email.strip(), normalized_email, display_name, user_id),
            )
        else:
            user_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO users (user_id, email, normalized_email, display_name, status)
                VALUES (%s, %s, %s, %s, 'active')
                """, (user_id, email.strip(), normalized_email, display_name),
            )
        # 即使接管了旧团队空间，账号仍有独立个人空间，避免个人与共享数据混在一起。
        workspace_id = str(uuid.uuid4())
        workspace_name = f"{display_name} 的个人空间"
        workspace_key = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", workspace_name)).strip().casefold()
        conn.execute(
            """
            INSERT INTO workspaces
                (workspace_id, workspace_name, normalized_name, owner_user_id, workspace_type)
            VALUES (%s, %s, %s, %s, 'personal')
            """, (workspace_id, workspace_name, workspace_key, user_id),
        )
        conn.execute(
            "INSERT INTO workspace_members (workspace_id, user_id, role) VALUES (%s, %s, 'owner')",
            (workspace_id, user_id),
        )
        conn.execute("UPDATE users SET default_workspace_id = %s WHERE user_id = %s", (workspace_id, user_id))
        conn.execute(
            "INSERT INTO user_credentials (user_id, password_hash) VALUES (%s, %s)",
            (user_id, encoded_password),
        )
        user_row = conn.execute(
            """
            SELECT user_id, email, normalized_email, display_name, department_id,
                   default_workspace_id, auth_version
            FROM users WHERE user_id = %s
            """, (user_id,),
        ).fetchone()
        bundle = _create_session(conn, user_row, ip_address, user_agent)
        _audit(conn, user_id, "auth.register", "success", object_type="user", object_id=user_id)
        # 新员工可在注册时直接完成交接；嵌套 connection 复用当前事务，
        # 接管码无效时用户注册和资源迁移会一起回滚，不留下半注册账号。
        if handover_token:
            accept_data_handover(user_id, handover_token)
        return bundle


def _check_login_rate(key: str) -> None:
    now = time.monotonic()
    with _login_attempts_lock:
        attempts = [value for value in _login_attempts.get(key, []) if now - value < 900]
        if len(attempts) >= 8:
            raise RateLimitError("登录尝试过多，请稍后再试", 900)
        attempts.append(now)
        _login_attempts[key] = attempts


def login_user(email: str, password: str, ip_address: str | None = None, user_agent: str | None = None) -> SessionBundle:
    normalized_email = normalize_email(email)
    rate_key = f"{ip_address or '-'}:{normalized_email}"
    _check_login_rate(rate_key)
    denied_actor_id: str | None = None
    with connection() as conn:
        row = conn.execute(
            """
            SELECT u.user_id, u.email, u.normalized_email, u.display_name, u.department_id,
                   u.default_workspace_id, u.auth_version, c.password_hash
            FROM users u JOIN user_credentials c ON c.user_id = u.user_id
            WHERE u.normalized_email = %s AND u.status = 'active'
            """, (normalized_email,),
        ).fetchone()
        candidate_hash = row[7] if row else _DUMMY_PASSWORD_HASH
        valid = password_hash.verify(password, candidate_hash)
        if not row or not valid:
            denied_actor_id = str(row[0]) if row else None
        else:
            bundle = _create_session(conn, row, ip_address, user_agent)
            _audit(conn, str(row[0]), "auth.login", "success", object_type="session", object_id=bundle.principal.session_id)
    if denied_actor_id is not None or not row or not valid:
        # 拒绝事件必须在登录事务结束后独立提交，否则随后抛出的异常会回滚审计。
        record_audit_event(denied_actor_id, "auth.login", outcome="denied")
        raise AuthenticationError("邮箱或密码错误")
    with _login_attempts_lock:
        _login_attempts.pop(rate_key, None)
    return bundle


def authenticate_session(session_token: str | None) -> AuthenticatedPrincipal | None:
    if not session_token:
        return None
    token_digest = _token_hash(session_token)
    with connection() as conn:
        row = conn.execute(
            """
            SELECT u.user_id, u.email, u.display_name, u.department_id, u.default_workspace_id,
                   s.session_id, s.csrf_token_hash, s.last_seen_at
            FROM user_sessions s JOIN users u ON u.user_id = s.user_id
            WHERE s.token_hash = %s AND s.revoked_at IS NULL
              AND s.expires_at > NOW() AND s.idle_expires_at > NOW()
              AND s.auth_version = u.auth_version AND u.status = 'active'
            """, (token_digest,),
        ).fetchone()
        if not row:
            return None
        now = datetime.now(timezone.utc)
        if row[7] is None or now - row[7] > timedelta(minutes=5):
            conn.execute(
                """
                UPDATE user_sessions
                SET last_seen_at = NOW(),
                    idle_expires_at = LEAST(expires_at, NOW() + (%s * INTERVAL '1 minute'))
                WHERE session_id = %s
                """, (settings.session_idle_minutes, row[5]),
            )
    return AuthenticatedPrincipal(
        user_id=str(row[0]), email=str(row[1]), display_name=str(row[2]), department_id=row[3],
        default_workspace_id=row[4], session_id=str(row[5]), csrf_token_hash=str(row[6]),
    )


def set_current_principal(principal: AuthenticatedPrincipal):
    return _principal_context.set(principal)


def reset_current_principal(token) -> None:
    _principal_context.reset(token)


def current_principal() -> AuthenticatedPrincipal:
    principal = _principal_context.get()
    if principal is None:
        raise AuthenticationError("需要登录")
    return principal


def session_is_active(principal: AuthenticatedPrincipal) -> bool:
    """长任务在每轮工具调用前复核会话，避免撤权后继续使用旧身份快照。"""
    with connection() as conn:
        row = conn.execute(
            """
            SELECT 1 FROM user_sessions s JOIN users u ON u.user_id = s.user_id
            WHERE s.session_id = %s AND s.user_id = %s AND s.revoked_at IS NULL
              AND s.expires_at > NOW() AND s.idle_expires_at > NOW()
              AND s.auth_version = u.auth_version AND u.status = 'active'
            """, (principal.session_id, principal.user_id),
        ).fetchone()
    return row is not None


def verify_csrf(principal: AuthenticatedPrincipal, header_token: str | None, cookie_token: str | None) -> bool:
    if not header_token or not cookie_token or not hmac.compare_digest(header_token, cookie_token):
        return False
    return hmac.compare_digest(_token_hash(header_token), principal.csrf_token_hash)


def revoke_session(session_id: str, user_id: str) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE user_sessions SET revoked_at = NOW() WHERE session_id = %s AND user_id = %s AND revoked_at IS NULL",
            (session_id, user_id),
        )
        _audit(conn, user_id, "auth.logout", "success", object_type="session", object_id=session_id)


def revoke_all_sessions(user_id: str) -> None:
    with connection() as conn:
        conn.execute("UPDATE user_sessions SET revoked_at = NOW() WHERE user_id = %s AND revoked_at IS NULL", (user_id,))
        _audit(conn, user_id, "auth.logout_all", "success", object_type="user", object_id=user_id)


def change_password(user_id: str, current_password: str, new_password: str, ip_address: str | None = None, user_agent: str | None = None) -> SessionBundle:
    if not 12 <= len(new_password) <= 128:
        raise ValueError("新密码长度必须为 12 到 128 个字符")
    with connection() as conn:
        row = conn.execute(
            "SELECT password_hash FROM user_credentials WHERE user_id = %s FOR UPDATE", (user_id,)
        ).fetchone()
        if not row or not password_hash.verify(current_password, row[0]):
            raise AuthenticationError("当前密码错误")
        conn.execute(
            "UPDATE user_credentials SET password_hash = %s, password_changed_at = NOW() WHERE user_id = %s",
            (password_hash.hash(new_password), user_id),
        )
        conn.execute("UPDATE users SET auth_version = auth_version + 1, updated_at = NOW() WHERE user_id = %s", (user_id,))
        conn.execute("UPDATE user_sessions SET revoked_at = NOW() WHERE user_id = %s AND revoked_at IS NULL", (user_id,))
        user_row = conn.execute(
            """
            SELECT user_id, email, normalized_email, display_name, department_id,
                   default_workspace_id, auth_version FROM users WHERE user_id = %s
            """, (user_id,),
        ).fetchone()
        bundle = _create_session(conn, user_row, ip_address, user_agent)
        _audit(conn, user_id, "auth.password_changed", "success", object_type="user", object_id=user_id)
        return bundle


def create_data_handover(source_user_id: str, target_email: str, current_password: str) -> dict:
    """重新验证发起人密码，并生成绑定指定接收账号的一次性交接码。"""
    target_normalized_email = normalize_email(target_email)
    if len(target_normalized_email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", target_normalized_email):
        raise HandoverError("请输入有效的接收员工邮箱")
    raw_token = secrets.token_urlsafe(32)
    handover_id = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    with connection() as conn:
        source = conn.execute(
            """
            SELECT u.normalized_email, u.status, c.password_hash
            FROM users u JOIN user_credentials c ON c.user_id = u.user_id
            WHERE u.user_id = %s FOR UPDATE
            """, (source_user_id,),
        ).fetchone()
        if not source or source[1] != "active" or not password_hash.verify(current_password, source[2]):
            raise AuthenticationError("当前密码错误")
        if source[0] == target_normalized_email:
            raise HandoverError("接收员工不能是当前账号")
        # 一个离职员工同一时间只有一个有效交接码；重发即撤销旧码。
        conn.execute(
            """
            UPDATE user_data_handovers SET revoked_at = NOW()
            WHERE source_user_id = %s AND accepted_at IS NULL AND revoked_at IS NULL
            """, (source_user_id,),
        )
        conn.execute(
            """
            INSERT INTO user_data_handovers
                (handover_id, source_user_id, target_normalized_email, token_hash, expires_at)
            VALUES (%s, %s, %s, %s, %s)
            """, (handover_id, source_user_id, target_normalized_email, _token_hash(raw_token), expires_at),
        )
        _audit(
            conn, source_user_id, "user_handover.created", "success",
            object_type="user_handover", object_id=handover_id,
            details=json.dumps({"target_email": target_normalized_email, "expires_at": expires_at.isoformat()}),
        )
    return {"handover_id": handover_id, "token": raw_token, "expires_at": expires_at}


def _handover_workspace_name(
    conn, workspace_id: str, source_name: str, workspace_type: str, target_user_id: str,
) -> tuple[str, str]:
    """为接收人解决同名空间冲突，并把个人空间改成可审计的继承团队空间。"""
    base_name = f"{source_name}（离职交接）" if workspace_type == "personal" else source_name
    candidate = base_name
    suffix = 1
    while True:
        key = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", candidate)).strip().casefold()
        conflict = conn.execute(
            """
            SELECT 1 FROM workspaces
            WHERE owner_user_id = %s AND normalized_name = %s AND workspace_id <> %s
            """, (target_user_id, key, workspace_id),
        ).fetchone()
        if not conflict:
            return candidate, key
        suffix += 1
        candidate = f"{base_name} {suffix}"


def accept_data_handover(target_user_id: str, raw_token: str) -> dict:
    """原子接收全部业务资源、成员权限与空间所有权，并立即停用原账号。"""
    if not raw_token or len(raw_token) > 512:
        raise HandoverError("接管码无效或已过期")
    with connection() as conn:
        handover = conn.execute(
            """
            SELECT h.handover_id, h.source_user_id, h.target_normalized_email,
                   u.normalized_email, u.status
            FROM user_data_handovers h
            JOIN users u ON u.user_id = %s
            WHERE h.token_hash = %s AND h.accepted_at IS NULL AND h.revoked_at IS NULL
              AND h.expires_at > NOW()
            FOR UPDATE OF h, u
            """, (target_user_id, _token_hash(raw_token)),
        ).fetchone()
        if not handover or handover[2] != handover[3] or handover[4] != "active":
            raise HandoverError("接管码无效、已过期或不属于当前账号")
        handover_id, source_user_id = str(handover[0]), str(handover[1])
        if source_user_id == target_user_id:
            raise HandoverError("不能接管当前账号自身的数据")
        source = conn.execute(
            "SELECT display_name, default_workspace_id, status FROM users WHERE user_id = %s FOR UPDATE",
            (source_user_id,),
        ).fetchone()
        if not source or source[2] != "active":
            raise HandoverError("原账号已不可用或数据已完成交接")

        owned = conn.execute(
            """
            SELECT workspace_id, workspace_name, workspace_type
            FROM workspaces WHERE owner_user_id = %s AND status = 'active'
            ORDER BY created_at, workspace_id FOR UPDATE
            """, (source_user_id,),
        ).fetchall()
        owned_ids = {str(row[0]) for row in owned}
        inherited_workspace_ids: list[str] = []
        for workspace_id_raw, workspace_name, workspace_type in owned:
            workspace_id = str(workspace_id_raw)
            new_name, new_key = _handover_workspace_name(
                conn, workspace_id, str(workspace_name), str(workspace_type), target_user_id,
            )
            # 先停用旧 owner 成员关系，避免同一空间出现两个活动 owner。
            conn.execute(
                "UPDATE workspace_members SET status = 'disabled' WHERE workspace_id = %s AND user_id = %s",
                (workspace_id, source_user_id),
            )
            conn.execute(
                """
                INSERT INTO workspace_members (workspace_id, user_id, role, status)
                VALUES (%s, %s, 'owner', 'active')
                ON CONFLICT (workspace_id, user_id) DO UPDATE
                SET role = 'owner', status = 'active'
                """, (workspace_id, target_user_id),
            )
            # 个人空间不能继续代表离职账号，转换为带来源痕迹的团队空间。
            conn.execute(
                """
                UPDATE workspaces SET owner_user_id = %s, workspace_type = 'team',
                    workspace_name = %s, normalized_name = %s, updated_at = NOW()
                WHERE workspace_id = %s
                """, (target_user_id, new_name, new_key, workspace_id),
            )
            inherited_workspace_ids.append(workspace_id)

        # 对原员工只是成员的空间，接收人继承其角色；已有更高权限时不降级。
        memberships = conn.execute(
            """
            SELECT workspace_id, role FROM workspace_members
            WHERE user_id = %s AND status = 'active'
            """, (source_user_id,),
        ).fetchall()
        role_rank = {"viewer": 1, "editor": 2, "admin": 3, "owner": 4}
        for workspace_id_raw, source_role in memberships:
            workspace_id = str(workspace_id_raw)
            if workspace_id in owned_ids:
                continue
            existing = conn.execute(
                "SELECT role FROM workspace_members WHERE workspace_id = %s AND user_id = %s",
                (workspace_id, target_user_id),
            ).fetchone()
            target_role = str(existing[0]) if existing else None
            inherited_role = target_role if target_role and role_rank[target_role] >= role_rank[str(source_role)] else str(source_role)
            conn.execute(
                """
                INSERT INTO workspace_members (workspace_id, user_id, role, status)
                VALUES (%s, %s, %s, 'active')
                ON CONFLICT (workspace_id, user_id) DO UPDATE
                SET role = EXCLUDED.role, status = 'active'
                """, (workspace_id, target_user_id, inherited_role),
            )
            conn.execute(
                "UPDATE workspace_members SET status = 'disabled' WHERE workspace_id = %s AND user_id = %s",
                (workspace_id, source_user_id),
            )

        document_count = conn.execute(
            "UPDATE documents SET owner_user_id = %s, updated_at = NOW() WHERE owner_user_id = %s RETURNING document_id",
            (target_user_id, source_user_id),
        ).fetchall()
        project_count = conn.execute(
            "UPDATE code_projects SET owner_user_id = %s, updated_at = NOW() WHERE owner_user_id = %s RETURNING project_id",
            (target_user_id, source_user_id),
        ).fetchall()
        conn.execute(
            """
            INSERT INTO code_project_access (project_id, user_id, permission, status)
            SELECT project_id, %s, permission, 'active'
            FROM code_project_access WHERE user_id = %s AND status = 'active'
            ON CONFLICT (project_id, user_id) DO UPDATE SET
              permission = CASE
                WHEN code_project_access.permission = 'admin' OR EXCLUDED.permission = 'admin' THEN 'admin'
                WHEN code_project_access.permission = 'write' OR EXCLUDED.permission = 'write' THEN 'write'
                ELSE 'read'
              END,
              status = 'active'
            """, (target_user_id, source_user_id),
        )
        conn.execute(
            "UPDATE code_project_access SET status = 'disabled' WHERE user_id = %s",
            (source_user_id,),
        )
        # 保留用户行和 created_by_user_id 以维持审计链；仅撤销登录与业务访问能力。
        conn.execute(
            """
            UPDATE users SET status = 'disabled', auth_version = auth_version + 1,
                default_workspace_id = NULL, updated_at = NOW()
            WHERE user_id = %s
            """, (source_user_id,),
        )
        conn.execute(
            "UPDATE user_sessions SET revoked_at = NOW() WHERE user_id = %s AND revoked_at IS NULL",
            (source_user_id,),
        )
        conn.execute(
            """
            UPDATE user_data_handovers
            SET accepted_by_user_id = %s, accepted_at = NOW()
            WHERE handover_id = %s
            """, (target_user_id, handover_id),
        )
        conn.execute(
            """
            UPDATE user_data_handovers SET revoked_at = NOW()
            WHERE source_user_id = %s AND handover_id <> %s
              AND accepted_at IS NULL AND revoked_at IS NULL
            """, (source_user_id, handover_id),
        )
        summary = {
            "source_user_id": source_user_id,
            "inherited_workspace_ids": inherited_workspace_ids,
            "workspace_count": len(inherited_workspace_ids),
            "document_count": len(document_count),
            "code_project_count": len(project_count),
        }
        _audit(
            conn, target_user_id, "user_handover.accepted", "success",
            object_type="user_handover", object_id=handover_id,
            details=json.dumps(summary, ensure_ascii=False),
        )
        return summary


class AuthenticationMiddleware:
    """在业务路由前完成 Session、Origin 和 CSRF 校验，并保持流式请求上下文。"""

    PUBLIC_PATHS = {
        "/", "/api/auth/register", "/api/auth/login", "/docs", "/openapi.json", "/redoc",
        "/api/health/live", "/api/health/ready", "/favicon.ico",
        "/.well-known/appspecific/com.chrome.devtools.json",
    }
    MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

    def __init__(self, app):
        self.app = app

    @staticmethod
    async def _reject(scope, receive, send, status_code: int, detail: str, error_code: str) -> None:
        """认证中间件直接返回稳定 JSON；保留 detail 兼容现有前端。"""
        await JSONResponse(
            error_payload(detail, status_code=status_code, error_code=error_code, retryable=False),
            status_code=status_code,
        )(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()
        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
        # 登录和注册没有现成 CSRF Cookie，因此至少要求浏览器 Origin 与当前 Host 一致。
        if path in {"/api/auth/register", "/api/auth/login"} and method == "POST":
            origin = headers.get("origin")
            host = headers.get("host", "")
            if origin and origin.rstrip("/") not in {f"http://{host}", f"https://{host}"}:
                await self._reject(scope, receive, send, 403, "请求来源不受信任", "ORIGIN_REJECTED")
                return
        if path in self.PUBLIC_PATHS or method == "OPTIONS" or not path.startswith("/api/"):
            await self.app(scope, receive, send)
            return

        cookies = {}
        for item in headers.get("cookie", "").split(";"):
            if "=" in item:
                key, value = item.strip().split("=", 1)
                cookies[key] = value
        principal = authenticate_session(cookies.get(settings.session_cookie_name))
        if principal is None:
            error_code = "SESSION_EXPIRED" if cookies.get(settings.session_cookie_name) else "AUTHENTICATION_REQUIRED"
            await self._reject(scope, receive, send, 401, "需要登录", error_code)
            return

        if method in self.MUTATING_METHODS:
            origin = headers.get("origin")
            host = headers.get("host", "")
            if origin and origin.rstrip("/") not in {f"http://{host}", f"https://{host}"}:
                await self._reject(scope, receive, send, 403, "请求来源不受信任", "ORIGIN_REJECTED")
                return
            if not verify_csrf(
                principal, headers.get("x-csrf-token"), cookies.get(settings.csrf_cookie_name),
            ):
                await self._reject(scope, receive, send, 403, "CSRF 校验失败", "CSRF_REJECTED")
                return

        context_token = set_current_principal(principal)
        scope.setdefault("state", {})["principal"] = principal
        try:
            await self.app(scope, receive, send)
        finally:
            reset_current_principal(context_token)
