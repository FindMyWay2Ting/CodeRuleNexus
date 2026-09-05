"""应用启动所需的数据库初始化入口。"""

from .access_control import initialize_access_control
from .auth import initialize_auth
from .code_wiki import initialize_code_wiki
from .db import initialize
from .db import connection


REQUIRED_SCHEMA_VERSION = "2026-09-code-snapshots-v2"


def initialize_application() -> None:
    """按外键依赖顺序创建基础表、代码事实表、空间表和认证约束。"""
    initialize()
    initialize_code_wiki()
    initialize_access_control()
    initialize_auth()


def verify_application_schema() -> None:
    """启动进程只验证迁移结果；DDL 由显式初始化命令执行一次。"""
    with connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE version = %s",
            (REQUIRED_SCHEMA_VERSION,),
        ).fetchone()
    if not row:
        raise RuntimeError("database schema is not initialized; run python -m scripts.init_database")
