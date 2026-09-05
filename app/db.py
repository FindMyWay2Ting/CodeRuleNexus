import json
import re
import uuid
import math
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import psycopg

from .config import settings


_transaction_connection: ContextVar[psycopg.Connection | None] = ContextVar("db_transaction", default=None)


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """提供可嵌套事务；内层仓储调用复用外层连接，不会提前提交。"""
    active = _transaction_connection.get()
    if active is not None:
        yield active
        return
    conn = psycopg.connect(settings.database_url)
    context_token = _transaction_connection.set(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _transaction_connection.reset(context_token)
        conn.close()


def initialize() -> None:
    """创建 MVP 所需表，并把旧版只有分块的数据迁移到 documents 主表。"""
    with connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # pg_trgm 为中文短语、英文术语和代码标识符提供词面相似度检索。
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                -- 每份导入知识的稳定唯一标识，供分块、引用和后续权限关联。
                document_id UUID PRIMARY KEY,
                -- 当前知识所属的工作空间；MVP 用于最小范围隔离。
                workspace_id TEXT NOT NULL,
                -- 知识来源类型：wiki 表示代码/项目知识，rag 表示文档知识。
                source_type TEXT NOT NULL CHECK (source_type IN ('wiki', 'rag')),
                -- 页面展示和用户检索时使用的文档标题。
                title TEXT NOT NULL,
                -- 原始文件名或项目名称，用于保留来源名称。
                source_name TEXT NOT NULL,
                -- 原始文件或项目在本机/仓库中的路径。
                source_path TEXT,
                -- Wiki 所属项目名称；RAG 文档为空。
                project_name TEXT,
                -- 工作空间显示名称；仅 workspace 归属需要，个人和部门知识为空。
                workspace_name TEXT,
                -- 业务分类，例如 code、development-standard、sop。
                category TEXT,
                -- 文档作者；当前可为空，后续可接入 GitLab 或文档系统作者。
                author TEXT,
                -- 文档业务版本，例如 v2.1；不是数据库 schema 版本。
                version TEXT,
                -- 原始内容 SHA-256，用于判断是否重复导入。
                content_hash TEXT NOT NULL,
                -- 文档状态：active 可检索，invalid 保留但不再参与检索。
                status TEXT NOT NULL DEFAULT 'active',
                -- 文档被标记失效的时间。
                invalidated_at TIMESTAMPTZ,
                -- 标记失效的原因，便于后续审计和恢复。
                invalid_reason TEXT,
                -- 当前文档的自动修订号，从 1 开始递增。
                current_revision INTEGER NOT NULL DEFAULT 0,
                -- 归属范围：personal 个人、department 部门、workspace 工作空间。
                scope_type TEXT NOT NULL DEFAULT 'workspace'
                    CHECK (scope_type IN ('personal', 'department', 'workspace')),
                -- 个人归属者；当前由服务端 CURRENT_USER_ID 提供。
                owner_user_id TEXT,
                -- 首次导入该文档的真实用户；共享文档也保留创建者。
                created_by_user_id TEXT,
                -- 部门归属标识；认证后应来自用户部门关系，而不是前端任意输入。
                owner_department_id TEXT,
                -- 首次创建文档记录的时间，也就是首次导入时间。
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                -- 文档元数据最后一次更新的时间。
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active'")
        conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS invalidated_at TIMESTAMPTZ")
        conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS invalid_reason TEXT")
        conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS current_revision INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS scope_type TEXT NOT NULL DEFAULT 'workspace'"
        )
        conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS owner_user_id TEXT")
        conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS created_by_user_id TEXT")
        conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS owner_department_id TEXT")
        conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS project_name TEXT")
        conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS workspace_name TEXT")
        # 旧数据没有工作空间名称，使用原有 workspace_id 补齐，保证管理页有可读归属。
        conn.execute("UPDATE documents SET workspace_name = workspace_id WHERE workspace_name IS NULL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_revisions (
                revision_id UUID PRIMARY KEY,
                document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                revision_no INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                author TEXT NOT NULL,
                change_type TEXT NOT NULL CHECK (change_type IN ('created', 'updated')),
                source_path TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (document_id, revision_no)
            )
            """
        )
        # 除了代码内注释，也写入 PostgreSQL 元数据，便于在 pgAdmin 中直接查看字段含义。
        conn.execute(
            """
            COMMENT ON TABLE documents IS '文档级知识主表：一条记录代表一份 Wiki 或 RAG 知识来源';
            COMMENT ON COLUMN documents.document_id IS '文档唯一 ID，分块和引用通过该字段关联';
            COMMENT ON COLUMN documents.workspace_id IS '知识所属工作空间，用于基础隔离';
            COMMENT ON COLUMN documents.source_type IS '知识类型：wiki 或 rag';
            COMMENT ON COLUMN documents.title IS '文档标题';
            COMMENT ON COLUMN documents.source_name IS '原始文件名或项目名称';
            COMMENT ON COLUMN documents.source_path IS '原始文件或项目路径';
            COMMENT ON COLUMN documents.project_name IS 'Wiki 所属项目名称；RAG 文档为空';
            COMMENT ON COLUMN documents.workspace_name IS '工作空间显示名称；workspace 归属使用';
            COMMENT ON COLUMN documents.category IS '业务分类';
            COMMENT ON COLUMN documents.author IS '文档作者';
            COMMENT ON COLUMN documents.version IS '业务文档版本';
            COMMENT ON COLUMN documents.content_hash IS '原始内容 SHA-256，用于幂等导入';
            COMMENT ON COLUMN documents.status IS '文档状态：active 或 invalid';
            COMMENT ON COLUMN documents.invalidated_at IS '文档标记失效的时间';
            COMMENT ON COLUMN documents.invalid_reason IS '文档失效原因';
            COMMENT ON COLUMN documents.current_revision IS '当前自动修订号，从 1 开始';
            COMMENT ON COLUMN documents.scope_type IS '知识归属范围：personal、department 或 workspace';
            COMMENT ON COLUMN documents.owner_user_id IS '个人知识归属者；由服务端认证身份写入';
            COMMENT ON COLUMN documents.created_by_user_id IS '首次导入文档的用户；用于共享知识治理和审计';
            COMMENT ON COLUMN documents.owner_department_id IS '部门归属标识；后续由认证用户部门关系提供';
            COMMENT ON COLUMN documents.created_at IS '文档首次导入时间';
            COMMENT ON COLUMN documents.updated_at IS '文档元数据最后更新时间';
            """
        )
        # 归属范围属于文档唯一性的一部分；同一内容可以分别归属于个人和部门。
        conn.execute("DROP INDEX IF EXISTS uq_documents_workspace_hash")
        index_row = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND indexname = 'uq_documents_workspace_hash_scope'"
        ).fetchone()
        if not index_row or "created_by_user_id" not in index_row[0]:
            conn.execute("DROP INDEX IF EXISTS uq_documents_workspace_hash_scope")
            conn.execute(
                "CREATE UNIQUE INDEX uq_documents_workspace_hash_scope "
                "ON documents (workspace_id, content_hash, scope_type, COALESCE(owner_user_id, ''), "
                "COALESCE(owner_department_id, ''), COALESCE(created_by_user_id, ''))"
            )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                -- 分块自身的数据库自增 ID，不作为对外引用 ID。
                id BIGSERIAL PRIMARY KEY,
                -- 对外稳定的分块 UUID；父子关系和后续引用都使用它，不依赖数据库自增序号。
                chunk_id UUID NOT NULL UNIQUE,
                -- 子块所属父块的 UUID；父块自身为空，历史扁平块也为空。
                parent_chunk_id UUID,
                -- 分块层级：parent 保存完整上下文，child 负责 Embedding 和召回。
                chunk_level TEXT NOT NULL DEFAULT 'child' CHECK (chunk_level IN ('parent', 'child')),
                -- 所属文档 ID，连接 documents 主表。
                document_id UUID,
                -- 从文档继承的工作空间，用于检索前过滤。
                workspace_id TEXT NOT NULL,
                -- 从文档继承的 Wiki/RAG 类型，用于路由过滤。
                source_type TEXT NOT NULL CHECK (source_type IN ('wiki', 'rag')),
                -- 从文档继承的来源名称，便于快速展示和兼容旧数据。
                source_name TEXT NOT NULL,
                -- 从文档继承的来源路径。
                source_path TEXT,
                -- 当前分块的可读引用，例如 file.md#chunk-2。
                source_ref TEXT,
                -- 分块在原文中的顺序，从 0 开始。
                chunk_index INTEGER NOT NULL,
                -- 子块在同一父块中的顺序；父块为空。
                child_index INTEGER,
                -- 内容元素类型，例如 section、page_text、table、code 或 document。
                element_type TEXT NOT NULL DEFAULT 'document',
                -- Markdown 标题层级路径；非 Markdown 内容可为空。
                section_path TEXT,
                -- PDF 原始页码；非 PDF 内容可为空。
                page_number INTEGER,
                -- 文档或代码语言，例如 markdown、python、java、pdf。
                language TEXT,
                -- 代码类、函数或方法符号；非代码内容可为空。
                code_symbol TEXT,
                -- 当前块的字符数，用于上下文预算、诊断和评测。
                char_count INTEGER NOT NULL DEFAULT 0,
                -- 解析器版本、表格编号等非固定扩展字段。
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                -- 分块原文，作为 LLM 的检索证据。
                content TEXT NOT NULL,
                -- 子块向量，用于 pgvector 语义检索；父块不生成向量，因此允许为空。
                embedding vector({settings.embedding_dimensions}),
                -- 分块写入数据库的时间。
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS document_id UUID")
        # 兼容旧表：先增加可空列并回填 UUID，再收紧 chunk_id 约束。旧分块保留为无父块的 child。
        conn.execute("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS chunk_id UUID")
        conn.execute("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS parent_chunk_id UUID")
        conn.execute("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS chunk_level TEXT NOT NULL DEFAULT 'child'")
        conn.execute("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS child_index INTEGER")
        conn.execute("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS element_type TEXT NOT NULL DEFAULT 'document'")
        conn.execute("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS section_path TEXT")
        conn.execute("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS page_number INTEGER")
        conn.execute("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS language TEXT")
        conn.execute("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS code_symbol TEXT")
        conn.execute("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS char_count INTEGER NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb")
        conn.execute("ALTER TABLE knowledge_chunks ALTER COLUMN embedding DROP NOT NULL")
        missing_chunk_ids = conn.execute("SELECT id FROM knowledge_chunks WHERE chunk_id IS NULL").fetchall()
        for (legacy_chunk_row_id,) in missing_chunk_ids:
            conn.execute(
                "UPDATE knowledge_chunks SET chunk_id = %s, char_count = length(content) WHERE id = %s",
                (uuid.uuid4(), legacy_chunk_row_id),
            )
        conn.execute("ALTER TABLE knowledge_chunks ALTER COLUMN chunk_id SET NOT NULL")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_chunks_chunk_id ON knowledge_chunks (chunk_id)")
        conn.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'ck_knowledge_chunks_level'
                ) THEN
                    ALTER TABLE knowledge_chunks
                    ADD CONSTRAINT ck_knowledge_chunks_level
                    CHECK (chunk_level IN ('parent', 'child'));
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'fk_knowledge_chunks_parent'
                ) THEN
                    ALTER TABLE knowledge_chunks
                    ADD CONSTRAINT fk_knowledge_chunks_parent
                    FOREIGN KEY (parent_chunk_id) REFERENCES knowledge_chunks(chunk_id) ON DELETE CASCADE;
                END IF;
            END $$
            """
        )
        conn.execute(
            """
            COMMENT ON TABLE knowledge_chunks IS '文档分块表：保存检索文本和对应向量';
            COMMENT ON COLUMN knowledge_chunks.id IS '分块数据库自增 ID';
            COMMENT ON COLUMN knowledge_chunks.chunk_id IS '稳定分块 UUID，父子关系和对外引用使用';
            COMMENT ON COLUMN knowledge_chunks.parent_chunk_id IS '子块所属父块 UUID；父块和历史扁平块为空';
            COMMENT ON COLUMN knowledge_chunks.chunk_level IS '分块层级：parent 完整上下文，child 检索单元';
            COMMENT ON COLUMN knowledge_chunks.document_id IS '所属文档 ID，关联 documents.document_id';
            COMMENT ON COLUMN knowledge_chunks.workspace_id IS '分块所属工作空间，用于权限过滤';
            COMMENT ON COLUMN knowledge_chunks.source_type IS '分块来源类型：wiki 或 rag';
            COMMENT ON COLUMN knowledge_chunks.source_name IS '分块来源名称';
            COMMENT ON COLUMN knowledge_chunks.source_path IS '分块来源路径';
            COMMENT ON COLUMN knowledge_chunks.source_ref IS '用于回答引用的原文定位标识';
            COMMENT ON COLUMN knowledge_chunks.chunk_index IS '分块在文档中的顺序号';
            COMMENT ON COLUMN knowledge_chunks.child_index IS '子块在同一父块内的顺序，从 0 开始';
            COMMENT ON COLUMN knowledge_chunks.element_type IS '结构元素类型：section、page_text、table、code 或 document';
            COMMENT ON COLUMN knowledge_chunks.section_path IS 'Markdown 标题层级路径';
            COMMENT ON COLUMN knowledge_chunks.page_number IS 'PDF 原始页码';
            COMMENT ON COLUMN knowledge_chunks.language IS '文档或代码语言';
            COMMENT ON COLUMN knowledge_chunks.code_symbol IS '代码类、函数或方法符号';
            COMMENT ON COLUMN knowledge_chunks.char_count IS '分块字符数，用于预算和诊断';
            COMMENT ON COLUMN knowledge_chunks.metadata IS '解析器版本、表格编号等可扩展结构化元数据';
            COMMENT ON COLUMN knowledge_chunks.content IS '分块文本内容';
            COMMENT ON COLUMN knowledge_chunks.embedding IS 'Embedding 模型生成的向量；父块为空，仅子块参与召回';
            COMMENT ON COLUMN knowledge_chunks.created_at IS '分块入库时间';
            """
        )
        # 早期版本没有 document_id。按来源分组创建兼容文档，再回填分块关联。
        legacy_rows = conn.execute(
            """
            SELECT DISTINCT source_type, source_name, source_path
            FROM knowledge_chunks
            WHERE document_id IS NULL
            """
        ).fetchall()
        for source_type, source_name, source_path in legacy_rows:
            legacy_hash = f"legacy:{settings.workspace_id}:{source_type}:{source_path or ''}:{source_name}"
            inserted_document = conn.execute(
                """
                INSERT INTO documents
                (document_id, workspace_id, source_type, title, source_name, source_path,
                 workspace_name, category, version, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING document_id
                """,
                (
                    uuid.uuid4(), settings.workspace_id, source_type, source_name,
                    source_name, source_path, settings.workspace_id, source_type, "legacy", legacy_hash,
                ),
            ).fetchone()
            # psycopg 返回单列结果时仍是 tuple；统一取出 UUID，避免回填分块时把 tuple 当成 ID。
            if inserted_document is not None:
                document_id = inserted_document[0]
            else:
                document_id = conn.execute(
                    """
                    SELECT document_id FROM documents
                    WHERE workspace_id = %s AND content_hash = %s
                    """,
                    (settings.workspace_id, legacy_hash),
                ).fetchone()[0]
            conn.execute(
                """
                UPDATE knowledge_chunks
                SET document_id = %s
                WHERE workspace_id = %s AND source_type = %s
                  AND source_name = %s AND source_path IS NOT DISTINCT FROM %s
                  AND document_id IS NULL
                """,
                (document_id, settings.workspace_id, source_type, source_name, source_path),
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_workspace_type "
            "ON knowledge_chunks (workspace_id, source_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document_id "
            "ON knowledge_chunks (document_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_parent_id "
            "ON knowledge_chunks (parent_chunk_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_retrieval_level "
            "ON knowledge_chunks (workspace_id, source_type, chunk_level)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_document_revisions_document_id ON document_revisions (document_id)")
        # 给 document_id 上线前已有的文档补一条初始修订记录，保持历史链路完整。
        legacy_documents = conn.execute(
            """
            SELECT document_id, content_hash, source_path, COALESCE(author, 'User')
            FROM documents
            WHERE current_revision = 0
            """
        ).fetchall()
        for legacy_document_id, content_hash, source_path, author in legacy_documents:
            conn.execute(
                """
                UPDATE documents SET current_revision = 1, author = %s WHERE document_id = %s
                """,
                (author, legacy_document_id),
            )
            conn.execute(
                """
                INSERT INTO document_revisions
                (revision_id, document_id, revision_no, content_hash, author, change_type, source_path)
                VALUES (%s, %s, 1, %s, %s, 'created', %s)
                ON CONFLICT (document_id, revision_no) DO NOTHING
                """,
                (uuid.uuid4(), legacy_document_id, content_hash, author, source_path),
            )


def ensure_document(metadata: dict) -> tuple[str, bool, int]:
    """登记文档修订：相同内容跳过，来源内容变化则递增修订号并清理旧分块。"""
    document_id = uuid.uuid4()
    workspace_id = metadata.get("workspace_id") or settings.workspace_id
    with connection() as conn:
        existing_source = conn.execute(
            """
            SELECT document_id, content_hash, current_revision
            FROM documents
            WHERE workspace_id = %s AND source_type = %s AND source_path = %s
              AND scope_type = %s
              AND owner_user_id IS NOT DISTINCT FROM %s
              AND owner_department_id IS NOT DISTINCT FROM %s
              AND created_by_user_id IS NOT DISTINCT FROM %s
            """,
            (
                workspace_id, metadata["source_type"], metadata.get("source_path"),
                metadata["scope_type"], metadata.get("owner_user_id"), metadata.get("owner_department_id"),
                metadata.get("created_by_user_id"),
            ),
        ).fetchone()
        if existing_source:
            existing_id, old_hash, current_revision = existing_source
            if old_hash == metadata["content_hash"]:
                return str(existing_id), False, current_revision
            next_revision = current_revision + 1
            conn.execute("DELETE FROM knowledge_chunks WHERE document_id = %s", (existing_id,))
            conn.execute(
                """
                UPDATE documents
                SET title = %s, author = %s, content_hash = %s, current_revision = %s,
                    project_name = %s, workspace_name = %s,
                    scope_type = %s, owner_user_id = %s, owner_department_id = %s,
                    created_by_user_id = COALESCE(created_by_user_id, %s),
                    status = 'active', invalidated_at = NULL, invalid_reason = NULL, updated_at = NOW()
                WHERE document_id = %s
                """,
                (
                    metadata["title"], metadata["author"], metadata["content_hash"], next_revision,
                    metadata.get("project_name"), metadata.get("workspace_name"),
                    metadata["scope_type"], metadata.get("owner_user_id"), metadata.get("owner_department_id"),
                    metadata.get("created_by_user_id"),
                    existing_id,
                ),
            )
            conn.execute(
                """
                INSERT INTO document_revisions
                (revision_id, document_id, revision_no, content_hash, author, change_type, source_path)
                VALUES (%s, %s, %s, %s, %s, 'updated', %s)
                """,
                (uuid.uuid4(), existing_id, next_revision, metadata["content_hash"], metadata["author"], metadata.get("source_path")),
            )
            return str(existing_id), True, next_revision

        existing_hash = conn.execute(
            """
            SELECT document_id, current_revision
            FROM documents
            WHERE workspace_id = %s AND content_hash = %s
              AND scope_type = %s
              AND owner_user_id IS NOT DISTINCT FROM %s
              AND owner_department_id IS NOT DISTINCT FROM %s
              AND created_by_user_id IS NOT DISTINCT FROM %s
            """,
            (
                workspace_id, metadata["content_hash"], metadata["scope_type"],
                metadata.get("owner_user_id"), metadata.get("owner_department_id"),
                metadata.get("created_by_user_id"),
            ),
        ).fetchone()
        if existing_hash:
            return str(existing_hash[0]), False, existing_hash[1]

        result = conn.execute(
            """
            INSERT INTO documents
            (document_id, workspace_id, source_type, title, source_name, source_path,
             project_name, workspace_name, category, author, version, content_hash, current_revision,
             scope_type, owner_user_id, owner_department_id, created_by_user_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, %s, NULL, %s, 1, %s, %s, %s, %s)
            RETURNING document_id
            """,
            (
                document_id, workspace_id, metadata["source_type"], metadata["title"],
                metadata["source_name"], metadata.get("source_path"), metadata.get("project_name"),
                metadata.get("workspace_name"), metadata["author"], metadata["content_hash"],
                metadata["scope_type"], metadata.get("owner_user_id"), metadata.get("owner_department_id"),
                metadata.get("created_by_user_id"),
            ),
        )
        new_id = result.fetchone()[0]
        conn.execute(
            """
            INSERT INTO document_revisions
            (revision_id, document_id, revision_no, content_hash, author, change_type, source_path)
            VALUES (%s, %s, 1, %s, %s, 'created', %s)
            """,
            (uuid.uuid4(), new_id, metadata["content_hash"], metadata["author"], metadata.get("source_path")),
        )
        return str(new_id), True, 1


def find_reusable_document(metadata: dict) -> str | None:
    """在调用 Embedding 前检查幂等数据，避免重复导入时再次请求外部模型。"""
    workspace_id = metadata.get("workspace_id") or settings.workspace_id
    with connection() as conn:
        existing_source = conn.execute(
            """
            SELECT document_id
            FROM documents
            WHERE workspace_id = %s
              AND source_type = %s
              AND source_path = %s
              AND content_hash = %s
              AND scope_type = %s
              AND owner_user_id IS NOT DISTINCT FROM %s
              AND owner_department_id IS NOT DISTINCT FROM %s
              AND created_by_user_id IS NOT DISTINCT FROM %s
            """,
            (
                workspace_id,
                metadata["source_type"],
                metadata.get("source_path"),
                metadata["content_hash"],
                metadata["scope_type"],
                metadata.get("owner_user_id"),
                metadata.get("owner_department_id"),
                metadata.get("created_by_user_id"),
            ),
        ).fetchone()
        if existing_source:
            return str(existing_source[0])

        # 同一归属范围内完全相同的内容可以直接复用；不同归属不能互相泄漏。
        existing_hash = conn.execute(
            """
            SELECT document_id
            FROM documents
            WHERE workspace_id = %s AND content_hash = %s
              AND scope_type = %s
              AND owner_user_id IS NOT DISTINCT FROM %s
              AND owner_department_id IS NOT DISTINCT FROM %s
              AND created_by_user_id IS NOT DISTINCT FROM %s
            """,
            (
                workspace_id, metadata["content_hash"], metadata["scope_type"],
                metadata.get("owner_user_id"), metadata.get("owner_department_id"),
                metadata.get("created_by_user_id"),
            ),
        ).fetchone()
        return str(existing_hash[0]) if existing_hash else None


def insert_chunks(rows: list[dict], document_id: str) -> int:
    """把父块和子块写入 knowledge_chunks；父块先写入，子块再通过 UUID 建立关联。"""
    if not rows:
        return 0
    with connection() as conn:
        # 父块没有 embedding，先插入父块并建立 source_ref 到 chunk_id 的映射。
        parent_ids: dict[str, str] = {}
        for row in rows:
            if row.get("chunk_level") != "parent":
                continue
            chunk_id = row["chunk_id"]
            parent_ids[row["parent_key"]] = chunk_id
            conn.execute(
                """
                INSERT INTO knowledge_chunks
                (chunk_id, parent_chunk_id, chunk_level, document_id, workspace_id, source_type,
                 source_name, source_path, source_ref, chunk_index, child_index, element_type,
                 section_path, page_number, language, code_symbol, char_count, metadata, content, embedding)
                VALUES (%s, NULL, 'parent', %s, %s, %s, %s, %s, %s, %s, NULL, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, NULL)
                """,
                (
                    chunk_id, document_id, row.get("workspace_id", settings.workspace_id), row["source_type"], row["source_name"],
                    row.get("source_path"), row.get("source_ref"), row["chunk_index"], row.get("element_type", "document"),
                    row.get("section_path"), row.get("page_number"), row.get("language"), row.get("code_symbol"),
                    len(row["content"]), json.dumps(row.get("metadata", {}), ensure_ascii=False), row["content"],
                ),
            )
        for row in rows:
            if row.get("chunk_level") == "parent":
                continue
            parent_id = parent_ids.get(row.get("parent_key"))
            conn.execute(
                """
                INSERT INTO knowledge_chunks
                (chunk_id, parent_chunk_id, chunk_level, document_id, workspace_id, source_type,
                 source_name, source_path, source_ref, chunk_index, child_index, element_type,
                 section_path, page_number, language, code_symbol, char_count, metadata, content, embedding)
                VALUES (%s, %s, 'child', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::vector)
                """,
                (
                    row["chunk_id"], parent_id, document_id,
                    row.get("workspace_id", settings.workspace_id),
                    row["source_type"],
                    row["source_name"],
                    row.get("source_path"),
                    row.get("source_ref"),
                    row["chunk_index"],
                    row.get("child_index"),
                    row.get("element_type", "document"),
                    row.get("section_path"), row.get("page_number"), row.get("language"), row.get("code_symbol"),
                    len(row["content"]), json.dumps(row.get("metadata", {}), ensure_ascii=False),
                    row["content"],
                    json.dumps(row["embedding"]),
                ),
            )
    # 对外返回可检索子块数量；父块是上下文容器，不应让用户误以为也生成了向量。
    return sum(1 for row in rows if row.get("chunk_level") != "parent")


def _visibility_clause() -> str:
    """统一生成文档归属过滤；个人、部门和 workspace 三类范围都必须经过这里。"""
    return """
      AND (
        d.scope_type = 'workspace'
        OR (d.scope_type = 'personal' AND d.owner_user_id = %s)
        OR (d.scope_type = 'department' AND d.owner_department_id = %s)
      )
    """


def search_chunks(
    query_embedding: list[float],
    source_types: list[str],
    limit: int,
    user_id: str | None = None,
    department_id: str | None = None,
    workspace_id: str | None = None,
) -> list[dict]:
    """在 workspace、知识类型和当前用户可见归属范围内做余弦检索。"""
    workspace_id = workspace_id or settings.workspace_id
    with connection() as conn:
        result = conn.execute(
            """
            SELECT c.id, c.chunk_id, c.parent_chunk_id, c.document_id, c.source_type,
                   c.source_name, c.source_path, c.source_ref, c.chunk_index, c.child_index,
                   c.element_type, c.section_path, c.page_number, c.language, c.code_symbol,
                   c.content, 1 - (embedding <=> %s::vector) AS score
            FROM knowledge_chunks c
            JOIN documents d ON d.document_id = c.document_id AND d.workspace_id = c.workspace_id
            WHERE c.workspace_id = %s AND c.source_type = ANY(%s)
              AND c.chunk_level = 'child' AND c.embedding IS NOT NULL
              AND d.status = 'active'
            """
            + _visibility_clause()
            + """
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (
                json.dumps(query_embedding), workspace_id, source_types,
                user_id, department_id, json.dumps(query_embedding), limit,
            ),
        )
        columns = [desc.name for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]


def search_keyword_chunks(
    query: str,
    source_types: list[str],
    limit: int,
    user_id: str | None = None,
    department_id: str | None = None,
    workspace_id: str | None = None,
) -> list[dict]:
    """使用查询词片段做 pg_trgm 词面召回，避免长中文问题稀释关键词相似度。"""
    terms = _keyword_terms(query)
    if not terms:
        return []
    workspace_id = workspace_id or settings.workspace_id
    with connection() as conn:
        result = conn.execute(
            """
            SELECT c.id, c.chunk_id, c.parent_chunk_id, c.document_id, c.source_type,
                   c.source_name, c.source_path, c.source_ref, c.chunk_index, c.child_index,
                   c.element_type, c.section_path, c.page_number, c.language, c.code_symbol,
                   c.content, MAX(similarity(c.content, term.value)) AS keyword_score
            FROM knowledge_chunks c
            JOIN documents d ON d.document_id = c.document_id AND d.workspace_id = c.workspace_id
            CROSS JOIN LATERAL unnest(%s::text[]) AS term(value)
            WHERE c.workspace_id = %s AND c.source_type = ANY(%s)
              AND c.chunk_level = 'child' AND c.embedding IS NOT NULL
              AND d.status = 'active'
              AND c.content %% term.value
            """
            + _visibility_clause()
            + """
            GROUP BY c.id, c.chunk_id, c.parent_chunk_id, c.document_id, c.source_type,
                     c.source_name, c.source_path, c.source_ref, c.chunk_index, c.child_index,
                     c.element_type, c.section_path, c.page_number, c.language, c.code_symbol, c.content
            ORDER BY MAX(similarity(c.content, term.value)) DESC
            LIMIT %s
            """,
            (terms, workspace_id, source_types, user_id, department_id, limit),
        )
        columns = [desc.name for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]


def _keyword_terms(query: str) -> list[str]:
    """生成适合 pg_trgm 的中英文查询片段；短于两个字符的片段没有稳定区分度。"""
    terms: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", query):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            terms.add(token)
            for width in (2, 3, 4):
                terms.update(token[index:index + width] for index in range(len(token) - width + 1))
        elif len(token) >= 2:
            terms.add(token.lower())
    return sorted(terms, key=len, reverse=True)[:80]


def hybrid_search(
    query_embedding: list[float],
    query: str,
    source_types: list[str],
    limit: int,
    user_id: str | None = None,
    department_id: str | None = None,
    workspace_id: str | None = None,
) -> list[dict]:
    """融合向量和词面两路召回，并在两路召回前统一应用归属过滤。"""
    vector_results = search_chunks(query_embedding, source_types, limit, user_id, department_id, workspace_id)
    keyword_results = search_keyword_chunks(query, source_types, limit, user_id, department_id, workspace_id)

    # Reciprocal Rank Fusion 只使用名次，不直接混合余弦分数和 trigram 分数。
    rrf_k = 60
    fused: dict[int, dict] = {}
    for rank, item in enumerate(vector_results, start=1):
        current = fused.setdefault(item["id"], dict(item))
        current["vector_score"] = float(item["score"])
        current["fused_score"] = current.get("fused_score", 0.0) + 1 / (rrf_k + rank)
    for rank, item in enumerate(keyword_results, start=1):
        current = fused.setdefault(item["id"], dict(item))
        current["keyword_score"] = float(item["keyword_score"])
        current["fused_score"] = current.get("fused_score", 0.0) + 1 / (rrf_k + rank)

    ranked = sorted(fused.values(), key=lambda item: item["fused_score"], reverse=True)
    # 先保证文档多样性，再交给 Reranker；同一文档最多占 3 个候选位置，避免一个长文档挤掉其他来源。
    diversified: list[dict] = []
    per_document_count: dict[str, int] = {}
    for item in ranked:
        document_key = str(item["document_id"])
        if per_document_count.get(document_key, 0) >= 3:
            continue
        diversified.append(item)
        per_document_count[document_key] = per_document_count.get(document_key, 0) + 1
        if len(diversified) == limit:
            break
    return diversified


def _estimate_tokens(text: str) -> int:
    """用保守字符比例估算 Token，避免为预算控制强绑定某一家模型 tokenizer。"""
    return max(1, math.ceil(len(text.strip()) / 2))


def expand_context(
    results: list[dict],
    max_tokens: int,
    workspace_id: str | None = None,
    user_id: str | None = None,
    department_id: str | None = None,
) -> list[dict]:
    """将命中的子块扩展为父块上下文，并去重、按预算保留高相关证据。"""
    if not results or max_tokens <= 0:
        return []
    workspace_id = workspace_id or settings.workspace_id
    user_id = user_id or settings.current_user_id
    department_id = department_id or settings.current_department_id
    parent_ids = [str(item["parent_chunk_id"]) for item in results if item.get("parent_chunk_id")]
    parents: dict[str, dict] = {}
    if parent_ids:
        with connection() as conn:
            parent_result = conn.execute(
                """
                SELECT c.chunk_id, c.document_id, c.source_type, c.source_name, c.source_path,
                       c.source_ref, c.chunk_index, c.element_type, c.section_path,
                       c.page_number, c.language, c.code_symbol, c.content
                FROM knowledge_chunks c
                JOIN documents d ON d.document_id = c.document_id AND d.workspace_id = c.workspace_id
                WHERE c.chunk_id = ANY(%s::uuid[])
                  AND c.workspace_id = %s
                  AND c.chunk_level = 'parent' AND d.status = 'active'
                """
                + _visibility_clause(),
                (parent_ids, workspace_id, user_id, department_id),
            )
            columns = [desc.name for desc in parent_result.description]
            parents = {str(row[0]): dict(zip(columns, row)) for row in parent_result.fetchall()}

    selected: list[dict] = []
    seen_contexts: set[str] = set()
    used_tokens = 0
    for result in results:
        parent_id = str(result.get("parent_chunk_id")) if result.get("parent_chunk_id") else None
        context_key = f"parent:{parent_id}" if parent_id and parent_id in parents else f"child:{result.get('chunk_id', result.get('id'))}"
        if context_key in seen_contexts:
            continue
        seen_contexts.add(context_key)

        context = dict(result)
        if parent_id and parent_id in parents:
            parent = parents[parent_id]
            context.update(parent)
            # 保留子块分数和引用，记录命中的子块，便于解释“为什么扩展了这个父块”。
            context["matched_ref"] = result.get("source_ref")
            context["source_ref"] = parent.get("source_ref") or result.get("source_ref")
            context["context_level"] = "parent"
        else:
            context["context_level"] = "child"

        estimated_tokens = _estimate_tokens(context.get("content", ""))
        remaining = max_tokens - used_tokens
        if remaining <= 0:
            break
        if estimated_tokens > remaining:
            # 父块放不下时优先保留命中的子块；仅在单个子块也过长时才截断。
            child_text = result.get("content", "")
            child_tokens = _estimate_tokens(child_text)
            if child_tokens <= remaining:
                context = dict(result)
                context["context_level"] = "child_fallback"
                context["matched_ref"] = result.get("source_ref")
                estimated_tokens = child_tokens
            else:
                max_chars = max(2, remaining * 2)
                context["content"] = child_text[:max_chars]
                context["context_level"] = "child_truncated"
                estimated_tokens = _estimate_tokens(context["content"])
        context["context_tokens"] = estimated_tokens
        selected.append(context)
        used_tokens += estimated_tokens
    return selected


def list_knowledge(
    source_type: str | None = None,
    scope_type: str | None = None,
    user_id: str | None = None,
    department_id: str | None = None,
    workspace_id: str | None = None,
) -> list[dict]:
    """按文档聚合列表，供知识管理页面展示导入时间和分块数量。"""
    workspace_id = workspace_id or settings.workspace_id
    conditions = ["workspace_id = %s"]
    params: list = [workspace_id]
    if source_type is not None:
        conditions.append("source_type = %s")
        params.append(source_type)
    if scope_type is not None:
        conditions.append("scope_type = %s")
        params.append(scope_type)

    with connection() as conn:
        result = conn.execute(
            f"""
            SELECT document_id, source_type, title, source_name, source_path,
                   author, content_hash, current_revision,
                   project_name, workspace_name, scope_type, owner_user_id, owner_department_id,
                   status, invalidated_at, invalid_reason,
                   created_at AS imported_at, updated_at,
                   (SELECT COUNT(*) FROM knowledge_chunks c WHERE c.document_id = d.document_id AND c.chunk_level = 'child') AS chunk_count
            FROM documents d
            WHERE {' AND '.join(conditions)}
              {_visibility_clause()}
            ORDER BY imported_at DESC, title ASC
            """,
            params + [user_id, department_id],
        )
        columns = [desc.name for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]


def list_document_revisions(document_id: str, workspace_id: str | None = None) -> list[dict]:
    """按文档返回修订历史，供知识管理页面展示版本和更新记录。"""
    workspace_id = workspace_id or settings.workspace_id
    with connection() as conn:
        result = conn.execute(
            """
            SELECT revision_id, document_id, revision_no, content_hash,
                   author, change_type, source_path, created_at
            FROM document_revisions
            WHERE document_id = %s
              AND document_id IN (
                  SELECT document_id
                  FROM documents
                  WHERE document_id = %s AND workspace_id = %s
              )
            ORDER BY revision_no DESC
            """,
            (document_id, document_id, workspace_id),
        )
        columns = [desc.name for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]


def invalidate_document(document_id: str, reason: str | None = None, workspace_id: str | None = None) -> bool:
    """保留文档和分块，但将其标记为 invalid，使检索自动忽略。"""
    workspace_id = workspace_id or settings.workspace_id
    with connection() as conn:
        result = conn.execute(
            """
            UPDATE documents
            SET status = 'invalid', invalidated_at = NOW(), invalid_reason = %s, updated_at = NOW()
            WHERE document_id = %s AND workspace_id = %s AND status = 'active'
            RETURNING document_id
            """,
            (reason, document_id, workspace_id),
        )
        return result.fetchone() is not None


def restore_document(document_id: str, workspace_id: str | None = None) -> bool:
    """恢复失效文档；关联分块重新允许进入向量检索。"""
    workspace_id = workspace_id or settings.workspace_id
    with connection() as conn:
        result = conn.execute(
            """
            UPDATE documents
            SET status = 'active', invalidated_at = NULL, invalid_reason = NULL, updated_at = NOW()
            WHERE document_id = %s AND workspace_id = %s AND status = 'invalid'
            RETURNING document_id
            """,
            (document_id, workspace_id),
        )
        return result.fetchone() is not None


def delete_document(document_id: str, workspace_id: str | None = None) -> bool:
    """永久删除文档及其分块；调用方必须在用户界面进行二次确认。"""
    workspace_id = workspace_id or settings.workspace_id
    with connection() as conn:
        exists = conn.execute(
            "SELECT document_id FROM documents WHERE document_id = %s AND workspace_id = %s",
            (document_id, workspace_id),
        ).fetchone()
        if not exists:
            return False
        conn.execute("DELETE FROM knowledge_chunks WHERE document_id = %s", (document_id,))
        conn.execute("DELETE FROM documents WHERE document_id = %s AND workspace_id = %s", (document_id, workspace_id))
        return True
