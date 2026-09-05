import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """应用配置集中读取自项目根目录的 .env 文件。"""

    database_url: str = os.getenv("DATABASE_URL", "postgresql://knowledge:knowledge_dev_password@localhost:5432/knowledge")
    llm_api_base: str = os.getenv("LLM_API_BASE", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    chat_model: str = os.getenv("CHAT_MODEL", "")
    embedding_api_base: str = os.getenv("EMBEDDING_API_BASE", "")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    reranker_api_url: str = os.getenv("RERANKER_API_URL", "")
    reranker_api_key: str = os.getenv("RERANKER_API_KEY", "")
    reranker_model: str = os.getenv("RERANKER_MODEL", "qwen3-rerank")
    reranker_candidate_limit: int = int(os.getenv("RERANKER_CANDIDATE_LIMIT", "20"))
    # Agent 自主决定何时结束；轮次和工具数只是防止异常循环的运行时安全预算。
    # 面试演示需要允许 Agent 充分调查；这里保留较高的异常熔断上限，
    # 正常结束仍由 Agent 的 finish_investigation、重复调用和无新证据规则决定。
    code_agent_max_rounds: int = max(1, int(os.getenv("CODE_AGENT_MAX_ROUNDS", "50")))
    code_agent_max_tool_calls: int = max(1, int(os.getenv("CODE_AGENT_MAX_TOOL_CALLS", "100")))
    knowledge_agent_max_rounds: int = max(1, int(os.getenv("KNOWLEDGE_AGENT_MAX_ROUNDS", "50")))
    knowledge_agent_max_tool_calls: int = max(1, int(os.getenv("KNOWLEDGE_AGENT_MAX_TOOL_CALLS", "100")))
    # Python SCIP 在 Windows 上通过 Linux 容器运行，避免官方 CLI 的路径正则兼容问题。
    scip_python_image: str = os.getenv("SCIP_PYTHON_IMAGE", "sourcegraph/scip-python:v0.6.6")
    scip_python_timeout: int = int(os.getenv("SCIP_PYTHON_TIMEOUT", "300"))
    # 代码快照属于持久化数据，不应隐式绑定某一次代码 checkout。
    # 相对路径以项目根目录解析；滚动发布或干净重建时应配置独立的绝对路径。
    code_repository_root: str = os.getenv("CODE_REPOSITORY_ROOT", "./data/code_repositories")
    workspace_id: str = os.getenv("WORKSPACE_ID", "workspace-001")
    # 当前工作空间显示名称；MVP 阶段默认沿用工作空间 ID，后续可接入工作空间表。
    current_workspace_name: str = os.getenv("CURRENT_WORKSPACE_NAME", "workspace-001")
    # 仅用于旧数据库迁移占位身份；HTTP 请求身份始终来自可撤销 Session。
    current_user_id: str = os.getenv("CURRENT_USER_ID", "User")
    current_department_id: str = os.getenv("CURRENT_DEPARTMENT_ID", "department-001")
    # 浏览器只保存不可读的随机 Session Cookie；生产 HTTPS 部署必须设为 true。
    session_cookie_name: str = os.getenv("SESSION_COOKIE_NAME", "knowledge_session")
    csrf_cookie_name: str = os.getenv("CSRF_COOKIE_NAME", "knowledge_csrf")
    session_cookie_secure: bool = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
    session_absolute_hours: int = int(os.getenv("SESSION_ABSOLUTE_HOURS", "168"))
    session_idle_minutes: int = int(os.getenv("SESSION_IDLE_MINUTES", "720"))
    registration_enabled: bool = os.getenv("REGISTRATION_ENABLED", "true").lower() == "true"
    # 旧单用户数据只能凭一次性接管码认领；留空表示彻底关闭旧数据接管。
    legacy_bootstrap_token: str = os.getenv("LEGACY_BOOTSTRAP_TOKEN", "")
    # 多用户 HTTP 服务不接受任意服务器路径；本地管理员确有需要时才能显式开启。
    allow_server_path_scan: bool = os.getenv("ALLOW_SERVER_PATH_SCAN", "false").lower() == "true"
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "8"))
    # 父块保存较完整的章节/页面/代码定义；子块更短，只对子块生成向量并参与召回。
    parent_chunk_size: int = int(os.getenv("PARENT_CHUNK_SIZE", "2400"))
    child_chunk_size: int = int(os.getenv("CHILD_CHUNK_SIZE", "700"))
    child_chunk_overlap: int = int(os.getenv("CHILD_CHUNK_OVERLAP", "100"))
    # 传给 Chat LLM 的证据预算；当前用字符估算 Token，后续可按具体模型接 tokenizer。
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))
    # 费用按服务商账单配置；未配置时仍记录 Token，但不虚构成本金额。
    embedding_input_cost_per_1k_tokens: float = float(os.getenv("EMBEDDING_INPUT_COST_PER_1K_TOKENS", "0"))
    chat_input_cost_per_1k_tokens: float = float(os.getenv("CHAT_INPUT_COST_PER_1K_TOKENS", "0"))
    chat_output_cost_per_1k_tokens: float = float(os.getenv("CHAT_OUTPUT_COST_PER_1K_TOKENS", "0"))
    reranker_cost_per_1k_tokens: float = float(os.getenv("RERANKER_COST_PER_1K_TOKENS", "0"))


settings = Settings()
