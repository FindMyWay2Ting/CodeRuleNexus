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
    workspace_id: str = os.getenv("WORKSPACE_ID", "workspace-001")
    # 当前工作空间显示名称；MVP 阶段默认沿用工作空间 ID，后续可接入工作空间表。
    current_workspace_name: str = os.getenv("CURRENT_WORKSPACE_NAME", "workspace-001")
    # 当前还没有认证，先用服务端配置模拟当前用户和部门；不能信任浏览器上传的 user_id。
    current_user_id: str = os.getenv("CURRENT_USER_ID", "User")
    current_department_id: str = os.getenv("CURRENT_DEPARTMENT_ID", "department-001")
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
