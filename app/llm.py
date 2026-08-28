from openai import OpenAI
import httpx

from .config import settings


def _client(api_base: str, api_key: str) -> OpenAI:
    """创建 OpenAI-compatible 客户端，聊天和 Embedding 可使用不同配置。"""
    if not api_base or not api_key:
        raise RuntimeError("LLM API configuration is incomplete")
    return OpenAI(base_url=api_base, api_key=api_key)


def _usage_value(usage, name: str) -> int:
    """兼容 OpenAI SDK 的对象 usage 和部分兼容服务返回的字典 usage。"""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(name, 0) or 0)
    return int(getattr(usage, name, 0) or 0)


def embed(text: str, return_usage: bool = False):
    """将一个文本分块转换成固定维度的向量；可选返回服务端 Token 用量。"""
    response = _client(settings.embedding_api_base, settings.embedding_api_key).embeddings.create(
        model=settings.embedding_model,
        input=[text],
        dimensions=settings.embedding_dimensions,
        encoding_format="float",
    )
    embedding = response.data[0].embedding
    if not return_usage:
        return embedding
    return embedding, {"prompt_tokens": _usage_value(getattr(response, "usage", None), "prompt_tokens")}


def answer(question: str, context: str, return_usage: bool = False):
    """只把检索证据交给模型；可选返回输入和输出 Token 用量。"""
    response = _client(settings.llm_api_base, settings.llm_api_key).chat.completions.create(
        model=settings.chat_model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是企业知识库助手。只能依据提供的证据回答，不要编造。"
                    "如果证据不足，明确说明无法确认。回答要简洁，并保留来源编号。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n证据：\n{context}",
            },
        ],
    )
    answer_text = response.choices[0].message.content or "无法生成回答。"
    if not return_usage:
        return answer_text
    usage = getattr(response, "usage", None)
    return answer_text, {
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
    }


def rerank(query: str, results: list[dict], top_n: int) -> list[dict]:
    """使用阿里云原生 Text Rerank API 对向量初召回结果进行二次排序。"""
    if not results:
        return []
    if not settings.reranker_api_url or not settings.reranker_api_key:
        raise RuntimeError("Reranker API configuration is incomplete")

    response = httpx.post(
        settings.reranker_api_url,
        headers={"Authorization": f"Bearer {settings.reranker_api_key}"},
        json={
            "model": settings.reranker_model,
            "input": {"query": query, "documents": [item["content"] for item in results]},
            "parameters": {"return_documents": True, "top_n": top_n},
        },
        timeout=60.0,
    )
    response.raise_for_status()
    payload = response.json()
    ranked = payload.get("output", {}).get("results") or payload.get("results") or []

    reordered = []
    for item in ranked:
        index = item.get("index")
        if isinstance(index, int) and 0 <= index < len(results):
            result = dict(results[index])
            result["rerank_score"] = float(item.get("relevance_score", 0.0))
            reordered.append(result)
    return reordered
