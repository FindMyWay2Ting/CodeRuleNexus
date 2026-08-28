"""运行固定 RAG 检索评测集，输出路由和召回指标。"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.db import expand_context, hybrid_search, list_knowledge
from app.llm import answer, embed, rerank
from app.main import route_question


def load_cases(path: Path) -> list[dict]:
    """读取一行一个 JSON 对象的评测集，便于追加样本和做 Git diff。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def result_text(item: dict) -> str:
    """将来源元数据和分块正文合并，作为可解释的命中判断范围。"""
    return " ".join(str(item.get(field, "")) for field in ("source_name", "source_path", "source_ref", "content")).lower()


def _is_relevant(item: dict, expected_sources: list[str]) -> bool:
    """用标注的来源关键词判断一个结果是否相关，避免把相似但错误的文档算作命中。"""
    return any(term in result_text(item) for term in expected_sources)


def precision_at_k(results: list[dict], expected_sources: list[str], k: int) -> float:
    """Precision@K：前 K 个结果中，来源属于标注集合的比例。"""
    if k <= 0:
        return 0.0
    return sum(_is_relevant(item, expected_sources) for item in results[:k]) / k


def ndcg_at_k(results: list[dict], expected_sources: list[str], k: int) -> float:
    """二值相关性 NDCG@K，既关注命中数量，也惩罚相关结果排在后面。"""
    if k <= 0 or not expected_sources:
        return 0.0
    gains = [1 if _is_relevant(item, expected_sources) else 0 for item in results[:k]]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_hits = min(sum(gains), k)
    ideal_dcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _source_recall(items: list[dict], expected_sources: list[str], parent_only: bool = False) -> float:
    """按期望来源计算召回；parent_only 用于验证父块是否真正恢复了上下文。"""
    if not expected_sources:
        return 0.0
    matched = 0
    for source in expected_sources:
        if any(source in result_text(item) and (not parent_only or item.get("context_level") == "parent") for item in items):
            matched += 1
    return matched / len(expected_sources)


def build_evidence(items: list[dict]) -> str:
    """生成与生产问答相同格式的带编号证据，保证引用评测和线上行为一致。"""
    return "\n\n".join(
        f"[{index + 1}] 类型={item.get('source_type', '')} 来源={item.get('source_ref', '')}\n{item.get('content', '')}"
        for index, item in enumerate(items)
    )


def evaluate_answer(answer_text: str, case: dict, sources: list[dict]) -> dict:
    """评估答案要点、引用存在性/正确性和拒答行为；不评价文风。"""
    lower_answer = answer_text.lower()
    expected_terms = [term.lower() for term in case.get("answer_terms", case.get("required_terms", []))]
    covered_terms = [term for term in expected_terms if term in lower_answer]
    citation_terms = [term.lower() for term in case.get("citation_source_terms", case.get("expected_source_terms", []))]
    cited_indexes = {int(index) for index in re.findall(r"\[(\d+)\]", answer_text)}
    cited_sources = [sources[index - 1] for index in cited_indexes if 1 <= index <= len(sources)]
    citation_correct = bool(cited_sources) and any(_is_relevant(item, citation_terms) for item in cited_sources)
    refusal_detected = bool(re.search(r"无法确认|证据不足|没有相关|未找到|不能确定|无法回答", answer_text))
    should_refuse = bool(case.get("should_refuse", False))
    return {
        "answer_term_coverage": len(covered_terms) / len(expected_terms) if expected_terms else (1.0 if not should_refuse else 0.0),
        "answer_covered_terms": covered_terms,
        "citation_present": bool(cited_indexes),
        "citation_correct": citation_correct,
        "refusal_detected": refusal_detected,
        "refusal_correct": refusal_detected == should_refuse,
    }


def evaluate_case(case: dict, available_sources: list[str], use_reranker: bool, top_k: int, evaluate_generation: bool) -> dict:
    """执行单个样本，覆盖路由、检索排序、父块上下文和可选生成评测。"""
    route = route_question(case["question"])
    source_types = {"wiki": ["wiki"], "rag": ["rag"], "hybrid": ["wiki", "rag"]}[route]
    candidates = hybrid_search(embed(case["question"]), case["question"], source_types, 20)
    results = candidates
    retrieval_mode = "hybrid"
    if use_reranker and candidates:
        try:
            results = rerank(case["question"], candidates, top_k)
            retrieval_mode = "hybrid+rerank"
        except Exception as exc:
            retrieval_mode = f"hybrid+rerank-fallback:{type(exc).__name__}"
    results = results[:top_k]
    context_results = expand_context(results, settings.max_context_tokens)

    expected_sources = [term.lower() for term in case["expected_source_terms"]]
    expected_terms = [term.lower() for term in case.get("required_terms", [])]
    source_available = any(
        term in source_name
        for term in expected_sources
        for source_name in available_sources
    )
    matching_ranks = [
        index
        for index, item in enumerate(results, start=1)
        if any(term in result_text(item) for term in expected_sources)
    ]
    combined = " ".join(result_text(item) for item in context_results)
    covered_terms = [term for term in expected_terms if term in combined]
    first_rank = matching_ranks[0] if matching_ranks else None
    output = {
        "id": case["id"],
        "evaluation_type": case.get("evaluation_type", "retrieval"),
        "route": route,
        "expected_route": case["expected_route"],
        "route_correct": route == case["expected_route"],
        "source_hit_at_k": bool(matching_ranks),
        "source_available": source_available,
        "first_source_rank": first_rank,
        "mrr": 1 / first_rank if first_rank else 0.0,
        "required_term_coverage": len(covered_terms) / len(expected_terms) if expected_terms else 1.0,
        "covered_terms": covered_terms,
        "precision_at_k": precision_at_k(results, expected_sources, top_k),
        "ndcg_at_k": ndcg_at_k(results, expected_sources, top_k),
        "context_recall_at_k": _source_recall(context_results, expected_sources),
        "parent_context_recall_at_k": _source_recall(context_results, expected_sources, parent_only=True),
        "parent_context_count": sum(item.get("context_level") == "parent" for item in context_results),
        "context_tokens": sum(item.get("context_tokens", 0) for item in context_results),
        "retrieval_mode": retrieval_mode,
        "top_sources": [item.get("source_name") for item in results],
    }
    if evaluate_generation:
        try:
            generated = answer(case["question"], build_evidence(context_results))
            output["answer"] = generated
            output.update(evaluate_answer(generated, case, context_results))
        except Exception as exc:
            output["answer_error"] = f"{type(exc).__name__}: {exc}"
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the two-level knowledge base retrieval pipeline")
    parser.add_argument("--dataset", default="evals/rag_eval.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", help="also call the configured reranker API")
    parser.add_argument("--answer", action="store_true", help="also call Chat LLM and evaluate answer/citation/refusal")
    args = parser.parse_args()

    cases = load_cases(Path(args.dataset))
    available_sources = [str(item["source_name"]).lower() for item in list_knowledge()]
    results = [evaluate_case(case, available_sources, args.rerank, args.top_k, args.answer) for case in cases]
    retrieval_cases = [item for item in results if item.get("evaluation_type", "retrieval") != "refusal"]
    route_accuracy = sum(item["route_correct"] for item in results) / len(results)
    recall_at_k = sum(item["source_hit_at_k"] for item in retrieval_cases) / len(retrieval_cases) if retrieval_cases else 0.0
    mrr = sum(item["mrr"] for item in retrieval_cases) / len(retrieval_cases) if retrieval_cases else 0.0
    term_coverage = sum(item["required_term_coverage"] for item in retrieval_cases) / len(retrieval_cases) if retrieval_cases else 0.0
    available_cases = [item for item in retrieval_cases if item["source_available"]]
    answer_cases = [item for item in retrieval_cases if "answer_term_coverage" in item]
    refusal_cases = [item for item in results if item.get("evaluation_type") == "refusal" and "refusal_correct" in item]
    eligible_recall = (
        sum(item["source_hit_at_k"] for item in available_cases) / len(available_cases)
        if available_cases else 0.0
    )
    summary = {
        "cases": len(results),
        "top_k": args.top_k,
        "rerank": args.rerank,
        "answer_evaluation": args.answer,
        "route_accuracy": round(route_accuracy, 4),
        "source_recall_at_k": round(recall_at_k, 4),
        "source_mrr": round(mrr, 4),
        "required_term_coverage": round(term_coverage, 4),
        "knowledge_coverage": round(len(available_cases) / len(results), 4),
        "source_recall_at_k_when_source_exists": round(eligible_recall, 4),
        "precision_at_k": round(sum(item["precision_at_k"] for item in retrieval_cases) / len(retrieval_cases), 4) if retrieval_cases else 0.0,
        "ndcg_at_k": round(sum(item["ndcg_at_k"] for item in retrieval_cases) / len(retrieval_cases), 4) if retrieval_cases else 0.0,
        "context_recall_at_k": round(sum(item["context_recall_at_k"] for item in retrieval_cases) / len(retrieval_cases), 4) if retrieval_cases else 0.0,
        "parent_context_recall_at_k": round(sum(item["parent_context_recall_at_k"] for item in retrieval_cases) / len(retrieval_cases), 4) if retrieval_cases else 0.0,
        "answer_cases": len(answer_cases),
        "refusal_cases": len(refusal_cases),
        "answer_term_coverage": round(sum(item["answer_term_coverage"] for item in answer_cases) / len(answer_cases), 4) if answer_cases else None,
        "citation_correctness": round(sum(item["citation_correct"] for item in answer_cases) / len(answer_cases), 4) if answer_cases else None,
        "refusal_accuracy": round(sum(item["refusal_correct"] for item in refusal_cases) / len(refusal_cases), 4) if refusal_cases else None,
        "details": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
