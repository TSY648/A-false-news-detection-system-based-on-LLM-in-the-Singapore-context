"""
联调测试：
    news_req_gemini_for_claim  -> 模块 1 / 2
    retriever                  -> 模块 3.1 Pinecone 向量检索
    news_req_req_tavily        -> 模块 3.2 实时联网检索
"""

import json
import sys

from news_req_gemini_for_claim import build_search_tasks
from news_req_req_tavily import tavily_search
from retriever import Retriever
from reranker import rerank_evidence


DEMO_TEXTS = [
    "SINGAPORE: Singapore's total fertility rate dropped significantly to a new low of 0.87 in 2025, Deputy Prime Minister Gan Kim Yong said in parliament on Thursday (Feb 26)."
]


def to_llm_evidence(evidence_list: list[dict]) -> list[dict]:
    """Convert detailed retrieval results into the minimal title+content format for the LLM."""
    return [
        {
            "title": item.get("title") or "",
            "content": item.get("content") or "",
        }
        for item in evidence_list
    ]


def run_pipeline(raw_text: str) -> list[dict]:
    """完整流水线：claim 提取 -> 双路检索 -> 分路重排序 -> 各保留 Top-3。"""
    tasks = build_search_tasks(raw_text)
    all_results = []

    retriever = Retriever()
    retriever.bootstrap_from_seed()

    for task in tasks:
        pinecone_evidence = retriever.retrieve_by_claim(task.claim, top_k=5)
        web_evidence = tavily_search(
            query=task.query,
            max_results=5,
            start_date=task.start_date,
            end_date=task.end_date,
        )
        merged_evidence = pinecone_evidence + web_evidence

        pinecone_top_evidence = rerank_evidence(
            claim=task.claim,
            query=task.query,
            evidence_list=pinecone_evidence,
            top_k=3,
        )
        web_top_evidence = rerank_evidence(
            claim=task.claim,
            query=task.query,
            evidence_list=web_evidence,
            top_k=3,
        )
        final_evidence = pinecone_top_evidence + web_top_evidence
        pinecone_top_evidence_for_llm = to_llm_evidence(pinecone_top_evidence)
        web_top_evidence_for_llm = to_llm_evidence(web_top_evidence)
        evidence_for_llm = pinecone_top_evidence_for_llm + web_top_evidence_for_llm

        all_results.append(
            {
                "raw_text": task.raw_text,
                "claim": task.claim,
                "query": task.query,
                "today_date": task.today_date,
                "start_date": task.start_date,
                "end_date": task.end_date,
                "cache_hit": bool(pinecone_evidence),
                "pinecone_evidence_count": len(pinecone_evidence),
                "web_evidence_count": len(web_evidence),
                "merged_evidence_count": len(merged_evidence),
                "pinecone_top_evidence_count": len(pinecone_top_evidence),
                "web_top_evidence_count": len(web_top_evidence),
                "evidence_count": len(final_evidence),
                "pinecone_evidence": pinecone_evidence,
                "web_evidence": web_evidence,
                "merged_evidence": merged_evidence,
                "pinecone_top_evidence": pinecone_top_evidence,
                "web_top_evidence": web_top_evidence,
                "evidence": final_evidence,
                "pinecone_top_evidence_for_llm": pinecone_top_evidence_for_llm,
                "web_top_evidence_for_llm": web_top_evidence_for_llm,
                "evidence_for_llm": evidence_for_llm,
            }
        )

    return all_results


def print_results(results: list[dict]):
    for i, r in enumerate(results, 1):
        print(f"\n{'=' * 70}")
        print(f"  Claim #{i}")
        print(f"{'=' * 70}")
        print(f"  原始输入 : {r['raw_text']}")
        print(f"  Claim    : {r['claim']}")
        print(f"  Query    : {r['query']}")
        print(f"  今日日期 : {r['today_date']}")
        print(f"  搜索区间 : {r['start_date'] or '无'} ~ {r['end_date'] or '无'}")
        print(f"  Cache Hit: {r['cache_hit']}")
        print(f"  Pinecone证据 : {r['pinecone_evidence_count']}")
        print(f"  联网证据 : {r['web_evidence_count']}")
        print(f"  合并证据 : {r['merged_evidence_count']}")
        print(f"  Pinecone Top-3 : {r['pinecone_top_evidence_count']}")
        print(f"  联网Top-3 : {r['web_top_evidence_count']}")
        print(f"  最终证据数: {r['evidence_count']}")

        if not r["evidence"]:
            print("  (未检索到相关证据)")
            continue

        print("\n  [Pinecone Top Evidence]")
        for j, e in enumerate(r["pinecone_top_evidence"], 1):
            score = e.get("score")
            raw_score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
            rerank_score = e.get("rerank_score")
            rerank_score_text = f"{rerank_score:.4f}" if isinstance(rerank_score, (int, float)) else "N/A"
            print(f"\n  --- Pinecone Evidence {j} (raw: {raw_score_text}, rerank: {rerank_score_text}) ---")
            print(f"  Title   : {e.get('title')}")
            print(f"  URL     : {e.get('url')}")
            print(f"  Type    : {e.get('source_type', 'unknown')}")
            print(f"  Rank    : {e.get('rank')}")
            print(f"  Content : {e.get('content', '')[:200]}...")

        print("\n  [Web Top Evidence]")
        for j, e in enumerate(r["web_top_evidence"], 1):
            score = e.get("score")
            raw_score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
            rerank_score = e.get("rerank_score")
            rerank_score_text = f"{rerank_score:.4f}" if isinstance(rerank_score, (int, float)) else "N/A"
            print(f"\n  --- Web Evidence {j} (raw: {raw_score_text}, rerank: {rerank_score_text}) ---")
            print(f"  Title   : {e.get('title')}")
            print(f"  URL     : {e.get('url')}")
            print(f"  Type    : {e.get('source_type', 'unknown')}")
            print(f"  Rank    : {e.get('rank')}")
            print(f"  Content : {e.get('content', '')[:200]}...")

    print(f"\n{'=' * 70}")
    print("完整 JSON 输出：")
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    if len(sys.argv) > 1:
        texts = [" ".join(sys.argv[1:])]
    else:
        texts = DEMO_TEXTS

    for idx, text in enumerate(texts):
        print(f"\n{'#' * 70}")
        print(f"# 测试用例 {idx + 1}: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"{'#' * 70}")

        results = run_pipeline(text)
        print_results(results)


if __name__ == "__main__":
    main()
