"""
Integration test:
gemini_for_claim (claim extraction + query generation)
    -> req_tavily (Tavily retrieval)

Usage:
    python test_pipeline.py
    python test_pipeline.py "Heard that Singapore will have a lockdown yesterday"
"""

import copy
import sys
import json
from typing import Any, Dict, Optional

from pipeline_runtime_config import setup_pipeline_runtime_interactive
from judge_router import aggregate_news_label, judge_claim
from gemini_for_claim import build_search_tasks
from model_config import get_claim_extract_model, get_judge_model
from req_tavily import tavily_search
from result_store import persist_pipeline_result
from vector_store import ChromaEvidenceStore


DEMO_TEXTS = [
    "Heard that Singapore will have a lockdown yesterday, and CPF withdrawals are not possible."
]

TAVILY_SCORE_THRESHOLD = 0.45
VECTOR_DISTANCE_THRESHOLD = 0.35


def _build_event_date(start_date: Optional[str], end_date: Optional[str]) -> str:
    if start_date and end_date:
        return f"{start_date} to {end_date}"
    if start_date and not end_date:
        return f"from {start_date}"
    if end_date and not start_date:
        return f"until {end_date}"
    return "unspecified"


def _filter_tavily_evidence(evidence: list[dict]) -> list[dict]:
    if not evidence:
        return []
    filtered = [e for e in evidence if float(e.get("score", 0.0)) >= TAVILY_SCORE_THRESHOLD]
    if filtered:
        return filtered
    # Fallback: keep top-1 to avoid empty evidence path.
    return [max(evidence, key=lambda x: float(x.get("score", 0.0)))]


def _best_vector_distance(item: dict) -> float:
    distances = [
        m.get("distance")
        for m in item.get("matched_chunks", [])
        if m.get("distance") is not None
    ]
    if not distances:
        return 1e9
    return min(float(d) for d in distances)


def _filter_vector_evidence(evidence: list[dict]) -> list[dict]:
    if not evidence:
        return []
    filtered = [e for e in evidence if _best_vector_distance(e) <= VECTOR_DISTANCE_THRESHOLD]
    if filtered:
        return filtered
    # Fallback: keep top-1 nearest source to avoid empty evidence path.
    return [min(evidence, key=_best_vector_distance)]


# ---------------- Core pipeline ----------------
def run_pipeline(
    raw_text: str,
    model_claim: Optional[str] = None,
    model_judge: Optional[str] = None,
) -> dict:
    # Data stream:
    # raw_text -> claim/query extraction -> Tavily + VectorDB retrieval
    # -> claim-level judge -> news-level aggregation
    mc = model_claim or get_claim_extract_model()
    mj = model_judge or get_judge_model()
    tasks, module1_prompt, module1_response_text = build_search_tasks(raw_text, model_name=mc)
    today_date = tasks[0].today_date if tasks else ""
    vector_store = ChromaEvidenceStore()

    claims = []
    for task in tasks:
        event_date = _build_event_date(task.start_date, task.end_date)
        tavily_evidence_raw = tavily_search(
            query=task.query,
            start_date=task.start_date,
            end_date=task.end_date,
        )
        vector_evidence_raw = vector_store.search_and_expand(query=task.claim, n_results=3)
        tavily_evidence = _filter_tavily_evidence(tavily_evidence_raw)
        vector_evidence = _filter_vector_evidence(vector_evidence_raw)
        verdict = judge_claim(
            claim=task.claim,
            tavily_evidence=tavily_evidence,
            vector_evidence=vector_evidence,
            event_date=event_date,
            today_date=today_date,
            model_name=mj,
        )
        claims.append(
            {
                "claim": task.claim,
                "query": task.query,
                "start_date": task.start_date,
                "end_date": task.end_date,
                "event_date": event_date,
                "tavily_evidence_count": len(tavily_evidence),
                "vector_evidence_count": len(vector_evidence),
                "tavily_evidence": tavily_evidence,
                "vector_evidence": vector_evidence,
                "verdict": verdict,
            }
        )

    news_label = aggregate_news_label(claims) if claims else "Not Enough Evidence"
    return {
        "raw_text": raw_text,
        "today_date": today_date,
        "claim_count": len(claims),
        "news_label": news_label,
        "claims": claims,
        "model_claim_extract": mc,
        "model_judge": mj,
        "module1_prompt": module1_prompt,
        "module1_response_text": module1_response_text,
    }


# ---------------- Print results ----------------
def print_results(result: dict):
    print(f"\n  Raw input  : {result['raw_text']}")
    print(f"  Today date : {result['today_date']}")
    print(f"  Claim count: {result['claim_count']}")
    print(f"  News label : {result['news_label']}")

    for i, c in enumerate(result["claims"], 1):
        print(f"\n{'='*70}")
        print(f"  Claim #{i}")
        print(f"{'='*70}")
        print(f"  Claim    : {c['claim']}")
        print(f"  Query    : {c['query']}")
        print(f"  Search window: {c['start_date'] or 'N/A'} ~ {c['end_date'] or 'N/A'}")
        print(f"  Event Date: {c['event_date']}")
        print(f"  Tavily evidence count : {c['tavily_evidence_count']}")
        print(f"  Vector evidence count : {c['vector_evidence_count']}")
        print(f"  Verdict label  : {c['verdict']['label']}")
        print(f"  Verdict reason : {c['verdict']['reason']}")
        print(f"  Citations      : {c['verdict'].get('citations', [])}")

        if c["tavily_evidence"]:
            print("  [Tavily Evidence]")
            for j, e in enumerate(c["tavily_evidence"], 1):
                print(f"\n  --- T{j} ---")
                print(f"  Title   : {e['title']}")
                print(f"  Score   : {e['score']:.4f}")
                print(f"  URL     : {e['url']}")
                print(f"  Content : {e['content'][:200]}...")
        else:
            print("  [Tavily Evidence] (No relevant evidence retrieved)")

        if c["vector_evidence"]:
            print("  [VectorDB Evidence]")
            for j, e in enumerate(c["vector_evidence"], 1):
                distances = [m.get("distance") for m in e.get("matched_chunks", []) if m.get("distance") is not None]
                best_distance = min(distances) if distances else None
                print(f"\n  --- V{j} source_id={e['source_id']} ---")
                print(f"  Title   : {e['title']}")
                if best_distance is not None:
                    print(f"  Distance: {best_distance:.4f}")
                else:
                    print("  Distance: N/A")
                print(f"  URL     : {e['url']}")
                print(f"  Type    : {e['source_type']}")
                print(f"  Matched : {len(e.get('matched_chunks', []))} chunks")
                print(f"  Content : {e['full_text'][:220]}...")
        else:
            print("  [VectorDB Evidence] (No relevant evidence retrieved)")

    print(f"\n{'='*70}")
    print(
        "Full JSON output (excluding module1_prompt / module1_response_text / module3_prompt / "
        "module3_response_text to avoid overly long terminal output; the persisted CSV still "
        "contains the full text):"
    )
    slim: Dict[str, Any] = copy.deepcopy(result)
    slim.pop("module1_prompt", None)
    slim.pop("module1_response_text", None)
    for c in slim.get("claims", []):
        v = c.get("verdict")
        if isinstance(v, dict):
            v.pop("module3_prompt", None)
            v.pop("module3_response_text", None)
    print(json.dumps(slim, ensure_ascii=False, indent=2))


# ---------------- Test entry point ----------------
def main():
    claim_m, judge_m, emb_m = setup_pipeline_runtime_interactive()
    if len(sys.argv) > 1:
        texts = [" ".join(sys.argv[1:])]
    else:
        user_text = input("Enter raw_text to test (press Enter to use the default demo): ").strip()
        texts = [user_text] if user_text else DEMO_TEXTS

    for idx, text in enumerate(texts):
        print(f"\n{'#'*70}")
        print(f"# Test case {idx + 1}: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"{'#'*70}")

        result = run_pipeline(text)
        print_results(result)
        news_run_id = persist_pipeline_result(
            result,
            model_claim_extract=result.get("model_claim_extract"),
            model_judge=result.get("model_judge"),
        )
        print(f"\nWritten to SQLite database. news_run.id={news_run_id}")
        print(f"Claim extract model: {claim_m}")
        print(f"Judge model          : {judge_m}")
        print(f"Embedding model      : {emb_m}")


if __name__ == "__main__":
    main()
