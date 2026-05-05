from typing import Any, Dict, List, Optional

from qwen_judge import (
    LABEL_NEE,
    LABEL_REFUTED,
    LABEL_SUPPORTED,
    judge_claim_with_qwen,
)


def judge_claim(
    claim: str,
    tavily_evidence: List[Dict[str, Any]],
    vector_evidence: List[Dict[str, Any]],
    event_date: Optional[str] = None,
    today_date: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    return judge_claim_with_qwen(
        claim=claim,
        tavily_evidence=tavily_evidence,
        vector_evidence=vector_evidence,
        event_date=event_date,
        today_date=today_date,
        model_name=model_name,
    )


def aggregate_news_label(claim_results: List[Dict[str, Any]]) -> str:
    labels = [item.get("verdict", {}).get("label") for item in claim_results]
    if LABEL_REFUTED in labels:
        return LABEL_REFUTED
    if LABEL_NEE in labels:
        return LABEL_NEE
    return LABEL_SUPPORTED


__all__ = ["judge_claim", "aggregate_news_label"]
