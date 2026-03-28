import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}

GOVERNMENT_DOMAINS = {
    "gov.sg",
    "police.gov.sg",
    "moh.gov.sg",
    "cpf.gov.sg",
    "mom.gov.sg",
    "moe.gov.sg",
    "pofmaoffice.gov.sg",
}

MAINSTREAM_DOMAINS = {
    "channelnewsasia.com",
    "straitstimes.com",
    "businesstimes.com.sg",
    "sg.news.yahoo.com",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _extract_domain(url: str) -> str:
    if not url:
        return ""

    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _overlap_ratio(target_tokens: set[str], candidate_text: str) -> float:
    if not target_tokens:
        return 0.0

    candidate_tokens = _tokenize(candidate_text)
    if not candidate_tokens:
        return 0.0

    return len(target_tokens & candidate_tokens) / len(target_tokens)


def _source_quality_bonus(evidence: Dict[str, Any]) -> Tuple[float, str]:
    source_type = str(evidence.get("source_type") or "").lower()
    domain = _extract_domain(str(evidence.get("url") or ""))

    if domain in GOVERNMENT_DOMAINS:
        return 0.22, "government-domain"
    if domain in MAINSTREAM_DOMAINS:
        return 0.12, "mainstream-domain"
    if source_type in {"vector_db", "demo_seed", "policy_doc", "factually", "pofma"}:
        return 0.16, "trusted-local-db"
    if source_type:
        return 0.05, f"source-type:{source_type}"
    return 0.0, "unknown-source"


def _content_quality_score(evidence: Dict[str, Any]) -> Tuple[float, List[str]]:
    content = str(evidence.get("content") or "")
    title = str(evidence.get("title") or "")

    score = 0.0
    reasons: List[str] = []

    if len(content) >= 120:
        score += 0.10
        reasons.append("content-length-good")
    elif len(content) >= 60:
        score += 0.05
        reasons.append("content-length-usable")
    else:
        score -= 0.08
        reasons.append("content-too-short")

    if title:
        score += 0.04
        reasons.append("has-title")
    else:
        score -= 0.03
        reasons.append("missing-title")

    if evidence.get("url"):
        score += 0.03
        reasons.append("has-url")
    else:
        score -= 0.05
        reasons.append("missing-url")

    return score, reasons


def deduplicate_evidence(evidence_list: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []

    for evidence in evidence_list:
        url = _normalized_text(str(evidence.get("url") or ""))
        title = _normalized_text(str(evidence.get("title") or ""))
        content = _normalized_text(str(evidence.get("content") or ""))
        key = url or f"{title}::{content[:160]}"

        if not key or key in seen:
            continue

        seen.add(key)
        deduped.append(evidence)

    return deduped


def rerank_evidence(
    claim: str,
    query: str,
    evidence_list: Iterable[Dict[str, Any]],
    *,
    top_k: int = 3,
    min_score: float = 0.30,
) -> List[Dict[str, Any]]:
    claim_tokens = _tokenize(claim)
    query_tokens = _tokenize(query)
    unique_evidence = deduplicate_evidence(list(evidence_list))

    rescored: List[Dict[str, Any]] = []

    for evidence in unique_evidence:
        title = str(evidence.get("title") or "")
        content = str(evidence.get("content") or "")
        combined_text = f"{title} {content}".strip()

        raw_score = evidence.get("score")
        base_score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0

        claim_overlap = _overlap_ratio(claim_tokens, combined_text)
        query_overlap = _overlap_ratio(query_tokens, combined_text)

        source_bonus, source_reason = _source_quality_bonus(evidence)
        quality_score, quality_reasons = _content_quality_score(evidence)
        relevance_penalty = 0.0
        relevance_reasons: List[str] = []

        if claim_overlap < 0.34 and query_overlap < 0.34 and source_bonus < 0.16:
            relevance_penalty -= 0.20
            relevance_reasons.append("weak-lexical-match")

        final_score = (
            0.45 * base_score
            + 0.30 * claim_overlap
            + 0.15 * query_overlap
            + source_bonus
            + quality_score
            + relevance_penalty
        )

        reasons = [
            f"base={base_score:.3f}",
            f"claim_overlap={claim_overlap:.3f}",
            f"query_overlap={query_overlap:.3f}",
            source_reason,
            *quality_reasons,
            *relevance_reasons,
        ]

        enriched = dict(evidence)
        enriched["rerank_score"] = round(final_score, 4)
        enriched["rerank_reasons"] = reasons
        rescored.append(enriched)

    rescored.sort(
        key=lambda item: (
            item.get("rerank_score", 0.0),
            item.get("score", 0.0) if isinstance(item.get("score"), (int, float)) else 0.0,
        ),
        reverse=True,
    )

    filtered = [
        item
        for item in rescored
        if item.get("rerank_score", 0.0) >= min_score
        and "weak-lexical-match" not in item.get("rerank_reasons", [])
    ]

    final_results = filtered[:top_k] if filtered else rescored[:top_k]
    for rank, item in enumerate(final_results, 1):
        item["rank"] = rank

    return final_results
