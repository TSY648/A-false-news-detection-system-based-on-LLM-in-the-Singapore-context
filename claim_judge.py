import json
import re
from typing import Any, Dict, List, Optional

from google import genai

from model_config import get_judge_api_key, get_judge_model


LABEL_SUPPORTED = "Supported"
LABEL_REFUTED = "Refuted"
LABEL_NEE = "Not Enough Evidence"


def extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"```json\s*(\{.*\})\s*```", text, re.S)
    if m:
        return json.loads(m.group(1))

    m = re.search(r"(\{.*\})", text, re.S)
    if m:
        return json.loads(m.group(1))

    raise ValueError("Could not parse JSON from model output:\n" + text)


def _format_search_evidences(tavily_evidence: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    if not tavily_evidence:
        lines.append("- (none)")
    for idx, e in enumerate(tavily_evidence, start=1):
        lines.append(
            f"- [T{idx}] title={e.get('title','')} url={e.get('url','')} "
            f"content={str(e.get('content',''))}"
        )
    return "\n".join(lines)


def build_judge_prompt(
    claim: str,
    tavily_evidence: List[Dict[str, Any]],
    vector_evidence: List[Dict[str, Any]],
    event_date: Optional[str] = None,
    today_date: Optional[str] = None,
) -> str:
    """Full prompt sent to the judgment model, identical to the actual judge_claim call."""
    search_evidences_block = _format_search_evidences(tavily_evidence)
    historical_evidences_block = _format_historical_evidences(vector_evidence)
    return f"""
You are a strict fact-checking judge for Singapore-context claims.

Task:
Given one CLAIM and two evidence channels (search_evidences + historical_evidences), decide exactly one label:
- Supported
- Refuted
- Not Enough Evidence

Rules:
1) Use only provided evidence. Do not rely on external memory.
2) If evidence is weak, conflicting, or not directly about the claim => Not Enough Evidence.
3) If strong evidence directly contradicts the claim => Refuted.
4) If strong evidence directly confirms the claim => Supported.
5) Return concise rationale and cite evidence IDs like [T1], [V2].

Return JSON only:
{{
  "label": "Supported | Refuted | Not Enough Evidence",
  "reason": "short explanation",
  "citations": ["T1","V2"]
}}

CLAIM:
{claim}

event_date:
{event_date or "unspecified"}

today_date:
{today_date or "unspecified"}

relevant_evidences:
    search_evidences:
    {search_evidences_block}

    historical_evidences:
    {historical_evidences_block}
""".strip()


def _format_historical_evidences(vector_evidence: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    if not vector_evidence:
        lines.append("- (none)")
    for idx, e in enumerate(vector_evidence, start=1):
        lines.append(
            f"- [V{idx}] title={e.get('title','')} url={e.get('url','')} "
            f"full_text={str(e.get('full_text',''))}"
        )
    return "\n".join(lines)


def judge_claim(
    claim: str,
    tavily_evidence: List[Dict[str, Any]],
    vector_evidence: List[Dict[str, Any]],
    event_date: Optional[str] = None,
    today_date: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    model = (model_name or "").strip() or get_judge_model()
    client = genai.Client(api_key=get_judge_api_key())
    prompt = build_judge_prompt(
        claim=claim,
        tavily_evidence=tavily_evidence,
        vector_evidence=vector_evidence,
        event_date=event_date,
        today_date=today_date,
    )

    resp = client.models.generate_content(model=model, contents=prompt)
    module3_response_text = getattr(resp, "text", None) or ""
    parsed = extract_json(module3_response_text)

    label = str(parsed.get("label", "")).strip()
    if label not in {LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEE}:
        label = LABEL_NEE

    return {
        "label": label,
        "reason": str(parsed.get("reason", "")).strip(),
        "citations": parsed.get("citations", []),
        "module3_prompt": prompt,
        "module3_response_text": module3_response_text,
    }


def aggregate_news_label(claim_results: List[Dict[str, Any]]) -> str:
    labels = [item.get("verdict", {}).get("label") for item in claim_results]
    if LABEL_REFUTED in labels:
        return LABEL_REFUTED
    if LABEL_NEE in labels:
        return LABEL_NEE
    return LABEL_SUPPORTED
