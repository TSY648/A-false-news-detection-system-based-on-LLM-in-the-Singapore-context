import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from model_config import DEFAULT_QWEN_JUDGE_BASE_URL


LABEL_SUPPORTED = "Supported"
LABEL_REFUTED = "Refuted"
LABEL_NEE = "Not Enough Evidence"

DEFAULT_QWEN_BASE_URL = DEFAULT_QWEN_JUDGE_BASE_URL
DEFAULT_QWEN_MODEL = "qwen-judge"


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


def _truncate_text(text: str, limit: int = 2200) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + " ..."


def _format_search_evidences(tavily_evidence: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for item in tavily_evidence:
        rows.append(
            {
                "title": str(item.get("title", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "content": _truncate_text(item.get("content", "")),
            }
        )
    return rows


def _format_historical_evidences(vector_evidence: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for item in vector_evidence:
        rows.append(
            {
                "title": str(item.get("title", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "content": _truncate_text(item.get("full_text") or item.get("content", "")),
            }
        )
    return rows


def build_qwen_judge_input(
    claim: str,
    tavily_evidence: List[Dict[str, Any]],
    vector_evidence: List[Dict[str, Any]],
    event_date: Optional[str] = None,
    today_date: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "prompt": "As a Singapore misinformation detection expert, judge the truthfulness of the claim based on the evidence.",
        "claim": claim,
        "event_date": event_date or "",
        "today_date": today_date or "",
        "search_evidences": _format_search_evidences(tavily_evidence),
        "historical_evidences": _format_historical_evidences(vector_evidence),
    }


def build_qwen_prompt(input_payload: Dict[str, Any]) -> str:
    return (
        "You are a strict Singapore fact-checking judgment model.\n"
        "Judge the truthfulness of the claim using only the provided evidence. Do not use external knowledge.\n"
        "In the input JSON:\n"
        "- search_evidences correspond to Tavily search evidence and should be cited as T1/T2/...\n"
        "- historical_evidences correspond to vector database evidence and should be cited as V1/V2/...\n"
        "The label must be exactly one of the following: Supported, Refuted, Not Enough Evidence.\n"
        "Rules:\n"
        "1. If strong evidence directly supports the claim, output Supported.\n"
        "2. If strong evidence directly contradicts the claim, output Refuted.\n"
        "3. If evidence is insufficient, not directly relevant, or conflicting, output Not Enough Evidence.\n"
        "4. The reason must be short and clear.\n"
        "5. citations must be an array of evidence IDs, such as [\"T1\", \"V2\"].\n"
        "6. Output JSON only. Do not include any extra explanation.\n\n"
        "Output format:\n"
        "{\n"
        "  \"label\": \"Supported | Refuted | Not Enough Evidence\",\n"
        "  \"reason\": \"short explanation\",\n"
        "  \"citations\": [\"T1\", \"V2\"]\n"
        "}\n\n"
        "Input JSON:\n"
        f"{json.dumps(input_payload, ensure_ascii=False, indent=2)}"
    )


def _get_qwen_base_url() -> str:
    return (
        os.environ.get("QWEN_JUDGE_BASE_URL", "").strip()
        or os.environ.get("JUDGE_BASE_URL", "").strip()
        or DEFAULT_QWEN_BASE_URL
    )


def _get_qwen_api_key() -> str:
    return (
        os.environ.get("QWEN_JUDGE_API_KEY", "").strip()
        or os.environ.get("JUDGE_API_KEY", "").strip()
        or "EMPTY"
    )


def _get_qwen_model(model_name: Optional[str] = None) -> str:
    return (
        (model_name or "").strip()
        or os.environ.get("QWEN_JUDGE_MODEL", "").strip()
        or os.environ.get("CLAIM_JUDGE_MODEL", "").strip()
        or DEFAULT_QWEN_MODEL
    )


def _post_openai_compat_chat(
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    timeout: int = 180,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful fact-checking judge. Return JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="ignore")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Qwen judge request failed: HTTP {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Qwen judge connection failed: {exc}") from exc

    parsed = json.loads(raw)
    choices = parsed.get("choices") or []
    if not choices:
        raise RuntimeError(f"Qwen judge response missing choices: {raw}")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        content = "\n".join(parts)
    return str(content or "").strip()


def judge_claim_with_qwen(
    claim: str,
    tavily_evidence: List[Dict[str, Any]],
    vector_evidence: List[Dict[str, Any]],
    event_date: Optional[str] = None,
    today_date: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    input_payload = build_qwen_judge_input(
        claim=claim,
        tavily_evidence=tavily_evidence,
        vector_evidence=vector_evidence,
        event_date=event_date,
        today_date=today_date,
    )
    prompt = build_qwen_prompt(input_payload)
    response_text = _post_openai_compat_chat(
        base_url=_get_qwen_base_url(),
        api_key=_get_qwen_api_key(),
        model_name=_get_qwen_model(model_name),
        prompt=prompt,
    )
    parsed = extract_json(response_text)

    label = str(parsed.get("label", "")).strip()
    if label not in {LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEE}:
        label = LABEL_NEE

    citations = parsed.get("citations", [])
    if not isinstance(citations, list):
        citations = []

    return {
        "label": label,
        "reason": str(parsed.get("reason", "")).strip(),
        "citations": [str(item).strip() for item in citations if str(item).strip()],
        "module3_prompt": prompt,
        "module3_response_text": response_text,
    }
