import re
import json
from dataclasses import dataclass, asdict
from datetime import date
from typing import Any, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from google import genai

from model_config import (
    get_claim_extract_api_key,
    get_claim_extract_base_url,
    get_claim_extract_model,
    get_claim_extract_provider,
)


def _get_client() -> genai.Client:
    return genai.Client(api_key=get_claim_extract_api_key())


def _should_use_qwen_for_claim_extract(model_name: Optional[str] = None) -> bool:
    provider = (get_claim_extract_provider() or "").strip().lower()
    if provider in {"qwen", "qwen_openai_compat", "openai_compat"}:
        return True
    model = (model_name or "").strip() or get_claim_extract_model()
    return "qwen" in model.lower()


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
                "content": "You are a careful claim extraction model. Return JSON only.",
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
        raise RuntimeError(f"Claim extract request failed: HTTP {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Claim extract connection failed: {exc}") from exc

    parsed = json.loads(raw)
    choices = parsed.get("choices") or []
    if not choices:
        raise RuntimeError(f"Claim extract response missing choices: {raw}")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        content = "\n".join(parts)
    return str(content or "").strip()


# ---------------- Extract JSON from model output ----------------
def extract_json(text: str) -> Any:
    """Parse JSON from model output, handling fences or extra surrounding text."""
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
    if m:
        return json.loads(m.group(1))

    m = re.search(r"(\[.*\])", text, re.S)
    if m:
        return json.loads(m.group(1))

    m = re.search(r"(\{.*\})", text, re.S)
    if m:
        return json.loads(m.group(1))

    raise ValueError("Could not parse JSON from model output:\n" + text)


# ---------------- Data model ----------------
@dataclass
class ClaimSearchTask:
    """A complete search task for one claim, ready to pass into tavily_search()."""

    raw_text: str
    claim: str
    query: str
    today_date: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# ---------------- Core logic ----------------
def build_search_tasks(
    raw_text: str, model_name: Optional[str] = None
) -> Tuple[List[ClaimSearchTask], str, str]:
    """
    One model call completes claim extraction, English query generation, and
    temporal window parsing.

    Returns:
        (ClaimSearchTask list, full module-1 prompt, full raw module-1 response)

    model_name:
        Optional model override. By default this reads CLAIM_EXTRACT_MODEL
        from the environment (see model_config).
    """
    today_str = date.today().isoformat()

    prompt = f"""
You are a fact-checking assistant. Given a piece of text and today's date:
1. Extract each verifiable factual claim.
2. Generate a short search-engine-friendly ENGLISH query for it.

Claim rules:
- Each claim must be about ONE distinct event or fact. If the text mentions TWO separate topics, they MUST be TWO separate claims.
- Only merge when facts describe the SAME event (e.g., "X dropped to 0.87" + "0.87 is a historic low" -> one claim).
- Do NOT include opinions, emotions, insults, or advice.
- Preserve uncertainty markers (e.g., "reportedly", "rumored", "may", "heard that").

Query rules:
- Expand abbreviations (e.g., CPF -> Central Provident Fund).
- Include key entities if implied (country/city/organization).
- Do NOT include filler words like "rumors", "issues", "policy changes".

Time rules:
- If a time reference exists, resolve it to start_date and end_date (YYYY-MM-DD) using today's date: {today_str}.
  "yesterday"      -> start_date = day before today, end_date = today
  "this week"      -> start_date = Monday of this week, end_date = Sunday of this week
  "last week"      -> start_date = Monday of last week, end_date = Sunday of last week
  specific date    -> start_date = that date - 3 days, end_date = that date + 3 days
- If no time reference, set both to null.

Output MUST be valid JSON ONLY (no markdown, no extra text).

JSON format (array with one or more objects):
[
  {{
    "claim": "...",
    "query": "...",
    "start_date": "YYYY-MM-DD or null",
    "end_date": "YYYY-MM-DD or null"
  }},
  {{
    "claim": "...",
    "query": "...",
    "start_date": "YYYY-MM-DD or null",
    "end_date": "YYYY-MM-DD or null"
  }}
]

Today's date:
{today_str}
Text:
{raw_text}
""".strip()

    model = (model_name or "").strip() or get_claim_extract_model()
    if _should_use_qwen_for_claim_extract(model):
        module1_response_text = _post_openai_compat_chat(
            base_url=get_claim_extract_base_url(),
            api_key=get_claim_extract_api_key(),
            model_name=model,
            prompt=prompt,
        )
    else:
        client = _get_client()
        resp = client.models.generate_content(model=model, contents=prompt)
        module1_response_text = getattr(resp, "text", None) or ""
    data = extract_json(module1_response_text)

    tasks: List[ClaimSearchTask] = []
    for item in data:
        claim = (item.get("claim") or "").strip()
        query = (item.get("query") or "").strip()
        if not claim or not query:
            continue
        tasks.append(
            ClaimSearchTask(
                raw_text=raw_text,
                claim=claim,
                query=query,
                today_date=today_str,
                start_date=item.get("start_date"),
                end_date=item.get("end_date"),
            )
        )
    return tasks, prompt, module1_response_text


# ---------------- Test entry point ----------------
def main():
    raw_text = "Heard that Singapore will have a lockdown yesterday, and CPF withdrawals are not possible. "
    tasks, _, _ = build_search_tasks(raw_text)
    for t in tasks:
        print(json.dumps(asdict(t), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
