import os
import re
import json
import getpass
from dataclasses import dataclass, asdict
from datetime import date
from typing import Any, List, Optional

from google import genai

MODEL_NAME = "gemini-3-flash-preview"


def get_client() -> genai.Client:
    if not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = getpass.getpass("Paste GEMINI_API_KEY (hidden): ")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def get_resp_text(resp) -> str:
    """Best-effort extraction of plain text from google-genai responses."""
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t

    cands = getattr(resp, "candidates", None)
    if cands:
        try:
            parts = cands[0].content.parts
            texts = [p.text for p in parts if hasattr(p, "text") and p.text]
            if texts:
                return "\n".join(texts)
        except Exception:
            pass

    return str(resp)


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


# ── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class ClaimSearchTask:
    """一条 claim 对应的完整搜索任务，可直接传给 tavily_search。"""
    raw_text: str
    claim: str
    query: str
    today_date: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# ── Core Logic ───────────────────────────────────────────────────────────────

def build_search_tasks(raw_text: str) -> List[ClaimSearchTask]:
    """
    一次模型调用完成：claim 提取 + 英文 query 生成 + 时间区间解析。
    返回 ClaimSearchTask 列表，可直接传给 tavily_search。
    """
    client = get_client()
    today_str = date.today().isoformat()

    prompt = f"""
You are a fact-checking assistant. Given a piece of text and today's date, do TWO things for each verifiable factual claim:
1. Extract the claim.
2. Generate a search-engine-friendly ENGLISH query for it.

- MERGE closely related facts into ONE claim (e.g., "X dropped to 0.87" + "0.87 is a historic low" → one claim).
- Aim for the MINIMUM number of claims that fully cover the text. Typical: 1-3 claims.
- Do NOT include opinions, emotions, insults, or advice.
- Preserve uncertainty markers (e.g., "reportedly", "rumored", "may", "heard that").
- Expand abbreviations in the query (e.g., CPF → Central Provident Fund).
- Include key entities in the query if implied (country/city/organization).
- Generate a time reference as needed. This time reference will be used by the search engine and must cover a period of time, represented by 'start_date' and 'end_date'.
- If a time reference exists (e.g., "yesterday", "this week", "last week", a specific date), resolve it to start_date and end_date in YYYY-MM-DD format using today's date: {today_str}.
  Examples (assuming today is {today_str}):
    "yesterday"  → start_date, end_date =  today
    "this week"  → start_date = Monday of this week, end_date = Sunday of this week
    "last week"  → start_date = Monday of last week, end_date = Sunday of last week
    specific date mentioned → start_date - end_date = one week before or after that date
  If no time reference, set both to null.
- Output MUST be valid JSON ONLY (no markdown, no extra text).

JSON format:
[
  {{
    "claim": "...",
    "query": "...",
    "start_date": "YYYY-MM-DD or null",
    "end_date": "YYYY-MM-DD or null"
  }}
]

Text:
 Today's date: {today_str}
 Text:
{raw_text}
""".strip()

    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    data = extract_json(get_resp_text(resp))

    tasks: List[ClaimSearchTask] = []
    for item in data:
        claim = (item.get("claim") or "").strip()
        query = (item.get("query") or "").strip()
        if not claim or not query:
            continue
        tasks.append(ClaimSearchTask(
            raw_text=raw_text,
            claim=claim,
            query=query,
            today_date=today_str,
            start_date=item.get("start_date"),
            end_date=item.get("end_date"),
        ))
    return tasks


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    raw_text = (
        "Heard that Singapore will have a lockdown yesterday, "
        "and CPF withdrawals are not possible."
    )

    tasks = build_search_tasks(raw_text)
    for t in tasks:
        print(json.dumps(asdict(t), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
