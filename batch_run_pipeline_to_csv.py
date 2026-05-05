"""
Run the full pipeline in batch
(claim extraction -> dual-channel retrieval -> judgment -> aggregation)
and export the results to CSV.

The input CSV must contain at least one column: raw_text
"""

import csv
import json
import os
import time
import uuid
from typing import Dict, List

from pipeline_runtime_config import setup_pipeline_runtime_interactive
from test_pipeline import run_pipeline


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Edit this path directly if needed (default: same directory as this repo)
INPUT_CSV_PATH = os.path.join(_BASE_DIR, "batch_input_raw_texts.csv")
OUTPUT_CSV_PATH = os.path.join(_BASE_DIR, "batch_pipeline_results.csv")
RAW_TEXT_COLUMN = "raw_text"


OUTPUT_FIELDS = [
    "request_id",
    "status",
    "error_message",
    "elapsed_ms",
    "raw_text",
    "today_date",
    "news_label",
    "claim_count",
    "claim_index",
    "claim",
    "query",
    "start_date",
    "end_date",
    "event_date",
    "claim_label",
    "claim_reason",
    "claim_citations_json",
    "tavily_evidence_count",
    "vector_evidence_count",
    "tavily_evidence_json",
    "vector_evidence_json",
    "module1_prompt",
    "module1_response_text",
    "module3_prompt",
    "module3_response_text",
]


def load_raw_texts(csv_path: str, raw_text_column: str) -> List[str]:
    texts: List[str] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if raw_text_column not in (reader.fieldnames or []):
            raise ValueError(f"Missing required column '{raw_text_column}' in input CSV.")
        for row in reader:
            text = (row.get(raw_text_column) or "").strip()
            if text:
                texts.append(text)
    return texts


def flatten_result_to_rows(request_id: str, raw_text: str, elapsed_ms: int, result: Dict) -> List[Dict]:
    rows: List[Dict] = []
    claims = result.get("claims", [])

    if not claims:
        rows.append(
            {
                "request_id": request_id,
                "status": "success",
                "error_message": "",
                "elapsed_ms": elapsed_ms,
                "raw_text": raw_text,
                "today_date": result.get("today_date", ""),
                "news_label": result.get("news_label", ""),
                "claim_count": result.get("claim_count", 0),
                "claim_index": "",
                "claim": "",
                "query": "",
                "start_date": "",
                "end_date": "",
                "event_date": "",
                "claim_label": "",
                "claim_reason": "",
                "claim_citations_json": "[]",
                "tavily_evidence_count": 0,
                "vector_evidence_count": 0,
                "tavily_evidence_json": "[]",
                "vector_evidence_json": "[]",
                "module1_prompt": result.get("module1_prompt", ""),
                "module1_response_text": result.get("module1_response_text", ""),
                "module3_prompt": "",
                "module3_response_text": "",
            }
        )
        return rows

    for i, claim_item in enumerate(claims, start=1):
        verdict = claim_item.get("verdict", {}) or {}
        rows.append(
            {
                "request_id": request_id,
                "status": "success",
                "error_message": "",
                "elapsed_ms": elapsed_ms,
                "raw_text": raw_text,
                "today_date": result.get("today_date", ""),
                "news_label": result.get("news_label", ""),
                "claim_count": result.get("claim_count", 0),
                "claim_index": i,
                "claim": claim_item.get("claim", ""),
                "query": claim_item.get("query", ""),
                "start_date": claim_item.get("start_date", ""),
                "end_date": claim_item.get("end_date", ""),
                "event_date": claim_item.get("event_date", ""),
                "claim_label": verdict.get("label", ""),
                "claim_reason": verdict.get("reason", ""),
                "claim_citations_json": json.dumps(verdict.get("citations", []), ensure_ascii=False),
                "tavily_evidence_count": claim_item.get("tavily_evidence_count", 0),
                "vector_evidence_count": claim_item.get("vector_evidence_count", 0),
                "tavily_evidence_json": json.dumps(claim_item.get("tavily_evidence", []), ensure_ascii=False),
                "vector_evidence_json": json.dumps(claim_item.get("vector_evidence", []), ensure_ascii=False),
                "module1_prompt": result.get("module1_prompt", ""),
                "module1_response_text": result.get("module1_response_text", ""),
                "module3_prompt": (claim_item.get("verdict") or {}).get("module3_prompt", ""),
                "module3_response_text": (claim_item.get("verdict") or {}).get(
                    "module3_response_text", ""
                ),
            }
        )
    return rows


def main():
    claim_m, judge_m, emb_m = setup_pipeline_runtime_interactive()

    raw_texts = load_raw_texts(INPUT_CSV_PATH, RAW_TEXT_COLUMN)
    output_rows: List[Dict] = []

    for idx, raw_text in enumerate(raw_texts, start=1):
        request_id = str(uuid.uuid4())
        print(f"[{idx}/{len(raw_texts)}] Running pipeline request_id={request_id}")

        t0 = time.time()
        try:
            result = run_pipeline(raw_text)
            elapsed_ms = int((time.time() - t0) * 1000)
            rows = flatten_result_to_rows(request_id, raw_text, elapsed_ms, result)
            output_rows.extend(rows)
        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            output_rows.append(
                {
                    "request_id": request_id,
                    "status": "failed",
                    "error_message": str(e),
                    "elapsed_ms": elapsed_ms,
                    "raw_text": raw_text,
                    "today_date": "",
                    "news_label": "",
                    "claim_count": 0,
                    "claim_index": "",
                    "claim": "",
                    "query": "",
                    "start_date": "",
                    "end_date": "",
                    "event_date": "",
                    "claim_label": "",
                    "claim_reason": "",
                    "claim_citations_json": "[]",
                    "tavily_evidence_count": 0,
                    "vector_evidence_count": 0,
                    "tavily_evidence_json": "[]",
                    "vector_evidence_json": "[]",
                    "module1_prompt": "",
                    "module1_response_text": "",
                    "module3_prompt": "",
                    "module3_response_text": "",
                }
            )

    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Input rows          : {len(raw_texts)}")
    print(f"Output rows         : {len(output_rows)}")
    print(f"Output CSV          : {OUTPUT_CSV_PATH}")
    print(f"Claim extract model : {claim_m}")
    print(f"Judge model         : {judge_m}")
    print(f"Embedding model     : {emb_m}")


if __name__ == "__main__":
    main()
