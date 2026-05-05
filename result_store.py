import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from model_config import get_claim_extract_model, get_judge_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.environ.get(
    "FACTCHECK_DB_PATH",
    os.path.join(BASE_DIR, "factcheck_results.db"),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_run (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL UNIQUE,
                raw_text TEXT NOT NULL,
                today_date TEXT,
                claim_count INTEGER NOT NULL,
                final_label TEXT NOT NULL,
                final_reason TEXT,
                model_claim_extract TEXT,
                model_judge TEXT,
                module1_prompt TEXT,
                module1_response_text TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS claim_result (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                news_run_id INTEGER NOT NULL,
                claim_index INTEGER NOT NULL,
                claim_text TEXT NOT NULL,
                query_text TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                label TEXT NOT NULL,
                reason TEXT,
                citations_json TEXT,
                tavily_evidence_count INTEGER NOT NULL DEFAULT 0,
                vector_evidence_count INTEGER NOT NULL DEFAULT 0,
                tavily_evidence_json TEXT,
                vector_evidence_json TEXT,
                module3_prompt TEXT,
                module3_response_text TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(news_run_id) REFERENCES news_run(id)
            )
            """
        )
        # Backward-compatible migration for older local DBs.
        news_cols = {row["name"] for row in conn.execute("PRAGMA table_info(news_run)").fetchall()}
        if "module1_prompt" not in news_cols:
            conn.execute("ALTER TABLE news_run ADD COLUMN module1_prompt TEXT")
        if "module1_response_text" not in news_cols:
            conn.execute("ALTER TABLE news_run ADD COLUMN module1_response_text TEXT")

        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(claim_result)").fetchall()
        }
        if "tavily_evidence_json" not in columns:
            conn.execute("ALTER TABLE claim_result ADD COLUMN tavily_evidence_json TEXT")
        if "vector_evidence_json" not in columns:
            conn.execute("ALTER TABLE claim_result ADD COLUMN vector_evidence_json TEXT")
        if "module3_prompt" not in columns:
            conn.execute("ALTER TABLE claim_result ADD COLUMN module3_prompt TEXT")
        if "module3_response_text" not in columns:
            conn.execute("ALTER TABLE claim_result ADD COLUMN module3_response_text TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claim_result_news_run_id ON claim_result(news_run_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claim_result_label ON claim_result(label)"
        )
        conn.commit()


def persist_pipeline_result(
    result: Dict[str, Any],
    db_path: str = DEFAULT_DB_PATH,
    request_id: Optional[str] = None,
    model_claim_extract: Optional[str] = None,
    model_judge: Optional[str] = None,
) -> int:
    init_db(db_path)
    mc = model_claim_extract if model_claim_extract is not None else get_claim_extract_model()
    mj = model_judge if model_judge is not None else get_judge_model()
    rid = request_id or str(uuid.uuid4())
    created_at = _utc_now_iso()

    with get_conn(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO news_run (
                request_id, raw_text, today_date, claim_count, final_label, final_reason,
                model_claim_extract, model_judge, module1_prompt, module1_response_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                result.get("raw_text", ""),
                result.get("today_date"),
                int(result.get("claim_count", 0)),
                result.get("news_label", "Not Enough Evidence"),
                None,  # final_reason reserved; not written yet
                mc,
                mj,
                result.get("module1_prompt"),
                result.get("module1_response_text"),
                created_at,
            ),
        )
        news_run_id = int(cursor.lastrowid)

        for idx, claim in enumerate(result.get("claims", []), start=1):
            verdict = claim.get("verdict", {}) or {}
            conn.execute(
                """
                INSERT INTO claim_result (
                    news_run_id, claim_index, claim_text, query_text, start_date, end_date,
                    label, reason, citations_json, tavily_evidence_count, vector_evidence_count,
                    tavily_evidence_json, vector_evidence_json, module3_prompt, module3_response_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    news_run_id,
                    idx,
                    claim.get("claim", ""),
                    claim.get("query", ""),
                    claim.get("start_date"),
                    claim.get("end_date"),
                    verdict.get("label", "Not Enough Evidence"),
                    verdict.get("reason", ""),
                    json.dumps(verdict.get("citations", []), ensure_ascii=False),
                    int(claim.get("tavily_evidence_count", 0)),
                    int(claim.get("vector_evidence_count", 0)),
                    json.dumps(claim.get("tavily_evidence", []), ensure_ascii=False),
                    json.dumps(claim.get("vector_evidence", []), ensure_ascii=False),
                    verdict.get("module3_prompt"),
                    verdict.get("module3_response_text"),
                    created_at,
                ),
            )

        conn.commit()
        return news_run_id
