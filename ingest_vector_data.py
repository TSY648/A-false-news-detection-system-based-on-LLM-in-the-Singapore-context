"""
Write historical fact-check and policy documents into the Chroma vector database.

Expected input CSV fields:
source_id,title,content,source_type,url,published_at

source_id and content are required.
"""

import argparse
import csv
import os
import pickle
import shutil
from datetime import datetime
from typing import List, Optional

from runtime_config import setup_ingest_runtime_interactive
from vector_store import DEFAULT_PERSIST_DIR, ChromaEvidenceStore, SourceDocument

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(_BASE_DIR, "vector_docs.csv")
COLLECTION_NAME = "sg_factcheck_knowledge_base"
CHUNK_WORDS = 300
OVERLAP_WORDS = 50


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _backup_existing_dir(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    backup_path = f"{path}_backup_{_timestamp()}"
    shutil.move(path, backup_path)
    return backup_path


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest CSV documents into ChromaDB.")
    parser.add_argument("--csv-file", default=CSV_FILE_PATH, help="Path to the source CSV file.")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Chroma persist directory.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Backup the current Chroma directory and rebuild it from scratch.",
    )
    return parser.parse_args()


def load_docs_from_csv(path: str) -> List[SourceDocument]:
    docs: List[SourceDocument] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=2):
            source_id = (row.get("source_id") or "").strip()
            title = (row.get("title") or "").strip()
            content = (row.get("content") or "").strip()
            if not source_id or not content:
                raise ValueError(f"Invalid CSV row at line {idx}: source_id/content is required.")
            docs.append(
                SourceDocument(
                    source_id=source_id,
                    title=title or source_id,
                    content=content,
                    source_type=(row.get("source_type") or "policy_or_factcheck").strip(),
                    url=(row.get("url") or "").strip(),
                    published_at=(row.get("published_at") or "").strip(),
                )
            )
    return docs


def main():
    args = parse_args()
    model_name = setup_ingest_runtime_interactive()
    docs = load_docs_from_csv(args.csv_file)

    backup_path = None
    if args.rebuild:
        backup_path = _backup_existing_dir(args.persist_dir)

    store = ChromaEvidenceStore(
        collection_name=COLLECTION_NAME,
        persist_dir=args.persist_dir,
    )
    total_chunks = store.upsert_documents(
        docs,
        chunk_words=CHUNK_WORDS,
        overlap_words=OVERLAP_WORDS,
    )
    persisted_count = store.collection.count()
    fallback_path = os.path.join(args.persist_dir, "fallback_store.pkl")
    fallback_count = 0
    if os.path.exists(fallback_path):
        with open(fallback_path, "rb") as f:
            rows = pickle.load(f)
        if isinstance(rows, list):
            fallback_count = len(rows)

    print(f"CSV file            : {args.csv_file}")
    print(f"Persist dir         : {args.persist_dir}")
    if backup_path:
        print(f"Backup dir          : {backup_path}")
    print(f"Ingested source docs: {len(docs)}")
    print(f"Upserted chunks     : {total_chunks}")
    print(f"Persisted count     : {persisted_count}")
    print(f"Fallback rows       : {fallback_count}")
    print(f"Collection          : {COLLECTION_NAME}")
    print(f"Embedding model     : {model_name}")


if __name__ == "__main__":
    main()
