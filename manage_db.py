import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Optional

from database import EvidenceDatabase
from document_ingest import chunk_documents, load_documents_from_directory, load_documents_from_json
from retriever import Retriever


def parse_metadata(metadata_text: Optional[str], metadata_file: Optional[str]) -> Dict[str, Any]:
    if metadata_text and metadata_file:
        raise ValueError("Use either --metadata or --metadata-file, not both.")

    if metadata_text:
        try:
            return json.loads(metadata_text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(metadata_text)
            if not isinstance(parsed, dict):
                raise ValueError("--metadata must parse to an object/dict.")
            return parsed

    if metadata_file:
        path = Path(metadata_file)
        return json.loads(path.read_text(encoding="utf-8"))

    return {}


def build_common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["pinecone", "chroma"], default=None)


def cmd_add(args: argparse.Namespace) -> None:
    db = EvidenceDatabase(provider=args.provider)
    metadata = parse_metadata(args.metadata, args.metadata_file)
    db.upsert_document(doc_id=args.id, text=args.text, metadata=metadata)
    print(json.dumps({"ok": True, "action": "add", "id": args.id}, ensure_ascii=False, indent=2))


def cmd_get(args: argparse.Namespace) -> None:
    db = EvidenceDatabase(provider=args.provider)
    result = db.get_document_by_id(args.id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_list(args: argparse.Namespace) -> None:
    db = EvidenceDatabase(provider=args.provider)
    result = db.list_documents(
        limit=args.limit,
        offset=args.offset,
        prefix=args.prefix,
        pagination_token=args.pagination_token,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    db = EvidenceDatabase(provider=args.provider)
    result = db.search(query=args.query, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_update(args: argparse.Namespace) -> None:
    db = EvidenceDatabase(provider=args.provider)
    metadata = parse_metadata(args.metadata, args.metadata_file)
    db.update_document(doc_id=args.id, new_text=args.text, metadata=metadata)
    print(json.dumps({"ok": True, "action": "update", "id": args.id}, ensure_ascii=False, indent=2))


def cmd_update_metadata(args: argparse.Namespace) -> None:
    db = EvidenceDatabase(provider=args.provider)
    metadata = parse_metadata(args.metadata, args.metadata_file)
    db.update_metadata_only(doc_id=args.id, metadata=metadata)
    print(
        json.dumps(
            {"ok": True, "action": "update_metadata", "id": args.id},
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_delete(args: argparse.Namespace) -> None:
    db = EvidenceDatabase(provider=args.provider)
    db.delete_document(args.id)
    print(json.dumps({"ok": True, "action": "delete", "id": args.id}, ensure_ascii=False, indent=2))


def cmd_seed(args: argparse.Namespace) -> None:
    retriever = Retriever(seed_path=args.seed_file)
    count = retriever.bootstrap_from_seed(force=args.force)
    print(
        json.dumps(
            {"ok": True, "action": "seed", "inserted_count": count},
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_ingest_json(args: argparse.Namespace) -> None:
    db = EvidenceDatabase(provider=args.provider)
    documents = load_documents_from_json(args.input_file)
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    inserted = db.upsert_documents(chunks)
    print(
        json.dumps(
            {
                "ok": True,
                "action": "ingest_json",
                "document_count": len(documents),
                "chunk_count": len(chunks),
                "inserted_count": inserted,
                "provider": db.provider,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_ingest_dir(args: argparse.Namespace) -> None:
    db = EvidenceDatabase(provider=args.provider)
    documents = load_documents_from_directory(args.input_dir)
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    inserted = db.upsert_documents(chunks)
    print(
        json.dumps(
            {
                "ok": True,
                "action": "ingest_dir",
                "document_count": len(documents),
                "chunk_count": len(chunks),
                "inserted_count": inserted,
                "provider": db.provider,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the evidence vector database.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add or upsert one document.")
    build_common_parser(add_parser)
    add_parser.add_argument("--id", required=True)
    add_parser.add_argument("--text", required=True)
    add_parser.add_argument("--metadata")
    add_parser.add_argument("--metadata-file")
    add_parser.set_defaults(func=cmd_add)

    get_parser = subparsers.add_parser("get", help="Fetch one document by id.")
    build_common_parser(get_parser)
    get_parser.add_argument("--id", required=True)
    get_parser.set_defaults(func=cmd_get)

    list_parser = subparsers.add_parser("list", help="List documents.")
    build_common_parser(list_parser)
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--offset", type=int, default=0)
    list_parser.add_argument("--prefix")
    list_parser.add_argument("--pagination-token")
    list_parser.set_defaults(func=cmd_list)

    search_parser = subparsers.add_parser("search", help="Semantic search by query.")
    build_common_parser(search_parser)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=3)
    search_parser.set_defaults(func=cmd_search)

    update_parser = subparsers.add_parser("update", help="Replace text and metadata for one document.")
    build_common_parser(update_parser)
    update_parser.add_argument("--id", required=True)
    update_parser.add_argument("--text", required=True)
    update_parser.add_argument("--metadata")
    update_parser.add_argument("--metadata-file")
    update_parser.set_defaults(func=cmd_update)

    meta_parser = subparsers.add_parser("update-metadata", help="Update metadata only for one document.")
    build_common_parser(meta_parser)
    meta_parser.add_argument("--id", required=True)
    meta_parser.add_argument("--metadata")
    meta_parser.add_argument("--metadata-file")
    meta_parser.set_defaults(func=cmd_update_metadata)

    delete_parser = subparsers.add_parser("delete", help="Delete one document by id.")
    build_common_parser(delete_parser)
    delete_parser.add_argument("--id", required=True)
    delete_parser.set_defaults(func=cmd_delete)

    seed_parser = subparsers.add_parser("seed", help="Bootstrap the DB from the local seed file.")
    seed_parser.add_argument("--seed-file")
    seed_parser.add_argument("--force", action="store_true")
    seed_parser.set_defaults(func=cmd_seed)

    ingest_json_parser = subparsers.add_parser(
        "ingest-json",
        help="Load document JSON, chunk it, and ingest chunks into the vector DB.",
    )
    build_common_parser(ingest_json_parser)
    ingest_json_parser.add_argument("--input-file", required=True)
    ingest_json_parser.add_argument("--chunk-size", type=int, default=180)
    ingest_json_parser.add_argument("--overlap", type=int, default=30)
    ingest_json_parser.set_defaults(func=cmd_ingest_json)

    ingest_dir_parser = subparsers.add_parser(
        "ingest-dir",
        help="Load .txt/.md files from a directory, chunk them, and ingest chunks.",
    )
    build_common_parser(ingest_dir_parser)
    ingest_dir_parser.add_argument("--input-dir", required=True)
    ingest_dir_parser.add_argument("--chunk-size", type=int, default=180)
    ingest_dir_parser.add_argument("--overlap", type=int, default=30)
    ingest_dir_parser.set_defaults(func=cmd_ingest_dir)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
