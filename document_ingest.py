import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 30) -> List[str]:
    """
    Split a long document into overlapping word chunks.

    chunk_size and overlap are measured in approximate words so the same logic
    works well across plain text, scraped articles, and policy documents.
    """

    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    words = cleaned.split()
    if len(words) <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            continue

        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


def make_chunk_id(document_id: str, chunk_index: int, chunk_text_value: str) -> str:
    digest = hashlib.sha1(chunk_text_value.encode("utf-8")).hexdigest()[:12]
    return f"{document_id}::chunk-{chunk_index:04d}::{digest}"


def build_chunk_metadata(
    base_metadata: Optional[Dict[str, Any]],
    *,
    document_id: str,
    chunk_index: int,
    chunk_count: int,
    chunk_text_value: str,
) -> Dict[str, Any]:
    metadata = dict(base_metadata or {})
    metadata["document_id"] = document_id
    metadata["chunk_index"] = chunk_index
    metadata["chunk_count"] = chunk_count
    metadata["char_count"] = len(chunk_text_value)
    metadata.setdefault("source_type", "document_chunk")
    return metadata


def chunk_document(
    *,
    document_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 180,
    overlap: int = 30,
) -> List[Dict[str, Any]]:
    chunks = chunk_text(text=text, chunk_size=chunk_size, overlap=overlap)
    prepared: List[Dict[str, Any]] = []

    for index, chunk in enumerate(chunks):
        prepared.append(
            {
                "id": make_chunk_id(document_id, index, chunk),
                "text": chunk,
                "metadata": build_chunk_metadata(
                    metadata,
                    document_id=document_id,
                    chunk_index=index,
                    chunk_count=len(chunks),
                    chunk_text_value=chunk,
                ),
            }
        )

    return prepared


def parse_document_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    document_id = str(raw.get("id") or raw.get("doc_id") or "").strip()
    text = str(raw.get("text") or raw.get("content") or "").strip()
    metadata = dict(raw.get("metadata") or {})

    if raw.get("title") and "title" not in metadata:
        metadata["title"] = raw["title"]
    if raw.get("source") and "source" not in metadata:
        metadata["source"] = raw["source"]
    if raw.get("source_type") and "source_type" not in metadata:
        metadata["source_type"] = raw["source_type"]
    if raw.get("published_at") and "published_at" not in metadata:
        metadata["published_at"] = raw["published_at"]
    if raw.get("tags") and "tags" not in metadata:
        metadata["tags"] = raw["tags"]

    if not document_id:
        raise ValueError("Each document record must include id or doc_id.")
    if not text:
        raise ValueError(f"Document {document_id} is missing text/content.")

    return {
        "id": document_id,
        "text": text,
        "metadata": metadata,
    }


def load_documents_from_json(path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("JSON input must be an object or a list of objects.")

    return [parse_document_record(item) for item in data]


def load_documents_from_directory(directory: str) -> List[Dict[str, Any]]:
    root = Path(directory)
    documents: List[Dict[str, Any]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md"}:
            continue

        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue

        relative_path = path.relative_to(root).as_posix()
        documents.append(
            {
                "id": relative_path.replace("/", "__"),
                "text": text,
                "metadata": {
                    "title": path.stem,
                    "source": relative_path,
                    "source_type": "file_ingest",
                },
            }
        )

    return documents


def chunk_documents(
    documents: Iterable[Dict[str, Any]],
    *,
    chunk_size: int = 180,
    overlap: int = 30,
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []

    for document in documents:
        prepared.extend(
            chunk_document(
                document_id=document["id"],
                text=document["text"],
                metadata=document.get("metadata"),
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return prepared
