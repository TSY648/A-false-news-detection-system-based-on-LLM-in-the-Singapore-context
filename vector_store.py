import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from google import genai

from model_config import get_embedding_api_key, get_embedding_model


DEFAULT_COLLECTION_NAME = "sg_factcheck_knowledge_base"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", os.path.join(BASE_DIR, "chroma_db"))


@dataclass
class SourceDocument:
    source_id: str
    title: str
    content: str
    source_type: str = "policy_or_factcheck"
    url: str = ""
    published_at: str = ""


def build_factcheck_query_text(content: str) -> str:
    """
    Asymmetric text format (query side):
    task: fact checking | query: {content}
    """
    c = " ".join((content or "").split())
    return f"task: fact checking | query: {c}"


def build_factcheck_document_text(title: str, content: str) -> str:
    """
    Asymmetric text format (document side):
    title: {title} | text: {content}
    """
    t = " ".join((title or "").split()) or "none"
    c = " ".join((content or "").split())
    return f"title: {t} | text: {c}"


def split_text(text: str, chunk_words: int = 300, overlap_words: int = 50) -> List[str]:
    clean = " ".join(text.split())
    if not clean:
        return []
    words = clean.split(" ")
    if len(words) <= chunk_words:
        return [clean]

    chunks: List[str] = []
    step = max(1, chunk_words - overlap_words)
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step
    return chunks


class GeminiEmbeddingClient:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        key = api_key or get_embedding_api_key()
        self.client = genai.Client(api_key=key)
        self.model_name = (model_name or "").strip() or get_embedding_model()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            items = getattr(response, "embeddings", []) or []
            if not items:
                raise RuntimeError("Embedding response has no embeddings.")
            item = items[0]
            values = getattr(item, "values", None)
            if values is None and isinstance(item, dict):
                values = item.get("values")
            vectors.append(values or [])
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding count does not match input text count.")
        return vectors


class ChromaEvidenceStore:
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        embedding_client: Optional[GeminiEmbeddingClient] = None,
    ):
        self.embedding_client = embedding_client or GeminiEmbeddingClient()
        self.persist_dir = persist_dir
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _fallback_store_path(self) -> str:
        return os.path.join(self.persist_dir, "fallback_store.pkl")

    def _load_fallback_rows(self) -> List[Dict[str, Any]]:
        path = self._fallback_store_path()
        if not os.path.exists(path):
            return []
        with open(path, "rb") as f:
            rows = pickle.load(f)
        return rows if isinstance(rows, list) else []

    def _save_fallback_rows(self, rows: List[Dict[str, Any]]) -> None:
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(self._fallback_store_path(), "wb") as f:
            pickle.dump(rows, f)

    def _upsert_fallback_rows(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        existing = self._load_fallback_rows()
        by_id: Dict[str, Dict[str, Any]] = {
            row["id"]: row for row in existing if isinstance(row, dict) and "id" in row
        }
        for row_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            by_id[row_id] = {
                "id": row_id,
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }
        self._save_fallback_rows(list(by_id.values()))

    @staticmethod
    def _cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 1.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0
        cosine_sim = dot / (norm_a * norm_b)
        cosine_sim = max(-1.0, min(1.0, cosine_sim))
        return 1.0 - cosine_sim

    def _fallback_query(self, query_embedding: List[float], n_results: int) -> Dict[str, List[List[Any]]]:
        rows = self._load_fallback_rows()
        if not rows:
            raise RuntimeError(
                "Current ChromaDB persist directory is corrupted and fallback store is missing. "
                "Please rebuild it with: python ingest_vector_data.py --rebuild"
            )

        scored_rows = []
        for row in rows:
            doc_text = row.get("document", "")
            meta = row.get("metadata", {}) or {}
            embedding = row.get("embedding", []) or []
            if not embedding:
                continue
            distance = self._cosine_distance(query_embedding, embedding)
            scored_rows.append((distance, doc_text, meta))

        scored_rows.sort(key=lambda item: item[0])
        top_rows = scored_rows[:n_results]
        return {
            "documents": [[row[1] for row in top_rows]],
            "metadatas": [[row[2] for row in top_rows]],
            "distances": [[row[0] for row in top_rows]],
        }

    def upsert_documents(self, docs: List[SourceDocument], chunk_words: int = 300, overlap_words: int = 50) -> int:
        all_ids: List[str] = []
        all_documents: List[str] = []
        all_metadatas: List[Dict[str, Any]] = []

        for doc in docs:
            chunks = split_text(doc.content, chunk_words=chunk_words, overlap_words=overlap_words)
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{doc.source_id}::chunk::{idx}"
                all_ids.append(chunk_id)
                embedding_text = build_factcheck_document_text(title=doc.title, content=chunk)
                all_documents.append(embedding_text)
                all_metadatas.append(
                    {
                        "source_id": doc.source_id,
                        "title": doc.title,
                        "url": doc.url,
                        "source_type": doc.source_type,
                        "published_at": doc.published_at,
                        "chunk_index": idx,
                        "raw_chunk": chunk,
                    }
                )

        if not all_documents:
            return 0

        embeddings = self.embedding_client.embed_texts(all_documents)
        self._upsert_fallback_rows(all_ids, all_documents, all_metadatas, embeddings)
        self.collection.upsert(
            ids=all_ids,
            documents=all_documents,
            metadatas=all_metadatas,
            embeddings=embeddings,
        )
        return len(all_ids)

    def search_and_expand(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        query_text = build_factcheck_query_text(content=query)
        query_embedding = self.embedding_client.embed_texts([query_text])[0]
        use_fallback_for_expand = False
        try:
            res = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            message = str(exc).lower()
            if "hnsw" not in message and "index" not in message:
                raise
            print(
                "Warning: Chroma HNSW index unavailable. "
                "Falling back to brute-force vector scan. "
                "If this keeps happening, rebuild the DB with: "
                "python ingest_vector_data.py --rebuild"
            )
            res = self._fallback_query(query_embedding=query_embedding, n_results=n_results)
            use_fallback_for_expand = True

        metadatas = (res.get("metadatas") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]

        if not metadatas:
            return []

        matched_by_source: Dict[str, Dict[str, Any]] = {}
        for meta, doc_text, distance in zip(metadatas, documents, distances):
            source_id = meta.get("source_id", "")
            if not source_id:
                continue
            bucket = matched_by_source.setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "title": meta.get("title", ""),
                    "url": meta.get("url", ""),
                    "source_type": meta.get("source_type", ""),
                    "published_at": meta.get("published_at", ""),
                    "matched_chunks": [],
                },
            )
            bucket["matched_chunks"].append(
                {
                    "chunk_index": meta.get("chunk_index", -1),
                    "distance": float(distance) if distance is not None else None,
                    "content": meta.get("raw_chunk", doc_text),
                }
            )

        expanded: List[Dict[str, Any]] = []
        fallback_rows = self._load_fallback_rows() if use_fallback_for_expand else []
        for source_id, summary in matched_by_source.items():
            if use_fallback_for_expand:
                rows = [
                    (row.get("metadata", {}) or {}, row.get("document", ""))
                    for row in fallback_rows
                    if (row.get("metadata", {}) or {}).get("source_id") == source_id
                ]
            else:
                all_chunks = self.collection.get(
                    where={"source_id": source_id},
                    include=["documents", "metadatas"],
                )
                rows = list(zip(all_chunks.get("metadatas", []), all_chunks.get("documents", [])))
            rows.sort(key=lambda x: x[0].get("chunk_index", 0))
            full_text = "\n".join([m.get("raw_chunk", d) for m, d in rows])
            summary["full_text"] = full_text
            summary["chunk_count"] = len(rows)
            expanded.append(summary)

        return expanded
