import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from database import EvidenceDatabase


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SEED_PATH = BASE_DIR / "local_evidence_seed.json"


class Retriever:
    def __init__(
        self,
        persist_path: Optional[str] = None,
        seed_path: Optional[str] = None,
    ):
        self.db = EvidenceDatabase(persist_path=persist_path)
        self.seed_path = Path(seed_path) if seed_path else DEFAULT_SEED_PATH

    def add_document(self, doc_id: str, text: str, metadata: Optional[dict] = None):
        self.db.upsert_document(doc_id=doc_id, text=text, metadata=metadata)

    def add_documents(self, documents: Iterable[Dict[str, Any]]) -> int:
        return self.db.upsert_documents(documents)

    def bootstrap_from_seed(self, force: bool = False) -> int:
        """
        Load a local curated corpus into the vector store.

        Replace the demo seed file with real POFMA / Factually / policy data
        when the team is ready to ingest the production corpus.
        """

        if self.db.count() > 0 and not force:
            return 0

        if not self.seed_path.exists():
            return 0

        with self.seed_path.open("r", encoding="utf-8") as fh:
            raw_documents = json.load(fh)

        prepared = []
        for item in raw_documents:
            doc_id = str(item.get("id") or item.get("doc_id") or "").strip()
            text = str(item.get("content") or item.get("text") or "").strip()

            if not doc_id or not text:
                continue

            metadata = {
                "title": item.get("title"),
                "source": item.get("source"),
                "source_type": item.get("source_type", "vector_db"),
                "published_at": item.get("published_at"),
                "tags": item.get("tags", []),
            }
            prepared.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata,
                }
            )

        return self.db.upsert_documents(prepared)

    def retrieve_by_claim(
        self,
        claim: str,
        top_k: int = 3,
        max_distance: Optional[float] = 1.2,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Use the claim to retrieve nearby evidence from the local vector store.

        `distance` is kept for debugging, while `score` is normalized for easier
        downstream use in cache-hit logic and later reranking.
        """

        raw = self.db.search(query=claim, top_k=top_k, metadata_filter=metadata_filter)

        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return []

        ids = raw["ids"][0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        results = []

        for i, doc_id in enumerate(ids):
            metadata = metas[i] if metas and i < len(metas) and metas[i] else {}
            distance = float(dists[i]) if dists and i < len(dists) else None

            if max_distance is not None and distance is not None and distance > max_distance:
                continue

            score = None if distance is None else max(0.0, 1.0 - (distance / 2.0))

            results.append(
                {
                    "id": doc_id,
                    "title": metadata.get("title"),
                    "url": metadata.get("source") or "local_db",
                    "content": docs[i] if i < len(docs) else "",
                    "distance": distance,
                    "score": score,
                    "source_type": metadata.get("source_type", "vector_db"),
                    "published_at": metadata.get("published_at"),
                    "cache_hit": True,
                }
            )

        return results
