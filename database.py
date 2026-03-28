import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import chromadb
import requests

from embedder import SafeEmbeddingFunction


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PERSIST_PATH = BASE_DIR / "chroma_store"
DEFAULT_COLLECTION_NAME = "evidence_collection_module31"
DEFAULT_PROVIDER = os.getenv("VECTOR_DB_PROVIDER", "").strip().lower() or None

PINECONE_API_VERSION = os.getenv("PINECONE_API_VERSION", "2025-10")
PINECONE_CONTROL_PLANE_URL = os.getenv("PINECONE_CONTROL_PLANE_URL", "https://api.pinecone.io")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ci-evidence-index")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "module31")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "384"))
PINECONE_TIMEOUT = int(os.getenv("PINECONE_TIMEOUT_SECONDS", "30"))


class ChromaEvidenceStore:
    def __init__(
        self,
        embedding_function: SafeEmbeddingFunction,
        persist_path: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.persist_path = Path(persist_path) if persist_path else DEFAULT_PERSIST_PATH
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> int:
        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        return len(ids)

    def query(self, query: str, top_k: int = 3, metadata_filter: Optional[Dict] = None) -> dict:
        return self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=metadata_filter,
        )

    def count(self) -> int:
        return self.collection.count()

    def fetch(self, doc_id: str) -> Optional[dict]:
        result = self.collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        if not ids:
            return None

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        return {
            "id": ids[0],
            "text": documents[0] if documents else "",
            "metadata": metadatas[0] if metadatas else {},
        }

    def list_documents(self, limit: int = 100, offset: int = 0, prefix: Optional[str] = None) -> dict:
        result = self.collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        records = []
        for i, doc_id in enumerate(ids):
            if prefix and not str(doc_id).startswith(prefix):
                continue
            records.append(
                {
                    "id": doc_id,
                    "text": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
            )

        return {
            "documents": records,
            "next_offset": offset + len(ids),
            "pagination_token": None,
        }

    def update_metadata(self, doc_id: str, metadata: dict):
        self.collection.update(ids=[doc_id], metadatas=[metadata])

    def delete(self, doc_id: str):
        self.collection.delete(ids=[doc_id])


class PineconeEvidenceStore:
    """
    Minimal Pinecone REST client for dense-vector search.

    This keeps the project independent from local Pinecone SDK packaging issues
    while still using Pinecone as the cloud vector database.
    """

    def __init__(
        self,
        embedding_function: SafeEmbeddingFunction,
        index_name: str = PINECONE_INDEX_NAME,
        namespace: str = PINECONE_NAMESPACE,
        dimension: int = PINECONE_DIMENSION,
        metric: str = PINECONE_METRIC,
        cloud: str = PINECONE_CLOUD,
        region: str = PINECONE_REGION,
        timeout_seconds: int = PINECONE_TIMEOUT,
    ):
        self.embedding_function = embedding_function
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.timeout_seconds = timeout_seconds
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_host = os.getenv("PINECONE_INDEX_HOST")

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required when using Pinecone.")

        self._ensure_index()

    @property
    def control_headers(self) -> Dict[str, str]:
        return {
            "Api-Key": self.api_key,
            "Content-Type": "application/json",
            "X-Pinecone-Api-Version": PINECONE_API_VERSION,
        }

    @property
    def data_headers(self) -> Dict[str, str]:
        return self.control_headers

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[dict] = None,
        expected_statuses: tuple[int, ...] = (200,),
    ) -> dict:
        response = requests.request(
            method=method,
            url=url,
            headers=self.control_headers if url.startswith(PINECONE_CONTROL_PLANE_URL) else self.data_headers,
            json=json_body,
            timeout=self.timeout_seconds,
        )

        if response.status_code not in expected_statuses:
            raise RuntimeError(
                f"Pinecone request failed: {response.status_code} {response.text}"
            )

        if not response.text.strip():
            return {}

        return response.json()

    def _describe_index(self) -> Optional[dict]:
        url = f"{PINECONE_CONTROL_PLANE_URL}/indexes/{self.index_name}"
        response = requests.get(url, headers=self.control_headers, timeout=self.timeout_seconds)

        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to describe Pinecone index {self.index_name}: "
                f"{response.status_code} {response.text}"
            )
        return response.json()

    def _create_index(self) -> None:
        url = f"{PINECONE_CONTROL_PLANE_URL}/indexes"
        payload = {
            "name": self.index_name,
            "dimension": self.dimension,
            "metric": self.metric,
            "vector_type": "dense",
            "spec": {
                "serverless": {
                    "cloud": self.cloud,
                    "region": self.region,
                }
            },
            "deletion_protection": "disabled",
        }
        self._request("POST", url, json_body=payload, expected_statuses=(200, 201, 202))

    def _ensure_index(self) -> None:
        if self.index_host:
            return

        description = self._describe_index()
        if description is None:
            self._create_index()

        for _ in range(20):
            description = self._describe_index()
            if description and description.get("status", {}).get("ready"):
                self.index_host = description["host"]
                return
            time.sleep(2)

        raise RuntimeError(
            f"Pinecone index {self.index_name} was not ready in time. "
            "Set PINECONE_INDEX_HOST manually if the index already exists."
        )

    def _data_url(self, path: str) -> str:
        if not self.index_host:
            raise RuntimeError("Pinecone index host is not available.")
        return f"https://{self.index_host}{path}"

    def fetch(self, doc_id: str) -> Optional[dict]:
        response = requests.get(
            self._data_url("/vectors/fetch"),
            headers=self.data_headers,
            params={"ids": [doc_id], "namespace": self.namespace},
            timeout=self.timeout_seconds,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Pinecone fetch failed: {response.status_code} {response.text}"
            )

        payload = response.json()
        vectors = payload.get("vectors") or {}
        item = vectors.get(doc_id)
        if not item:
            return None

        metadata = item.get("metadata") or {}
        return {
            "id": item.get("id", doc_id),
            "text": metadata.get("text", ""),
            "metadata": metadata,
            "values": item.get("values"),
        }

    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        prefix: Optional[str] = None,
        pagination_token: Optional[str] = None,
    ) -> dict:
        params: Dict[str, Any] = {
            "namespace": self.namespace,
            "limit": limit,
        }
        if prefix:
            params["prefix"] = prefix
        if pagination_token:
            params["paginationToken"] = pagination_token

        response = requests.get(
            self._data_url("/vectors/list"),
            headers=self.data_headers,
            params=params,
            timeout=self.timeout_seconds,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Pinecone list failed: {response.status_code} {response.text}"
            )

        payload = response.json()
        vector_rows = payload.get("vectors") or []
        ids = [row.get("id") for row in vector_rows if row.get("id")]

        documents = []
        for doc_id in ids:
            item = self.fetch(doc_id)
            if item is not None:
                documents.append(item)

        next_token = (payload.get("pagination") or {}).get("next")
        return {
            "documents": documents,
            "next_offset": offset + len(documents),
            "pagination_token": next_token,
        }


    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> int:
        vectors = []
        embeddings = self.embedding_function.embed_documents(texts)

        for doc_id, text, metadata, embedding in zip(ids, texts, metadatas, embeddings):
            payload_metadata = dict(metadata)
            payload_metadata["text"] = text
            vectors.append(
                {
                    "id": doc_id,
                    "values": embedding,
                    "metadata": payload_metadata,
                }
            )

        response = self._request(
            "POST",
            self._data_url("/vectors/upsert"),
            json_body={"namespace": self.namespace, "vectors": vectors},
            expected_statuses=(200,),
        )
        return int(response.get("upsertedCount", len(vectors)))

    def query(self, query: str, top_k: int = 3, metadata_filter: Optional[Dict] = None) -> dict:
        query_vector = self.embedding_function.embed_query([query])[0]
        payload: Dict[str, Any] = {
            "namespace": self.namespace,
            "vector": query_vector,
            "topK": top_k,
            "includeValues": False,
            "includeMetadata": True,
        }
        if metadata_filter:
            payload["filter"] = metadata_filter

        response = self._request(
            "POST",
            self._data_url("/query"),
            json_body=payload,
            expected_statuses=(200,),
        )

        ids = []
        documents = []
        metadatas = []
        distances = []

        for match in response.get("matches", []):
            metadata = match.get("metadata") or {}
            score = match.get("score")
            distance = None if score is None else max(0.0, 1.0 - float(score))

            ids.append(match.get("id"))
            documents.append(metadata.get("text", ""))
            metadatas.append(metadata)
            distances.append(distance)

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def count(self) -> int:
        response = self._request(
            "POST",
            self._data_url("/describe_index_stats"),
            json_body={},
            expected_statuses=(200,),
        )
        namespaces = response.get("namespaces") or {}
        namespace_info = namespaces.get(self.namespace) or {}
        return int(
            namespace_info.get("vectorCount")
            or namespace_info.get("vector_count")
            or 0
        )

    def update_metadata(self, doc_id: str, metadata: dict):
        self._request(
            "POST",
            self._data_url("/vectors/update"),
            json_body={
                "id": doc_id,
                "namespace": self.namespace,
                "setMetadata": metadata,
            },
            expected_statuses=(200,),
        )

    def delete(self, doc_id: str):
        self._request(
            "POST",
            self._data_url("/vectors/delete"),
            json_body={"namespace": self.namespace, "ids": [doc_id]},
            expected_statuses=(200,),
        )


class EvidenceDatabase:
    def __init__(
        self,
        persist_path: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        provider: Optional[str] = None,
    ):
        self.embedding_function = SafeEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.provider = (provider or DEFAULT_PROVIDER or ("pinecone" if os.getenv("PINECONE_API_KEY") else "chroma")).lower()

        if self.provider == "pinecone":
            self.store = PineconeEvidenceStore(embedding_function=self.embedding_function)
        elif self.provider == "chroma":
            self.store = ChromaEvidenceStore(
                embedding_function=self.embedding_function,
                persist_path=persist_path,
                collection_name=collection_name,
            )
        else:
            raise ValueError(f"Unsupported vector DB provider: {self.provider}")

    @staticmethod
    def _normalize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}

        for key, value in (metadata or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = json.dumps(value, ensure_ascii=False)

        return cleaned

    def upsert_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None):
        self.store.upsert(
            ids=[doc_id],
            texts=[text],
            metadatas=[self._normalize_metadata(metadata)],
        )

    def upsert_documents(self, documents: Iterable[Dict[str, Any]]) -> int:
        ids = []
        texts = []
        metadatas = []

        for item in documents:
            doc_id = str(item.get("id") or "").strip()
            text = str(item.get("text") or "").strip()

            if not doc_id or not text:
                continue

            ids.append(doc_id)
            texts.append(text)
            metadatas.append(self._normalize_metadata(item.get("metadata")))

        if not ids:
            return 0

        return self.store.upsert(ids=ids, texts=texts, metadatas=metadatas)

    def search(self, query: str, top_k: int = 3, metadata_filter: Optional[Dict] = None):
        if not query.strip() or self.count() == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        return self.store.query(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

    def count(self) -> int:
        return self.store.count()

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.store.fetch(doc_id)

    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        prefix: Optional[str] = None,
        pagination_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.provider == "pinecone":
            return self.store.list_documents(
                limit=limit,
                offset=offset,
                prefix=prefix,
                pagination_token=pagination_token,
            )
        return self.store.list_documents(limit=limit, offset=offset, prefix=prefix)

    def update_document(self, doc_id: str, new_text: str, metadata: Optional[Dict] = None):
        self.upsert_document(doc_id=doc_id, text=new_text, metadata=metadata)

    def update_metadata_only(self, doc_id: str, metadata: Optional[Dict] = None):
        normalized = self._normalize_metadata(metadata)
        self.store.update_metadata(doc_id=doc_id, metadata=normalized)

    def delete_document(self, doc_id: str):
        self.store.delete(doc_id)
