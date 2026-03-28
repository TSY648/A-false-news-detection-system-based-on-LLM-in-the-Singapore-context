import os
from typing import Any, Dict, List

from sklearn.feature_extraction.text import HashingVectorizer


class HashingEmbeddingFunction:
    """Offline-safe embedding function for the local evidence store."""

    def __init__(self, n_features: int = 384):
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
        )

    def __call__(self, input: List[str]) -> List[List[float]]:
        texts = [str(item or "") for item in input]
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype("float32").tolist()

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self(input)

    @staticmethod
    def name() -> str:
        return "hashing_embedding"

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction(n_features=config.get("n_features", 384))

    def get_config(self) -> Dict[str, Any]:
        return {"n_features": self.n_features}

    @staticmethod
    def default_space() -> str:
        return "l2"

    @staticmethod
    def supported_spaces() -> List[str]:
        return ["l2", "cosine", "ip"]

    @staticmethod
    def is_legacy() -> bool:
        return False


class SafeEmbeddingFunction:
    """
    Default to a fully offline embedding function.

    If the team later wants stronger semantic embeddings, set
    USE_SENTENCE_TRANSFORMER_EMBEDDINGS=1 in the environment and ensure the
    model is already available locally.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._fallback = HashingEmbeddingFunction()
        self._primary = None

        if os.getenv("USE_SENTENCE_TRANSFORMER_EMBEDDINGS") == "1":
            try:
                from chromadb.utils import embedding_functions

                self._primary = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                )
            except Exception:
                self._primary = None

    @property
    def backend_name(self) -> str:
        return "sentence-transformer" if self._primary is not None else "hashing"

    def __call__(self, input: List[str]) -> List[List[float]]:
        texts = [str(item or "") for item in input]

        if self._primary is not None:
            try:
                return self._primary(texts)
            except Exception:
                self._primary = None

        return self._fallback(texts)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self(input)

    @staticmethod
    def name() -> str:
        return "safe_embedding"

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "SafeEmbeddingFunction":
        return SafeEmbeddingFunction(model_name=config.get("model_name", "all-MiniLM-L6-v2"))

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend_name": self.backend_name,
        }

    @staticmethod
    def default_space() -> str:
        return "l2"

    @staticmethod
    def supported_spaces() -> List[str]:
        return ["l2", "cosine", "ip"]

    @staticmethod
    def is_legacy() -> bool:
        return False
