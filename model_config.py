"""
Model IDs and API keys are provided through environment variables.
Interactive scripts write into CLAIM_EXTRACT_*, CLAIM_JUDGE_*, and EMBEDDING_*.

Each stage reads only its dedicated key variables. They are not mixed or
silently reused across stages.

Model-name environment variables:
CLAIM_EXTRACT_MODEL, CLAIM_JUDGE_MODEL, EMBEDDING_MODEL
"""

import os
from typing import Optional

DEFAULT_CLAIM_EXTRACT_MODEL = "qwen-module1"
DEFAULT_CLAIM_JUDGE_MODEL = "qwen-judge"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2-preview"
DEFAULT_QWEN_CLAIM_EXTRACT_BASE_URL = "https://4rbjsqvcwwwofc-8000.proxy.runpod.net/v1"
DEFAULT_QWEN_JUDGE_BASE_URL = "https://y8zhkdn2w1vdsz-8000.proxy.runpod.net/v1"


def _first_nonempty_env(*keys: str) -> Optional[str]:
    for k in keys:
        v = os.environ.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def get_claim_extract_model() -> str:
    return _first_nonempty_env("MODULE1_MODEL", "CLAIM_EXTRACT_MODEL") or DEFAULT_CLAIM_EXTRACT_MODEL


def get_claim_extract_provider() -> str:
    return _first_nonempty_env("MODULE1_PROVIDER", "CLAIM_EXTRACT_PROVIDER") or ""


def get_claim_extract_base_url() -> str:
    return (
        _first_nonempty_env("MODULE1_BASE_URL", "CLAIM_EXTRACT_BASE_URL")
        or DEFAULT_QWEN_CLAIM_EXTRACT_BASE_URL
    )


def get_judge_model() -> str:
    return _first_nonempty_env("CLAIM_JUDGE_MODEL") or DEFAULT_CLAIM_JUDGE_MODEL


def get_embedding_model() -> str:
    return _first_nonempty_env("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


def get_claim_extract_api_key() -> str:
    k = _first_nonempty_env("MODULE1_API_KEY", "CLAIM_EXTRACT_API_KEY")
    if not k:
        raise ValueError("Missing CLAIM_EXTRACT_API_KEY.")
    return k


def get_judge_api_key() -> str:
    k = _first_nonempty_env("JUDGE_API_KEY")
    if not k:
        raise ValueError("Missing JUDGE_API_KEY.")
    return k


def get_embedding_api_key() -> str:
    k = _first_nonempty_env("EMBEDDING_API_KEY")
    if not k:
        raise ValueError("Missing EMBEDDING_API_KEY.")
    return k
