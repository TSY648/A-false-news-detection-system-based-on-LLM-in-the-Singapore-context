import getpass
import os

from model_config import (
    DEFAULT_CLAIM_JUDGE_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QWEN_CLAIM_EXTRACT_BASE_URL,
    DEFAULT_QWEN_JUDGE_BASE_URL,
    get_claim_extract_model,
    get_claim_extract_provider,
)


def _prompt_model(label: str, default: str) -> str:
    value = input(f"{label} (Enter for default: {default}): ").strip()
    return value or default


def _prompt_required_key(env_name: str, label: str) -> None:
    raw = getpass.getpass(f"{label} API Key (required): ").strip()
    if not raw:
        raise ValueError(f"{label} API Key cannot be empty.")
    os.environ[env_name] = raw


def _prompt_optional_key(env_name: str, label: str, default: str = "EMPTY") -> None:
    raw = getpass.getpass(f"{label} API Key (optional): ").strip()
    os.environ[env_name] = raw or default


def _prompt_optional_value(env_name: str, label: str, default: str) -> str:
    raw = input(f"{label} (Enter for default: {default}): ").strip()
    value = raw or default
    os.environ[env_name] = value
    return value


def _is_qwen_model(model_name: str) -> bool:
    return "qwen" in (model_name or "").strip().lower()


def setup_pipeline_runtime_interactive() -> tuple[str, str, str]:
    print("--- [1/4] Claim Extract ---")
    claim_default = get_claim_extract_model()
    claim_model = _prompt_model("Claim extract model", claim_default)
    os.environ["CLAIM_EXTRACT_MODEL"] = claim_model
    os.environ["MODULE1_MODEL"] = claim_model
    claim_provider = get_claim_extract_provider()
    if claim_provider in {"qwen", "qwen_openai_compat", "openai_compat"} or _is_qwen_model(claim_model):
        os.environ["CLAIM_EXTRACT_PROVIDER"] = "qwen_openai_compat"
        os.environ["MODULE1_PROVIDER"] = "qwen_openai_compat"
        _prompt_optional_value(
            "MODULE1_BASE_URL",
            "Qwen/OpenAI-compatible claim-extract base URL",
            DEFAULT_QWEN_CLAIM_EXTRACT_BASE_URL,
        )
        _prompt_optional_key("MODULE1_API_KEY", "Claim extract")
    else:
        _prompt_required_key("CLAIM_EXTRACT_API_KEY", "Claim extract")

    print("\n--- [2/4] Judge ---")
    judge_model = _prompt_model("Judge model", DEFAULT_CLAIM_JUDGE_MODEL)
    os.environ["CLAIM_JUDGE_MODEL"] = judge_model
    os.environ["JUDGE_PROVIDER"] = "qwen_openai_compat"
    os.environ["QWEN_JUDGE_MODEL"] = judge_model
    _prompt_optional_value("JUDGE_BASE_URL", "Qwen/OpenAI-compatible base URL", DEFAULT_QWEN_JUDGE_BASE_URL)
    _prompt_optional_key("JUDGE_API_KEY", "Judge")

    print("\n--- [3/4] Embedding ---")
    embedding_model = _prompt_model("Embedding model", DEFAULT_EMBEDDING_MODEL)
    _prompt_required_key("EMBEDDING_API_KEY", "Embedding")
    os.environ["EMBEDDING_MODEL"] = embedding_model

    print("\n--- [4/4] Tavily Search ---")
    if not (os.environ.get("TAVILY_API_KEY") or "").strip():
        _prompt_required_key("TAVILY_API_KEY", "Tavily")
    else:
        print("TAVILY_API_KEY already exists; skipping input.")

    return claim_model, judge_model, embedding_model


def setup_ingest_runtime_interactive() -> str:
    print("--- Vector DB Ingest ---")
    embedding_model = _prompt_model("Embedding model", DEFAULT_EMBEDDING_MODEL)
    _prompt_required_key("EMBEDDING_API_KEY", "Embedding")
    os.environ["EMBEDDING_MODEL"] = embedding_model
    return embedding_model
