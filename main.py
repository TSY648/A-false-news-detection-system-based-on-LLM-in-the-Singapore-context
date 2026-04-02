import json

from env_config import load_project_env
from news_req_test_pipeline import run_pipeline
from retriever import Retriever


if __name__ == "__main__":
    load_project_env()

    raw_text = (
        "SINGAPORE: Singapore's total fertility rate dropped significantly "
        "to a new low of 0.87 in 2025, Deputy Prime Minister Gan Kim Yong "
        "said in parliament on Thursday (Feb 26)."
    )

    retriever = Retriever()
    seeded = retriever.bootstrap_from_seed()

    results = run_pipeline(raw_text)

    print(f"Seeded Pinecone evidence docs: {seeded}")
    print(json.dumps(results, ensure_ascii=False, indent=2))
