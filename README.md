# Singapore Fact-Checking Pipeline

## 1. Project Overview

This project is a fact-checking pipeline for Singapore-related claims. It is designed to:

1. accept a user's raw query or long text,
2. use Gemini to split it into atomic factual claims,
3. generate a search-friendly English query for each claim,
4. retrieve evidence from two channels:
   - Pinecone vector database
   - Tavily web search
5. rerank the two channels separately,
6. keep the top 3 evidence items from each channel,
7. output evidence in a format that can be passed to a downstream LLM for final reasoning.

The current repository focuses on the evidence preparation stage. The final verdict-generation step can be built on top of the existing `evidence_for_llm` output.

## 2. Current Pipeline

The implemented workflow in this repo is:

`raw_text -> Gemini claim extraction -> Pinecone retrieval + Tavily retrieval -> separate reranking -> top 3 from Pinecone + top 3 from Tavily -> evidence_for_llm`

Important design choices:

- Pinecone is the only vector database backend.
- Vector retrieval uses `claim`, not `query`.
- Tavily retrieval uses the English `query`.
- Pinecone evidence and Tavily evidence are reranked separately.
- The final combined evidence list contains up to 6 items.
- The downstream LLM evidence format is unified to:

```json
{
  "title": "...",
  "content": "..."
}
```

## 3. Repository Structure

Most of the actual system code lives in [System](/d:/桌面/CI/System).

Key files:

- [System/news_req_gemini_for_claim.py](/d:/桌面/CI/System/news_req_gemini_for_claim.py)  
  Gemini-based claim extraction and English query generation.

- [System/news_req_req_tavily.py](/d:/桌面/CI/System/news_req_req_tavily.py)  
  Tavily search wrapper with Singapore-focused domain whitelist.

- [System/database.py](/d:/桌面/CI/System/database.py)  
  Pinecone REST client and unified database access layer.

- [System/retriever.py](/d:/桌面/CI/System/retriever.py)  
  Converts Pinecone search output into the project's evidence format.

- [System/reranker.py](/d:/桌面/CI/System/reranker.py)  
  Relevance scoring, deduplication, noise filtering, and top-k reranking.

- [System/news_req_test_pipeline.py](/d:/桌面/CI/System/news_req_test_pipeline.py)  
  Main orchestration pipeline.

- [System/document_ingest.py](/d:/桌面/CI/System/document_ingest.py)  
  Document chunking and ingestion preprocessing.

- [System/manage_db.py](/d:/桌面/CI/System/manage_db.py)  
  CLI for Pinecone CRUD, search, and bulk ingestion.

- [System/embedder.py](/d:/桌面/CI/System/embedder.py)  
  Embedding layer. Defaults to offline-safe hashing embeddings.

- [System/local_evidence_seed.json](/d:/桌面/CI/System/local_evidence_seed.json)  
  Demo seed data. Useful for smoke tests, but should be replaced by real curated data later.

- [System/main.py](/d:/桌面/CI/System/main.py)  
  Minimal demo entry point.

- [System/env_config.py](/d:/桌面/CI/System/env_config.py)  
  Auto-loads local `.env` / `pinecone.env` files for Pinecone configuration.

## 4. Data Flow

### 4.1 Gemini Output

For each input text, Gemini produces one or more tasks like:

```json
{
  "raw_text": "...",
  "claim": "...",
  "query": "...",
  "today_date": "2026-04-03",
  "start_date": "2026-04-01",
  "end_date": "2026-04-02"
}
```

Meaning of each field:

- `claim`: used for Pinecone semantic retrieval
- `query`: used for Tavily web search
- `start_date` / `end_date`: optional date constraints for search

### 4.2 Pinecone Retrieval Input

Pinecone matching uses:

```python
retriever.retrieve_by_claim(task.claim, top_k=5)
```

So the direct query input to the vector database is a single string:

```json
"claim text"
```

That string is embedded and sent to Pinecone as a query vector.

### 4.3 Pinecone Retrieval Output

The project-level evidence format returned by [System/retriever.py](/d:/桌面/CI/System/retriever.py) is:

```json
[
  {
    "id": "...",
    "title": "...",
    "url": "...",
    "content": "...",
    "distance": 0.92,
    "score": 0.54,
    "source_type": "...",
    "published_at": "...",
    "cache_hit": true
  }
]
```

### 4.4 Tavily Retrieval Output

The Tavily evidence format is:

```json
[
  {
    "title": "...",
    "url": "...",
    "content": "...",
    "score": 0.99
  }
]
```

### 4.5 Final LLM Input Evidence

Although the internal retrieval format keeps more fields for ranking and debugging, the pipeline also produces a simplified LLM-ready format:

```json
[
  {
    "title": "...",
    "content": "..."
  }
]
```

This is available as `evidence_for_llm` in the pipeline output.

## 5. Reranking Strategy

The project does not mix Pinecone and Tavily evidence into a single ranking pool anymore.

Current behavior:

1. retrieve Pinecone candidates,
2. retrieve Tavily candidates,
3. rerank Pinecone evidence separately,
4. rerank Tavily evidence separately,
5. keep top 3 from Pinecone,
6. keep top 3 from Tavily,
7. concatenate both lists.

Final evidence count:

- minimum: `0`
- maximum: `6`

The reranker currently considers:

- base retrieval score
- lexical overlap with `claim`
- lexical overlap with `query`
- source quality bonus
- content quality
- deduplication
- weak-match filtering

## 6. Environment Setup

### 6.1 Python Environment

The repo has been used with the existing virtual environment in:

- `D:\桌面\CI\rag_venv`

From the project root:

```powershell
.\rag_venv\Scripts\Activate.ps1
```

or run commands directly with:

```powershell
.\rag_venv\Scripts\python.exe ...
```

### 6.2 Pinecone Configuration

Pinecone is loaded from a local `.env` file. The system automatically checks:

- `D:\桌面\CI\.env`
- `D:\桌面\CI\System\.env`
- `D:\桌面\CI\pinecone.env`
- `D:\桌面\CI\System\pinecone.env`

Recommended location:

- `D:\桌面\CI\.env`

Example:

```env
PINECONE_API_KEY=your_real_pinecone_api_key
PINECONE_INDEX_NAME=ci-evidence-index
PINECONE_NAMESPACE=module31
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_DIMENSION=384
PINECONE_METRIC=cosine
```

Notes:

- `PINECONE_INDEX_NAME` must use lower-case letters, numbers, or `-`.
- Do not use `_` in Pinecone index names.
- If `PINECONE_INDEX_HOST` is known, you can add it to skip describe/poll.

### 6.3 Gemini and Tavily Configuration

This repo currently keeps the existing team logic for Gemini and Tavily:

- Gemini: if `GEMINI_API_KEY` is not set, the script prompts for it at runtime.
- Tavily: current key handling remains in [System/news_req_req_tavily.py](/d:/桌面/CI/System/news_req_req_tavily.py).

If your team later wants to centralize all keys in `.env`, that can be added separately.

## 7. Quick Start

### 7.1 Run the Demo Pipeline

From `D:\桌面\CI\System`:

```powershell
python main.py
```

Or from the project root:

```powershell
.\rag_venv\Scripts\python.exe System\main.py
```

What `main.py` currently does:

- initializes the Pinecone retriever,
- optionally seeds demo evidence from [System/local_evidence_seed.json](/d:/桌面/CI/System/local_evidence_seed.json),
- runs the full pipeline on one demo news sentence,
- prints the JSON result.

### 7.2 Run the Full Test Pipeline with Custom Input

From `D:\桌面\CI\System`:

```powershell
python news_req_test_pipeline.py "Heard that Singapore will have a lockdown yesterday, and CPF withdrawals are not possible."
```

## 8. Bulk Ingestion into Pinecone

### 8.1 Supported Ingestion Paths

The project supports two main ingestion paths:

1. JSON document ingestion
2. directory-based `.txt` / `.md` ingestion

The ingestion flow is:

`raw document -> chunking -> embedding -> upsert to Pinecone`

### 8.2 JSON Input Format

Expected JSON shape:

```json
[
  {
    "id": "doc-001",
    "title": "MOH advisory",
    "source": "https://www.moh.gov.sg/example",
    "source_type": "policy_doc",
    "published_at": "2026-04-01",
    "tags": ["moh", "policy"],
    "content": "Full document text goes here."
  }
]
```

`id` and `content` are required. Other fields are optional but recommended.

### 8.3 Ingest JSON into Pinecone

From `D:\桌面\CI\System`:

```powershell
python manage_db.py ingest-json --input-file your_docs.json
```

With custom chunk settings:

```powershell
python manage_db.py ingest-json --input-file your_docs.json --chunk-size 180 --overlap 30
```

### 8.4 Ingest a Directory into Pinecone

```powershell
python manage_db.py ingest-dir --input-dir your_docs_folder
```

This recursively loads `.txt` and `.md` files, chunks them, and upserts them into Pinecone.

## 9. Pinecone CRUD and Search Commands

All commands below run from `D:\桌面\CI\System`.

### Add One Record

```powershell
python manage_db.py add --id doc-1 --text "test document" --metadata "{'title':'Test','source_type':'demo'}"
```

### Get One Record

```powershell
python manage_db.py get --id doc-1
```

### List Records

```powershell
python manage_db.py list --limit 10
```

### Semantic Search

```powershell
python manage_db.py search --query "Singapore fertility rate 0.87" --top-k 3
```

### Update One Record

```powershell
python manage_db.py update --id doc-1 --text "updated text" --metadata "{'title':'Updated'}"
```

### Update Metadata Only

```powershell
python manage_db.py update-metadata --id doc-1 --metadata "{'title':'Updated title only'}"
```

### Delete One Record

```powershell
python manage_db.py delete --id doc-1
```

### Seed Demo Data

```powershell
python manage_db.py seed
```

Use `--force` to seed again even if the namespace already has data:

```powershell
python manage_db.py seed --force
```

## 10. Output of `run_pipeline`

The main pipeline returns a list of per-claim results. A simplified example:

```json
[
  {
    "raw_text": "...",
    "claim": "...",
    "query": "...",
    "today_date": "2026-04-03",
    "start_date": "2026-04-01",
    "end_date": "2026-04-02",
    "cache_hit": true,
    "pinecone_evidence_count": 5,
    "web_evidence_count": 5,
    "merged_evidence_count": 10,
    "pinecone_top_evidence_count": 3,
    "web_top_evidence_count": 3,
    "evidence_count": 6,
    "pinecone_evidence": [],
    "web_evidence": [],
    "merged_evidence": [],
    "pinecone_top_evidence": [],
    "web_top_evidence": [],
    "evidence": [],
    "pinecone_top_evidence_for_llm": [],
    "web_top_evidence_for_llm": [],
    "evidence_for_llm": []
  }
]
```

Recommended downstream usage:

- Use `pinecone_top_evidence` and `web_top_evidence` for debugging and analysis.
- Use `evidence_for_llm` as the direct evidence payload for the final reasoning model.

## 11. Suggested Downstream LLM Input

A downstream reasoning module can consume the output like this:

```json
{
  "prompt": "As a Singapore fact-checking assistant, determine whether each claim is true based on the evidence.",
  "input": {
    "claims": [
      {
        "claim": "CPF withdrawals are not possible",
        "event_date": "",
        "evidences": [
          {
            "title": "CPF withdrawal rules 2026",
            "content": "Members can withdraw their CPF savings at age 55..."
          },
          {
            "title": "Official FAQ",
            "content": "CPF withdrawals are subject to specific age and account rules..."
          }
        ]
      }
    ]
  }
}
```

## 12. Current Limitations

- The seed data in [System/local_evidence_seed.json](/d:/桌面/CI/System/local_evidence_seed.json) is demo-only.
- Real curated sources such as POFMA, Factually, MOH, CPF Board, and other official policy documents still need to be ingested into Pinecone.
- The final LLM verdict-generation module is not yet part of this repo's main pipeline.
- The default embedding backend is hashing-based for offline safety; it is stable and lightweight, but not as semantically strong as a fully loaded sentence-transformer model.
- You may still see Python 3.9 end-of-life warnings from Google libraries. These are warnings, not immediate pipeline failures.

## 13. Recommended Next Steps

1. Replace demo seed data with real curated evidence.
2. Ingest official and media documents into Pinecone.
3. Add the final verdict-generation module that consumes `evidence_for_llm`.
4. Optionally centralize Gemini and Tavily key management if the team wants a single `.env` workflow.
5. Upgrade from Python 3.9 when convenient.

## 14. Ownership Summary

If you only need to understand the system quickly, focus on these files first:

- [System/news_req_test_pipeline.py](/d:/桌面/CI/System/news_req_test_pipeline.py)  
  Main data flow.

- [System/news_req_gemini_for_claim.py](/d:/桌面/CI/System/news_req_gemini_for_claim.py)  
  Claim and query generation.

- [System/retriever.py](/d:/桌面/CI/System/retriever.py)  
  Pinecone evidence retrieval.

- [System/reranker.py](/d:/桌面/CI/System/reranker.py)  
  Evidence ranking and filtering.

- [System/manage_db.py](/d:/桌面/CI/System/manage_db.py)  
  Data ingestion and database operations.
