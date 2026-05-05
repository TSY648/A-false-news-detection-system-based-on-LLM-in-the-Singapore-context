# Singapore-Context News Fact-Checking Pipeline (CI)

A multi-stage pipeline for **short-text / rumor-style news**. It uses a large language model to extract verifiable claims and search queries, searches within a **Singapore domain whitelist**, combines results with evidence from a local **vector knowledge base**, judges each claim, and finally aggregates the claim-level results into a news-level conclusion. The full prompts, raw model outputs, and evidence can be written to **SQLite** or exported to **CSV** for auditing and reproducibility.

## Pipeline Overview

| Stage | Description |
|------|------|
| **Module 1** | Gemini: extracts claims from the raw text, generates English Tavily queries, and resolves time ranges. |
| **Module 2** | **Tavily**: searches within whitelisted domains and filters results by score. **Chroma + Gemini Embedding**: retrieves semantically related evidence by claim and expands matching chunks. |
| **Module 3** | Gemini: combines both evidence streams and outputs `Supported` / `Refuted` / `Not Enough Evidence`, with reasons and citations. |
| **Aggregation** | For multiple claims: if any claim is `Refuted`, the news label is `Refuted`; otherwise, if any claim is `Not Enough Evidence`, the news label is `Not Enough Evidence`; otherwise it is `Supported`. |

The JSON printed in the terminal omits long fields such as `module1_prompt`, `module1_response_text`, `module3_prompt`, and `module3_response_text`, but **database writes and batch CSV exports keep the full text**.

## Environment Requirements

- Python **3.10+**. The development environment used Python 3.13.
- Valid APIs: **Google Gemini** for claim extraction, judging, and vector embeddings; **Tavily** for search.

## Installation

```bash
cd NTU/CI   # Or the project root after cloning
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

| Variable | Purpose |
|------|------|
| `CLAIM_EXTRACT_API_KEY` | Module 1 Gemini. |
| `JUDGE_API_KEY` | Module 3 Gemini. |
| `EMBEDDING_API_KEY` | Vector embedding for reading and writing the vector database. |
| `TAVILY_API_KEY` | Tavily search. |
| `CLAIM_EXTRACT_MODEL` / `CLAIM_JUDGE_MODEL` / `EMBEDDING_MODEL` | Optional model ID overrides. See `model_config.py`. |
| `FACTCHECK_DB_PATH` | Optional SQLite path. Defaults to `factcheck_results.db` in the project directory. |
| `CHROMA_PERSIST_DIR` | Optional Chroma persistence directory. Defaults to `chroma_db` in the project directory. |

The interactive scripts prompt for the **API keys** for each stage at runtime.

Example for one-time export:

```bash
export TAVILY_API_KEY="your_key"
export CLAIM_EXTRACT_API_KEY="..."
export JUDGE_API_KEY="..."
export EMBEDDING_API_KEY="..."
```

## Common Commands

### Single Pipeline Run

Interactive key input, with optional raw text from the command line:

```bash
python test_pipeline.py
python test_pipeline.py "Heard that Singapore will go into lockdown tomorrow"
```

After the run finishes, the result is written to SQLite if `persist_pipeline_result` is enabled. See `main` in `test_pipeline.py`.

### Batch Run: Input CSV to Output CSV

1. Prepare an input file with at least one column: **`raw_text`**. You can refer to the format of `batch_input_raw_texts.csv` in this repository.
2. If needed, edit `INPUT_CSV_PATH` / `OUTPUT_CSV_PATH` at the top of `batch_run_pipeline_to_csv.py`. By default, they point to files in the current directory.
3. Run:

```bash
python batch_run_pipeline_to_csv.py
```

The output columns include news-level fields, claim-level fields, evidence JSON, `module1_prompt`, `module1_response_text`, `module3_prompt`, `module3_response_text`, and related fields.

### Vector Database: Write CSV to Chroma

CSV columns: `source_id,title,content,source_type,url,published_at`. `source_id` and `content` are required. By default, the script reads `sample_vector_docs.csv` from the same directory.

```bash
python ingest_vector_data.py
```

## Project Structure

The following table lists the main files in this directory that are related to the pipeline.

| File | Stage | Description |
|------|------|------|
| `gemini_for_claim.py` | **Module 1** | `build_search_tasks`: uses one Gemini call to extract claims, English queries, and time ranges; returns the prompt and raw model output. |
| `req_tavily.py` | **Module 2** | `tavily_search`: runs Tavily search within the Singapore domain whitelist; reads `TAVILY_API_KEY` from environment variables. |
| `vector_store.py` | **Module 2** | `ChromaEvidenceStore`: persistent Chroma collection, Gemini Embedding writes and queries, and `search_and_expand`, which retrieves and expands chunks by claim. |
| `ingest_vector_data.py` | **Module 2 (offline)** | Batch-writes CSV rows into the vector database. It shares the same `vector_store` used by the online pipeline. This is not required for every news item; it is used to build or update the knowledge base. |
| `test_pipeline.py` | **Orchestration + Entry Point** | `run_pipeline`: calls Module 1, the two Module 2 retrieval paths, Module 3 judging, and `aggregate_news_label`. It includes Tavily/vector evidence filtering by score and distance, plus `print_results` and `main`, which writes to SQLite. |
| `batch_run_pipeline_to_csv.py` | **Orchestration + Entry Point** | Batch-reads a `raw_text` CSV, calls `run_pipeline`, flattens results into multiple rows, and writes the output CSV. |
| `claim_judge.py` | **Module 3** | `build_judge_prompt` and `judge_claim`: builds the judging prompt, calls Gemini, parses JSON labels/reasons/citations, and returns `module3_prompt` and `module3_response_text`. |
| `result_store.py` | **Persistence** | SQLite: `news_run` / `claim_result` tables, `init_db` migration, and `persist_pipeline_result`, including module prompts, raw model outputs, and evidence JSON. |
| `model_config.py` | **Configuration** | Default model IDs for each stage and environment-variable lookup for `CLAIM_EXTRACT_*`, `JUDGE_*`, and `EMBEDDING_*`, shared by Modules 1, 2, and 3. |
| `runtime_config.py` | **Configuration** | `setup_pipeline_runtime_interactive` / `setup_ingest_runtime_interactive`: interactive terminal input for keys and model names, including Tavily. |
| `requirements.txt` | **Dependencies** | Python packages such as `google-genai`, `tavily-python`, and `chromadb`. |
| `README.md` | **Documentation** | This document. |
| `batch_input_raw_texts.csv` | **Sample Data** | Batch pipeline input example. It must contain at least a `raw_text` column. |
| `sample_vector_docs.csv` | **Sample Data** | Default example for `ingest_vector_data.py`; see the vector database section above for its fields. |

## Dataset and Model Files

Due to file size limitations, the datasets and LoRA model files are provided through the following link: https://drive.google.com/drive/folders/1puIWGaDKMZBiSAeKtzGlZY5RDH3SKrZs?usp=drive_link

## License

If this project is used for coursework or research, add the required license statement according to your department or institution. This directory does not currently include an open-source license file by default.

---
