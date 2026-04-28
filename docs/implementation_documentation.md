# RAG Fundamentals Implementation Documentation

## 1. Project Overview

This project implements a local Retrieval-Augmented Generation (RAG) pipeline focused on comparing chunking strategies for retrieval quality and answer quality.

The pipeline runs the same queries across:

- fixed-size chunking (`500` tokens, `0` overlap),
- overlapping chunking (`500` tokens, `125` overlap),
- semantic chunking (LangChain `SemanticChunker`).

Each strategy is indexed in a separate Chroma collection, queried with the same top-k setting, and evaluated side-by-side through saved outputs and charts.

## 2. Folder Structure

```text
rag-assignment/
├── app.py
├── config.py
├── ingestion.py
├── chunking.py
├── embeddings.py
├── vectordb.py
├── retrieval.py
├── llm.py
├── compare.py
├── visualization.py
├── requirements.txt
├── .env.example
├── sample.pdf
├── docs/
│   ├── implementation_documentation.md
│   └── assignment_deliverables_report.md
└── outputs/
    ├── retrieved_chunks/
    ├── final_answers/
    └── charts/
```

## 3. Library Choices and Rationale

- **LangChain**: orchestration layer for loaders, chunking, embeddings, vector store integration, and LLM interaction.
- **PyPDF (`PyPDFLoader`)**: reliable PDF ingestion from local file.
- **Sentence Transformers (`all-MiniLM-L6-v2`)**: compact, strong baseline embedding model for semantic retrieval.
- **ChromaDB**: persistent local vector database; easy to inspect and store per-strategy collections.
- **Ollama + Mistral**: local generation with no cloud dependency.
- **Matplotlib**: simple visualization of chunk counts and retrieval score comparisons.

## 4. Configuration and Environment

Configuration is centralized in `config.py` and `.env`.

Key environment variables:

- `SOURCE_DOCUMENT` (default `sample.pdf`)
- `TOP_K`
- `CHROMA_BASE_DIR`
- `OUTPUT_DIR`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `EMBEDDING_MODEL`
- `EMBEDDING_LOCAL_FILES_ONLY`

`get_settings()` also ensures required output folders exist before runtime.

## 5. Ingestion Flow

Implemented in `ingestion.py`:

1. Reads source path from config.
2. Chooses loader by extension:
   - `.pdf` -> `PyPDFLoader`
   - `.txt` / `.md` -> `TextLoader`
3. Filters empty pages/documents.
4. Raises explicit error when source exists but has no extractable text.

This avoids silent failures later in vector indexing.

## 6. Chunking Strategy Implementation

Implemented in `chunking.py`.

### 6.1 Fixed-size

- `TokenTextSplitter(chunk_size=500, chunk_overlap=0)`
- Prioritizes deterministic boundaries and simple indexing behavior.

### 6.2 Overlapping

- `TokenTextSplitter(chunk_size=500, chunk_overlap=125)`
- Preserves context across boundaries (25% overlap).

### 6.3 Semantic

- `SemanticChunker(embeddings=..., breakpoint_threshold_type="percentile")`
- Splits based on semantic shifts instead of rigid token count.

All strategies remove blank chunks after splitting.

## 7. Embedding Layer

Implemented in `embeddings.py`.

- Uses `HuggingFaceEmbeddings` with model `sentence-transformers/all-MiniLM-L6-v2`.
- Supports `local_files_only` mode via env when network-restricted.
- Includes clear runtime error message for model download/cache failures.

## 8. Vector Store Layer

Implemented in `vectordb.py`.

- Creates one collection per strategy (`rag_fixed`, `rag_overlap`, `rag_semantic`).
- Uses persistent directories under `chroma_store`.
- Validates chunk list is non-empty before indexing to avoid Chroma upsert errors.

## 9. Retrieval Layer

Implemented in `retrieval.py`.

- Uses `similarity_search_with_score`.
- Fetches expanded candidate set (`k*3`) and deduplicates by normalized chunk text.
- Returns unique top-k chunks to improve evaluation readability and reduce repeated contexts.

## 10. LLM Answer Generation

Implemented in `llm.py`.

- Primary path: `ChatOllama` with local Mistral model.
- Prompt constrains the model to answer from retrieved context only.
- Fallback path: if LLM call fails, returns a context-backed fallback response using top retrieved chunk. This keeps end-to-end pipeline execution resilient during local service issues.

## 11. Comparison Flow

Implemented in `compare.py` and orchestrated by `app.py`.

For each strategy:

1. Split documents
2. Build Chroma collection
3. Retrieve top-k for each query
4. Generate final answer
5. Save outputs:
   - `outputs/retrieved_chunks/<strategy>.json`
   - `outputs/final_answers/<strategy>.json`
6. Aggregate metrics:
   - `chunk_count`
   - `avg_score`
   - per-query average score

Global summary is saved at `outputs/comparison_summary.json`.

## 12. Visualization

Implemented in `visualization.py`:

- `chunk_counts.png`: chunk count per strategy
- `avg_scores.png`: average retrieval score per strategy
- `query_strategy_scores.png`: query-vs-strategy score heatmap

Backend is set to `Agg` for stable local/headless execution.

## 13. Run Instructions

From project root:

```bash
cd rag-assignment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Expected terminal output includes completion message, output directory, and per-strategy summary (chunk count + average score).

## 14. Ollama + Mistral Integration Notes

- Ensure Ollama is running: `ollama serve`
- Ensure model exists: `ollama pull mistral`
- Confirm endpoint: `curl http://localhost:11434/api/tags`
- `.env` should point to the same base URL/model.

## 15. Semantic vs Fixed Chunking (Code-Level Difference)

- Fixed chunking applies rigid token windows regardless of sentence/topic boundaries.
- Semantic chunking computes embedding-driven breakpoints and tends to keep related statements together.
- In practice, semantic often produces more chunks than fixed and can improve query-specific relevance while increasing indexing granularity.

## 16. Implementation Status

This implementation is end-to-end runnable with local PDF input, local embeddings, local vector DB persistence, local LLM generation, per-strategy output exports, and visualization artifacts for assignment evaluation.
