# Assignment Deliverables Report: Chunking Strategy Comparison in RAG

## 1. Scope

This submission compares three chunking strategies in a local RAG pipeline using the same source PDF, same embedding model, same retriever top-k, and same LLM.

Strategies:

1. Fixed-size chunking (`500`, no overlap)
2. Overlapping chunking (`500`, `125` overlap / 25%)
3. Semantic chunking (`SemanticChunker`)

## 2. Experimental Setup

- Source: `sample.pdf`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB: Chroma
- LLM: Ollama (`mistral`)
- Top-k: `4`
- Queries:
  - What is the main objective of this document?
  - List the key points discussed in the document.
  - What conclusions or recommendations are provided?

## 3. Quantitative Comparison

From `outputs/comparison_summary.json`:

- **Fixed**
  - chunk_count: `15`
  - avg_score: `1.4655`
- **Overlap**
  - chunk_count: `18`
  - avg_score: `1.4754`
- **Semantic**
  - chunk_count: `23`
  - avg_score: `1.4766`

Per-query average retrieval scores:

- **Main objective**
  - Fixed: `1.3509`
  - Overlap: `1.4010`
  - Semantic: `1.3986`
- **Key points**
  - Fixed: `1.3821`
  - Overlap: `1.3828`
  - Semantic: `1.3586`
- **Conclusions/recommendations**
  - Fixed: `1.6636`
  - Overlap: `1.6424`
  - Semantic: `1.6726`

## 4. Retrieved Chunks (Top-K Highlights)

Full top-k outputs for every query and strategy are saved in:

- `outputs/retrieved_chunks/fixed.json`
- `outputs/retrieved_chunks/overlap.json`
- `outputs/retrieved_chunks/semantic.json`

Representative top retrieved chunk snippets:

### Fixed

- Main objective:
  - score `1.2502`: "Additionally, the impact of Hybrid Search ..."
  - score `1.4273`: "formatting offers superior ranking consistency ..."
- Key points:
  - score `1.3268`: "impact of Hybrid Search ..."
  - score `1.3908`: "formatting offers superior ranking consistency ..."
- Conclusions:
  - score `1.5551`: "impact of Hybrid Search ..."
  - score `1.7364`: "structured data formats like JSON-LD ..."

### Overlap

- Main objective:
  - score `1.3894`: "industry lacks empirical guidance on how to structure content ..."
  - score `1.4385`: journal findings/conclusions section snippet
- Key points:
  - score `1.3407`: "context window (Top 3) ... broad conceptual queries ..."
  - score `1.3790`: "density and intent alignment ..."
- Conclusions:
  - score `1.6055`: sentence-level retriever explanation
  - score `1.6782`: "Hybrid GEO proposal ..." snippet

### Semantic

- Main objective:
  - score `1.3976`: "Lost in the Middle phenomenon ..."
  - score `1.4230`: journal findings section snippet
- Key points:
  - score `1.3080`: "Hybrid GEO proposal ..."
  - score `1.4151`: variable definitions and structure details
- Conclusions:
  - score `1.5748`: variable definitions and indexing setup
  - score `1.7435`: "Hybrid GEO proposal ..." and recommendations

## 5. Final LLM Responses by Strategy

Full responses are saved in:

- `outputs/final_answers/fixed.json`
- `outputs/final_answers/overlap.json`
- `outputs/final_answers/semantic.json`

Observed response patterns:

- **Fixed**: generally coherent, but often includes repeated metadata-heavy context in top chunks.
- **Overlap**: tends to give broader, structured summaries due to context continuity across chunk boundaries.
- **Semantic**: produces focused explanations around conceptual sections (e.g., proposals, hypotheses, conclusions).

## 6. Precision vs Recall Discussion

- **Precision behavior**:
  - Semantic and overlap retrieval slightly outperform fixed in aggregate score.
  - Semantic chunking gives sharper topic-focused retrieval for conceptual queries.
- **Recall behavior**:
  - Overlap improves continuity and helps preserve adjacent context, useful when relevant statements cross boundaries.
  - Fixed chunks are simplest but can miss adjacent supporting context.

## 7. Which Strategy Worked Best

Based on this run:

- **Semantic chunking** performed best overall by average score and in the conclusions query.
- **Overlap chunking** was strongest for objective-oriented broad query phrasing.
- **Fixed chunking** remained competitive but less consistent on nuanced sections.

## 8. Tradeoff Analysis

- **Fixed**
  - Pros: simple, fast, easy to reason about
  - Cons: rigid boundaries can break context
- **Overlap**
  - Pros: better continuity and robustness
  - Cons: extra chunks increase index size and potential redundancy
- **Semantic**
  - Pros: meaning-aware segments, strong relevance on concept-heavy queries
  - Cons: highest chunk count and heavier preprocessing

## 9. Visualization Deliverables

Generated charts:

- `outputs/charts/chunk_counts.png`
- `outputs/charts/avg_scores.png`
- `outputs/charts/query_strategy_scores.png`

These visualize chunk volume differences, overall retrieval scores, and per-query strategy behavior.

## 10. Final Conclusion

The project successfully demonstrates that chunking strategy directly affects retrieval behavior and final answer quality in RAG. For this document and query set, semantic chunking gave the best overall retrieval quality, overlap chunking improved robustness for broad questions, and fixed chunking provided a solid baseline with the lowest complexity.
