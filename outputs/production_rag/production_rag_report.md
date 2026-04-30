# Production RAG Retrieval Evaluation

## Scope

- Baseline: vector similarity retrieval from Chroma
- Improved: keyword (TF-IDF) + embedding hybrid, score-based reranking, metadata filtering

## Before vs After (Top-K)

### Query: What is the main objective of this document?

**Baseline Top-K**

1. score=1.3751 | Copyright © 202 5 The Author(s): This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/) International Journal of Scientific Research i...
2. score=1.3976 | The ―Lost in the Middle‖ Phenomenon Liu et al. [4] demonstrated that LLMs stru ggle to access information located in the middle of long input contexts. This finding supports the ar...
3. score=1.4230 | International Journal of Scientific Research in Science, Engineering and Technology | www.ijsrset.com | Vol 13 | Issue 1 Aashish Dhakal et al Int J Sci Res Sci Eng Technol, January...
4. score=1.4385 | International Journal of Scientific Research in Science, Engineering and Technology | www.ijsrset.com | Vol 13 | Issue 1 Aashish Dhakal et al Int J Sci Res Sci Eng Technol, January...

**Improved Top-K**

1. rerank=0.5350 (hybrid=0.5050, section=objective) | METHODOLOGY A. Theoretical Framework We model the retrieval task as a Nearest Neighbor Search (NNS) problem in a vector space Rd. Let D ={d1, d2, ..., dn} be the set of document se...

### Query: List the key points discussed in the document.

**Baseline Top-K**

1. score=1.3080 | B. The ―Hybrid GEO‖ Proposal Based on this da ta, we propose a hybrid documentation standard. Technical writers should not choose between Prose or Q&A; they should layer them. Reco...
2. score=1.3528 | 1. AWS Cloud Architecture: (Topic: Cold Starts) 2. Green Computing: (Topic: Power Usage Ef- fectiveness - PUE) 3. Django Framework: (Topic: N+1 Query Prob- lem) 4. Blockchain Techn...
3. score=1.4151 | Variable Definitions 1) Standard Prose (Whole). A continuous narra - tive paragraph typical of blog posts. Indexed as a sin- gle vector. 2) Standard Prose (Chunked). The same narra...
4. score=1.5073 | International Journal of Scientific Research in Science, Engineering and Technology | www.ijsrset.com | Vol 13 | Issue 1 Aashish Dhakal et al Int J Sci Res Sci Eng Technol, January...

**Improved Top-K**

1. rerank=0.2886 (hybrid=0.2896, section=key_points) | Liu et al. (2023) [4] Context Window ―Lost in the Middle‖: LLMs recall start/end info better than middle info. Highlights the need for chunking but does not prescribe a specific fo...

### Query: What conclusions or recommendations are provided?

**Baseline Top-K**

1. score=1.5748 | Variable Definitions 1) Standard Prose (Whole). A continuous narra - tive paragraph typical of blog posts. Indexed as a sin- gle vector. 2) Standard Prose (Chunked). The same narra...
2. score=1.6994 | International Journal of Scientific Research in Science, Engineering and Technology | www.ijsrset.com | Vol 13 | Issue 1 Aashish Dhakal et al Int J Sci Res Sci Eng Technol, January...
3. score=1.7435 | B. The ―Hybrid GEO‖ Proposal Based on this da ta, we propose a hybrid documentation standard. Technical writers should not choose between Prose or Q&A; they should layer them. Reco...
4. score=1.7604 | Copyright © 202 5 The Author(s): This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/) International Journal of Scientific Research i...

**Improved Top-K**

1. rerank=0.6810 (hybrid=0.7969, section=conclusion) | B. The ―Hybrid GEO‖ Proposal Based on this da ta, we propose a hybrid documentation standard. Technical writers should not choose between Prose or Q&A; they should layer them. Reco...
2. rerank=0.6205 (hybrid=0.7162, section=conclusion) | International Journal of Scientific Research in Science, Engineering and Technology | www.ijsrset.com | Vol 13 | Issue 1 Aashish Dhakal et al Int J Sci Res Sci Eng Technol, January...
3. rerank=0.6199 (hybrid=0.7155, section=conclusion) | The Agentic Paradox: Retrieval vs. Action A nuance emerging from this study is the distinc- tion between RAG (Retrieval) and Agents (Action). While our data proves JSON is poor for...

### Query: How does chunking strategy affect retrieval quality?

**Baseline Top-K**

1. score=1.0002 | 1. AWS Cloud Architecture: (Topic: Cold Starts) 2. Green Computing: (Topic: Power Usage Ef- fectiveness - PUE) 3. Django Framework: (Topic: N+1 Query Prob- lem) 4. Blockchain Techn...
2. score=1.0011 | Copyright © 202 5 The Author(s): This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/) International Journal of Scientific Research i...
3. score=1.0030 | The ―Lost in the Middle‖ Phenomenon Liu et al. [4] demonstrated that LLMs stru ggle to access information located in the middle of long input contexts. This finding supports the ar...
4. score=1.0187 | This study investigates the impact of document structure (Standard Prose, Q&A Pairs, Fact -Dense Lists, and Structured JSON) and indexing strategies (Whole Document vs. Semantic Ch...

**Improved Top-K**

1. rerank=0.7433 (hybrid=0.8005, section=general) | Variable Definitions 1) Standard Prose (Whole). A continuous narra - tive paragraph typical of blog posts. Indexed as a sin- gle vector. 2) Standard Prose (Chunked). The same narra...
2. rerank=0.7305 (hybrid=0.8311, section=general) | 1. AWS Cloud Architecture: (Topic: Cold Starts) 2. Green Computing: (Topic: Power Usage Ef- fectiveness - PUE) 3. Django Framework: (Topic: N+1 Query Prob- lem) 4. Blockchain Techn...
3. rerank=0.7068 (hybrid=0.8471, section=key_points) | Liu et al. (2023) [4] Context Window ―Lost in the Middle‖: LLMs recall start/end info better than middle info. Highlights the need for chunking but does not prescribe a specific fo...
4. rerank=0.6908 (hybrid=0.7782, section=general) | International Journal of Scientific Research in Science, Engineering and Technology | www.ijsrset.com | Vol 13 | Issue 1 Aashish Dhakal et al Int J Sci Res Sci Eng Technol, January...

## LLM Output Comparison

### Query: What is the main objective of this document?

**Baseline Answer**

 The main objective of this document appears to be comparing Semantic Chunking and Structural Formatting in optimizing technical documentation for Retrieval-Augmented Generation (RAG) systems, with a focus on determining which method is more effective in terms of precision (Win Rate) and consistency (Mean Reciprocal Rank or MRR).

**Improved Answer**

 The main objective of this document, as stated in the context, is to find the structure type T that maximizes the relevance score S(q, di) across a distribution of queries. This is achieved by using an embedding function E and defining the relevance score as the Cosine Similarity between the query (q) and each document segment (di).

### Query: List the key points discussed in the document.

**Baseline Answer**

 The key points discussed in the document are as follows:

1. The Hybrid GEO Proposal for a hybrid documentation standard that layers natural prose and Q&A formats.
2. The implications for API Documentation, suggesting the need to generate natural language descriptions alongside schema definitions to make them AI-Ready.
3. A research study on the structure of text in relation to its vector representation and retrievability, focusing on document structures like Standard Prose (Whole/Chunked), Q&A Format, Fact-Dense List, Structured JSON, and their impact on Retrieval-Augmented Generation (RAG) systems.
4. The study's methodology, which includes creating five variations of content for each domain, analyzing the results using Win Rate (Precision) and Mean Reciprocal Rank, and comparing it with related literature.
5. The lack of empirical guidance on how to structure content for RAG systems, and the need to address this gap by conducting a comparative analysis of document structures.
6. The shift from sparse representations to dense vector representations in Information Retrieval (IR), with Sentence-BERT (SBERT) being introduced as the backbone of most modern RAG systems.

**Improved Answer**

 The key points discussed in the document by Liu et al. (2023) are as follows:

1. Long-term memory (LLMs) recall start and end information better than middle information, which is often referred to as the "Lost in the Middle" phenomenon.
2. This finding highlights the need for chunking or organizing information into smaller parts to improve memory retention.
3. However, the document does not prescribe a specific formatting standard for this chunking process.

### Query: What conclusions or recommendations are provided?

**Baseline Answer**

 The conclusions and recommendations provided are as follows:

1. Based on the win rate analysis, semantically chunked prose achieved the highest precision, confirming the hypothesis that high-granularity indexing allows the retriever to "snipe" specific queries.

2. However, in terms of consistency (Mean Reciprocal Rank or MRR), Q&A formatting emerged as the most stable structure.

3. Therefore, a hybrid documentation standard is proposed where technical writers should layer both prose and Q&A. The main body of documentation should be written in natural prose for precision, while an FAQ section should be appended at the bottom of the page to act as a "Retrieval Anchor" if the chunking strategy fails to capture nuance.

4. For API documentation, it is suggested that engineers generate natural language descriptions alongside schema definitions to make API docs more visible to RAG systems.

**Improved Answer**

 The conclusions and recommendations provided are as follows:

1. A hybrid documentation standard is proposed, where technical writers should layer natural prose and a Frequently Asked Questions (FAQ) section. The main body of the documentation should be written in natural prose, while an FAQ section should be appended at the bottom of the page to act as a Retrieval Anchor.

2. For API documentation, engineers are recommended to generate natural language descriptions alongside schema definitions to make API docs AI-Ready.

3. The paper suggests a dual-layer approach for technical documentation sites: a visible layer with natural language descriptions and Q&A for the Vector Index, and a hidden layer with JSON schemas wrapped in HTML <details> tags or separate metadata fields, which are excluded from the embedding process but passed to the LLM context once the document is retrieved.

4. Future work should investigate the role of Cross-Encoders (Re-ranking models) and the impact of Hybrid Search on these structural findings.

5. The shift to AI-mediated documentation consumption requires a shift in writing standards.

### Query: How does chunking strategy affect retrieval quality?

**Baseline Answer**

 The chunking strategy affects retrieval quality positively, as demonstrated in the study. Semantically Chunked Prose yields the highest precision (Top-1 Win Rate: 28%) and effectively "snipes" specific queries. However, it's important to note that Question-Answer formatting offers superior ranking consistency (MRR: 0.54), acting as a retrieval safety net. The optimal granularity of chunking is crucial; too small may lack context, while too large may dilute the signal.

**Improved Answer**

 The chunking strategy affects retrieval quality by optimizing for precision, particularly for specific, fact-based queries. By splitting a narrative paragraph into atomic semantic units (Chunked Standard Prose), it increases the chance of being the primary citation (Rank 1) in a retrieval system, making it more suitable for queries seeking precise information such as error codes. However, this strategy may not be ideal for broader or conceptual queries due to its focus on smaller vectors and precision.

## Insights

- TF-IDF + embedding hybrid generally improved lexical alignment for direct keyword queries.
- Metadata filtering was most useful for intent-specific questions (objective / key points / conclusion).
- Score-based reranking reduced noisy chunks that had decent vector similarity but weak query overlap.
- Retrieval still fails when source chunks do not explicitly contain requested wording.

## Tradeoff: Simplicity vs Performance

- Baseline vector retrieval is simple and fast but can miss exact keyword intent.
- Hybrid retrieval adds minimal code and gives a reliable relevance lift.
- Metadata filtering improves precision but depends on metadata quality.
- Lightweight reranking is cheaper than full cross-encoder reranking while still improving ordering.

