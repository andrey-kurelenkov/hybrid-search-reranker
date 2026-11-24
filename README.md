# Hybrid Search & Reranking on SciFact

Two-stage retrieval pipeline (lexical + semantic retrieval with rank fusion and neural reranking) evaluated on the BEIR **SciFact** dataset.

## Architecture

```text
                        User Query
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
      Lexical Retrieval            Dense Retrieval
         (BM25Okapi)           (BAAI/bge-small-en-v1.5)
             │                        + FAISS IP
       Top-50 candidates           Top-50 candidates
             └──────────────┬──────────────┘
                            ▼
             Reciprocal Rank Fusion (k = 60)
                            │
                      Top-10 / Top-20
                            ▼
            Cross-Encoder Reranker (Planned)
                            │
                      Final Top-10
```

## Benchmark Results (SciFact Test Set)

Evaluated across 300 test queries using standard IR metrics at cutoff `k = 10`:

| Stage | Method | Recall@10 | MRR@10 | NDCG@10 | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Baseline 1 | BM25 (`BM25Okapi`) | 0.7740 | 0.6186 | 0.6519 | Exact term matching |
| Baseline 2 | Dense (`bge-small-en-v1.5` + FAISS) | 0.8296 | 0.6725 | 0.7057 | Cosine similarity via `IndexFlatIP` |
| Hybrid | **BM25 + Dense (RRF, k=60)** | **0.8332** | **0.6792** | **0.7117** | Merged from Top-50 candidate pools |
| Reranking | Hybrid + Cross-Encoder | *TBD* | *TBD* | *TBD* | Second-stage reranking |

## Implementation Details

* **Reciprocal Rank Fusion (RRF):** Used instead of raw score addition because BM25 scores are unbounded (`[0, +inf)`) while cosine similarity lies in `[-1, 1]`. RRF relies strictly on document ranks (`1 / (60 + rank)`), avoiding scale normalization and weight tuning.
* **Candidate Pool Depth:** Each retriever fetches top-50 candidates prior to fusion so documents ranking moderately well in both systems (e.g., ranks 10–20) can surface into the final top-10.
* **Embedding Caching & Prompts:** Document embeddings are precomputed and cached locally (`doc_embeddings.npy`). Queries are prefixed with the standard BGE retrieval instruction (`"Represent this sentence for searching relevant passages: "`).

## Repository Structure

```text
├── data_loader.py       # SciFact corpus, queries, and qrels loader
├── bm25_baseline.py     # Custom tokenizer and BM25 baseline evaluation
├── dense_retriever.py   # BGE-small encoding, FAISS indexing, and dense evaluation
├── hybrid_search.py     # RRF merging and comparative evaluation (Recall, MRR, NDCG)
└── requirements.txt     # Project dependencies
```

## Setup & Usage

```bash
git clone https://github.com/andrey-kurelenkov/hybrid-search-reranker
cd hybrid-search-reranker
pip install -r requirements.txt

python hybrid_search.py
```

*Note: On the first run, `hybrid_search.py` encodes the corpus and saves `doc_embeddings.npy` to disk. Subsequent runs load the cached matrix directly.*
