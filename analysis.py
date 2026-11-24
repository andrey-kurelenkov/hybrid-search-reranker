import os
import time
import numpy as np
import faiss
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from data_loader import load_scifact_data
from bm25_baseline import tokenize
from hybrid_search import rrf_merge, calc_metrics


def collect_candidates_and_scores(corpus, queries, max_candidates=50):
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[doc_id] for doc_id in doc_ids]
    tokenized_corpus = [tokenize(doc) for doc in doc_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    bi_encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    if os.path.exists("doc_embeddings.npy"):
        d_embeddings = np.load("doc_embeddings.npy")
    else:
        d_embeddings = bi_encoder.encode(
            doc_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        d_embeddings = np.array(d_embeddings, dtype="float32")
        np.save("doc_embeddings.npy", d_embeddings)

    index = faiss.IndexFlatIP(d_embeddings.shape[1])
    index.add(d_embeddings)

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    q_ids = list(queries.keys())
    q_texts = [
        f"Represent this sentence for searching relevant passages: {queries[q_id]}"
        for q_id in q_ids
    ]

    t_dense_start = time.time()
    q_embeddings = bi_encoder.encode(
        q_texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    q_embeddings = np.array(q_embeddings, dtype="float32")
    _, indices = index.search(q_embeddings, 50)
    dense_latency_ms = ((time.time() - t_dense_start) / len(q_ids)) * 1000

    hybrid_results = {}
    cached_rerank = {}
    bm25_times, rrf_times, pair_times = [], [], []

    for i, q_id in enumerate(tqdm(q_ids, desc="Collecting & Reranking Top-50")):
        dense_top50 = [doc_ids[idx] for idx in indices[i]]

        t0 = time.time()
        bm25_scores = bm25.get_scores(tokenize(queries[q_id]))
        top50_bm25_idx = np.argsort(bm25_scores)[::-1][:50]
        bm25_top50 = [doc_ids[idx] for idx in top50_bm25_idx]
        bm25_times.append((time.time() - t0) * 1000)

        t1 = time.time()
        hybrid_merged = rrf_merge(bm25_top50, dense_top50, k=60)
        rrf_times.append((time.time() - t1) * 1000)

        hybrid_results[q_id] = hybrid_merged
        candidates = hybrid_merged[:max_candidates]

        pairs = [(queries[q_id], corpus[doc_id]) for doc_id in candidates]
        t2 = time.time()
        rerank_scores = reranker.predict(pairs)
        pair_times.append(((time.time() - t2) * 1000) / len(candidates))

        cached_rerank[q_id] = list(zip(candidates, rerank_scores))

    latency_stats = {
        "bm25_ms": np.mean(bm25_times),
        "dense_ms": dense_latency_ms,
        "rrf_ms": np.mean(rrf_times),
        "per_pair_ms": np.mean(pair_times),
    }
    return hybrid_results, cached_rerank, latency_stats


def run_ablation_study(cached_rerank, qrels, latency_stats, k=10):
    print("\n" + "=" * 72)
    print("ABLATION STUDY: Reranking Pool Depth (Top-N Candidates)")
    print("=" * 72)
    print(f"Base Retrieval Latency -> BM25: {latency_stats['bm25_ms']:.2f} ms | "
          f"Dense: {latency_stats['dense_ms']:.2f} ms | "
          f"RRF: {latency_stats['rrf_ms']:.3f} ms\n")

    header = f"{'Top-N':<8} | {'Recall@10':<10} | {'MRR@10':<10} | {'NDCG@10':<10} | {'Est. Latency (ms/q)':<20}"
    print(header)
    print("-" * len(header))

    base_retrieval_ms = (
        latency_stats["bm25_ms"] + latency_stats["dense_ms"] + latency_stats["rrf_ms"]
    )

    for top_n in (10, 15, 25, 50):
        recalls, mrrs, ndcgs = [], [], []

        for q_id, scored_list in cached_rerank.items():
            subset = scored_list[:top_n]
            reranked = sorted(subset, key=lambda pair: pair[-1], reverse=True)
            reranked_ids = [doc_id for doc_id, _ in reranked]

            r, rr, n = calc_metrics(reranked_ids, qrels[q_id], k=k)
            recalls.append(r)
            mrrs.append(rr)
            ndcgs.append(n)

        est_latency = base_retrieval_ms + latency_stats["per_pair_ms"] * top_n
        print(
            f"{top_n:<8} | {np.mean(recalls):<10.4f} | {np.mean(mrrs):<10.4f} | "
            f"{np.mean(ndcgs):<10.4f} | {est_latency:<20.2f}"
        )


def run_error_analysis(corpus, queries, qrels, hybrid_results, cached_rerank, top_n=25, max_examples=5):
    print("\n" + "=" * 72)
    print(f"ERROR ANALYSIS: Cases where Hybrid RRF was #1, but Reranker (Top-{top_n}) dropped it")
    print("=" * 72)

    found = 0
    for q_id, scored_list in cached_rerank.items():
        hybrid_ids = hybrid_results[q_id]
        _, rr_hybrid, _ = calc_metrics(hybrid_ids, qrels[q_id], k=10)

        subset = scored_list[:top_n]
        reranked = sorted(subset, key=lambda pair: pair[-1], reverse=True)
        reranked_ids = [doc_id for doc_id, _ in reranked]
        _, rr_rerank, _ = calc_metrics(reranked_ids, qrels[q_id], k=10)

        if rr_hybrid == 1.0 and rr_rerank < 0.5:
            found += 1
            gold_id = next(iter(hybrid_ids))
            wrong_top1_id = next(iter(reranked_ids))

            new_rank = reranked_ids.index(gold_id) + 1 if gold_id in reranked_ids else -1

            print(f"\n[Case #{found}] Query ID: {q_id}")
            print(f"Claim: {queries[q_id]}")
            print(f"-> Gold Doc ({gold_id}) [Hybrid #1 -> Reranker #{new_rank}]: {corpus[gold_id][:240]}...")
            print(f"-> Reranker Top-1 ({wrong_top1_id}): {corpus[wrong_top1_id][:240]}...")

            if found >= max_examples:
                break


if __name__ == "__main__":
    corpus, queries, qrels = load_scifact_data()
    hybrid_results, cached_rerank, latency_stats = collect_candidates_and_scores(
        corpus, queries, max_candidates=50
    )
    run_ablation_study(cached_rerank, qrels, latency_stats, k=10)
    run_error_analysis(corpus, queries, qrels, hybrid_results, cached_rerank, top_n=25, max_examples=5)