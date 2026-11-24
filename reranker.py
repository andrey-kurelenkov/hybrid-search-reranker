import os
import time
import numpy as np
import faiss
from tqdm import tqdm 
from rank_bm25 import BM25Okapi
from data_loader import load_scifact_data
from sentence_transformers import SentenceTransformer, CrossEncoder
from bm25_baseline import tokenize
from hybrid_search import rrf_merge, calc_metrics

def evaluate_reranker(corpus, queries, qrels, k=10, top_n_candidates=25):
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[doc_id] for doc_id in doc_ids]
    tokenized_corpus = [tokenize(doc) for doc in doc_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    if os.path.exists("doc_embeddings.npy"):
        print("Загружаем готовые эмбеддинги с диска...")
        d_embeddings = np.load("doc_embeddings.npy")
    else:
        d_embeddings = model.encode(
            doc_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        d_embeddings = np.array(d_embeddings, dtype="float32")
        np.save("doc_embeddings.npy", d_embeddings)
    
    size = d_embeddings.shape[1]
    index = faiss.IndexFlatIP(size)
    index.add(d_embeddings)

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    q_ids = list(queries.keys())
    q_texts = [f"Represent this sentence for searching relevant passages: {queries[q_id]}" for q_id in q_ids]
    q_embeddings = model.encode(
        q_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    q_embeddings = np.array(q_embeddings, dtype="float32")
    scores, indices = index.search(q_embeddings, 50)

    recall_rerank, mrr_rerank, ndcg_rerank = [], [], []
    start_time = time.time()

    for i, q_id in enumerate(tqdm(q_ids)):
        dense_top50 = [doc_ids[idx] for idx in indices[i]]

        bm25_scores = bm25.get_scores(tokenize(queries[q_id]))
        top50_bm25_idx = np.argsort(bm25_scores)[::-1][:50]
        bm25_top50 = [doc_ids[idx] for idx in top50_bm25_idx]   

        hybrid_merged = rrf_merge(bm25_top50, dense_top50, k = 60)
        candidates = hybrid_merged[:top_n_candidates]

        cand = []
        for doc_id in candidates:
            cand.append((queries[q_id], corpus[doc_id]))

        rerank_scores = reranker.predict(cand)
        scored_directions = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        reranked_ids = [doc_id for doc_id, score in scored_directions]
        recall_rerank_cur, mrr_rerank_cur, ndcg_rerank_cur = calc_metrics(reranked_ids, qrels[q_id], k=k) 
        recall_rerank.append(recall_rerank_cur)
        mrr_rerank.append(mrr_rerank_cur)
        ndcg_rerank.append(ndcg_rerank_cur)

    total_time = time.time() - start_time
    avg_latency_ms = (total_time / len(q_ids)) * 1000

    print(f"Recall@{k}: {np.mean(recall_rerank):.4f}")
    print(f"MRR@{k}:    {np.mean(mrr_rerank):.4f}")
    print(f"NDCG@{k}:   {np.mean(ndcg_rerank):.4f}")
    print(f"Latency:   {avg_latency_ms:.2f} ms/query (Total: {total_time:.2f}s)")


if __name__ == "__main__":
    corpus, queries, qrels = load_scifact_data()
    evaluate_reranker(corpus, queries, qrels, k=10, top_n_candidates=25)
    

        
            

        

