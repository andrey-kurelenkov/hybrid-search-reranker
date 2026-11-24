import numpy as np
import math
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from data_loader import load_scifact_data
from tqdm import tqdm
import os
from bm25_baseline import tokenize
def calc_metrics(retrieved_ids, relevant_docs, k):
    hits = len(set(retrieved_ids[:k]).intersection(relevant_docs))
    recall = hits / len(relevant_docs) 

    rr = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_docs:
            rr = 1.0 / rank
            break

    ndcg_cur = compute_ndcg(retrieved_ids[:k], relevant_docs, k=k)

    return recall, rr, ndcg_cur
    
def rrf_merge(bm25_ids, dense_ids, k = 60):
    hash_table = {}
    for ind, doc_id in enumerate(bm25_ids, start=1):
        score = 1 / (k + ind)
        hash_table[doc_id] = hash_table.get(doc_id, 0.0) + score
    for ind, doc_id in enumerate(dense_ids, start=1):
            score = 1 / (k + ind)
            hash_table[doc_id] = hash_table.get(doc_id, 0.0) + score
    return sorted(hash_table, key=lambda doc_id:hash_table[doc_id], reverse=True)

def compute_ndcg(retrieved_ids: list, relevant_ids: set, k: int = 10) -> float:
    if not relevant_ids:
        return 0.0
    dcg = 0.0
    idcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_ids), k)
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(rank + 1)
    return dcg / idcg

            
def evaluate_hybrid(corpus: dict, queries: dict, qrels: dict, k: int = 10):
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

    recall_bm25 = []
    mrr_bm25 = []
    ndcg_bm25 = []
    recall_dense = []
    mrr_dense = []
    ndcg_dense = []
    recall_hybrid = []
    mrr_hybrid = []
    ndcg_hybrid = []

    for i, q_id in enumerate(tqdm(q_ids)):
        dense_top50 = [doc_ids[idx] for idx in indices[i]]

        bm25_scores = bm25.get_scores(tokenize(queries[q_id]))
        top50_bm25_idx = np.argsort(bm25_scores)[::-1][:50]
        bm25_top50 = [doc_ids[idx] for idx in top50_bm25_idx]   

        hybrid_merged = rrf_merge(bm25_top50, dense_top50, k = 60)
        hybrid_top_k = hybrid_merged[:k]

        relevant_docs = qrels[q_id]
        recall, rr, ndcg_cur = calc_metrics(bm25_top50, relevant_docs, k=k)
        recall_bm25.append(recall)
        mrr_bm25.append(rr)
        ndcg_bm25.append(ndcg_cur)
        recall, rr, ndcg_cur = calc_metrics(dense_top50, relevant_docs, k=k)
        recall_dense.append(recall)
        mrr_dense.append(rr)
        ndcg_dense.append(ndcg_cur)
        recall, rr, ndcg_cur = calc_metrics(hybrid_top_k, relevant_docs, k=k)
        recall_hybrid.append(recall)
        mrr_hybrid.append(rr)
        ndcg_hybrid.append(ndcg_cur)
    m_recall_bm25 = np.mean(recall_bm25)
    m_mrr_bm25 = np.mean(mrr_bm25)
    m_ndcg_bm25 = np.mean(ndcg_bm25)
    m_recall_dense = np.mean(recall_dense)
    m_mrr_dense = np.mean(mrr_dense)
    m_ndcg_dense = np.mean(ndcg_dense)
    m_recall_hybrid = np.mean(recall_hybrid)
    m_mrr_hybrid = np.mean(mrr_hybrid)
    m_ndcg_hybrid = np.mean(ndcg_hybrid)

    print(f"BM25   | Recall@10: {m_recall_bm25:.4f} | MRR@10: {m_mrr_bm25:.4f} | NDCG@10: {m_ndcg_bm25:.4f}")
    print(f"Dense  | Recall@10: {m_recall_dense:.4f} | MRR@10: {m_mrr_dense:.4f} | NDCG@10: {m_ndcg_dense:.4f}")
    print(f"Hybrid | Recall@10: {m_recall_hybrid:.4f} | MRR@10: {m_mrr_hybrid:.4f} | NDCG@10: {m_ndcg_hybrid:.4f}")


if __name__ == "__main__":
    corpus, test_queries, qrels = load_scifact_data()
    evaluate_hybrid(corpus, test_queries, qrels, k=10)

