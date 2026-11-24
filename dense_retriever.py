from data_loader import load_scifact_data
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

def evaluate_dense(corpus: dict, queries: dict, qrels: dict, k: int = 10):
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[doc_id] for doc_id in doc_ids]

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    if os.path.exists("doc_embeddings.npy"):
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
    scores, indices = index.search(q_embeddings, k)

    recalls = []
    mrrs = []

    for i, q_id in enumerate(q_ids):
        retrieved_doc_ids = [doc_ids[idx] for idx in indices[i]]
        relevant_docs = qrels[q_id]

        hits = len(set(retrieved_doc_ids).intersection(relevant_docs))
        recall = hits / len(relevant_docs) 
        recalls.append(recall)


        rr = 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant_docs:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)

    print(f"Recall@{k}: {mean_recall:.4f}")
    print(f"MRR@{k}:    {mean_mrr:.4f}")

if __name__ == "__main__":
    corpus, test_queries, qrels = load_scifact_data()
    evaluate_dense(corpus, test_queries, qrels, k=10)