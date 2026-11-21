import re
from data_loader import load_scifact_data
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def evaluate_bm25(corpus: dict, queries: dict, qrels: dict, k: int = 10):
    doc_ids = list(corpus.keys())
    print("Токенизируем корпус документов...")
    tokenized_corpus = [tokenize(corpus[doc_id]) for doc_id in doc_ids]

    print("Строим индекс BM25...")
    bm25 = BM25Okapi(tokenized_corpus)

    recalls = []
    mrrs = []

    print(f"Оцениваем качество поиска на {len(queries)} запросах...")
    for q_id, query_text in tqdm(queries.items()):
        tokenized_query = tokenize(query_text)
        scores = bm25.get_scores(tokenized_query)

        # Берем индексы топ-K документов с наибольшим скором
        top_k_indices = np.argsort(scores)[::-1][:k]
        retrieved_doc_ids = [doc_ids[idx] for idx in top_k_indices]

        relevant_docs = qrels[q_id]

        # 1. Считаем Recall@K
        hits = len(set(retrieved_doc_ids) & relevant_docs)
        recalls.append(hits / len(relevant_docs))

        # 2. Считаем MRR@K (обратный ранг первого релевантного документа)
        rr = 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant_docs:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    print(f"\n--- Результаты BM25 Baseline (K={k}) ---")
    print(f"Recall@{k}: {mean_recall:.4f}")
    print(f"MRR@{k}:    {mean_mrr:.4f}")


if __name__ == "__main__":
    corpus, test_queries, qrels = load_scifact_data()
    evaluate_bm25(corpus, test_queries, qrels, k=10)
