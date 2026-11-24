from datasets import load_dataset


def load_scifact_data():
    corpus_ds = load_dataset("BeIR/scifact", "corpus")["corpus"]
    # Превращаем в словарь {doc_id: "Title. Text"}
    corpus = {
        row["_id"]: f"{row['title']}. {row['text']}".strip()
        for row in corpus_ds
    }

    queries_ds = load_dataset("BeIR/scifact", "queries")["queries"]
    queries = {row["_id"]: row["text"] for row in queries_ds}

    qrels_ds = load_dataset("BeIR/scifact-qrels")["test"]
    # Словарь {query_id: set(relevant_doc_ids)}
    qrels = {}
    for row in qrels_ds:
        if row["score"] > 0:
            q_id = str(row["query-id"])
            d_id = str(row["corpus-id"])
            qrels.setdefault(q_id, set()).add(d_id)

    # Оставляем только те запросы, для которых есть разметка в test
    test_queries = {q_id: queries[q_id] for q_id in qrels if q_id in queries}

    print(
        f"Документов: {len(corpus)}, Тестовых запросов: {len(test_queries)}"
    )
    return corpus, test_queries, qrels
