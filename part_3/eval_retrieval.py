"""
eval_retrieval.py

Part 3 Task 10. For each test query, I hand-picked which document(s) I'd
consider "relevant" (the answer key below -- this is the same judgement
call Task 1 asked for). Precision@3/Recall@3 are computed at the DOCUMENT
level: the top-3 retrieved CHUNKS are mapped back to their parent doc_id and
deduplicated before scoring, since two chunks from the same doc should not
count as two separate hits.
"""

from retriever import retrieve_chunks, dedupe_to_documents

# query -> set of relevant doc_ids (from knowledge_base.py). 1-2 docs each,
# picked by reading the KB and judging which doc(s) actually answer the query.
ANSWER_KEY = {
    "How many days do I have to return a pair of shoes I bought?": {"return_apparel_footwear"},
    "My laptop arrived with a cracked screen, can I return it?": {"return_electronics", "damaged_item_policy"},
    "I paid cash on delivery, when will my refund hit my account?": {"cod_refund_timeline"},
    "Does someone come pick up my return or do I have to ship it myself?": {"reverse_pickup_eligibility", "reverse_pickup_unavailable"},
    "How fast is standard delivery to a metro city?": {"delivery_sla_standard"},
    "Can I exchange a t-shirt for a bigger size instead of a refund?": {"exchange_policy"},
    "Is there a warranty on electronics after Flipkart's return window closes?": {"warranty_electronics"},
}


def precision_recall_at_k(query: str, relevant_docs: set, k: int = 3):
    chunks = retrieve_chunks(query, k=k)
    retrieved_docs = {c["doc_id"] for c in dedupe_to_documents(chunks)}

    hits = retrieved_docs & relevant_docs
    precision = len(hits) / len(retrieved_docs) if retrieved_docs else 0.0
    recall = len(hits) / len(relevant_docs) if relevant_docs else 0.0
    return precision, recall, retrieved_docs, hits


def main():
    precisions, recalls = [], []
    print(f"{'query':<65} {'P@3':>6} {'R@3':>6}")
    print("-" * 80)
    for query, relevant in ANSWER_KEY.items():
        p, r, retrieved_docs, hits = precision_recall_at_k(query, relevant, k=3)
        precisions.append(p)
        recalls.append(r)
        print(f"{query[:63]:<65} {p:6.2f} {r:6.2f}")
        print(f"    relevant={sorted(relevant)}  retrieved={sorted(retrieved_docs)}  hits={sorted(hits)}")

    print("-" * 80)
    print(f"Average Precision@3: {sum(precisions)/len(precisions):.3f}")
    print(f"Average Recall@3:    {sum(recalls)/len(recalls):.3f}")


if __name__ == "__main__":
    main()
