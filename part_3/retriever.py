import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "index/faiss.index"
META_PATH = "index/chunk_meta.json"

_model = None
_index = None
_chunk_meta = None


def _lazy_load():
    global _model, _index, _chunk_meta
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
        with open(META_PATH) as f:
            _chunk_meta = json.load(f)
    return _model, _index, _chunk_meta


def retrieve_chunks(query: str, k: int = 3):
    model, index, chunk_meta = _lazy_load()
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb.astype(np.float32), k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        chunk = dict(chunk_meta[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


def dedupe_to_documents(chunks):
    best_by_doc = {}
    for c in chunks:
        doc_id = c["doc_id"]
        if doc_id not in best_by_doc or c["score"] > best_by_doc[doc_id]["score"]:
            best_by_doc[doc_id] = c
    return sorted(best_by_doc.values(), key=lambda c: c["score"], reverse=True)


if __name__ == "__main__":
    for q in ["How long do I have to return shoes?", "When do I get my COD refund?"]:
        print(f"\nQuery: {q}")
        for c in retrieve_chunks(q, k=3):
            print(f"  [{c['score']:.3f}] {c['doc_id']} :: {c['text']}")
