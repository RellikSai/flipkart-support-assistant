import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from chunking import build_chunks

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "index/faiss.index"
META_PATH = "index/chunk_meta.json"


def main():
    chunks = build_chunks()
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks with {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Wrote {INDEX_PATH} ({index.ntotal} vectors, dim={dim})")
    print(f"Wrote {META_PATH}")


if __name__ == "__main__":
    main()
