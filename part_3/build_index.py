"""
build_index.py

Embeds every KB chunk with a free local sentence-transformer model and
builds a FAISS index over them. Both are free, local, no account/API key.

Run:
    python3 build_index.py

Writes:
    index/faiss.index       -- the FAISS vector index
    index/chunk_meta.json   -- chunk_id -> {doc_id, doc_title, text}, in the
                                same row order as the FAISS index so vector i
                                maps to chunk_meta[i]

Note on the embedding model: I use sentence-transformers/all-MiniLM-L6-v2
because it's small (~80MB), fast on CPU, and is explicitly called out as a
fine default in the brief. The very first run needs internet once to
download the model weights from HuggingFace (cached locally after that,
same as torchvision downloading Fashion-MNIST once in Part 2).
"""

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

    # normalize so inner product == cosine similarity
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
