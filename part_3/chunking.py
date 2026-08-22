"""
chunking.py

Splits each policy doc into one chunk per sentence. I went with sentence-wise
chunking instead of fixed-size or overlapping windows because these policy
docs are short (2-4 sentences) and each sentence is usually a single,
self-contained fact ("COD refunds take 7-9 days") -- fixed windows would
either cut a fact in half or glue two unrelated facts together, and
overlapping windows would just duplicate the same short doc three times over
for no benefit here.

Every chunk keeps parent_doc_id so retrieval and the groundedness check can
always map back up to the source document (Task 10 scores at doc level).
"""

import re

from knowledge_base import POLICY_DOCS


def split_sentences(text: str):
    # good enough for our own hand-written policy text (no abbreviations
    # like "Mr." that would trip up a naive splitter)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def build_chunks():
    """Returns a list of chunk dicts: {chunk_id, doc_id, doc_title, text}"""
    chunks = []
    for doc in POLICY_DOCS:
        sentences = split_sentences(doc["text"])
        for i, sentence in enumerate(sentences):
            chunks.append({
                "chunk_id": f"{doc['doc_id']}::{i}",
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "text": sentence,
            })
    return chunks


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"{len(POLICY_DOCS)} documents -> {len(chunks)} sentence-level chunks\n")
    for c in chunks[:6]:
        print(c["chunk_id"], "|", c["text"])
