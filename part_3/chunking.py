import re
from knowledge_base import POLICY_DOCS
def split_sentences(text: str):
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
