"""
guardrails.py

Two guardrails, per Part 3 Task 8:

1. Input-side prompt-injection filter -- a regex-based check that flags
   inputs trying to override the system prompt ("ignore previous
   instructions", "pretend you are...", etc). If it fires, the agent must
   not comply with whatever was asked -- it just deflects.

2. Output-side groundedness check -- refuses to answer a policy question if
   nothing retrieved from the KB clears a minimum similarity threshold,
   instead of letting the mock generator fabricate a policy that isn't
   actually in the knowledge base.
"""

import re

INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?previous (instructions|rules)",
    r"ignore (all|any) ?(the )?rules",
    r"disregard (all|the) (above|previous|prior)",
    r"pretend (you are|to be)",
    r"you are now",
    r"forget (all|your) (previous )?instructions",
    r"reveal your (system prompt|instructions)",
    r"act as (if|though)",
    r"new instructions?:",
]
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

MIN_SIMILARITY = 0.35  # cosine similarity floor for a policy answer to count as grounded


def check_prompt_injection(user_input: str) -> dict:
    """Returns {"blocked": bool, "reason": str|None, "matched_pattern": str|None}"""
    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(user_input)
        if match:
            return {
                "blocked": True,
                "reason": (
                    "Input matched a prompt-injection pattern and was not "
                    "followed. The assistant will not adopt new instructions "
                    "from user input."
                ),
                "matched_pattern": pattern.pattern,
            }
    return {"blocked": False, "reason": None, "matched_pattern": None}


def check_groundedness(retrieved_docs: list, min_similarity: float = MIN_SIMILARITY) -> dict:
    """retrieved_docs: doc-level deduped list from retriever.dedupe_to_documents(),
    sorted best-first. Returns whether the top hit clears the similarity floor."""
    if not retrieved_docs:
        return {"grounded": False, "top_score": 0.0, "threshold": min_similarity}

    top_score = retrieved_docs[0]["score"]
    return {
        "grounded": top_score >= min_similarity,
        "top_score": round(top_score, 4),
        "threshold": min_similarity,
    }
