"""
mock_llm.py

The default, required mode (Task 6). Zero network calls, zero API keys.
Given retrieved KB chunks and/or tool output, deterministically composes the
final structured answer. A live-LLM path can be swapped in later behind
USE_LIVE_LLM=1 (see agent_graph.py) but every graded transcript runs against
this file.
"""

import re

from prompts import INTENT_FEWSHOT

ORDER_ID_PATTERN = re.compile(r"order\s*#?\s*(\d{3,})", re.IGNORECASE)
IMAGE_PATH_PATTERN = re.compile(r"[\w./\\-]+\.(?:png|jpg|jpeg)", re.IGNORECASE)


def _word_overlap(a: str, b: str) -> int:
    return len(set(a.split()) & set(b.split()))


def classify_intent(user_input: str) -> str:
    """Rule-based intent router. Checks the sharpest signals first (an order
    number -> return_risk, an image path -> product_category), then falls
    back to nearest-neighbor keyword overlap against the few-shot examples
    from prompts.INTENT_FEWSHOT, and only defaults to "policy" if nothing
    matches at all."""
    lower = user_input.lower()

    if IMAGE_PATH_PATTERN.search(user_input) or (
        "category" in lower and ("photo" in lower or "image" in lower or "picture" in lower)
    ):
        return "product_category"

    if ORDER_ID_PATTERN.search(user_input) or "return risk" in lower or "likely to be returned" in lower:
        return "return_risk"

    # few-shot nearest-neighbor fallback
    best_intent, best_score = "policy", 0
    for ex in INTENT_FEWSHOT:
        score = _word_overlap(lower, ex["query"].lower())
        if score > best_score:
            best_score, best_intent = score, ex["intent"]
    return best_intent


def extract_order_id(user_input: str):
    m = ORDER_ID_PATTERN.search(user_input)
    return m.group(1) if m else None


def extract_image_path(user_input: str):
    m = IMAGE_PATH_PATTERN.search(user_input)
    return m.group(0) if m else None


def compose_policy_answer(retrieved_docs: list) -> dict:
    """retrieved_docs: doc-level deduped, best-first, already passed the
    groundedness check by the time this is called."""
    top = retrieved_docs[0]
    supporting = " ".join(d["text"] for d in retrieved_docs[:2])
    return {
        "answer": supporting,
        "source": "policy_kb",
        "confidence": round(top["score"], 2),
    }


def compose_refusal_answer(top_score: float, threshold: float) -> dict:
    return {
        "answer": (
            "I don't have a confident answer for that in the current policy "
            f"knowledge base (best match similarity {top_score:.2f} is below "
            f"the {threshold:.2f} grounding threshold), so I won't guess. "
            "Please rephrase, or a human agent can help with this one."
        ),
        "source": "policy_kb",
        "confidence": 0.0,
    }


def compose_return_risk_answer(tool_result: dict, order_id: str = None) -> dict:
    order_ref = f"order #{order_id}" if order_id else "this order"
    return {
        "answer": (
            f"{order_ref} is estimated at {tool_result['return_probability']:.0%} "
            f"probability of being returned -- risk bucket: {tool_result['risk_bucket']}. "
            f"(Anchored to t*_rf={tool_result['t_star_rf']}: Low < {tool_result['low_cut']}, "
            f"High >= {tool_result['high_cut']}.)"
        ),
        "source": "return_risk_tool",
        "confidence": round(
            tool_result["return_probability"] if tool_result["risk_bucket"] == "High"
            else 1 - tool_result["return_probability"] if tool_result["risk_bucket"] == "Low"
            else 0.5,
            2,
        ),
    }


def compose_image_classification_answer(tool_result: dict) -> dict:
    return {
        "answer": (
            f"This image looks like: {tool_result['predicted_category']} "
            f"(confidence {tool_result['confidence']:.0%})."
        ),
        "source": "image_classifier_tool",
        "confidence": tool_result["confidence"],
    }


def compose_injection_blocked_answer(reason: str) -> dict:
    return {
        "answer": (
            "I can't follow instructions embedded in a message -- I only "
            "answer Flipkart support questions using the policy knowledge "
            "base and the order/image tools. " + reason
        ),
        "source": "policy_kb",
        "confidence": 0.0,
    }
