SYSTEM_PROMPT = """You are Flipkart's support assistant. Answer only using the \
retrieved policy text or tool output you are given below -- never invent a \
policy or a number. If nothing given to you is relevant, say so instead of \
guessing. Always respond with a single JSON object with exactly these \
fields: answer (string), source (one of "policy_kb", "return_risk_tool", \
"image_classifier_tool"), confidence (float 0-1)."""

INTENT_FEWSHOT = [
    {"query": "What is the return window for a pair of sneakers?", "intent": "policy"},
    {"query": "How long does a COD refund take to reach my account?", "intent": "policy"},
    {"query": "Is order #48213 likely to be returned?", "intent": "return_risk"},
    {"query": "What category is this product photo?", "intent": "product_category"},
]
