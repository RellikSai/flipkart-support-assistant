"""
prompts.py

The system prompt and few-shot intent examples. Since MOCK_LLM mode never
actually calls a live LLM, this prompt isn't fed to a model at runtime --
but it's still what a live-LLM run (USE_LIVE_LLM=1) would use, and it's
annotated below against the 4S principles the brief asks for, plus role
prompting.
"""

# --- Role prompting ---------------------------------------------------------
# "You are Flipkart's support assistant" gives the model a clear persona and
# scope (support, not general chat) before anything else.
#
# --- 4S annotation ---------------------------------------------------------
# Specific : names the exact 3 intents it must route between and the exact
#            JSON schema it must return -- not "help the user however you can".
# Short    : one paragraph, no filler, no restating the whole policy KB inline.
# Surround : the retrieved KB chunk(s) / tool output are handed to the model
#            AFTER this prompt, wrapped in their own clearly delimited block
#            (see mock_llm.compose_answer), so the model's own instructions
#            never get mixed up with untrusted retrieved/user content.
# Single   : one job per call -- classify intent, OR compose the final answer.
#            The prompt below is not asked to do both at once.

SYSTEM_PROMPT = """You are Flipkart's support assistant. Answer only using the \
retrieved policy text or tool output you are given below -- never invent a \
policy or a number. If nothing given to you is relevant, say so instead of \
guessing. Always respond with a single JSON object with exactly these \
fields: answer (string), source (one of "policy_kb", "return_risk_tool", \
"image_classifier_tool"), confidence (float 0-1)."""

# --- Few-shot intent-classification examples --------------------------------
# 2+ required by the brief. These aren't decorative -- mock_llm.classify_intent
# actually uses them as anchor examples for its keyword-overlap fallback rule,
# so they visibly influence routing (see transcripts/06_fewshot_routing.md).
INTENT_FEWSHOT = [
    {"query": "What is the return window for a pair of sneakers?", "intent": "policy"},
    {"query": "How long does a COD refund take to reach my account?", "intent": "policy"},
    {"query": "Is order #48213 likely to be returned?", "intent": "return_risk"},
    {"query": "What category is this product photo?", "intent": "product_category"},
]
