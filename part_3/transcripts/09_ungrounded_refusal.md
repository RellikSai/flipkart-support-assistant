# Groundedness guardrail -- question with no sufficiently-similar KB chunk (refused)

> **Note:** generated in a sandbox with no internet access, using the offline TF-IDF retrieval stand-in described in `_sandbox_offline_check.py` (not the real sentence-transformers+FAISS retriever) and a plain-Python stand-in for the LangGraph engine (not the real `langgraph` library) and a stubbed image-classifier result (Part 2's model wasn't trained in this sandbox either). `guardrails.py`, `mock_llm.py`, and `tools.check_return_risk` are the REAL modules, and `check_return_risk` is scored by the REAL Part 1 model. Run `python3 run_transcripts.py` locally after `pip install -r requirements.txt` to regenerate this file against the real, required stack before submitting -- retrieval quality in particular should improve, since MiniLM embeddings capture semantic similarity (e.g. "shoes" ~ "footwear") that plain TF-IDF word-overlap misses.

### Turn 1
**User:** Can you recommend a good biryani recipe?

**Agent (final_answer):**
```json
{
  "answer": "I don't have a confident answer for that in the current policy knowledge base (best match similarity 0.00 is below the 0.20 grounding threshold), so I won't guess. Please rephrase, or a human agent can help with this one.",
  "source": "policy_kb",
  "confidence": 0.0
}
```

_retrieved docs (doc_id, score):_
- warranty_electronics (0.000)
- cancellation_policy (0.000)
