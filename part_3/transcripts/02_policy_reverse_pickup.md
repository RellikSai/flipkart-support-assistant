# Policy question -- reverse pickup eligibility

> **Note:** generated in a sandbox with no internet access, using the offline TF-IDF retrieval stand-in described in `_sandbox_offline_check.py` (not the real sentence-transformers+FAISS retriever) and a plain-Python stand-in for the LangGraph engine (not the real `langgraph` library) and a stubbed image-classifier result (Part 2's model wasn't trained in this sandbox either). `guardrails.py`, `mock_llm.py`, and `tools.check_return_risk` are the REAL modules, and `check_return_risk` is scored by the REAL Part 1 model. Run `python3 run_transcripts.py` locally after `pip install -r requirements.txt` to regenerate this file against the real, required stack before submitting -- retrieval quality in particular should improve, since MiniLM embeddings capture semantic similarity (e.g. "shoes" ~ "footwear") that plain TF-IDF word-overlap misses.

### Turn 1
**User:** Does someone come pick up my return or do I have to ship it myself?

**Agent (final_answer):**
```json
{
  "answer": "Self-ship return costs are reimbursed by Flipkart once the returned item is received and verified at the warehouse. The customer does not need to pack a shipping label themselves when reverse pickup is available -- the courier carries one.",
  "source": "policy_kb",
  "confidence": 0.27
}
```

_retrieved docs (doc_id, score):_
- reverse_pickup_unavailable (0.272)
- reverse_pickup_eligibility (0.240)
