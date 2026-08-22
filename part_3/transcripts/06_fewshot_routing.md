# Few-shot examples driving intent routing

> **Note:** generated in a sandbox with no internet access, using the offline TF-IDF retrieval stand-in described in `_sandbox_offline_check.py` (not the real sentence-transformers+FAISS retriever) and a plain-Python stand-in for the LangGraph engine (not the real `langgraph` library) and a stubbed image-classifier result (Part 2's model wasn't trained in this sandbox either). `guardrails.py`, `mock_llm.py`, and `tools.check_return_risk` are the REAL modules, and `check_return_risk` is scored by the REAL Part 1 model. Run `python3 run_transcripts.py` locally after `pip install -r requirements.txt` to regenerate this file against the real, required stack before submitting -- retrieval quality in particular should improve, since MiniLM embeddings capture semantic similarity (e.g. "shoes" ~ "footwear") that plain TF-IDF word-overlap misses.

### Turn 1
**User:** What is the return window for a pair of sneakers?

**Agent (final_answer):**
```json
{
  "answer": "Warranty claims after the return window has closed are handled directly through the manufacturer's authorized service centers, not through a Flipkart return. Home and furniture items follow a 15-day return window from the date of delivery.",
  "source": "policy_kb",
  "confidence": 0.38
}
```

_retrieved docs (doc_id, score):_
- warranty_electronics (0.378)
- return_home (0.335)
- cancellation_policy (0.296)

### Turn 2
**User:** How long does a COD refund take to reach my account?

**Agent (final_answer):**
```json
{
  "answer": "Customers must share valid bank account details for the refund to be initiated. The customer does not need to pack a shipping label themselves when reverse pickup is available -- the courier carries one.",
  "source": "policy_kb",
  "confidence": 0.29
}
```

_retrieved docs (doc_id, score):_
- cod_refund_timeline (0.294)
- reverse_pickup_eligibility (0.186)
