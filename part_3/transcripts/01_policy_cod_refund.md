# Policy question -- COD refund timeline

> **Note:** generated in a sandbox with no internet access, using the offline TF-IDF retrieval stand-in described in `_sandbox_offline_check.py` (not the real sentence-transformers+FAISS retriever) and a plain-Python stand-in for the LangGraph engine (not the real `langgraph` library) and a stubbed image-classifier result (Part 2's model wasn't trained in this sandbox either). `guardrails.py`, `mock_llm.py`, and `tools.check_return_risk` are the REAL modules, and `check_return_risk` is scored by the REAL Part 1 model. Run `python3 run_transcripts.py` locally after `pip install -r requirements.txt` to regenerate this file against the real, required stack before submitting -- retrieval quality in particular should improve, since MiniLM embeddings capture semantic similarity (e.g. "shoes" ~ "footwear") that plain TF-IDF word-overlap misses.

### Turn 1
**User:** I paid cash on delivery, when will my refund hit my account?

**Agent (final_answer):**
```json
{
  "answer": "For Cash on Delivery orders, refunds are issued to the customer's bank account or Flipkart wallet, since there is no original payment instrument to reverse. For prepaid orders paid by card, UPI, or wallet, refunds are credited back to the original payment method within 3-5 business days of the return being approved.",
  "source": "policy_kb",
  "confidence": 0.35
}
```

_retrieved docs (doc_id, score):_
- cod_refund_timeline (0.351)
- prepaid_refund_timeline (0.155)
