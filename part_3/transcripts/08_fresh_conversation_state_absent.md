# Fresh conversation -- new thread_id, no prior order id (state correctly reset)

### Turn 1 (brand-new thread, never mentioned an order)
**User:** What's the standard delivery SLA?

**Agent (final_answer):**
```json
{
  "answer": "Standard delivery for most in-stock items takes 2-7 business days depending on the destination pin code. Express delivery, where available, guarantees next-day delivery for orders placed before the daily cutoff time shown at checkout.",
  "source": "policy_kb",
  "confidence": 0.41
}
```

_retrieved docs (doc_id, score):_
- delivery_sla_standard (0.409)
- delivery_sla_express (0.151)
- return_apparel_footwear (0.131)
