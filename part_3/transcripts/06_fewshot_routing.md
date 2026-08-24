# Few-shot examples driving intent routing

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
