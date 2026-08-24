# Return-risk question -- calls check_return_risk (real Part 1 model)

### Turn 1
**User:** Is order #48213 likely to be returned?

**Agent (final_answer):**
```json
{
  "answer": "order #48213 is estimated at 53% probability of being returned -- risk bucket: Medium. (Anchored to t*_rf=0.46: Low < 0.46, High >= 0.61.)",
  "source": "return_risk_tool",
  "confidence": 0.5
}
```

_raw tool_result:_ `{'return_probability': 0.5252, 'risk_bucket': 'Medium', 't_star_rf': 0.46, 'low_cut': 0.46, 'high_cut': 0.61}`
