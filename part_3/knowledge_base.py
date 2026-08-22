"""
knowledge_base.py

The actual policy text the RAG pipeline is grounded in. I wrote these myself
(Flipkart-style, not copied from anywhere real) -- 14 short docs, each
2-4 sentences, covering the topics the brief asks for: return windows by
category, COD refund timelines, delivery SLAs, and reverse-pickup
eligibility, plus a few extra ones so the agent has something to say for
realistic follow-up questions too.

Each doc has a stable doc_id -- chunking.py splits these into one chunk per
sentence but always keeps a pointer back to doc_id, since Part 3 Task 10
(retrieval eval) and the groundedness check both score at the DOCUMENT level,
not the chunk level.
"""

POLICY_DOCS = [
    {
        "doc_id": "return_apparel_footwear",
        "title": "Return window -- Apparel & Footwear",
        "text": (
            "Apparel and footwear items can be returned within 30 days of delivery. "
            "The item must be unused, unwashed, and returned with its original tags "
            "and packaging intact. Innerwear, socks, and swimwear are not eligible "
            "for return once the hygiene seal has been broken."
        ),
    },
    {
        "doc_id": "return_electronics",
        "title": "Return window -- Electronics",
        "text": (
            "Most electronics can be returned within 10 days of delivery if the item "
            "is defective, damaged, or materially different from what was ordered. "
            "The original box, all accessories, and any free-gift items must be "
            "included. Electronics returned only because the customer changed their "
            "mind are not eligible after the box has been opened."
        ),
    },
    {
        "doc_id": "return_home",
        "title": "Return window -- Home & Furniture",
        "text": (
            "Home and furniture items follow a 15-day return window from the date "
            "of delivery. Large furniture items require a reverse pickup rather than "
            "a courier drop-off, and the item must be repackaged in its original "
            "crate or box wherever possible."
        ),
    },
    {
        "doc_id": "return_beauty",
        "title": "Return window -- Beauty & Personal Care",
        "text": (
            "Beauty and personal care products can only be returned if they arrive "
            "damaged, expired, or incorrect, within 7 days of delivery. Opened "
            "cosmetics, skincare, or fragrance items cannot be returned for hygiene "
            "reasons unless the item itself is defective."
        ),
    },
    {
        "doc_id": "cod_refund_timeline",
        "title": "COD refund timeline",
        "text": (
            "For Cash on Delivery orders, refunds are issued to the customer's bank "
            "account or Flipkart wallet, since there is no original payment "
            "instrument to reverse. COD refunds are typically processed within 7-9 "
            "business days after the returned item passes a quality check at the "
            "warehouse. Customers must share valid bank account details for the "
            "refund to be initiated."
        ),
    },
    {
        "doc_id": "prepaid_refund_timeline",
        "title": "Prepaid refund timeline",
        "text": (
            "For prepaid orders paid by card, UPI, or wallet, refunds are credited "
            "back to the original payment method within 3-5 business days of the "
            "return being approved. Card refunds may take a further 2-3 business "
            "days to reflect on the customer's statement depending on the issuing "
            "bank."
        ),
    },
    {
        "doc_id": "delivery_sla_standard",
        "title": "Standard delivery SLA",
        "text": (
            "Standard delivery for most in-stock items takes 2-7 business days "
            "depending on the destination pin code. Metro cities typically see "
            "delivery within 2-4 days, while remote or non-serviceable areas may "
            "take up to 7 days."
        ),
    },
    {
        "doc_id": "delivery_sla_express",
        "title": "Express delivery SLA",
        "text": (
            "Express delivery, where available, guarantees next-day delivery for "
            "orders placed before the daily cutoff time shown at checkout. Express "
            "delivery is currently limited to select metro and tier-1 city pin "
            "codes and select product categories."
        ),
    },
    {
        "doc_id": "reverse_pickup_eligibility",
        "title": "Reverse pickup eligibility",
        "text": (
            "Reverse pickup, where a courier collects the return directly from the "
            "customer's address, is available in most serviceable pin codes for "
            "apparel, footwear, electronics, and home items above a minimum size "
            "threshold. The customer does not need to pack a shipping label "
            "themselves when reverse pickup is available -- the courier carries one."
        ),
    },
    {
        "doc_id": "reverse_pickup_unavailable",
        "title": "Self-ship returns where reverse pickup is unavailable",
        "text": (
            "In pin codes where reverse pickup is not available, the customer must "
            "self-ship the item back using the prepaid shipping label provided in "
            "the Flipkart app. Self-ship return costs are reimbursed by Flipkart "
            "once the returned item is received and verified at the warehouse."
        ),
    },
    {
        "doc_id": "exchange_policy",
        "title": "Exchange policy",
        "text": (
            "Apparel and footwear items can be exchanged for a different size within "
            "the same 30-day return window instead of a refund, subject to stock "
            "availability of the requested size. Only one free exchange is allowed "
            "per order line item."
        ),
    },
    {
        "doc_id": "damaged_item_policy",
        "title": "Damaged or wrong item received",
        "text": (
            "If an item arrives damaged, defective, or different from what was "
            "ordered, the customer should report it within 48 hours of delivery "
            "through the Flipkart app, ideally with photos. Damaged or wrong-item "
            "claims are prioritized and typically resolved with a replacement or "
            "full refund without requiring the usual return-window wait."
        ),
    },
    {
        "doc_id": "cancellation_policy",
        "title": "Order cancellation policy",
        "text": (
            "Orders can be cancelled free of charge any time before they are "
            "shipped. Once an order has shipped, it can no longer be cancelled and "
            "must instead be returned after delivery following the applicable "
            "category return window."
        ),
    },
    {
        "doc_id": "warranty_electronics",
        "title": "Warranty on electronics",
        "text": (
            "Most electronics carry a manufacturer warranty separate from "
            "Flipkart's own return window, typically ranging from 6 months to 2 "
            "years depending on the brand and product. Warranty claims after the "
            "return window has closed are handled directly through the "
            "manufacturer's authorized service centers, not through a Flipkart "
            "return."
        ),
    },
]
