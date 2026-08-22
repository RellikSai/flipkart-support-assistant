"""
tools.py

The two real tools the agent can call. Both load actual saved artifacts
from Parts 1 and 2 -- nothing here is a hardcoded stand-in.
"""

import json

import joblib
import pandas as pd

RETURN_RISK_MODEL_PATH = "models/return_risk_model.pkl"
T_STAR_PATH = "t_star_rf.json"

_return_risk_model = None
_t_star_cutoffs = None


def _load_return_risk_model():
    global _return_risk_model, _t_star_cutoffs
    if _return_risk_model is None:
        _return_risk_model = joblib.load(RETURN_RISK_MODEL_PATH)
    if _t_star_cutoffs is None:
        with open(T_STAR_PATH) as f:
            _t_star_cutoffs = json.load(f)
    return _return_risk_model, _t_star_cutoffs


def check_return_risk(order_features: dict) -> dict:
    """Loads Part 1's tuned Random Forest pipeline (models/return_risk_model.pkl)
    and scores one order.

    order_features must have the same columns the model was trained on:
    product_category, price_inr, discount_pct, payment_method,
    customer_tenure_days, num_previous_orders, num_previous_returns,
    delivery_distance_km, delivery_days, is_weekend_order, rating_given
    (rating_given can be missing/None -- the pipeline's own imputer handles it,
    same as at training time).

    Risk buckets are anchored to t*_rf (the F1-maximising threshold computed
    on THIS model's own predict_proba in Part 1 Task 9), not fixed 0.3/0.6
    cut points -- see README for why a fixed split isn't self-calibrating.
    """
    model, cutoffs = _load_return_risk_model()

    row = pd.DataFrame([order_features])
    proba = float(model.predict_proba(row)[0, 1])

    low_cut = cutoffs["low_cut"]
    high_cut = cutoffs["high_cut"]
    if proba < low_cut:
        bucket = "Low"
    elif proba >= high_cut:
        bucket = "High"
    else:
        bucket = "Medium"

    return {
        "return_probability": round(proba, 4),
        "risk_bucket": bucket,
        "t_star_rf": cutoffs["t_star_rf"],
        "low_cut": low_cut,
        "high_cut": high_cut,
    }


def classify_product_image(image_path: str) -> dict:
    """Loads Part 2's saved classifier (models/product_classifier.pt) via the
    documented classify_image() snippet and returns the predicted category
    plus confidence. Only meant to be pointed at the real .png files exported
    to data/sample_images/ in Part 2 Task 8."""
    # imported lazily so tools.py doesn't hard-require torch/torchvision for
    # people only exercising the return-risk / policy paths
    from classify_image import classify_image

    result = classify_image(image_path)
    return {
        "predicted_category": result["predicted_class"],
        "confidence": round(result["confidence"], 4),
        "image_path": image_path,
    }


if __name__ == "__main__":
    # quick manual check against a realistic high-risk-looking order
    demo_order = {
        "product_category": "Apparel",
        "price_inr": 1800,
        "discount_pct": 45,
        "payment_method": "COD",
        "customer_tenure_days": 20,
        "num_previous_orders": 1,
        "num_previous_returns": 1,
        "delivery_distance_km": 900,
        "delivery_days": 8,
        "is_weekend_order": 1,
        "rating_given": None,
    }
    print(check_return_risk(demo_order))
