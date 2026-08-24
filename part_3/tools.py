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
   
    from classify_image import classify_image

    result = classify_image(image_path)
    return {
        "predicted_category": result["predicted_class"],
        "confidence": round(result["confidence"], 4),
        "image_path": image_path,
    }


if __name__ == "__main__":
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
