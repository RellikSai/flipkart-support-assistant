# Part 1 -- Return-Risk Scoring Pipeline

## Files
- `generate_orders.py` -- exact seeded dataset generator (do not modify the seed or lists).
- `orders_dataset.csv` -- generated output (6000 rows x 13 columns).
- `part1_pipeline.py` -- runs Tasks 2-9 end to end (verification, preprocessing,
  baseline, Logistic Regression + threshold sweep, Random Forest + GridSearchCV,
  feature/permutation importance, subgroup analysis, final artifact save).
- `models/return_risk_model.pkl` -- final tuned Random Forest pipeline
  (preprocessing + model as one fitted sklearn `Pipeline`), saved with `joblib`.
- `t_star_rf.json` -- the F1-maximising threshold `t*_rf` for the saved Random
  Forest's own `predict_proba`, plus the Low/Medium/High cut points derived
  from it. This is what Part 3's `check_return_risk` tool reads.
- `REPORT.md` -- full captured output of the pipeline run (all numbers below
  are copied from this file, not hand-typed).

## How to run
```bash
pip install scikit-learn pandas numpy joblib
python3 generate_orders.py      # writes orders_dataset.csv (6000 rows)
python3 part1_pipeline.py       # writes REPORT.md, t_star_rf.json, models/return_risk_model.pkl
```

## Key results (from this run's REPORT.md)
- Rows: 6000, columns: 13, overall return rate: **22.75%**
- `rating_given` missing: **13.05%** of rows
- Missingness mechanism: **MAR**, conditional on `payment_method`
  (COD missing rate 22.83% vs non-COD 6.06%, a 16.77-point gap)
- Baseline `DummyClassifier`: accuracy 0.7725, **F1(class 1) = 0.0**
  ("high accuracy, zero recall" trap)
- Logistic Regression @0.5: ROC-AUC 0.6253, F1 0.3921, recall 0.5788, precision 0.2964
- Logistic Regression best-F1 threshold: **t\* = 0.44** -> recall 0.7582,
  precision 0.2801 (recall +17.95 pts vs default, precision -1.63 pts)
- Random Forest GridSearchCV: best params `{max_depth: 6, n_estimators: 100}`,
  best CV ROC-AUC **0.6178**, held-out test ROC-AUC **0.6143** (gap 0.0036)
- Top-5 impurity feature importances: `payment_method_COD`, `price_inr`,
  `customer_tenure_days`, `delivery_distance_km`, `discount_pct`
- Permutation importance (test-set ROC-AUC drop) shows `customer_tenure_days`,
  `delivery_distance_km`, and `discount_pct` collapse to ~0 or negative,
  while `payment_method` and `price_inr` remain the dominant real signal --
  impurity importance overrates continuous columns because they offer many
  split points to fit noise to, regardless of true signal.
- Weakest subgroups: `Electronics` (recall 0.327 vs overall 0.509) by
  category; `Prepaid_Card` (recall 0.000) by payment method. Proposed fix:
  a category-specific decision threshold for Electronics, retuned via the
  same F1-sweep procedure on that category's rows alone.
- Final artifact: `models/return_risk_model.pkl` (tuned Random Forest
  pipeline), anchored threshold **t\*_rf = 0.46** -> Low if `p < 0.46`,
  High if `p >= 0.61`, else Medium.

See `REPORT.md` for the full threshold-sweep tables and per-category /
per-payment-method breakdowns.
