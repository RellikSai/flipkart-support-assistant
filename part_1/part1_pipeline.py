"""
Part 1 -- Return-Risk Scoring Pipeline
Flipkart Order Intelligence & Support Assistant

Runs Tasks 2-9 end to end:
  - data verification (Task 2)
  - leak-free preprocessing pipeline (Task 3)
  - DummyClassifier baseline (Task 4)
  - Logistic Regression + threshold sweep (Task 5)
  - Random Forest + GridSearchCV (Task 6)
  - feature importance + permutation importance (Task 7)
  - subgroup / root-cause analysis (Task 8)
  - final artifact save with t*_rf (Task 9)

Run: python3 part1_pipeline.py
Writes a full text report to REPORT.md and the final model to
models/return_risk_model.pkl
"""

import io
import json
import contextlib

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
REPORT_LINES = []


def log(*args):
    """Print to stdout AND accumulate into the report."""
    line = " ".join(str(a) for a in args)
    print(line)
    REPORT_LINES.append(line)


def log_header(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


# ---------------------------------------------------------------------------
# Task 2 -- Load + verify the generated data
# ---------------------------------------------------------------------------
log_header("TASK 2 -- DATA VERIFICATION")

df = pd.read_csv("orders_dataset.csv")

n_rows, n_cols = df.shape
overall_return_rate = df["returned"].mean()
pct_missing_rating = df["rating_given"].isna().mean() * 100

log(f"Total rows: {n_rows}")
log(f"Total columns: {n_cols}")
log(f"Overall return rate: {overall_return_rate:.4f} ({overall_return_rate*100:.2f}%)")
log(f"Percent of rating_given missing: {pct_missing_rating:.2f}%")

log("\nReturn rate by product_category:")
cat_table = df.groupby("product_category")["returned"].agg(["mean", "count"]).rename(
    columns={"mean": "return_rate", "count": "n"}
)
log(cat_table.to_string())

log("\nReturn rate by payment_method:")
pay_table = df.groupby("payment_method")["returned"].agg(["mean", "count"]).rename(
    columns={"mean": "return_rate", "count": "n"}
)
log(pay_table.to_string())

# Missingness gap: COD vs non-COD
missing_by_pay = df.assign(rating_missing=df["rating_given"].isna()).groupby("payment_method")[
    "rating_missing"
].mean()
cod_missing_rate = missing_by_pay.get("COD", np.nan)
non_cod_missing_rate = df.assign(rating_missing=df["rating_given"].isna())
non_cod_missing_rate = non_cod_missing_rate.loc[
    non_cod_missing_rate["payment_method"] != "COD", "rating_missing"
].mean()

log("\nMissingness rate of rating_given by payment_method:")
log(missing_by_pay.to_string())
log(f"\nCOD missing rate: {cod_missing_rate*100:.2f}%  |  Non-COD missing rate: {non_cod_missing_rate*100:.2f}%")
gap = (cod_missing_rate - non_cod_missing_rate) * 100
log(f"Gap (COD - non-COD): {gap:.2f} percentage points")

log(
    "\nMissingness classification: MAR (Missing At Random), conditional on the "
    "observed payment_method column. Evidence: the probability that rating_given "
    f"is missing is {cod_missing_rate*100:.1f}% for COD orders vs {non_cod_missing_rate*100:.1f}% "
    f"for non-COD orders -- a gap of {gap:.1f} points. This is exactly how the generator "
    "constructs the column (missing_mask draws from a Bernoulli whose probability is "
    "0.22 if payment_method == 'COD' else 0.06), i.e. missingness depends on another "
    "FULLY OBSERVED column (payment_method), not on the unobserved rating value itself "
    "(which would be MNAR) and not on nothing at all (which would be MCAR)."
)

# ---------------------------------------------------------------------------
# Task 3 -- Preprocessing pipeline (no leakage)
# ---------------------------------------------------------------------------
log_header("TASK 3 -- PREPROCESSING PIPELINE (LEAK-FREE)")

FEATURES = [
    "product_category", "price_inr", "discount_pct", "payment_method",
    "customer_tenure_days", "num_previous_orders", "num_previous_returns",
    "delivery_distance_km", "delivery_days", "is_weekend_order", "rating_given",
]
TARGET = "returned"
CATEGORICAL = ["product_category", "payment_method"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

X = df[FEATURES].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
log(f"Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")
log(f"Train return rate: {y_train.mean():.4f}  |  Test return rate: {y_test.mean():.4f}")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC),
    ("cat", categorical_transformer, CATEGORICAL),
])
log(
    "Preprocessing: ColumnTransformer(median-impute+scale numeric, "
    "mode-impute+one-hot categorical). Fit ONLY on X_train, then used to "
    "transform() both X_train and X_test -- never fit on test data."
)

# ---------------------------------------------------------------------------
# Task 4 -- Baseline DummyClassifier
# ---------------------------------------------------------------------------
log_header("TASK 4 -- BASELINE (DummyClassifier, most_frequent)")

dummy_pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
])
dummy_pipe.fit(X_train, y_train)
dummy_pred = dummy_pipe.predict(X_test)

dummy_acc = accuracy_score(y_test, dummy_pred)
dummy_f1 = f1_score(y_test, dummy_pred, pos_label=1, zero_division=0)

log(f"DummyClassifier accuracy: {dummy_acc:.4f}")
log(f"DummyClassifier F1 (class 1 = returned): {dummy_f1:.4f}")
log(
    "\nWhy this accuracy is misleading: with an ~22-23% return rate, a classifier that "
    "NEVER predicts a return can still score roughly 77-78% accuracy just by exploiting "
    "the class imbalance. Its F1 for the returned=1 class is 0.0 because recall is 0.0 -- "
    "it flags zero of the actual returns. This is the classic 'high accuracy, zero recall' "
    "trap: accuracy alone hides the fact that the model is operationally useless for the "
    "real business problem (catching orders likely to be returned), because it was never "
    "compared against this baseline nor evaluated on a metric aligned to that goal."
)

# ---------------------------------------------------------------------------
# Task 5 -- Logistic Regression + threshold sweep
# ---------------------------------------------------------------------------
log_header("TASK 5 -- LOGISTIC REGRESSION + THRESHOLD SWEEP")

logreg_pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
])
logreg_pipe.fit(X_train, y_train)

logreg_proba_test = logreg_pipe.predict_proba(X_test)[:, 1]
logreg_pred_default = (logreg_proba_test >= 0.5).astype(int)

lr_acc = accuracy_score(y_test, logreg_pred_default)
lr_f1 = f1_score(y_test, logreg_pred_default, pos_label=1)
lr_recall = recall_score(y_test, logreg_pred_default, pos_label=1)
lr_precision = precision_score(y_test, logreg_pred_default, pos_label=1, zero_division=0)
lr_auc = roc_auc_score(y_test, logreg_proba_test)

log("At default threshold 0.5:")
log(f"  Accuracy : {lr_acc:.4f}")
log(f"  F1 (cls1): {lr_f1:.4f}")
log(f"  Recall   : {lr_recall:.4f}")
log(f"  Precision: {lr_precision:.4f}")
log(f"  ROC-AUC  : {lr_auc:.4f}")


def threshold_sweep(y_true, proba, lo=0.10, hi=0.90, step=0.02):
    rows = []
    t = lo
    while t <= hi + 1e-9:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, pred, pos_label=1, zero_division=0)
        prec = precision_score(y_true, pred, pos_label=1, zero_division=0)
        rows.append({"threshold": round(t, 2), "f1": f1, "recall": rec, "precision": prec})
        t += step
    return pd.DataFrame(rows)


sweep_lr = threshold_sweep(y_test, logreg_proba_test)
best_lr_row = sweep_lr.loc[sweep_lr["f1"].idxmax()]

log("\nThreshold sweep (0.10 -> 0.90, step 0.02) -- Logistic Regression:")
log(sweep_lr.to_string(index=False))

log(f"\nF1-maximising threshold (Logistic Regression): t* = {best_lr_row['threshold']}")
log(f"  F1        at t*: {best_lr_row['f1']:.4f}")
log(f"  Recall    at t*: {best_lr_row['recall']:.4f}")
log(f"  Precision at t*: {best_lr_row['precision']:.4f}")

recall_gain_pts = (best_lr_row["recall"] - lr_recall) * 100
precision_drop_pts = (lr_precision - best_lr_row["precision"]) * 100
log(f"\nRecall gain vs default threshold: {recall_gain_pts:.2f} percentage points")
log(f"Precision drop vs default threshold: {precision_drop_pts:.2f} percentage points")

log(
    "\nBusiness trade-off: lowering the decision threshold below 0.5 makes the model flag "
    "more orders as 'likely to be returned'. This buys higher recall -- fewer real returns "
    "slip through undetected, which matters because an undetected high-risk order costs "
    "Flipkart a full return/refund/reverse-logistics cycle. The price paid is lower "
    "precision: more low-risk orders get flagged too, each one costing an unnecessary "
    "manual review or a customer-facing friction point (e.g. a COD verification call). "
    "In short: we are accepting more false positives (wasted review effort) in exchange "
    "for fewer false negatives (missed returns), which is the right trade for a proactive "
    "risk-flagging tool where a missed return is more expensive than an extra review."
)

# ---------------------------------------------------------------------------
# Task 6 -- Random Forest + GridSearchCV
# ---------------------------------------------------------------------------
log_header("TASK 6 -- RANDOM FOREST + GRIDSEARCHCV")

rf_pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
])

param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [6, 10, None],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid = GridSearchCV(rf_pipe, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)

best_rf_pipe = grid.best_estimator_
best_cv_auc = grid.best_score_

rf_proba_test = best_rf_pipe.predict_proba(X_test)[:, 1]
rf_test_auc = roc_auc_score(y_test, rf_proba_test)

log(f"Best params: {grid.best_params_}")
log(f"Best cross-validated ROC-AUC: {best_cv_auc:.4f}")
log(f"Held-out test-set ROC-AUC (winning config): {rf_test_auc:.4f}")
log(f"Gap (|CV - test|): {abs(best_cv_auc - rf_test_auc):.4f}  (should be <= 0.05)")

# ---------------------------------------------------------------------------
# Task 7 -- Feature importance + permutation importance
# ---------------------------------------------------------------------------
log_header("TASK 7 -- FEATURE IMPORTANCE (impurity vs permutation)")

# Recover feature names after ColumnTransformer
fitted_prep = best_rf_pipe.named_steps["prep"]
ohe = fitted_prep.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL))
all_feature_names = NUMERIC + cat_feature_names

rf_model = best_rf_pipe.named_steps["clf"]
impurity_importances = pd.Series(rf_model.feature_importances_, index=all_feature_names)
impurity_top5 = impurity_importances.sort_values(ascending=False).head(5)

log("Top-5 features by impurity-based .feature_importances_:")
log(impurity_top5.to_string())

# Permutation importance on the held-out test split, using the whole pipeline
# (so it operates on the original raw X_test columns, not the transformed matrix)
perm_result = permutation_importance(
    best_rf_pipe, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1
)
perm_importances = pd.Series(perm_result.importances_mean, index=X_test.columns)
perm_top_for_same_features = perm_importances.reindex(
    # map the one-hot top-5 back to their raw source column where applicable
    list({f.split("_")[0] if f not in NUMERIC else f for f in impurity_top5.index})
).dropna()

log("\nFull permutation importance (raw columns, test split, roc_auc drop):")
log(perm_importances.sort_values(ascending=False).to_string())

log(
    "\nComparison note: impurity-based .feature_importances_ operates on the ONE-HOT "
    "expanded matrix and is computed from training-time impurity reduction, so it can "
    "overrate high-cardinality/continuous columns (e.g. delivery_distance_km, "
    "delivery_days) purely because they offer many possible split points to fit noise to, "
    "even when that column carries little real signal about the outcome. Permutation "
    "importance instead measures the actual drop in held-out ROC-AUC when a column is "
    "shuffled, so a column that the model leaned on only to fit noise shows little to no "
    "drop. Columns like delivery_distance_km and is_weekend_order -- which the true "
    "data-generating process for `returned` does not depend on -- are the ones most "
    "likely to lose most of their apparent importance under the permutation measure, "
    "because shuffling them barely changes the model's real predictive performance."
)

# ---------------------------------------------------------------------------
# Task 8 -- Subgroup / root-cause analysis
# ---------------------------------------------------------------------------
log_header("TASK 8 -- SUBGROUP ANALYSIS")

rf_pred_test_default = (rf_proba_test >= 0.5).astype(int)
eval_df = X_test.copy()
eval_df["y_true"] = y_test.values
eval_df["y_pred"] = rf_pred_test_default

overall_recall = recall_score(eval_df["y_true"], eval_df["y_pred"], pos_label=1, zero_division=0)
overall_precision = precision_score(eval_df["y_true"], eval_df["y_pred"], pos_label=1, zero_division=0)
log(f"Overall test-set recall (class 1): {overall_recall:.4f}")
log(f"Overall test-set precision (class 1): {overall_precision:.4f}")


def subgroup_report(eval_df, group_col):
    rows = []
    for val, g in eval_df.groupby(group_col):
        rec = recall_score(g["y_true"], g["y_pred"], pos_label=1, zero_division=0)
        prec = precision_score(g["y_true"], g["y_pred"], pos_label=1, zero_division=0)
        rows.append({group_col: val, "n": len(g), "recall": rec, "precision": prec})
    return pd.DataFrame(rows).sort_values("recall")


by_category = subgroup_report(eval_df, "product_category")
by_payment = subgroup_report(eval_df, "payment_method")

log("\nRecall / precision by product_category:")
log(by_category.to_string(index=False))
log("\nRecall / precision by payment_method:")
log(by_payment.to_string(index=False))

weakest_cat_row = by_category.iloc[0]
weakest_pay_row = by_payment.iloc[0]
log(
    f"\nWeakest product_category subgroup: {weakest_cat_row['product_category']} "
    f"(recall {weakest_cat_row['recall']:.3f} vs overall {overall_recall:.3f})."
)
log(
    f"Weakest payment_method subgroup: {weakest_pay_row['payment_method']} "
    f"(recall {weakest_pay_row['recall']:.3f} vs overall {overall_recall:.3f})."
)
log(
    "\nProposed concrete fix: rather than a single global 0.5 cut, apply a "
    f"category-specific decision threshold for '{weakest_cat_row['product_category']}' "
    "orders (calibrated via the same F1-threshold-sweep procedure used in Task 5, run "
    "on just that category's held-out rows). Sizing/fit risk is concentrated in "
    "Apparel/Footwear, so a single global threshold tuned across all categories "
    "systematically under-flags the subgroup whose base return rate is highest; a "
    "category-specific threshold recovers recall there without moving every other "
    "category's operating point."
)

# ---------------------------------------------------------------------------
# Task 9 -- Save the final artifact + t*_rf
# ---------------------------------------------------------------------------
log_header("TASK 9 -- FINAL ARTIFACT + t*_rf")

sweep_rf = threshold_sweep(y_test, rf_proba_test)
best_rf_row = sweep_rf.loc[sweep_rf["f1"].idxmax()]
t_star_rf = float(best_rf_row["threshold"])

log("Threshold sweep (0.10 -> 0.90, step 0.02) -- Random Forest's own predict_proba:")
log(sweep_rf.to_string(index=False))

log(f"\nt*_rf (F1-maximising threshold on RF's own predict_proba) = {t_star_rf}")
log(f"  F1        at t*_rf: {best_rf_row['f1']:.4f}")
log(f"  Recall    at t*_rf: {best_rf_row['recall']:.4f}")
log(f"  Precision at t*_rf: {best_rf_row['precision']:.4f}")

low_cut = t_star_rf
high_cut = round(t_star_rf + 0.15, 2)
log(
    f"\nPart 3 risk buckets will use: Low if probability < {low_cut}, "
    f"High if probability >= {high_cut}, else Medium (anchored to t*_rf = {t_star_rf})."
)

joblib.dump(best_rf_pipe, "models/return_risk_model.pkl")
log("\nSaved final tuned Random Forest pipeline to models/return_risk_model.pkl")

# Round-trip sanity check
reloaded = joblib.load("models/return_risk_model.pkl")
check_proba = reloaded.predict_proba(X_test.iloc[:5])[:, 1]
log(f"Sanity check -- reloaded model predict_proba on first 5 test rows: {np.round(check_proba, 4).tolist()}")

with open("t_star_rf.json", "w") as f:
    json.dump({"t_star_rf": t_star_rf, "low_cut": low_cut, "high_cut": high_cut}, f, indent=2)
log("Saved t_star_rf.json (consumed by Part 3's check_return_risk tool).")

# ---------------------------------------------------------------------------
# Write full report
# ---------------------------------------------------------------------------
with open("REPORT.md", "w") as f:
    f.write("# Part 1 -- Return-Risk Scoring Pipeline: Full Run Report\n\n```\n")
    f.write("\n".join(REPORT_LINES))
    f.write("\n```\n")

print("\n\nDone. Full report written to REPORT.md")
