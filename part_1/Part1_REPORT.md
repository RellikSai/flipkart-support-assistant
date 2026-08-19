# Part 1 -- Return-Risk Scoring Pipeline: Full Run Report

```

==============================================================================
TASK 2 -- DATA VERIFICATION
==============================================================================
Total rows: 6000
Total columns: 13
Overall return rate: 0.2275 (22.75%)
Percent of rating_given missing: 13.05%

Return rate by product_category:
                  return_rate     n
product_category                   
Apparel              0.264275  1979
Beauty               0.200345   579
Electronics          0.186930  1316
Footwear             0.259570  1071
Home                 0.191469  1055

Return rate by payment_method:
                return_rate     n
payment_method                   
COD                0.307477  2501
Prepaid_Card       0.168154  1457
Prepaid_UPI        0.169199  1448
Wallet             0.178451   594

Missingness rate of rating_given by payment_method:
payment_method
COD             0.228309
Prepaid_Card    0.063143
Prepaid_UPI     0.056630
Wallet          0.063973

COD missing rate: 22.83%  |  Non-COD missing rate: 6.06%
Gap (COD - non-COD): 16.77 percentage points

Missingness classification: MAR (Missing At Random), conditional on the observed payment_method column. Evidence: the probability that rating_given is missing is 22.8% for COD orders vs 6.1% for non-COD orders -- a gap of 16.8 points. This is exactly how the generator constructs the column (missing_mask draws from a Bernoulli whose probability is 0.22 if payment_method == 'COD' else 0.06), i.e. missingness depends on another FULLY OBSERVED column (payment_method), not on the unobserved rating value itself (which would be MNAR) and not on nothing at all (which would be MCAR).

==============================================================================
TASK 3 -- PREPROCESSING PIPELINE (LEAK-FREE)
==============================================================================
Train rows: 4800  |  Test rows: 1200
Train return rate: 0.2275  |  Test return rate: 0.2275
Preprocessing: ColumnTransformer(median-impute+scale numeric, mode-impute+one-hot categorical). Fit ONLY on X_train, then used to transform() both X_train and X_test -- never fit on test data.

==============================================================================
TASK 4 -- BASELINE (DummyClassifier, most_frequent)
==============================================================================
DummyClassifier accuracy: 0.7725
DummyClassifier F1 (class 1 = returned): 0.0000

Why this accuracy is misleading: with an ~22-23% return rate, a classifier that NEVER predicts a return can still score roughly 77-78% accuracy just by exploiting the class imbalance. Its F1 for the returned=1 class is 0.0 because recall is 0.0 -- it flags zero of the actual returns. This is the classic 'high accuracy, zero recall' trap: accuracy alone hides the fact that the model is operationally useless for the real business problem (catching orders likely to be returned), because it was never compared against this baseline nor evaluated on a metric aligned to that goal.

==============================================================================
TASK 5 -- LOGISTIC REGRESSION + THRESHOLD SWEEP
==============================================================================
At default threshold 0.5:
  Accuracy : 0.5917
  F1 (cls1): 0.3921
  Recall   : 0.5788
  Precision: 0.2964
  ROC-AUC  : 0.6253

Threshold sweep (0.10 -> 0.90, step 0.02) -- Logistic Regression:
 threshold       f1   recall  precision
      0.10 0.370672 1.000000   0.227500
      0.12 0.370672 1.000000   0.227500
      0.14 0.370672 1.000000   0.227500
      0.16 0.370672 1.000000   0.227500
      0.18 0.370672 1.000000   0.227500
      0.20 0.370672 1.000000   0.227500
      0.22 0.369565 0.996337   0.226856
      0.24 0.371585 0.996337   0.228380
      0.26 0.370014 0.985348   0.227773
      0.28 0.375698 0.985348   0.232097
      0.30 0.376167 0.959707   0.233929
      0.32 0.373708 0.926740   0.234043
      0.34 0.382195 0.912088   0.241748
      0.36 0.387869 0.890110   0.247959
      0.38 0.394979 0.864469   0.255965
      0.40 0.401055 0.835165   0.263889
      0.42 0.402224 0.794872   0.269231
      0.44 0.409091 0.758242   0.280108
      0.46 0.402128 0.692308   0.283358
      0.48 0.394977 0.633700   0.286899
      0.50 0.392060 0.578755   0.296435
      0.52 0.396122 0.523810   0.318486
      0.54 0.375189 0.454212   0.319588
      0.56 0.355848 0.395604   0.323353
      0.58 0.344583 0.355311   0.334483
      0.60 0.322457 0.307692   0.338710
      0.62 0.322581 0.274725   0.390625
      0.64 0.305164 0.238095   0.424837
      0.66 0.243523 0.172161   0.415929
      0.68 0.173669 0.113553   0.369048
      0.70 0.128834 0.076923   0.396226
      0.72 0.080268 0.043956   0.461538
      0.74 0.034965 0.018315   0.384615
      0.76 0.021352 0.010989   0.375000
      0.78 0.007220 0.003663   0.250000
      0.80 0.000000 0.000000   0.000000
      0.82 0.000000 0.000000   0.000000
      0.84 0.000000 0.000000   0.000000
      0.86 0.000000 0.000000   0.000000
      0.88 0.000000 0.000000   0.000000
      0.90 0.000000 0.000000   0.000000

F1-maximising threshold (Logistic Regression): t* = 0.44
  F1        at t*: 0.4091
  Recall    at t*: 0.7582
  Precision at t*: 0.2801

Recall gain vs default threshold: 17.95 percentage points
Precision drop vs default threshold: 1.63 percentage points

Business trade-off: lowering the decision threshold below 0.5 makes the model flag more orders as 'likely to be returned'. This buys higher recall -- fewer real returns slip through undetected, which matters because an undetected high-risk order costs Flipkart a full return/refund/reverse-logistics cycle. The price paid is lower precision: more low-risk orders get flagged too, each one costing an unnecessary manual review or a customer-facing friction point (e.g. a COD verification call). In short: we are accepting more false positives (wasted review effort) in exchange for fewer false negatives (missed returns), which is the right trade for a proactive risk-flagging tool where a missed return is more expensive than an extra review.

==============================================================================
TASK 6 -- RANDOM FOREST + GRIDSEARCHCV
==============================================================================
Best params: {'clf__max_depth': 6, 'clf__n_estimators': 100}
Best cross-validated ROC-AUC: 0.6178
Held-out test-set ROC-AUC (winning config): 0.6143
Gap (|CV - test|): 0.0036  (should be <= 0.05)

==============================================================================
TASK 7 -- FEATURE IMPORTANCE (impurity vs permutation)
==============================================================================
Top-5 features by impurity-based .feature_importances_:
payment_method_COD      0.166461
price_inr               0.137116
customer_tenure_days    0.107431
delivery_distance_km    0.097244
discount_pct            0.089011

Full permutation importance (raw columns, test split, roc_auc drop):
payment_method          0.097461
price_inr               0.008015
num_previous_returns    0.007110
product_category        0.004619
delivery_days          -0.000416
is_weekend_order       -0.001151
num_previous_orders    -0.001642
rating_given           -0.002549
delivery_distance_km   -0.002711
discount_pct           -0.002868
customer_tenure_days   -0.005192

Comparison note: impurity-based .feature_importances_ operates on the ONE-HOT expanded matrix and is computed from training-time impurity reduction, so it can overrate high-cardinality/continuous columns (e.g. delivery_distance_km, delivery_days) purely because they offer many possible split points to fit noise to, even when that column carries little real signal about the outcome. Permutation importance instead measures the actual drop in held-out ROC-AUC when a column is shuffled, so a column that the model leaned on only to fit noise shows little to no drop. Columns like delivery_distance_km and is_weekend_order -- which the true data-generating process for `returned` does not depend on -- are the ones most likely to lose most of their apparent importance under the permutation measure, because shuffling them barely changes the model's real predictive performance.

==============================================================================
TASK 8 -- SUBGROUP ANALYSIS
==============================================================================
Overall test-set recall (class 1): 0.5092
Overall test-set precision (class 1): 0.3188

Recall / precision by product_category:
product_category   n   recall  precision
     Electronics 261 0.326923   0.278689
        Footwear 217 0.500000   0.333333
         Apparel 385 0.530000   0.341935
          Beauty 116 0.612903   0.500000
            Home 221 0.647059   0.224490

Recall / precision by payment_method:
payment_method   n   recall  precision
  Prepaid_Card 283 0.000000   0.000000
   Prepaid_UPI 294 0.041667   0.666667
        Wallet 120 0.047619   0.500000
           COD 503 0.877419   0.317016

Weakest product_category subgroup: Electronics (recall 0.327 vs overall 0.509).
Weakest payment_method subgroup: Prepaid_Card (recall 0.000 vs overall 0.509).

Proposed concrete fix: rather than a single global 0.5 cut, apply a category-specific decision threshold for 'Electronics' orders (calibrated via the same F1-threshold-sweep procedure used in Task 5, run on just that category's held-out rows). Sizing/fit risk is concentrated in Apparel/Footwear, so a single global threshold tuned across all categories systematically under-flags the subgroup whose base return rate is highest; a category-specific threshold recovers recall there without moving every other category's operating point.

==============================================================================
TASK 9 -- FINAL ARTIFACT + t*_rf
==============================================================================
Threshold sweep (0.10 -> 0.90, step 0.02) -- Random Forest's own predict_proba:
 threshold       f1   recall  precision
      0.10 0.370672 1.000000   0.227500
      0.12 0.370672 1.000000   0.227500
      0.14 0.370672 1.000000   0.227500
      0.16 0.370672 1.000000   0.227500
      0.18 0.370672 1.000000   0.227500
      0.20 0.370672 1.000000   0.227500
      0.22 0.370672 1.000000   0.227500
      0.24 0.370672 1.000000   0.227500
      0.26 0.370924 1.000000   0.227690
      0.28 0.372188 1.000000   0.228643
      0.30 0.373021 0.992674   0.229661
      0.32 0.375972 0.974359   0.232925
      0.34 0.381805 0.937729   0.239700
      0.36 0.384013 0.897436   0.244267
      0.38 0.392679 0.864469   0.254037
      0.40 0.393966 0.813187   0.259953
      0.42 0.394990 0.750916   0.267974
      0.44 0.392324 0.673993   0.276692
      0.46 0.396181 0.608059   0.293805
      0.48 0.391590 0.545788   0.305328
      0.50 0.392102 0.509158   0.318807
      0.52 0.378539 0.465201   0.319095
      0.54 0.358974 0.410256   0.319088
      0.56 0.335766 0.336996   0.334545
      0.58 0.273684 0.238095   0.321782
      0.60 0.200528 0.139194   0.358491
      0.62 0.109422 0.065934   0.321429
      0.64 0.040404 0.021978   0.250000
      0.66 0.014184 0.007326   0.222222
      0.68 0.007299 0.003663   1.000000
      0.70 0.000000 0.000000   0.000000
      0.72 0.000000 0.000000   0.000000
      0.74 0.000000 0.000000   0.000000
      0.76 0.000000 0.000000   0.000000
      0.78 0.000000 0.000000   0.000000
      0.80 0.000000 0.000000   0.000000
      0.82 0.000000 0.000000   0.000000
      0.84 0.000000 0.000000   0.000000
      0.86 0.000000 0.000000   0.000000
      0.88 0.000000 0.000000   0.000000
      0.90 0.000000 0.000000   0.000000

t*_rf (F1-maximising threshold on RF's own predict_proba) = 0.46
  F1        at t*_rf: 0.3962
  Recall    at t*_rf: 0.6081
  Precision at t*_rf: 0.2938

Part 3 risk buckets will use: Low if probability < 0.46, High if probability >= 0.61, else Medium (anchored to t*_rf = 0.46).

Saved final tuned Random Forest pipeline to models/return_risk_model.pkl
Sanity check -- reloaded model predict_proba on first 5 test rows: [0.532, 0.3502, 0.3357, 0.515, 0.4638]
Saved t_star_rf.json (consumed by Part 3's check_return_risk tool).
```
