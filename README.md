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


 # Part 2 -- Product Image Categoriser via Transfer Learning

## Important: this needs to be run on your machine, not here
This code needs internet access once (to auto-download Fashion-MNIST and the
pretrained ResNet-18 ImageNet weights) and PyTorch, neither of which is
available in the sandbox that generated these files. Every number in this
README's "Results" section is a **placeholder to fill in from your own run's
`REPORT.md`** -- do not submit invented numbers. Run it locally or on a free
GPU runtime (Colab/Kaggle); on a CPU-only laptop it should still finish in
well under an hour thanks to the feature-caching trick described below.

## Files
- `part2_train.py` -- runs Tasks 1-7 end to end: loads Fashion-MNIST (pinned
  source), preprocesses for a pretrained backbone, trains a new head on
  **cached** frozen-ResNet-18 features (the documented CPU speed trick),
  fine-tunes late layers only if feature-extraction validation accuracy is
  below 80%, evaluates on the untouched test split, auto-diagnoses the top
  confused category pairs from the real confusion matrix, and saves the model.
- `export_sample_images.py` -- Task 8: exports one real test-split image per
  class (10 total, covering all categories) as actual `.png` files into
  `data/sample_images/`, named so the true label is obvious from the filename.
- `classify_image.py` -- the documented one-function loading + single-image
  prediction snippet. This is exactly what Part 3's `classify_product_image`
  tool imports and calls -- not a reimplementation.
- `models/product_classifier.pt` -- written by `part2_train.py`: model
  weights (`state_dict`) plus preprocessing metadata (image size, ImageNet
  mean/std, class names) needed to reload and run inference.
- `REPORT.md` / `part2_summary.json` -- written by `part2_train.py` from your
  actual run; this is what you copy real numbers out of for grading.

## How to run
```bash
pip install torch torchvision scikit-learn pandas numpy pillow

python3 part2_train.py            # trains, evaluates, saves models/product_classifier.pt
python3 export_sample_images.py   # writes 10 real .png files to data/sample_images/
python3 classify_image.py data/sample_images/00_t-shirt-top.png   # smoke test
```

## Design choices (per the brief)
- **Backbone**: ResNet-18, ImageNet-pretrained (`ResNet18_Weights.IMAGENET1K_V1`).
- **Input size**: 224x224 (ResNet-18's standard expected size); grayscale
  replicated to 3 channels; normalized with ImageNet mean/std.
- **Feature-extraction stage**: all backbone layers frozen; the frozen
  backbone is run **once** over every train/val/test image and its 512-d
  output cached, then only a `Linear(512, 10)` head is trained on those
  cached vectors -- mathematically identical to re-running the frozen
  backbone every epoch, but turns an hours-long CPU loop into a few minutes.
  Optimizer: Adam, lr=1e-3, batch size 256, 15 epochs (see `part2_train.py`
  header for exact constants).
- **Fine-tuning fallback**: only triggered if feature-extraction validation
  accuracy is below 80%. If triggered, unfreezes `layer4` only (keeps
  layers 1-3 frozen) and continues training end-to-end at lr=1e-4 for 5
  epochs -- the standard gradual-unfreezing strategy.
- **Splits**: standard Fashion-MNIST 60,000-image train / 10,000-image test;
  a stratified 6,000-image validation split is carved out of the 60,000
  training images (comfortably above the required 5,000), leaving the test
  split completely untouched until the final Task 5 evaluation.
- **Confusion-pair diagnosis**: computed from the real confusion matrix, not
  guessed -- `part2_train.py` ranks off-diagonal pairs by total
  misclassification count and prints the top pairs with a visual-similarity
  explanation (falls back to a generic explanation if the actual top pair
  isn't one of the commonly-known ones already documented in the script).

## Results (fill in from your run's REPORT.md / part2_summary.json)
- Train / val / test split sizes: 54,000 / 6,000 / 10,000
- Feature-extraction-only validation accuracy: `<fill in>`
- Fine-tuning required: `<yes/no -- fill in>`
- Final validation accuracy (after fine-tuning, if triggered): `<fill in>`
- **Final test-set accuracy: `<fill in>`** (target: >= 80%; if genuinely not
  reached after fine-tuning, report the real shortfall honestly along with
  the confusion-matrix diagnosis, per the brief -- never fabricate this number)
- Top confused category pairs (from the real confusion matrix): `<fill in
  the two pairs and counts printed by part2_train.py, plus the printed
  explanations>`

See `REPORT.md` (generated by `part2_train.py`) for the full confusion
matrix and per-class precision/recall table.

See `REPORT.md` for the full threshold-sweep tables and per-category /
per-payment-method breakdowns.
