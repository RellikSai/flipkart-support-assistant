# Flipkart Support Assistant

The project is divided into three parts:

1. **Part 1 — Return Risk Prediction**
2. **Part 2 — Product Image Classification**
3. **Part 3 — Support Assistant / RAG Agent**

---

# Repository Structure

```text
flipkart-support-assistant/
│
├── README.md
│
├── part_1/
│   ├── generate_orders.py
│   ├── orders_dataset.csv
│   ├── part1_pipeline.py
│   ├── Part1_REPORT.md
│   ├── return_risk_model.pkl
│   └── t_star_rf.json
│
├── part_2/
│   ├── classify_image.py
│   ├── export_sample_images.py
│   ├── part2_train.py
│   ├── part2_summary.json
│   ├── part2_REPORT.md
│   ├── data/
│   │    └── sample_images/
│   └── models/
│        └── product_classifier.pt
│
└── part_3/
    ├── agent_graph.py
    ├── build_index.py
    ├── chunking.py
    ├── eval_retrieval.py
    ├── guardrails.py
    ├── knowledge_base.py
    ├── mock_llm.py
    ├── prompts.py
    ├── retriever.py
    ├── run_transcripts.py
    ├── tools.py
    ├── requirements.txt
    ├── models/
    │   ├── return_risk_model.pkl
    │   └── product_classifier.pt
    ├── data/
    │   └── sample_images/
    ├── index/
    │   ├── faiss.index
    │   └── chunk_meta.json
    └── transcripts/
```

# Part 1 -- Return-Risk Scoring Pipeline

## Files
- `generate_orders.py` -- exact seeded dataset generator (do not modify the seed or lists).
- `orders_dataset.csv` -- generated output (6000 rows x 13 columns).
- `part1_pipeline.py` -- runs Tasks 2-9 end to end (verification, preprocessing,
  baseline, Logistic Regression + threshold sweep, Random Forest + GridSearchCV,
  feature/permutation importance, subgroup analysis, final artifact save).
- `return_risk_model.pkl` -- final tuned Random Forest pipeline
  (preprocessing + model as one fitted sklearn `Pipeline`), saved with `joblib`.
- `t_star_rf.json` -- the F1-maximising threshold `t*_rf` for the saved Random
  Forest's own `predict_proba`, plus the Low/Medium/High cut points derived
  from it. This is what Part 3's `check_return_risk` tool reads.
- `Part1_REPORT.md` -- full captured output of the pipeline run (all numbers below
  are copied from this file, not hand-typed).

## How to run
```bash
pip install scikit-learn pandas numpy joblib

python3 part_1/generate_orders.py      # writes part_1/orders_dataset.csv (6000 rows)
python3 part_1/part1_pipeline.py       # writes/updates Part 1 artifacts
```

## Key results (from this run's Part1_REPORT.md)
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
- Final artifact: `part_1/return_risk_model.pkl` (tuned Random Forest
  pipeline), anchored threshold **t\*_rf = 0.46** -> Low if `p < 0.46`,
  High if `p >= 0.61`, else Medium.


# Part 2 -- Product Image Categoriser via Transfer Learning
 
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
  **Not yet run** -- see "Still to do" below.
- `classify_image.py` -- the documented one-function loading + single-image
  prediction snippet. This is exactly what Part 3's `classify_product_image`
  tool imports and calls -- not a reimplementation.
- `models/product_classifier.pt` -- written by `part2_train.py`: model
  weights (`state_dict`) plus preprocessing metadata (image size, ImageNet
  mean/std, class names) needed to reload and run inference.
- `part2_REPORT.md` / `part2_summary.json` -- written by `part2_train.py` from the
  actual training run (on Google Colab, T4 GPU). Real numbers, not placeholders.
## How to run
```bash
pip install torch torchvision scikit-learn pandas numpy pillow

python3 part_2/part2_train.py            # trains, evaluates, saves part_2/models/product_classifier.pt
python3 part_2/export_sample_images.py   # writes 10 real .png files to part_2/data/sample_images/
python3 part_2/classify_image.py part_2/data/sample_images/00_t-shirt-top.png   # smoke test
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
## Results (from the actual run -- see part2_REPORT.md for full detail)
- Train / val / test split sizes: 54,000 / 6,000 / 10,000
- Feature-extraction-only validation accuracy: **0.8965**
- Fine-tuning required: **No** (0.8965 >= the 0.80 threshold)
- Final validation accuracy: **0.8965** (unchanged, no fine-tuning performed)
- **Final test-set accuracy: 0.8876** (target: >= 80% -- met, with margin)
- Top confused category pairs (from the real confusion matrix):
  - **Shirt <-> T-shirt/top** (218 misclassifications, both directions) --
    both share the same basic torso silhouette at 28x28 grayscale resolution;
    the collar/placket detail that actually distinguishes them is only a
    few pixels wide and is lost in the downsampled representation.
  - **Coat <-> Shirt** (196 misclassifications) -- both occupy a similar
    torso-plus-sleeves silhouette envelope; a coat's extra bulk and longer
    hem are subtle at this resolution, so the model leans on overall shape
    rather than fine cues like lapels or layering.
  - Worth noting: **Shirt** is the common thread across the top 3 confused
    pairs (also confused with Pullover, 121 misclassifications) and has by
    far the lowest per-class precision (0.65) of any category -- the model
    systematically over-predicts "Shirt" for ambiguous upper-body garments.
See `part2_REPORT.md` for the full confusion matrix and per-class precision/recall
table.

# Part 3 -- Flipkart Support Agent

## Files
- `knowledge_base.py` -- 14 hand-written policy documents (>=12 required).
- `chunking.py` -- splits each doc into one chunk per sentence, keeping a
  chunk -> parent doc_id mapping.
- `build_index.py` -- embeds every chunk with `all-MiniLM-L6-v2`
  (sentence-transformers) and builds a FAISS `IndexFlatIP` over them.
- `retriever.py` -- loads that index, does top-k similarity search, and
  dedupes results to one row per parent document.
- `tools.py` -- `check_return_risk` (loads Part 1's
  `models/return_risk_model.pkl`, buckets anchored to `t_star_rf.json`) and
  `classify_product_image` (calls Part 2's `classify_image.py`).
- `guardrails.py` -- input-side prompt-injection regex filter, output-side
  groundedness check (min cosine similarity `0.35` for the real embeddings).
- `prompts.py` -- the system prompt (annotated against the 4S principles +
  role prompting) and the 2+ few-shot intent-classification examples.
- `mock_llm.py` -- the required deterministic `MOCK_LLM` mode: rule-based
  intent classifier (few-shot-informed) + templated answer composer. Zero
  network calls, zero API keys.
- `agent_graph.py` -- the actual LangGraph graph: 5 nodes
  (`guardrail -> intent -> [retrieve | tool_call] -> generate`), one
  conditional edge out of `intent`, and a `MemorySaver` checkpointer keyed
  by `thread_id` for real multi-turn state.
- `run_transcripts.py` -- runs the 8+ required test conversations against
  the real agent and writes them to `transcripts/`.
- `eval_retrieval.py` -- Precision@3/Recall@3 at the document level against
  a 7-query hand-labeled answer key (>=5 required), with per-query
  arithmetic printed.
- `t_star_rf.json`, `models/return_risk_model.pkl` -- copied in from Part 1.
- `models/product_classifier.pt`, `data/sample_images/*.png` -- copy these
  in from Part 2 once you've run it (not included here since Part 2 wasn't
  run in this sandbox).

## How to run (the real stack)

```bash
cd part_3

pip install -r requirements.txt

# one-time setup
python3 build_index.py          # embeds the KB, writes index/faiss.index

# make sure Part 1's and Part 2's artifacts are actually present:
# models/return_risk_model.pkl, t_star_rf.json   (from Part 1)
# models/product_classifier.pt                   (from Part 2)
# data/sample_images/*.png                       (from Part 2)
# classify_image.py                              (from Part 2, tools.py imports it)

python3 run_transcripts.py      # writes transcripts/*.md (MOCK_LLM mode, the default)
python3 eval_retrieval.py       # prints Precision@3 / Recall@3 per query + averages
```

`USE_LIVE_LLM` is not wired up in this submission -- every path in
`agent_graph.py` goes through `mock_llm.py`. That satisfies the brief on its
own (a live-LLM extension is explicitly optional and never scored), so I
kept scope to the required MOCK_LLM path rather than adding an unused flag.

## 4S + role prompting annotation
See the top of `prompts.py` -- each principle (Specific, Short, Surround,
Single) plus role prompting is annotated inline next to `SYSTEM_PROMPT`.

## Guardrails
- **Input-side (prompt injection):** `guardrails.check_prompt_injection`
  matches patterns like "ignore previous instructions", "pretend you are...",
  "reveal your system prompt". See `transcripts/05_prompt_injection_blocked.md`
  -- the agent does not comply, it deflects.
- **Output-side (groundedness):** `guardrails.check_groundedness` refuses to
  answer a policy question if the best retrieved chunk's cosine similarity
  is below `0.35` (the real embeddings), printing the actual score and
  threshold instead of letting the mock generator invent a policy. See
  `transcripts/09_ungrounded_refusal.md`.

## Retrieval evaluation (Task 10)

| Query | P@3 | R@3 |
|---|---|---|
| How many days do I have to return a pair of shoes I bought? | 0.33 | 1.00 |
| My laptop arrived with a cracked screen, can I return it? | 0.00 | 0.00 |
| I paid cash on delivery, when will my refund hit my account? | 0.50 | 1.00 |
| Does someone come pick up my return or do I have to ship it myself? | 1.00 | 1.00 |
| How fast is standard delivery to a metro city? | 0.50 | 1.00 |
| Can I exchange a t-shirt for a bigger size instead of a refund? | 0.50 | 1.00 |
| Is there a warranty on electronics after Flipkart's return window closes? | 0.50 | 1.00 |

**Average Precision@3: 0.476  |  Average Recall@3: 0.857**

## Transcripts
All 9 (exceeds the 8+ requirement) are in `transcripts/`, each showing the
user turn(s) and the agent's structured JSON answer:
1. `01_policy_cod_refund.md` -- policy question via RAG
2. `02_policy_reverse_pickup.md` -- policy question via RAG
3. `03_return_risk_tool_call.md` -- calls `check_return_risk` (real Part 1 model)
4. `04_image_classifier_tool_call.md` -- calls `classify_product_image`
5. `05_prompt_injection_blocked.md` -- injection attempt, deflected
6. `06_fewshot_routing.md` -- few-shot examples driving intent routing
7. `07_multiturn_state_carried.md` -- order id from turn 1 referenced in turn 3
8. `08_fresh_conversation_state_absent.md` -- new thread, state correctly absent
9. `09_ungrounded_refusal.md` -- groundedness check refuses rather than fabricating

Transcripts 7 and 8 are the required matching pair: same kind of follow-up
question ("what about X for it?" style), one on a thread that already has
`last_order_id` set, one on a brand-new thread where it's `None`.
