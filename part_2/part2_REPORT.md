# Part 2 -- Product Image Categoriser: Full Run Report

Flipkart Order Intelligence & Support Assistant

---

## Task 1 -- Load Fashion-MNIST

| Split | Size | Notes |
|---|---|---|
| Train | 54,000 | |
| Validation | 6,000 | stratified out of the 60k train split |
| Test | 10,000 | untouched until final evaluation (Task 5) |

**Classes (10):** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## Task 2 -- Preprocessing for the Pretrained Backbone

- **Input size:** 224x224 (ResNet-18 standard)
- **Normalization:** ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- Grayscale channel replicated to 3 channels via `transforms.Grayscale(num_output_channels=3)`

---

## Task 3 -- Transfer-Learning Model (Feature Extraction)

**Backbone:** torchvision ResNet-18 (ImageNet-pretrained). All layers frozen for the feature-extraction stage; only the new classifier head is trained.

**Feature caching (CPU speed trick):**

| Split | Shape | Extraction time |
|---|---|---|
| Train | (54000, 512) | 142.8s |
| Validation | (6000, 512) | 15.5s |
| Test | (10000, 512) | 26.0s |

**Head:** single `Linear(512 -> 10)` layer
**Optimizer:** Adam, lr = 0.001, batch size = 256, epochs = 15

### Training curve (head on cached features)

| Epoch | Train loss | Val acc |
|---|---|---|
| 1 | 0.6764 | 0.8560 |
| 2 | 0.3984 | 0.8728 |
| 3 | 0.3605 | 0.8767 |
| 4 | 0.3411 | 0.8850 |
| 5 | 0.3278 | 0.8833 |
| 6 | 0.3191 | 0.8872 |
| 7 | 0.3112 | 0.8902 |
| 8 | 0.3048 | 0.8920 |
| 9 | 0.3001 | 0.8933 |
| 10 | 0.2968 | 0.8947 |
| 11 | 0.2917 | 0.8967 |
| 12 | 0.2874 | 0.8967 |
| 13 | 0.2852 | 0.8967 |
| 14 | 0.2820 | 0.8935 |
| 15 | 0.2795 | 0.8965 |

**Feature-extraction-only validation accuracy: `0.8965`**

---

## Task 4 -- Fine-Tuning Decision

Feature-extraction validation accuracy (`0.8965`) was **>= the 0.80 threshold**, so fine-tuning was **not required**.

| | Validation accuracy |
|---|---|
| Before (feature-extraction-only) | 0.8965 |
| After (no fine-tuning performed) | 0.8965 |

---

## Task 5 -- Final Test-Set Evaluation

### Final TEST-SET accuracy: **0.8876**  ✅ (exceeds the 80% requirement)

### Confusion matrix (rows = true label, columns = predicted label)

| True \ Pred | T-shirt/top | Trouser | Pullover | Dress | Coat | Sandal | Shirt | Sneaker | Bag | Ankle boot |
|---|---|---|---|---|---|---|---|---|---|---|
| **T-shirt/top** | 814 | 5 | 19 | 30 | 1 | 0 | 122 | 0 | 8 | 1 |
| **Trouser** | 2 | 973 | 2 | 18 | 1 | 0 | 3 | 0 | 1 | 0 |
| **Pullover** | 13 | 0 | 860 | 9 | 39 | 0 | 76 | 0 | 3 | 0 |
| **Dress** | 16 | 7 | 14 | 888 | 18 | 0 | 56 | 0 | 1 | 0 |
| **Coat** | 2 | 0 | 74 | 39 | 750 | 0 | 131 | 0 | 4 | 0 |
| **Sandal** | 0 | 0 | 0 | 0 | 0 | 948 | 1 | 38 | 2 | 11 |
| **Shirt** | 96 | 0 | 45 | 35 | 65 | 0 | 749 | 0 | 9 | 1 |
| **Sneaker** | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 963 | 4 | 23 |
| **Bag** | 1 | 0 | 2 | 3 | 0 | 1 | 6 | 0 | 986 | 1 |
| **Ankle boot** | 0 | 0 | 0 | 1 | 1 | 11 | 0 | 40 | 2 | 945 |

### Per-class precision / recall / F1

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| T-shirt/top | 0.8623 | 0.8140 | 0.8374 | 1000 |
| Trouser | 0.9878 | 0.9730 | 0.9804 | 1000 |
| Pullover | 0.8465 | 0.8600 | 0.8532 | 1000 |
| Dress | 0.8680 | 0.8880 | 0.8779 | 1000 |
| Coat | 0.8571 | 0.7500 | 0.8000 | 1000 |
| Sandal | 0.9773 | 0.9480 | 0.9624 | 1000 |
| Shirt | 0.6547 | 0.7490 | 0.6987 | 1000 |
| Sneaker | 0.9251 | 0.9630 | 0.9437 | 1000 |
| Bag | 0.9667 | 0.9860 | 0.9762 | 1000 |
| Ankle boot | 0.9623 | 0.9450 | 0.9536 | 1000 |

**Weakest class:** Shirt (precision 0.65, F1 0.70) — consistent with the confusion analysis below: Shirt is the single most-confused-with-everything class, pulling misclassifications from T-shirt/top, Coat, and Pullover alike.

---

## Task 6 -- Confusion Pattern Diagnosis

**Top confused category pairs (from the real confusion matrix):**

| Pair | Total misclassifications (both directions) |
|---|---|
| Shirt <-> T-shirt/top | 218 |
| Coat <-> Shirt | 196 |
| Pullover <-> Shirt | 121 |

### Shirt <-> T-shirt/top (218 misclassifications)
Shirt and T-shirt/top share the same basic torso silhouette — short sleeves, a straight hem, no strong texture cues at 28x28 grayscale resolution. The feature that actually distinguishes them (a collar and button placket) is only a few pixels wide, so it is easily lost in the downsampled, low-resolution representation the model sees.

### Coat <-> Shirt (196 misclassifications)
Shirt and Coat both occupy a similar torso-plus-sleeves silhouette envelope; a coat's extra bulk and longer hem are subtle at 28x28 resolution, so the model leans on overall shape rather than the fine cues (lapels, layering) a human would use.

**Note:** Shirt is the common thread across all three top-confused pairs (T-shirt/top, Coat, and Pullover all get confused with it), which matches Shirt having by far the lowest precision (0.65) of any class in the per-class table above — the model systematically over-predicts "Shirt" for ambiguous upper-body-garment images.

---

## Task 7 -- Save Artifact

Saved `models/product_classifier.pt` (state_dict + preprocessing metadata + class names).

---

## Summary

| Metric | Value | Requirement | Met? |
|---|---|---|---|
| Train / Val / Test split sizes | 54,000 / 6,000 / 10,000 | Val >= 5,000, test untouched | ✅ |
| Feature-extraction val accuracy | 0.8965 | >= 0.80 to skip fine-tuning | ✅ (fine-tuning not required) |
| Final test-set accuracy | **0.8876** | >= 0.80 | ✅ |
| Confusion matrix | Real, from model predictions | Not simulated | ✅ |
| Confused pairs named with explanation | Shirt<->T-shirt/top, Coat<->Shirt | >= 2 pairs | ✅ |
| Artifact saved | `models/product_classifier.pt` | Loadable via documented snippet | ✅ |
