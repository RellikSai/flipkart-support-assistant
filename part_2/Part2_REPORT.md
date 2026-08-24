# Part 2 -- Product Image Categoriser: Full Run Report

```

==============================================================================
TASK 1 -- LOAD FASHION-MNIST
==============================================================================
Train split size: 54000
Validation split size (stratified out of train): 6000
Test split size (untouched until final evaluation): 10000
Classes (10): ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

==============================================================================
TASK 2 -- PREPROCESSING FOR PRETRAINED BACKBONE
==============================================================================
Backbone input size used: 224x224 (documented ResNet-18 standard)
Normalization: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
Grayscale channel replicated to 3 channels via transforms.Grayscale(num_output_channels=3)

==============================================================================
TASK 3 -- BUILD TRANSFER-LEARNING MODEL (feature extraction)
==============================================================================
Backbone: torchvision ResNet-18 (ImageNet-pretrained). All layers frozen for the feature-extraction stage; only the new classifier head is trained.
Extracted train features: (54000, 512) in 3013.5s
Extracted val features: (6000, 512) in 343.9s
Extracted test features: (10000, 512) in 599.8s
Head: single Linear(512 -> 10) layer, optimizer=Adam, lr=0.001, batch_size=256, epochs=15

Training head on cached features (feature extraction stage)...
  epoch  1/15 | train loss 0.6764 | val acc 0.8560
  epoch  2/15 | train loss 0.3984 | val acc 0.8728
  epoch  3/15 | train loss 0.3605 | val acc 0.8767
  epoch  4/15 | train loss 0.3411 | val acc 0.8850
  epoch  5/15 | train loss 0.3278 | val acc 0.8833
  epoch  6/15 | train loss 0.3191 | val acc 0.8872
  epoch  7/15 | train loss 0.3112 | val acc 0.8902
  epoch  8/15 | train loss 0.3048 | val acc 0.8920
  epoch  9/15 | train loss 0.3001 | val acc 0.8933
  epoch 10/15 | train loss 0.2968 | val acc 0.8947
  epoch 11/15 | train loss 0.2917 | val acc 0.8967
  epoch 12/15 | train loss 0.2874 | val acc 0.8967
  epoch 13/15 | train loss 0.2852 | val acc 0.8967
  epoch 14/15 | train loss 0.2820 | val acc 0.8935
  epoch 15/15 | train loss 0.2795 | val acc 0.8965

Feature-extraction-only validation accuracy: 0.8965

==============================================================================
TASK 4 -- FINE-TUNE (only if needed)
==============================================================================
Feature-extraction val accuracy 0.8965 >= 0.8. Fine-tuning was NOT required.

Before (feature-extraction-only) val accuracy: 0.8965
After (no fine-tuning performed) val accuracy: 0.8965

==============================================================================
TASK 5 -- FINAL TEST-SET EVALUATION
==============================================================================
Final TEST-SET accuracy: 0.8876

Confusion matrix (rows = true, cols = predicted):
             T-shirt/top  Trouser  Pullover  Dress  Coat  Sandal  Shirt  Sneaker  Bag  Ankle boot
T-shirt/top          814        5        19     30     1       0    122        0    8           1
Trouser                2      973         2     18     1       0      3        0    1           0
Pullover              13        0       860      9    39       0     76        0    3           0
Dress                 16        7        14    888    18       0     56        0    1           0
Coat                   2        0        74     39   750       0    131        0    4           0
Sandal                 0        0         0      0     0     948      1       38    2          11
Shirt                 96        0        45     35    65       0    749        0    9           1
Sneaker                0        0         0      0     0      10      0      963    4          23
Bag                    1        0         2      3     0       1      6        0  986           1
Ankle boot             0        0         0      1     1      11      0       40    2         945

Per-class precision/recall/F1:
      class  precision  recall       f1  support
T-shirt/top   0.862288   0.814 0.837449     1000
    Trouser   0.987817   0.973 0.980353     1000
   Pullover   0.846457   0.860 0.853175     1000
      Dress   0.868035   0.888 0.877904     1000
       Coat   0.857143   0.750 0.800000     1000
     Sandal   0.977320   0.948 0.962437     1000
      Shirt   0.654720   0.749 0.698694     1000
    Sneaker   0.925072   0.963 0.943655     1000
        Bag   0.966667   0.986 0.976238     1000
 Ankle boot   0.962322   0.945 0.953582     1000

==============================================================================
TASK 6 -- CONFUSION PATTERN DIAGNOSIS
==============================================================================
Top confused category pairs (from this run's real confusion matrix):
  Shirt <-> T-shirt/top: 218 total misclassifications (both directions)
  Shirt <-> Coat: 196 total misclassifications (both directions)
  Shirt <-> Pullover: 121 total misclassifications (both directions)

Explanations for the top pairs:

Shirt <-> T-shirt/top (218 misclassifications):
Shirt and T-shirt/top share the same basic torso silhouette -- short sleeves, a straight hem, no strong texture cues at 28x28 grayscale resolution. The feature that actually distinguishes them (a collar and button placket) is only a few pixels wide, so it is easily lost in the downsampled, low-resolution representation the model sees.

Shirt <-> Coat (196 misclassifications):
Shirt and Coat both occupy a similar torso-plus-sleeves silhouette envelope; a coat's extra bulk and longer hem are subtle at 28x28 resolution, so the model leans on overall shape rather than the fine cues (lapels, layering) a human would use.

==============================================================================
TASK 7 -- SAVE ARTIFACT
==============================================================================
Saved models/product_classifier.pt (state_dict + preprocessing metadata + class names)
```
