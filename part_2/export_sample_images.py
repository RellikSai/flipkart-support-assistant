"""
Part 2, Task 8 -- Export real sample images as actual .png files.

torchvision.datasets.FashionMNIST stores its data as raw IDX binary files,
not a folder of individual images. Part 3's classify_product_image(image_path)
tool needs real image files to point at, so this script pulls at least 5 real
TEST-split images (covering different classes) and writes each one out as an
actual .png via PIL.Image.fromarray(...), named so the true label is obvious
from the filename.

Run this AFTER part2_train.py (it reuses the same torchvision cache dir).

Run:
    python3 export_sample_images.py
"""

import os

import numpy as np
from PIL import Image
from torchvision import datasets

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

OUT_DIR = "data/sample_images"
os.makedirs(OUT_DIR, exist_ok=True)

raw_test = datasets.FashionMNIST(root="data", train=False, download=True)
test_images = raw_test.data.numpy()   # (10000, 28, 28) uint8
test_labels = raw_test.targets.numpy()

# Pick one real example per class, in class order, so every export covers a
# different category (more informative for Part 3's demo transcripts than
# picking randomly and risking duplicate classes).
rng = np.random.default_rng(42)
chosen = []
for class_idx, class_name in enumerate(CLASS_NAMES):
    candidates = np.where(test_labels == class_idx)[0]
    pick = rng.choice(candidates)
    chosen.append((pick, class_idx, class_name))

for i, (idx, class_idx, class_name) in enumerate(chosen):
    img_array = test_images[idx]  # (28, 28) uint8
    safe_name = class_name.lower().replace("/", "-").replace(" ", "_")
    filename = f"{i:02d}_{safe_name}.png"
    path = os.path.join(OUT_DIR, filename)
    Image.fromarray(img_array).save(path)
    print(f"Saved {path}  (true label: {class_name}, test-set index: {idx})")

print(f"\nExported {len(chosen)} real test-split images to {OUT_DIR}/")
