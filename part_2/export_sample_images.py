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
test_images = raw_test.data.numpy()  
test_labels = raw_test.targets.numpy()

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
