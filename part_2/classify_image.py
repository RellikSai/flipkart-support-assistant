"""
Part 2 -- documented loading + single-image prediction snippet.

This is the exact function Part 3's classify_product_image(image_path) tool
calls. It loads models/product_classifier.pt (saved by part2_train.py) and
returns a predicted category label plus confidence for one image file.

Usage:
    from classify_image import classify_image
    result = classify_image("data/sample_images/07_sneaker.png")
    # -> {"predicted_class": "Sneaker", "confidence": 0.94, "all_probs": {...}}
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models, transforms
from PIL import Image

_MODEL = None
_META = None


def _load_model(weights_path="models/product_classifier.pt"):
    """Lazily load and cache the trained model + its preprocessing metadata.
    Called once per process; subsequent classify_image() calls reuse it."""
    global _MODEL, _META
    if _MODEL is not None:
        return _MODEL, _META

    checkpoint = torch.load(weights_path, map_location="cpu")
    class_names = checkpoint["class_names"]

    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Linear(512, len(class_names))
    backbone.load_state_dict(checkpoint["state_dict"])
    backbone.eval()

    _MODEL = backbone
    _META = {
        "class_names": class_names,
        "img_size": checkpoint["img_size"],
        "mean": checkpoint["imagenet_mean"],
        "std": checkpoint["imagenet_std"],
    }
    return _MODEL, _META


def classify_image(image_path: str) -> dict:
    """Load a single image file from disk, run it through the saved
    transfer-learning classifier, and return the predicted category plus
    the model's confidence for that prediction.

    Args:
        image_path: path to a real image file (e.g. one of the .png files
            exported by export_sample_images.py into data/sample_images/).

    Returns:
        {
          "predicted_class": str,
          "confidence": float,          # softmax probability of the top class
          "all_probs": {class_name: float, ...},
        }
    """
    model, meta = _load_model()

    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((meta["img_size"], meta["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=meta["mean"], std=meta["std"]),
    ])

    img = Image.open(image_path)
    tensor = preprocess(img).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_idx = int(torch.argmax(probs).item())
    class_names = meta["class_names"]

    return {
        "predicted_class": class_names[top_idx],
        "confidence": float(probs[top_idx]),
        "all_probs": {name: float(p) for name, p in zip(class_names, probs)},
    }


if __name__ == "__main__":
    # Quick manual smoke test against one of the exported sample images.
    import sys
    import glob

    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("data/sample_images/*.png"))[0]
    result = classify_image(path)
    print(f"Image: {path}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
