"""
Generate a small test dataset for training validation.

Creates CSV files and small random images under dataset/train and dataset/val.
"""

import os
import random

import numpy as np
from PIL import Image

# Attribute definitions: name -> number of classes
ATTRIBUTES = {
    "theme": 10,
    "sentiment": 3,
    "emotion": 8,
    "dominant_colour": 10,
    "attention_score": 3,
    "trust_safety": 3,
    "target_audience": 8,
    "predicted_ctr": 3,
    "likelihood_shares": 3,
}

# Number of samples per split
TRAIN_SAMPLES = 80
VAL_SAMPLES = 20

# Image dimensions
IMG_SIZE = 64


def generate_split(split_name: str, num_samples: int) -> None:
    """Generate CSV and images for one split."""
    csv_dir = f"dataset/{split_name}"
    img_dir = f"{csv_dir}/images"
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    for i in range(num_samples):
        filename = f"img_{i:04d}.png"
        # Create a small random RGB image
        arr = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(img_dir, filename))

        row = {"image_filename": filename}
        for attr, num_classes in ATTRIBUTES.items():
            row[attr] = random.randint(0, num_classes - 1)
        rows.append(row)

    # Write CSV
    import csv
    csv_path = os.path.join(csv_dir, f"{split_name}.csv")
    fieldnames = ["image_filename"] + list(ATTRIBUTES.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {num_samples} samples in {csv_dir}/")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    generate_split("train", TRAIN_SAMPLES)
    generate_split("val", VAL_SAMPLES)
    print("Done.")
