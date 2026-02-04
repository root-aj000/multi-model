import os
import cv2
import numpy as np
import pandas as pd
import random
import shutil

# Attribute names matching the model config
ATTRIBUTE_NAMES = [
    "theme", "sentiment", "emotion", "dominant_colour", "attention_score",
    "trust_safety", "target_audience", "predicted_ctr", "likelihood_shares"
]

# Number of classes per attribute (must match model_config.json)
ATTRIBUTE_CLASSES = {
    "theme": 10,
    "sentiment": 3,
    "emotion": 8,
    "dominant_colour": 10,
    "attention_score": 3,
    "trust_safety": 3,
    "target_audience": 8,
    "predicted_ctr": 3,
    "likelihood_shares": 3
}

# Label names for each attribute
ATTRIBUTE_LABELS = {
    "theme": ["Food", "Fashion", "Tech", "Health", "Travel", "Finance", "Entertainment", "Sports", "Education", "Other"],
    "sentiment": ["Positive", "Negative", "Neutral"],
    "emotion": ["Excitement", "Trust", "Joy", "Fear", "Anger", "Sadness", "Surprise", "Anticipation"],
    "dominant_colour": ["Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Black", "White", "Brown", "Multi"],
    "attention_score": ["High", "Medium", "Low"],
    "trust_safety": ["Safe", "Unsafe", "Questionable"],
    "target_audience": ["General", "Food Lovers", "Tech Enthusiasts", "Fashionistas", "Parents", "Professionals", "Fitness Enthusiasts", "Students"],
    "predicted_ctr": ["High", "Medium", "Low"],
    "likelihood_shares": ["High", "Medium", "Low"]
}

def create_dummy_image(path, size=(640, 480), color=(255, 255, 255)):
    """Creates a valid dummy JPG image (Document style)."""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = color
    # Add clear black text
    cv2.putText(img, "TEST DOCUMENT TEXT 123", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Another line of text", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def setup_dummy_data(base_dir, multi_attribute=True):
    """Sets up a temporary directory with dummy images and a csv.
    
    Args:
        base_dir: Directory to create test data in
        multi_attribute: If True, generate all attribute columns; if False, use legacy format
    """
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    image_dir = os.path.join(base_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    # Create 5 dummy images
    data = []
    for i in range(5):
        filename = f"img_{i}.jpg"
        path = os.path.join(image_dir, filename)
        create_dummy_image(path)
        
        row = {
            "image_path": filename,
            "text": f"This is a dummy text description {i}",
        }
        
        if multi_attribute:
            # Generate random labels for all attributes
            for attr in ATTRIBUTE_NAMES:
                num_classes = ATTRIBUTE_CLASSES[attr]
                label_idx = random.randint(0, num_classes - 1)
                row[f"{attr}_num"] = label_idx
                row[attr] = ATTRIBUTE_LABELS[attr][label_idx]
            
            # Add text-extracted fields
            row["keywords"] = "Test Dummy Keywords"
            row["monetary_mention"] = "50% OFF" if random.random() > 0.5 else "None"
            row["call_to_action"] = random.choice(["Order Now", "Shop Today", "Learn More", "None"])
            row["object_detected"] = random.choice(["Phone", "Food", "Clothing", "General"])
        else:
            # Legacy format
            row["label_num"] = random.choice([1, 2])  # 1: Neutral, 2: Positive
            row["label_text"] = "Positive" if row["label_num"] == 2 else "Neutral"
        
        data.append(row)
    
    csv_path = os.path.join(base_dir, "data.csv")
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    return csv_path, image_dir

def teardown_dummy_data(base_dir):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
