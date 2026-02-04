import unittest
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tests.test_utils import setup_dummy_data, teardown_dummy_data, ATTRIBUTE_NAMES
from preprocessing.dataset import CustomDataset
from models.fg_mfn import FG_MFN

class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_data_training"
        self.csv_path, self.image_dir = setup_dummy_data(self.test_dir, multi_attribute=True)
        
        # Multi-attribute config matching model_config.json
        self.cfg = {
            "IMAGE_BACKBONE": "resnet50",
            "TEXT_ENCODER": "bert-base-uncased",
            "FUSION_TYPE": "concat",
            "DROPOUT": 0.1,
            "HIDDEN_DIM": 128,  # Smaller dim for speed
            "ATTRIBUTES": {
                "theme": {"num_classes": 10, "labels": ["Food", "Fashion", "Tech", "Health", "Travel", "Finance", "Entertainment", "Sports", "Education", "Other"]},
                "sentiment": {"num_classes": 3, "labels": ["Positive", "Negative", "Neutral"]},
                "emotion": {"num_classes": 8, "labels": ["Excitement", "Trust", "Joy", "Fear", "Anger", "Sadness", "Surprise", "Anticipation"]},
                "dominant_colour": {"num_classes": 10, "labels": ["Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Black", "White", "Brown", "Multi"]},
                "attention_score": {"num_classes": 3, "labels": ["High", "Medium", "Low"]},
                "trust_safety": {"num_classes": 3, "labels": ["Safe", "Unsafe", "Questionable"]},
                "target_audience": {"num_classes": 8, "labels": ["General", "Food Lovers", "Tech Enthusiasts", "Fashionistas", "Parents", "Professionals", "Fitness Enthusiasts", "Students"]},
                "predicted_ctr": {"num_classes": 3, "labels": ["High", "Medium", "Low"]},
                "likelihood_shares": {"num_classes": 3, "labels": ["High", "Medium", "Low"]}
            }
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FG_MFN(self.cfg).to(self.device)

    def tearDown(self):
        teardown_dummy_data(self.test_dir)

    def test_one_epoch_training(self):
        dataset = CustomDataset(self.csv_path, image_dir=self.image_dir)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
        self.model.train()
        
        print(f"\n[INFO] Starting Multi-Attribute Training Test...", flush=True)
        
        for i, batch in enumerate(loader):
            images = batch["visual"].to(self.device)
            texts = batch["text"].to(self.device)
            masks = batch["attention_mask"].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images, texts, attention_mask=masks)
            
            # Compute multi-task loss
            total_loss = 0
            num_attrs = 0
            
            for attr in ATTRIBUTE_NAMES:
                if attr in outputs and attr in batch:
                    labels = batch[attr].to(self.device)
                    loss = criterion(outputs[attr], labels)
                    total_loss += loss
                    num_attrs += 1
            
            self.assertGreater(num_attrs, 0, "No attributes found in outputs or batch")
            total_loss = total_loss / num_attrs
            
            total_loss.backward()
            optimizer.step()
            
            print(f"[INFO] Batch {i+1} processed. Loss: {total_loss.item():.4f}", flush=True)
        
        print(f"[INFO] Multi-Attribute Training Test Completed Successfully.", flush=True)
        
        # Check that we have all expected outputs
        self.assertEqual(len(outputs), len(ATTRIBUTE_NAMES), "Model should output all attributes")

if __name__ == "__main__":
    unittest.main()
