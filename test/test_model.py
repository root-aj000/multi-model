import unittest
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.fg_mfn import FG_MFN, ATTRIBUTE_NAMES

class TestModel(unittest.TestCase):
    def setUp(self):
        # Multi-attribute config
        self.cfg = {
            "IMAGE_BACKBONE": "resnet50",
            "TEXT_ENCODER": "bert-base-uncased",
            "FUSION_TYPE": "concat",
            "DROPOUT": 0.1,
            "HIDDEN_DIM": 256,
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
        self.device = "cpu"  # Test on CPU for CI/local speed
        self.model = FG_MFN(self.cfg).to(self.device)

    def test_forward_pass(self):
        batch_size = 2
        
        # Dummy inputs
        images = torch.randn(batch_size, 3, 224, 224).to(self.device)
        texts = torch.randint(0, 1000, (batch_size, 64)).to(self.device)
        attention_mask = torch.ones((batch_size, 64)).to(self.device)
        
        outputs = self.model(images, texts, attention_mask=attention_mask)
        
        # Check that outputs is a dictionary
        self.assertIsInstance(outputs, dict, "Output should be a dictionary")
        
        # Check that all expected attributes are present
        for attr in ATTRIBUTE_NAMES:
            self.assertIn(attr, outputs, f"Output should contain {attr}")
        
        # Check shapes for each attribute
        for attr, logits in outputs.items():
            expected_classes = self.cfg["ATTRIBUTES"][attr]["num_classes"]
            self.assertEqual(logits.shape, (batch_size, expected_classes), 
                           f"{attr} should have shape [batch, {expected_classes}]")
    
    def test_backwards_compatibility(self):
        """Test that model works with legacy single-class config."""
        legacy_cfg = {
            "IMAGE_BACKBONE": "resnet50",
            "TEXT_ENCODER": "bert-base-uncased",
            "FUSION_TYPE": "concat",
            "NUM_CLASSES": 2,
            "DROPOUT": 0.1,
            "HIDDEN_DIM": 256
        }
        model = FG_MFN(legacy_cfg).to(self.device)
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(self.device)
        texts = torch.randint(0, 1000, (batch_size, 64)).to(self.device)
        
        outputs = model(images, texts)
        
        # Should have at least sentiment output
        self.assertIn("sentiment", outputs)
        self.assertEqual(outputs["sentiment"].shape, (batch_size, 2))

if __name__ == "__main__":
    unittest.main()
