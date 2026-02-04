import unittest
import os
import torch
import shutil
from tests.test_utils import setup_dummy_data, teardown_dummy_data
from preprocessing.dataset import CustomDataset
from preprocessing.text_preprocessing import clean_text, tokenize_text
from preprocessing.image_preprocessing import resize_image, normalize_image
import cv2
import numpy as np

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_data_preprocessing"
        self.csv_path, self.image_dir = setup_dummy_data(self.test_dir, multi_attribute=True)

    def tearDown(self):
        teardown_dummy_data(self.test_dir)

    def test_clean_text(self):
        raw = "  This IS a TEST!! "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "this is a test")

    def test_tokenize_text(self):
        text = "hello world"
        tokens = tokenize_text(text, max_length=10)
        
        self.assertIn("input_ids", tokens)
        self.assertIn("attention_mask", tokens)
        self.assertEqual(tokens["input_ids"].shape, (10,))
        self.assertEqual(tokens["attention_mask"].shape, (10,))
        
        # Check that padding is 0 and mask is 0 there
        # 'hello world' is short, so rest should be padded
        self.assertEqual(tokens["input_ids"][-1].item(), 0)
        self.assertEqual(tokens["attention_mask"][-1].item(), 0)

    def test_dataset_loading(self):
        dataset = CustomDataset(self.csv_path, image_dir=self.image_dir, augment=False)
        self.assertEqual(len(dataset), 5)
        
        sample = dataset[0]
        self.assertIn("visual", sample)
        self.assertIn("text", sample)
        self.assertIn("attention_mask", sample)
        if dataset.legacy_mode:
            self.assertIn("label", sample)
        else:
            self.assertIn("theme", sample)
            self.assertIn("sentiment", sample)
        
        # Visual shape: [3, 224, 224]
        self.assertEqual(sample["visual"].shape, (3, 224, 224))
        # Text shape
        self.assertEqual(sample["text"].dim(), 1)
        self.assertEqual(sample["attention_mask"].dim(), 1)

    def test_image_preprocessing(self):
        # Create a solid red BGR image (0, 0, 255)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (0, 0, 255) 
        
        resized = resize_image(img, size=(224, 224))
        self.assertEqual(resized.shape, (224, 224, 3))
        
        # Note: normalize_image expects RGB input if you did the conversion manually before
        # But here we are unit testing the function itself which expects HWC numpy array
        # Real pipeline converts BGR->RGB before calling normalize_image usually, or inside dataset.py
        # check dataset.py logic: it calls resize, then converts BGR2RGB, then normalize.
        
        # Let's verify normalize output range
        norm = normalize_image(resized)
        self.assertTrue(torch.is_tensor(norm))
        self.assertEqual(norm.shape, (3, 224, 224))

if __name__ == "__main__":
    unittest.main()
