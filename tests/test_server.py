import unittest
import os
import torch
import shutil
from PIL import Image
from tests.test_utils import setup_dummy_data, teardown_dummy_data
# Note: You are importing from old_predict here
from app.old_predict import predict, model
from utils.path import MODEL_CONFIG
from unittest.mock import patch

class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_data_predict"
        self.csv_path, self.image_dir = setup_dummy_data(self.test_dir)

    def tearDown(self):
        teardown_dummy_data(self.test_dir)

    # FIX 1: It is safer to patch 'app.old_predict.extract_text' because that is 
    # the file where the function is actually being run.
    @patch("app.old_predict.extract_text")
    def test_predict_function_execution(self, mock_extract):
        # Mock OCR output
        mock_extract.return_value = ("TEST DOCUMENT TEXT", 0.99)
        
        # Load a few images
        images = []
        image_files = os.listdir(self.image_dir)[:3]
        for f in image_files:
            img_path = os.path.join(self.image_dir, f)
            images.append(Image.open(img_path).convert("RGB"))
            
        # Run predict
        try:
            results = predict(images)
        except Exception as e:
            self.fail(f"Prediction failed with error: {e}")
            
        self.assertEqual(len(results), len(images))
        
        for res in results:
            self.assertIn("predicted_label_text", res)
            self.assertIn("confidence_score", res)
            self.assertIn("ocr_text", res)
            
            # FIX 2: Use assertIn or startswith
            # The model combines OCR text with dummy metadata, so exact match fails.
            self.assertIn("TEST DOCUMENT TEXT", res["ocr_text"]) 
            
            self.assertTrue(isinstance(res["confidence_score"], float))

if __name__ == "__main__":
    unittest.main()