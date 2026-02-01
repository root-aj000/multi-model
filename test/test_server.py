import unittest
import os
import torch
import shutil
from PIL import Image
from tests.test_utils import setup_dummy_data, teardown_dummy_data
from app.predict import predict, model
from utils.path import MODEL_CONFIG

class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_data_predict"
        self.csv_path, self.image_dir = setup_dummy_data(self.test_dir)
        
        # Ensure we can load a dummy model state if needed, or just use initialized model
        # For this test, we might use the global model from server.predict which is already initialized
        # but we need to make sure it's in eval mode.
        pass

    def tearDown(self):
        teardown_dummy_data(self.test_dir)

    @unittest.mock.patch("server.predict.extract_text")
    def test_predict_function_execution(self, mock_extract):
        # Mock OCR output so we don't crash PaddleX on dummy images
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
            self.assertEqual(res["ocr_text"], "TEST DOCUMENT TEXT") # Verify mock was used
            self.assertTrue(isinstance(res["confidence_score"], float))

if __name__ == "__main__":
    unittest.main()
