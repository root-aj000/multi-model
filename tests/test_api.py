import unittest
import os
import io
from fastapi.testclient import TestClient
from app.app import app
from tests.test_utils import create_dummy_image
import shutil

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.test_dir = "tests/temp_api"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a dummy image
        self.img_path = os.path.join(self.test_dir, "test_api.jpg")
        create_dummy_image(self.img_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        
        # FIX: Check specific keys instead of the whole dictionary
        # because 'timestamp' and 'uptime' change dynamically.
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["message"], "API is running")
        
        # Verify the extra fields exist (optional)
        self.assertIn("timestamp", data)
        self.assertIn("model_loaded", data)

    def test_predict_endpoint(self):
        # Open the image in binary mode
        with open(self.img_path, "rb") as f:
            # Prepare file upload
            files = {"files": ("test_api.jpg", f, "image/jpeg")}
            response = self.client.post("/predict", files=files)
            
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("predictions", data)
        self.assertTrue(len(data["predictions"]) > 0)
        
        pred = data["predictions"][0]
        self.assertIn("filename", pred)
        self.assertIn("predicted_label_text", pred)
        self.assertIn("confidence_score", pred)
        self.assertEqual(pred["filename"], "test_api.jpg")

if __name__ == "__main__":
    unittest.main()