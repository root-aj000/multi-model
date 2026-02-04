# import unittest
# import os
# import shutil
# from unittest.mock import patch, MagicMock
# # Import the module to path globals, but don't run prepare_dataset yet
# import dataset_generator.prepare_data as prep
# from tests.test_utils import create_dummy_image

# class TestDatasetGenerator(unittest.TestCase):
#     def setUp(self):
#         self.test_base = "tests/temp_generator"
#         self.raw_dir = os.path.join(self.test_base, "raw")
#         self.processed_dir = os.path.join(self.test_base, "processed")
#         self.log_dir = os.path.join(self.test_base, "logs")
        
#         os.makedirs(self.raw_dir, exist_ok=True)
#         # Create dummy raw images
#         for i in range(3):
#             create_dummy_image(os.path.join(self.raw_dir, f"raw_{i}.jpg"))

#     def tearDown(self):
#         if os.path.exists(self.test_base):
#             shutil.rmtree(self.test_base)

#     @patch("dataset_generator.prepare_data.extract_text")
#     def test_generation_logic(self, mock_extract):
#         # Mock OCR return value to avoid running actual PaddleOCR
#         mock_extract.return_value = ("DUMMY OCR TEXT", 0.99)
        
#         # Save original globals
#         orig_raw = prep.RAW_DATA_DIR
#         orig_proc = prep.PROCESSED_DATA_DIR
#         orig_out = prep.IMAGE_OUTPUT_DIR
#         orig_log = prep.LOG_DIR
        
#         try:
#             # Override globals to point to test dirs
#             prep.RAW_DATA_DIR = self.raw_dir
#             prep.PROCESSED_DATA_DIR = self.processed_dir
#             prep.IMAGE_OUTPUT_DIR = os.path.join(self.processed_dir, "images")
#             prep.LOG_DIR = self.log_dir
            
#             os.makedirs(prep.IMAGE_OUTPUT_DIR, exist_ok=True)
#             os.makedirs(prep.LOG_DIR, exist_ok=True)
            
#             # Run the function
#             prep.prepare_dataset()
            
#             # Verify outputs
#             self.assertTrue(os.path.exists(os.path.join(self.processed_dir, "train.csv")))
#             self.assertTrue(os.path.exists(os.path.join(self.processed_dir, "val.csv")))
            
#             # Verify CSV content
#             with open(os.path.join(self.processed_dir, "train.csv"), "r") as f:
#                 content = f.read()
#                 # Should contain the dummy text we mocked
#                 self.assertIn("DUMMY OCR TEXT", content)
            
#             # Verify images were processed (copied/renamed)
#             output_images = os.listdir(prep.IMAGE_OUTPUT_DIR)
#             self.assertTrue(len(output_images) > 0)
            
#         finally:
#             # Restore globals
#             prep.RAW_DATA_DIR = orig_raw
#             prep.PROCESSED_DATA_DIR = orig_proc
#             prep.IMAGE_OUTPUT_DIR = orig_out
#             prep.LOG_DIR = orig_log

# if __name__ == "__main__":
#     unittest.main()
