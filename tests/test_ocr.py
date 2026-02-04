# test_ocr.py
from PIL import Image
from app.predict import extract_text, get_ocr_status

# Check status before
print("Before extraction:")
status = get_ocr_status()
print(f"  PaddleOCR initialized: {status['paddleocr']['initialized']}")

# Create a test image with text
img = Image.new('RGB', (200, 50), color='white')

# Extract text (this triggers initialization)
print("\nExtracting text...")
text, confidence = extract_text(img)
print(f"  Text: '{text}'")
print(f"  Confidence: {confidence}")

# Check status after
print("\nAfter extraction:")
status = get_ocr_status()
print(f"  PaddleOCR initialized: {status['paddleocr']['initialized']}")

# Check status before
print("Before extraction:")
status = get_ocr_status()
print(f"  EasyOCR initialized: {status['easyocr']['initialized']}")

# Create a test image with text
img = Image.new('RGB', (200, 50), color='white')

# Extract text (this triggers initialization)
print("\nExtracting text...")
text, confidence = extract_text(img)
print(f"  Text: '{text}'")
print(f"  Confidence: {confidence}")

# Check status after
print("\nAfter extraction:")
status = get_ocr_status()
print(f"  EasyOCR initialized: {status['easyocr']['initialized']}")