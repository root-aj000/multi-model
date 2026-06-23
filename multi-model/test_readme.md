cd /home/aj-000/python_projects/m/02/backup/multi-model

# Basic usage — auto-detects checkpoint from config
python scripts/test_model.py --image ./0001.png

# Skip OCR (faster sanity check with empty text)
python scripts/test_model.py --image img.jpg --no-ocr

# Provide text directly (skip OCR)
python scripts/test_model.py --image 0001.png--text "BUY NOW 50% OFF"

# Explicit checkpoint
python scripts/test_model.py --image 0001.png \
  --checkpoint saved_models/best_model_epoch_14_acc_0.7228.pt

# Verbose debug output
python scripts/test_model.py --image img.jpg -v