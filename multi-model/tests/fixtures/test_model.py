"""
CLI script to test the trained FG_MFN model on a single image.

Mirrors the FastAPI /predict pipeline but runs entirely from the
command line — no auth, no server, no HTTP.

Usage:
    python scripts/test_model.py --image path/to/image.jpg
    python scripts/test_model.py --image img.jpg --config configs/model/model_config.json
    python scripts/test_model.py --image img.jpg --checkpoint saved_models/best.pt
    python scripts/test_model.py --image img.jpg --no-ocr   # skip OCR (use empty text)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

# Allow running from project root or from multi-model/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.models.factory import load_model
from lib.ocr.factory import create_ocr_engine
from lib.preprocessing.image.transforms import build_image_transform
from lib.preprocessing.text.cleaner import clean_text
from lib.preprocessing.text.tokenizer import load_tokenizer, tokenize_text
from lib.services.predictor import Predictor
from lib.utils.config import get_label_maps, load_config
from use_cases.prediction.predict_image import predict_image

logger = logging.getLogger(__name__)


def _resolve_image_size(config: dict) -> tuple:
    """Read image_size from config, defaulting to (512, 512)."""
    raw = config.get("image_size", [512, 512])
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (int(raw[0]), int(raw[1]))
    if isinstance(raw, int):
        return (raw, raw)
    return (512, 512)


def _resolve_checkpoint(config: dict, explicit: str | None) -> str:
    """Resolve checkpoint path using the same priority as the API."""
    if explicit:
        return explicit

    cfg_path = config.get("checkpoint_path")
    if cfg_path:
        return cfg_path

    ckpt_dir = Path(config.get("checkpoint_dir", "saved_models"))
    best = sorted(ckpt_dir.glob("best_model_*.pt"))
    if best:
        return str(best[-1])

    last = ckpt_dir / "last_model.pt"
    if last.exists():
        return str(last)

    any_pt = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if any_pt:
        return str(any_pt[0])

    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_dir}. "
        "Pass --checkpoint or set CHECKPOINT_PATH."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the FG_MFN model on a single image")
    parser.add_argument("--image", "-i", type=str, required=True,
                        help="Path to the input image (jpg/png/etc.)")
    parser.add_argument("--config", "-c", type=str,
                        default="configs/model/model_config.json",
                        help="Path to model_config.json")
    parser.add_argument("--checkpoint", "-k", type=str, default=None,
                        help="Path to .pt checkpoint (overrides config)")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Skip OCR and use empty text (faster, useful for sanity check)")
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="Override OCR text directly (skips OCR)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print debug logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 1

    # Resolve config path relative to project root (multi-model/), not cwd
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    print(f"Loading config: {config_path}")
    config = load_config(str(config_path))

    # Resolve checkpoint path relative to project root
    checkpoint_path = _resolve_checkpoint(config, args.checkpoint)
    ckpt = Path(checkpoint_path)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1
    checkpoint_path = str(ckpt)
    print(f"Using checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model_cfg = config.get("model", config)
    model = load_model(model_cfg, device, checkpoint_path)
    label_maps = get_label_maps(config)

    # Build image transform (must match training)
    image_size = _resolve_image_size(config)
    image_pipeline = build_image_transform(size=image_size, augment=False)

    # Load tokenizer if text is used
    text_max_length = config.get("text_max_length", 256)
    tokenizer = None
    if config.get("use_text", False):
        tokenizer = load_tokenizer(
            config.get("TEXT_ENCODER", "distilbert-base-uncased")
        )

    # OCR
    if args.text is not None:
        extracted_text = args.text
        ocr_confidence = 1.0
        print(f"Using provided text (--text): {extracted_text!r}")
    elif args.no_ocr:
        extracted_text = ""
        ocr_confidence = 0.0
        print("Skipping OCR (--no-ocr); using empty text")
    else:
        ocr_cfg = config.get("ocr", {})
        engine_name = ocr_cfg.get("engine", "easyocr")
        model_dir = Path(ocr_cfg.get("model_dir", "local/ocr"))
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
        language = ocr_cfg.get("language", "en")
        ocr_engine = create_ocr_engine(
            engine_name, model_dir,
            language=language,
            languages=[language],
        )
        # EasyOCR expects a numpy array (or file path / bytes), not a PIL Image.
        # Pass the file path directly — it's the most reliable input.
        print(f"Running OCR ({engine_name}, lang={language})...")
        extracted_text, ocr_confidence = ocr_engine.extract_text(str(image_path))


        
        print(f"OCR text: {extracted_text!r}  (confidence={ocr_confidence:.3f})")

    # Run prediction through the same use-case the API uses.
    # predict_image() always calls OCR internally, so we always pass the
    # ocr_engine. (We don't pre-run OCR here — let the use case handle it
    # exactly the way the API does, to keep behaviour identical.)
    pil_image = Image.open(image_path).convert("RGB")
    predictor = Predictor(model, label_maps)

    result = predict_image(
        image=pil_image,
        model=model,
        ocr_engine=ocr_engine,
        label_maps=label_maps,
        filename=image_path.name,
        config=config,
        predictor=predictor,
    )

    # Print result
    print("\n" + "=" * 60)
    print(f"Prediction for: {image_path}")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
