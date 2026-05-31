"""
Evaluation script for the FG_MFN model.

BUG-10 FIX: text_max_length now defaults to 256 (matching training config)
            instead of the previous hardcoded 128.
BUG-21 FIX: Uses lib.preprocessing.text.tokenizer.load_tokenizer exclusively
            (not the factory alias) to avoid the duplicate-name shadowing bug.
"""

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from lib.models.factory import load_model
from lib.preprocessing.dataset import load_dataset
from lib.preprocessing.image.transforms import build_image_transform
from lib.preprocessing.text.cleaner import clean_text
from lib.preprocessing.text.pipeline import build_text_pipeline
from lib.preprocessing.text.tokenizer import load_tokenizer, tokenize_text
from lib.utils.config import get_label_maps, load_config
from use_cases.training.evaluate import compute_metrics, evaluate_model, save_results

logger = logging.getLogger(__name__)

# Default batch size for evaluation
DEFAULT_BATCH_SIZE = 32

# Default output directory for results
DEFAULT_OUTPUT_DIR = "results"


def main() -> None:
    """
    Wire the evaluation use case and run evaluation.

    Parses command-line arguments, loads the model and dataset,
    runs evaluation, computes per-attribute metrics, and saves results.
    """
    parser = argparse.ArgumentParser(description="Evaluate the FG_MFN model")
    parser.add_argument("--config",     type=str, required=True,
                        help="Path to the evaluation configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--split",      type=str, default="test",
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--output",     type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for evaluation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = load_config(args.config)
    model_cfg = config.get("model", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load_model sets eval() mode (BUG-11 FIX)
    model = load_model(model_cfg, device, args.checkpoint)

    label_maps = get_label_maps(config)

    # Build the same image pipeline used during training.
    # image_size is read from config to match training exactly.
    raw_size = config.get("image_size", [512, 512])
    if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
        image_size = (int(raw_size[0]), int(raw_size[1]))
    elif isinstance(raw_size, int):
        image_size = (raw_size, raw_size)
    else:
        image_size = (512, 512)

    image_pipeline = build_image_transform(size=image_size, augment=False)

    text_pipeline = None
    if config.get("use_text", False):
        # BUG-21 FIX: use lib.preprocessing.text.tokenizer.load_tokenizer only
        tokenizer = load_tokenizer(
            config.get("TEXT_ENCODER", "distilbert-base-uncased")
        )
        # BUG-10 FIX: default to 256, not 128
        max_length = config.get("text_max_length", 256)

        # BUG-17 FIX: no redundant default arg in closure
        def tokenizer_fn(text: str) -> dict:
            return tokenize_text(text, max_length=max_length, tokenizer=tokenizer)

        text_pipeline = build_text_pipeline(clean_text, tokenizer_fn)

    dataset = load_dataset(
        args.split,
        label_maps=label_maps,
        image_pipeline=image_pipeline,
        text_pipeline=text_pipeline,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # BUG-05 FIX: pass text_max_length so evaluate_model uses the right seq_len
    text_max_length = config.get("text_max_length", 256)
    evaluation_results = evaluate_model(
        model, dataloader, device, text_max_length=text_max_length,
    )
    metrics = compute_metrics(
        evaluation_results["all_preds"],
        evaluation_results["all_labels"],
        per_attr_preds=evaluation_results.get("per_attr_preds"),
        per_attr_labels=evaluation_results.get("per_attr_labels"),
    )
    save_results(metrics, args.output)

    print(f"\nEvaluation complete on '{args.split}' split.")
    print(f"Macro-average accuracy: {metrics['accuracy']:.4f}")
    if "per_attribute" in metrics:
        print("\nPer-attribute accuracies:")
        for attr, acc in metrics["per_attribute"].items():
            print(f"  {attr:20s}: {acc:.4f}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
