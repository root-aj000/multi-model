"""
Model analysis utility script.

Analyzes the FG_MFN model architecture, counting trainable parameters
and estimating inference latency. Benchmark parameters (num_samples,
batch size, seq length) are read from model_config.json (analyze section)
and can be overridden via CLI flags.
"""

import argparse
import json
import logging
import time
from typing import Any, Dict

import torch

from lib.models.fg_mfn import FG_MFN

logger = logging.getLogger(__name__)

# Fixed physical constant — not user-configurable
_IMAGE_CHANNELS = 3


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.

    Args:
        model: The PyTorch model to count parameters for.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: torch.nn.Module) -> int:
    """
    Count the total number of all parameters in a model (including frozen).

    Args:
        model: The PyTorch model to count parameters for.

    Returns:
        Total number of all parameters.
    """
    return sum(p.numel() for p in model.parameters())


def benchmark_inference(
    model: torch.nn.Module,
    num_samples: int = 100,
    batch_size: int = 1,
    image_height: int = 224,
    image_width: int = 224,
    seq_length: int = 128,
) -> Dict[str, float]:
    """
    Benchmark inference latency on random inputs.

    Args:
        model: The model to benchmark.
        num_samples: Number of sample forward passes to run.
        batch_size: Batch size for each forward pass.
        image_height: Height of dummy images.
        image_width: Width of dummy images.
        seq_length: Length of dummy text sequences.

    Returns:
        Dictionary with 'mean_latency_ms' and 'throughput_samples_per_sec'.
    """
    model.eval()
    device = next(model.parameters()).device

    dummy_images = torch.randn(
        batch_size, _IMAGE_CHANNELS,
        image_height, image_width,
    ).to(device)
    dummy_input_ids = torch.randint(
        0, 1000, (batch_size, seq_length),
    ).to(device)
    dummy_attention_mask = torch.ones(
        batch_size, seq_length,
    ).to(device)

    latencies = []
    with torch.no_grad():
        for _ in range(num_samples):
            start_time = time.perf_counter()
            model(dummy_images, dummy_input_ids, dummy_attention_mask)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

    mean_latency = sum(latencies) / len(latencies)
    throughput = (batch_size / mean_latency) * 1000

    return {
        "mean_latency_ms": mean_latency,
        "throughput_samples_per_sec": throughput,
    }


def analyze_model(
    cfg: Dict[str, Any],
    num_samples: int = 100,
) -> Dict[str, Any]:
    """
    Analyze the model suitability for production.

    Creates the model from configuration, counts parameters, and
    optionally benchmarks inference latency. Image size and sequence
    length for benchmarking are read from cfg (image_size and
    analyze.benchmark_seq_length) when available.

    Args:
        cfg: Model configuration dictionary (full config, not just model keys).
        num_samples: Number of sample inputs for latency analysis.

    Returns:
        Dictionary containing analysis metrics including parameter
        counts and optional latency benchmarks.
    """
    model = FG_MFN(cfg)
    trainable_params = count_parameters(model)
    total_params = count_total_parameters(model)

    result = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "num_attribute_heads": len(model.attribute_heads),
        "attribute_names": list(model.attribute_heads.keys()),
    }

    if num_samples > 0:
        analyze_cfg = cfg.get("analyze", {})
        raw_size = cfg.get("image_size", [224, 224])
        if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
            img_w, img_h = int(raw_size[0]), int(raw_size[1])
        elif isinstance(raw_size, int):
            img_w = img_h = raw_size
        else:
            img_w = img_h = 224

        latency_results = benchmark_inference(
            model,
            num_samples=num_samples,
            batch_size=analyze_cfg.get("benchmark_batch_size", 1),
            image_height=img_h,
            image_width=img_w,
            seq_length=analyze_cfg.get("benchmark_seq_length", 128),
        )
        result.update(latency_results)

    return result


def main() -> None:
    """
    Parse arguments and run model analysis.

    Loads the model configuration, runs analysis, and prints results.
    Benchmark parameters default to values in config (analyze section).
    """
    parser = argparse.ArgumentParser(
        description="Analyze model parameters and size",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model configuration",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of sample inputs for latency benchmarking (overrides config analyze.num_samples)",
    )
    args = parser.parse_args()

    with open(args.config) as file_handle:
        cfg = json.load(file_handle)

    analyze_cfg = cfg.get("analyze", {})
    num_samples = (
        args.num_samples
        if args.num_samples is not None
        else analyze_cfg.get("num_samples", 100)
    )

    result = analyze_model(cfg, num_samples)

    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
