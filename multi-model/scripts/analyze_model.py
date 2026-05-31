"""
Model analysis utility script.

Analyzes the FG_MFN model architecture, counting trainable parameters
and estimating inference latency.
"""

import argparse
import json
import logging
import time
from typing import Any, Dict

import torch

from lib.models.fg_mfn import FG_MFN

logger = logging.getLogger(__name__)

# Default number of sample inputs for latency benchmarking
DEFAULT_NUM_SAMPLES = 100

# Default batch size for benchmarking
DEFAULT_BATCH_SIZE = 1

# Default image input dimensions
DEFAULT_IMAGE_CHANNELS = 3
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_IMAGE_WIDTH = 224

# Default text sequence length
DEFAULT_SEQ_LENGTH = 128


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
    num_samples: int = DEFAULT_NUM_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    seq_length: int = DEFAULT_SEQ_LENGTH,
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
        batch_size, DEFAULT_IMAGE_CHANNELS,
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
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> Dict[str, Any]:
    """
    Analyze the model suitability for production.

    Creates the model from configuration, counts parameters, and
    optionally benchmarks inference latency.

    Args:
        cfg: Model configuration dictionary.
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
        latency_results = benchmark_inference(model, num_samples)
        result.update(latency_results)

    return result


def main() -> None:
    """
    Parse arguments and run model analysis.

    Loads the model configuration, runs analysis, and prints results.
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
        default=DEFAULT_NUM_SAMPLES,
        help="Number of sample inputs for latency benchmarking",
    )
    args = parser.parse_args()

    with open(args.config) as file_handle:
        cfg = json.load(file_handle)

    result = analyze_model(cfg, args.num_samples)

    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
