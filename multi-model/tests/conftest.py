"""
Pytest configuration for the multi-model test suite.

Adds the project root directory to sys.path and provides
comprehensive fixtures for unit and integration testing.
"""

import sys
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock

# Add the multi-model project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import numpy as np
import pandas as pd
from PIL import Image


# ============================================================================
# Session-Scoped Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Create persistent test data directory for the session."""
    tmpdir = Path(tempfile.mkdtemp(prefix="pytest_test_data_"))
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="session")
def device():
    """Get the device to use for testing (CPU or CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Function-Scoped Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_image():
    """Create a sample image tensor."""
    return torch.randn(3, 224, 224)


@pytest.fixture
def sample_images_batch():
    """Create a batch of sample images."""
    batch_size = 8
    return torch.randn(batch_size, 3, 224, 224)


@pytest.fixture
def sample_text_tokens():
    """Create sample text tokens."""
    batch_size = 4
    seq_len = 128
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }


@pytest.fixture
def sample_labels_single_attribute():
    """Create sample labels for single attribute."""
    batch_size = 8
    num_classes = 3
    return {"sentiment": torch.randint(0, num_classes, (batch_size,))}


@pytest.fixture
def sample_labels_multi_attribute():
    """Create sample labels for multiple attributes."""
    batch_size = 8
    return {
        "sentiment": torch.randint(0, 3, (batch_size,)),
        "theme": torch.randint(0, 5, (batch_size,)),
    }


@pytest.fixture
def sample_model_config():
    """Create a sample model configuration."""
    return {
        "IMAGE_BACKBONE": "resnet18",
        "TEXT_ENCODER": "distilbert-base-uncased",
        "HIDDEN_DIM": 256,
        "DROPOUT": 0.3,
        "FUSION_TYPE": "concat",
        "ATTRIBUTES": {
            "sentiment": {"num_classes": 3, "labels": ["negative", "neutral", "positive"]},
        },
    }


@pytest.fixture
def sample_training_config():
    """Create a sample training configuration."""
    return {
        "dataset_path": "/tmp/data",
        "train_csv": "train.csv",
        "val_csv": "val.csv",
        "images_dir": "images",
        "batch_size": 8,
        "num_epochs": 2,
        "learning_rate": 0.001,
        "warmup_epochs": 1,
        "mixup_alpha": 0.2,
        "early_stopping_patience": 5,
        "scheduler_type": "cosine",
    }


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer."""
    optimizer = Mock()
    optimizer.step = Mock()
    optimizer.zero_grad = Mock()
    optimizer.state_dict = Mock(return_value={})
    return optimizer


@pytest.fixture
def mock_criterion():
    """Create a mock criterion."""
    criterion = Mock()
    criterion.return_value = torch.tensor(0.5)
    return criterion


# ============================================================================
# Dataset Fixtures
# ============================================================================

@pytest.fixture
def sample_dataset_with_files(temp_dir):
    """Create a sample dataset with actual files."""
    # Create CSV
    csv_path = temp_dir / "data.csv"
    df = pd.DataFrame({
        "image_filename": [f"img_{i}.jpg" for i in range(5)],
        "label": np.random.randint(0, 3, 5),
        "theme": np.random.choice(["food", "fashion", "tech"], 5),
    })
    df.to_csv(csv_path, index=False)
    
    # Create images directory
    images_dir = temp_dir / "images"
    images_dir.mkdir()
    
    # Create dummy images
    for i in range(5):
        img = Image.new("RGB", (224, 224), color=(i*50, 100, 150))
        img.save(images_dir / f"img_{i}.jpg")
    
    return csv_path, images_dir


# ============================================================================
# Pytest Configuration & Hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    yield
