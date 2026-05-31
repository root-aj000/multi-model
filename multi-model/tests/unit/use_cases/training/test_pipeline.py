"""
Tests for the training pipeline builder.
"""

import torch
from torch.utils.data import TensorDataset

from use_cases.training.pipeline import (
    create_data_loaders,
    setup_training_components,
)


def test_create_data_loaders_returns_tuple():
    """create_data_loaders returns a tuple of two DataLoaders."""
    dummy_data = TensorDataset(torch.randn(10, 3), torch.randint(0, 2, (10,)))
    train_loader, val_loader = create_data_loaders(
        dummy_data, dummy_data, batch_size=4, num_workers=0,
    )
    assert train_loader is not None
    assert val_loader is not None


def test_setup_training_components_returns_dict():
    """setup_training_components returns dict with optimizer, criterion, scheduler."""
    model = torch.nn.Linear(10, 2)
    cfg = {"learning_rate": 0.001}
    device = torch.device("cpu")
    components = setup_training_components(model, cfg, device)
    assert "optimizer" in components
    assert "criterion" in components
    assert "scheduler" in components


def test_setup_training_components_uses_config_lr():
    """setup_training_components uses the learning rate from config."""
    model = torch.nn.Linear(10, 2)
    cfg = {"learning_rate": 0.01}
    device = torch.device("cpu")
    components = setup_training_components(model, cfg, device)
    # Scheduler warmup modifies the active lr; check initial_lr which
    # PyTorch schedulers store as the original configured value.
    assert components["optimizer"].param_groups[0]["initial_lr"] == 0.01
