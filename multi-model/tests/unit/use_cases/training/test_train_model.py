import torch
from use_cases.training.train_model import mixup_data, apply_warmup


def test_mixup_data_returns_tuple():
    """mixup_data returns a 4-tuple."""
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])
    result = mixup_data(images, labels, alpha=1.0)
    assert len(result) == 4


def test_apply_warmup():
    """apply_warmup modifies the optimizer learning rate."""
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    apply_warmup(optimizer, epoch=0, warmup_epochs=5, base_lr=0.1)
    assert optimizer.param_groups[0]["lr"] < 0.1
