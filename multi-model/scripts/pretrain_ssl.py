import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from lib.ssl.simclr import SimCLR


class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=1e-4, eta=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, eta=eta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            eta = group["eta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd > 0:
                    grad = grad.add(p, alpha=wd)
                p_norm = p.norm()
                g_norm = grad.norm()
                trust_ratio = 1.0
                if p_norm > 0 and g_norm > 0:
                    trust_ratio = eta * p_norm / (g_norm + 1e-8)
                p.add_(grad, alpha=-lr * trust_ratio)
        return loss


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=None, sigma=(0.1, 2.0)):
        super().__init__()
        self.sigma = sigma
        if kernel_size is None:
            kernel_size = int(0.1 * 224)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size

    def forward(self, x):
        sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()
        k = self.kernel_size
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_1d = gauss.view(1, 1, -1)
        C = x.size(1)
        kernel_3d = kernel_1d.expand(C, 1, -1)
        pad = k // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        x = F.conv2d(x, kernel_3d.unsqueeze(2), groups=C)
        x = F.conv2d(x, kernel_3d.unsqueeze(3), groups=C)
        return x


class SimCLRTransform:
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def adjust_learning_rate(optimizer, epoch, warmup_epochs, total_epochs, base_lr, batch_size):
    if epoch < warmup_epochs:
        lr = base_lr * batch_size / 256 * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = base_lr * batch_size / 256 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def main():
    parser = argparse.ArgumentParser(description="SimCLR pretraining")
    parser.add_argument("--backbone", default="resnet50", type=str)
    parser.add_argument("--projection-dim", default=128, type=int)
    parser.add_argument("--temperature", default=0.5, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=0.3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = models.__dict__[args.backbone](pretrained=False)
    backbone.fc = nn.Identity()
    model = SimCLR(backbone, args.projection_dim, args.temperature)
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    transform = SimCLRTransform()
    dataset = ImageFolder(root=args.dataset_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, os.cpu_count() or 4),
        pin_memory=True,
        drop_last=True,
    )

    for epoch in range(args.epochs):
        adjust_learning_rate(
            optimizer, epoch, warmup_epochs=10,
            total_epochs=args.epochs, base_lr=args.lr,
            batch_size=args.batch_size,
        )
        model.train()
        total_loss = 0.0
        for images, _ in loader:
            x1, x2 = images
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            loss, _, _ = model(x1, x2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x1.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1:3d}/{args.epochs}]  loss: {avg_loss:.4f}")

    if isinstance(model, nn.DataParallel):
        backbone_state = model.module.backbone.state_dict()
    else:
        backbone_state = model.backbone.state_dict()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(backbone_state, args.output_path)
    print(f"Backbone saved to {args.output_path}")


if __name__ == "__main__":
    main()
