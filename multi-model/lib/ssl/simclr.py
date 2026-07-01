import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        projection_dim: int = 128,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.backbone = backbone

        backbone_out_dim = getattr(backbone, "native_dim", None)
        if backbone_out_dim is None:
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                backbone_out_dim = backbone(dummy).size(-1)

        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, projection_dim),
        )

    @staticmethod
    def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.size(0)
        z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0)
        sim = z @ z.T
        sim = sim / 0.5
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels])
        sim = sim - torch.eye(2 * batch_size, device=z.device, dtype=sim.dtype) * 1e9
        loss = F.cross_entropy(sim, labels)
        return loss

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.backbone(x1)
        h2 = self.backbone(x2)
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        loss = self.info_nce_loss(z1, z2)
        return loss, z1, z2
