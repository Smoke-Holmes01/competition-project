import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class RetrievalEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int | None = None,
        backbone_name: str = "resnet18",
        embedding_dim: int | None = None,
        pretrained: bool = False,
        normalize: bool | None = None,
    ):
        super().__init__()

        if backbone_name != "resnet18":
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.output_dim = embedding_dim if embedding_dim is not None else (num_classes or 46)
        self.normalize = embedding_dim is not None if normalize is None else normalize
        self.fc = nn.Linear(in_features, self.output_dim)

    def forward(self, x):
        feat = self.backbone(x)
        output = self.fc(feat)
        if self.normalize:
            output = F.normalize(output, p=2, dim=1)
        return output
