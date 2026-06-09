import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    AlexNet_Weights,
    DenseNet121_Weights,
    MobileNet_V2_Weights,
    ResNet18_Weights,
    ViT_B_16_Weights,
)
from typing import Callable


class MobileNetV2Features(nn.Module):
    def __init__(self, use_pretrained: bool) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if use_pretrained else None
        model = models.mobilenet_v2(weights=weights)
        self.features = model.features
        self.feature_dim = model.last_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.mean([2, 3])


class ResNet18Features(nn.Module):
    def __init__(self, use_pretrained: bool) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        model = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = model.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


class DenseNet121Features(nn.Module):
    def __init__(self, use_pretrained: bool) -> None:
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if use_pretrained else None
        model = models.densenet121(weights=weights)
        self.features = model.features
        self.feature_dim = model.classifier.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.relu(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)


class AlexNetFeatures(nn.Module):
    def __init__(self, use_pretrained: bool) -> None:
        super().__init__()
        weights = AlexNet_Weights.DEFAULT if use_pretrained else None
        model = models.alexnet(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier_features = nn.Sequential(*list(model.classifier.children())[:-1])
        self.feature_dim = model.classifier[-1].in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier_features(x)


class ViTB16Features(nn.Module):
    def __init__(self, use_pretrained: bool) -> None:
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if use_pretrained else None
        self.model = models.vit_b_16(weights=weights)
        self.feature_dim = self.model.heads.head.in_features
        self.model.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


FEATURE_EXTRACTOR_BUILDERS: dict[str, type] = {
    "mobilenet_v2": MobileNetV2Features,
    "resnet18": ResNet18Features,
    "densenet121": DenseNet121Features,
    "alexnet": AlexNetFeatures,
    "vit_b_16": ViTB16Features,
}


def build_feature_extractor(model_name: str, use_pretrained: bool) -> tuple[nn.Module, int]:
    cls = FEATURE_EXTRACTOR_BUILDERS[model_name]
    extractor = cls(use_pretrained)
    return extractor, extractor.feature_dim  # type: ignore[union-attr]
