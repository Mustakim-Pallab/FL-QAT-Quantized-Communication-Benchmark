import torch.nn as nn
from torchvision import models
from torchvision.models import (
    AlexNet_Weights,
    DenseNet121_Weights,
    MobileNet_V2_Weights,
    ResNet18_Weights,
    ViT_B_16_Weights,
)


def build_vanilla_model(model_name: str, num_classes: int, use_pretrained: bool) -> nn.Module:
    if model_name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if use_pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if use_pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if model_name == "alexnet":
        weights = AlexNet_Weights.DEFAULT if use_pretrained else None
        model = models.alexnet(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if model_name == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT if use_pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")
