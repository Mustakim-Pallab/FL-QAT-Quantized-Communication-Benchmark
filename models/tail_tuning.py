import torch.nn as nn


def set_trainable(module: nn.Module, trainable: bool = True) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def enable_adapter_tail_tuning(feature_extractor: nn.Module, model_name: str, blocks: int) -> None:
    blocks = max(int(blocks), 1)

    if model_name == "mobilenet_v2":
        for module in list(feature_extractor.features.children())[-blocks:]:
            set_trainable(module, True)
        return

    if model_name == "resnet18":
        set_trainable(feature_extractor.features[-2], True)
        return

    if model_name == "densenet121":
        set_trainable(feature_extractor.features.denseblock4, True)
        set_trainable(feature_extractor.features.norm5, True)
        return

    if model_name == "alexnet":
        for module in list(feature_extractor.features.children())[-5:]:
            set_trainable(module, True)
        for module in list(feature_extractor.classifier_features.children())[-blocks:]:
            set_trainable(module, True)
        return

    if model_name == "vit_b_16":
        encoder_layers = list(feature_extractor.model.encoder.layers.children())
        for module in encoder_layers[-blocks:]:
            set_trainable(module, True)
        set_trainable(feature_extractor.model.encoder.ln, True)
        return

    raise ValueError(f"Unsupported model: {model_name}")
