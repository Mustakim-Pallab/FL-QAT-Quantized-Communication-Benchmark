import torch
import torch.nn as nn

from config import Config
from models.adapter import build_adapter
from models.feature_extractors import build_feature_extractor
from models.tail_tuning import enable_adapter_tail_tuning


class AdapterModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, config: Config, dataset_name: str | None = None) -> None:
        super().__init__()
        self.model_name = model_name
        self.feature_extractor, feature_dim = build_feature_extractor(model_name, config.use_pretrained)
        self.feature_dim = feature_dim
        self.adapter = build_adapter(feature_dim, config.adapter_hidden_dim, config)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes),
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        tail_blocks = (config.adapter_tail_blocks or {}).get(model_name, 1)
        if config.should_tune_adapter_tail(dataset_name):
            enable_adapter_tail_tuning(self.feature_extractor, model_name, tail_blocks)

    def train(self, mode: bool = True) -> AdapterModel:
        super().train(mode)
        self.feature_extractor.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if any(param.requires_grad for param in self.feature_extractor.parameters()):
            features = self.feature_extractor(x)
        else:
            with torch.no_grad():
                features = self.feature_extractor(x)
        features = self.adapter(features)
        return self.classifier(features)


def build_adapter_model(model_name: str, num_classes: int, config: Config,
                        dataset_name: str | None = None) -> AdapterModel:
    return AdapterModel(model_name, num_classes, config, dataset_name=dataset_name)
