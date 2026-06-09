import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torchvision import models
from torchvision.models import (
    AlexNet_Weights,
    DenseNet121_Weights,
    MobileNet_V2_Weights,
    ResNet18_Weights,
    ViT_B_16_Weights,
)

from config import (
    ADAPTER_ATTENTION_HEADS,
    ADAPTER_ATTENTION_TOKENS,
    ADAPTER_DROPOUT,
    ADAPTER_HIDDEN_DIM,
    ADAPTER_TAIL_BLOCKS,
    ADAPTER_TRAIN_BACKBONE_TAIL,
    ADAPTER_TYPE,
    BRAIN_TUMOR_TUNE_ADAPTER_TAIL,
    USE_PRETRAINED,
)

from config import is_brain_tumor_dataset


class Adapter(nn.Module):
    def __init__(self, dim, hidden=512, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(hidden, dim)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = self.up(self.dropout(self.act(self.down(self.norm(x)))))
        return x + residual


class AttentionAdapter(nn.Module):
    def __init__(self, dim, hidden=512, heads=8, tokens=8, dropout=0.1):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(
                f"Attention adapter hidden dim ({hidden}) must be divisible by heads ({heads})"
            )

        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.memory_tokens = nn.Parameter(torch.empty(1, tokens, hidden))
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.up = nn.Linear(hidden, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.memory_tokens, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def split_heads(self, x):
        batch, tokens, hidden = x.shape
        x = x.view(batch, tokens, self.heads, self.head_dim)
        return x.transpose(1, 2)

    def merge_heads(self, x):
        batch, heads, tokens, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, tokens, heads * head_dim)

    def forward(self, x):
        query = self.down(self.norm(x)).unsqueeze(1)
        memory = self.memory_tokens.expand(x.size(0), -1, -1)
        context = torch.cat([query, memory], dim=1)

        q = self.split_heads(self.q_proj(query))
        k = self.split_heads(self.k_proj(context))
        v = self.split_heads(self.v_proj(context))
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        attended = torch.matmul(self.attn_dropout(attn), v)
        attended = self.out_proj(self.merge_heads(attended))

        adapted = attended + self.ffn(attended)
        residual = self.up(self.dropout(adapted.squeeze(1)))
        return x + residual


def build_adapter(dim, hidden):
    if ADAPTER_TYPE == "attention":
        return AttentionAdapter(
            dim,
            hidden=hidden,
            heads=ADAPTER_ATTENTION_HEADS,
            tokens=ADAPTER_ATTENTION_TOKENS,
            dropout=ADAPTER_DROPOUT,
        )

    if ADAPTER_TYPE == "mlp":
        return Adapter(dim, hidden, dropout=ADAPTER_DROPOUT)

    raise ValueError(f"Unsupported adapter type: {ADAPTER_TYPE}")


class MobileNetV2Features(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if USE_PRETRAINED else None
        model = models.mobilenet_v2(weights=weights)
        self.features = model.features
        self.feature_dim = model.last_channel

    def forward(self, x):
        x = self.features(x)
        return x.mean([2, 3])


class ResNet18Features(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if USE_PRETRAINED else None
        model = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = model.fc.in_features

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)


class DenseNet121Features(nn.Module):
    def __init__(self):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if USE_PRETRAINED else None
        model = models.densenet121(weights=weights)
        self.features = model.features
        self.feature_dim = model.classifier.in_features

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)


class AlexNetFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        weights = AlexNet_Weights.DEFAULT if USE_PRETRAINED else None
        model = models.alexnet(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier_features = nn.Sequential(*list(model.classifier.children())[:-1])
        self.feature_dim = model.classifier[-1].in_features

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier_features(x)


class ViTB16Features(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if USE_PRETRAINED else None
        self.model = models.vit_b_16(weights=weights)
        self.feature_dim = self.model.heads.head.in_features
        self.model.heads = nn.Identity()

    def forward(self, x):
        return self.model(x)


def build_feature_extractor(model_name):
    builders = {
        "mobilenet_v2": MobileNetV2Features,
        "resnet18": ResNet18Features,
        "densenet121": DenseNet121Features,
        "alexnet": AlexNetFeatures,
        "vit_b_16": ViTB16Features,
    }
    extractor = builders[model_name]()
    return extractor, extractor.feature_dim


class AdapterModel(nn.Module):
    def __init__(self, model_name, num_classes, adapter_hidden=512):
        super().__init__()
        self.model_name = model_name
        self.feature_extractor, feature_dim = build_feature_extractor(model_name)
        self.feature_dim = feature_dim
        self.adapter = build_adapter(feature_dim, adapter_hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes),
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        if should_tune_adapter_tail(model_name):
            enable_adapter_tail_tuning(
                self.feature_extractor,
                model_name,
                ADAPTER_TAIL_BLOCKS.get(model_name, 1),
            )

    def train(self, mode=True):
        super().train(mode)
        self.feature_extractor.eval()
        return self

    def forward(self, x):
        if any(param.requires_grad for param in self.feature_extractor.parameters()):
            features = self.feature_extractor(x)
        else:
            with torch.no_grad():
                features = self.feature_extractor(x)
        features = self.adapter(features)
        return self.classifier(features)


def should_tune_adapter_tail(model_name):
    return ADAPTER_TRAIN_BACKBONE_TAIL or (
        BRAIN_TUMOR_TUNE_ADAPTER_TAIL
        and is_brain_tumor_dataset()
    )


def set_trainable(module, trainable=True):
    for param in module.parameters():
        param.requires_grad = trainable


def enable_adapter_tail_tuning(feature_extractor, model_name, blocks):
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


def build_vanilla_model(model_name, num_classes):
    if model_name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if USE_PRETRAINED else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if USE_PRETRAINED else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if USE_PRETRAINED else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if model_name == "alexnet":
        weights = AlexNet_Weights.DEFAULT if USE_PRETRAINED else None
        model = models.alexnet(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if model_name == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT if USE_PRETRAINED else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def build_adapter_model(model_name, num_classes):
    return AdapterModel(model_name, num_classes, adapter_hidden=ADAPTER_HIDDEN_DIM)
