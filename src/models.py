from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    ResNet18_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    densenet121,
    resnet18,
    resnet101,
    resnet152,
)


def build_model(name: str = "densenet121", num_labels: int = 14, pretrained: bool = True):
    name = name.lower()

    if name == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_labels)
        return model

    builders = {
        "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
        "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V1),
        "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V1),
    }
    if name in builders:
        builder, default_weights = builders[name]
        model = builder(weights=default_weights if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_labels)
        return model

    raise ValueError(f"Unsupported model: {name}")
