from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from .base import BaseImageEncoder


class ResNetEncoder(BaseImageEncoder):
    """
    Frozen ResNet image encoder.
    """

    SUPPORTED_BACKBONES = {
        "resnet18": (models.resnet18, 512),
        "resnet34": (models.resnet34, 512),
        "resnet50": (models.resnet50, 2048),
        "resnet101": (models.resnet101, 2048),
        "resnet152": (models.resnet152, 2048),
    }

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        device: Optional[str] = None,
        image_size: int = 224,
    ) -> None:
        if backbone not in self.SUPPORTED_BACKBONES:
            supported = ", ".join(self.SUPPORTED_BACKBONES.keys())
            raise ValueError(f"Unsupported ResNet backbone: {backbone}. Supported: {supported}")

        self.backbone_name = backbone
        self.pretrained = pretrained
        self.image_size = image_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model_fn, self.feature_dim = self.SUPPORTED_BACKBONES[backbone]

        if pretrained:
            weights = self._get_default_weights(backbone)
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)

        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _get_default_weights(self, backbone: str):
        mapping = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
            "resnet101": models.ResNet101_Weights.DEFAULT,
            "resnet152": models.ResNet152_Weights.DEFAULT,
        }
        return mapping[backbone]

    def encode_pil(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)
            features = features.flatten(start_dim=1)

        embedding = features.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return self.l2_normalize(embedding)

    def encode_path(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        return self.encode_pil(image)
