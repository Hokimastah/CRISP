from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from .base import BaseImageEncoder


class ResNetEncoder(BaseImageEncoder):
    """
    ResNet image encoder for CRISP.

    The encoder can be trained first through CRISPClassifier.fit_backbone().
    After training, it should be frozen before embeddings are indexed into the
    memory bank. This keeps all stored embeddings compatible during retrieval.
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
        freeze: bool = True,
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

        # Keep only the convolutional backbone and global pooling.
        # The classification head is created temporarily during fit_backbone().
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.set_frozen(freeze)

    def _get_default_weights(self, backbone: str):
        mapping = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
            "resnet101": models.ResNet101_Weights.DEFAULT,
            "resnet152": models.ResNet152_Weights.DEFAULT,
        }
        return mapping[backbone]

    def set_frozen(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze the ResNet backbone.
        """
        for param in self.model.parameters():
            param.requires_grad = not freeze

        if freeze:
            self.model.eval()
        else:
            self.model.train()

    def freeze(self) -> None:
        self.set_frozen(True)

    def unfreeze(self) -> None:
        self.set_frozen(False)

    def forward_features(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Return non-normalized feature tensors for training or embedding.
        """
        features = self.model(tensor)
        return features.flatten(start_dim=1)

    def encode_pil(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            features = self.forward_features(tensor)

        if was_training:
            self.model.train()

        embedding = features.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return self.l2_normalize(embedding)

    def encode_path(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        return self.encode_pil(image)

    def save_weights(self, path: str) -> None:
        """
        Save the trained ResNet feature extractor weights.
        """
        checkpoint = {
            "backbone": self.backbone_name,
            "feature_dim": self.feature_dim,
            "image_size": self.image_size,
            "state_dict": self.model.state_dict(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True) if Path(path).parent != Path(".") else None
        torch.save(checkpoint, path)

    def load_weights(self, path: str, strict: bool = True, freeze: bool = True) -> None:
        """
        Load ResNet feature extractor weights and optionally freeze the backbone.
        """
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict, strict=strict)
        self.set_frozen(freeze)
