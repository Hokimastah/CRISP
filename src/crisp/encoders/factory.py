from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseImageEncoder


def build_encoder(
    encoder: str = "resnet50",
    device: Optional[str] = None,
    pretrained: bool = True,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
) -> BaseImageEncoder:
    encoder_kwargs = encoder_kwargs or {}
    encoder = encoder.lower()

    if encoder.startswith("resnet"):
        from .resnet import ResNetEncoder

        return ResNetEncoder(
            backbone=encoder,
            pretrained=pretrained,
            device=device,
            **encoder_kwargs,
        )

    if encoder == "clip":
        from .clip_encoder import CLIPEncoder

        return CLIPEncoder(device=device, **encoder_kwargs)

    raise ValueError(
        "Unsupported encoder. Use one of: resnet18, resnet34, resnet50, resnet101, resnet152, clip, arcface"
    )
