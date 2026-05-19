from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseImageEncoder


class CLIPEncoder(BaseImageEncoder):
    """
    Frozen CLIP image encoder using open_clip_torch.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
    ) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "CLIPEncoder requires open_clip_torch. Install with: pip install -e '.[clip]'"
            ) from exc

        self.model_name = model_name
        self.pretrained = pretrained
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )

        self.model.to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.feature_dim = int(getattr(self.model.visual, "output_dim", 0) or 512)

    def encode_pil(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(tensor)

        embedding = features.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return self.l2_normalize(embedding)

    def encode_path(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        return self.encode_pil(image)
