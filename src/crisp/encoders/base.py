from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class BaseImageEncoder(ABC):
    """
    Abstract interface for image encoders used by CRISP.
    """

    feature_dim: int

    @abstractmethod
    def encode_pil(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_path(self, image_path: str) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / max(norm, eps)
