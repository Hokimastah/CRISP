from .base import BaseImageEncoder
from .factory import build_encoder

__all__ = [
    "BaseImageEncoder",
    "build_encoder",
    "ResNetEncoder",
]


def __getattr__(name):
    if name == "ResNetEncoder":
        from .resnet import ResNetEncoder

        return ResNetEncoder
    raise AttributeError(f"module 'crisp.encoders' has no attribute {name!r}")
