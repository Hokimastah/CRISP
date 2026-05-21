__version__ = "0.2.0"

__all__ = [
    "CRISPClassifier",
    "MemoryBank",
]


def __getattr__(name):
    if name == "CRISPClassifier":
        from .classifier import CRISPClassifier

        return CRISPClassifier
    if name == "MemoryBank":
        from .memory import MemoryBank

        return MemoryBank
    raise AttributeError(f"module 'crisp' has no attribute {name!r}")
