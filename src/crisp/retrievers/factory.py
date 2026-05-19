from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseRetriever
from .numpy_backend import NumpyRetriever


def build_retriever(
    retriever: str = "numpy",
    retriever_kwargs: Optional[Dict[str, Any]] = None,
) -> BaseRetriever:
    retriever_kwargs = retriever_kwargs or {}
    retriever = retriever.lower()

    if retriever == "numpy":
        return NumpyRetriever(**retriever_kwargs)

    if retriever == "annoy":
        from .annoy_backend import AnnoyRetriever
        return AnnoyRetriever(**retriever_kwargs)

    if retriever == "faiss":
        from .faiss_backend import FaissRetriever
        return FaissRetriever(**retriever_kwargs)

    raise ValueError("Unsupported retriever. Use one of: numpy, annoy, faiss")
