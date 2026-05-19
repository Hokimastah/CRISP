from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from crisp.memory import MemoryBank


class BaseRetriever(ABC):
    """
    Abstract interface for vector retrieval backends.
    """

    @abstractmethod
    def build(self, memory: MemoryBank) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        memory: MemoryBank,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError
