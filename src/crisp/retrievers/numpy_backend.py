from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from crisp.memory import MemoryBank
from .base import BaseRetriever


class NumpyRetriever(BaseRetriever):
    """
    Exact brute-force cosine retrieval using NumPy.
    """

    def __init__(self) -> None:
        self.matrix = None
        self.index_size = 0

    def build(self, memory: MemoryBank) -> None:
        if len(memory) == 0:
            self.matrix = None
            self.index_size = 0
            return

        self.matrix = memory.embeddings_matrix()
        self.index_size = len(memory)

    def search(
        self,
        query_embedding: np.ndarray,
        memory: MemoryBank,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if len(memory) == 0:
            raise ValueError("MemoryBank is empty. Add samples before prediction.")

        if self.matrix is None or self.index_size != len(memory):
            self.build(memory)

        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        # Embeddings are L2-normalized, so dot product equals cosine similarity.
        similarities = self.matrix @ query_embedding

        k = min(top_k, len(memory))
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            item = memory.get(int(idx))
            results.append({
                "index": int(idx),
                "label": item.label,
                "similarity": float(similarities[idx]),
                "metadata": item.metadata,
            })

        return results
