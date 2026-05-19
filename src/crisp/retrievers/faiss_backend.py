from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from crisp.memory import MemoryBank
from .base import BaseRetriever


class FaissRetriever(BaseRetriever):
    """
    Similarity search using FAISS IndexFlatIP.
    """

    def __init__(self) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "FaissRetriever requires faiss-cpu. Install with: pip install -e '.[faiss]'"
            ) from exc

        self.faiss = faiss
        self.index = None
        self.index_size = 0
        self.dim = None

    def build(self, memory: MemoryBank) -> None:
        if len(memory) == 0:
            self.index = None
            self.index_size = 0
            self.dim = None
            return

        matrix = memory.embeddings_matrix().astype(np.float32)
        self.dim = int(matrix.shape[1])

        index = self.faiss.IndexFlatIP(self.dim)
        index.add(matrix)

        self.index = index
        self.index_size = len(memory)

    def search(
        self,
        query_embedding: np.ndarray,
        memory: MemoryBank,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if len(memory) == 0:
            raise ValueError("MemoryBank is empty. Add samples before prediction.")

        if self.index is None or self.index_size != len(memory):
            self.build(memory)

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        k = min(top_k, len(memory))

        similarities, indices = self.index.search(query, k)

        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            item = memory.get(int(idx))
            results.append({
                "index": int(idx),
                "label": item.label,
                "similarity": float(similarity),
                "metadata": item.metadata,
            })

        return results
