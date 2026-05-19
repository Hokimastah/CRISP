from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from crisp.memory import MemoryBank
from .base import BaseRetriever


class AnnoyRetriever(BaseRetriever):
    """
    Approximate nearest neighbor retrieval using Annoy.
    """

    def __init__(
        self,
        n_trees: int = 10,
        metric: str = "angular",
        search_k: int = -1,
    ) -> None:
        try:
            from annoy import AnnoyIndex
        except ImportError as exc:
            raise ImportError(
                "AnnoyRetriever requires annoy. Install with: pip install -e '.[annoy]'"
            ) from exc

        self.AnnoyIndex = AnnoyIndex
        self.n_trees = n_trees
        self.metric = metric
        self.search_k = search_k
        self.index = None
        self.dim = None
        self.index_size = 0

    def build(self, memory: MemoryBank) -> None:
        if len(memory) == 0:
            self.index = None
            self.dim = None
            self.index_size = 0
            return

        matrix = memory.embeddings_matrix()
        self.dim = int(matrix.shape[1])

        index = self.AnnoyIndex(self.dim, self.metric)

        for i, vector in enumerate(matrix):
            index.add_item(i, vector.tolist())

        index.build(self.n_trees)

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

        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        k = min(top_k, len(memory))

        indices, distances = self.index.get_nns_by_vector(
            query_embedding.tolist(),
            k,
            search_k=self.search_k,
            include_distances=True,
        )

        results = []
        for idx, distance in zip(indices, distances):
            item = memory.get(int(idx))

            if self.metric == "angular":
                similarity = 1.0 - (float(distance) ** 2) / 2.0
            else:
                similarity = -float(distance)

            results.append({
                "index": int(idx),
                "label": item.label,
                "similarity": float(similarity),
                "metadata": item.metadata,
            })

        return results
