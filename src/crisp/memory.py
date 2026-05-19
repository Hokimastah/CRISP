from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MemoryItem:
    embedding: np.ndarray
    label: str
    metadata: Dict[str, Any]


class MemoryBank:
    """
    Add-only vector memory bank for CRISP.
    """

    def __init__(self) -> None:
        self.items: List[MemoryItem] = []

    def __len__(self) -> int:
        return len(self.items)

    def add(
        self,
        embedding: np.ndarray,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        embedding = np.asarray(embedding, dtype=np.float32)
        metadata = metadata or {}
        self.items.append(MemoryItem(embedding=embedding, label=str(label), metadata=metadata))

    def embeddings_matrix(self) -> np.ndarray:
        if not self.items:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([item.embedding for item in self.items]).astype(np.float32)

    def labels(self) -> List[str]:
        return [item.label for item in self.items]

    def metadata(self) -> List[Dict[str, Any]]:
        return [item.metadata for item in self.items]

    def get(self, index: int) -> MemoryItem:
        return self.items[index]

    def save(self, path: str) -> None:
        data = [
            {
                "embedding": item.embedding,
                "label": item.label,
                "metadata": item.metadata,
            }
            for item in self.items
        ]
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.items = [
            MemoryItem(
                embedding=np.asarray(row["embedding"], dtype=np.float32),
                label=str(row["label"]),
                metadata=dict(row.get("metadata", {})),
            )
            for row in data
        ]
