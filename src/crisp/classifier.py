from __future__ import annotations

from typing import Any, Dict, Optional

from tqdm import tqdm

from .encoders import build_encoder
from .memory import MemoryBank
from .retrievers import build_retriever
from .utils import infer_label_from_parent, list_images
from .voting import majority_vote, weighted_vote


class CRISPClassifier:
    """
    Continual Retrieval & Indexing System for Perception.

    The encoder is frozen. Incremental learning is performed by adding new
    image embeddings into the memory bank.
    """

    def __init__(
        self,
        encoder: str = "resnet50",
        retriever: str = "numpy",
        pretrained: bool = True,
        device: Optional[str] = None,
        top_k: int = 5,
        voting: str = "weighted",
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        retriever_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.encoder_name = encoder
        self.retriever_name = retriever
        self.pretrained = pretrained
        self.device = device
        self.top_k = top_k
        self.voting = voting

        self.encoder = build_encoder(
            encoder=encoder,
            device=device,
            pretrained=pretrained,
            encoder_kwargs=encoder_kwargs,
        )
        self.memory = MemoryBank()
        self.retriever = build_retriever(
            retriever=retriever,
            retriever_kwargs=retriever_kwargs,
        )

    def add_image(
        self,
        image_path: str,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        embedding = self.encoder.encode_path(image_path)
        meta = metadata or {}
        meta.setdefault("path", image_path)

        self.memory.add(
            embedding=embedding,
            label=label,
            metadata=meta,
        )

    def add_folder(self, folder: str) -> None:
        image_paths = list_images(folder)

        for image_path in tqdm(image_paths, desc="Indexing images"):
            label = infer_label_from_parent(image_path)
            self.add_image(
                str(image_path),
                label=label,
                metadata={"path": str(image_path)},
            )

        self.retriever.build(self.memory)

    def predict(
        self,
        image_path: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        embedding = self.encoder.encode_path(image_path)

        neighbors = self.retriever.search(
            query_embedding=embedding,
            memory=self.memory,
            top_k=top_k or self.top_k,
        )

        if self.voting == "weighted":
            vote_result = weighted_vote(neighbors)
        elif self.voting == "majority":
            vote_result = majority_vote(neighbors)
        else:
            raise ValueError("voting must be either 'weighted' or 'majority'.")

        best_similarity = float(neighbors[0]["similarity"]) if neighbors else None
        status = "known"
        predicted_label = vote_result["predicted_label"]

        if threshold is not None and best_similarity is not None:
            if best_similarity < threshold:
                status = "unknown"
                predicted_label = None

        return {
            "status": status,
            "predicted_label": predicted_label,
            "scores": vote_result["scores"],
            "best_similarity": best_similarity,
            "neighbors": neighbors,
            "encoder": self.encoder_name,
            "retriever": self.retriever_name,
        }

    def save(self, path: str) -> None:
        self.memory.save(path)

    def load(self, path: str) -> None:
        self.memory.load(path)
        self.retriever.build(self.memory)

    def __len__(self) -> int:
        return len(self.memory)
