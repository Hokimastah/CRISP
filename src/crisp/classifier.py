from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .encoders import build_encoder
from .memory import MemoryBank
from .retrievers import build_retriever
from .utils import infer_label_from_parent, list_images
from .voting import majority_vote, weighted_vote

try:
    from insightface.app import FaceAnalysis
except ImportError:  # pragma: no cover - optional dependency
    FaceAnalysis = None


class CRISPClassifier:
    """
    Continual Retrieval & Indexing System for Perception.

    Supported encoders:
      - resnet18/34/50/101/152
      - clip
      - arcface (face-specific embedding)

    Recommended workflow for ResNet encoders:
      1. Train the backbone with fit_backbone().
      2. Freeze the backbone.
      3. Build the memory bank using add_folder().
      4. Predict by retrieval and voting.

    Incremental updates after the backbone is frozen are add-only memory updates.
    If the backbone weights change, the memory bank must be rebuilt.
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
        self.encoder_name = encoder.lower()
        self.retriever_name = retriever
        self.pretrained = pretrained
        self.device = str(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
        self.top_k = top_k
        self.voting = voting
        self.encoder_kwargs = encoder_kwargs or {}
        self.class_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_class: Optional[Dict[int, str]] = None

        if self.encoder_name == "arcface":
            if FaceAnalysis is None:
                raise ImportError(
                    "ArcFace requires insightface. Install it with: pip install -e '.[arcface]'"
                )
            self.encoder = self._build_arcface_encoder(**self.encoder_kwargs)
        else:
            self.encoder = build_encoder(
                encoder=self.encoder_name,
                device=self.device,
                pretrained=pretrained,
                encoder_kwargs=encoder_kwargs,
            )

        self.memory = MemoryBank()
        self.retriever = build_retriever(
            retriever=retriever,
            retriever_kwargs=retriever_kwargs,
        )

    def _build_arcface_encoder(self, **kwargs):
        name = kwargs.get("name", "antelope")
        det_size = kwargs.get("det_size", (112, 112))
        provider = "CUDAExecutionProvider" if self.device.startswith("cuda") else "CPUExecutionProvider"
        app = FaceAnalysis(name=name, providers=[provider])
        app.prepare(ctx_id=0 if self.device.startswith("cuda") else -1, det_size=det_size)
        return app

    def _encode_path(self, image_path: str) -> np.ndarray:
        if self.encoder_name == "arcface":
            import cv2

            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Unable to read image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.encoder.get(img)
            if not faces:
                raise ValueError(f"No face detected in {image_path}")
            embedding = faces[0].embedding
            embedding = embedding / max(np.linalg.norm(embedding), 1e-12)
            return embedding.astype(np.float32)

        return self.encoder.encode_path(image_path)

    def fit_backbone(
        self,
        train_folder: str,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        num_workers: int = 0,
        save_path: Optional[str] = None,
        freeze_after: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Fine-tune a ResNet backbone before using it as a frozen CRISP encoder.

        The training dataset must use the ImageFolder format:

        train_folder/
        ├── class_a/
        ├── class_b/
        └── class_c/

        This method trains the ResNet feature extractor with a temporary linear
        classification head. Only the feature extractor weights are kept for
        retrieval-based classification. The temporary head is discarded.
        """
        if not self.encoder_name.startswith("resnet"):
            raise ValueError("fit_backbone() is currently supported only for ResNet encoders.")

        if not hasattr(self.encoder, "forward_features"):
            raise TypeError("The selected encoder does not expose forward_features().")

        train_path = Path(train_folder)
        if not train_path.exists():
            raise FileNotFoundError(f"Training folder not found: {train_folder}")

        from torchvision import datasets

        dataset = datasets.ImageFolder(str(train_path), transform=self.encoder.transform)
        if len(dataset) == 0:
            raise ValueError(f"No training images found in: {train_folder}")
        if len(dataset.classes) < 2:
            raise ValueError("fit_backbone() requires at least two class folders.")

        self.class_to_idx = dict(dataset.class_to_idx)
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.startswith("cuda"),
        )

        self.encoder.unfreeze()
        head = nn.Linear(self.encoder.feature_dim, len(dataset.classes)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            list(self.encoder.model.parameters()) + list(head.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        history: List[Dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            self.encoder.model.train()
            head.train()

            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            progress = tqdm(loader, desc=f"Training backbone epoch {epoch}/{epochs}")
            for images, labels in progress:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                features = self.encoder.forward_features(images)
                logits = head(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                batch_size_actual = int(labels.size(0))
                total_loss += float(loss.item()) * batch_size_actual
                total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                total_samples += batch_size_actual

                progress.set_postfix({
                    "loss": total_loss / max(total_samples, 1),
                    "acc": total_correct / max(total_samples, 1),
                })

            epoch_result = {
                "epoch": float(epoch),
                "loss": total_loss / max(total_samples, 1),
                "accuracy": total_correct / max(total_samples, 1),
            }
            history.append(epoch_result)

        if freeze_after:
            self.encoder.freeze()

        if save_path is not None:
            self.save_backbone(save_path)

        return history

    def save_backbone(self, path: str) -> None:
        if not hasattr(self.encoder, "save_weights"):
            raise ValueError("save_backbone() is available only for encoders with save_weights().")
        self.encoder.save_weights(path)

    def load_backbone(self, path: str, strict: bool = True, freeze: bool = True) -> None:
        if not hasattr(self.encoder, "load_weights"):
            raise ValueError("load_backbone() is available only for encoders with load_weights().")
        self.encoder.load_weights(path, strict=strict, freeze=freeze)

    def add_image(
        self,
        image_path: str,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        embedding = self._encode_path(image_path)
        meta = metadata or {}
        meta.setdefault("path", image_path)

        self.memory.add(
            embedding=embedding,
            label=label,
            metadata=meta,
        )

    def add_folder(self, folder: str) -> None:
        image_paths = list_images(folder)
        if not image_paths:
            raise ValueError(f"No images found in folder: {folder}")

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
        embedding = self._encode_path(image_path)

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

        if threshold is not None and best_similarity is not None and best_similarity < threshold:
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
