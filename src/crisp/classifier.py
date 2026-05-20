from __future__ import annotations

from typing import Any, Dict, Optional

from tqdm import tqdm

from .encoders import build_encoder
from .memory import MemoryBank
from .retrievers import build_retriever
from .utils import infer_label_from_parent, list_images
from .voting import majority_vote, weighted_vote

# tambahan untuk ArcFace
try:
    from insightface.app import FaceAnalysis
    import numpy as np
except ImportError:
    FaceAnalysis = None
    np = None


class CRISPClassifier:
    """
    Continual Retrieval & Indexing System for Perception.

    Encoder dapat berupa:
      - resnet18/34/50/101/152
      - clip
      - arcface (face-specific embedding)

    Incremental learning dilakukan dengan menambahkan embedding baru
    ke memory bank. Voting menggunakan weighted atau majority.
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
        self.device = device or ("cuda" if np and np.cuda.is_available() else "cpu")
        self.top_k = top_k
        self.voting = voting
        self.encoder_kwargs = encoder_kwargs or {}

        # Build encoder
        if self.encoder_name == "arcface":
            if FaceAnalysis is None:
                raise ImportError("ArcFace requires `insightface` package. Install via `pip install insightface`.")
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
        """
        Membuat ArcFace encoder menggunakan InsightFace.
        kwargs bisa berisi:
            - name: model ArcFace (default 'antelope')
            - det_size: ukuran deteksi (default (112,112))
        """
        name = kwargs.get("name", "antelope")
        det_size = kwargs.get("det_size", (112, 112))
        app = FaceAnalysis(name=name, providers=['CUDAExecutionProvider' if self.device=='cuda' else 'CPUExecutionProvider'])
        app.prepare(ctx_id=0 if self.device=='cuda' else -1, det_size=det_size)
        return app

    def _encode_path(self, image_path: str):
        """
        Wrapper encoder. ArcFace return embedding dari FaceAnalysis,
        ResNet/CLIP menggunakan build_encoder.encode_path()
        """
        if self.encoder_name == "arcface":
            import cv2
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.encoder.get(img)
            if not faces:
                raise ValueError(f"No face detected in {image_path}")
            embedding = faces[0].embedding  # 512-dim
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.astype(np.float32)
        else:
            return self.encoder.encode_path(image_path)

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