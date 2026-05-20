import torch
import numpy as np
from insightface.app import FaceAnalysis

class ArcFaceEncoder:
    def __init__(self, device='cuda'):
        self.device = device
        self.app = FaceAnalysis(name='antelope', providers=['CUDAExecutionProvider' if device=='cuda' else 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0 if device=='cuda' else -1, det_size=(112, 112))

    def encode(self, img):
        # img: numpy array HxWxC, RGB
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        embedding = faces[0].embedding  # 512 dim float32
        embedding = embedding / np.linalg.norm(embedding)  # normalize
        return embedding.astype(np.float32)