# CRISP

**CRISP** stands for **Continual Retrieval & Indexing System for Perception**.

CRISP is an installable Python library for image classification using frozen image encoders, a vector memory bank, retrieval-based top-k ranking, and voting-based classification.

## Supported components

**Encoders**

- Frozen ResNet: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- Frozen CLIP image encoder: `clip`

**Retrievers**

- NumPy brute-force retrieval: `numpy`
- Annoy approximate nearest neighbor retrieval: `annoy`
- FAISS similarity search: `faiss`

**Voting**

- Similarity-weighted voting: `weighted`
- Majority voting: `majority`

## Architecture

```text
Input Image
  -> Image Preprocessing
  -> Frozen Encoder
     -> ResNet / CLIP
  -> L2-normalized Embedding
  -> Vector Memory Bank
  -> Retrieval Backend
     -> NumPy / Annoy / FAISS
  -> Top-k Nearest Neighbors
  -> Voting
  -> Predicted Class / Unknown Class
```

## Installation

Basic local install:

```bash
pip install -e .
```

Install all optional variants:

```bash
pip install -e ".[all]"
```

Install optional modules separately:

```bash
pip install -e ".[annoy]"
pip install -e ".[faiss]"
pip install -e ".[clip]"
```

## Dataset structure

```text
dataset/
  class_a/
    image_001.jpg
    image_002.jpg
  class_b/
    image_003.jpg
    image_004.jpg
```

## Basic usage

```python
from crisp import CRISPClassifier

clf = CRISPClassifier(
    encoder="resnet50",
    retriever="numpy",
    device="cuda",
    top_k=5,
    voting="weighted"
)

clf.add_folder("dataset")
clf.save("memory_bank.pkl")

result = clf.predict("test_image.jpg")
print(result["predicted_label"])
print(result["scores"])
```

## CLIP usage

```python
from crisp import CRISPClassifier

clf = CRISPClassifier(
    encoder="clip",
    encoder_kwargs={
        "model_name": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k"
    },
    retriever="faiss",
    device="cuda",
    top_k=5,
    voting="weighted"
)

clf.add_folder("dataset")
result = clf.predict("test_image.jpg")
print(result)
```

## CLI

Create memory bank:

```bash
crisp index --data dataset --output memory.pkl --encoder resnet50 --retriever numpy
```

Predict:

```bash
crisp predict --image test.jpg --memory memory.pkl --encoder resnet50 --retriever numpy --top-k 5
```

## Unknown class detection

Use a similarity threshold:

```python
result = clf.predict("test_image.jpg", threshold=0.65)
print(result["status"])
```

If the best similarity is below the threshold, the status is `unknown`.
