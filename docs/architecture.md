# CRISP Architecture

CRISP is a retrieval-based image classification system. The revised workflow supports a trainable ResNet phase before the encoder is frozen.

```text
Training Dataset
    |
    v
Train ResNet Backbone + Temporary Linear Head
    |
    v
Discard Temporary Head
    |
    v
Freeze Trained Backbone
    |
    v
Input Image
    |
    v
Image Preprocessing
    |
    v
Frozen Encoder
    |----------------|
    | Trained ResNet |
    | CLIP           |
    | ArcFace        |
    |----------------|
    |
    v
L2-Normalized Embedding
    |
    v
Vector Memory Bank
    |
    v
Retriever
    |----------------|
    | NumPy          |
    | Annoy          |
    | FAISS          |
    |----------------|
    |
    v
Top-k Nearest Neighbors
    |
    v
Voting
    |----------------|
    | Weighted       |
    | Majority       |
    |----------------|
    |
    v
Predicted Class / Unknown Class
```

## Key rule

If the backbone weights are changed, all old memory-bank embeddings become incompatible. Rebuild the memory bank after training or loading different backbone weights.
