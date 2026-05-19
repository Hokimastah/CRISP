# CRISP Architecture

```text
Input Image
    |
    v
Image Preprocessing
    |
    v
Frozen Encoder
    |----------------|
    | ResNet         |
    | CLIP           |
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
