from pathlib import Path

from crisp import CRISPClassifier


DATASET_DIR = "dataset_train"
TEST_IMAGE = "test_image.jpg"
BACKBONE_WEIGHTS = "resnet50_crisp.pt"


def run_variant(name, encoder, retriever, encoder_kwargs=None, retriever_kwargs=None):
    print(f"\n=== {name} ===")

    clf = CRISPClassifier(
        encoder=encoder,
        retriever=retriever,
        encoder_kwargs=encoder_kwargs,
        retriever_kwargs=retriever_kwargs,
        device="cuda",
        top_k=5,
        voting="weighted",
    )

    if encoder.startswith("resnet") and Path(BACKBONE_WEIGHTS).exists():
        clf.load_backbone(BACKBONE_WEIGHTS, freeze=True)

    clf.add_folder(DATASET_DIR)
    result = clf.predict(TEST_IMAGE)

    print("Prediction:", result["predicted_label"])
    print("Scores:", result["scores"])


def main():
    run_variant("Trained ResNet50 + NumPy", "resnet50", "numpy")
    run_variant("Trained ResNet50 + Annoy", "resnet50", "annoy", retriever_kwargs={"n_trees": 20})
    run_variant("Trained ResNet50 + FAISS", "resnet50", "faiss")

    run_variant(
        "CLIP + NumPy",
        "clip",
        "numpy",
        encoder_kwargs={
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
        },
    )


if __name__ == "__main__":
    main()
