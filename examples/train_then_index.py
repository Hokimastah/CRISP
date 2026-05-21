from crisp import CRISPClassifier


def main():
    clf = CRISPClassifier(
        encoder="resnet50",
        retriever="numpy",
        pretrained=True,
        device="cuda",
        top_k=5,
        voting="weighted",
    )

    history = clf.fit_backbone(
        train_folder="dataset_train",
        epochs=10,
        batch_size=32,
        lr=1e-4,
        save_path="resnet50_crisp.pt",
    )
    print(history)

    # The backbone is now frozen. Build embeddings using the trained backbone.
    clf.add_folder("dataset_train")
    clf.save("memory_bank.pkl")

    result = clf.predict("test_image.jpg")
    print(result)


if __name__ == "__main__":
    main()
