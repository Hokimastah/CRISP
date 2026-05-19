from crisp import CRISPClassifier


def main():
    clf = CRISPClassifier(
        encoder="resnet50",
        retriever="numpy",
        pretrained=True,
        device="cpu",
        top_k=5,
        voting="weighted",
    )

    clf.add_folder("dataset")
    clf.save("memory_bank.pkl")

    result = clf.predict("test_image.jpg")
    print(result["predicted_label"])
    print(result["scores"])


if __name__ == "__main__":
    main()
