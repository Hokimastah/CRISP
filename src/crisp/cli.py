from __future__ import annotations

import argparse
import json

from .classifier import CRISPClassifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CRISP: Continual Retrieval & Indexing System for Perception"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train-backbone",
        help="Train a ResNet backbone first, then freeze and save its weights",
    )
    train_parser.add_argument("--data", required=True, help="Training dataset folder with class subfolders")
    train_parser.add_argument("--output", required=True, help="Output backbone weights path, e.g. resnet50_crisp.pt")
    train_parser.add_argument("--encoder", default="resnet50", help="resnet18, resnet34, resnet50, resnet101, resnet152")
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--no-pretrained", action="store_true", help="Start ResNet from random weights")

    index_parser = subparsers.add_parser("index", help="Create memory bank from folder dataset")
    index_parser.add_argument("--data", required=True, help="Dataset folder with class subfolders")
    index_parser.add_argument("--output", required=True, help="Output memory bank path, e.g. memory.pkl")
    index_parser.add_argument("--encoder", default="resnet50", help="resnet50, resnet18, clip, arcface, etc.")
    index_parser.add_argument("--retriever", default="numpy", choices=["numpy", "annoy", "faiss"])
    index_parser.add_argument("--device", default=None)
    index_parser.add_argument("--backbone-weights", default=None, help="Optional trained ResNet backbone weights")
    index_parser.add_argument("--no-pretrained", action="store_true", help="Start ResNet from random weights")

    predict_parser = subparsers.add_parser("predict", help="Predict image class")
    predict_parser.add_argument("--image", required=True, help="Image path")
    predict_parser.add_argument("--memory", required=True, help="Memory bank path")
    predict_parser.add_argument("--encoder", default="resnet50", help="resnet50, resnet18, clip, arcface, etc.")
    predict_parser.add_argument("--retriever", default="numpy", choices=["numpy", "annoy", "faiss"])
    predict_parser.add_argument("--device", default=None)
    predict_parser.add_argument("--top-k", type=int, default=5)
    predict_parser.add_argument("--threshold", type=float, default=None)
    predict_parser.add_argument("--voting", choices=["weighted", "majority"], default="weighted")
    predict_parser.add_argument("--backbone-weights", default=None, help="Optional trained ResNet backbone weights")
    predict_parser.add_argument("--no-pretrained", action="store_true", help="Start ResNet from random weights")

    args = parser.parse_args()

    if args.command == "train-backbone":
        clf = CRISPClassifier(
            encoder=args.encoder,
            retriever="numpy",
            pretrained=not args.no_pretrained,
            device=args.device,
        )
        history = clf.fit_backbone(
            train_folder=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            save_path=args.output,
            freeze_after=True,
        )
        print(json.dumps({"saved_backbone": args.output, "history": history}, indent=2))

    elif args.command == "index":
        clf = CRISPClassifier(
            encoder=args.encoder,
            retriever=args.retriever,
            pretrained=not args.no_pretrained,
            device=args.device,
        )
        if args.backbone_weights:
            clf.load_backbone(args.backbone_weights, freeze=True)
        clf.add_folder(args.data)
        clf.save(args.output)
        print(f"Saved memory bank to {args.output}")

    elif args.command == "predict":
        clf = CRISPClassifier(
            encoder=args.encoder,
            retriever=args.retriever,
            pretrained=not args.no_pretrained,
            device=args.device,
            top_k=args.top_k,
            voting=args.voting,
        )
        if args.backbone_weights:
            clf.load_backbone(args.backbone_weights, freeze=True)
        clf.load(args.memory)
        result = clf.predict(args.image, threshold=args.threshold)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
