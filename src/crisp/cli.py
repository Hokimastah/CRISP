from __future__ import annotations

import argparse
import json

from .classifier import CRISPClassifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CRISP: Continual Retrieval & Indexing System for Perception"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Create memory bank from folder dataset")
    index_parser.add_argument("--data", required=True, help="Dataset folder with class subfolders")
    index_parser.add_argument("--output", required=True, help="Output memory bank path, e.g. memory.pkl")
    index_parser.add_argument("--encoder", default="resnet50", help="resnet50, resnet18, clip, etc.")
    index_parser.add_argument("--retriever", default="numpy", choices=["numpy", "annoy", "faiss"])
    index_parser.add_argument("--device", default=None)

    predict_parser = subparsers.add_parser("predict", help="Predict image class")
    predict_parser.add_argument("--image", required=True, help="Image path")
    predict_parser.add_argument("--memory", required=True, help="Memory bank path")
    predict_parser.add_argument("--encoder", default="resnet50", help="resnet50, resnet18, clip, etc.")
    predict_parser.add_argument("--retriever", default="numpy", choices=["numpy", "annoy", "faiss"])
    predict_parser.add_argument("--device", default=None)
    predict_parser.add_argument("--top-k", type=int, default=5)
    predict_parser.add_argument("--threshold", type=float, default=None)
    predict_parser.add_argument("--voting", choices=["weighted", "majority"], default="weighted")

    args = parser.parse_args()

    if args.command == "index":
        clf = CRISPClassifier(
            encoder=args.encoder,
            retriever=args.retriever,
            device=args.device,
        )
        clf.add_folder(args.data)
        clf.save(args.output)
        print(f"Saved memory bank to {args.output}")

    elif args.command == "predict":
        clf = CRISPClassifier(
            encoder=args.encoder,
            retriever=args.retriever,
            device=args.device,
            top_k=args.top_k,
            voting=args.voting,
        )
        clf.load(args.memory)
        result = clf.predict(args.image, threshold=args.threshold)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
