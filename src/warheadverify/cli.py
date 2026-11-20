import argparse
import json
from pathlib import Path
import pandas as pd
from .data import load_dataset
from .modeling import train_model, load_model, evaluate_model
from .agent import VerificationAgent

def build_parser():
    parser = argparse.ArgumentParser(prog="warheadverify")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--model-dir", default="artifacts")
    train.add_argument("--model-type", choices=["mlp", "logreg"], default="mlp")
    train.add_argument("--seed", type=int, default=17)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--model-dir", default="artifacts")
    evaluate.add_argument("--model-type", choices=["mlp", "logreg"], default="mlp")
    verify = sub.add_parser("verify")
    verify.add_argument("--claim", required=True)
    verify.add_argument("--evidence", required=True)
    verify.add_argument("--model-dir", default="artifacts")
    verify.add_argument("--model-type", choices=["mlp", "logreg"], default="mlp")
    verify.add_argument("--top-k", type=int, default=1)
    verify.add_argument("--output", default="")
    return parser

def main():
    args = build_parser().parse_args()
    if args.command == "train":
        frame = load_dataset(args.data)
        train_model(frame, args.model_dir, args.model_type, args.seed)
        print(f"Trained {args.model_type} model on {len(frame)} examples.")
        print(f"Saved model in: {args.model_dir}")
    elif args.command == "evaluate":
        frame = load_dataset(args.data)
        model = load_model(args.model_dir, args.model_type)
        print(json.dumps(evaluate_model(model, frame), indent=2))
    elif args.command == "verify":
        model = load_model(args.model_dir, args.model_type)
        evidence_frame = pd.DataFrame({"evidence_text": [args.evidence]})
        result = VerificationAgent(model, evidence_frame).verify(args.claim, args.top_k)
        rendered = json.dumps(result, indent=2)
        print(rendered)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
