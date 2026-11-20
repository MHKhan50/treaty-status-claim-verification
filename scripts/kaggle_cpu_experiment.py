from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score

from warheadverify.data import load_dataset, grouped_splits
from warheadverify.modeling import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_demo.csv"
OUT = ROOT / "reports" / "kaggle_cpu_results.csv"
SEEDS = [11, 23, 37, 51, 79]
N_SPLITS = 3


def main():
    frame = load_dataset(DATA)
    rows = []

    # Smoke-test only: replace the fictional fixture before substantive research.
    for fold, (train_idx, test_idx) in enumerate(
        grouped_splits(frame, n_splits=N_SPLITS),
        start=1,
    ):
        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx]

        train_documents = set(train["document_id"])
        test_documents = set(test["document_id"])

        if not train_documents.isdisjoint(test_documents):
            raise RuntimeError(
                f"Document leakage detected in fold {fold}"
            )

        for model_type in ("logreg", "mlp"):
            for seed in SEEDS:
                model = make_pipeline(model_type, seed)
                model.fit(train["claim"], train["label"])
                pred = model.predict(test["claim"])

                rows.append({
                    "model": model_type,
                    "seed": seed,
                    "fold": fold,
                    "macro_f1": f1_score(
                        test["label"],
                        pred,
                        average="macro",
                        zero_division=0,
                    ),
                    "balanced_accuracy": balanced_accuracy_score(
                        test["label"],
                        pred,
                    ),
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_train_documents": len(train_documents),
                    "n_test_documents": len(test_documents),
                    "document_overlap": len(
                        train_documents.intersection(test_documents)
                    ),
                    "dataset_status": "synthetic_demo_only",
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT, index=False)

    print(f"Saved {len(results)} experiment rows to {OUT}")
    print(f"Models: {results['model'].nunique()}")
    print(f"Seeds: {results['seed'].nunique()}")
    print(f"Folds: {results['fold'].nunique()}")
    print(
        "Maximum document overlap:",
        results["document_overlap"].max(),
    )


if __name__ == "__main__":
    main()
