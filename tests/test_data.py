from pathlib import Path

import pandas as pd
import pytest

from warheadverify.data import load_dataset

DATA = Path(__file__).parents[1] / "data" / "synthetic_demo.csv"


def test_demo_dataset_loads():
    frame = load_dataset(DATA)
    assert len(frame) == 18
    assert frame["document_id"].nunique() == 12
    assert set(frame["label"]) == {
        "supported",
        "contradicted",
        "insufficient_evidence",
    }


def test_missing_columns_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("claim,label\na,supported\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_dataset(path)


def test_empty_document_id_rejected(tmp_path):
    frame = pd.read_csv(DATA)
    frame.loc[0, "document_id"] = ""
    path = tmp_path / "empty_document.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="document_id"):
        load_dataset(path)


def test_duplicate_claim_ids_rejected(tmp_path):
    frame = pd.read_csv(DATA)
    frame.loc[1, "claim_id"] = frame.loc[0, "claim_id"]
    path = tmp_path / "duplicate_claim.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="claim_id"):
        load_dataset(path)


def test_grouped_splits_prevent_document_leakage():
    from warheadverify.data import grouped_splits

    frame = load_dataset(DATA)

    for train_idx, test_idx in grouped_splits(frame, n_splits=3):
        train_documents = set(frame.iloc[train_idx]["document_id"])
        test_documents = set(frame.iloc[test_idx]["document_id"])

        assert train_documents.isdisjoint(test_documents)


def test_grouped_splits_reject_too_many_folds():
    from warheadverify.data import grouped_splits

    frame = load_dataset(DATA)

    with pytest.raises(ValueError, match="unique documents"):
        list(grouped_splits(frame, n_splits=13))
