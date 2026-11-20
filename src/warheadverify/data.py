from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

REQUIRED_COLUMNS = {
    "claim_id",
    "document_id",
    "claim",
    "label",
    "evidence_text",
    "source_name",
    "source_url",
    "is_synthetic",
}

VALID_LABELS = {
    "supported",
    "contradicted",
    "insufficient_evidence",
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path).fillna("")

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    invalid = set(frame["label"].unique()) - VALID_LABELS
    if invalid:
        raise ValueError(f"Invalid labels: {sorted(invalid)}")

    if frame.empty:
        raise ValueError("Dataset is empty")

    for column in ("claim_id", "document_id", "claim", "evidence_text"):
        empty_rows = frame[column].astype(str).str.strip().eq("")
        if empty_rows.any():
            raise ValueError(
                f"Column '{column}' contains {int(empty_rows.sum())} empty values"
            )

    if frame["claim_id"].duplicated().any():
        raise ValueError("claim_id values must be unique")

    return frame


def grouped_splits(frame: pd.DataFrame, n_splits: int = 3):
    """Yield train/test indices without splitting document groups."""
    if "document_id" not in frame.columns:
        raise ValueError("Dataset must contain document_id")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    n_groups = frame["document_id"].nunique()
    if n_groups < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique documents; found {n_groups}"
        )

    splitter = GroupKFold(n_splits=n_splits)

    yield from splitter.split(
        frame["claim"],
        frame["label"],
        groups=frame["document_id"],
    )
