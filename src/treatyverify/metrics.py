"""Classification metrics with country-level bootstrap intervals, abstention curves and retrieval metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from .corpus import LABELS


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, labels=list(LABELS), average="macro", zero_division=0))


def per_class_f1(y, p) -> dict:
    vals = f1_score(y, p, labels=list(LABELS), average=None, zero_division=0)
    return {lab: float(v) for lab, v in zip(LABELS, vals)}


def bootstrap_ci(y, p, groups, fn=macro_f1, n=1000, seed=0) -> list:
    """Resample groups (countries) with replacement; percentile 95% interval."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        take = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))])
        vals.append(fn(y[take], p[take]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def report(y, p, groups, n_boot=1000) -> dict:
    return {"macro_f1": macro_f1(y, p), "macro_f1_ci95": bootstrap_ci(y, p, groups, n=n_boot),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)), "per_class_f1": per_class_f1(y, p),
            "n": int(len(y))}


def risk_coverage(y, p, confidence, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)) -> dict:
    y, p, c = np.asarray(y), np.asarray(p), np.asarray(confidence, dtype=float)
    order = np.argsort(-c, kind="stable")
    out = {}
    for cov in coverages:
        keep = order[: max(1, int(round(cov * len(y))))]
        out[f"{cov:.1f}"] = {"accuracy": float((y[keep] == p[keep]).mean()), "macro_f1": macro_f1(y[keep], p[keep])}
    return out


def retrieval_metrics(gold_ids, ranked_ids, ks=(1, 3, 5)) -> dict:
    """Recall@k and MRR over claims that have a gold document."""
    pairs = [(g, r) for g, r in zip(gold_ids, ranked_ids) if g]
    out = {f"recall@{k}": float(np.mean([g in r[:k] for g, r in pairs])) for k in ks}
    out["mrr"] = float(np.mean([1 / (r.index(g) + 1) if g in r else 0.0 for g, r in pairs]))
    out["n"] = len(pairs)
    return out
