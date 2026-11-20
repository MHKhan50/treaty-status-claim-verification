"""Lexical baselines: majority class, claim-only and claim + evidence TF-IDF logistic regression and MLP."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

from .metrics import macro_f1


def _pair(claims, evidence):
    return [f"{c} [EVIDENCE] {e}" for c, e in zip(claims, evidence)]


def fit_predict(kind, train, val, test, ev_train=None, ev_val=None, ev_test=None, seed=0):
    """kind: majority | claim_lr | pair_lr | pair_mlp. Regularisation chosen on validation macro-F1."""
    y = train["label"].to_numpy()
    if kind == "majority":
        major = train["label"].value_counts().idxmax()
        return np.array([major] * len(test)), {"label": major}
    use_pair = kind.startswith("pair")
    x_tr = _pair(train["claim"], ev_train) if use_pair else list(train["claim"])
    x_va = _pair(val["claim"], ev_val) if use_pair else list(val["claim"])
    x_te = _pair(test["claim"], ev_test) if use_pair else list(test["claim"])
    best = None
    grid = [0.3, 1.0, 3.0, 10.0] if kind != "pair_mlp" else [1e-4, 1e-3]
    for g in grid:
        clf = (LogisticRegression(C=g, max_iter=3000, class_weight="balanced", random_state=seed) if kind != "pair_mlp"
               else MLPClassifier(hidden_layer_sizes=(256,), alpha=g, max_iter=60, early_stopping=True, random_state=seed))
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2), clf).fit(x_tr, y)
        score = macro_f1(val["label"], model.predict(x_va))
        if best is None or score > best[0]:
            best = (score, g, model)
    return best[2].predict(x_te), {"param": best[1], "val_macro_f1": best[0]}
