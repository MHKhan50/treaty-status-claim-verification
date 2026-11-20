from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

def make_pipeline(model_type="mlp", seed=17):
    if model_type not in {"mlp", "logreg"}:
        raise ValueError("model_type must be 'mlp' or 'logreg'")
    classifier = (
        MLPClassifier(hidden_layer_sizes=(16,), solver="lbfgs", max_iter=1000, random_state=seed, alpha=0.01)
        if model_type == "mlp"
        else LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    )
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=12000,
                                  strip_accents="unicode", sublinear_tf=True)),
        ("classifier", classifier),
    ])

def train_model(frame, model_dir, model_type="mlp", seed=17):
    model = make_pipeline(model_type, seed)
    model.fit(frame["claim"].astype(str), frame["label"].astype(str))
    target = Path(model_dir)
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target / f"{model_type}.joblib")
    return model

def evaluate_model(model, frame):
    predictions = model.predict(frame["claim"].astype(str))
    return {
        "accuracy": float(accuracy_score(frame["label"], predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(frame["label"], predictions)),
        "macro_f1": float(f1_score(frame["label"], predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            frame["label"], predictions, output_dict=True, zero_division=0
        ),
    }

def load_model(model_dir, model_type="mlp"):
    path = Path(model_dir) / f"{model_type}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run `warheadverify train` first.")
    return joblib.load(path)
