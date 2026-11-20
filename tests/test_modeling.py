from pathlib import Path
from warheadverify.data import load_dataset
from warheadverify.modeling import train_model, evaluate_model, load_model

DATA = Path(__file__).parents[1] / "data" / "synthetic_demo.csv"

def test_train_save_load_and_evaluate(tmp_path):
    frame = load_dataset(DATA)
    model = train_model(frame, tmp_path, model_type="logreg", seed=5)
    metrics = evaluate_model(model, frame)
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    loaded = load_model(tmp_path, "logreg")
    assert len(loaded.predict(["A fictional treaty requires reporting."])) == 1
