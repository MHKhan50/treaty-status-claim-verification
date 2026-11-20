from pathlib import Path
from warheadverify.data import load_dataset
from warheadverify.modeling import train_model
from warheadverify.agent import VerificationAgent

DATA = Path(__file__).parents[1] / "data" / "synthetic_demo.csv"

def test_agent_returns_auditable_structure(tmp_path):
    frame = load_dataset(DATA)
    model = train_model(frame, tmp_path, model_type="logreg")
    agent = VerificationAgent(model, frame[["evidence_text"]].head(5))
    result = agent.verify("A fictional accord requires annual reporting.")
    assert result["verdict"] in {"supported", "contradicted", "insufficient_evidence"}
    assert "retrieved_evidence" in result
    assert "limitations" in result
