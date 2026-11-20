import pytest
from warheadverify.retrieval import EvidenceRetriever

def test_retriever_returns_ranked_evidence():
    retriever = EvidenceRetriever([
        "Article 1 requires annual public reporting.",
        "Article 2 establishes a fictional meeting schedule."
    ])
    result = retriever.retrieve("annual public reporting", top_k=1)
    assert len(result) == 1
    assert "annual" in result[0]["evidence_text"]
    assert 0.0 <= result[0]["similarity"] <= 1.0

def test_invalid_top_k_rejected():
    retriever = EvidenceRetriever(["one evidence passage"])
    with pytest.raises(ValueError):
        retriever.retrieve("query", top_k=0)
