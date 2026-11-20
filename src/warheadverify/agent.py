from .retrieval import EvidenceRetriever

class VerificationAgent:
    # Deterministic workflow; it does not autonomously access external systems.
    def __init__(self, model, evidence_frame, sufficiency_threshold=0.04):
        self.model = model
        self.evidence_frame = evidence_frame.reset_index(drop=True)
        self.retriever = EvidenceRetriever(self.evidence_frame["evidence_text"].tolist())
        self.sufficiency_threshold = float(sufficiency_threshold)

    def verify(self, claim, top_k=3):
        retrieved = self.retriever.retrieve(claim, top_k=top_k)
        predicted = str(self.model.predict([claim])[0])
        best_score = retrieved[0]["similarity"] if retrieved else 0.0
        verdict = predicted if best_score >= self.sufficiency_threshold else "insufficient_evidence"
        return {
            "claim": claim,
            "predicted_label": predicted,
            "verdict": verdict,
            "evidence_sufficiency_score": best_score,
            "retrieved_evidence": retrieved,
            "limitations": [
                "Similarity is a retrieval heuristic, not proof of entailment.",
                "The model is trained on the supplied dataset and may not generalise.",
                "Human review of original documents is required for substantive conclusions."
            ],
        }
