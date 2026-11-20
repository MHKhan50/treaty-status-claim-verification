from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EvidenceRetriever:
    def __init__(self, evidence_texts):
        self.evidence_texts = [str(item) for item in evidence_texts]
        if not self.evidence_texts:
            raise ValueError("At least one evidence passage is required")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        self.matrix = self.vectorizer.fit_transform(self.evidence_texts)

    def retrieve(self, query, top_k=3):
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        order = scores.argsort()[::-1][:top_k]
        return [{"evidence_text": self.evidence_texts[int(i)], "similarity": float(scores[i])}
                for i in order]
