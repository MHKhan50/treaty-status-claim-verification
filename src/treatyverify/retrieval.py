"""Evidence retrieval over the document corpus: TF-IDF (word n-grams) and an optional dense encoder."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EvidenceRetriever:
    """TF-IDF retriever. ``retrieve`` returns ranked (doc_id, score) pairs."""

    def __init__(self, doc_ids, texts):
        self.doc_ids = list(doc_ids)
        self.texts = [str(t) for t in texts]
        if not self.texts:
            raise ValueError("At least one evidence document is required")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def scores(self, queries):
        return cosine_similarity(self.vectorizer.transform(list(queries)), self.matrix)

    def retrieve(self, query, top_k=3):
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        s = self.scores([query]).ravel()
        order = np.argsort(-s, kind="stable")[:top_k]
        return [(self.doc_ids[i], float(s[i])) for i in order]

    def rank(self, queries, top_k=5):
        s = self.scores(queries)
        order = np.argsort(-s, axis=1, kind="stable")[:, :top_k]
        return [[self.doc_ids[j] for j in row] for row in order], np.take_along_axis(s, order, axis=1)


class DenseRetriever(EvidenceRetriever):
    """Sentence-embedding retriever (e.g. intfloat/e5-small-v2); needs sentence-transformers."""

    def __init__(self, doc_ids, texts, model_name="intfloat/e5-small-v2", device=None):
        from sentence_transformers import SentenceTransformer
        self.doc_ids, self.texts = list(doc_ids), [str(t) for t in texts]
        self.model = SentenceTransformer(model_name, device=device)
        self.prefix = ("query: ", "passage: ") if "e5" in model_name else ("", "")
        self.matrix = self.model.encode([self.prefix[1] + t for t in self.texts], normalize_embeddings=True,
                                        batch_size=128, show_progress_bar=False)

    def scores(self, queries):
        q = self.model.encode([self.prefix[0] + t for t in queries], normalize_embeddings=True, batch_size=128,
                              show_progress_bar=False)
        return q @ self.matrix.T
