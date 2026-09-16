"""
Phase 10 — Historical retrieval.

Beginner explanation:
- An embedding converts text into numbers so that texts with similar
  meanings tend to have similar numerical representations.
- We'd ideally use a sentence-transformers embedding model + FAISS (a
  fast local search system for finding similar vectors) for this. Both
  are unavailable in this sandbox (no internet to download the model or
  the package — see requirements.txt). So retrieval here uses TF-IDF
  vectors + cosine similarity, which is a weaker but honest, locally
  computable form of the same idea: turn text into numbers, then find
  the historical messages whose numbers are closest to the new message.
- If sentence-transformers + faiss ARE installed (e.g. you run this on
  a machine with internet), this module automatically upgrades to real
  embeddings — see `_try_load_embedder()` below.

retrieve_similar_cases(message, brand, k) is the public function used by
the pipeline.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.config import load_config, project_path


@dataclass
class RetrievedCase:
    similarity: float
    brand: str
    customer_message: str
    historical_response: str
    intent: str
    thread_id: str
    same_brand: bool = False


def _try_load_embedder():
    """Returns a sentence-transformers model if it's installed, else None."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


class HistoricalRetriever:
    def __init__(self, history_df: pd.DataFrame, cfg: dict):
        self.history = history_df.reset_index(drop=True)
        self.cfg = cfg
        self.embedder = _try_load_embedder()

        if self.embedder is not None:
            self.mode = "sentence_transformers"
            self.doc_vectors = self.embedder.encode(
                self.history["customer_message"].tolist(), normalize_embeddings=True
            )
        else:
            self.mode = "tfidf_fallback"
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            self.doc_vectors = self.vectorizer.fit_transform(self.history["customer_message"])

    def _query_vector(self, message: str):
        if self.mode == "sentence_transformers":
            return self.embedder.encode([message], normalize_embeddings=True)
        return self.vectorizer.transform([message])

    def retrieve(self, message: str, brand: str, k: int = 3) -> list:
        q_vec = self._query_vector(message)
        sims = cosine_similarity(q_vec, self.doc_vectors).flatten()

        boost = self.cfg["retrieval"]["same_brand_boost"]
        same_brand_mask = (self.history["brand"] == brand).to_numpy()
        boosted_sims = sims + same_brand_mask * boost

        top_idx = np.argsort(-boosted_sims)[:k]
        results = []
        for idx in top_idx:
            row = self.history.iloc[idx]
            results.append(
                RetrievedCase(
                    similarity=float(sims[idx]),  # report RAW similarity, not the boosted score
                    brand=row["brand"],
                    customer_message=row["customer_message"],
                    historical_response=row["agent_response"],
                    intent=row.get("intent", "unknown"),
                    thread_id=row.get("thread_id", "unknown"),
                    same_brand=bool(same_brand_mask[idx]),
                )
            )
        return results


_RETRIEVER_SINGLETON = None


def get_retriever() -> HistoricalRetriever:
    global _RETRIEVER_SINGLETON
    if _RETRIEVER_SINGLETON is None:
        cfg = load_config()
        history_df = pd.read_csv(project_path(cfg["dataset"]["train_path"]))
        _RETRIEVER_SINGLETON = HistoricalRetriever(history_df, cfg)
    return _RETRIEVER_SINGLETON


def retrieve_similar_cases(message: str, brand: str, k: int = None) -> list:
    cfg = load_config()
    k = k or cfg["retrieval"]["top_k"]
    retriever = get_retriever()
    return retriever.retrieve(message, brand, k)


if __name__ == "__main__":
    # quick manual check
    results = retrieve_similar_cases("My refund still hasn't shown up after a week", "AmazonHelp", k=3)
    for r in results:
        print(f"sim={r.similarity:.3f} same_brand={r.same_brand} brand={r.brand} intent={r.intent}")
        print(f"  msg: {r.customer_message}")
        print(f"  resp: {r.historical_response}")
