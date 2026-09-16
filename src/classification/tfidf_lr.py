"""
Phase 8 — Baseline 2: TF-IDF + Logistic Regression.

Beginner explanation:
- TF-IDF turns each message into a vector of numbers based on which
  words appear and how distinctive they are (a word that appears in
  every message, like "the", gets a low weight; a rare, on-topic word
  like "refund" gets a high weight).
- Logistic Regression then learns, from labelled examples, which
  combinations of those word-weights point to which intent.

This is also the classifier used by the live pipeline (run_agent).
IMPORTANT: trained on WEAK/keyword-rule labels (see
src/classification/weak_label_train_data.py), not human-reviewed
labels — the real dataset has no ground-truth intent column at all.
Evaluated against the golden set, which IS individually reviewed (see
its own docstring for exactly what that means and doesn't mean). See
DECISION_LOG.md D17.
"""
import json
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import classification_metrics
from src.utils.config import load_config, project_path

MODEL_PATH = "outputs/metrics/tfidf_lr_model.pkl"


def build_pipeline(cfg) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=cfg["classification"]["tfidf_max_features"],
                    ngram_range=tuple(cfg["classification"]["tfidf_ngram_range"]),
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=cfg["classification"]["logreg_C"],
                    max_iter=cfg["classification"]["logreg_max_iter"],
                ),
            ),
        ]
    )


def train_and_save(cfg=None):
    cfg = cfg or load_config()
    train_df = pd.read_csv(project_path(cfg["dataset"]["train_weak_labelled_path"]))
    pipe = build_pipeline(cfg)
    pipe.fit(train_df["customer_message"], train_df["intent"])

    model_path = project_path(MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    return pipe


def load_model(cfg=None):
    cfg = cfg or load_config()
    model_path = project_path(MODEL_PATH)
    if not model_path.exists():
        return train_and_save(cfg)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def main():
    cfg = load_config()
    train_df = pd.read_csv(project_path(cfg["dataset"]["train_weak_labelled_path"]))
    golden_df = pd.read_csv(project_path(cfg["golden_set"]["path"]))

    pipe = train_and_save(cfg)
    preds = pipe.predict(golden_df["customer_message"])
    probs = pipe.predict_proba(golden_df["customer_message"]).max(axis=1)

    labels = sorted(set(train_df["intent"]) | set(golden_df["intent"]))
    metrics = classification_metrics(golden_df["intent"], preds, labels=labels)

    print(f"Accuracy:    {metrics['accuracy']:.3f}")
    print(f"Macro F1:    {metrics['macro_f1']:.3f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.3f}")
    print(f"Mean predicted confidence: {probs.mean():.3f}")

    out_path = project_path("outputs/metrics/baseline2_tfidf_lr.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
