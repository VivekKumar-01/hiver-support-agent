"""
Phase 7 — Baseline 1: majority-class classifier.

Beginner note: this baseline always predicts the single most common
intent from training data, no matter what the input is. It exists so we
have a floor to beat — if our real model can't beat "always guess the
most common class", it isn't adding value.
"""
import json

import pandas as pd

from src.evaluation.metrics import classification_metrics
from src.utils.config import load_config, project_path


class MajorityClassifier:
    def __init__(self):
        self.majority_label = None

    def fit(self, y):
        self.majority_label = pd.Series(y).value_counts().idxmax()
        return self

    def predict(self, X):
        return [self.majority_label] * len(X)


def main():
    cfg = load_config()
    train_df = pd.read_csv(project_path(cfg["dataset"]["train_weak_labelled_path"]))
    golden_df = pd.read_csv(project_path(cfg["golden_set"]["path"]))

    clf = MajorityClassifier().fit(train_df["intent"])
    preds = clf.predict(golden_df["customer_message"])

    labels = sorted(set(train_df["intent"]) | set(golden_df["intent"]))
    metrics = classification_metrics(golden_df["intent"], preds, labels=labels)

    print(f"Majority class learned from training data: '{clf.majority_label}'")
    print(f"Accuracy:    {metrics['accuracy']:.3f}")
    print(f"Macro F1:    {metrics['macro_f1']:.3f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.3f}")

    out_path = project_path("outputs/metrics/baseline1_majority.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"majority_label": clf.majority_label, **metrics}, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
