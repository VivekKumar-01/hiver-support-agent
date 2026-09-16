"""
Phase 15 — Evaluation harness.

Runs the FULL pipeline (run_agent) over every row of the golden set and
computes:
  - intent classification metrics (accuracy, macro/weighted F1, confusion matrix)
  - escalation metrics (precision/recall/confusion matrix vs expected_action)
  - retrieval similarity distribution
  - reply-quality metrics: NOT MEASURED (no LLM judge / no API key available —
    see LLM-as-judge section of README; this is explicitly reported as
    "Not measured yet" rather than fabricated)

Run:
    python -m src.evaluation.run_evaluation
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.metrics import classification_metrics, bootstrap_ci
from src.pipeline import run_agent
from src.utils.config import load_config, project_path


def main(golden_set_path: str = None):
    cfg = load_config()
    path = golden_set_path or cfg["golden_set"]["path"]
    golden = pd.read_csv(project_path(path) if not str(path).startswith("/") else path)

    predicted_intents, true_intents = [], []
    predicted_actions, expected_actions = [], []
    confidences, similarities = [], []
    all_results = []

    for _, row in golden.iterrows():
        result = run_agent(row["customer_message"], row["brand"])
        all_results.append({**result, "golden_id": row["id"], "true_intent": row["intent"],
                             "expected_action": row["expected_action"]})

        predicted_intents.append(result["predicted_intent"])
        true_intents.append(row["intent"])
        predicted_actions.append(result["decision"])
        expected_actions.append(row["expected_action"])
        confidences.append(result["confidence"])
        similarities.append(result["retrieved_cases"][0]["similarity"] if result["retrieved_cases"] else 0.0)

    # ---- 1. Intent classification metrics (this IS the "proposed system" row) ----
    labels = sorted(golden["intent"].unique())
    intent_metrics = classification_metrics(true_intents, predicted_intents, labels=labels)

    from sklearn.metrics import accuracy_score, f1_score
    intent_accuracy_ci = bootstrap_ci(true_intents, predicted_intents, accuracy_score)
    intent_macro_f1_ci = bootstrap_ci(
        true_intents, predicted_intents,
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0, labels=labels),
    )

    # ---- 2. Escalation metrics ----
    action_labels = ["AUTO_HANDLE", "ESCALATE"]
    escalation_metrics = classification_metrics(expected_actions, predicted_actions, labels=action_labels)
    escalation_accuracy_ci = bootstrap_ci(expected_actions, predicted_actions, accuracy_score)

    # false auto-handling = predicted AUTO_HANDLE when expected ESCALATE (the dangerous error)
    false_auto_handle = sum(
        1 for p, e in zip(predicted_actions, expected_actions) if p == "AUTO_HANDLE" and e == "ESCALATE"
    )
    false_escalate = sum(
        1 for p, e in zip(predicted_actions, expected_actions) if p == "ESCALATE" and e == "AUTO_HANDLE"
    )

    # ---- 3. Retrieval similarity distribution ----
    sim_series = pd.Series(similarities)
    retrieval_summary = {
        "mean_top1_similarity": float(sim_series.mean()),
        "median_top1_similarity": float(sim_series.median()),
        "min_top1_similarity": float(sim_series.min()),
        "max_top1_similarity": float(sim_series.max()),
    }

    # ---- Save everything ----
    metrics_dir = project_path("outputs/metrics")
    figures_dir = project_path("outputs/figures")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "proposed_system_intent_metrics.json", "w") as f:
        json.dump({**intent_metrics, "accuracy_ci_95": intent_accuracy_ci,
                   "macro_f1_ci_95": intent_macro_f1_ci}, f, indent=2)
    with open(metrics_dir / "escalation_metrics.json", "w") as f:
        json.dump(
            {**escalation_metrics, "false_auto_handle_count": false_auto_handle,
             "false_escalate_count": false_escalate, "n_examples": len(golden),
             "accuracy_ci_95": escalation_accuracy_ci},
            f, indent=2,
        )
    with open(metrics_dir / "retrieval_summary.json", "w") as f:
        json.dump(retrieval_summary, f, indent=2)
    with open(metrics_dir / "reply_quality_metrics.json", "w") as f:
        json.dump(
            {"status": "Not measured yet",
             "reason": "No LLM API key available in this sandboxed environment; "
                        "LLM-as-judge module exists at src/evaluation/llm_judge.py but was not "
                        "executed. See README.md > LLM-as-judge for how to run it with a key."},
            f, indent=2,
        )

    pd.DataFrame(
        [{"golden_id": r["golden_id"], "brand": r["brand"], "true_intent": r["true_intent"],
          "predicted_intent": r["predicted_intent"], "confidence": r["confidence"],
          "expected_action": r["expected_action"], "predicted_action": r["decision"],
          "triggered_rule": r["triggered_rule"], "top_similarity": r["retrieved_cases"][0]["similarity"]
          if r["retrieved_cases"] else 0.0, "generated_reply": r["generated_reply"]}
         for r in all_results]
    ).to_csv(metrics_dir / "predictions_detailed.csv", index=False)

    # ---- Figures ----
    # baseline comparison
    with open(metrics_dir / "baseline1_majority.json") as f:
        b1 = json.load(f)
    with open(metrics_dir / "baseline2_tfidf_lr.json") as f:
        b2 = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4))
    systems = ["Majority\nbaseline", "TF-IDF + LR\n(proposed)"]
    macro_f1s = [b1["macro_f1"], b2["macro_f1"]]
    ax.bar(systems, macro_f1s, color=["#999999", "#2b6cb0"])
    ax.set_ylabel("Macro F1")
    ax.set_title("Intent classification: baseline comparison (real golden-set results)")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(macro_f1s):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline_comparison.png", dpi=120)
    plt.close(fig)

    # intent distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    golden["intent"].value_counts().plot(kind="bar", ax=ax, color="#2b6cb0")
    ax.set_title("Golden set intent distribution")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(figures_dir / "intent_distribution.png", dpi=120)
    plt.close(fig)

    # escalation confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm = escalation_metrics["confusion_matrix"]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(action_labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(action_labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Expected")
    ax.set_title("Escalation confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(figures_dir / "escalation_confusion_matrix.png", dpi=120)
    plt.close(fig)

    # retrieval similarity distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sim_series.plot(kind="hist", bins=20, ax=ax, color="#2b6cb0")
    ax.set_title("Top-1 retrieval similarity distribution (golden set)")
    ax.set_xlabel("cosine similarity")
    fig.tight_layout()
    fig.savefig(figures_dir / "retrieval_similarity_distribution.png", dpi=120)
    plt.close(fig)

    # ---- Console summary ----
    print("=" * 60)
    print("INTENT CLASSIFICATION (proposed system, on golden set)")
    print(f"  Accuracy:    {intent_metrics['accuracy']:.3f}  (95% CI: {intent_accuracy_ci['ci_lower']:.3f}-{intent_accuracy_ci['ci_upper']:.3f})")
    print(f"  Macro F1:    {intent_metrics['macro_f1']:.3f}  (95% CI: {intent_macro_f1_ci['ci_lower']:.3f}-{intent_macro_f1_ci['ci_upper']:.3f})")
    print(f"  Weighted F1: {intent_metrics['weighted_f1']:.3f}")
    print("\nESCALATION")
    print(f"  Accuracy:              {escalation_metrics['accuracy']:.3f}  (95% CI: {escalation_accuracy_ci['ci_lower']:.3f}-{escalation_accuracy_ci['ci_upper']:.3f})")
    print(f"  Macro F1:              {escalation_metrics['macro_f1']:.3f}")
    print(f"  False AUTO-HANDLE (dangerous): {false_auto_handle} / {len(golden)}")
    print(f"  False ESCALATE (wasteful):     {false_escalate} / {len(golden)}")
    print("\nRETRIEVAL")
    print(f"  Mean top-1 similarity: {retrieval_summary['mean_top1_similarity']:.3f}")
    print("\nREPLY QUALITY: Not measured yet (no LLM API key in this environment)")
    print(f"\nSaved metrics -> {metrics_dir}")
    print(f"Saved figures -> {figures_dir}")


if __name__ == "__main__":
    main()
