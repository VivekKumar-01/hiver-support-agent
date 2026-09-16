"""
Why this script exists
-----------------------
The golden set (Phase 6) was sampled from the SAME synthetic template
pool used to train the TF-IDF + Logistic Regression classifier, so
100% accuracy on it is expected, not impressive — the messages are
close paraphrases of training examples by construction. That is exactly
the kind of "misleading headline number" Phase 28 asks us to call out.

This script is a small, HAND-WRITTEN set of harder, out-of-distribution
examples (different phrasing, some multi-intent, some ambiguous, some
with no template match at all) run through the real pipeline, to find
GENUINE failure modes for the Phase 27 failure analysis. These labels
ARE assigned by us reading the message (closest thing to "human
labelling" in this project) and are marked as such.

Run:
    python -m scripts.stress_test
"""
import json

import pandas as pd

from src.pipeline import run_agent
from src.utils.config import project_path

# (message, brand, human_expected_intent, human_expected_action, note)
STRESS_CASES = [
    ("ugh this is the 3rd time ive had to message you about my package not showing up, so done with this",
     "AmazonHelp", "shipping_delivery", "ESCALATE", "Repeated-contact frustration signal, no explicit keyword"),
    ("can you guys help me get into my account, forgot everything about it lol",
     "SpotifyCares", "account_access", "ESCALATE", "Casual phrasing, no direct password/2FA words"),
    ("not sure if this is a billing thing or a bug but I got charged and the app also crashed right after",
     "AppleSupport", "billing_refund", "ESCALATE", "Genuinely multi-intent (billing + technical)"),
    ("is priority support included if I upgrade my plan",
     "SpotifyCares", "general_feedback", "AUTO_HANDLE", "Pure pre-sales question, low risk"),
    ("my card was charged by someone in another country, I never authorized this!!",
     "Delta", "security_fraud", "ESCALATE", "Fraud without the literal word 'fraud'"),
    ("driver never showed up and now im being charged a cancellation fee, thats not fair",
     "Uber_Support", "billing_refund", "ESCALATE", "Dispute framed as unfairness, not a refund request"),
    ("just checking in, any update on the thing I emailed about last week",
     "AmazonHelp", "order_status", "ESCALATE", "Vague follow-up, no order number or details at all"),
    ("thanks for fixing that so fast, you guys rock",
     "AppleSupport", "general_feedback", "AUTO_HANDLE", "Clear positive feedback, easy case"),
]


def main():
    rows = []
    for message, brand, expected_intent, expected_action, note in STRESS_CASES:
        result = run_agent(message, brand)
        rows.append(
            {
                "brand": brand,
                "customer_message": message,
                "human_expected_intent": expected_intent,
                "predicted_intent": result["predicted_intent"],
                "intent_correct": result["predicted_intent"] == expected_intent,
                "human_expected_action": expected_action,
                "predicted_action": result["decision"],
                "action_correct": result["decision"] == expected_action,
                "confidence": result["confidence"],
                "top_similarity": result["retrieved_cases"][0]["similarity"] if result["retrieved_cases"] else 0.0,
                "generated_reply": result["generated_reply"],
                "note": note,
            }
        )

    df = pd.DataFrame(rows)
    out_path = project_path("outputs/metrics/stress_test_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Stress test: {df['intent_correct'].sum()}/{len(df)} intents correct, "
          f"{df['action_correct'].sum()}/{len(df)} actions correct")
    print(f"Saved -> {out_path}")
    print(df[["customer_message", "human_expected_intent", "predicted_intent", "intent_correct",
               "human_expected_action", "predicted_action", "action_correct"]].to_string())


if __name__ == "__main__":
    main()
