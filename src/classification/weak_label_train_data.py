"""
IMPORTANT ARCHITECTURE NOTE — read before trusting classifier training numbers.

The real Kaggle dataset has NO intent labels at all — "intent" is not a
column anywhere in the raw export. The only intent-labelled data in this
entire project is the 240-example golden set
(data/evaluation/golden_set.csv), which the assistant hand-reviewed
(see its docstring and docs/LABEL_SELF_CONSISTENCY.md for exactly what
that does and doesn't mean).

That leaves a real problem: training a supervised TF-IDF + Logistic
Regression classifier needs labelled training examples, and manually
labelling all 4,368 real training messages was not feasible in the time
budget (and would just be a bigger version of the same single-annotator
limitation already disclosed for the golden set).

SOLUTION USED HERE, disclosed honestly: a keyword/rule-based WEAK
labeller assigns a "silver" intent label to each training message. This
is standard practice for bootstrapping a classifier when no labelled
training corpus exists (distant/weak supervision) — but it means
training labels are LOWER QUALITY than golden-set labels, and the
classifier is only as good as these keyword rules. This is exactly the
kind of "weakly labelled" data the master brief asks to be clearly
distinguished from human-reviewed labels (Phase 6) — here we go further
and distinguish it from the golden set too.

Messages matching no keyword rule are weak-labelled "other" (silver),
same as the golden set's genuine "other" class.

Run:
    python -m src.classification.weak_label_train_data
"""
import pandas as pd

from src.utils.config import load_config, project_path

# Ordered rules: first match wins. Deliberately simple and readable —
# these are NOT the same as the escalation risk keywords in config.yaml,
# though there is some overlap (both were designed by the same reasoning
# about what these words usually indicate).
RULES = [
    ("security_fraud", ["hacked", "unauthorized", "fraud", "scam", "phishing", "breached",
                         "compromised", "without my permission", "stole", "stolen"]),
    ("cancellation_request", ["cancel my", "cancel the", "want to cancel", "please cancel",
                               "how do i cancel", "unsubscribe", "cancel subscription",
                               "cancel this order", "cancel it"]),
    ("account_access", ["can't log in", "cant log in", "cannot log in", "locked out",
                         "password reset", "reset my password", "forgot password",
                         "forgot my password", "can't sign in", "cant sign in",
                         "verification code", "2fa", "sign in to my", "log into my"]),
    ("billing_refund", ["refund", "charged", "charge me", "overcharged", "double charge",
                         "billed", "billing", "money back", "reimburse"]),
    ("shipping_delivery", ["damaged", "never arrived", "didn't receive", "did not receive",
                            "wrong item", "missing package", "missing order", "lost package",
                            "wrong address", "package was"]),
    ("order_status", ["where is my order", "where's my order", "track my order",
                       "order status", "when will my order", "has my order shipped",
                       "update on my order", "where is my package"]),
    ("cancellation_request", []),  # placeholder kept for rule ordering clarity, no-op
    ("technical_issue", ["crash", "crashing", "bug", "glitch", "not working", "isn't working",
                          "doesn't work", "won't work", "error", "broken", "freeze", "freezing"]),
]


def weak_label(text: str) -> str:
    t = text.lower()
    for intent, keywords in RULES:
        if keywords and any(kw in t for kw in keywords):
            return intent
    return "other"


def main():
    cfg = load_config()
    train_path = project_path(cfg["dataset"]["train_path"])
    df = pd.read_csv(train_path)

    df["intent"] = df["customer_message"].apply(weak_label)
    df["label_type"] = "weak_keyword_rule"  # explicitly distinguished from golden-set labels

    out_path = project_path("data/processed/train_weak_labelled.csv")
    df.to_csv(out_path, index=False)

    print(f"Weak-labelled {len(df)} training rows -> {out_path}")
    print(df["intent"].value_counts())


if __name__ == "__main__":
    main()
