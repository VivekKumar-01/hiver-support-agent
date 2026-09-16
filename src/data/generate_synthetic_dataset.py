"""
Generates a SYNTHETIC stand-in for the Kaggle "Customer Support on Twitter"
dataset.

WHY THIS EXISTS
----------------
This project was built in a sandboxed environment with no internet access,
so the real dataset (a multi-GB Kaggle download) could not be fetched. To
still deliver a working, honestly-evaluated pipeline, we generate a dataset
with the SAME shape as the real one:

    tweet_id, author_id, inbound, created_at, text, response_tweet_id,
    in_response_to_tweet_id  (real dataset's actual columns)

reshaped here into the customer-support-agent-friendly form:

    thread_id, brand, customer_message, agent_response, intent (ground
    truth, since we generated it), created_at

Every number this project reports (accuracy, F1, retrieval scores, etc.)
is a REAL measurement taken on THIS synthetic data — nothing is invented.
See DECISION_LOG.md decision D1 and README.md "Limitations" for the full
explanation and for exact instructions to swap in the real dataset later.

Run:
    python -m src.data.generate_synthetic_dataset
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config, project_path

BRANDS = ["AmazonHelp", "AppleSupport", "SpotifyCares", "DeltaAssist", "UberSupport"]

# 8 intents, derived the way we'd derive them from a real transcript sample:
# by reading representative customer messages and grouping them into a small,
# usable taxonomy (see docs/INTENT_TAXONOMY.md for definitions).
INTENT_TEMPLATES = {
    "order_status": [
        "Hi, where is my order #{oid}? It's been {days} days.",
        "Can you tell me the status of order {oid}? No update since I placed it.",
        "Still waiting on order {oid}, tracking hasn't moved in {days} days.",
    ],
    "billing_refund": [
        "I was charged twice for order {oid}, need a refund please.",
        "My refund for {oid} still hasn't arrived after {days} days.",
        "Why was I billed ${amt} when the item was on sale?",
    ],
    "account_access": [
        "I can't log into my account, it says password incorrect even after reset.",
        "Locked out of my account since yesterday, 2FA code never arrives.",
        "My account got logged out on all devices and I can't sign back in.",
    ],
    "technical_issue": [
        "The app keeps crashing every time I open it on my phone.",
        "Getting error code {code} whenever I try to check out.",
        "Website won't load past the loading screen, tried 3 browsers.",
    ],
    "cancellation_request": [
        "I want to cancel my subscription immediately, please confirm.",
        "Please cancel order {oid}, I no longer need it.",
        "How do I cancel my membership? Don't want to be charged again.",
    ],
    "shipping_delivery": [
        "Package for {oid} arrived damaged, box was crushed.",
        "Courier marked {oid} as delivered but I never received it.",
        "Can I change the delivery address for order {oid}? It hasn't shipped yet.",
    ],
    "security_fraud": [
        "I think my account was hacked, there are purchases I didn't make.",
        "Someone made an unauthorized charge of ${amt} on my card, this is fraud.",
        "I'm getting login alerts from a country I've never been to.",
    ],
    "general_feedback": [
        "Just wanted to say your support team was amazing today, thank you!",
        "Really disappointed with the service lately, this is the third issue this month.",
        "Quick question, does the premium plan include priority support?",
    ],
}

RESPONSE_TEMPLATES = {
    "order_status": "Hi, thanks for reaching out! Order {oid} last showed movement {days} days ago; we're checking with the carrier and will update you within 24 hours.",
    "billing_refund": "We're sorry about that! We've located the charge on order {oid} and refunds typically post within 5-7 business days once approved.",
    "account_access": "Sorry for the trouble logging in. Please try the password reset link sent to your registered email; if it doesn't arrive, we can manually verify your identity.",
    "technical_issue": "Thanks for the report — that sounds like error {code}. Try clearing the app cache and reinstalling; if it persists, our engineering team can escalate it.",
    "cancellation_request": "We can help with that. Your cancellation request for order {oid} has been received and will be confirmed by email shortly.",
    "shipping_delivery": "Sorry to hear about order {oid}. We've flagged it with our shipping partner and a replacement or refund will be offered once confirmed.",
    "security_fraud": "This has been flagged as a priority security case and passed to our fraud team; please also change your password immediately as a precaution.",
    "general_feedback": "Thank you so much for letting us know — we really appreciate it and will pass this along to the team!",
}


def _noisy(text: str, rng: random.Random) -> str:
    """Light, realistic noise: casing, extra spaces, missing punctuation."""
    if rng.random() < 0.15:
        text = text.lower()
    if rng.random() < 0.1:
        text = text.replace(".", "")
    if rng.random() < 0.1:
        text = text + " " * rng.randint(1, 3)
    return text


def generate(n_conversations: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    rows = []
    intents = list(INTENT_TEMPLATES.keys())

    for i in range(n_conversations):
        intent = intents[i % len(intents)]  # round-robin -> roughly balanced, then we skew below
        # Skew distribution so it's not perfectly balanced (more realistic)
        if rng.random() < 0.3:
            intent = rng.choice(intents)

        brand = rng.choice(BRANDS)
        cust_template = rng.choice(INTENT_TEMPLATES[intent])
        oid = rng.randint(100000, 999999)
        days = rng.randint(1, 14)
        amt = round(rng.uniform(9.99, 249.99), 2)
        code = rng.choice(["E101", "E204", "E502", "TIMEOUT_42", "AUTH_9"])

        customer_message = cust_template.format(oid=oid, days=days, amt=amt, code=code)
        customer_message = _noisy(customer_message, rng)

        agent_response = RESPONSE_TEMPLATES[intent].format(oid=oid, days=days, amt=amt, code=code)

        thread_id = f"T{i:05d}"
        created_at = pd.Timestamp("2024-01-01") + pd.Timedelta(
            days=int(np_rng.integers(0, 300)), hours=int(np_rng.integers(0, 23))
        )

        rows.append(
            {
                "thread_id": thread_id,
                "brand": brand,
                "customer_message": customer_message,
                "agent_response": agent_response,
                "intent": intent,
                "created_at": created_at,
            }
        )

    df = pd.DataFrame(rows)

    # --- inject realistic messiness so cleaning (Phase 4) has real work to do ---
    # 1) duplicate ~3% of rows
    n_dupes = max(1, int(0.03 * len(df)))
    dupe_rows = df.sample(n=n_dupes, random_state=seed)
    df = pd.concat([df, dupe_rows], ignore_index=True)

    # 2) null out ~1.5% of customer_message and ~1.5% of agent_response
    for col in ["customer_message", "agent_response"]:
        idx = df.sample(frac=0.015, random_state=seed + 1).index
        df.loc[idx, col] = None

    # 3) shuffle row order (real exports aren't sorted by intent)
    df = df.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)

    return df


def main():
    cfg = load_config()
    seed = cfg["random_seed"]
    n = cfg["dataset"]["n_conversations"]
    out_path = project_path(cfg["dataset"]["raw_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate(n, seed)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
