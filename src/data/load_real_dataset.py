"""
Phase 2/3 (real data) — Loads the REAL Kaggle "Customer Support on
Twitter" export and reshapes it into this project's working schema:

    thread_id, brand, customer_message, agent_response, created_at

Adapter logic (documented per README section 18):
  - The raw file has one row per tweet: tweet_id, author_id, inbound,
    created_at, text, response_tweet_id, in_response_to_tweet_id.
  - A "brand" reply is any row where inbound == False; its author_id
    IS the brand handle (e.g. "AppleSupport", "AmazonHelp").
  - We pair each brand reply to the customer tweet it was written in
    response to (in_response_to_tweet_id -> that tweet's text), keeping
    only pairs where the original tweet is inbound == True (i.e. a real
    customer, not a brand replying to another brand).
  - Only the FIRST brand reply in a thread is kept per customer tweet
    (a customer tweet can technically have multiple response_tweet_ids;
    we take the earliest by tweet_id to avoid one customer message
    appearing multiple times with different responses, which would
    inflate/duplicate a "case").
  - @mentions and URLs are stripped from both sides (real support tweets
    are full of "@105835" handles and t.co links that add noise without
    information for intent classification).

Run:
    python -m src.data.load_real_dataset --brands AmazonHelp AppleSupport Uber_Support SpotifyCares Delta
"""
import argparse
import re

import pandas as pd

from src.utils.config import load_config, project_path

RAW_TWCS_PATH = "/home/claude/raw_data/twcs/twcs.csv"  # source file, not committed (too large)

MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")


def strip_noise(text: str) -> str:
    text = MENTION_RE.sub("", str(text))
    text = URL_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_pairs(df: pd.DataFrame, brands: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["created_at_parsed"] = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce"
    )

    inbound = df[df["inbound"]].set_index("tweet_id")
    brand_replies = df[(~df["inbound"]) & (df["author_id"].isin(brands))].copy()
    brand_replies = brand_replies.dropna(subset=["in_response_to_tweet_id"])
    brand_replies["in_response_to_tweet_id"] = brand_replies["in_response_to_tweet_id"].astype(int)

    # keep only the earliest brand reply per source customer tweet
    brand_replies = brand_replies.sort_values("tweet_id").drop_duplicates(
        subset="in_response_to_tweet_id", keep="first"
    )

    rows = []
    for _, reply in brand_replies.iterrows():
        cust_id = reply["in_response_to_tweet_id"]
        if cust_id not in inbound.index:
            continue
        cust = inbound.loc[cust_id]
        # inbound.loc can return a DataFrame if tweet_id somehow duplicated; guard
        if isinstance(cust, pd.DataFrame):
            cust = cust.iloc[0]

        cust_text = strip_noise(cust["text"])
        reply_text = strip_noise(reply["text"])
        if len(cust_text) < 3 or len(reply_text) < 3:
            continue

        rows.append(
            {
                "thread_id": f"T{int(cust_id)}",
                "brand": reply["author_id"],
                "customer_message": cust_text,
                "agent_response": reply_text,
                "created_at": reply["created_at_parsed"],
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brands", nargs="+", default=None,
                         help="Brand author_id handles to include, e.g. AppleSupport AmazonHelp")
    parser.add_argument("--max_per_brand", type=int, default=1200,
                         help="Reproducible cap per brand after pairing, to keep the project fast (Phase 3: sampling, not full 3M rows)")
    args = parser.parse_args()

    cfg = load_config()
    brands = args.brands or cfg["dataset"]["real"]["brands"]
    seed = cfg["random_seed"]

    print(f"Loading raw file: {RAW_TWCS_PATH}")
    raw = pd.read_csv(RAW_TWCS_PATH)
    print(f"Raw shape: {raw.shape}")

    pairs = build_pairs(raw, brands)
    print(f"Paired (brand reply -> customer tweet) rows before per-brand cap: {len(pairs)}")
    print(pairs["brand"].value_counts())

    # reproducible per-brand cap so the shipped dataset stays small/fast (Phase 3)
    capped_parts = []
    for brand, group in pairs.groupby("brand"):
        n = min(args.max_per_brand, len(group))
        capped_parts.append(group.sample(n=n, random_state=seed))
    sampled = pd.concat(capped_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_path = project_path(cfg["dataset"]["raw_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False)
    print(f"\nSaved {len(sampled)} rows -> {out_path}")
    print(sampled["brand"].value_counts())

    # dedicated sample_data.csv: 500-1000 AppleSupport-only rows, for the
    # "reproduces in under 15 minutes" requirement
    apple = sampled[sampled["brand"] == "AppleSupport"]
    n_apple = min(1000, max(500, len(apple))) if len(apple) >= 500 else len(apple)
    apple_sample = apple.sample(n=min(n_apple, len(apple)), random_state=seed)
    apple_path = project_path(cfg["dataset"]["sample_path"])
    apple_sample.to_csv(apple_path, index=False)
    print(f"Saved AppleSupport sample ({len(apple_sample)} rows) -> {apple_path}")


if __name__ == "__main__":
    main()
