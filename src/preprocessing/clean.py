"""
Phase 4 — Data cleaning.

Beginner note: "cleaning" here means making the raw export usable and
trustworthy: dropping rows we can't use, removing exact duplicates, and
normalizing whitespace/casing so downstream models see consistent text.

UPDATED for real data: real Twitter support text has HTML entities
(&gt;, &amp;), agent sign-off codes (^AZ, *TMT — individual agent
initials brands append to replies), and highly variable casing/punctuation
that the synthetic dataset never had. Handled explicitly below and
documented since these are exactly the kind of real-world artifacts a
naive read of "clean the text" would miss.

Run:
    python -m src.preprocessing.clean
"""
import html
import re

import pandas as pd

from src.utils.config import load_config, project_path

SIGNOFF_RE = re.compile(r"\s*[\^*/][A-Za-z]{2,4}\s*$")  # e.g. "...^AZ", "...*TMT", "...  /TB" at end of a reply


def normalize_text(text: str) -> str:
    text = html.unescape(str(text))       # &gt; -> >, &amp; -> &, etc.
    text = SIGNOFF_RE.sub("", text)        # strip trailing agent-initial sign-offs
    text = re.sub(r"\s+", " ", text).strip()  # collapse repeated whitespace
    return text


def is_mostly_ascii(text: str, threshold: float = 0.85) -> bool:
    """
    Real-data scope decision (see DECISION_LOG.md D16): this project's
    intent taxonomy and templates were derived from English-language
    examples. Rather than silently mis-classify non-English messages, we
    explicitly filter them out and document the exclusion, instead of
    quietly showing English-only accuracy as if it applied to all traffic.
    """
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) >= threshold


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # 1. drop rows missing the two fields we can't function without
    df = df.dropna(subset=["customer_message", "agent_response"]).copy()

    # 2. normalize text
    df["customer_message"] = df["customer_message"].apply(normalize_text)
    df["agent_response"] = df["agent_response"].apply(normalize_text)

    # 3. drop exact full-row duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=["thread_id", "customer_message", "agent_response"])

    # 4. drop rows where the message became empty, is too short to carry
    #    intent signal, or is not (mostly) English (see is_mostly_ascii)
    df = df[df["customer_message"].str.len() >= 8]
    df = df[df["customer_message"].apply(is_mostly_ascii)]

    after = len(df)
    print(f"Cleaning: {before} -> {after} rows ({before - after} removed)")
    return df.reset_index(drop=True)


def thread_level_split(df: pd.DataFrame, seed: int, test_frac: float = 0.25):
    """
    Splits by thread_id (not by row) so the same conversation never
    appears in both train and test — avoids the leakage warned about
    in Phase 3 / 11 of the brief.
    """
    thread_ids = df["thread_id"].unique()
    rng = pd.Series(thread_ids).sample(frac=1.0, random_state=seed)
    n_test = int(len(rng) * test_frac)
    test_ids = set(rng.iloc[:n_test])
    train_ids = set(rng.iloc[n_test:])

    train_df = df[df["thread_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["thread_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


def main():
    cfg = load_config()
    raw_path = project_path(cfg["dataset"]["raw_path"])
    df = pd.read_csv(raw_path)

    print("BEFORE (sample):")
    print(df[["customer_message", "agent_response"]].head(2).to_string())

    cleaned = clean_dataframe(df)

    print("\nAFTER (sample):")
    print(cleaned[["customer_message", "agent_response"]].head(2).to_string())

    processed_path = project_path(cfg["dataset"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(processed_path, index=False)
    print(f"\nSaved cleaned dataset -> {processed_path} ({len(cleaned)} rows)")

    train_df, test_df = thread_level_split(cleaned, seed=cfg["random_seed"])
    train_path = project_path(cfg["dataset"]["train_path"])
    test_path = project_path(cfg["dataset"]["test_path"])
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Train: {len(train_df)} rows -> {train_path}")
    print(f"Test:  {len(test_df)} rows -> {test_path}")

    # sanity check: no thread_id overlap
    overlap = set(train_df["thread_id"]) & set(test_df["thread_id"])
    print(f"Thread ID overlap between train/test: {len(overlap)} (should be 0)")


if __name__ == "__main__":
    main()
