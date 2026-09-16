"""
Phase 2 — Dataset inspection.

Never assume columns/shape — always print the actual thing. Run this
before writing any cleaning code.

Run:
    python -m src.data.inspect_dataset
"""
import pandas as pd

from src.utils.config import load_config, project_path


def main():
    cfg = load_config()
    path = project_path(cfg["dataset"]["raw_path"])
    df = pd.read_csv(path)

    print("=" * 60)
    print(f"FILE: {path}")
    print(f"SHAPE: {df.shape[0]} rows x {df.shape[1]} columns")
    print("=" * 60)

    print("\nCOLUMNS AND DTYPES:")
    print(df.dtypes)

    print("\nMISSING VALUES PER COLUMN:")
    print(df.isna().sum())

    print("\nFULL-ROW DUPLICATES:", df.duplicated().sum())
    print("DUPLICATE customer_message values:", df["customer_message"].duplicated().sum())
    print("DUPLICATE thread_id values:", df["thread_id"].duplicated().sum())

    print("\nUNIQUE BRANDS:", df["brand"].nunique())
    print(df["brand"].value_counts())

    print("\nINTENT COLUMN PRESENT:", "intent" in df.columns)
    if "intent" in df.columns:
        print(df["intent"].value_counts())

    print("\nSAMPLE ROWS:")
    print(df.sample(5, random_state=cfg["random_seed"]).to_string())


if __name__ == "__main__":
    main()
