"""
Top-level evaluation CLI.

    python evaluate.py --golden_set data/evaluation/golden_set.csv

Runs the full pipeline over every row of the given golden set and
reports intent classification, escalation, and retrieval metrics with
95% bootstrap confidence intervals.
"""
import argparse
import sys

from src.evaluation.run_evaluation import main as run_eval_main


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Hiver Support Agent against a golden set.")
    parser.add_argument("--golden_set", type=str, default=None,
                         help="Path to a golden-set CSV (absolute, or relative to the project root). "
                              "Defaults to the path in config.yaml (data/evaluation/golden_set.csv).")
    args = parser.parse_args()
    run_eval_main(golden_set_path=args.golden_set)


if __name__ == "__main__":
    sys.exit(main())
