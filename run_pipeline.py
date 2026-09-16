"""
Top-level pipeline runner CLI.

    python run_pipeline.py --brand AppleSupport --sample

`--sample` uses the fast, dedicated AppleSupport sample_data.csv
(1,000 real rows, see data/README.md) so the full flow — build, run,
and see results — completes in well under 15 minutes on a normal laptop
without needing the ~500MB raw Kaggle file. Without `--sample`, it runs
against whatever dataset is configured in config/config.yaml
(data.source: "real", the full 5-brand 6,000-row working set).

Examples:
    python run_pipeline.py --brand AppleSupport --sample
    python run_pipeline.py --brand AmazonHelp --message "Where is my order?"
    python run_pipeline.py --brand AppleSupport --sample --steps all
"""
import argparse
import json
import sys

import pandas as pd

from src.pipeline import run_agent
from src.utils.config import load_config, project_path


def run_step(module_name: str):
    import importlib
    print(f"\n{'=' * 60}\nSTEP: {module_name}\n{'=' * 60}")
    mod = importlib.import_module(module_name)
    mod.main()


def main():
    parser = argparse.ArgumentParser(description="Run the Hiver Support Agent pipeline.")
    parser.add_argument("--brand", type=str, default="AppleSupport",
                         help="Brand to run a demo message against, e.g. AppleSupport, AmazonHelp, "
                              "Uber_Support, SpotifyCares, Delta")
    parser.add_argument("--message", type=str, default=None,
                         help="Customer message to run through the pipeline. If omitted with --sample, "
                              "a real example is pulled from the AppleSupport sample data.")
    parser.add_argument("--sample", action="store_true",
                         help="Use the small, fast AppleSupport sample_data.csv instead of the full "
                              "5-brand working dataset — reproduces in well under 15 minutes.")
    parser.add_argument("--steps", choices=["demo", "all"], default="demo",
                         help="'demo' (default): just run one message through run_agent(). "
                              "'all': also (re)run data prep + training end-to-end first.")
    args = parser.parse_args()

    cfg = load_config()

    if args.steps == "all":
        run_step("src.preprocessing.clean")
        run_step("scripts.build_golden_set")
        run_step("src.classification.weak_label_train_data")
        run_step("src.classification.baseline_majority")
        run_step("src.classification.tfidf_lr")

    message = args.message
    if message is None:
        if args.sample:
            sample_path = project_path(cfg["dataset"]["sample_path"])
            df = pd.read_csv(sample_path)
            message = df.iloc[0]["customer_message"]
            print(f"(--sample: no --message given, using a real example)\n")
        else:
            message = "My refund hasn't arrived yet, it's been over a week."
            print(f"(no --message given, using a default example)\n")

    print(f"Brand: {args.brand}")
    print(f"Message: {message}\n")

    result = run_agent(message, args.brand)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
