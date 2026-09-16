# Data

## Real dataset (current)

`raw/real_customer_support.csv` is built from the REAL Kaggle "Customer
Support on Twitter" export (`twcs.csv`, ~2.81M tweets), reshaped by
`src/data/load_real_dataset.py` — see that file's docstring for the
exact pairing logic (agent reply <-> the customer tweet it responds to).
The raw ~500MB source file itself is NOT shipped in this repo (too
large); see "Reproducing from the raw file" below.

| File | Produced by | Rows (last run) |
|---|---|---|
| `raw/real_customer_support.csv` | `src/data/load_real_dataset.py` | 6,000 (1,200 x 5 brands) |
| `raw/sample_data_applesupport.csv` | `src/data/load_real_dataset.py` (standalone AppleSupport-only pull) | 1,000 |
| `processed/processed.csv` | `src/preprocessing/clean.py` | 5,824 |
| `processed/train.csv` | `src/preprocessing/clean.py` (thread-level split) | 4,368 |
| `processed/train_weak_labelled.csv` | `src/classification/weak_label_train_data.py` | 4,368 |
| `processed/test.csv` | `src/preprocessing/clean.py` (thread-level split) | 1,456 |
| `evaluation/golden_set.csv` | `scripts/build_golden_set.py` + `scripts/label_supplemental_examples.py` | 240 |

### `sample_data_applesupport.csv` — what it is and isn't

This is a **standalone, independently-drawn** 1,000-row AppleSupport-only
pull from the full ~105,739 available real AppleSupport pairs (not a
subset of `real_customer_support.csv`). It exists so `run_pipeline.py
--brand AppleSupport --sample` runs fast without needing the full
~500MB raw file. Because it's drawn independently, **it can overlap**
with `real_customer_support.csv` / `train.csv` / `test.csv` — that's
fine for its purpose (a fast demo/quick-start message source) but it is
**never used for scored evaluation**. All real metrics in this project
come from `golden_set.csv`, which is drawn only from `test.csv` and is
leakage-checked against `train.csv` at the thread-id level (see
`tests/test_pipeline.py::test_thread_level_split_has_no_overlap`).

### `train_weak_labelled.csv` — read before trusting training numbers

The raw dataset has no intent column at all. `train.csv` (used for
retrieval history) has no labels; `train_weak_labelled.csv` (used to
train the classifier) has **keyword-rule-assigned labels**, explicitly
lower-quality than the golden set's individually-reviewed labels — see
`src/classification/weak_label_train_data.py` docstring and
`DECISION_LOG.md` D17.

### Reproducing from the raw file

1. Download `customer-support-on-twitter` from Kaggle.
2. Update `config/config.yaml > dataset.real.source_file` to point at
   your local `twcs.csv`.
3. Run `python -m src.data.load_real_dataset` (add `--brands` /
   `--max_per_brand` to change scope).

## Superseded: synthetic dataset

The earlier milestone of this project (before real data was provided)
used a synthetic generator (`src/data/generate_synthetic_dataset.py`),
kept in the repo for reference but no longer the default — see
`DECISION_LOG.md` D1.

Raw/processed CSVs are excluded from git via `.gitignore` and regenerated
with the commands above; only this README and `golden_set.csv` are meant
to be committed as-is.
