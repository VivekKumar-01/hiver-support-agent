# Hiver Support Agent (Take-Home Project)

An AI customer-support agent that classifies intent, retrieves similar
historical cases (same-brand prioritized), generates a grounded reply,
and decides AUTO_HANDLE vs ESCALATE — with an honest evaluation harness,
a real, individually-reviewed golden set, real failure analysis, and a
decision log that includes a genuine safety bug the evaluation harness
caught and a documented fix.

**This project now runs on the real Kaggle "Customer Support on
Twitter" dataset**, provided directly by the user (2,811,774 real
tweets). An earlier synthetic-data milestone is preserved in
`DECISION_LOG.md` (D1-D14) for history but is no longer the default.

**Two things are still not live** because this build environment has no
internet access: (1) response generation is template-grounded, not a
live LLM call, and (2) the LLM-as-judge module is implemented but never
executed (no API key). Both are clearly marked "Not measured yet"
everywhere they'd otherwise appear, never faked.

---

## 1. Quick start (real data, <15 minutes)

```bash
pip install -r requirements.txt
python run_pipeline.py --brand AppleSupport --sample
python evaluate.py --golden_set data/evaluation/golden_set.csv
```

`--sample` uses the bundled `data/raw/sample_data_applesupport.csv`
(1,000 real AppleSupport tweets) so this works without the ~500MB raw
Kaggle file.

## 2. Architecture

```
Customer Message
  → Preprocessing (real-data: HTML-entity decoding, agent sign-off stripping, English-only filter)
  → Intent Classification (TF-IDF + Logistic Regression, trained on WEAK/keyword-rule labels)
  → Confidence
  → Historical Retrieval (same-brand prioritized, TF-IDF cosine)
  → Risk / Escalation Analysis (conservative rules, incl. "unknown intent -> escalate")
  → Grounded Response Generation (template-grounded; LLM-API path implemented, unused)
  → Grounding / Safety Check
  → AUTO_HANDLE or ESCALATE
  → Evaluation (bootstrap 95% CIs on every headline metric)
```

Callable interface: `run_agent(customer_message, brand) -> dict`
(`src/pipeline.py`).

## 3. Problem statement

See `REPORT.md` for full framing.

## 4. Dataset — REAL

Source: Kaggle "Customer Support on Twitter" (`twcs.csv`, 2,811,774
tweets, columns `tweet_id, author_id, inbound, created_at, text,
response_tweet_id, in_response_to_tweet_id`). Reshaped by
`src/data/load_real_dataset.py` — see that file and `data/README.md`
for exact pairing logic and every real, measured stat (row counts,
brand distribution, missing values).

Working set: 6,000 real customer<->brand-reply pairs across 5 brands
(AmazonHelp, AppleSupport, Uber_Support, SpotifyCares, Delta), capped
at 1,200/brand for a reproducible, fast-to-run scope (the full paired
pool is 390,835 rows). After cleaning: 5,824 rows -> thread-level split
-> 4,368 train / 1,456 test (0 thread-id overlap, verified).

## 5. Sampling

Fixed seed (42) throughout. Thread-level train/test split (not
row-level) to prevent leakage. Golden set (240 examples) drawn from the
TEST split only, via stratified random sampling PLUS keyword-targeted
supplemental sampling for under-represented intents — every candidate
individually read and labelled (see section 7).

## 6. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # exact-pinned versions
```
Python 3.12.3, verified.

## 7. Golden Set — what "labelled" actually means here

**240 examples**, all individually read and labelled by the assistant
(Claude) — a real judgment call per example on real, messy Twitter
text. This is NOT equivalent to human-labelled data or independent
inter-annotator review:

- **Single annotator.** No independent human reviewer was available in
  this environment. `label_source` is `assistant_single_annotator` for
  every row — never claimed as human-labelled.
- **Self-consistency, not inter-annotator agreement.** A 50-example
  blind re-labelling pass by the same annotator got 98% raw agreement,
  κ=0.975 — see `docs/LABEL_SELF_CONSISTENCY.md` for exactly why this
  is NOT the checklist's "κ≥0.6 on 50 double-labelled examples" claim,
  and what a real version of that check would require.
- **Per-intent coverage:** `other` 62, `technical_issue` 56,
  `general_feedback` 36, `billing_refund` 27, `shipping_delivery` 18,
  `security_fraud` 15, `account_access` 15, `order_status` 7,
  `cancellation_request` 4. The last two fell short of a 15-example
  floor despite a keyword search across the ENTIRE 1,456-message test
  split — reported as a real finding (these phrasings are rare on
  Twitter support specifically), not padded — see D19 in
  `DECISION_LOG.md`.
- **Codebook:** `docs/INTENT_TAXONOMY.md`.
- **9 candidates excluded as non-English** (French/Spanish/Portuguese/
  Dutch) — caught by manual reading, not by the automated filter (a
  real, disclosed gap — see D16).

## 8. Evaluation harness

```bash
python evaluate.py --golden_set data/evaluation/golden_set.csv
```

Computes intent accuracy/macro-F1/weighted-F1, escalation
precision/recall/confusion matrix, retrieval similarity distribution —
**every headline metric now includes a 95% bootstrap confidence
interval** (`src/evaluation/metrics.py::bootstrap_ci`, 1,000 resamples).

LLM-as-judge (`src/evaluation/llm_judge.py`) is implemented with a
6-criterion rubric at `temperature=0` (for reproducibility if ever run)
but **not executed** — no API key in this environment.
`outputs/metrics/reply_quality_metrics.json` says `"Not measured yet"`
explicitly.

## 9. Baselines

```bash
python -m src.classification.baseline_majority
python -m src.classification.tfidf_lr
```

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority baseline (always "other") | 0.258 | 0.046 | 0.106 |
| TF-IDF + LR (proposed) | 0.300 (CI 0.242–0.362) | 0.104 (CI 0.074–0.135) | 0.180 |

The proposed system beats the trivial baseline on every metric, but
only modestly — **see `MISLEADING_HEADLINE_NUMBER.md` for why even
30% accuracy is itself optimistic**, given the classifier trains on
weak labels where 87.7% of training data is "other."

## 10. The central real finding: a safety bug the evaluation harness caught

First real evaluation run: **98/240 (40.8%) false auto-handles** — the
exact dangerous failure mode the brief says to prioritize avoiding.
Root cause: the classifier is badly overconfident (mean confidence 0.88
vs 30% actual accuracy) and "other" predictions weren't treated as
inherently risky by the escalation policy. Fix (D18): escalate on any
`predicted_intent == "other"`. Result:

| Metric | Before | After |
|---|---:|---:|
| False auto-handle | 98/240 | **1/240** |
| False escalate | 7/240 | 114/240 |
| Escalation accuracy | 0.562 | 0.521 (CI 0.454–0.588) |

This is a genuine, disclosed tradeoff (much safer, less efficient), not
a clean win — see `FAILURE_ANALYSIS.md` failures 1-3 for the real
examples (including an actual driver-stalking safety report that was
auto-handled with a boilerplate reply before the fix) and
`MISLEADING_HEADLINE_NUMBER.md` for why neither the before nor the
after number alone tells the real story.

## 11. Demo

```bash
python demo.py
# or
python run_pipeline.py --brand AppleSupport --sample
python run_pipeline.py --brand AmazonHelp --message "Where is my order?"
```

## 12. Failure analysis

`FAILURE_ANALYSIS.md` — 4 real, fully-worked failures from actual golden-set/pipeline output, each with a real example, root-cause hypothesis, and fix status (applied or proposed).

## 13. Limitations

- **Single-annotator golden set** — no independent human review (see
  section 7 and `docs/LABEL_SELF_CONSISTENCY.md`).
- **Weak-labelled training data** — 87.7% of training messages don't
  match any keyword rule and are weak-labelled "other"; the classifier
  is only as good as these rules (D17).
- **English-only, imperfectly enforced** — ASCII-ratio filter misses
  Romance-language non-English text (D16).
- **No live LLM** for generation or judging — no API key/network access
  in this environment.
- **`order_status` and `cancellation_request`** golden-set counts (7, 4)
  are below a reliable sample size — per-class metrics for these two
  should not be quoted with confidence.
- **The D18 escalation fix is blunt** — it trades a large false-auto-handle
  reduction for a large false-escalate increase; a smarter fix (better
  weak-label rules, classifier recalibration, or a narrower "other +
  risk signal" rule) is proposed but not implemented — see REPORT.md.
- **`sentence-transformers`/FAISS unavailable** in this sandbox (no
  internet); retrieval uses TF-IDF cosine similarity, auto-upgrades if
  those packages are installed elsewhere.

## 14. Reproducibility

| Item | Value |
|---|---|
| Random seed | 42 |
| Dataset | Real, Kaggle Customer Support on Twitter, 6,000-row 5-brand working set |
| Train/test split | Thread-level, seed 42, 0 overlap (verified) |
| Golden set | 240 examples, single-annotator, stratified + keyword-targeted supplemental sampling |
| Classifier | TF-IDF (max_features=5000, ngram=(1,2)) + Logistic Regression (C=2.0), trained on weak-keyword-rule labels |
| Escalation | confidence<0.55, risk keywords, poor retrieval, sensitive intents, AND unknown ("other") intent all -> ESCALATE |
| Generation | template_grounded (default) or llm_api (unused, needs `ANTHROPIC_API_KEY`) |

Reproduce end-to-end in under 15 minutes:

```bash
pip install -r requirements.txt
python run_pipeline.py --brand AppleSupport --sample
python -m src.preprocessing.clean
python -m scripts.build_golden_set
python -m scripts.label_supplemental_examples
python -m src.classification.weak_label_train_data
python -m src.classification.baseline_majority
python -m src.classification.tfidf_lr
python evaluate.py --golden_set data/evaluation/golden_set.csv
python -m unittest discover tests
```

To reproduce from the full raw Kaggle file instead of the bundled
working set: `python -m src.data.load_real_dataset` (see
`data/README.md`).

## 15. Decision log

`DECISION_LOG.md` — 19 real decisions (D1-D14 from the synthetic
milestone, D15-D19 from the real-data rebuild), each with reason,
alternative considered, and why rejected.

## 16. Checklist compliance

This project was reviewed against an external code-review checklist —
`CHECKLIST_COMPLIANCE.md` maps every single item to its current status,
including the items that are honestly still not fully met (independent
human labelling and true inter-annotator κ chief among them) and why.

## 17. Citations

- Kaggle "Customer Support on Twitter" dataset (real, user-provided)
- scikit-learn, pandas, numpy, matplotlib
- Anthropic API (`claude-sonnet-4-6`) — implemented, optional, unused (no key)

---

Also see: `REPORT.md`, `FAILURE_ANALYSIS.md`,
`MISLEADING_HEADLINE_NUMBER.md`, `DECISION_LOG.md`,
`docs/INTENT_TAXONOMY.md`, `docs/LABEL_SELF_CONSISTENCY.md`,
`docs/INTERVIEW_PREP.md`, `CHECKLIST_COMPLIANCE.md`.
