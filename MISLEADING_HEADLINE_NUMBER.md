# What is misleading about my headline number?

This document was originally written for the synthetic-data milestone
of this project, where the "misleading number" was a suspiciously
perfect 1.00 F1 caused by synthetic template artifacts. With the real
dataset, the story flipped: **the real headline numbers are much worse,
and the most important finding isn't a metric at all — it's a policy
bug the evaluation harness caught.**

## The headline numbers

| Metric | Value |
|---|---:|
| Intent accuracy (proposed system, golden set) | 0.300 (95% CI 0.242–0.362) |
| Intent macro F1 | 0.104 (95% CI 0.074–0.135) |
| Escalation accuracy | 0.521 (95% CI 0.454–0.588) |
| False auto-handle | 1 / 240 |
| False escalate | 114 / 240 |

A naive read: "30% intent accuracy, terrible" or "only 1 dangerous
false auto-handle, great." **Both readings are misleading in specific,
mechanistic ways** — not just "the data is noisy":

## 1. The classifier is trained on weak (keyword-rule) labels, and 87.7% of training data falls into "other"

The real dataset has no intent labels at all. To train a classifier at
scale, `src/classification/weak_label_train_data.py` assigns labels via
simple keyword rules. Running it on the 4,368 real training messages
found that **3,830 of them (87.7%) match none of the 8 substantive
keyword rules** and fall into "other." This means the classifier's
"ground truth" during training is itself only ~12% reliably labelled —
its 30% golden-set accuracy is a measurement of a model trained on
mostly-weak labels, not a clean supervised classifier. The mechanism:
narrow keyword rules (e.g. `billing_refund` needs literally "refund" or
"charged") systematically miss the linguistic variety of real complaints
("that's not fair I was billed twice" contains none of the trigger
words used). Fixing this requires either a larger/smarter weak-labelling
ruleset or (better) a genuinely labelled training set.

## 2. The classifier is badly overconfident on real text — this is the mechanism behind the danger, not just a side note

Mean predicted confidence across the golden set was ~0.88 while actual
accuracy was 30%. Concretely: of the 98 original false auto-handles
(before the D18 fix), every single one had confidence between 0.83 and
0.98 — the model was MOST confident exactly when it was wrong. This is
a known pathology of linear models (logistic regression on sparse
TF-IDF) trained on an imbalanced, weakly-labelled corpus: the "other"
class dominates training, so the model learns to assign it high
confidence by default, including on real security_fraud/billing_refund
messages it has never seen phrased that way.

## 3. The escalation policy had a real, evaluation-caught safety bug — and fixing it traded one error type for another

Before D18: false auto-handle = 98/240 (40.8%) — the exact dangerous
failure mode Phase 26 of the brief warns about, caused by "other"
predictions never being treated as inherently risky. After adding "an
unclassified intent escalates" (D18): false auto-handle dropped to
1/240, but false escalate rose from 7/240 to 114/240, and *overall*
escalation accuracy fell (0.562 -> 0.521). Reporting only the "after"
accuracy number (0.521) without this context would hide that it
represents a large, deliberate safety improvement, not model quality;
reporting only the "before" number (0.562) would hide a serious safety
bug entirely. **Neither single number tells the real story — the
before/after pair does.**

## 4. Golden-set label provenance and quantity, per class

`account_access`, `security_fraud`, and `shipping_delivery` needed
keyword-targeted (not purely random) supplemental sampling to reach the
15-example floor — random sampling alone found almost none. Two
intents, `order_status` (7) and `cancellation_request` (4), never
reached 15 despite a search across the ENTIRE 1,456-message test split.
This is a real finding about the dataset (neutral status checks and
explicit cancellations are rare on Twitter support), reported honestly
rather than padded — but it also means any per-class metric for those
two intents is statistically unreliable (n<15) and shouldn't be quoted
as if it were.

## 5. Self-consistency (κ=0.975) is not inter-annotator agreement

See `docs/LABEL_SELF_CONSISTENCY.md` in full. Short version: one
annotator (the assistant) re-labelling its own prior work agrees with
itself far more than two independent humans would agree with each
other. This number measures rule self-consistency, not label
correctness or real-world reliability.

## 6. "other" is 25.8% of the golden set, by design, not suppressed

Unlike a system that might quietly avoid an "other" bucket to look more
decisive, this taxonomy uses "other" honestly, and it is the single
largest class in the real evaluation set (62/240). That's a genuine,
disclosed finding about how much real Twitter support traffic doesn't
fit a small, pre-defined intent taxonomy — not evidence of a
data-quality problem to be papered over.

## What should be reported instead of any single headline number

- Golden-set metrics with 95% CIs (now included in
  `outputs/metrics/*.json` via bootstrap resampling — see
  `src/evaluation/metrics.py::bootstrap_ci`), not point estimates alone.
- The false-auto-handle vs. false-escalate **pair**, before and after
  D18, together — never one without the other.
- Per-class metrics flagged as unreliable below n=15 (`order_status`,
  `cancellation_request`).
- The weak-label training distribution (87.7% "other") alongside any
  classifier accuracy number, since it explains most of the gap between
  this project's real 30% accuracy and the earlier synthetic project's
  fabricated-looking 100%.
