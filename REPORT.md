# Final Report — Hiver Support Agent (Real Data)

## 1. Problem framing

Customer support teams get a high volume of repetitive-but-varied
messages. The task: build a system that reads an inbound message,
figures out what the customer wants, finds relevant precedent, drafts a
grounded reply, and — critically — decides whether it's safe to send
that reply automatically or whether a human should handle it. The hard
part isn't generating plausible text; it's knowing when *not* to trust
the system's own output — which turned out to be the central, literal
finding of this project (see §10).

## 2. What "good" means here

Good is **not** "the classifier gets high accuracy" — on real data it
doesn't, and that's disclosed rather than hidden. Good is: the system
rarely auto-sends something wrong or unsafe, the evaluation harness
actually finds real problems instead of rubber-stamping the system, and
every number in this report is something that was actually measured
(including the ones that look bad).

## 3. Architecture

```
Customer Message → Preprocessing → Intent Classification (TF-IDF + LR,
trained on weak/keyword-rule labels) → Confidence → Historical Retrieval
(same-brand prioritized) → Risk/Escalation Analysis (conservative rules)
→ Grounded Response Generation → Grounding/Safety Check → AUTO_HANDLE or
ESCALATE → Evaluation (with bootstrap confidence intervals)
```

Single callable interface: `run_agent(customer_message, brand)`.

## 4. What we chose not to build

No Docker/Kubernetes/cloud deployment, no distributed vector store, no
fine-tuned/large generative model, no Streamlit UI, no true multi-intent
representation, no second classification approach beyond TF-IDF+LR, and
— new for the real-data phase — no attempt at multilingual support or
confidence recalibration (both flagged as real gaps, not silently
skipped — see §12, §13).

## 5. Dataset and sampling — REAL

The user provided the actual Kaggle "Customer Support on Twitter"
dataset (2,811,774 real tweets). `src/data/load_real_dataset.py` pairs
each brand agent's reply with the customer tweet it responded to,
across 5 brands (AmazonHelp, AppleSupport, Uber_Support, SpotifyCares,
Delta), capped at 1,200 pairs/brand for a fast, reproducible 6,000-row
working set (full available pool: 390,835 pairs). After cleaning
(HTML-entity decoding, agent sign-off stripping, English-only
filtering): 5,824 rows. Thread-level train/test split (seed 42): 4,368
train / 1,456 test, 0 thread-id overlap (verified by test).

## 6. Baselines

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority baseline (always "other") | 0.258 | 0.046 | 0.106 |
| TF-IDF + LR (proposed) | 0.300 (CI 0.242–0.362) | 0.104 (CI 0.074–0.135) | 0.180 |

The proposed system beats the trivial baseline on all three metrics,
but only modestly. §11 explains the mechanism behind why 30% accuracy
is itself an optimistic ceiling for this classifier design.

## 7. Proposed system

TF-IDF + Logistic Regression trained on **weak, keyword-rule-assigned
labels** (`src/classification/weak_label_train_data.py`) — the real
dataset has no intent column at all, so training labels had to be
bootstrapped rather than human-provided (D17). Retrieval uses TF-IDF
cosine similarity with a same-brand boost. Escalation is a conservative
rule stack, with one rule added mid-project after real evaluation
results demanded it (D18, see §10).

## 8. Evaluation

- **Golden set (n=240):** individually read and labelled by the
  assistant (single annotator — see README §7 for exactly what that
  does and doesn't mean).
- **Stress test (n=8, hand-written):** 1/8 intents, 6/8 escalation
  actions correct — consistent with, not contradicting, the golden-set
  picture (unlike the synthetic milestone, real data didn't need a
  separate stress test to surface real failures).
- **LLM-as-judge:** implemented, 6-criterion rubric, temperature=0 for
  reproducibility — **not executed**, no API key in this environment.
- **Bootstrap 95% confidence intervals** on every headline metric
  (`src/evaluation/metrics.py::bootstrap_ci`, 1,000 resamples) — added
  specifically because a bare point estimate with no uncertainty range
  was flagged as a red flag in an external review.

## 9. Results

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority baseline | 0.258 | 0.046 | 0.106 |
| TF-IDF + LR (golden set, n=240) | 0.300 (CI 0.242–0.362) | 0.104 (CI 0.074–0.135) | 0.180 |
| TF-IDF + LR (stress test, n=8) | 0.125 | — | — |

| Metric | Before D18 fix | After D18 fix |
|---|---:|---:|
| Escalation accuracy | 0.562 | 0.521 (CI 0.454–0.588) |
| False auto-handle (dangerous) | 98 / 240 | **1 / 240** |
| False escalate (wasteful) | 7 / 240 | 114 / 240 |

| Metric | Result |
|---|---:|
| Reply quality (LLM judge) | Not measured yet |
| Mean top-1 retrieval similarity | 0.322 |
| Self-consistency (NOT inter-annotator agreement) | κ=0.975, n=50 |

## 10. The central finding: a real safety bug, caught and fixed by the evaluation harness itself

The first full evaluation run found 98/240 (40.8%) false auto-handles —
the exact dangerous failure mode the brief prioritizes avoiding. Every
one was triggered by the classifier predicting `other` with spuriously
high confidence (mean 0.88) on a message that was actually
`security_fraud`, `billing_refund`, or `account_access`. The single most
serious real example: a genuine driver-stalking safety report (with a
license plate number) was classified `other` and auto-handled with a
generic templated reply. Adding one rule — escalate on any
`predicted_intent == "other"` (D18) — dropped false auto-handles to
1/240, at the cost of false escalates rising from 7 to 114 and overall
escalation accuracy falling from 0.562 to 0.521. This is reported as a
genuine, disclosed tradeoff, not a clean win — see
`MISLEADING_HEADLINE_NUMBER.md` and `FAILURE_ANALYSIS.md` failures 1-3
for the full mechanism and the real examples.

## 11. Failure analysis

Full detail in `FAILURE_ANALYSIS.md` (4 real failures, each with an
actual example, root cause, and fix status). Summary: the training
data's weak keyword-rule labels miss the linguistic diversity of real
complaints (e.g. neither "followed... plate number" nor "spoken with
customer service twice, not helped" contain any of the rule keywords
for their true intents), the resulting classifier is badly
overconfident, and the D18 escalation fix — while it closes the
dangerous gap — is blunt enough to over-escalate harmless messages that
happen to also land in "other."

## 12. Misleading headline number

Full detail in `MISLEADING_HEADLINE_NUMBER.md`. Short version: neither
"30% accuracy" nor "1/240 false auto-handles" is meaningful without its
mechanism (87.7% weak-labelled "other" training data; a before/after
policy comparison) attached. This is the opposite problem from the
earlier synthetic milestone (there, numbers were misleadingly perfect;
here, a naive reading of the "after" number alone would hide how much
worse things were before a specific, documented fix).

## 13. Limitations

- Golden set is single-annotator (assistant), not independently
  human-reviewed; self-consistency (κ=0.975) is not inter-annotator
  agreement (`docs/LABEL_SELF_CONSISTENCY.md`).
- Training labels are weak/keyword-rule-based; 87.7% of real training
  messages don't match any rule and default to "other."
- `order_status` (7) and `cancellation_request` (4) golden-set counts
  are below a reliable sample size despite genuine targeted search.
- English-only scope, imperfectly enforced (ASCII-ratio filter misses
  Romance-language non-English text).
- No live LLM for generation or judging.
- The D18 escalation fix is a blunt instrument, not a calibrated
  solution — see §14.

## 14. One more week

1. **Recalibrate classifier confidence** (e.g. Platt scaling / temperature
   scaling) so confidence actually tracks accuracy — the single highest-
   leverage fix, since most of §10's danger stemmed from overconfidence,
   not the underlying prediction errors themselves.
2. **Expand weak-label keyword rules**, especially for physical-safety
   language (not just financial/account-compromise terms) — directly
   motivated by Failure 1.
3. **Get a second human labeler** for the golden set (or at least the
   50-example subset) to get a real inter-annotator κ, replacing the
   self-consistency substitute.
4. **Add a narrower "other + risk signal" escalation rule** to reduce
   the 114 false escalates from D18 without reopening the false-auto-
   handle gap — e.g. only force-escalate "other" predictions below a
   raised confidence threshold, or combine with a lightweight
   sentiment/risk heuristic.
5. **Real language detection** (`langdetect`/`fasttext`) instead of the
   ASCII-ratio heuristic.
6. Get an LLM API key working end-to-end for live generation and
   LLM-as-judge, then measure real reply-quality and human-vs-LLM
   agreement.

## 15. Conclusion

The architecture works end-to-end on real data, and — more importantly
than any single metric — the evaluation harness did exactly what an
evaluation harness is for: it found a real, dangerous safety bug (40.8%
false auto-handle rate) that would not have been visible from the
system's design alone, and the fix that followed is documented with its
real tradeoffs rather than presented as a clean win. The honest
headline numbers here (30% intent accuracy, 1/240 false auto-handles
after a costly-but-necessary fix) are far less impressive-looking than
the synthetic milestone's 100% accuracy — and that gap is itself the
most important result in this report.
