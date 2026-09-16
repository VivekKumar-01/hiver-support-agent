# Decision Log

Real engineering decisions actually made while building this project,
in the format the brief requests.

---

**D1 — Use a synthetic dataset instead of the real Kaggle dump**
Reason: The build environment has no internet access, so the Kaggle
"Customer Support on Twitter" dataset (multi-GB download) could not be
fetched, and no cached copy existed locally.
Alternative: Skip the dataset entirely and hand-write 20-30 examples.
Why rejected: Too small to support a train/test split, a meaningful
golden set, or believable baseline comparisons.
Impact: Every number in this repo is real and measured, but measured on
synthetic, template-generated data — see README > Limitations for what
that does and doesn't prove, and exactly how to swap in the real data.

**D2 — Template-grounded generation instead of a live LLM call**
Reason: No LLM API key is available in this sandbox and it has no
network access to reach one anyway.
Alternative: Fabricate example LLM outputs and label them as real.
Why rejected: Directly violates the brief's non-negotiable rule against
fabricating results.
Impact: Generated replies are less fluent than an LLM would produce,
but are provably grounded (they literally cannot state an order number
that wasn't in the retrieved evidence). The `llm_api` code path is
implemented in `src/generation/generator.py` and `src/evaluation/llm_judge.py`
and activates automatically if you add an `ANTHROPIC_API_KEY`.

**D3 — Fixed, hand-designed intent taxonomy rather than discovered via clustering**
Reason: Because the dataset itself is synthetic (D1), there is no real
message distribution to cluster — the taxonomy IS the generation schema.
Alternative: Run k-means over TF-IDF vectors of the synthetic messages
and call the resulting clusters "discovered."
Why rejected: Circular — clustering data generated from 8 known
templates would just rediscover the same 8 templates while looking like
independent analysis. That would misrepresent the process to a reader.
Impact: `docs/INTENT_TAXONOMY.md` documents the taxonomy AND states
plainly it was designed rather than mined, with instructions for how to
do real clustering-based derivation once real data is available.

**D4 — Thread-level train/test split, not row-level**
Reason: Multiple rows can share a `thread_id` (a conversation). Splitting
by row risks the same conversation leaking into both train and test.
Alternative: Simple random row split (faster to write).
Why rejected: Classic leakage source explicitly warned about in the brief.
Impact: Verified zero thread-id overlap between train/test (see test
`test_thread_level_split_has_no_overlap` and the cleaning script output).

**D5 — TF-IDF + Logistic Regression as the production classifier (skip Phase 9 semantic classifier)**
Reason: Under a 6-hour budget, retrieval + generation + escalation +
evaluation are worth more than a third classification approach. Baseline
2 already reaches 1.00 macro F1 on the golden set (see D7 for why that
number is misleading), so a third classifier would not have added
decision-relevant evidence.
Alternative: Also implement a sentence-embedding nearest-centroid classifier.
Why rejected: Priority order in the brief (Phase 9 is explicitly
lower-priority than Phases 10-15) plus the time budget.
Impact: `src/classification/tfidf_lr.py` doubles as both "Baseline 2"
and the live pipeline's classifier.

**D6 — TF-IDF cosine similarity for retrieval instead of sentence-transformers + FAISS**
Reason: `sentence-transformers` and `faiss` could not be installed
(no internet in this sandbox).
Alternative: Skip retrieval entirely, or hardcode example retrieved cases.
Why rejected: Retrieval is core to the assignment (Phase 10) and the
brief explicitly forbids faking retrieved evidence.
Impact: `src/retrieval/retriever.py` auto-upgrades to
sentence-transformers if it detects the package installed, so the exact
same code runs better retrieval on a machine with internet — no
rewrite needed.

**D7 — Report the golden-set 1.00 macro F1 as misleading, not as a headline success**
Reason: The golden set is sampled from the same template pool used to
train the classifier, so near-perfect separability is expected, not
evidence of real-world robustness.
Alternative: Report it at face value as "100% accurate."
Why rejected: Would be a textbook example of the misleading-headline-number
trap Phase 28 explicitly asks us to watch for.
Impact: Built `scripts/stress_test.py` — a small, hand-written,
out-of-distribution example set — which found real failures (6/8 intent,
7/8 action correct). See FAILURE_ANALYSIS.md and MISLEADING_HEADLINE_NUMBER.md.

**D8 — Golden set `expected_action` labels come from an explicit rule, documented as such (not "human-labelled")**
Reason: True manual, one-by-one human review of 180 examples wasn't
feasible in the time budget, and mislabeling a rule-based pass as
"human-labelled" would violate the brief's labelling-honesty requirement.
Alternative: Call it human-labelled anyway since a human wrote the rule.
Why rejected: The rule, not a human, made each per-row call — that's a
meaningfully different (and weaker) form of ground truth and needs to be
named accurately.
Impact: `golden_set.csv` has a `label_source` column set to
`synthetic_ground_truth` for every row, and the escalation-metric
"1.00 accuracy" result is flagged as circular in
MISLEADING_HEADLINE_NUMBER.md (see D9 below for why).

**D9 — Conservative, rule-based escalation thresholds instead of a tuned classifier**
Reason: With only 6 hours and a synthetic dataset, tuning thresholds
against synthetic "ground truth" that was generated by a similar rule
(D8) would be circular, not genuine tuning.
Alternative: Fit a small logistic model on confidence + similarity +
risk-keyword-count to predict escalation.
Why rejected: Would launder the same circularity through a model
instead of removing it, while looking more "rigorous."
Impact: Thresholds in `config.yaml > escalation` are simple, documented,
conservative defaults (e.g. confidence < 0.55 → escalate), explicitly
NOT claimed to be tuned on held-out data. See MISLEADING_HEADLINE_NUMBER.md.

**D10 — Prioritize false-auto-handle avoidance over auto-handle rate**
Reason: An auto-sent reply that shouldn't have been auto-sent (e.g. to a
fraud case) is worse than an unnecessary human handoff.
Alternative: Optimize thresholds to maximize the AUTO_HANDLE rate.
Why rejected: Directly contradicts the brief's stated escalation
philosophy (Phase 26: "focus particularly on false auto-handling").
Impact: Any risk keyword, low confidence, poor retrieval, or
financially-sensitive intent unconditionally escalates — see
`src/escalation/policy.py`.

**D11 — Grounding check can override an AUTO_HANDLE decision after generation**
Reason: Even a template-grounded reply could technically restate a
number found elsewhere in the retrieved set in a misleading way; a
safety net after generation is cheap insurance.
Alternative: Trust the escalation decision made before generation and
never re-check.
Why rejected: The brief explicitly asks for a post-generation grounding
check (Phase 13), independent of the pre-generation escalation policy.
Impact: `src/pipeline.py` calls `check_grounding()` after generating a
reply and force-escalates if it finds an unsupported claim.

**D12 — Skip Streamlit, ship a CLI**
Reason: Per the brief's explicit priority order (Phase 39, priority 15
of 15), UI is the lowest priority and should never be built at the
expense of evaluation/documentation.
Alternative: Build a minimal Streamlit UI anyway for demo polish.
Why rejected: Time budget; CLI fully exercises the real pipeline and is
sufficient for the 5-minute demo script.
Impact: `demo.py` provides an interactive CLI; no `app.py` was built.

**D13 — Macro F1 as the primary classification metric, not accuracy**
Reason: Golden-set intents are not perfectly balanced (18-25 examples
per class); macro F1 weights every class equally so a model can't hide
poor performance on rarer intents behind strong performance on common ones.
Alternative: Report accuracy only.
Why rejected: Accuracy can look strong while a minority class (e.g.
`security_fraud`, the highest-stakes intent) is being predicted poorly —
exactly the failure mode Phase 28 warns about.
Impact: All classification reporting includes accuracy, macro F1, AND
weighted F1 side by side (see `outputs/metrics/*.json`).

**D14 — Use unittest instead of pytest for tests actually executed here**
Reason: `pytest` could not be installed in this sandbox (no internet).
Alternative: Skip automated tests.
Why rejected: Phase 31 explicitly requires tests; skipping them is a
much bigger loss than using a stdlib substitute.
Impact: `tests/test_pipeline.py` uses `unittest.TestCase`, which is
pytest-compatible — it runs unchanged under `pytest tests/` on a machine
that has it (see `requirements.txt`), and was verified to pass with
`python -m unittest discover tests` here (11/11 passing).

---

## Real-data update (dataset provided by the user)

The decisions below were made after the user provided the real Kaggle
dataset, superseding the synthetic-data-only version of this project.
D1-D14 above are kept for history (they explain the synthetic
milestone); D15+ describe the real-data rebuild.

**D15 — Adopt the real dataset immediately, keep synthetic generator only for reference**
Reason: the user provided the actual `twcs.csv` (2,811,774 real tweets),
removing the reason for D1's synthetic substitution.
Alternative: keep both pipelines fully interchangeable via a config flag.
Why rejected (partially): `config.yaml > dataset.source` still has a
"synthetic" option and the generator script still exists, but every
default path, script, and documented number in this project now points
at real data — maintaining true parity would double the validation
burden for no benefit once real data is available.
Impact: `src/data/load_real_dataset.py` is now the primary ingestion
path; `data/README.md` documents both.

**D16 — English-only scope, enforced by an ASCII-ratio filter plus manual exclusion**
Reason: the taxonomy and templates are English; non-English messages
would be silently mis-classified rather than legitimately handled.
Alternative: attempt multilingual support.
Why rejected: out of scope for the time budget, and worse, a naive
ASCII-ratio filter alone is NOT sufficient — Romance-language text
(French/Spanish/Portuguese/Dutch) is still mostly ASCII and slipped
through. Caught only because every golden-set candidate was actually
read (9 excluded on sight — see `scripts/build_golden_set.py`).
Impact: this is a disclosed, real scope limit, not a silent gap — see
MISLEADING_HEADLINE_NUMBER.md. A production system needs real language
detection (e.g. `langdetect` or `fasttext`), not an ASCII heuristic.

**D17 — Weak/keyword-rule labels to train the classifier at scale; golden set stays individually reviewed**
Reason: the real dataset has NO intent column anywhere. Manually
labelling all 4,368 training messages was infeasible in the time
budget, and would just be a bigger version of the single-annotator
limitation already disclosed for the golden set.
Alternative: train only on the 240 golden-set examples (split into a
smaller train/test), or skip a learned classifier and use keyword rules
directly in production.
Why rejected: 240 examples is too few to train a usable TF-IDF+LR model
well, and the golden set must stay 100% held-out from training for the
evaluation to mean anything. A rules-only production system was
considered but rejected because it would be indistinguishable from just
shipping `weak_label_train_data.py`'s rules as "the system," which
provides no genuine ML baseline to evaluate.
Impact: `src/classification/weak_label_train_data.py` produces
`train_weak_labelled.csv`, explicitly labelled `label_type:
weak_keyword_rule`, kept fully separate from and never merged with the
golden set. This is disclosed as a real limitation: the classifier can
only be as good as the weak rules it was trained from, and the eval
results directly show the cost of that (macro F1 0.104 vs. golden-set
macro F1 1.00 on synthetic data — see MISLEADING_HEADLINE_NUMBER.md).

**D18 — Escalate on `predicted_intent == "other"` (added AFTER seeing real evaluation results)**
Reason: the first real evaluation run found 98/240 (40.8%) false
auto-handles — the dangerous error type the brief explicitly says to
prioritize avoiding (Phase 26) — and every single one was triggered via
the `default_auto_handle` rule on a message the classifier had (often
confidently, but wrongly) predicted as `other`.
Alternative: try to fix this by recalibrating classifier confidence
(e.g. Platt scaling) instead of changing the escalation policy.
Why rejected (for now): recalibration is the more "correct" long-term
fix (see REPORT.md > One More Week) but requires more held-out data
than is available and more time than remained in the budget. Adding an
explicit "unknown intent -> escalate" rule is a one-line, immediately
verifiable, conservative fix consistent with the whole policy's design
philosophy (D10).
Impact: false auto-handles dropped from 98/240 to 1/240 — but false
escalates rose from 7/240 to 114/240 and overall escalation accuracy
fell from 0.562 to 0.521. This is a real, disclosed precision/recall
tradeoff, not a free win — see MISLEADING_HEADLINE_NUMBER.md and
FAILURE_ANALYSIS.md for the full before/after and why the tradeoff is
still the right call given the brief's own stated priority.

**D19 — Supplement rare intents with keyword-targeted (not random) real examples, and disclose the two that still fell short**
Reason: the initial 220-message stratified random sample left
`cancellation_request` (1 example), `order_status` (7),
`security_fraud` (6), `account_access` (7), and `shipping_delivery` (9)
below the 15-example floor.
Alternative: pad these classes with borderline/loosely-related examples
from the random sample to hit 15 without additional search.
Why rejected: that would quietly lower label quality to hit a number —
exactly the kind of thing this project's evaluation methodology exists
to catch, not commit.
Impact: `security_fraud`, `account_access`, and `shipping_delivery`
reached >=15 via genuine keyword-targeted search + individual review
(several matches were rejected as false positives — see
`scripts/label_supplemental_examples.py` docstring). `order_status` (7)
and `cancellation_request` (4) did NOT reach 15 despite a broadened
search across the entire 1,456-message test split — reported as a real
finding about this dataset (neutral status questions and explicit
cancellation requests are rare on Twitter support specifically, likely
because those interactions happen via app/website instead) rather than
padded to hit the target.
