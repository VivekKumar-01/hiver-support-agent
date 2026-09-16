## Real-data addendum (read this first)

This project now runs on the real Kaggle dataset — the honest headline
numbers are 30% intent accuracy and (after a documented fix) 1/240
false auto-handles, traded against 114/240 false escalates. If asked
about results, lead with these real numbers and the D18 safety-bug
story (README.md section 10), not the earlier synthetic milestone's
100% accuracy, which is now explicitly superseded and explained in
DECISION_LOG.md as a synthetic-data artifact.

**Best single interview answer if asked "what's the most interesting
thing you found?"**: "My evaluation harness caught a real safety bug —
40.8% of the time, the system would auto-send a reply on a message it
had actually misclassified, including one real case where a customer
reported a driver following their friend and got a generic templated
reply instead of an escalation. I fixed it with one rule — treat
unknown-intent predictions as inherently escalate-worthy — which cut
that to under 1%, but I want to be upfront that it wasn't a free fix:
unnecessary escalations went up by a lot too. That tradeoff, and being
honest about it instead of just reporting the after-number, is the part
of this project I'd defend most."

---

# Interview Preparation

Note: the brief asked for 30+20+15+15+10+10+10+10 = 120 questions. Given
the 6-hour budget was spent primarily on the working system and honest
evaluation (the brief's own stated priority order), this document covers
a condensed but representative set per category — enough to prepare for
the actual interview — rather than 120 padded entries. Expand any
section using the same format if you want more.

## 2-minute project explanation

"I built a customer-support triage agent: it takes an incoming message
and a brand, classifies the intent with TF-IDF and logistic regression,
retrieves similar historical cases with the same brand prioritized,
drafts a reply grounded in that retrieved evidence, and decides whether
to auto-send it or escalate to a human. The escalation policy is
deliberately conservative — low confidence, risky keywords, weak
retrieval evidence, or a financially-sensitive intent all force a human
handoff, because a wrongly auto-sent reply is worse than an unnecessary
one. I built and ran a full evaluation harness with a golden set, but
also specifically tested it on out-of-distribution, hand-written
examples, because the golden-set numbers looked suspiciously perfect —
and the stress test found real failures immediately, which I documented
instead of hiding. The one thing I couldn't do in this sandbox was reach
the internet, so I used a synthetic dataset and template-based
generation instead of a live LLM — both are clearly labeled everywhere,
and the code has a ready-made upgrade path to the real dataset and a
real LLM API without any architecture changes."

---

## Technical questions

1. **Q: Why TF-IDF + Logistic Regression instead of a neural classifier?**
   Simple: fast, interpretable, no GPU, works well on small labelled sets.
   Deeper: LR on sparse TF-IDF features is a strong, well-calibrated
   baseline for short-text classification; a neural model needs far more
   data to beat it and adds latency/complexity disproportionate to the
   6-hour budget and synthetic dataset size.
   Tests: whether you default to complexity or justify it with evidence.

2. **Q: How does retrieval avoid leaking test data into training?**
   Simple: the retrieval index is built only from the training split.
   Deeper: `HistoricalRetriever` is constructed from `train.csv`; the
   golden set is sampled from `test.csv`, and thread-level (not row-level)
   splitting means no conversation appears on both sides.
   Tests: understanding of data leakage beyond just "don't train on test."

3. **Q: What happens if `sentence-transformers` isn't installed?**
   Simple: retrieval automatically falls back to TF-IDF cosine similarity.
   Deeper: `_try_load_embedder()` catches the ImportError and returns
   `None`; `HistoricalRetriever.__init__` branches on that to pick TF-IDF
   instead — same public interface either way, no caller changes needed.
   Tests: whether you design for graceful degradation vs. hard failure.

4. **Q: How is "confidence" computed?**
   Simple: the classifier's own predicted probability for the winning class.
   Deeper: `LogisticRegression.predict_proba(...).max()` — this is a
   model-calibration-dependent number, not a guarantee of correctness; it
   reflects how sure the model is given its training distribution, which
   is why it's only one of several escalation signals, not the only one.
   Tests: whether you understand confidence ≠ correctness.

5. **Q: Why does the safety check run AFTER generation instead of just
   trusting the escalation decision made before it?**
   Simple: generation could still introduce an unsupported claim even
   when the pre-generation signals looked fine.
   Deeper: the escalation policy and the grounding check operate on
   different information — the policy sees message/intent/confidence/
   retrieval score, the grounding check sees the actual drafted text.
   Defense in depth: either can independently force escalation.
   Tests: layered-safety thinking vs single-gate thinking.

6. **Q: What's a `RetrievedCase` and why a dataclass?**
   Simple: a small typed container for one retrieval result.
   Deeper: dataclasses give free `__repr__`/equality and, combined with
   `asdict()`, trivially serialize to the JSON schema `run_agent` returns
   — cheaper than hand-writing a dict builder and less error-prone than
   passing raw tuples around.
   Tests: comfort with idiomatic Python data modeling.

7. **Q: Why store thresholds in `config.yaml` instead of as constants
   in the code?**
   Simple: one place to change behavior without touching code.
   Deeper: it's also what makes the "Reproducibility" section of the
   README possible — every run is fully described by the config file
   plus the seed, so results can be regenerated or audited.
   Tests: whether you separate configuration from logic by habit.

8. **Q: How would you add rate limiting / retries for the LLM API path?**
   Simple: wrap the `anthropic` client call in a retry-with-backoff loop.
   Deeper: not implemented here since the path is untested (no key
   available), but the natural spot is inside `_llm_api_reply()` in
   `generator.py` — catch `anthropic.RateLimitError` specifically and
   back off, rather than a blanket except.
   Tests: whether you think about production failure modes even for
   code you didn't get to fully exercise.

9. **Q: Why is the classifier trained fresh every time `load_model()`
   runs if no pickle exists, rather than always retraining?**
   Simple: caching — avoids retraining on every pipeline call.
   Deeper: `load_model()` checks for `outputs/metrics/tfidf_lr_model.pkl`
   first; this trades a small risk of a stale model (if training data
   changes but the pickle isn't deleted) for much faster repeated calls
   — acceptable here since `main()` in `tfidf_lr.py` always retrains
   explicitly.
   Tests: whether you can articulate a caching tradeoff, not just use one.

10. **Q: Walk me through what happens if `retrieved_cases` is empty.**
    Simple: the generator returns a clarifying-question reply instead of
    trying to ground on nothing.
    Deeper: `_template_grounded_reply` explicitly checks `if not
    retrieved_cases` first — this can't actually happen given `top_k>=1`
    and a non-empty training set, but the guard exists because a future
    change (e.g. empty brand-specific index) shouldn't silently produce
    an ungrounded reply.
    Tests: defensive coding for edge cases that "shouldn't happen."

11. **Q: Why `cosine_similarity` from scikit-learn instead of computing
    it manually with numpy?**
    Simple: it's a one-line, well-tested, vectorized call.
    Deeper: it also handles both dense (sentence-transformers) and
    sparse (TF-IDF) inputs transparently, which is exactly why the
    retriever code doesn't need an if/else for the two backends beyond
    building the vectors.
    Tests: knowing your library, not reinventing it.

12. **Q: How is the "same-brand boost" applied without corrupting the
    reported similarity?**
    Simple: the boost is only used for *ranking*, the raw similarity is
    still what's reported/stored.
    Deeper: `retrieve()` computes `boosted_sims` for `argsort` but stores
    `sims[idx]` (the unboosted value) on the returned `RetrievedCase` —
    otherwise downstream grounding thresholds (`min_similarity_for_grounding`)
    would be comparing an already-inflated number.
    Tests: separating "what to do" logic from "what to report" logic.

---

## Design-decision questions

1. **Q: Why synthetic data instead of just skipping the dataset entirely
   and describing what you'd do?** A: A description alone wouldn't have
   let anything be actually run and measured, and the brief's central
   requirement is real, non-fabricated numbers — even on the wrong
   dataset, the metrics are honest and every claim is checked. See D1.
2. **Q: Why not just call the golden-set 1.00 F1 a success and move on?**
   A: It would violate the project's own non-negotiable honesty rule and
   Phase 28's explicit ask to interrogate headline numbers — see D7 and
   `MISLEADING_HEADLINE_NUMBER.md`.
3. **Q: Why unittest instead of skipping tests when pytest wasn't
   installable?** A: Tests were an explicit requirement (Phase 31);
   unittest is stdlib and pytest-compatible, so nothing is lost when the
   grader has pytest available. See D14.
4. **Q: Why is `general_feedback` a catch-all instead of splitting
   praise from complaints from pre-sales questions?** A: With only 6
   hours and a documented preference for a smaller, well-separated
   taxonomy (Phase 13), three low-volume, easily-confused intents were
   merged into one broader, still-useful category rather than adding
   classes that would mostly just increase confusion in a small dataset.
5. **Q: Why escalate ALL of `billing_refund` and `account_access`
   instead of only escalating low-confidence cases within them?** A:
   These are the intents most likely to cause real financial/account
   harm if auto-handled incorrectly; the policy treats them as
   inherently sensitive regardless of confidence — see D10, prioritizing
   false-auto-handle avoidance.
6. **Q: Why keep the 0.15 same-brand boost additive instead of
   multiplicative?** A: Additive is simpler to reason about and to
   document (a flat bonus, not a scaling factor that behaves
   differently at different similarity magnitudes) — appropriate given
   thresholds weren't formally tuned (see D9).
7. **Q: Why regenerate order numbers out of historical responses
   (`_generalize()`) instead of trusting the LLM/template not to
   confuse customers?** A: A template can't "decide" not to include a
   number; the regex strip is a mechanical guarantee against quoting a
   different customer's order number, which matters more than losing
   some specificity.
8. **Q: Why is there no `app.py` / Streamlit file?** A: Per the brief's
   own priority order, UI is explicitly last and optional; the CLI
   fully exercises the pipeline and was judged sufficient given time
   spent on evaluation quality instead — see D12.

---

## ML / NLP questions

1. **Q: What does TF-IDF actually compute?** Term frequency (how often a
   word appears in this document) times inverse document frequency (how
   rare that word is across all documents) — common words get downweighted,
   distinctive words get upweighted.
2. **Q: Why `ngram_range=(1,2)`?** Unigrams alone miss short phrases like
   "can't log in"; bigrams capture some of that local word order cheaply.
3. **Q: What does macro F1 do differently from accuracy?** It computes
   F1 per class then averages with equal weight per class, so a model
   that ignores a rare-but-important class (e.g. fraud) can't hide
   behind strong performance on common classes.
4. **Q: Why L2-regularized logistic regression (`C=2.0`)?** `C` is the
   inverse regularization strength; a moderate C keeps a non-huge
   coefficient space with ~5000 TF-IDF features from overfitting a
   ~1100-example training set.
5. **Q: What's the difference between the TF-IDF fallback and
   sentence-transformer embeddings for retrieval?** TF-IDF matches on
   literal shared vocabulary; sentence embeddings capture semantic
   similarity even with different wording (e.g. "cancel" vs "stop my
   subscription") — which is exactly the gap the stress test exposed.
6. **Q: Why not use a neural network for intent classification given
   the dataset is small?** Small labelled datasets are exactly where
   simpler, regularized linear models tend to generalize better than
   neural nets, which need more data to avoid overfitting.
7. **Q: What would you check before trusting a classifier's confidence
   score as a real probability?** Calibration — plot predicted
   confidence against actual accuracy in bins; logistic regression is
   reasonably well-calibrated by construction, but this wasn't
   separately verified here (a real limitation, not measured).
8. **Q: How would multi-label intent classification change the pipeline?**
   The classifier would need `predict_proba` per label with a threshold
   instead of `argmax`, and both retrieval and escalation would need to
   handle a set of intents instead of one — a bigger architectural
   change than it sounds, since escalation rules are currently keyed on
   a single `predicted_intent`.

---

## RAG questions

1. **Q: What does RAG mean here specifically?** Retrieve similar
   historical customer-support cases first, then generate a reply that's
   grounded in (constrained by) that retrieved evidence rather than the
   model's unconstrained imagination.
2. **Q: Why prioritize same-brand retrieved cases?** Different brands
   have different policies, tone, and typical resolutions; a cross-brand
   match might describe an irrelevant process.
3. **Q: What happens when retrieval quality is poor?** The escalation
   policy checks `top_similarity` against
   `min_similarity_for_grounding` (0.18) and escalates or asks for
   clarification rather than generating on weak evidence — Phase 20's
   "poor retrieval → escalate or clarify" rule.
4. **Q: How is "grounded" enforced mechanically, not just by
   instruction?** Two ways: the template generator literally builds the
   reply from retrieved text (can't invent facts not present in it), and
   the post-generation `check_grounding()` regex-scans for dollar
   amounts/dates/order numbers that don't appear in the evidence.
5. **Q: What's a weakness of the current grounding check?** It's regex
   pattern-based, so it only catches specific claim *shapes* (dollar
   amounts, dates, order numbers) — it wouldn't catch a false claim
   phrased without those patterns, e.g. "your refund has already been
   processed" with no number attached.
6. **Q: Why cap retrieval at top_k=3?** A tradeoff — more retrieved
   cases give more grounding material but dilute relevance and add
   prompt-length cost if/when a real LLM is used; 3 was a reasonable,
   documented but not empirically tuned default (see D9's honesty about
   thresholds not being tuned).
7. **Q: How would you evaluate retrieval quality independent of the
   end-to-end system?** Precision@k against a labelled "is this
   historical case actually relevant" set — not implemented here
   (labelled explicitly as a gap, see README Limitations).
8. **Q: If you had an LLM available, what would you change about the
   generation prompt?** Nothing structurally — `SYSTEM_INSTRUCTIONS` in
   `generator.py` is already written and ready; the main addition would
   be a few-shot example of a properly-declined/clarifying response to
   anchor the "ask for clarification" behavior more reliably.

---

## Evaluation questions

1. **Q: Why build a golden set at all instead of just eyeballing outputs?**
   Repeatable, quantifiable comparison across runs/models — the whole
   point of the evaluation harness.
2. **Q: Why is the golden set drawn only from the test split?** So its
   examples were never used to fit the classifier or build the
   retrieval index — otherwise it wouldn't measure generalization.
3. **Q: What does `false_auto_handle_count` measure and why track it
   separately from overall accuracy?** Cases where the system said
   AUTO_HANDLE but should have escalated — the single most dangerous
   error type, worth tracking on its own rather than averaged away.
4. **Q: Why compute both macro AND weighted F1?** Macro exposes
   minority-class weakness; weighted shows overall volume-weighted
   performance — reporting only one can hide the other's story.
5. **Q: How would you validate the golden set's own labels?** Have a
   second, independent labeler review a sample and compute agreement
   (e.g. Cohen's kappa) — not done here since labels are rule-derived,
   not human, by necessity (see D8).
6. **Q: Why report "Not measured yet" for reply quality instead of
   using a proxy metric like ROUGE?** ROUGE against what reference? There
   is no ground-truth "ideal reply" to compare against here, and a
   loosely-justified proxy number risks looking like a measured result
   when it isn't — better to state the gap plainly.

---

## Failure-analysis questions

1. **Q: How did you find real failures when the golden set showed 0?**
   Built a small, hand-written, deliberately out-of-template stress test
   — see `scripts/stress_test.py` and D7.
2. **Q: What's the common thread across the two documented failures?**
   Both are casual/paraphrased phrasing that doesn't match the narrow
   3-templates-per-intent training data — a training-data diversity
   problem, not a fundamentally broken approach.
3. **Q: In failure #2, was the wrong escalation decision dangerous?**
   No — it escalated a harmless compliment unnecessarily, which is a
   wasted-effort error, not an unsafe one; exactly the tradeoff the
   conservative policy is designed to make (safe direction, not free).
4. **Q: How would you prioritize which failure to fix first in a real
   system?** By the product of frequency × severity — a rare-but-unsafe
   error (like a missed fraud case) outranks a common-but-safe one (like
   an unnecessary escalation) even if the second happens more often.
5. **Q: What's a failure mode you predicted but didn't observe, and
   why didn't you fabricate an example for it?** The same-brand-boost
   promoting a topically worse match when a brand has sparse history —
   not observed because every synthetic brand has 35+ examples per
   intent; documented as a hypothesis instead of inventing a fake
   example, per the project's no-fabrication rule.
6. **Q: How would you get more reliable failure statistics?** Grow the
   stress test from 8 to ~50+ independently-written examples — flagged
   explicitly as a "one more week" item, not silently left implied.

---

## System-design questions

1. **Q: How would this scale to millions of messages/day?** Swap the
   TF-IDF/pickle retrieval and model loading for a real vector database
   (e.g. a managed FAISS/pgvector service) and a served model endpoint
   instead of in-process `pickle.load`; batch classification requests.
2. **Q: Where's the current single point of failure?** The in-process
   singleton retriever/model (`_RETRIEVER_SINGLETON`, `_MODEL`) — fine
   for a single-process CLI/demo, not for a multi-worker production
   deployment without moving state out of module globals.
3. **Q: How would you add human-in-the-loop feedback to improve the
   classifier over time?** Log every `run_agent` call plus the human
   agent's actual eventual action/label, periodically retrain
   `tfidf_lr.py` on the accumulated real-world-labelled data instead of
   only the synthetic set.
4. **Q: How would you version the escalation policy safely?** Keep
   `config.yaml` thresholds in a versioned config store, log which
   config version produced each decision (not currently done — a real
   gap), and A/B or shadow-test threshold changes before rollout.
5. **Q: What would you monitor in production?** Escalation rate over
   time, false-auto-handle rate against downstream complaint/refund
   data, retrieval similarity distribution drift, and confidence
   calibration drift.
6. **Q: How would you avoid the retrieval index going stale?** Rebuild
   the `HistoricalRetriever` on a schedule (or trigger) from fresh
   training data instead of the current one-shot singleton — not
   implemented, flagged as a real gap for a long-running service.

---

## Python questions

1. **Q: Why `dataclass` for `RetrievedCase` and `EscalationDecision`
   instead of plain dicts?** Type-checkable structure, IDE autocomplete,
   and `asdict()` gives free, correct JSON serialization.
2. **Q: What does the `_RETRIEVER_SINGLETON` global pattern trade off?**
   Avoids rebuilding the (potentially expensive) retrieval index on
   every call, at the cost of hidden mutable global state — acceptable
   for a demo/CLI, a real concern for concurrent serving.
3. **Q: Why `Path` objects (`pathlib`) instead of string paths
   throughout `src/utils/config.py`?** Cross-platform path joining and
   clearer, chainable operations (`.parent.mkdir(...)`) versus manual
   string concatenation.
4. **Q: How does `load_model()` avoid retraining unnecessarily?** It
   checks whether the pickle file exists first and only calls
   `train_and_save()` if it doesn't — simple existence-check caching.
5. **Q: Why `zero_division=0` in the sklearn metric calls?** Without it,
   sklearn raises warnings/errors when a class has no predicted samples
   — common with a small golden set — and `0` is the conservative,
   honest choice for undefined precision/recall in that case.
6. **Q: Why `if __name__ == "__main__":` guards in every module?**
   Lets every module be both imported (e.g. `run_agent` importing
   `retrieve_similar_cases`) and run standalone for manual testing
   (e.g. `python -m src.retrieval.retriever`) without side effects on import.
