# Checklist Compliance

Direct mapping of the external review checklist to this project's
current, real state. ✅ = met, ⚠️ = partially met (explained), ❌ = not
met (explained, and whether it's fixable without new resources).

## Code and Reproducibility

| Item | Status | Notes |
|---|---|---|
| README has install/data/run steps | ✅ | README.md §1, §6, §14 |
| Ships sample_data.csv (500–1000 AppleSupport tweets) | ✅ | `data/raw/sample_data_applesupport.csv`, 1,000 real rows |
| Dependencies pinned | ✅ | `requirements.txt`, exact `==` versions |
| API keys via env vars, explained | ✅ | `.env.example`, README §6/10 |
| `python run_pipeline.py --brand AppleSupport --sample` | ✅ | Implemented, tested |
| `python evaluate.py --golden_set golden_200.json` | ⚠️ | `evaluate.py --golden_set <path>` works with the actual CSV golden set; the checklist's exact filename (`golden_200.json`) doesn't match this project's format/naming (`golden_set.csv`, 240 rows) — the flag and mechanism are real and functional, just not that literal filename |

## Golden Set Quality

| Item | Status | Notes |
|---|---:|---|
| 150–250 hand-labelled examples | ⚠️ | 240 examples, individually assistant-labelled — NOT human-labelled (see README §7). This is the most significant remaining gap. |
| Each intent ≥15 examples incl. "other" | ⚠️ | 7/9 classes reach it; `order_status` (7) and `cancellation_request` (4) genuinely don't, despite full-test-split keyword search — disclosed, not padded (D19) |
| `should_escalate` labelled every row | ✅ | `expected_action` column, every row |
| Sampling notes: time range, stratification, seed | ✅ | README §5, `data/README.md`; real date range 2008–2017 noted in inspection output |
| Codebook exists | ✅ | `docs/INTENT_TAXONOMY.md` |
| Inter-annotator agreement (50 double-labelled, κ≥0.6) | ❌ | No second annotator available. Self-consistency (same annotator, blind re-pass) measured instead: κ=0.975 — explicitly NOT the same claim, see `docs/LABEL_SELF_CONSISTENCY.md`. Fixable only with a second human labeler. |

## Evaluation Harness

| Item | Status | Notes |
|---|---|---|
| Intent accuracy, macro F1, escalation precision/recall | ✅ | `outputs/metrics/*.json` |
| LLM-judge rubric, ≥4 criteria | ✅ | 6 criteria, `src/evaluation/llm_judge.py` |
| Judge runs at temperature=0 | ✅ (code) / ❌ (execution) | Set in code (`config.yaml > judge.temperature`); never actually run — no API key |
| Human agreement on ≥50 examples | ❌ | Same root cause as inter-annotator agreement above — no second human |
| Cohen's κ / weighted κ, per-criterion | ❌ | N/A — judge never executed, no human comparison exists |
| Rubric revision history if κ<0.6 | ➖ | N/A, nothing was measured to revise against |

## Baselines

| Item | Status | Notes |
|---|---|---|
| Trivial baseline runs end-to-end | ✅ | `src/classification/baseline_majority.py` |
| Simple baseline runs end-to-end | ✅ | `src/classification/tfidf_lr.py` |
| Results table, all systems | ✅ | README §9 |
| Honest analysis if a metric loses to simple baseline | ➖ | Proposed system beats the trivial baseline on every metric (barely) — see README §9 and `MISLEADING_HEADLINE_NUMBER.md` for why even the win is not strong evidence |

## Report

| Item | Status |
|---|---|
| "What I chose NOT to build" | ✅ REPORT.md §4 |
| Results vs two baselines | ✅ REPORT.md §9 |
| Top 5 failure modes, real example + hypothesis | ✅ 4 fully real, evidenced failures (`FAILURE_ANALYSIS.md`) — one short of 5, all real (no filler/hypothetical ones this time, unlike the synthetic milestone) |
| "Misleading headline number," says something real | ✅ Mechanism-level: weak-label coverage, confidence miscalibration, the escalation-policy before/after tradeoff |
| "One more week" | ✅ REPORT.md §13 |
| ≤6 pages | ✅ |

## Decision Log

| Item | Status |
|---|---|
| 10–15 non-obvious decisions | ✅ 19 total (14 synthetic-milestone + 5 real-data-rebuild) |
| Each has decision + why | ✅ |
| Covers brand choice, taxonomy, baselines, judge design, scope cuts | ✅ | D15 (brand/source choice), D16 (language scope), D17 (weak labels/baseline design), D18 (escalation policy fix), D19 (golden-set sampling); judge design (6 criteria, temperature=0) documented in `src/evaluation/llm_judge.py` and README §8 |

## Red Flags — self-check

| Red flag | Present? |
|---|---|
| Headline number has no CI | ❌ Fixed — every headline metric now has a 95% bootstrap CI |
| Judge reports % agreement only, no κ | ➖ N/A, judge never run — nothing to misreport |
| "other" rate artificially low | ❌ — "other" is 25.8% of the golden set, the single largest class, reported plainly |
| Only success examples shown | ❌ — all 4 documented failures are real errors, including a serious one (driver-stalking report auto-handled) |
| "Misleading" section vague | ❌ — mechanism-level (weak-label %, confidence-vs-accuracy gap, before/after policy numbers) |
| Baselines assumed not run | ❌ — both executed, real output saved |

## Bottom line

The two items that remain genuinely unmet — human-labelled golden set
and true inter-annotator/human-judge agreement — require a second human
being involved in this project, which isn't something achievable from
within this environment. Everything else on the checklist that was
fixable with real data, real code, and honest documentation has been
addressed, including finding and fixing a real safety bug (D18) that
the evaluation harness surfaced.
