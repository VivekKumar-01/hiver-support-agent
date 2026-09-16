# Failure Analysis (Real Data)

This supersedes the synthetic-data version of this document. With real
data, the golden set itself surfaces genuine failures at scale (240
examples, 30% intent accuracy) — no separate stress test was needed to
find real problems, though `scripts/stress_test.py` (8 hand-written,
non-golden-set messages) is still included and found 1/8 intents and
6/8 escalation actions correct, consistent with the golden-set picture.

## Failure 1 (most severe) — a driver-stalking safety report was auto-handled with a generic templated reply

1. **Failure type:** Intent misclassification -> dangerous false auto-handle (before the D18 fix)
2. **Actual example (AppleSupport/Uber_Support, real tweet):**
   *"Hey ... One of your drivers in a new silver Mercedes followed my
   friend in Hollywood last night. Possible plate number: 7ZHJ231.
   Please follow up."*
3. **Expected behavior:** `security_fraud` (a genuine safety/security
   report), ESCALATE immediately.
4. **Actual behavior (before D18):** Classified as `other` with 0.93
   confidence, decision AUTO_HANDLE, and the system generated: *"Thanks
   for reaching out to Uber_Support support. Based on how we've handled
   similar other cases before, here's what typically happens next:
   Happy to help, Harry! Please DM us the email and phone number linked
   to your account and we will follow up."* — a generic, unrelated
   templated reply to a stalking report.
5. **Why it failed:** The weak-training-label rule set for
   `security_fraud` looks for words like "hacked"/"fraud"/"stolen" —
   this message describes a real safety incident without any of those
   trigger words, so it was weak-labelled (and then learned) as "other."
   The risk-keyword escalation check has the same blind spot (its list
   is "fraud, unauthorized, lawsuit, ... hacked, stolen, ..." — none of
   which appear here either).
6. **Hypothesis:** Both the weak-label rules AND the risk-keyword list
   were designed around financial/account-compromise language and don't
   cover physical-safety language ("followed," "plate number") at all —
   a genuine taxonomy/keyword-coverage gap, not a model capacity problem.
7. **Fix applied:** D18 (escalate on any `predicted_intent == "other"`)
   catches this specific case as a side effect, since the message landed
   in "other" — but that's incidental, not a real fix for the
   underlying gap. **Not yet fixed:** add physical-safety keywords
   ("followed," "stalking," "license plate," "in danger") to the risk
   list directly, so safety reports are caught on their own merits
   rather than by the blunter "unknown intent" catch-all. Flagged as a
   "one more week" item in REPORT.md.

## Failure 2 — a repeated, unresolved billing dispute was auto-handled with a boilerplate reply

1. **Failure type:** Intent misclassification -> false auto-handle (before D18)
2. **Actual example (AmazonHelp, real tweet):** *"Had purchased Amazon
   echo dot my prime services had to be restarted from 6th November but
   that never happened. I have spoken with the customer service team
   twice, but they have not helped. Don't know how they function and I
   have no idea when will this be resolved!"*
3. **Expected behavior:** `billing_refund`/subscription intent, ESCALATE
   (customer explicitly states two prior support contacts failed).
4. **Actual behavior (before D18):** Classified `other` at 0.88
   confidence, AUTO_HANDLE, generic reply offering a generic form link —
   ignoring that the customer already said this exact path failed twice.
5. **Why it failed:** The weak-label rule for `billing_refund` keys on
   "refund"/"charged"/"billed" — this message describes a
   service-not-restarted problem without ever using those words.
6. **Hypothesis:** Same root cause as Failure 1 — weak-label keyword
   coverage is too narrow for the linguistic diversity of real
   complaints, and it compounds with the classifier's overconfidence
   (see MISLEADING_HEADLINE_NUMBER.md point 2): 0.88 confidence on a
   wrong prediction, not a borderline call.
7. **Possible fix:** Same as Failure 1 (better weak-label rules / real
   labelled training data), plus a targeted rule: if a message states
   "spoken with customer service twice"/"already contacted"/similar
   repeated-contact language, escalate regardless of predicted intent —
   repeated failed contact is itself a strong, easy-to-detect escalation
   signal that nothing in the current policy checks for.

## Failure 3 — the D18 fix overcorrects: many harmless "other" messages are now escalated unnecessarily

1. **Failure type:** False escalate (introduced by the D18 fix itself)
2. **Actual example pattern:** Positive/neutral messages that the
   classifier correctly-ish routes to "other" (e.g. praise, light
   humor, simple policy questions) now escalate unconditionally, because
   D18 treats every "other" prediction as escalate-worthy regardless of
   the message's actual tone or risk.
3. **Expected behavior:** AUTO_HANDLE for clearly low-risk messages.
4. **Actual behavior:** ESCALATE, contributing to the false-escalate
   count rising from 7/240 to 114/240 after D18.
5. **Why it happens:** D18 is a blunt, intent-only rule — it can't
   distinguish "other, safety concern" (Failure 1) from "other, just
   saying thanks" using only the predicted label.
6. **Hypothesis:** The real fix isn't a smarter escalation rule but a
   better classifier/taxonomy — if `general_feedback` positives were
   reliably classified instead of falling into `other`, D18 wouldn't
   need to catch them.
7. **Possible fix:** A cheap, high-value addition: a simple sentiment/
   risk heuristic (e.g., presence of positive words + absence of any
   risk keyword) that allows "other" messages to AUTO_HANDLE only when
   they look unambiguously positive/neutral — narrowing D18's blast
   radius without reopening Failure 1/2's danger. Not implemented here
   — flagged for "one more week."

## Failure 4 — non-English text slipping past the language filter

1. **Failure type:** Data-quality / scope-boundary failure, not a
   pipeline prediction failure.
2. **Actual example:** French, Spanish, Portuguese, and Dutch messages
   (9 found among 220 read candidates) passed the ASCII-ratio filter in
   `src/preprocessing/clean.py` because Romance-language text is still
   mostly ASCII characters.
3. **Expected behavior:** Filtered out before reaching the golden set
   (the taxonomy and templates are English-only by design — D16).
4. **Actual behavior:** Caught only because every golden-set candidate
   was individually read before labelling; NOT caught automatically.
5. **Why it failed:** ASCII-ratio is a script-detection heuristic (Latin
   vs. non-Latin alphabets), not a language-detection heuristic.
6. **Hypothesis:** Any non-English messages remaining in `train.csv` /
   `train_weak_labelled.csv` (not individually reviewed, unlike the
   golden set) are silently mislabelled by the weak-label keyword rules
   and contribute noise to classifier training — not measured, but
   architecturally certain to some degree.
7. **Possible fix:** Add a real language-detection library (e.g.
   `langdetect`) to `clean_dataframe()` instead of the ASCII heuristic.
