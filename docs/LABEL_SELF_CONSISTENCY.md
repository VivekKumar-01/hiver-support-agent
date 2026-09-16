# Label Self-Consistency Check (NOT Inter-Annotator Agreement)

## What was actually done

The checklist item is "Inter-annotator agreement reported (50 double-labelled, κ ≥ 0.6)".
That specific claim requires **two independent human annotators**. This
project has **one annotator: the assistant (Claude)**, so true
inter-annotator agreement cannot be computed or claimed — doing so would
misrepresent what happened.

What *was* done, as an honest partial substitute: 50 examples were
sampled from the golden set (`data/evaluation/golden_set.csv`), their
labels hidden, and the assistant re-read and re-labelled each one from
scratch in a second pass, then the two label sets were compared.

## Result

- Raw agreement: 49/50 (0.98)
- Cohen's κ: 0.975

## Why this number should NOT be read as evidence of label quality

This is a **self-consistency** check, not inter-annotator agreement, and
the difference matters a lot here, not just semantically:

1. **Same annotator, same rules, same taxonomy document in "mind" both
   times.** Two independent humans might interpret an ambiguous
   category boundary (e.g. "is a driver-behavior complaint
   `security_fraud` or `other`?") differently from each other. The same
   annotator applying the same internalized rule twice will tend to
   agree with itself far more than two different people would agree
   with each other — that's exactly what a κ this high (0.975) most
   likely reflects: rule self-consistency, not correctness or
   real-world inter-rater reliability.
2. **No true blinding.** A human re-labelling task blinds the labeler
   from their own prior answer by using time distance, different
   session, etc. Here, the "blind" pass happened in the same
   conversation shortly after the original labelling, with the same
   taxonomy document freshly in context — a much weaker blind than an
   independent reviewer days later would provide.
3. **The one disagreement is informative on its own**: "iPhone 6 - both
   home & work Wi-fi - & after I've closed the app & re-started the
   phone" was labelled `technical_issue` originally and `other` on
   re-read — it's a genuinely ambiguous, context-free fragment (likely a
   continuation of a longer thread we don't have), which is exactly the
   kind of real-world message where two independent humans would also
   likely disagree.

## What this substitute DOES demonstrate

- The intent taxonomy and its definitions (`docs/INTENT_TAXONOMY.md`)
  are specific enough to apply consistently by the same
  reader/annotator across two passes — i.e., the categories aren't so
  vague that even a single consistent rule-follower contradicts itself
  often.
- It does NOT demonstrate that a second, independent human would label
  these messages the same way. That check was not possible in this
  environment (no second annotator available) and is listed as a
  concrete "one more week" item in `REPORT.md`.

## If you want real inter-annotator agreement

Have a second person (ideally someone who didn't write
`docs/INTENT_TAXONOMY.md`) independently label the same 50-example
subset (`data/evaluation/golden_set.csv`, ids in the `id` column) using
only the taxonomy document, then run:

```python
from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(assistant_labels, human_labels)
```

κ ≥ 0.6 on that comparison would be real evidence; κ ≥ 0.6 on the
self-consistency check above is not a comparable claim.
