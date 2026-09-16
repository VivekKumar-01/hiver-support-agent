# Intent Taxonomy

8 intents, chosen to be the smallest set that (a) covers the message
types seen in the sampled data and (b) maps cleanly onto different
escalation behavior — which is what actually matters downstream.

| Intent | Definition | Inclusion criteria | Exclusion criteria | Example |
|---|---|---|---|---|
| `order_status` | Customer wants to know where an order/delivery currently stands. | Asks "where is", "status of", "tracking" for a placed order. | Complaints about damage/non-delivery (→ `shipping_delivery`). | "Still waiting on order 973835, tracking hasn't moved in 11 days." |
| `billing_refund` | Customer disputes a charge or wants money back. | Mentions refund, duplicate charge, wrong price. | Cancellation without a billing dispute (→ `cancellation_request`). | "Why was I billed $199 when the item was on sale?" |
| `account_access` | Customer cannot log in / access their account. | Password reset, 2FA, locked out. | Account was compromised by someone else (→ `security_fraud`). | "Locked out of my account since yesterday, 2FA code never arrives." |
| `technical_issue` | Product/app/site is malfunctioning. | Crashes, error codes, won't load. | Billing errors caused by a bug (ambiguous — see Failure Analysis #3). | "Getting error code E502 whenever I try to check out." |
| `cancellation_request` | Customer wants to stop a subscription/order. | Explicit "cancel". | Refund-focused cancellation (→ `billing_refund` if refund is the main ask). | "Please cancel order 823468, I no longer need it." |
| `shipping_delivery` | Physical delivery problem (damage, wrong address, lost). | Item arrived damaged, "never received", address change pre-shipment. | Simple ETA question (→ `order_status`). | "Package for 667059 arrived damaged, box was crushed." |
| `security_fraud` | Account compromise or unauthorized transaction. | Words/phrasing implying someone else accessed the account or card. | Ordinary billing dispute with no compromise implied (→ `billing_refund`). | "I think my account was hacked, there are purchases I didn't make." |
| `general_feedback` | Praise, general complaints not tied to a specific order, or pre-sales questions. | No specific order/account action needed. | Anything actionable (falls into one of the above). | "Does the premium plan include priority support?" |

## How this taxonomy was derived

For the real Kaggle dataset, Phase 13 of the brief calls for deriving
intents from actually reading a sample of messages, not assuming a
predefined list. Because this project uses a **synthetic** dataset (see
README > Limitations), the taxonomy was designed up front as the
generation schema rather than discovered by reading real messages
after the fact. The categories above were still chosen for the reasons
a real derivation would use: they are mutually distinguishable, cover
plausible high-frequency customer-support contact reasons, and map to
meaningfully different escalation behavior (e.g. `security_fraud` is
always escalated; `general_feedback` is usually safe to auto-handle).

**On a real dataset**, the correct process (not performed here, since
there is no real data available) is: sample ~200-300 raw messages,
read them, cluster by hand or with a quick embedding + k-means pass,
merge near-duplicate buckets, and only then lock the taxonomy — see
DECISION_LOG.md decision D3.

## Real-data addendum

With the real dataset, "other" turned out to be the single LARGEST
class in the golden set (62/240, 25.8%) — real Twitter support traffic
includes a lot of flight-logistics questions, driver-behavior
complaints, feature requests, off-topic social replies, and
context-free thread fragments that don't fit any of the 8 substantive
intents. This is reported honestly as a taxonomy-coverage finding, not
suppressed — see `MISLEADING_HEADLINE_NUMBER.md` point 6.

Two intents could not reach a 15-example floor in the golden set even
after a full-test-split keyword search: `order_status` (7 examples) and
`cancellation_request` (4 examples). Real customers rarely ask a neutral
"where is my order" question or explicitly request a cancellation via
Twitter specifically — those interactions likely happen through the app
or website instead. See D19 in `DECISION_LOG.md`.
