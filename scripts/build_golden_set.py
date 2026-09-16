"""
Phase 6 (real data) — Golden Evaluation Set, built from real messages.

LABELLING METHODOLOGY — read this before trusting these labels:
Every label below was assigned by the assistant (Claude) individually
reading each real customer message and its context (brand, surrounding
thread fragment) and choosing the best-fit intent from
docs/INTENT_TAXONOMY.md, or "other" when nothing fit. This is a
substantive upgrade over the earlier synthetic project's rule-derived
labels — a real judgment call was made per example, on real, messy
Twitter text. It is NOT, however, equivalent to the checklist's
"hand-labelled by a human annotator" requirement or an independent
human review: there is exactly one annotator (the assistant), so no
true inter-annotator agreement / Cohen's kappa can be computed. A
"self-consistency" check (the assistant re-labelling a subset blind to
its earlier answers) is reported separately in
docs/LABEL_SELF_CONSISTENCY.md as an honest partial substitute — see
that file for what it does and does not demonstrate.

Sampling: candidates were stratified by brand from the TEST split
(never seen by the classifier), read in full, and 9 were excluded for
being non-English (see EXCLUDED_NON_ENGLISH below — the ASCII-ratio
filter in cleaning only catches non-Latin scripts, so Romance-language
text like French/Spanish/Portuguese/Dutch slipped through and had to be
caught here by actually reading the messages).

`cancellation_request` had only 1 naturally-occurring example in the
first 220-message random sample (a real class-imbalance finding, not a
bug) — to meet the "every intent needs >=15 examples" bar without
fabricating anything, a SEPARATE, keyword-targeted supplemental sample
was drawn and labelled the same way (see scripts/label_supplemental_cancellations.py).

Run:
    python -m scripts.build_golden_set
"""
import pandas as pd

from src.utils.config import load_config, project_path

LABELS = {
    0: ("general_feedback", "ESCALATE", "Vague complaint, no specific actionable detail"),
    1: ("technical_issue", "ESCALATE", "Vague but implies something broken, needs clarification"),
    2: ("general_feedback", "ESCALATE", "Service-process complaint (long hold, disconnects)"),
    3: ("technical_issue", "ESCALATE", "Context-free 'it still doesn't work'"),
    4: ("other", "ESCALATE", "Context-less fragment, no actionable content"),
    5: ("general_feedback", "AUTO_HANDLE", "Positive/promotional tweet"),
    6: ("security_fraud", "ESCALATE", "Driver safety/stalking report with plate number"),
    7: ("billing_refund", "ESCALATE", "Charged when app wasn't working, wants refund"),
    8: ("general_feedback", "AUTO_HANDLE", "Pre-sales compatibility question"),
    9: ("other", "ESCALATE", "Context-free fragment"),
    10: ("general_feedback", "ESCALATE", "Major delay complaint, no specific ask"),
    11: ("shipping_delivery", "AUTO_HANDLE", "Delivery delay complaint"),
    12: ("billing_refund", "ESCALATE", "Disputes adequacy of a refund already given"),
    13: ("billing_refund", "ESCALATE", "Subscription/restart billing issue"),
    14: ("technical_issue", "AUTO_HANDLE", "Audio glitch bug report"),
    15: ("billing_refund", "ESCALATE", "Unrecognized large charge"),
    16: ("billing_refund", "ESCALATE", "Charged despite driver refusing ride"),
    17: ("billing_refund", "ESCALATE", "Unwanted renewal charge + service not delivered"),
    18: ("shipping_delivery", "AUTO_HANDLE", "Recurring late food delivery"),
    19: ("technical_issue", "AUTO_HANDLE", "Clipboard/paste bug"),
    20: ("billing_refund", "ESCALATE", "Charged, order never received"),
    21: ("other", "AUTO_HANDLE", "Flight upgrade/waitlist policy question"),
    22: ("general_feedback", "ESCALATE", "Complaint about automated support quality"),
    23: ("shipping_delivery", "AUTO_HANDLE", "Sarcastic complaint about parcel condition/location"),
    24: ("other", "ESCALATE", "In-flight delay, no info given, time-sensitive"),
    25: ("shipping_delivery", "AUTO_HANDLE", "Wrong/incomplete food order delivered"),
    26: ("other", "ESCALATE", "In-flight medical assistance compensation claim"),
    27: ("order_status", "AUTO_HANDLE", "Order timing question fragment"),
    28: ("technical_issue", "AUTO_HANDLE", "Hardware stopped working after 3 months"),
    29: ("general_feedback", "AUTO_HANDLE", "Feature-return request"),
    30: ("other", "AUTO_HANDLE", "General policy question"),
    31: ("technical_issue", "ESCALATE", "Third-party app integration not working, vague"),
    32: ("other", "ESCALATE", "Flight delay, no reason given, time-sensitive"),
    33: ("general_feedback", "ESCALATE", "No response to prior DM, process complaint"),
    34: ("technical_issue", "AUTO_HANDLE", "iOS version crash bug"),
    35: ("other", "ESCALATE", "Context-free fragment"),
    36: ("technical_issue", "AUTO_HANDLE", "Product quality complaint (charger)"),
    37: ("other", "AUTO_HANDLE", "Content/algorithm curiosity question"),
    38: ("other", "AUTO_HANDLE", "Version info fragment, no request"),
    39: ("general_feedback", "AUTO_HANDLE", "Positive loyalty tweet"),
    40: ("technical_issue", "ESCALATE", "App compatibility broke paid hardware workflow"),
    41: ("order_status", "ESCALATE", "Ride status, time-sensitive, no answer"),
    42: ("general_feedback", "ESCALATE", "Process/bureaucracy complaint"),
    43: ("other", "ESCALATE", "Ambiguous baggage-related fragment"),
    44: ("technical_issue", "AUTO_HANDLE", "Charging/battery bug"),
    45: ("shipping_delivery", "ESCALATE", "Order tampered with by driver"),
    46: ("general_feedback", "AUTO_HANDLE", "Mixed positive feedback about event/flight"),
    47: ("technical_issue", "AUTO_HANDLE", "Recurring shuffle bug"),
    48: ("other", "AUTO_HANDLE", "Off-topic personal note"),
    49: ("general_feedback", "AUTO_HANDLE", "Pre-sales/plan duration question"),
    50: ("other", "ESCALATE", "Context-free urgent fragment"),
    51: ("shipping_delivery", "AUTO_HANDLE", "Delivery location error, needs help"),
    52: ("other", "AUTO_HANDLE", "How-to/storage question"),
    53: ("other", "ESCALATE", "Lost item, driver contact blocked"),
    54: ("other", "AUTO_HANDLE", "Plan/policy question"),
    55: ("general_feedback", "ESCALATE", "Vague 'no help' complaint"),
    56: ("other", "AUTO_HANDLE", "Promotional-email complaint"),
    57: ("technical_issue", "AUTO_HANDLE", "WiFi connectivity bug"),
    58: ("billing_refund", "ESCALATE", "Inadequate compensation dispute (33 cents)"),
    59: ("other", "ESCALATE", "Truncated/unclear fragment"),
    60: ("technical_issue", "AUTO_HANDLE", "Photo import bug"),
    61: ("EXCLUDE", "EXCLUDE", "Non-English (Spanish)"),
    62: ("general_feedback", "AUTO_HANDLE", "Feature request"),
    63: ("general_feedback", "AUTO_HANDLE", "Feature request"),
    64: ("technical_issue", "AUTO_HANDLE", "Fewer daily mixes than expected (bug)"),
    65: ("billing_refund", "ESCALATE", "Cancellation-fee dispute"),
    66: ("general_feedback", "AUTO_HANDLE", "Market-expansion request"),
    67: ("technical_issue", "ESCALATE", "Reset didn't fix issue, following up"),
    68: ("general_feedback", "AUTO_HANDLE", "Light positive/humorous feedback"),
    69: ("technical_issue", "AUTO_HANDLE", "Promo code functionality bug"),
    70: ("other", "AUTO_HANDLE", "Content licensing question"),
    71: ("general_feedback", "ESCALATE", "Vague service-quality complaint"),
    72: ("EXCLUDE", "EXCLUDE", "Non-English (French)"),
    73: ("account_access", "ESCALATE", "Password rejected repeatedly"),
    74: ("other", "AUTO_HANDLE", "Service-availability-by-region question"),
    75: ("shipping_delivery", "AUTO_HANDLE", "Paid-tier shipping SLA missed"),
    76: ("technical_issue", "ESCALATE", "Data loss (deleted photos) — high severity"),
    77: ("other", "AUTO_HANDLE", "Contact-info request"),
    78: ("general_feedback", "AUTO_HANDLE", "Positive follow-up (resolved)"),
    79: ("technical_issue", "AUTO_HANDLE", "Feature-specific bug"),
    80: ("technical_issue", "ESCALATE", "In-flight wifi malfunction, time-sensitive"),
    81: ("technical_issue", "ESCALATE", "Recurring error blocking core function"),
    82: ("other", "ESCALATE", "Context-free fragment"),
    83: ("order_status", "AUTO_HANDLE", "Order number provided, implied status request"),
    84: ("technical_issue", "AUTO_HANDLE", "Device-specific app bug"),
    85: ("technical_issue", "AUTO_HANDLE", "Feature not working (AirDrop)"),
    86: ("billing_refund", "ESCALATE", "Reimbursement request, no contact channel found"),
    87: ("EXCLUDE", "EXCLUDE", "Non-English (Portuguese)"),
    88: ("other", "ESCALATE", "Version info fragment, no request"),
    89: ("security_fraud", "ESCALATE", "Reports drivers manipulating cash-vs-card payment"),
    90: ("billing_refund", "ESCALATE", "Wrong/cold order, threatens public complaint"),
    91: ("other", "AUTO_HANDLE", "Context-free fragment ('approved')"),
    92: ("EXCLUDE", "EXCLUDE", "Non-English (French)"),
    93: ("other", "ESCALATE", "Context-free fragment"),
    94: ("technical_issue", "AUTO_HANDLE", "Bluetooth icon glitch"),
    95: ("EXCLUDE", "EXCLUDE", "Non-English (Dutch)"),
    96: ("general_feedback", "AUTO_HANDLE", "Positive feedback"),
    97: ("general_feedback", "AUTO_HANDLE", "Positive feedback (upgrade)"),
    98: ("technical_issue", "ESCALATE", "WiFi bug, persists after restart"),
    99: ("billing_refund", "ESCALATE", "Subscription expired with no renewal warning"),
    100: ("general_feedback", "ESCALATE", "Complaint about support-team effectiveness"),
    101: ("EXCLUDE", "EXCLUDE", "Non-English (Spanish)"),
    102: ("other", "AUTO_HANDLE", "Seat-change request, mixed feedback"),
    103: ("technical_issue", "AUTO_HANDLE", "Feature-specific follow-up (Siri)"),
    104: ("other", "ESCALATE", "Ambiguous repeated-complaint fragment"),
    105: ("technical_issue", "AUTO_HANDLE", "Fewer daily mixes than expected"),
    106: ("other", "AUTO_HANDLE", "Airport-specific question"),
    107: ("order_status", "AUTO_HANDLE", "Tracking number shared, implied status check"),
    108: ("billing_refund", "AUTO_HANDLE", "Promo/discount not applied to a payment method"),
    109: ("account_access", "ESCALATE", "Account needs restart/restore"),
    110: ("other", "AUTO_HANDLE", "Feature/how-to question (multi-stop ride)"),
    111: ("general_feedback", "ESCALATE", "Vague 'no help' complaint"),
    112: ("billing_refund", "ESCALATE", "Repeated unexpected charges after a refund"),
    113: ("other", "AUTO_HANDLE", "Presale-code eligibility question"),
    114: ("billing_refund", "ESCALATE", "Order never arrived, explicit refund ask"),
    115: ("technical_issue", "AUTO_HANDLE", "Verification code blocking checkout"),
    116: ("other", "ESCALATE", "Confused/unclear fragment"),
    117: ("technical_issue", "AUTO_HANDLE", "Platform-specific playback issue"),
    118: ("general_feedback", "AUTO_HANDLE", "Positive feedback"),
    119: ("general_feedback", "ESCALATE", "Threatens to churn over algorithm complaint"),
    120: ("other", "AUTO_HANDLE", "Awaiting-response fragment"),
    121: ("other", "AUTO_HANDLE", "Version info fragment"),
    122: ("other", "ESCALATE", "Baggage-fee policy dispute"),
    123: ("other", "ESCALATE", "Contact info given, unspecified issue"),
    124: ("technical_issue", "ESCALATE", "Widespread post-update crashing"),
    125: ("other", "AUTO_HANDLE", "Content-availability question"),
    126: ("other", "ESCALATE", "Driver-side account issue, not a customer support case"),
    127: ("billing_refund", "ESCALATE", "Disputed cleaning-fee charge"),
    128: ("technical_issue", "AUTO_HANDLE", "Device-specific playback bug"),
    129: ("technical_issue", "AUTO_HANDLE", "Mapping/ETA inaccuracy"),
    130: ("other", "AUTO_HANDLE", "Content-availability question"),
    131: ("account_access", "ESCALATE", "Locked out, 2FA number outdated"),
    132: ("other", "ESCALATE", "Driver cancels after confirming destination"),
    133: ("other", "AUTO_HANDLE", "Contact-info request"),
    134: ("technical_issue", "ESCALATE", "Device-wide resetting bug, high value item"),
    135: ("other", "AUTO_HANDLE", "How-to/migration question"),
    136: ("EXCLUDE", "EXCLUDE", "Non-English (French)"),
    137: ("other", "AUTO_HANDLE", "Fare-rules policy question"),
    138: ("other", "AUTO_HANDLE", "Order-quantity policy question"),
    139: ("other", "ESCALATE", "Context-free 'I'm confused'"),
    140: ("technical_issue", "AUTO_HANDLE", "Metadata bug"),
    141: ("technical_issue", "AUTO_HANDLE", "Autocorrect bug"),
    142: ("order_status", "AUTO_HANDLE", "Delivery-schedule question"),
    143: ("billing_refund", "ESCALATE", "Unexpected extra charges"),
    144: ("technical_issue", "AUTO_HANDLE", "Autocorrect bug"),
    145: ("other", "AUTO_HANDLE", "Off-topic social request"),
    146: ("technical_issue", "AUTO_HANDLE", "Playback-state bug"),
    147: ("technical_issue", "AUTO_HANDLE", "Autocorrect bug"),
    148: ("billing_refund", "ESCALATE", "Wants refund for accidental subscription"),
    149: ("billing_refund", "ESCALATE", "Payment problem, needs region-specific support"),
    150: ("order_status", "ESCALATE", "Sarcastic escalation of delivery complaint"),
    151: ("general_feedback", "AUTO_HANDLE", "Positive feedback despite delay"),
    152: ("technical_issue", "AUTO_HANDLE", "Autocorrect/capitalization bug"),
    153: ("technical_issue", "AUTO_HANDLE", "App crash bug"),
    154: ("billing_refund", "ESCALATE", "Charged for ride in wrong city"),
    155: ("billing_refund", "ESCALATE", "ETA changed, threatened cancellation fee"),
    156: ("other", "ESCALATE", "Driver went off-route, safety-adjacent"),
    157: ("general_feedback", "AUTO_HANDLE", "Positive feedback"),
    158: ("technical_issue", "AUTO_HANDLE", "Multitasking bug"),
    159: ("shipping_delivery", "ESCALATE", "Order not received as promised"),
    160: ("EXCLUDE", "EXCLUDE", "Non-English (Spanish)"),
    161: ("general_feedback", "ESCALATE", "Complaint about staff conduct"),
    162: ("other", "AUTO_HANDLE", "Off-topic social tweet"),
    163: ("technical_issue", "AUTO_HANDLE", "Update-related glitch"),
    164: ("billing_refund", "ESCALATE", "Wrong card charged, confused process"),
    165: ("technical_issue", "ESCALATE", "Unresolved promo-code issue, repeated contact"),
    166: ("other", "ESCALATE", "Loyalty-program expiry policy complaint"),
    167: ("shipping_delivery", "ESCALATE", "Repeated delivery-address errors, forced cancellation"),
    168: ("general_feedback", "AUTO_HANDLE", "Positive feedback"),
    169: ("technical_issue", "AUTO_HANDLE", "Post-update battery drain"),
    170: ("other", "AUTO_HANDLE", "Context-free 'DMing now'"),
    171: ("other", "AUTO_HANDLE", "Plan-migration policy question"),
    172: ("general_feedback", "ESCALATE", "Frustrated, wants personal contact"),
    173: ("technical_issue", "AUTO_HANDLE", "Self-resolved bug report"),
    174: ("other", "ESCALATE", "Context-free device-model fragment"),
    175: ("security_fraud", "ESCALATE", "Unordered food delivered, customer flags as security issue"),
    176: ("general_feedback", "AUTO_HANDLE", "Positive feedback"),
    177: ("technical_issue", "AUTO_HANDLE", "Ride not showing in app"),
    178: ("account_access", "ESCALATE", "Account closed, wants reinstatement"),
    179: ("technical_issue", "AUTO_HANDLE", "Post-update freezing bug"),
    180: ("technical_issue", "ESCALATE", "Repeated bug, churn threat"),
    181: ("general_feedback", "AUTO_HANDLE", "Light humorous feedback"),
    182: ("technical_issue", "AUTO_HANDLE", "Autocorrect bug"),
    183: ("billing_refund", "ESCALATE", "Charged for a trip not taken"),
    184: ("billing_refund", "ESCALATE", "Promo not honored, unresponsive support"),
    185: ("other", "ESCALATE", "Context-free fragment"),
    186: ("technical_issue", "AUTO_HANDLE", "Autocorrect bug"),
    187: ("technical_issue", "ESCALATE", "Persistent connectivity bug, purchase threat"),
    188: ("general_feedback", "AUTO_HANDLE", "Positive feedback"),
    189: ("other", "ESCALATE", "Unexplained cancellation, no notice given"),
    190: ("technical_issue", "ESCALATE", "Update broke core function, churn threat"),
    191: ("other", "AUTO_HANDLE", "Partner-account linking question"),
    192: ("general_feedback", "ESCALATE", "General dissatisfaction complaint"),
    193: ("account_access", "ESCALATE", "No account, needs help starting one"),
    194: ("account_access", "ESCALATE", "Cannot access account, high frustration"),
    195: ("general_feedback", "AUTO_HANDLE", "Positive follow-up"),
    196: ("EXCLUDE", "EXCLUDE", "Non-English (German)"),
    197: ("other", "AUTO_HANDLE", "Context-free numeric fragment"),
    198: ("billing_refund", "ESCALATE", "Disputes a surcharge"),
    199: ("other", "AUTO_HANDLE", "Off-topic personal/emotional tweet"),
    200: ("technical_issue", "ESCALATE", "Device won't power on after hardware issue"),
    201: ("cancellation_request", "ESCALATE", "Wants to cancel but lost access to registered email"),
    202: ("technical_issue", "AUTO_HANDLE", "Recurring bug, wants urgent fix"),
    203: ("technical_issue", "AUTO_HANDLE", "Device/OS-specific playback bug"),
    204: ("general_feedback", "AUTO_HANDLE", "Positive feedback"),
    205: ("other", "AUTO_HANDLE", "Baggage-guarantee policy question"),
    206: ("security_fraud", "ESCALATE", "Reports a phishing site impersonating the brand"),
    207: ("other", "AUTO_HANDLE", "Baggage-allowance policy question"),
    208: ("account_access", "ESCALATE", "Account repeatedly disabled"),
    209: ("technical_issue", "AUTO_HANDLE", "Post-update hardware noise bug"),
    210: ("other", "AUTO_HANDLE", "Regional contact-number request"),
    211: ("other", "ESCALATE", "Context-free 'help me please'"),
    212: ("order_status", "AUTO_HANDLE", "Order status not visible on site"),
    213: ("security_fraud", "ESCALATE", "Reports a driver operating with mismatched plates"),
    214: ("technical_issue", "AUTO_HANDLE", "Playlist feature not showing"),
    215: ("other", "ESCALATE", "Context-free, distress-toned fragment"),
    216: ("billing_refund", "ESCALATE", "Refund marked successful but not received"),
    217: ("other", "AUTO_HANDLE", "Booking-change request (add infant)"),
    218: ("security_fraud", "ESCALATE", "Alleges repeated fraud via GPS jamming"),
    219: ("billing_refund", "ESCALATE", "Seeks compensation for cancelled-flight costs"),
}

EXCLUDED_NON_ENGLISH = [61, 72, 87, 92, 95, 101, 136, 160, 196]

CANCELLATION_KEYWORDS = ["cancel my", "cancel the", "cancel this", "want to cancel",
                          "please cancel", "how do i cancel", "unsubscribe", "cancel subscription"]


def find_supplemental_cancellations(test_df: pd.DataFrame, seed: int, n_needed: int) -> pd.DataFrame:
    mask = test_df["customer_message"].str.lower().apply(
        lambda t: any(k in t for k in CANCELLATION_KEYWORDS)
    )
    pool = test_df[mask]
    n = min(n_needed, len(pool))
    return pool.sample(n=n, random_state=seed)


def main():
    cfg = load_config()
    test_path = project_path(cfg["dataset"]["test_path"])
    df = pd.read_csv(test_path)

    candidates = pd.read_csv("/tmp/golden_candidates.csv")

    rows = []
    for idx, row in candidates.iterrows():
        if idx not in LABELS:
            continue
        intent, action, note = LABELS[idx]
        if intent == "EXCLUDE":
            continue
        rows.append(
            {
                "id": f"G{idx:04d}",
                "brand": row["brand"],
                "customer_message": row["customer_message"],
                "intent": intent,
                "expected_action": action,
                "label_source": cfg["golden_set"]["labelling_method"],
                "sampling_method": "stratified_random",
                "notes": note,
            }
        )

    golden = pd.DataFrame(rows)

    current_cancel = (golden["intent"] == "cancellation_request").sum()
    print(f"cancellation_request from random sample: {current_cancel}")
    if current_cancel < 15:
        already_used = set(golden["customer_message"])
        pool = df[~df["customer_message"].isin(already_used)]
        supp = find_supplemental_cancellations(pool, cfg["random_seed"], n_needed=40)
        print(f"Supplemental cancellation-keyword candidates found: {len(supp)}")
        supp.reset_index(drop=True).to_csv("/tmp/cancellation_candidates.csv", index=False)
        for i, r in supp.reset_index(drop=True).iterrows():
            print(f"{i:3d}|{r['brand'][:12]:12s}|{r['customer_message'][:150]}")

    out_path = project_path(cfg["golden_set"]["path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    golden.to_csv(out_path, index=False)

    print(f"\nGolden set size (before supplemental top-up): {len(golden)}")
    print(f"Excluded as non-English: {len(EXCLUDED_NON_ENGLISH)}")
    print("\nIntent coverage:")
    print(golden["intent"].value_counts())
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
