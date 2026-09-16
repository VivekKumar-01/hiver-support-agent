"""
Phase 13 — Grounding / safety checks.

Simple, explainable check: scan the generated reply for claims about
refunds/dates/account status that DON'T appear anywhere in the retrieved
evidence. This is a heuristic, not a perfect fact-checker — documented
as a limitation in README.md.
"""
import re

CLAIM_PATTERNS = {
    "specific_dollar_amount": r"\$\d+(\.\d{2})?",
    "specific_date": r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b",
    "order_number": r"\border\s*#?\s*\d{5,}\b",
}


def check_grounding(reply: str, evidence_text: str) -> dict:
    """Returns which claim types appear in the reply but not in the evidence."""
    unsupported = []
    for claim_type, pattern in CLAIM_PATTERNS.items():
        reply_matches = set(re.findall(pattern, reply.lower()))
        evidence_matches = set(re.findall(pattern, evidence_text.lower()))
        # only flag matches present in reply that are NOT backed by evidence
        unbacked = reply_matches - evidence_matches
        if unbacked:
            unsupported.append({"claim_type": claim_type, "values": list(unbacked)})

    return {"is_grounded": len(unsupported) == 0, "unsupported_claims": unsupported}
