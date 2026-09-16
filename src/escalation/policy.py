"""
Phase 12 — Escalation policy.

Deliberately simple and conservative (see DECISION_LOG.md D9, D11):
any one red flag is enough to escalate. We would rather over-escalate
(a human looks at something that was actually fine) than
under-escalate (the bot auto-answers something risky) — see
FAILURE_ANALYSIS.md / REPORT.md "false auto-handling" discussion.
"""
from dataclasses import dataclass

from src.utils.config import load_config


@dataclass
class EscalationDecision:
    decision: str          # "AUTO_HANDLE" or "ESCALATE"
    reason: str
    triggered_rule: str


def decide_escalation(
    message: str,
    predicted_intent: str,
    confidence: float,
    top_similarity: float,
    cfg: dict = None,
) -> EscalationDecision:
    cfg = cfg or load_config()
    esc_cfg = cfg["escalation"]
    ret_cfg = cfg["retrieval"]

    text_lower = message.lower()

    # Rule 1: explicit risk keywords (security/fraud/legal)
    hit_keywords = [kw for kw in esc_cfg["risk_keywords"] if kw in text_lower]
    if hit_keywords:
        return EscalationDecision(
            "ESCALATE",
            f"Risk keyword(s) detected: {', '.join(hit_keywords)}.",
            "risk_keyword",
        )

    # Rule 2: intent itself is inherently high-risk
    if predicted_intent == "security_fraud":
        return EscalationDecision(
            "ESCALATE", "Predicted intent is security/fraud related.", "high_risk_intent"
        )

    # Rule 3: low classifier confidence
    if confidence < esc_cfg["min_confidence_for_auto_handle"]:
        return EscalationDecision(
            "ESCALATE",
            f"Classifier confidence {confidence:.2f} is below threshold "
            f"{esc_cfg['min_confidence_for_auto_handle']:.2f}.",
            "low_confidence",
        )

    # Rule 4: poor retrieval evidence -> nothing grounded to answer with
    if top_similarity < ret_cfg["min_similarity_for_grounding"]:
        action = esc_cfg["poor_retrieval_action"]
        return EscalationDecision(
            action,
            f"Best retrieved similarity {top_similarity:.2f} is below grounding threshold "
            f"{ret_cfg['min_similarity_for_grounding']:.2f}; not enough historical evidence to answer safely.",
            "poor_retrieval",
        )

    # Rule 5: financially-sensitive intents are escalated conservatively
    if predicted_intent in ("billing_refund", "account_access"):
        return EscalationDecision(
            "ESCALATE",
            f"Intent '{predicted_intent}' involves account or financial impact; "
            "escalated by conservative policy rather than auto-actioned.",
            "sensitive_intent_policy",
        )

    # Rule 6 (added after real-data evaluation — see DECISION_LOG.md D18):
    # an "other" prediction means the classifier could not map the message
    # to any known category. Auto-handling an UNKNOWN intent is unsafe by
    # construction — on real data this was the single largest source of
    # false auto-handles (misclassified security_fraud/billing_refund
    # messages landing in "other" with spuriously high confidence).
    if predicted_intent == "other":
        return EscalationDecision(
            "ESCALATE",
            "Predicted intent is 'other' (unclassified) — escalated rather than "
            "auto-handling an unknown request type.",
            "unclassified_intent_policy",
        )

    return EscalationDecision(
        "AUTO_HANDLE",
        "High confidence, low risk, and strong historical evidence.",
        "default_auto_handle",
    )
