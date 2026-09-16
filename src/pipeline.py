"""
Phase 14 — End-to-end pipeline.

    run_agent(customer_message, brand) -> dict

This is the single callable interface everything else (CLI, evaluation
harness, tests) is built on.
"""
from dataclasses import asdict

from src.classification.tfidf_lr import load_model
from src.retrieval.retriever import retrieve_similar_cases
from src.generation.generator import generate_reply, _format_evidence
from src.generation.safety import check_grounding
from src.escalation.policy import decide_escalation
from src.utils.config import load_config

_CFG = None
_MODEL = None


def _get_cfg():
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model(_get_cfg())
    return _MODEL


def run_agent(customer_message: str, brand: str) -> dict:
    cfg = _get_cfg()
    model = _get_model()

    # 1. Intent classification + confidence
    intent = model.predict([customer_message])[0]
    confidence = float(model.predict_proba([customer_message]).max())

    # 2. Historical retrieval (same-brand prioritized)
    retrieved = retrieve_similar_cases(customer_message, brand, k=cfg["retrieval"]["top_k"])
    top_similarity = retrieved[0].similarity if retrieved else 0.0

    # 3. Escalation decision (uses confidence + retrieval quality + risk signals)
    escalation = decide_escalation(customer_message, intent, confidence, top_similarity, cfg)

    # 4. Grounded reply generation
    if escalation.decision == "ESCALATE":
        reply = (
            f"Thanks for reaching out to {brand} support — this has been routed to a "
            "human agent who will follow up shortly."
        )
        grounding = {"is_grounded": True, "unsupported_claims": []}  # no factual claims made
    else:
        reply = generate_reply(customer_message, brand, intent, confidence, retrieved, cfg)
        evidence_text = _format_evidence(retrieved)
        grounding = check_grounding(reply, evidence_text)
        # Phase 13: if generation made an unsupported claim, override to escalate
        if not grounding["is_grounded"]:
            escalation.decision = "ESCALATE"
            escalation.reason = (
                f"Generated reply contained unsupported claims {grounding['unsupported_claims']}; "
                "escalated by safety check rather than sent as-is."
            )
            escalation.triggered_rule = "grounding_check_failed"
            reply = (
                f"Thanks for reaching out to {brand} support — this has been routed to a "
                "human agent to make sure you get an accurate answer."
            )

    return {
        "customer_message": customer_message,
        "brand": brand,
        "predicted_intent": intent,
        "confidence": round(confidence, 4),
        "retrieved_cases": [asdict(r) for r in retrieved],
        "generated_reply": reply,
        "decision": escalation.decision,
        "escalation_reason": escalation.reason,
        "triggered_rule": escalation.triggered_rule,
        "grounding": grounding,
    }


if __name__ == "__main__":
    result = run_agent("My refund hasn't arrived yet, it's been over a week.", "AmazonHelp")
    import json

    print(json.dumps(result, indent=2, default=str))
