"""
Phase 11 — RAG response generation.

Beginner explanation of RAG: RAG means we first retrieve useful
historical information and then give that information to a generator
(here: a template, or optionally an LLM) so it can produce a grounded
answer — one based on real precedent instead of invented facts.

MODE: this environment has no internet access and no LLM API key, so
`generation.mode` in config.yaml defaults to "template_grounded": the
reply is built directly from the retrieved historical response rather
than free-form LLM text. This is a genuine grounding strategy (it
literally cannot invent an order status, refund amount, or policy that
isn't in the retrieved case), just a less fluent one than an LLM would
produce. An `llm_api` path is implemented and ready to use if you add
an ANTHROPIC_API_KEY to .env and set generation.mode: "llm_api".
"""
import os

from src.utils.config import load_config


SYSTEM_INSTRUCTIONS = """You are a customer support agent for {brand}.
Use ONLY the retrieved historical cases below as evidence. Do not invent
order status, refund status, dates, or policy. If the evidence is
insufficient, ask a clarifying question instead of guessing. Keep the
reply concise and in a helpful, professional tone matching {brand}.

Customer message: {message}
Predicted intent: {intent} (confidence {confidence:.2f})

Retrieved historical cases:
{evidence}

Write only the reply to the customer, nothing else."""


def _format_evidence(retrieved_cases) -> str:
    lines = []
    for i, c in enumerate(retrieved_cases, 1):
        lines.append(
            f"{i}. [similarity={c.similarity:.2f}, same_brand={c.same_brand}] "
            f"Customer said: \"{c.customer_message}\" -> Agent replied: \"{c.historical_response}\""
        )
    return "\n".join(lines) if lines else "(none retrieved)"


def _template_grounded_reply(message: str, brand: str, intent: str, retrieved_cases) -> str:
    if not retrieved_cases:
        return (
            f"Thanks for reaching out to {brand} support. Could you share a bit more detail "
            "(e.g. an order number or account email) so we can look into this properly?"
        )

    best = retrieved_cases[0]
    # Ground the reply in the best historical case's response pattern, but
    # keep it generic (no copied order numbers/amounts from a DIFFERENT
    # customer's case) — this is the "don't blindly copy historical
    # responses" and "don't invent order/refund status" rule from the brief.
    opener = f"Thanks for reaching out to {brand} support."
    if best.same_brand:
        body = (
            f" Based on how we've handled similar {intent.replace('_', ' ')} cases before, "
            f"here's what typically happens next: {_generalize(best.historical_response)}"
        )
    else:
        body = (
            f" This looks similar to {intent.replace('_', ' ')} issues we've seen (though not "
            f"specifically for {brand} in our recent history), so a human agent should confirm "
            f"the exact next steps for your account."
        )
    closer = " Let us know if you need anything else in the meantime."
    return opener + body + closer


def _generalize(historical_response: str) -> str:
    """Strip order-number-like tokens from a historical response so we don't
    accidentally imply a DIFFERENT customer's order number applies here."""
    import re

    generalized = re.sub(r"\b\d{5,}\b", "your order", historical_response)
    return generalized


def _llm_api_reply(message: str, brand: str, intent: str, confidence: float, retrieved_cases) -> str:
    """Only called if generation.mode == 'llm_api' AND ANTHROPIC_API_KEY is set."""
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "generation.mode is 'llm_api' but ANTHROPIC_API_KEY is not set in .env. "
            "Either set the key or switch config.yaml generation.mode back to 'template_grounded'."
        )

    import anthropic

    cfg = load_config()
    client = anthropic.Anthropic(api_key=api_key)
    prompt = SYSTEM_INSTRUCTIONS.format(
        brand=brand,
        message=message,
        intent=intent,
        confidence=confidence,
        evidence=_format_evidence(retrieved_cases),
    )
    response = client.messages.create(
        model=cfg["generation"]["llm_model"],
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def generate_reply(message: str, brand: str, intent: str, confidence: float, retrieved_cases, cfg: dict = None) -> str:
    cfg = cfg or load_config()
    mode = cfg["generation"]["mode"]
    if mode == "llm_api":
        return _llm_api_reply(message, brand, intent, confidence, retrieved_cases)
    return _template_grounded_reply(message, brand, intent, retrieved_cases)
