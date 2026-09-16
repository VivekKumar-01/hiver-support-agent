"""
Phase 16 — LLM-as-judge.

STATUS: NOT EXECUTED IN THIS PROJECT. This environment has no internet
access and no LLM API key, so no judge scores exist anywhere in
outputs/. Running main() without a key will raise a clear error rather
than silently returning fake scores — see NON-NEGOTIABLE RULES in the
brief ("never fabricate ... LLM judge scores").

If you have an ANTHROPIC_API_KEY, add it to a .env file at the project
root and run:

    python -m src.evaluation.llm_judge

This scores every row in outputs/metrics/predictions_detailed.csv on
correctness, relevance, helpfulness, groundedness, brand_consistency,
and safety (1-5 each) and saves outputs/metrics/llm_judge_scores.json.
"""
import json
import os

import pandas as pd

from src.utils.config import project_path

JUDGE_PROMPT = """You are an evaluation judge for a customer support AI agent.
Score the generated reply on a 1-5 scale for each dimension below.
Return ONLY valid JSON, no other text.

Customer message: {message}
Brand: {brand}
Generated reply: {reply}

Score these dimensions (1=very poor, 5=excellent):
- correctness: is the reply factually consistent with what a support agent should say?
- relevance: does it address the customer's actual message?
- helpfulness: does it move the customer toward resolution?
- groundedness: does it avoid inventing facts not implied by typical support context?
- brand_consistency: does the tone fit a professional support brand?
- safety: does it avoid risky promises (refunds, legal, etc.) it shouldn't make?

Return JSON: {{"correctness": int, "relevance": int, "helpfulness": int,
"groundedness": int, "brand_consistency": int, "safety": int,
"overall": float, "reason": "one sentence"}}"""


def judge_reply(message: str, brand: str, reply: str, api_key: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = JUDGE_PROMPT.format(message=message, brand=brand, reply=reply)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return json.loads(text)


def main():
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No ANTHROPIC_API_KEY found in environment/.env. "
            "LLM-as-judge cannot run without a real API key, and this project "
            "does not fabricate judge scores. Add a key to .env and rerun."
        )

    df = pd.read_csv(project_path("outputs/metrics/predictions_detailed.csv"))
    scores = []
    for _, row in df.iterrows():
        s = judge_reply(row["customer_message"], row["brand"], row["generated_reply"], api_key)
        scores.append(s)

    out_path = project_path("outputs/metrics/llm_judge_scores.json")
    with open(out_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Judged {len(scores)} replies -> {out_path}")


if __name__ == "__main__":
    main()
