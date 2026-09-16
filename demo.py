"""
Phase 32 — CLI demo.

Run:
    python demo.py

Uses the real pipeline (src.pipeline.run_agent) — no hardcoded output.
"""
import json

from src.pipeline import run_agent


def main():
    print("=" * 60)
    print("Hiver Support Agent — CLI Demo")
    print("=" * 60)

    brand = input("\nBrand (e.g. AmazonHelp, AppleSupport, SpotifyCares, DeltaAssist, UberSupport): ").strip()
    if not brand:
        brand = "AmazonHelp"
        print(f"(no brand entered, defaulting to {brand})")

    message = input("Customer message: ").strip()
    if not message:
        message = "My refund hasn't arrived yet, it's been over a week."
        print(f"(no message entered, using example: \"{message}\")")

    result = run_agent(message, brand)

    print("\n--- Intent ---")
    print(result["predicted_intent"])

    print("\n--- Confidence ---")
    print(f"{result['confidence']:.2f}")

    print("\n--- Retrieved historical cases ---")
    for i, case in enumerate(result["retrieved_cases"], 1):
        print(f"{i}. sim={case['similarity']:.2f} same_brand={case['same_brand']} "
              f"[{case['brand']}] \"{case['customer_message']}\" -> \"{case['historical_response']}\"")

    print("\n--- Suggested reply ---")
    print(result["generated_reply"])

    print("\n--- Decision ---")
    print(result["decision"])

    print("\n--- Reason ---")
    print(result["escalation_reason"])

    print("\n(full structured output)")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
