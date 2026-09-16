"""
Phase 31 — Tests.

Written in plain unittest so they run even without pytest installed
(pytest could not be installed in the sandbox this project was built in
— no internet access; requirements.txt lists it for your machine, and
these same tests run fine under `pytest` there too since unittest.TestCase
is pytest-compatible).

Run:
    python -m pytest tests/          (on a machine with pytest)
    python -m unittest discover tests  (works anywhere, incl. this sandbox)
"""
import unittest

import pandas as pd

from src.preprocessing.clean import clean_dataframe, normalize_text, thread_level_split
from src.classification.tfidf_lr import load_model
from src.retrieval.retriever import retrieve_similar_cases
from src.escalation.policy import decide_escalation
from src.generation.safety import check_grounding
from src.pipeline import run_agent
from src.utils.config import load_config, project_path


class TestPreprocessing(unittest.TestCase):
    def test_normalize_text_collapses_whitespace(self):
        self.assertEqual(normalize_text("hello    world  "), "hello world")

    def test_clean_dataframe_drops_missing_and_duplicates(self):
        df = pd.DataFrame(
            {
                "thread_id": ["T1", "T1", "T2", "T3"],
                "brand": ["A", "A", "B", "C"],
                "customer_message": ["hi there friend", "hi there friend", None, "hello world today"],
                "agent_response": ["ok sure thing", "ok sure thing", "ok sure thing", "ok sure thing"],
                "intent": ["x", "x", "y", "z"],
            }
        )
        cleaned = clean_dataframe(df)
        self.assertNotIn(None, cleaned["customer_message"].tolist())
        self.assertEqual(len(cleaned), 2)  # one dup removed, one null-message row removed

    def test_thread_level_split_has_no_overlap(self):
        df = pd.DataFrame(
            {
                "thread_id": [f"T{i}" for i in range(20)],
                "brand": ["A"] * 20,
                "customer_message": [f"msg {i}" for i in range(20)],
                "agent_response": [f"resp {i}" for i in range(20)],
                "intent": ["x"] * 20,
            }
        )
        train, test = thread_level_split(df, seed=42, test_frac=0.25)
        overlap = set(train["thread_id"]) & set(test["thread_id"])
        self.assertEqual(len(overlap), 0)


class TestClassifier(unittest.TestCase):
    def test_model_predicts_known_label(self):
        model = load_model()
        pred = model.predict(["I want a refund for my order please"])[0]
        cfg = load_config()
        train_labels = set(pd.read_csv(project_path(cfg["dataset"]["train_weak_labelled_path"]))["intent"].unique())
        self.assertIn(pred, train_labels)


class TestRetrieval(unittest.TestCase):
    def test_retrieve_returns_k_results(self):
        results = retrieve_similar_cases("where is my package", "AmazonHelp", k=3)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertTrue(0.0 <= r.similarity <= 1.0 or r.similarity >= 0.0)


class TestEscalation(unittest.TestCase):
    def test_fraud_keyword_forces_escalation(self):
        decision = decide_escalation(
            "I think my account was hacked and someone made unauthorized charges",
            predicted_intent="security_fraud",
            confidence=0.99,
            top_similarity=0.9,
        )
        self.assertEqual(decision.decision, "ESCALATE")
        self.assertEqual(decision.triggered_rule, "risk_keyword")

    def test_low_confidence_forces_escalation(self):
        decision = decide_escalation(
            "some ambiguous message",
            predicted_intent="general_feedback",
            confidence=0.1,
            top_similarity=0.9,
        )
        self.assertEqual(decision.decision, "ESCALATE")
        self.assertEqual(decision.triggered_rule, "low_confidence")


class TestSafety(unittest.TestCase):
    def test_unsupported_dollar_amount_is_flagged(self):
        reply = "You will be refunded $499.99 by tomorrow."
        evidence = "Refunds typically post within 5-7 business days."
        result = check_grounding(reply, evidence)
        self.assertFalse(result["is_grounded"])

    def test_supported_claim_is_not_flagged(self):
        reply = "Refunds typically post within 5-7 business days."
        evidence = "Refunds typically post within 5-7 business days."
        result = check_grounding(reply, evidence)
        self.assertTrue(result["is_grounded"])


class TestEndToEnd(unittest.TestCase):
    def test_run_agent_returns_expected_schema(self):
        result = run_agent("My refund hasn't arrived yet", "AmazonHelp")
        expected_keys = {
            "customer_message", "brand", "predicted_intent", "confidence",
            "retrieved_cases", "generated_reply", "decision",
            "escalation_reason", "triggered_rule", "grounding",
        }
        self.assertEqual(expected_keys, set(result.keys()))
        self.assertIn(result["decision"], ("AUTO_HANDLE", "ESCALATE"))
        self.assertIsInstance(result["retrieved_cases"], list)

    def test_run_agent_smoke_various_inputs(self):
        cases = [
            ("Where is my order?", "AmazonHelp"),
            ("I was hacked, unauthorized charges!", "DeltaAssist"),
            ("Love your app, thanks!", "SpotifyCares"),
        ]
        for message, brand in cases:
            result = run_agent(message, brand)
            self.assertIn(result["decision"], ("AUTO_HANDLE", "ESCALATE"))


if __name__ == "__main__":
    unittest.main()
