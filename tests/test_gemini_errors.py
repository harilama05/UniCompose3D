"""Regression tests for the diagnostic lost on persistent Gemini HTTP failures."""
import unittest
from unittest.mock import Mock, patch

from unicompose3d.clients import Gemini, ProviderError
from unicompose3d.schemas import PlacementPlan


class GeminiErrorTests(unittest.TestCase):
    def failure(self, code, payload=None, non_json=False):
        transport = Mock()
        response = transport.post.return_value
        response.status_code = code
        response.json.return_value = payload
        if non_json:
            response.json.side_effect = ValueError("not JSON")
        client = Gemini("private-test-key", "test-model", transport)
        with self.assertRaises(ProviderError) as caught:
            client._generate([], PlacementPlan)
        self.assertEqual(transport.post.call_count, 1)
        return str(caught.exception)

    def test_503_preserves_google_reason_and_model(self):
        message = self.failure(503, {"error": {"status": "UNAVAILABLE", "message": "This model is currently experiencing high demand."}})
        self.assertIn("UNAVAILABLE", message)
        self.assertIn("high demand", message)
        self.assertIn("test-model", message)
        self.assertNotIn("Check the API key", message)

    def test_429_preserves_distinction_between_rate_and_daily_quota(self):
        message = self.failure(429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Daily request quota exceeded."}})
        self.assertIn("Daily request quota exceeded", message)
        self.assertIn("429", message)

    def test_provider_message_redacts_explicit_and_environment_keys(self):
        with patch.dict("os.environ", {"TRIPO_API_KEY": "other-private-key"}):
            message = self.failure(403, {"error": {"message": "private-test-key other-private-key https://example.com/?key=hidden"}})
        self.assertNotIn("private-test-key", message)
        self.assertNotIn("other-private-key", message)
        self.assertNotIn("key=hidden", message)

    def test_non_json_error_does_not_echo_html_body(self):
        message = self.failure(503, non_json=True)
        self.assertIn("503", message)
        self.assertIn("did not return a detailed JSON diagnostic", message)

    def test_unexpected_error_shape_still_reports_http(self):
        for payload in (None, [], {"error": "unexpected"}):
            with self.subTest(payload=payload):
                self.assertIn("404", self.failure(404, payload))


if __name__ == "__main__":
    unittest.main()
