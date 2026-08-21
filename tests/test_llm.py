"""Test für llm.py — LLMClient mit gemocktem requests (kein echtes Netz).

Mock-Strategie: monkeypatch von concilium.llm.requests.post,
damit kein echter HTTP-Call stattfindet.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.llm import LLMClient  # noqa: E402


class _MockResponse:
    """Mock für requests.Response."""

    def __init__(self, status_code: int, content: str = "Test-Antwort"):
        self.status_code = status_code
        self._content = content
        self.text = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")  # noqa: TRY002

    def json(self):
        return {
            "choices": [
                {"message": {"content": self._content}}
            ]
        }


class TestLLMClient:
    """Tests für LLMClient.chat()."""

    def test_chat_returns_text(self):
        """chat() soll den Text aus der choices-Response zurückgeben."""
        client = LLMClient(base_url="http://fake:8080/v1", api_key="test-key", model="test-model")

        mock_resp = _MockResponse(200, "Hallo vom LLM")
        with patch("concilium.llm.requests.post", return_value=mock_resp) as mock_post:
            result = client.chat([{"role": "user", "content": "Hallo"}])

        assert result == "Hallo vom LLM"
        mock_post.assert_called_once()

        # Prüfe, dass der Payload korrekt ist
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        # payload ist positional oder keyword — je nach mock
        if "json" in call_kwargs[1]:
            payload = call_kwargs[1]["json"]
        else:
            # positional args: (url, json=payload)
            payload = call_kwargs[1].get("json", {})
        assert payload["model"] == "test-model"
        assert payload["messages"] == [{"role": "user", "content": "Hallo"}]

    def test_retry_on_500(self):
        """Bei HTTP 500 soll der Client retry-en und beim Erfolg zurückgeben."""
        client = LLMClient(base_url="http://fake:8080/v1", api_key="test-key", model="test-model")

        # Erste Anfrage: 500, zweite Anfrage: 200
        responses = [_MockResponse(500, "Server Error"), _MockResponse(200, "Erfolg nach Retry")]

        with patch("concilium.llm.requests.post", side_effect=responses):
            with patch("concilium.llm.time.sleep"):  # Backoff beschleunigen
                result = client.chat([{"role": "user", "content": "Test"}])

        assert result == "Erfolg nach Retry"

    def test_retry_on_429(self):
        """Bei HTTP 429 (rate limit) soll ebenfalls retry-en."""
        client = LLMClient(base_url="http://fake:8080/v1", api_key="test-key", model="test-model")

        responses = [_MockResponse(429, "Rate limited"), _MockResponse(200, "OK nach 429")]

        with patch("concilium.llm.requests.post", side_effect=responses):
            with patch("concilium.llm.time.sleep"):
                result = client.chat([{"role": "user", "content": "Test"}])

        assert result == "OK nach 429"

    def test_all_retries_fail_raises(self):
        """Wenn alle Retries fehlschlagen, soll RuntimeError geworfen werden."""
        client = LLMClient(base_url="http://fake:8080/v1", api_key="test-key", model="test-model")

        responses = [_MockResponse(500, "Error")] * 3  # 1 + 2 retries

        with patch("concilium.llm.requests.post", side_effect=responses):
            with patch("concilium.llm.time.sleep"):
                with pytest.raises(RuntimeError, match="fehlgeschlagen"):
                    client.chat([{"role": "user", "content": "Test"}])

    def test_timeout_retry(self):
        """Bei Timeout soll retry-en."""
        import requests as real_requests

        client = LLMClient(base_url="http://fake:8080/v1", api_key="test-key", model="test-model")

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise real_requests.Timeout("Timeout!")
            return _MockResponse(200, "Erfolg nach Timeout")

        with patch("concilium.llm.requests.post", side_effect=side_effect):
            with patch("concilium.llm.time.sleep"):
                result = client.chat([{"role": "user", "content": "Test"}])

        assert result == "Erfolg nach Timeout"

    def test_env_defaults(self):
        """LLMClient soll Defaults aus Umgebungsvariablen ziehen."""
        os.environ["LLM_BASE_URL"] = "http://env-test:9999/v1"
        os.environ["LLM_API_KEY"] = "env-key-123"
        os.environ["LLM_MODEL"] = "env-model"

        client = LLMClient()
        assert client.base_url == "http://env-test:9999/v1"
        assert client.api_key == "env-key-123"
        assert client.model == "env-model"

        # Cleanup
        del os.environ["LLM_BASE_URL"]
        del os.environ["LLM_API_KEY"]
        del os.environ["LLM_MODEL"]


class TestParseJson:
    """Tests für parse_json Helper aus agents.py."""

    def test_valid_json(self):
        from concilium.agents import parse_json

        result = parse_json('{"key": "value", "score": 3}')
        assert result["key"] == "value"
        assert result["score"] == 3

    def test_json_in_text(self):
        from concilium.agents import parse_json

        result = parse_json('Hier ist meine Antwort: {"key": "value"} das war es.')
        assert result["key"] == "value"

    def test_json_in_codeblock(self):
        from concilium.agents import parse_json

        result = parse_json('```json\n{"key": "value", "score": 5}\n```')
        assert result["key"] == "value"
        assert result["score"] == 5

    def test_no_json_returns_raw(self):
        from concilium.agents import parse_json

        result = parse_json("Das ist kein JSON.")
        assert "_raw" in result
        assert result["_raw"] == "Das ist kein JSON."
