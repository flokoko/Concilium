"""Tests für LLM-Modell-Fallback (Aufgabe 3).

Mock-Strategie: monkeypatch von concilium.llm.requests.post, sodass kein
echter HTTP-Call stattfindet.
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
        return {"choices": [{"message": {"content": self._content}}]}


class TestFallbackModel:
    """Tests für das Fallback-Modell bei 429/5xx-Erschöpfung."""

    def test_fallback_after_429_exhaustion(self):
        """Bei dauerhaftem 429 wechselt der Client auf das Fallback-Modell."""
        client = LLMClient(
            base_url="http://fake:8080/v1",
            api_key="test-key",
            model="primary-model",
            fallback_model="fallback-model",
        )

        # 3x 429 für Primärmodell, dann 200 für Fallback
        responses = [
            _MockResponse(429, "Rate limited"),
            _MockResponse(429, "Rate limited"),
            _MockResponse(429, "Rate limited"),
            _MockResponse(200, "Erfolg vom Fallback"),
        ]

        with patch("concilium.llm.requests.post", side_effect=responses) as mock_post:
            with patch("concilium.llm.time.sleep"):
                result = client.chat([{"role": "user", "content": "Test"}])

        assert result == "Erfolg vom Fallback"

        # Prüfe: 4 Calls total (3 primär + 1 fallback)
        assert mock_post.call_count == 4

        # Der 4. Call sollte das Fallback-Modell enthalten
        last_call_kwargs = mock_post.call_args_list[3]
        payload = last_call_kwargs[1]["json"] if "json" in last_call_kwargs[1] else last_call_kwargs[0][1]
        if "json" not in last_call_kwargs[1]:
            payload = last_call_kwargs[1].get("json", {})
        assert payload["model"] == "fallback-model"

    def test_fallback_after_500_exhaustion(self):
        """Bei dauerhaftem 500 wechselt der Client auf das Fallback-Modell."""
        client = LLMClient(
            base_url="http://fake:8080/v1",
            api_key="test-key",
            model="primary-model",
            fallback_model="fallback-model",
        )

        # 3x 500 für Primärmodell, dann 200 für Fallback
        responses = [
            _MockResponse(500, "Server Error"),
            _MockResponse(500, "Server Error"),
            _MockResponse(500, "Server Error"),
            _MockResponse(200, "Erfolg vom Fallback"),
        ]

        with patch("concilium.llm.requests.post", side_effect=responses):
            with patch("concilium.llm.time.sleep"):
                result = client.chat([{"role": "user", "content": "Test"}])

        assert result == "Erfolg vom Fallback"

    def test_no_fallback_without_fallback_model(self):
        """Ohne fallback_model → kein Wechsel, RuntimeError wie vorher."""
        client = LLMClient(
            base_url="http://fake:8080/v1",
            api_key="test-key",
            model="primary-model",
        )
        assert client.fallback_model is None

        # Alle 3 Versuche 429 → RuntimeError
        responses = [_MockResponse(429, "Rate limited")] * 3

        with patch("concilium.llm.requests.post", side_effect=responses):
            with patch("concilium.llm.time.sleep"):
                with pytest.raises(RuntimeError, match="fehlgeschlagen"):
                    client.chat([{"role": "user", "content": "Test"}])

    def test_fallback_also_fails_raises_combined_error(self):
        """Wenn sowohl Primär- als auch Fallback-Modell fehlschlagen → RuntimeError."""
        client = LLMClient(
            base_url="http://fake:8080/v1",
            api_key="test-key",
            model="primary-model",
            fallback_model="fallback-model",
        )

        # Alle Versuche 429 (primär + fallback)
        responses = [_MockResponse(429, "Rate limited")] * 6

        with patch("concilium.llm.requests.post", side_effect=responses):
            with patch("concilium.llm.time.sleep"):
                with pytest.raises(RuntimeError, match="fehlgeschlagen"):
                    client.chat([{"role": "user", "content": "Test"}])

    def test_env_fallback_model(self):
        """LLM_FALLBACK_MODEL Umgebungsvariable wird gelesen."""
        os.environ["LLM_FALLBACK_MODEL"] = "env-fallback-model"
        client = LLMClient(base_url="http://fake:8080/v1", api_key="test-key", model="primary")
        assert client.fallback_model == "env-fallback-model"
        del os.environ["LLM_FALLBACK_MODEL"]

    def test_env_fallback_model_empty_means_none(self):
        """Leere LLM_FALLBACK_MODEL → None (kein Fallback)."""
        os.environ["LLM_FALLBACK_MODEL"] = ""
        client = LLMClient(base_url="http://fake:8080/v1", api_key="test-key", model="primary")
        assert client.fallback_model is None
        del os.environ["LLM_FALLBACK_MODEL"]

    def test_no_fallback_on_timeout(self):
        """Bei Timeout wird KEIN Fallback aktiviert (nur bei 429/5xx)."""
        import requests as real_requests

        client = LLMClient(
            base_url="http://fake:8080/v1",
            api_key="test-key",
            model="primary-model",
            fallback_model="fallback-model",
        )

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            raise real_requests.Timeout("Timeout!")

        with patch("concilium.llm.requests.post", side_effect=side_effect):
            with patch("concilium.llm.time.sleep"):
                with pytest.raises(RuntimeError, match="Timeout"):
                    client.chat([{"role": "user", "content": "Test"}])

        # Nur 3 Versuche (alle Timeout) — kein 4. Versuch mit Fallback
        assert call_count[0] == 3
