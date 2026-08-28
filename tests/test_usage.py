"""Test für usage.py und Token-Usage-Logging im LLMClient.

Tests:
1. record_usage schreibt CSV-Zeile + Header.
2. record_usage mit usage=None tut nichts.
3. summarize_usage aggregiert mehrere Zeilen korrekt.
4. summarize_usage mit fehlender Datei → leeres dict.
5. LLMClient.chat setzt total_usage kumulativ.
6. CLI: --usage parst und ruft summarize_usage auf.
"""

from __future__ import annotations

import csv
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.usage import record_usage, summarize_usage  # noqa: E402

# ---------------------------------------------------------------------------
# 1. record_usage schreibt CSV-Zeile + Header
# ---------------------------------------------------------------------------


class TestRecordUsage:
    """Tests für record_usage."""

    def test_record_usage_writes_row_and_header(self, tmp_path):
        """record_usage schreibt eine CSV-Zeile und legt den Header an."""
        usage_file = str(tmp_path / "usage" / "usage.csv")
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        record_usage("AAPL", usage, usage_file=usage_file)

        assert os.path.isfile(usage_file)
        with open(usage_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["prompt_tokens"] == "100"
        assert rows[0]["completion_tokens"] == "50"
        assert rows[0]["total_tokens"] == "150"
        assert rows[0]["timestamp"]  # nicht leer

    def test_record_usage_none_does_nothing(self, tmp_path):
        """record_usage mit usage=None schreibt nichts."""
        usage_file = str(tmp_path / "usage" / "usage.csv")
        record_usage("AAPL", None, usage_file=usage_file)

        assert not os.path.isfile(usage_file)

    def test_record_usage_no_total_tokens_does_nothing(self, tmp_path):
        """record_usage ohne total_tokens schreibt nichts."""
        usage_file = str(tmp_path / "usage" / "usage.csv")
        record_usage("AAPL", {"prompt_tokens": 100}, usage_file=usage_file)

        assert not os.path.isfile(usage_file)

    def test_record_usage_appends_existing_file(self, tmp_path):
        """record_usage appended an eine bestehende Datei (Header nicht erneut)."""
        usage_file = str(tmp_path / "usage.csv")
        usage1 = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        usage2 = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}

        record_usage("AAPL", usage1, usage_file=usage_file)
        record_usage("MSFT", usage2, usage_file=usage_file)

        with open(usage_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["ticker"] == "AAPL"
        assert rows[1]["ticker"] == "MSFT"


# ---------------------------------------------------------------------------
# 3. summarize_usage aggregiert korrekt
# ---------------------------------------------------------------------------


class TestSummarizeUsage:
    """Tests für summarize_usage."""

    def test_summarize_multiple_rows(self, tmp_path):
        """summarize_usage aggregiert mehrere Zeilen korrekt."""
        usage_file = str(tmp_path / "usage.csv")
        record_usage("AAPL", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                      usage_file=usage_file)
        record_usage("AAPL", {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
                      usage_file=usage_file)
        record_usage("MSFT", {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
                      usage_file=usage_file)

        summary = summarize_usage(usage_file=usage_file)

        assert summary["anzahl_calls"] == 3
        assert summary["summe_prompt_tokens"] == 350
        assert summary["summe_completion_tokens"] == 175
        assert summary["summe_total_tokens"] == 525
        assert summary["anzahl_ticker"] == 2
        assert summary["ticker_tokens"]["AAPL"] == 450
        assert summary["ticker_tokens"]["MSFT"] == 75

    def test_summarize_missing_file_returns_empty(self, tmp_path):
        """summarize_usage mit fehlender Datei → leeres dict mit Nullen."""
        usage_file = str(tmp_path / "does_not_exist.csv")
        summary = summarize_usage(usage_file=usage_file)

        assert summary["anzahl_calls"] == 0
        assert summary["summe_prompt_tokens"] == 0
        assert summary["summe_completion_tokens"] == 0
        assert summary["summe_total_tokens"] == 0
        assert summary["anzahl_ticker"] == 0
        assert summary["ticker_tokens"] == {}

    def test_summarize_empty_file_returns_empty(self, tmp_path):
        """summarize_usage mit leerer Datei (nur Header) → leeres dict."""
        usage_file = str(tmp_path / "usage.csv")
        # Nur Header schreiben
        with open(usage_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "timestamp", "ticker", "prompt_tokens", "completion_tokens", "total_tokens",
            ])
            writer.writeheader()

        summary = summarize_usage(usage_file=usage_file)

        assert summary["anzahl_calls"] == 0


# ---------------------------------------------------------------------------
# 5. LLMClient: chat setzt total_usage kumulativ
# ---------------------------------------------------------------------------


class _MockResponseWithUsage:
    """Mock für requests.Response mit usage-Feld."""

    def __init__(self, status_code: int, content: str = "Test", usage: dict | None = None):
        self.status_code = status_code
        self._content = content
        self.text = content
        self._usage = usage

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")  # noqa: TRY002

    def json(self):
        result = {
            "choices": [
                {"message": {"content": self._content}}
            ]
        }
        if self._usage is not None:
            result["usage"] = self._usage
        return result


class TestLLMClientUsage:
    """Tests für LLMClient total_usage-Akkumulation."""

    def test_total_usage_accumulates_across_calls(self):
        """chat() addiert usage kumulativ auf total_usage."""
        from concilium.llm import LLMClient

        client = LLMClient(base_url="http://fake:8080/v1", api_key="test-key", model="test-model")

        resp1 = _MockResponseWithUsage(200, "Antwort 1", {
            "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
        })
        resp2 = _MockResponseWithUsage(200, "Antwort 2", {
            "prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300,
        })

        with patch("concilium.llm.requests.post", side_effect=[resp1, resp2]):
            client.chat([{"role": "user", "content": "Erste Frage"}])
            client.chat([{"role": "user", "content": "Zweite Frage"}])

        assert client.total_usage["prompt_tokens"] == 300
        assert client.total_usage["completion_tokens"] == 150
        assert client.total_usage["total_tokens"] == 450
        assert client.last_usage is not None
        assert client.last_usage["total_tokens"] == 300

    def test_total_usage_no_usage_in_response(self):
        """Wenn die API kein usage liefert, bleibt total_usage bei Null."""
        from concilium.llm import LLMClient

        client = LLMClient(base_url="http://fake:8080/v1", api_key="test-key", model="test-model")

        resp = _MockResponseWithUsage(200, "Antwort", usage=None)

        with patch("concilium.llm.requests.post", return_value=resp):
            client.chat([{"role": "user", "content": "Test"}])

        assert client.total_usage["prompt_tokens"] == 0
        assert client.total_usage["completion_tokens"] == 0
        assert client.total_usage["total_tokens"] == 0
        assert client.last_usage is None

    def test_last_usage_resets_on_new_call(self):
        """last_usage wird am Anfang von chat() auf None gesetzt."""
        from concilium.llm import LLMClient

        client = LLMClient(base_url="http://fake:8080/v1", api_key="test-key", model="test-model")

        resp_with_usage = _MockResponseWithUsage(200, "Antwort", {
            "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
        })
        resp_no_usage = _MockResponseWithUsage(200, "Antwort 2", usage=None)

        with patch("concilium.llm.requests.post", side_effect=[resp_with_usage, resp_no_usage]):
            client.chat([{"role": "user", "content": "Erste"}])
            assert client.last_usage is not None
            client.chat([{"role": "user", "content": "Zweite"}])
            assert client.last_usage is None

        # total_usage behält die kumulativen Werte
        assert client.total_usage["total_tokens"] == 150


# ---------------------------------------------------------------------------
# 6. CLI: --usage parst und ruft summarize_usage auf
# ---------------------------------------------------------------------------


class TestCliUsage:
    """Tests für --usage CLI-Flag."""

    def test_usage_flag_calls_summarize(self):
        """--usage ruft summarize_usage auf und gibt Daten aus."""
        from concilium.cli import main

        mock_summary = {
            "anzahl_calls": 3,
            "summe_prompt_tokens": 350,
            "summe_completion_tokens": 175,
            "summe_total_tokens": 525,
            "anzahl_ticker": 2,
            "ticker_tokens": {"AAPL": 450, "MSFT": 75},
        }

        with patch("concilium.cli.summarize_usage", return_value=mock_summary):
            result = main(["--usage"])

        assert result == 0

    def test_usage_flag_no_data(self):
        """--usage mit keinen Daten → Exit 0, Hinweismeldung."""
        from concilium.cli import main

        mock_summary = {
            "anzahl_calls": 0,
            "summe_prompt_tokens": 0,
            "summe_completion_tokens": 0,
            "summe_total_tokens": 0,
            "anzahl_ticker": 0,
            "ticker_tokens": {},
        }

        with patch("concilium.cli.summarize_usage", return_value=mock_summary):
            result = main(["--usage"])

        assert result == 0

    def test_usage_flag_conflict_with_ticker(self, capsys):
        """--usage + --ticker → Fehler, Exit 1."""
        from concilium.cli import main

        result = main(["--usage", "--ticker", "AAPL"])
        assert result == 1
