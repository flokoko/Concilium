"""Tests für Batch-Modus (--tickers) und CLI-Parsing-Logik."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.cli import main  # noqa: E402


class TestCLIParsingMutualExclusion:
    """Test CLI-Parsing: --ticker und --tickers schließen sich aus."""

    def test_ticker_and_tickers_mutual_exclusion(self):
        """--ticker und --tickers zusammen → Fehler (Exit 2 von argparse.error)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--ticker", "AAPL", "--tickers", "NVDA,MSFT"])
        # parser.error() exitet mit Code 2
        assert exc_info.value.code == 2

    def test_evaluate_and_ticker_error(self):
        """--evaluate + --ticker → Fehler, Exit 1."""
        result = main(["--evaluate", "--ticker", "AAPL"])
        assert result == 1

    def test_evaluate_and_tickers_error(self):
        """--evaluate + --tickers → Fehler, Exit 1."""
        result = main(["--evaluate", "--tickers", "AAPL,NVDA"])
        assert result == 1

    def test_neither_ticker_nor_tickers_error(self):
        """Weder --ticker noch --tickers → Fehler (Exit 2 von argparse.error)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--no-llm"])
        assert exc_info.value.code == 2


class TestBatchMode:
    """Test Batch-Modus: fehlgeschlagener Ticker crasht nicht den Batch."""

    def test_batch_continues_on_failure(self):
        """Ein fehlgeschlagener Tiker crasht nicht den Batch — Fehler wird gezählt."""
        call_count = [0]

        def mock_run_pipeline(ticker, **kwargs):
            call_count[0] += 1
            if ticker == "BAD":
                raise ValueError("Ungültiger Ticker")
            # Erfolgreicher Ticker
            return {
                "ticker": ticker,
                "data": {"ticker": ticker, "fundamentals": {}, "technicals": {},
                         "sentiment": {}, "news": [], "history": []},
                "no_llm": True,
            }

        def mock_generate_report(result, reports_dir=None):
            return f"# Report für {result.get('ticker', '?')}"

        with patch("concilium.cli.run_pipeline", side_effect=mock_run_pipeline):
            with patch("concilium.cli.generate_report", side_effect=mock_generate_report):
                with patch("concilium.cli.LLMClient"):
                    result = main(["--tickers", "AAPL,BAD,MSFT", "--no-llm"])

        # Beide Tickser wurden versucht
        assert call_count[0] == 3
        # Mindestens einer erfolgreich → Exit 0
        assert result == 0

    def test_batch_all_fail_exit_1(self):
        """Alle Ticker fehlgeschlagen → Exit 1."""
        def mock_run_pipeline(ticker, **kwargs):
            raise ValueError(f"Fehler für {ticker}")

        def mock_generate_report(result, reports_dir=None):
            return "# dummy"

        with patch("concilium.cli.run_pipeline", side_effect=mock_run_pipeline):
            with patch("concilium.cli.generate_report", side_effect=mock_generate_report):
                with patch("concilium.cli.LLMClient"):
                    result = main(["--tickers", "BAD1,BAD2", "--no-llm"])

        assert result == 1

    def test_batch_single_ticker_works(self):
        """Ein Ticker in --tickers funktioniert."""
        def mock_run_pipeline(ticker, **kwargs):
            return {
                "ticker": ticker,
                "data": {"ticker": ticker, "fundamentals": {}, "technicals": {},
                         "sentiment": {}, "news": [], "history": []},
                "no_llm": True,
            }

        def mock_generate_report(result, reports_dir=None):
            return f"# Report für {result.get('ticker', '?')}"

        with patch("concilium.cli.run_pipeline", side_effect=mock_run_pipeline):
            with patch("concilium.cli.generate_report", side_effect=mock_generate_report):
                with patch("concilium.cli.LLMClient"):
                    result = main(["--tickers", "AAPL", "--no-llm"])

        assert result == 0
