"""Tests für Watchlist-Support (--watchlist, _read_watchlist, watchlist.txt).

Alle Tests sind OFFLINE-fähig: kein yfinance, kein Netzwerk.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.cli import _read_watchlist, main  # noqa: E402

# --------------------------------------------------------------------------- #
# _read_watchlist
# --------------------------------------------------------------------------- #


class TestReadWatchlist:
    """Testet _read_watchlist: Parsen, Kommentare, Leerzeilen, Fehlersicherheit."""

    def test_reads_tickers_correctly(self, tmp_path):
        """Liest Ticker korrekt, ignoriert Kommentare und Leerzeilen."""
        wl = tmp_path / "watchlist.txt"
        wl.write_text(
            "# Kommentar\n"
            "AAPL\n"
            "\n"
            "  MSFT  \n"
            "# NVDA auskommentiert\n"
            "NVDA\n"
            "TSM\n",
            encoding="utf-8",
        )
        result = _read_watchlist(str(wl))
        assert result == ["AAPL", "MSFT", "NVDA", "TSM"]

    def test_trims_whitespace(self, tmp_path):
        """Whitespace um Ticker wird getrimmt."""
        wl = tmp_path / "watchlist.txt"
        wl.write_text("  AAPL  \n\tMSFT\t\n  NVDA  \n", encoding="utf-8")
        result = _read_watchlist(str(wl))
        assert result == ["AAPL", "MSFT", "NVDA"]

    def test_empty_file_returns_empty_list(self, tmp_path):
        """Leere Datei → leere Liste."""
        wl = tmp_path / "watchlist.txt"
        wl.write_text("", encoding="utf-8")
        result = _read_watchlist(str(wl))
        assert result == []

    def test_only_comments_and_blanks_returns_empty(self, tmp_path):
        """Nur Kommentare und Leerzeilen → leere Liste."""
        wl = tmp_path / "watchlist.txt"
        wl.write_text("# nur kommentare\n\n#  \n", encoding="utf-8")
        result = _read_watchlist(str(wl))
        assert result == []

    def test_missing_file_returns_empty_list(self):
        """Fehlende Datei → leere Liste (crasht nicht)."""
        result = _read_watchlist("/nonexistent/path/watchlist.txt")
        assert result == []

    def test_env_override(self, tmp_path, monkeypatch):
        """CONCILIUM_WATCHLIST übersteuert den Pfad."""
        wl = tmp_path / "my_watchlist.txt"
        wl.write_text("GOOGL\nMETA\n", encoding="utf-8")
        monkeypatch.setenv("CONCILIUM_WATCHLIST", str(wl))
        result = _read_watchlist()  # kein expliziter Pfad → Env
        assert result == ["GOOGL", "META"]

    def test_explicit_path_overrides_env(self, tmp_path, monkeypatch):
        """Expliziter Pfad hat Vorrang über CONCILIUM_WATCHLIST-Env."""
        wl_env = tmp_path / "env_wl.txt"
        wl_env.write_text("ENV_TKR\n", encoding="utf-8")
        wl_explicit = tmp_path / "explicit_wl.txt"
        wl_explicit.write_text("EXPLICIT_TKR\n", encoding="utf-8")
        monkeypatch.setenv("CONCILIUM_WATCHLIST", str(wl_env))
        result = _read_watchlist(str(wl_explicit))
        assert result == ["EXPLICIT_TKR"]


# --------------------------------------------------------------------------- #
# --watchlist Modus (CLI main)
# --------------------------------------------------------------------------- #


class TestWatchlistMode:
    """Testet --watchlist in der CLI: evaluate vorn, dann Batch, Fehler-Resistenz."""

    def test_watchlist_runs_evaluate_before_analysis(self, tmp_path, monkeypatch):
        """--watchlist ruft evaluate_journal + _write_calibration_json VOR run_pipeline auf."""
        wl = tmp_path / "watchlist.txt"
        wl.write_text("AAPL\nMSFT\n", encoding="utf-8")

        call_order: list[str] = []

        def mock_evaluate_journal(*args, **kwargs):
            call_order.append("evaluate_journal")
            return {
                "anzahl_entscheidungen": 5,
                "hit_rate_gesamt": 0.4,
                "nach_aktion": {},
            }

        def mock_write_calibration(eval_result, **kwargs):
            call_order.append("_write_calibration_json")

        def mock_run_pipeline(ticker, **kwargs):
            call_order.append(f"run_pipeline:{ticker}")
            return {
                "ticker": ticker,
                "data": {"ticker": ticker, "fundamentals": {}, "technicals": {},
                         "sentiment": {}, "news": [], "history": []},
                "no_llm": True,
            }

        def mock_generate_report(result, reports_dir=None):
            return f"# Report für {result.get('ticker', '?')}"

        def mock_track_record_report(eval_result):
            return "# Track Record Report"

        monkeypatch.setenv("CONCILIUM_WATCHLIST", str(wl))

        with patch("concilium.cli.evaluate_journal", side_effect=mock_evaluate_journal):
            with patch("concilium.cli._write_calibration_json", side_effect=mock_write_calibration):
                with patch("concilium.cli.run_pipeline", side_effect=mock_run_pipeline):
                    with patch("concilium.cli.generate_report", side_effect=mock_generate_report):
                        with patch("concilium.cli.generate_track_record_report",
                                   side_effect=mock_track_record_report):
                            with patch("concilium.cli.LLMClient"):
                                result = main(["--watchlist", "--no-llm"])

        # evaluate + calibration kamen VOR run_pipeline
        assert call_order[0] == "evaluate_journal"
        assert call_order[1] == "_write_calibration_json"
        assert "run_pipeline:AAPL" in call_order
        assert "run_pipeline:MSFT" in call_order
        # Beide Ticker wurden analysiert
        assert call_order.index("run_pipeline:AAPL") < call_order.index("run_pipeline:MSFT")
        assert result == 0

    def test_watchlist_continues_on_failure(self, tmp_path, monkeypatch):
        """Ein fehlgeschlagener Ticker crasht nicht den Watchlist-Batch."""
        wl = tmp_path / "watchlist.txt"
        wl.write_text("AAPL\nBAD\nMSFT\n", encoding="utf-8")

        call_count = [0]

        def mock_evaluate_journal(*args, **kwargs):
            return {"anzahl_entscheidungen": 0, "hit_rate_gesamt": None, "nach_aktion": {}}

        def mock_run_pipeline(ticker, **kwargs):
            call_count[0] += 1
            if ticker == "BAD":
                raise ValueError("Ungültiger Ticker")
            return {
                "ticker": ticker,
                "data": {"ticker": ticker, "fundamentals": {}, "technicals": {},
                         "sentiment": {}, "news": [], "history": []},
                "no_llm": True,
            }

        def mock_generate_report(result, reports_dir=None):
            return f"# Report für {result.get('ticker', '?')}"

        def mock_track_record_report(eval_result):
            return "# Track Record Report"

        monkeypatch.setenv("CONCILIUM_WATCHLIST", str(wl))

        with patch("concilium.cli.evaluate_journal", side_effect=mock_evaluate_journal):
            with patch("concilium.cli._write_calibration_json"):
                with patch("concilium.cli.run_pipeline", side_effect=mock_run_pipeline):
                    with patch("concilium.cli.generate_report", side_effect=mock_generate_report):
                        with patch("concilium.cli.generate_track_record_report",
                                   side_effect=mock_track_record_report):
                            with patch("concilium.cli.LLMClient"):
                                result = main(["--watchlist", "--no-llm"])

        # Alle 3 Ticker wurden versucht
        assert call_count[0] == 3
        # Mindestens 2 erfolgreich → Exit 0
        assert result == 0

    def test_watchlist_all_fail_exit_1(self, tmp_path, monkeypatch):
        """Alle Ticker fehlgeschlagen → Exit 1."""
        wl = tmp_path / "watchlist.txt"
        wl.write_text("BAD1\nBAD2\n", encoding="utf-8")

        def mock_evaluate_journal(*args, **kwargs):
            return {"anzahl_entscheidungen": 0, "hit_rate_gesamt": None, "nach_aktion": {}}

        def mock_run_pipeline(ticker, **kwargs):
            raise ValueError(f"Fehler für {ticker}")

        def mock_generate_report(result, reports_dir=None):
            return "# dummy"

        def mock_track_record_report(eval_result):
            return "# Track Record Report"

        monkeypatch.setenv("CONCILIUM_WATCHLIST", str(wl))

        with patch("concilium.cli.evaluate_journal", side_effect=mock_evaluate_journal):
            with patch("concilium.cli._write_calibration_json"):
                with patch("concilium.cli.run_pipeline", side_effect=mock_run_pipeline):
                    with patch("concilium.cli.generate_report", side_effect=mock_generate_report):
                        with patch("concilium.cli.generate_track_record_report",
                                   side_effect=mock_track_record_report):
                            with patch("concilium.cli.LLMClient"):
                                result = main(["--watchlist", "--no-llm"])

        assert result == 1

    def test_watchlist_empty_returns_error(self, tmp_path, monkeypatch):
        """Leere Watchlist → Fehler, Exit 1."""
        wl = tmp_path / "watchlist.txt"
        wl.write_text("# nur kommentare\n\n", encoding="utf-8")
        monkeypatch.setenv("CONCILIUM_WATCHLIST", str(wl))

        with patch("concilium.cli.LLMClient"):
            result = main(["--watchlist", "--no-llm"])
        assert result == 1

    def test_watchlist_missing_file_returns_error(self, monkeypatch):
        """Fehlende Watchlist-Datei → Fehler, Exit 1."""
        monkeypatch.setenv("CONCILIUM_WATCHLIST", "/nonexistent/watchlist.txt")
        with patch("concilium.cli.LLMClient"):
            result = main(["--watchlist", "--no-llm"])
        assert result == 1


# --------------------------------------------------------------------------- #
# Mutual Exclusion
# --------------------------------------------------------------------------- #


class TestWatchlistMutex:
    """Testet Mutual Exclusion von --watchlist mit anderen Modi."""

    def test_watchlist_and_ticker_error(self):
        """--watchlist + --ticker → Fehler, Exit 1."""
        result = main(["--watchlist", "--ticker", "AAPL"])
        assert result == 1

    def test_watchlist_and_tickers_error(self):
        """--watchlist + --tickers → Fehler, Exit 1."""
        result = main(["--watchlist", "--tickers", "AAPL,NVDA"])
        assert result == 1

    def test_watchlist_and_portfolio_error(self):
        """--watchlist + --portfolio → Fehler, Exit 1."""
        result = main(["--watchlist", "--portfolio", "AAPL,NVDA"])
        assert result == 1
