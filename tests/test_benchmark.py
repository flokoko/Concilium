"""Tests für die regionale Benchmark-Map (analog TradingAgents' benchmark_map).

Der Benchmark-Index wird deterministisch aus dem Börsen-Suffix des Tickers
abgeleitet: .DE → ^GDAXI (DAX), .L → ^FTSE (FTSE 100), .T → ^N225 (Nikkei 225)
usw. Ohne Suffix (US) und bei unbekanntem Suffix → SPY (Fallback). Crasht nie.
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.evaluate import benchmark_for_ticker, realised_return_for_row  # noqa: E402
from concilium.journal import JOURNAL_HEADER  # noqa: E402

# --------------------------------------------------------------------------- #
# Helper: synthetische Preisdaten
# --------------------------------------------------------------------------- #


def _make_prices(start_price: float, n_days: int, drift: float = 0.0) -> list[dict]:
    """Erzeugt eine Liste von Preis-Dicts für n_days Tage."""
    prices: list[dict] = []
    base_date = datetime.now() - timedelta(days=n_days + 5)
    price = start_price
    for i in range(n_days):
        d = base_date + timedelta(days=i)
        price = price * (1.0 + drift)
        prices.append({
            "date": d.strftime("%Y-%m-%d"),
            "close": round(price, 2),
            "high": round(price * 1.01, 2),
            "low": round(price * 0.99, 2),
        })
    return prices


# --------------------------------------------------------------------------- #
# Tests: benchmark_for_ticker
# --------------------------------------------------------------------------- #


class TestBenchmarkForTicker:
    """Testet benchmark_for_ticker: Börsen-Suffix → regionaler Index."""

    def test_de_maps_to_dax(self):
        assert benchmark_for_ticker("RWE.DE") == "^GDAXI"

    def test_l_maps_to_ftse(self):
        assert benchmark_for_ticker("SHEL.L") == "^FTSE"

    def test_t_maps_to_nikkei(self):
        assert benchmark_for_ticker("7203.T") == "^N225"

    def test_hk_maps_to_hang_seng(self):
        assert benchmark_for_ticker("0700.HK") == "^HSI"

    def test_ns_maps_to_nifty(self):
        assert benchmark_for_ticker("RELIANCE.NS") == "^NSEI"

    def test_bo_maps_to_sensex(self):
        assert benchmark_for_ticker("500325.BO") == "^BSESN"

    def test_to_maps_to_tsx(self):
        assert benchmark_for_ticker("RY.TO") == "^GSPTSE"

    def test_ax_maps_to_asx(self):
        assert benchmark_for_ticker("BHP.AX") == "^AXJO"

    def test_ss_maps_to_sse(self):
        assert benchmark_for_ticker("600519.SS") == "000001.SS"

    def test_sz_maps_to_szse(self):
        assert benchmark_for_ticker("000001.SZ") == "399001.SZ"

    def test_no_suffix_maps_to_spy(self):
        assert benchmark_for_ticker("AAPL") == "SPY"

    def test_unknown_suffix_falls_back_to_spy(self):
        assert benchmark_for_ticker("SAP.F") == "SPY"

    def test_case_insensitive_suffix(self):
        assert benchmark_for_ticker("rwe.de") == "^GDAXI"

    def test_strips_whitespace(self):
        assert benchmark_for_ticker("  RWE.DE  ") == "^GDAXI"

    def test_empty_ticker_returns_spy(self):
        assert benchmark_for_ticker("") == "SPY"

    def test_none_ticker_returns_spy(self):
        assert benchmark_for_ticker(None) == "SPY"

    def test_never_raises_on_garbage_input(self):
        """Crasht nie — gibt immer einen String zurück."""
        for garbage in ({"a": 1}, ["X"], 42, "SAP...", "...", ".", "RWE.DE.EXTRA"):
            result = benchmark_for_ticker(garbage)  # type: ignore[arg-type]
            assert isinstance(result, str)
            assert result


# --------------------------------------------------------------------------- #
# Tests: realised_return_for_row nutzt den regionalen Benchmark
# --------------------------------------------------------------------------- #


class TestRealisedReturnBenchmark:
    """Testet dass realised_return_for_row den richtigen Benchmark lädt."""

    def test_de_ticker_uses_dax_benchmark(self):
        """RWE.DE → Benchmark ^GDAXI wird geladen (nicht SPY)."""
        prices = _make_prices(100, 60, drift=0.01)
        dax_prices = _make_prices(13000, 60, drift=0.005)
        loaded: list[str] = []

        def mock_load(ticker, *, lookback_days=30):
            loaded.append(ticker)
            if ticker == "^GDAXI":
                return dax_prices
            if ticker == "SPY":
                raise AssertionError("SPY darf für RWE.DE nicht geladen werden")
            return prices

        row = {
            "ticker": "RWE.DE",
            "timestamp": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),
            "action": "KAUFEN",
        }
        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = realised_return_for_row(row, lookback_days=30)

        assert result is not None
        assert result["benchmark"] == "^GDAXI"
        assert result["benchmark_return_pct"] is not None
        assert result["alpha_pct"] == result["raw_return_pct"] - result["benchmark_return_pct"]
        assert "^GDAXI" in loaded

    def test_us_ticker_still_uses_spy(self):
        """AAPL (kein Suffix) → weiterhin SPY, Felder umbenannt."""
        prices = _make_prices(100, 60, drift=0.01)
        spy_prices = _make_prices(100, 60, drift=0.005)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return spy_prices
            return prices

        row = {
            "ticker": "AAPL",
            "timestamp": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),
            "action": "KAUFEN",
        }
        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = realised_return_for_row(row, lookback_days=30)

        assert result is not None
        assert result["benchmark"] == "SPY"
        assert result["benchmark_return_pct"] is not None
        assert result["alpha_pct"] is not None

    def test_benchmark_none_when_benchmark_fails(self):
        """Benchmark-Daten nicht ladbar → benchmark_return_pct=None, alpha_pct=None."""
        prices = _make_prices(100, 60, drift=0.01)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return None
            return prices

        row = {
            "ticker": "AAPL",
            "timestamp": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),
            "action": "KAUFEN",
        }
        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = realised_return_for_row(row, lookback_days=30)

        assert result is not None
        assert result["benchmark"] == "SPY"
        assert result["benchmark_return_pct"] is None
        assert result["alpha_pct"] is None

    def test_unknown_suffix_ticker_uses_spy_fallback(self):
        """SAP.F (unbekanntes Suffix) → SPY-Fallback."""
        prices = _make_prices(100, 60, drift=0.01)
        spy_prices = _make_prices(100, 60, drift=0.0)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return spy_prices
            return prices

        row = {
            "ticker": "SAP.F",
            "timestamp": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),
            "action": "KAUFEN",
        }
        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = realised_return_for_row(row, lookback_days=30)

        assert result is not None
        assert result["benchmark"] == "SPY"

# --------------------------------------------------------------------------- #
# Tests: dynamische "Alpha vs {benchmark}"-Texte in feedback.py
# --------------------------------------------------------------------------- #


class TestDynamicBenchmarkLabels:
    """Testet dass Reflexions-Texte den regionalen Benchmark nennen."""

    def _write_journal_row(self, writer, ticker: str) -> None:
        """Schreibt eine abgelaufene KAUFEN-Zeile für ticker."""
        full_row = {k: "" for k in JOURNAL_HEADER}
        full_row.update({
            "ticker": ticker,
            "action": "KAUFEN",
            "timestamp": (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d %H:%M:%S"),
        })
        writer.writerow(full_row)

    def _setup_journal(self, tmp_path, tickers: list[str]) -> None:
        """Schreibt eine Journal-CSV (journal/decisions.csv) für die Tickers."""
        import shutil

        journal_dir = tmp_path / "journal"
        journal_dir.mkdir(exist_ok=True)
        path = str(tmp_path / "decisions.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
            writer.writeheader()
            for ticker in tickers:
                self._write_journal_row(writer, ticker)
        shutil.copy(path, str(journal_dir / "decisions.csv"))

    def test_reflection_labels_dax_for_de_ticker(self, tmp_path, monkeypatch):
        """RWE.DE → Reflexions-Text enthält 'Alpha vs ^GDAXI', nicht SPY."""
        monkeypatch.chdir(tmp_path)
        self._setup_journal(tmp_path, ["RWE.DE"])
        prices = _make_prices(100, 60, drift=0.01)
        dax = _make_prices(13000, 60, drift=0.005)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "^GDAXI":
                return dax
            return prices

        from concilium.feedback import build_reflection_context
        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_reflection_context("RWE.DE")

        assert result != ""
        assert "Alpha vs ^GDAXI" in result
        assert "SPY" not in result

    def test_reflection_labels_spy_for_us_ticker(self, tmp_path, monkeypatch):
        """AAPL → Reflexions-Text enthält weiterhin 'Alpha vs SPY'."""
        monkeypatch.chdir(tmp_path)
        self._setup_journal(tmp_path, ["AAPL"])
        prices = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.005)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return spy
            return prices

        from concilium.feedback import build_reflection_context
        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_reflection_context("AAPL")

        assert result != ""
        assert "Alpha vs SPY" in result

    def test_cross_ticker_labels_benchmark_per_ticker(self, tmp_path, monkeypatch):
        """Cross-Ticker: pro Zeile der Benchmark des jeweiligen Tickers."""
        monkeypatch.chdir(tmp_path)
        self._setup_journal(tmp_path, ["RWE.DE", "MSFT"])
        prices = _make_prices(100, 60, drift=0.01)
        dax = _make_prices(13000, 60, drift=0.005)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "^GDAXI":
                return dax
            return prices

        from concilium.feedback import build_cross_ticker_context
        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_cross_ticker_context("AAPL")

        assert result != ""
        assert "Alpha vs ^GDAXI" in result
        assert "Alpha vs SPY" in result
