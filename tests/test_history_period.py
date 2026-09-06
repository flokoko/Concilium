"""Tests für Phase 5: Kurs-Historie wird mit period="2y" geladen (~500 Handelstage).

Abgedeckte Verhaltensweisen:
  (a) collect_ticker_data fetcht die Ticker-Historie mit period="2y"
      (der Backtest braucht >=200 Tage NACH SMA200-Start für Crossovers —
      mit 1y/~250 Tagen blieb für den auswertbaren Zeitraum zu wenig übrig,
      vgl. CEG-Live-Lauf mit 0 Signalen).
  (b) Der aktuelle Kurs bleibt der NEUESTE Close der Historie
      (längeres Fenster darf die "letzter gültiger Wert"-Logik nicht ändern).
  (c) SMA50/SMA200 sind mit 2y-Historie robust verfügbar.

Alle Tests laufen offline: yfinance + Social-Quellen + Makro werden gemockt.
"""

from __future__ import annotations

import os
import sys
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# src zum Pfad hinzufügen
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

from concilium.data import collect_ticker_data  # noqa: E402

# Fixtures / Helfer -----------------------------------------------------------


def _make_hist_2y(n_days: int = 500) -> pd.DataFrame:
    """Synthetische 2y-Kurshistorie (werktäglich, steigend, endet 2026-09-04).

    Steigende Closes (100.0 + 0.3 * i), damit erster und letzter Close
    eindeutig unterscheidbar sind.
    """
    idx = pd.bdate_range(end="2026-09-04", periods=n_days)
    closes = [100.0 + 0.3 * i for i in range(n_days)]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000_000] * n_days,
        },
        index=idx,
    )


def _make_mock_ticker(hist: pd.DataFrame) -> MagicMock:
    """yf.Ticker-Mock, dessen .history-Mock die gegebene Historie liefert.

    t.history bleibt ein Mock (kein return_value fixiert), damit die
    Aufruf-Argumente (period=...) assertion-fähig sind.
    """
    t = MagicMock()
    t.info = {
        "longName": "Test Corp",
        "shortName": "TEST",
        "sector": "Tech",
        "industry": "Semis",
        "currency": "USD",
    }
    t.news = []
    t.history.return_value = hist
    return t


def _collect(hist: pd.DataFrame) -> tuple[dict, MagicMock]:
    """collect_ticker_data mit gemocktem yfinance + Social-/Makro-Quellen.

    Liefert (result, mock_ticker) — der Mock-Ticker dient der
    Aufruf-Argument-Prüfung (period="2y").
    """
    mock_ticker = _make_mock_ticker(hist)
    with ExitStack() as stack:
        stack.enter_context(
            patch("concilium.data.yf.Ticker", return_value=mock_ticker)
        )
        stack.enter_context(patch("concilium.data._fetch_google_news", return_value=[]))
        stack.enter_context(patch("concilium.data._fetch_stocktwits", return_value=[]))
        stack.enter_context(patch("concilium.data._fetch_reddit", return_value=[]))
        stack.enter_context(patch("concilium.data._fetch_macro_data", return_value={}))
        stack.enter_context(
            patch("concilium.data._get_sp500_momentum", return_value=1.5)
        )
        stack.enter_context(patch("concilium.data._save_cache"))
        result = collect_ticker_data("TEST")
    return result, mock_ticker


# (a) Fetch-Period -------------------------------------------------------------


class TestHistoryFetchedWith2y:
    """Die Ticker-Historie wird mit period="2y" (~500 Handelstage) geladen."""

    def test_history_call_uses_period_2y(self):
        """t.history wird mit period="2y" aufgerufen (nicht mehr "1y")."""
        _, mock_ticker = _collect(_make_hist_2y())

        assert mock_ticker.history.call_count == 1
        call_kwargs = mock_ticker.history.call_args.kwargs
        assert call_kwargs.get("period") == "2y"
        assert call_kwargs.get("auto_adjust") is False

    def test_full_2y_history_passed_through(self):
        """Die volle 2y-Historie (~500 Tage) landet unverändert in result['history']."""
        hist = _make_hist_2y(500)
        result, _ = _collect(hist)

        assert len(result["history"]) == 500


# (b) Aktueller Kurs bleibt der neueste Close ----------------------------------


class TestCurrentPriceIsNewestClose:
    """Längeres Fenster darf die letzte-Close-Logik nicht ändern."""

    def test_current_price_is_last_close(self):
        """current_price = NEUESTER Close der 500-Tage-Historie (nicht der erste)."""
        hist = _make_hist_2y(500)
        result, _ = _collect(hist)

        first_close = float(hist["Close"].iloc[0])
        last_close = float(hist["Close"].iloc[-1])
        assert first_close != last_close  # Testdaten sind eindeutig

        assert result["technicals"]["current_price"] == pytest.approx(last_close)

    def test_last_history_record_is_newest_date(self):
        """Der letzte history-Record trägt das neueste Datum (chronologisch)."""
        hist = _make_hist_2y(500)
        result, _ = _collect(hist)

        records = result["history"]
        assert records[0]["date"] < records[-1]["date"]
        assert records[-1]["date"] == hist.index.max().strftime("%Y-%m-%d")


# (c) Indikatoren mit 2y robust verfügbar ---------------------------------------


class TestIndicatorsAvailableWith2y:
    """SMA50/SMA200 sind mit 2y-Historie berechnet (kein None)."""

    def test_sma50_and_sma200_available(self):
        """Mit ~500 Tagen sind SMA50 und SMA200 gültig (nicht None)."""
        hist = _make_hist_2y(500)
        result, _ = _collect(hist)

        technicals = result["technicals"]
        assert technicals["sma50"] is not None
        assert technicals["sma200"] is not None
        # SMA200 = Mittelwert der letzten 200 Closes
        expected_sma200 = float(hist["Close"].tail(200).mean())
        assert technicals["sma200"] == pytest.approx(expected_sma200)


# (d) Report-Zeitraum aus echten Daten -----------------------------------------


class TestReportZeitraumFromData:
    """Der Report-Zeitraum wird aus den tatsächlichen Backtest-Daten abgeleitet.

    Regression-Guard (bestehendes Verhalten, bewusst gepinnt): report.py
    rendert "Zeitraum | startdatum – enddatum" aus dem backtest-dict —
    diese Daten stammen aus der Kurs-Historie (run_backtest), nicht aus
    einer Hartkodierung. Mit 2y-Historie wird der Zeitraum automatisch
    länger (SMA200-Start rückt nach vorn, hier: Tag 199 von ~500).
    """

    def test_zeitraum_line_matches_backtest_dates(self):
        from concilium.backtest import run_backtest
        from concilium.report import generate_report

        hist = _make_hist_2y(500)
        collected, _ = _collect(hist)
        bt = run_backtest({"history": collected["history"]})

        assert bt["strategie_rendite"] is not None  # Backtest-Sektion rendert
        # startdatum = SMA200-Start (Tag 199) — aus den Historie-Daten abgeleitet
        assert bt["startdatum"] == hist.index[199].strftime("%Y-%m-%d")
        assert bt["enddatum"] == hist.index.max().strftime("%Y-%m-%d")

        report = generate_report({
            "ticker": "TEST",
            "data": collected,
            "no_llm": True,
            "backtest": bt,
        })

        expected_line = f"| Zeitraum | {bt['startdatum']} – {bt['enddatum']} |"
        assert expected_line in report
