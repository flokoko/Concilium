"""Tests für Währungsrisiko (Feature 1) und erweiterte Makro-Daten (Feature 2).

Prüft:
  - collect_ticker_data: eur_risiko True/False, eurusd geladen oder None
  - _fetch_macro_data: neue Felder vorhanden, None bei Fehler, crasht nicht
  - _build_data_text: Währungsrisiko-Block bei eur_risiko, Makro-Zeile mit neuen Kennzahlen
  - report.py: Währungsrisiko-Hinweis + neue Makro-Zeilen
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from concilium.agents import _build_data_text  # noqa: E402
from concilium.data import _fetch_macro_data, collect_ticker_data  # noqa: E402
from concilium.report import generate_report  # noqa: E402

# ---------------------------------------------------------------------------
# Helper: Mock-yfinance-Ticker erstellen
# ---------------------------------------------------------------------------


def _make_mock_ticker(info: dict, hist_len: int = 250) -> MagicMock:
    """Erstellt ein Mock-Ticker-Objekt, das collect_ticker_data akzeptiert."""
    import pandas as pd

    t = MagicMock()
    t.info = info
    dates = pd.date_range(end="2026-01-01", periods=hist_len)
    hist = pd.DataFrame(
        {
            "Close": [100.0 + i * 0.1 for i in range(hist_len)],
            "Volume": [1_000_000] * hist_len,
            "Open": [99.0] * hist_len,
            "High": [101.0] * hist_len,
            "Low": [98.0] * hist_len,
        },
        index=dates,
    )
    t.history.return_value = hist
    t.news = None
    return t


_FULL_INFO_USD = {
    "marketCap": 2_000_000_000_000,
    "trailingPE": 28.5,
    "trailingEps": 6.50,
    "totalRevenue": 400_000_000_000,
    "revenueGrowth": 0.08,
    "profitMargins": 0.25,
    "pegRatio": 1.2,
    "fiftyTwoWeekHigh": 200.0,
    "fiftyTwoWeekLow": 120.0,
    "trailingAnnualDividendYield": 0.005,
    "beta": 1.2,
    "currency": "USD",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "longName": "Test Apple Inc.",
    "currentPrice": 180.0,
    "targetMeanPrice": 200.0,
    "targetHighPrice": 250.0,
    "targetLowPrice": 150.0,
    "recommendationKey": "buy",
    "numberOfAnalystOpinions": 30,
    "recommendationMean": 1.8,
}

_FULL_INFO_EUR = {
    **_FULL_INFO_USD,
    "currency": "EUR",
    "longName": "Test RWE AG",
    "sector": "Utilities",
}


# ---------------------------------------------------------------------------
# Feature 1a: EUR-Risiko-Felder in collect_ticker_data
# ---------------------------------------------------------------------------


class TestEurRisikoFields:
    """eur_risiko und eur_risiko_hinweis werden korrekt gesetzt."""

    @patch("concilium.data.yf.Ticker")
    def test_usd_ticker_eur_risiko_true(self, mock_ticker_class):
        """USD-Ticker → eur_risiko True, Hinweis enthält 'USD'."""
        mock_ticker_class.return_value = _make_mock_ticker(_FULL_INFO_USD)
        data = collect_ticker_data("TEST")
        f = data["fundamentals"]
        assert f["eur_risiko"] is True
        assert "Währungsrisiko" in f["eur_risiko_hinweis"]
        assert "USD" in f["eur_risiko_hinweis"]

    @patch("concilium.data.yf.Ticker")
    def test_eur_ticker_eur_risiko_false(self, mock_ticker_class):
        """EUR-Ticker → eur_risiko False, Hinweis leer."""
        mock_ticker_class.return_value = _make_mock_ticker(_FULL_INFO_EUR)
        data = collect_ticker_data("TEST")
        f = data["fundamentals"]
        assert f["eur_risiko"] is False
        assert f["eur_risiko_hinweis"] == ""

    @patch("concilium.data.yf.Ticker")
    def test_eurusd_field_present(self, mock_ticker_class):
        """eurusd-Feld ist vorhanden (float oder None)."""
        mock_ticker_class.return_value = _make_mock_ticker(_FULL_INFO_USD)
        data = collect_ticker_data("TEST")
        f = data["fundamentals"]
        assert "eurusd" in f
        assert f["eurusd"] is None or isinstance(f["eurusd"], float)

    @patch("concilium.data.yf.Ticker")
    def test_eur_de_ticker_eur_risiko_false(self, mock_ticker_class):
        """EUR.DE Währung → eur_risiko False."""
        info = {**_FULL_INFO_EUR, "currency": "EUR.DE"}
        mock_ticker_class.return_value = _make_mock_ticker(info)
        data = collect_ticker_data("TEST")
        assert data["fundamentals"]["eur_risiko"] is False


# ---------------------------------------------------------------------------
# Feature 2: _fetch_macro_data erweitert
# ---------------------------------------------------------------------------


class TestMacroDataExtended:
    """_fetch_macro_data liefert die neuen Felder und crasht nicht."""

    def test_new_keys_present(self):
        """Die neuen Makro-Keys sind im Ergebnis-dict vorhanden."""
        with patch("concilium.data.yf.Ticker"):
            result = _fetch_macro_data()
        assert "eurusd" in result
        assert "vix" in result
        assert "sp500_trend" in result
        assert "oel_preis" in result
        assert "oel_name" in result

    def test_new_keys_none_on_failure(self):
        """Bei yfinance-Fehler sind die neuen Werte None (best effort)."""
        with patch("concilium.data.yf.Ticker", side_effect=Exception("no network")):
            result = _fetch_macro_data()
        assert result["eurusd"] is None
        assert result["vix"] is None
        assert result["sp500_trend"] is None
        assert result["oel_preis"] is None
        assert result["oel_name"] == "WTI"

    def test_does_not_crash(self):
        """_fetch_macro_data crasht nie, auch bei Exception."""
        with patch("concilium.data.yf.Ticker", side_effect=RuntimeError("crash")):
            result = _fetch_macro_data()
        assert isinstance(result, dict)

    def test_sp500_trend_values(self):
        """sp500_trend wird korrekt als steigend/fallend/flach bestimmt."""
        import pandas as pd

        # Mock für ^GSPC: steigender Trend (>0.5%)
        rising_dates = pd.date_range(end="2026-01-01", periods=22)
        rising_hist = pd.DataFrame({"Close": [100.0 + i * 0.5 for i in range(22)]}, index=rising_dates)

        def ticker_factory(symbol):
            t = MagicMock()
            if symbol == "^GSPC":
                t.history.return_value = rising_hist
                t.info = {}
            else:
                t.history.return_value = pd.DataFrame()
                t.info = {}
            return t

        with patch("concilium.data.yf.Ticker", side_effect=ticker_factory):
            result = _fetch_macro_data()
        assert result["sp500_trend"] == "steigend"


# ---------------------------------------------------------------------------
# Feature 1b: Währungsrisiko-Block in _build_data_text
# ---------------------------------------------------------------------------


class TestWaehrungsrisikoBlock:
    """Währungsrisiko-Block erscheint bei eur_risiko True, nicht bei False."""

    def test_block_appears_when_eur_risiko_true(self):
        """Bei eur_risiko True erscheint der WÄHRUNGSRISIKO-Block."""
        data = {
            "ticker": "AAPL",
            "fundamentals": {
                "name": "Apple Inc.",
                "currency": "USD",
                "eur_risiko": True,
                "eur_risiko_hinweis": "Währungsrisiko: Der Ticker notiert in USD.",
                "eurusd": 1.08,
            },
            "technicals": {},
            "sentiment": {},
            "news": [],
            "macro": {},
            "peers": [],
        }
        text = _build_data_text(data, role="alle")
        assert "=== WÄHRUNGSRISIKO ===" in text
        assert "USD" in text
        assert "EURUSD" in text

    def test_block_absent_when_eur_risiko_false(self):
        """Bei eur_risiko False erscheint kein WÄHRUNGSRISIKO-Block."""
        data = {
            "ticker": "RWE.DE",
            "fundamentals": {
                "name": "RWE AG",
                "currency": "EUR",
                "eur_risiko": False,
                "eur_risiko_hinweis": "",
                "eurusd": None,
            },
            "technicals": {},
            "sentiment": {},
            "news": [],
            "macro": {},
            "peers": [],
        }
        text = _build_data_text(data, role="alle")
        assert "=== WÄHRUNGSRISIKO ===" not in text

    def test_block_appears_for_risk_role(self):
        """Der Block erscheint auch bei role='risk'."""
        data = {
            "ticker": "AAPL",
            "fundamentals": {
                "name": "Apple Inc.",
                "currency": "USD",
                "eur_risiko": True,
                "eur_risiko_hinweis": "Währungsrisiko: USD.",
                "eurusd": 1.08,
            },
            "technicals": {},
            "sentiment": {},
            "news": [],
            "macro": {},
            "peers": [],
        }
        text = _build_data_text(data, role="risk")
        assert "=== WÄHRUNGSRISIKO ===" in text

    def test_eurusd_na_when_none(self):
        """Bei eurusd=None wird 'N/A' im Block angezeigt."""
        data = {
            "ticker": "AAPL",
            "fundamentals": {
                "name": "Apple Inc.",
                "currency": "USD",
                "eur_risiko": True,
                "eur_risiko_hinweis": "Währungsrisiko.",
                "eurusd": None,
            },
            "technicals": {},
            "sentiment": {},
            "news": [],
            "macro": {},
            "peers": [],
        }
        text = _build_data_text(data, role="alle")
        assert "=== WÄHRUNGSRISIKO ===" in text
        assert "N/A" in text


# ---------------------------------------------------------------------------
# Feature 2: Makro-Block in _build_data_text
# ---------------------------------------------------------------------------


class TestMacroBlockExtended:
    """Makro-Zeile enthält die neuen Kennzahlen."""

    def test_macro_extra_line_present(self):
        """Die erweiterte Makro-Zeile mit EURUSD, VIX etc. taucht auf."""
        data = {
            "ticker": "TEST",
            "fundamentals": {"name": "Test"},
            "technicals": {},
            "sentiment": {},
            "news": [],
            "macro": {
                "us_10y_yield": 4.2,
                "us_10y_trend": "steigend",
                "eurusd": 1.08,
                "vix": 15.5,
                "sp500_trend": "steigend",
                "oel_preis": 78.5,
            },
            "peers": [],
        }
        text = _build_data_text(data, role="alle")
        assert "EURUSD:" in text
        assert "VIX:" in text
        assert "Öl (WTI):" in text
        assert "S&P500-Trend:" in text
        assert "Risiko-Off-Regime" in text

    def test_macro_extra_absent_when_all_none(self):
        """Wenn alle neuen Makro-Werte None sind, fehlt die extra Zeile."""
        data = {
            "ticker": "TEST",
            "fundamentals": {"name": "Test"},
            "technicals": {},
            "sentiment": {},
            "news": [],
            "macro": {
                "us_10y_yield": 4.2,
                "eurusd": None,
                "vix": None,
                "sp500_trend": None,
                "oel_preis": None,
            },
            "peers": [],
        }
        text = _build_data_text(data, role="alle")
        assert "Risiko-Off-Regime" not in text

    def test_macro_partial_values(self):
        """Nur die vorhandenen Werte werden gezeigt, fehlende ignoriert."""
        data = {
            "ticker": "TEST",
            "fundamentals": {"name": "Test"},
            "technicals": {},
            "sentiment": {},
            "news": [],
            "macro": {
                "us_10y_yield": 4.2,
                "eurusd": 1.08,
                "vix": None,
                "sp500_trend": None,
                "oel_preis": None,
            },
            "peers": [],
        }
        text = _build_data_text(data, role="alle")
        # EURUSD-Wert erscheint
        assert "EURUSD:" in text
        # VIX-Wert erscheint nicht (nur im Hinweis-String, nicht als Kennzahl)
        assert "VIX:" not in text


# ---------------------------------------------------------------------------
# Report: Währungsrisiko + neue Makro-Zeilen
# ---------------------------------------------------------------------------


class TestReportWaehrungsrisiko:
    """Report zeigt Währungsrisiko-Hinweis bei nicht-EUR-Tickern."""

    def test_report_shows_waehrungsrisiko(self):
        """Report enthält 'Währungsrisiko'-Zeile bei eur_risiko True."""
        result = {
            "ticker": "AAPL",
            "data": {
                "ticker": "AAPL",
                "fundamentals": {
                    "name": "Apple Inc.",
                    "currency": "USD",
                    "eur_risiko": True,
                    "eur_risiko_hinweis": "Währungsrisiko: Der Ticker notiert in USD.",
                    "eurusd": 1.08,
                },
                "technicals": {"current_price": 180.0},
                "sentiment": {},
                "news": [],
                "macro": {},
                "peers": [],
            },
            "no_llm": True,
        }
        report = generate_report(result)
        assert "Währungsrisiko" in report
        assert "EURUSD" in report

    def test_report_no_waehrungsrisiko_for_eur(self):
        """Report enthält keine 'Währungsrisiko'-Zeile bei EUR-Ticker."""
        result = {
            "ticker": "RWE.DE",
            "data": {
                "ticker": "RWE.DE",
                "fundamentals": {
                    "name": "RWE AG",
                    "currency": "EUR",
                    "eur_risiko": False,
                    "eur_risiko_hinweis": "",
                    "eurusd": None,
                },
                "technicals": {"current_price": 30.0},
                "sentiment": {},
                "news": [],
                "macro": {},
                "peers": [],
            },
            "no_llm": True,
        }
        report = generate_report(result)
        # Keine Währungsrisiko-Zeile
        assert "**Währungsrisiko:**" not in report


class TestReportMacroExtended:
    """Report zeigt die neuen Makro-Kennzahlen in der Makro-Tabelle."""

    def test_report_shows_new_macro_lines(self):
        """Report enthält EUR/USD, VIX, S&P500-Trend, Ölpreis in Makro-Sektion."""
        result = {
            "ticker": "TEST",
            "data": {
                "ticker": "TEST",
                "fundamentals": {
                    "name": "Test Inc.",
                    "currency": "USD",
                },
                "technicals": {"current_price": 100.0},
                "sentiment": {},
                "news": [],
                "macro": {
                    "us_10y_yield": 4.2,
                    "us_10y_trend": "steigend",
                    "sp500_pe": 22.0,
                    "eurusd": 1.08,
                    "vix": 15.5,
                    "sp500_trend": "steigend",
                    "oel_preis": 78.5,
                },
                "peers": [],
            },
            "no_llm": True,
        }
        report = generate_report(result)
        assert "EUR/USD" in report
        assert "VIX" in report
        assert "S&P 500 Trend (1M)" in report
        assert "Ölpreis (WTI)" in report

    def test_report_no_macro_lines_when_none(self):
        """Keine neuen Makro-Zeilen wenn Werte None."""
        result = {
            "ticker": "TEST",
            "data": {
                "ticker": "TEST",
                "fundamentals": {
                    "name": "Test Inc.",
                    "currency": "USD",
                },
                "technicals": {"current_price": 100.0},
                "sentiment": {},
                "news": [],
                "macro": {
                    "us_10y_yield": 4.2,
                    "sp500_pe": 22.0,
                    "eurusd": None,
                    "vix": None,
                    "sp500_trend": None,
                    "oel_preis": None,
                },
                "peers": [],
            },
            "no_llm": True,
        }
        report = generate_report(result)
        assert "EUR/USD" not in report
        assert "Ölpreis (WTI)" not in report


# --------------------------------------------------------------------------- #
# Feature 1c: Währungsrisiko-Score im Portfolio-Fit-Report-Abschnitt
# --------------------------------------------------------------------------- #


class TestReportPortfolioFitWaehrungsrisikoScore:
    """Report zeigt Währungsrisiko-Score im Portfolio-Fit-Abschnitt (1c)."""

    def test_report_shows_waehrungsrisiko_score_usd(self):
        """Report enthält 'Währungsrisiko-Score' bei USD-Ticker mit Portfolio-Fit."""
        result = {
            "ticker": "AAPL",
            "data": {
                "ticker": "AAPL",
                "fundamentals": {
                    "name": "Apple Inc.",
                    "currency": "USD",
                    "eur_risiko": True,
                    "eur_risiko_hinweis": "Währungsrisiko: USD.",
                    "eurusd": 1.08,
                },
                "technicals": {"current_price": 180.0},
                "sentiment": {},
                "news": [],
                "macro": {},
                "peers": [],
            },
            "analysts": {"fundamental": {}, "technical": {}, "sentiment": {}},
            "debate": {},
            "trade": {"aktion": "HALTEN"},
            "risk": {},
            "final": {"entscheidung": "GENEHMIGT"},
            "portfolio_fit": {
                "rolle": "Portfolio-Fit-Analyst",
                "portfolio_fit_score": 3,
                "ziel_gewichtung_pct": 2.5,
                "konzentrationsrisiko_bewertung": "AAPL bereits gewichtet.",
                "sektor_overlap_bewertung": "Tech-Overlap.",
                "begründung": "Mäßiger Fit.",
                "portfolio_daten_verfuegbar": True,
                "waehrungsrisiko_score": 3,
            },
            "no_llm": False,
        }
        report = generate_report(result)
        assert "Währungsrisiko-Score:" in report
        assert "3/5" in report
        assert "(USD)" in report

    def test_report_no_waehrungsrisiko_score_for_eur(self):
        """Report enthält keinen 'Währungsrisiko-Score' bei EUR-Ticker (None)."""
        result = {
            "ticker": "RWE.DE",
            "data": {
                "ticker": "RWE.DE",
                "fundamentals": {
                    "name": "RWE AG",
                    "currency": "EUR",
                    "eur_risiko": False,
                    "eur_risiko_hinweis": "",
                    "eurusd": None,
                },
                "technicals": {"current_price": 30.0},
                "sentiment": {},
                "news": [],
                "macro": {},
                "peers": [],
            },
            "portfolio_fit": {
                "rolle": "Portfolio-Fit-Analyst",
                "portfolio_fit_score": 4,
                "ziel_gewichtung_pct": 5.0,
                "konzentrationsrisiko_bewertung": "OK.",
                "sektor_overlap_bewertung": "OK.",
                "begründung": "Gut.",
                "portfolio_daten_verfuegbar": True,
                "waehrungsrisiko_score": None,
            },
            "no_llm": False,
        }
        report = generate_report(result)
        assert "Währungsrisiko-Score:" not in report
