"""Tests für erweiterte Fundamentalkennzahlen — FCF, Nettoverschuldung, PEG-Konsistenz.

Mockt yfinance Ticker-Objekt, um collect_ticker_data offline zu testen.
Prüft:
  - Neue Felder im fundamentals-dict
  - Abgeleitete Werte (net_debt, fcf_margin, net_debt_to_ebitda, forward_pe)
  - PEG-Konsistenz-Warnung
  - _validate_fundamentals erkennt neue Felder (kein Crash)
  - factors.py nutzt neue Felder
  - _build_data_text zeigt neue Felder
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from concilium.agents import _build_data_text  # noqa: E402
from concilium.data import _validate_fundamentals, collect_ticker_data  # noqa: E402
from concilium.factors import compute_multi_factor_score  # noqa: E402

# ---------------------------------------------------------------------------
# Helper: Mock-yfinance-Ticker erstellen
# ---------------------------------------------------------------------------


def _make_mock_ticker(info: dict, hist_len: int = 250) -> MagicMock:
    """Erstellt ein Mock-Ticker-Objekt, das collect_ticker_data akzeptiert."""
    import pandas as pd

    t = MagicMock()
    t.info = info
    # Historie: 250 Tage Close/Volume
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


# Vollständiges info-dict mit allen neuen Feldern
_FULL_INFO = {
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
    # Neue Felder
    "freeCashflow": 100_000_000_000,
    "operatingCashflow": 120_000_000_000,
    "totalDebt": 120_000_000_000,
    "totalCash": 60_000_000_000,
    "ebitda": 130_000_000_000,
    "currentRatio": 1.0,
    "returnOnEquity": 1.5,
    "grossMargins": 0.45,
    "operatingMargins": 0.30,
    "priceToBook": 45.0,
    "bookValue": 4.0,
    "forwardEps": 7.0,
}


class TestExtendedFundamentalsFields:
    """Neue Felder sind im fundamentals-dict vorhanden."""

    @patch("concilium.data.yf.Ticker")
    def test_new_fields_present(self, mock_ticker_class):
        """Alle neuen Felder sind im fundamentals-dict."""
        mock_ticker_class.return_value = _make_mock_ticker(_FULL_INFO)
        data = collect_ticker_data("TEST")
        f = data["fundamentals"]
        # Neue info-Felder
        assert "free_cash_flow" in f
        assert "operating_cashflow" in f
        assert "total_debt" in f
        assert "total_cash" in f
        assert "ebitda" in f
        assert "current_ratio" in f
        assert "return_on_equity" in f
        assert "gross_margin" in f
        assert "operating_margin" in f
        assert "price_to_book" in f
        assert "book_value" in f
        assert "forward_eps" in f
        # Abgeleitete Felder
        assert "net_debt" in f
        assert "fcf_margin" in f
        assert "net_debt_to_ebitda" in f
        assert "forward_pe" in f
        # PEG-Warnung
        assert "peg_konsistenz_warnung" in f

    @patch("concilium.data.yf.Ticker")
    def test_new_fields_values(self, mock_ticker_class):
        """Werte der neuen Felder sind korrekt aus info.get()."""
        mock_ticker_class.return_value = _make_mock_ticker(_FULL_INFO)
        data = collect_ticker_data("TEST")
        f = data["fundamentals"]
        assert f["free_cash_flow"] == 100_000_000_000.0
        assert f["operating_cashflow"] == 120_000_000_000.0
        assert f["total_debt"] == 120_000_000_000.0
        assert f["total_cash"] == 60_000_000_000.0
        assert f["ebitda"] == 130_000_000_000.0
        assert f["current_ratio"] == 1.0
        assert f["return_on_equity"] == 1.5
        assert f["gross_margin"] == 0.45
        assert f["operating_margin"] == 0.30
        assert f["price_to_book"] == 45.0
        assert f["book_value"] == 4.0
        assert f["forward_eps"] == 7.0


class TestDerivedMetrics:
    """Abgeleitete Kennzahlen werden korrekt berechnet."""

    @patch("concilium.data.yf.Ticker")
    def test_net_debt(self, mock_ticker_class):
        """net_debt = total_debt - total_cash."""
        mock_ticker_class.return_value = _make_mock_ticker(_FULL_INFO)
        data = collect_ticker_data("TEST")
        f = data["fundamentals"]
        assert f["net_debt"] == 60_000_000_000.0  # 120e9 - 60e9

    @patch("concilium.data.yf.Ticker")
    def test_fcf_margin(self, mock_ticker_class):
        """fcf_margin = free_cash_flow / revenue * 100."""
        mock_ticker_class.return_value = _make_mock_ticker(_FULL_INFO)
        data = collect_ticker_data("TEST")
        f = data["fundamentals"]
        expected = 100_000_000_000 / 400_000_000_000 * 100
        assert abs(f["fcf_margin"] - expected) < 0.01

    @patch("concilium.data.yf.Ticker")
    def test_net_debt_to_ebitda(self, mock_ticker_class):
        """net_debt_to_ebitda = net_debt / ebitda."""
        mock_ticker_class.return_value = _make_mock_ticker(_FULL_INFO)
        data = collect_ticker_data("TEST")
        f = data["fundamentals"]
        expected = 60_000_000_000 / 130_000_000_000
        assert abs(f["net_debt_to_ebitda"] - expected) < 0.001

    @patch("concilium.data.yf.Ticker")
    def test_forward_pe(self, mock_ticker_class):
        """forward_pe = currentPrice / forwardEps."""
        mock_ticker_class.return_value = _make_mock_ticker(_FULL_INFO)
        data = collect_ticker_data("TEST")
        f = data["fundamentals"]
        expected = 180.0 / 7.0
        assert abs(f["forward_pe"] - expected) < 0.01

    @patch("concilium.data.yf.Ticker")
    def test_net_debt_none_when_partial(self, mock_ticker_class):
        """net_debt ist None, wenn total_debt oder total_cash fehlt."""
        info = {**_FULL_INFO, "totalDebt": None, "totalCash": 60e9}
        mock_ticker_class.return_value = _make_mock_ticker(info)
        data = collect_ticker_data("TEST")
        assert data["fundamentals"]["net_debt"] is None

    @patch("concilium.data.yf.Ticker")
    def test_fcf_margin_none_when_revenue_missing(self, mock_ticker_class):
        """fcf_margin ist None, wenn revenue fehlt oder 0 ist."""
        info = {**_FULL_INFO, "totalRevenue": 0}
        mock_ticker_class.return_value = _make_mock_ticker(info)
        data = collect_ticker_data("TEST")
        assert data["fundamentals"]["fcf_margin"] is None

    @patch("concilium.data.yf.Ticker")
    def test_net_debt_to_ebitda_none_when_ebitda_zero(self, mock_ticker_class):
        """net_debt_to_ebitda ist None, wenn ebitda 0 oder negativ ist."""
        info = {**_FULL_INFO, "ebitda": 0}
        mock_ticker_class.return_value = _make_mock_ticker(info)
        data = collect_ticker_data("TEST")
        assert data["fundamentals"]["net_debt_to_ebitda"] is None

    @patch("concilium.data.yf.Ticker")
    def test_forward_pe_none_when_eps_zero(self, mock_ticker_class):
        """forward_pe ist None, wenn forward_eps <= 0."""
        info = {**_FULL_INFO, "forwardEps": 0}
        mock_ticker_class.return_value = _make_mock_ticker(info)
        data = collect_ticker_data("TEST")
        assert data["fundamentals"]["forward_pe"] is None


class TestPegConsistencyWarning:
    """PEG-Konsistenz-Warnung wird korrekt gesetzt."""

    @patch("concilium.data.yf.Ticker")
    def test_warning_set_when_negative_growth(self, mock_ticker_class):
        """PEG-Warnung gesetzt, wenn PEG vorhanden und revenue_growth < 0."""
        info = {**_FULL_INFO, "pegRatio": 1.5, "revenueGrowth": -0.05}
        mock_ticker_class.return_value = _make_mock_ticker(info)
        data = collect_ticker_data("TEST")
        w = data["fundamentals"]["peg_konsistenz_warnung"]
        assert w is not None
        assert "negativem Umsatzwachstum" in w

    @patch("concilium.data.yf.Ticker")
    def test_no_warning_when_positive_growth(self, mock_ticker_class):
        """Keine PEG-Warnung bei positivem Umsatzwachstum."""
        info = {**_FULL_INFO, "pegRatio": 1.5, "revenueGrowth": 0.08}
        mock_ticker_class.return_value = _make_mock_ticker(info)
        data = collect_ticker_data("TEST")
        assert data["fundamentals"]["peg_konsistenz_warnung"] is None

    @patch("concilium.data.yf.Ticker")
    def test_no_warning_when_peg_missing(self, mock_ticker_class):
        """Keine PEG-Warnung, wenn peg_ratio fehlt."""
        info = {**_FULL_INFO, "pegRatio": None, "revenueGrowth": -0.05}
        mock_ticker_class.return_value = _make_mock_ticker(info)
        data = collect_ticker_data("TEST")
        assert data["fundamentals"]["peg_konsistenz_warnung"] is None


class TestValidateFundamentalsExtended:
    """_validate_fundamentals crasht nicht mit neuen Feldern."""

    def test_validate_with_new_fields_no_crash(self):
        """Neue Felder werden von _validate_fundamentals nicht als Fehler gewertet."""
        fundamentals = {
            "dividend_yield": 0.01,
            "profit_margin": 0.2,
            "market_cap": 2e12,
            "revenue": 1e11,
            "free_cash_flow": 1e10,
            "net_debt": 5e9,
            "fcf_margin": 10.0,
            "net_debt_to_ebitda": 1.5,
            "forward_pe": 25.0,
            "peg_konsistenz_warnung": None,
        }
        warnings = _validate_fundamentals(fundamentals)
        # Sollte keine Warnung auslösen (alle Werte plausibel)
        assert warnings == []

    def test_validate_with_peg_warning_string(self):
        """peg_konsistenz_warnung als String crasht nicht."""
        fundamentals = {
            "dividend_yield": 0.01,
            "profit_margin": 0.2,
            "market_cap": 2e12,
            "revenue": 1e11,
            "peg_konsistenz_warnung": "PEG positiv trotz negativem Umsatzwachstum",
        }
        warnings = _validate_fundamentals(fundamentals)
        # Die PEG-Warnung ist ein String, kein numerischer Wert → kein Crash
        assert isinstance(warnings, list)


class TestFactorsExtended:
    """factors.py nutzt die neuen Felder für Quality-Score."""

    def test_quality_score_improves_with_fcf_margin(self):
        """Hohe FCF-Marge verbessert den Quality-Score."""
        base = {
            "profit_margin": 0.10,
            "revenue_growth": 0.05,
        }
        without_fcf = compute_multi_factor_score(base)
        with_fcf = compute_multi_factor_score({**base, "fcf_margin": 25.0})
        assert with_fcf["quality_score"] > without_fcf["quality_score"]

    def test_quality_score_decreases_with_high_leverage(self):
        """Hohe Nettoverschuldung/EBITDA verschlechtert den Quality-Score."""
        base = {
            "profit_margin": 0.10,
            "revenue_growth": 0.05,
            "fcf_margin": 5.0,
        }
        low_debt = compute_multi_factor_score({**base, "net_debt_to_ebitda": 0.5})
        high_debt = compute_multi_factor_score({**base, "net_debt_to_ebitda": 6.0})
        assert low_debt["quality_score"] > high_debt["quality_score"]

    def test_quality_score_net_cash_is_strong(self):
        """Negative Nettoverschuldung (Nettoliquidität) → Top-Score."""
        base = {
            "profit_margin": 0.10,
            "revenue_growth": 0.05,
            "fcf_margin": 5.0,
            "net_debt_to_ebitda": -2.0,
        }
        result = compute_multi_factor_score(base)
        # Mit Net-Cash + FCF + Marge + Wachstum → Score sollte >= 3.5
        assert result["quality_score"] is not None
        assert result["quality_score"] >= 3.5

    def test_quality_score_unchanged_without_new_fields(self):
        """Ohne neue Felder ist der Quality-Score unverändert (nur margin+growth+div)."""
        base = {
            "profit_margin": 0.20,
            "revenue_growth": 0.10,
            "dividend_yield": 0.03,
        }
        result = compute_multi_factor_score(base)
        # 3 Komponenten: margin(4) + growth(4) + div(4) = 4.0
        assert result["quality_score"] == 4.0


class TestBuildDataTextExtended:
    """_build_data_text zeigt die neuen Felder im FUNDAMENTALS-Abschnitt."""

    def test_new_fields_in_text(self):
        """Die neuen Kennzahlen tauchen im FUNDAMENTALS-Text auf."""
        data = {
            "ticker": "TEST",
            "fundamentals": {
                "name": "Test Inc.",
                "sector": "Tech",
                "industry": "Software",
                "free_cash_flow": 1e10,
                "net_debt": 5e9,
                "fcf_margin": 15.0,
                "net_debt_to_ebitda": 1.5,
                "forward_pe": 20.0,
                "ebitda": 3e10,
                "current_ratio": 1.5,
                "return_on_equity": 0.30,
                "gross_margin": 0.50,
                "operating_margin": 0.25,
                "price_to_book": 5.0,
                "book_value": 30.0,
                "forward_eps": 5.0,
            },
            "technicals": {},
            "sentiment": {},
            "news": [],
            "macro": {},
            "peers": [],
        }
        text = _build_data_text(data, role="fundamental")
        assert "Free Cash Flow" in text
        assert "Nettoverschuldung" in text
        assert "FCF-Marge" in text
        assert "Net-Debt/EBITDA" in text
        assert "Forward KGV" in text
        assert "EBITDA" in text
        assert "Current Ratio" in text
        assert "ROE" in text
        assert "Bruttomarge" in text
        assert "Price-to-Book" in text

    def test_peg_warning_in_text(self):
        """PEG-Konsistenz-Warnung taucht im Text auf, wenn gesetzt."""
        data = {
            "ticker": "TEST",
            "fundamentals": {
                "name": "Test Inc.",
                "sector": "Tech",
                "industry": "Software",
                "peg_konsistenz_warnung": "PEG positiv trotz negativem Umsatzwachstum — plausibel prüfen",
            },
            "technicals": {},
            "sentiment": {},
            "news": [],
            "macro": {},
            "peers": [],
        }
        text = _build_data_text(data, role="fundamental")
        assert "PEG-Konsistenz" in text
        assert "negativem Umsatzwachstum" in text

    def test_no_peg_warning_when_none(self):
        """Keine PEG-Warnung im Text, wenn None."""
        data = {
            "ticker": "TEST",
            "fundamentals": {
                "name": "Test Inc.",
                "sector": "Tech",
                "industry": "Software",
                "peg_konsistenz_warnung": None,
            },
            "technicals": {},
            "sentiment": {},
            "news": [],
            "macro": {},
            "peers": [],
        }
        text = _build_data_text(data, role="fundamental")
        assert "PEG-Konsistenz" not in text
