"""Tests für Datenqualitäts-Validierung und Mrd/Bio-Formatierung.

Prüft:
  - _validate_fundamentals: ADR-Fehler, plausible Werte, None-Werte
  - _fmt / _fmt_num: Mrd/Bio-Schwellen korrekt
  - collect_ticker_data: data_warnings-Schlüssel vorhanden
"""

from __future__ import annotations

import os
import sys

import pytest

# src zum Pfad hinzufügen
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from concilium.agents import _fmt_num  # noqa: E402
from concilium.data import _validate_fundamentals  # noqa: E402
from concilium.report import _fmt  # noqa: E402

# ---------------------------------------------------------------------------
# _validate_fundamentals
# ---------------------------------------------------------------------------


class TestValidateFundamentals:
    """Tests für die Datenqualitäts-Validierung."""

    def test_validate_fundamentals_dividend_yield(self):
        """ADR-Fehler: dividend_yield=1.05 (105%) muss Warnung auslösen."""
        fundamentals = {
            "dividend_yield": 1.05,
            "profit_margin": 0.2,
            "market_cap": 2e12,
            "revenue": 1e11,
        }
        warnings = _validate_fundamentals(fundamentals)
        assert len(warnings) >= 1
        assert any("Dividendenrendite" in w or "unplausibel" in w.lower() for w in warnings)

    def test_validate_fundamentals_clean(self):
        """Normale Werte dürfen keine Warnung auslösen."""
        fundamentals = {
            "dividend_yield": 0.01,
            "profit_margin": 0.2,
            "market_cap": 2e12,
            "revenue": 1e11,
        }
        warnings = _validate_fundamentals(fundamentals)
        assert warnings == [], f"Unerwartete Warnungen: {warnings}"

    def test_validate_missing_values(self):
        """None-Werte dürfen nicht crashen und keine Warnung auslösen."""
        fundamentals = {
            "dividend_yield": None,
            "profit_margin": None,
            "market_cap": None,
            "revenue": None,
        }
        warnings = _validate_fundamentals(fundamentals)
        assert warnings == [], f"Unerwartete Warnungen bei None: {warnings}"

    def test_validate_profit_margin_too_high(self):
        """Gewinnmarge > 100% muss warnen."""
        fundamentals = {"profit_margin": 1.5}
        warnings = _validate_fundamentals(fundamentals)
        assert any("Gewinnmarge" in w for w in warnings)

    def test_validate_low_market_cap(self):
        """Marktkap < 100 Mio muss warnen."""
        fundamentals = {"market_cap": 5e7}
        warnings = _validate_fundamentals(fundamentals)
        assert any("Marktkapitalisierung" in w for w in warnings)

    def test_validate_high_valuation_ratio(self):
        """market_cap/revenue > 50 muss als Hinweis warnen."""
        fundamentals = {"market_cap": 1e12, "revenue": 1e9}
        warnings = _validate_fundamentals(fundamentals)
        assert any("Bewertung" in w for w in warnings)

    def test_validate_empty_dict(self):
        """Leeres dict darf nicht crashen und keine Warnung auslösen."""
        warnings = _validate_fundamentals({})
        assert warnings == []


# ---------------------------------------------------------------------------
# _fmt (report.py) — Mrd/Bio-Formatierung
# ---------------------------------------------------------------------------


class TestFmtBioMrd:
    """Tests für die Mrd/Bio-Schwellen in _fmt (report.py)."""

    def test_fmt_bio(self):
        """1e12 → Bio (deutsche Billion)."""
        result = _fmt(2.16e12)
        assert result.endswith("Bio"), f"Erwartet '…Bio', got '{result}'"
        assert result == "2.16 Bio"

    def test_fmt_mrd(self):
        """4.4e10 → Mrd (deutsche Milliarde)."""
        result = _fmt(4.4e10)
        assert result.endswith("Mrd"), f"Erwartet '…Mrd', got '{result}'"
        assert result == "44.00 Mrd"

    def test_fmt_mio(self):
        """1e6 → Mio."""
        result = _fmt(5e6)
        assert result.endswith("Mio")

    def test_fmt_k(self):
        """1e3 → K."""
        result = _fmt(2500)
        assert result.endswith("K")

    def test_fmt_small(self):
        """< 1e3 → rohe Zahl."""
        result = _fmt(42.5)
        assert "K" not in result
        assert "Mio" not in result

    def test_fmt_none(self):
        """None → N/A."""
        assert _fmt(None) == "N/A"


# ---------------------------------------------------------------------------
# _fmt_num (agents.py) — Mrd/Bio-Formatierung
# ---------------------------------------------------------------------------


class TestFmtNumBioMrd:
    """Tests für die Mrd/Bio-Schwellen in _fmt_num (agents.py)."""

    def test_fmt_num_bio(self):
        """1e12 → Bio (nicht 'B')."""
        result = _fmt_num(2.16e12)
        assert "Bio" in result, f"Erwartet 'Bio' in '{result}'"
        assert "B" != result.strip().split()[-1], "Sollte nicht 'B' als Endung haben"

    def test_fmt_num_mrd(self):
        """4.4e10 → Mrd."""
        result = _fmt_num(4.4e10)
        assert "Mrd" in result

    def test_fmt_num_none(self):
        """None → N/A."""
        assert _fmt_num(None) == "N/A"


# ---------------------------------------------------------------------------
# collect_ticker_data — data_warnings-Schlüssel
# ---------------------------------------------------------------------------


def _has_network() -> bool:
    """Prüft, ob yfinance Daten abrufen kann."""
    try:
        import yfinance as yf

        t = yf.Ticker("AAPL")
        hist = t.history(period="5d")
        return hist is not None and not hist.empty
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _has_network(), reason="Kein Netzwerkzugriff für yfinance")
class TestDataWarningsKey:
    """Tests für data_warnings in collect_ticker_data."""

    def test_data_warnings_key_tsm(self):
        """TSM muss data_warnings als Liste zurückgeben."""
        from concilium.data import collect_ticker_data

        data = collect_ticker_data("TSM")
        assert "data_warnings" in data
        assert isinstance(data["data_warnings"], list)

    def test_data_warnings_key_aapl(self):
        """AAPL muss data_warnings als Liste zurückgeben (ohne Crash)."""
        from concilium.data import collect_ticker_data

        data = collect_ticker_data("AAPL")
        assert "data_warnings" in data
        assert isinstance(data["data_warnings"], list)
