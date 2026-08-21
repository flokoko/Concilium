"""Tests für rollenspezifischen Datenkontext und Stimme/Score-Konsistenz-Wächter.

Feature A: _build_data_text mit role-Parameter filtert Sektionen rollenspezifisch.
Feature B: _analyst_consistency_warning erkennt inkonsistente Stimmung/Score-Kombinationen.

Alle Tests sind offline (kein Netzwerk) — der LLMClient wird gemockt.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import (  # noqa: E402
    _analyst_consistency_warning,
    _build_data_text,
    analyst_team,
)

# --- Fixtures ---

_FULL_DATA = {
    "ticker": "TEST",
    "fundamentals": {
        "name": "Test Inc.",
        "sector": "Technology",
        "industry": "Semiconductors",
        "market_cap": 1_000_000_000,
        "pe_ratio": 25.0,
        "eps": 4.0,
        "revenue": 500_000_000,
        "revenue_growth": 0.15,
        "profit_margin": 0.20,
        "peg_ratio": 1.5,
        "dividend_yield": 0.02,
        "beta": 1.2,
        "fifty_two_week_high": 60.0,
        "fifty_two_week_low": 30.0,
        "recommendation_key": "buy",
        "recommendation_mean": 2.1,
        "analyst_count": 15,
        "analyst_target_mean": 55.0,
        "analyst_target_high": 70.0,
        "analyst_target_low": 45.0,
        "analyst_upside_pct": 0.10,
    },
    "technicals": {
        "current_price": 50.0,
        "sma50": 48.0,
        "sma200": 45.0,
        "rsi14": 55.0,
        "macd": {"macd": 0.5, "signal": 0.3},
        "bollinger": {"lower": 44.0, "middle": 50.0, "upper": 56.0, "position": 0.5},
        "current_volume": 1_000_000,
        "avg_volume_30d": 1_200_000,
    },
    "sentiment": {
        "positiv": 5,
        "negativ": 2,
        "neutral": 3,
        "dominant": "positiv",
        "sample_size": 10,
        "weighted": False,
    },
    "news": ["Test headline 1", "Test headline 2"],
    "macro": {
        "us_10y_yield": 4.2,
        "us_10y_yield_1m_ago": 4.0,
        "us_10y_trend": "steigend",
        "sp500_pe": 22.0,
        "sp500_market_cap": 45_000_000_000_000,
        "sp500_source": "none",
    },
    "peers": [
        {"ticker": "PEER1", "pe_ratio": 20.0, "market_cap": 800_000_000, "name": "Peer One"},
    ],
    "data_warnings": ["ADR-Risiko: EPS möglicherweise verzerrt"],
}


# ===========================================================================
# Feature A: _build_data_text mit role-Parameter
# ===========================================================================

class TestBuildDataTextRoleFundamental:
    """role='fundamental': FUNDAMENTALS ja, TECHNIK nein, SENTIMENT nein."""

    def test_contains_fundamentals(self):
        text = _build_data_text(_FULL_DATA, role="fundamental")
        assert "=== FUNDAMENTALS ===" in text

    def test_no_technik_section(self):
        text = _build_data_text(_FULL_DATA, role="fundamental")
        assert "=== TECHNIK ===" not in text

    def test_no_sentiment_section(self):
        text = _build_data_text(_FULL_DATA, role="fundamental")
        assert "=== SENTIMENT ===" not in text

    def test_contains_datenqualitaet(self):
        text = _build_data_text(_FULL_DATA, role="fundamental")
        assert "=== DATENQUALITÄTS-WARNUNGEN ===" in text

    def test_contains_makro(self):
        text = _build_data_text(_FULL_DATA, role="fundamental")
        assert "=== MAKRO / ZINSEN ===" in text

    def test_contains_peer_vergleich(self):
        text = _build_data_text(_FULL_DATA, role="fundamental")
        assert "=== PEER-VERGLEICH ===" in text

    def test_contains_stock_identity(self):
        text = _build_data_text(_FULL_DATA, role="fundamental")
        assert "Aktie: TEST" in text
        assert "Sektor: Technology" in text


class TestBuildDataTextRoleTechnik:
    """role='technik': TECHNIK ja, FUNDAMENTALS nein, SENTIMENT nein."""

    def test_contains_technik(self):
        text = _build_data_text(_FULL_DATA, role="technik")
        assert "=== TECHNIK ===" in text

    def test_no_fundamentals_section(self):
        text = _build_data_text(_FULL_DATA, role="technik")
        assert "=== FUNDAMENTALS ===" not in text

    def test_no_sentiment_section(self):
        text = _build_data_text(_FULL_DATA, role="technik")
        assert "=== SENTIMENT ===" not in text

    def test_contains_current_price(self):
        text = _build_data_text(_FULL_DATA, role="technik")
        assert "Aktueller Kurs" in text

    def test_contains_makro_kurz(self):
        text = _build_data_text(_FULL_DATA, role="technik")
        assert "=== MAKRO / ZINSEN (Kurz) ===" in text

    def test_no_peer_vergleich(self):
        text = _build_data_text(_FULL_DATA, role="technik")
        assert "=== PEER-VERGLEICH ===" not in text

    def test_contains_stock_identity(self):
        text = _build_data_text(_FULL_DATA, role="technik")
        assert "Aktie: TEST" in text

    def test_no_datenqualitaet(self):
        text = _build_data_text(_FULL_DATA, role="technik")
        assert "=== DATENQUALITÄTS-WARNUNGEN ===" not in text


class TestBuildDataTextRoleSentiment:
    """role='sentiment': SENTIMENT ja, FUNDAMENTALS nein, TECHNIK nein."""

    def test_contains_sentiment(self):
        text = _build_data_text(_FULL_DATA, role="sentiment")
        assert "=== SENTIMENT ===" in text

    def test_no_technik_section(self):
        text = _build_data_text(_FULL_DATA, role="sentiment")
        assert "=== TECHNIK ===" not in text

    def test_no_fundamentals_section(self):
        text = _build_data_text(_FULL_DATA, role="sentiment")
        assert "=== FUNDAMENTALS ===" not in text

    def test_contains_headlines(self):
        text = _build_data_text(_FULL_DATA, role="sentiment")
        assert "Test headline 1" in text

    def test_contains_stock_identity(self):
        text = _build_data_text(_FULL_DATA, role="sentiment")
        assert "Aktie: TEST" in text

    def test_no_makro(self):
        text = _build_data_text(_FULL_DATA, role="sentiment")
        assert "=== MAKRO" not in text

    def test_no_peer_vergleich(self):
        text = _build_data_text(_FULL_DATA, role="sentiment")
        assert "=== PEER-VERGLEICH ===" not in text


class TestBuildDataTextDefault:
    """Default (kein role): alle Sektionen (rückwärtskompatibel)."""

    def test_contains_fundamentals(self):
        text = _build_data_text(_FULL_DATA)
        assert "=== FUNDAMENTALS ===" in text

    def test_contains_technik(self):
        text = _build_data_text(_FULL_DATA)
        assert "=== TECHNIK ===" in text

    def test_contains_sentiment(self):
        text = _build_data_text(_FULL_DATA)
        assert "=== SENTIMENT ===" in text

    def test_contains_makro(self):
        text = _build_data_text(_FULL_DATA)
        assert "=== MAKRO / ZINSEN ===" in text

    def test_contains_peer_vergleich(self):
        text = _build_data_text(_FULL_DATA)
        assert "=== PEER-VERGLEICH ===" in text

    def test_contains_datenqualitaet(self):
        text = _build_data_text(_FULL_DATA)
        assert "=== DATENQUALITÄTS-WARNUNGEN ===" in text

    def test_explicit_alle_same_as_default(self):
        """role='alle' liefert dasselbe wie Default (kein role-Argument)."""
        text_default = _build_data_text(_FULL_DATA)
        text_alle = _build_data_text(_FULL_DATA, role="alle")
        assert text_default == text_alle


# ===========================================================================
# Feature B: _analyst_consistency_warning
# ===========================================================================

class TestAnalystConsistencyWarning:
    """Tests für den Stimme/Score-Konsistenz-Wächter."""

    def test_bullish_low_score_warns(self):
        """bullish + score 1 → Warnung."""
        result = _analyst_consistency_warning("bullish", 1)
        assert result != ""
        assert "bullish" in result.lower() or "inkonsistent" in result.lower()

    def test_bearish_high_score_warns(self):
        """bearish + score 5 → Warnung."""
        result = _analyst_consistency_warning("bearish", 5)
        assert result != ""
        assert "bearish" in result.lower() or "inkonsistent" in result.lower()

    def test_bullish_good_score_no_warning(self):
        """bullish + score 4 → keine Warnung (leerer String)."""
        result = _analyst_consistency_warning("bullish", 4)
        assert result == ""

    def test_bearish_low_score_no_warning(self):
        """bearish + score 2 → keine Warnung."""
        result = _analyst_consistency_warning("bearish", 2)
        assert result == ""

    def test_neutral_mid_score_no_warning(self):
        """neutral + score 3 → keine Warnung."""
        result = _analyst_consistency_warning("neutral", 3)
        assert result == ""

    def test_neutral_extreme_score_warns_low(self):
        """neutral + score 1 → Warnung."""
        result = _analyst_consistency_warning("neutral", 1)
        assert result != ""

    def test_neutral_extreme_score_warns_high(self):
        """neutral + score 5 → Warnung."""
        result = _analyst_consistency_warning("neutral", 5)
        assert result != ""

    def test_bullish_score_2_no_warning(self):
        """bullish + score 2 → keine Warnung (Grenze ist <=1)."""
        result = _analyst_consistency_warning("bullish", 2)
        assert result == ""

    def test_bearish_score_3_no_warning(self):
        """bearish + score 3 → keine Warnung (Grenze ist >=4)."""
        result = _analyst_consistency_warning("bearish", 3)
        assert result == ""

    def test_neutral_score_2_no_warning(self):
        """neutral + score 2 → keine Warnung."""
        result = _analyst_consistency_warning("neutral", 2)
        assert result == ""

    def test_neutral_score_4_no_warning(self):
        """neutral + score 4 → keine Warnung."""
        result = _analyst_consistency_warning("neutral", 4)
        assert result == ""

    def test_invalid_score_returns_empty(self):
        """Ungültiger Score (None/String) → leerer String, kein Crash."""
        assert _analyst_consistency_warning("bullish", None) == ""
        assert _analyst_consistency_warning("bullish", "abc") == ""

    def test_invalid_stimmung_returns_empty(self):
        """Unbekannte Stimmung → leerer String."""
        assert _analyst_consistency_warning("unknown", 3) == ""


# ===========================================================================
# Feature B: analyst_team injiziert konsistenz_warnung
# ===========================================================================

class _FakeLLM:
    """Einfacher Mock-LLM: gibt vordefinierte JSON-Antworten zurück.

    Dispatcht basierend auf dem System-Prompt-Inhalt, damit jeder Analyst
    eine andere (inkonsistente) Antwort bekommt.
    """

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3, **kwargs) -> str:
        role = messages[0]["content"]
        if "Fundamental" in role:
            # Inkonsistent: bullish + score 1
            return json.dumps({
                "rolle": "Fundamental-Analyst",
                "stimmung": "bullish",
                "score": 1,
                "zusammenfassung": "Gut aber Score inkonsistent",
            })
        if "technisch" in role or "Technik" in role:
            # Konsistent: neutral + score 3
            return json.dumps({
                "rolle": "Technik-Analyst",
                "stimmung": "neutral",
                "score": 3,
                "zusammenfassung": "Seitwärts",
            })
        if "Sentiment" in role:
            # Inkonsistent: bearish + score 5
            return json.dumps({
                "rolle": "Sentiment-Analyst",
                "stimmung": "bearish",
                "score": 5,
                "zusammenfassung": "Negativ aber Score inkonsistent",
            })
        return json.dumps({"rolle": "Unknown", "stimmung": "neutral", "score": 3})


_MINIMAL_DATA = {
    "ticker": "TEST",
    "fundamentals": {"name": "Test Inc.", "sector": "Tech"},
    "technicals": {"current_price": 50.0},
    "sentiment": {"positiv": 1, "negativ": 0, "neutral": 1},
}


class TestAnalystTeamConsistencyWarning:
    """analyst_team injiziert konsistenz_warnung bei inkonsistenten Analyst-Resultaten."""

    def test_inconsistent_analyst_gets_warning(self):
        """Fundamental (bullish+1) und Sentiment (bearish+5) bekommen Warnung."""
        result = analyst_team(_MINIMAL_DATA, _FakeLLM())

        # Fundamental: bullish + score 1 → inkonsistent
        assert "konsistenz_warnung" in result["fundamental"]
        assert result["fundamental"]["konsistenz_warnung"] != ""

    def test_consistent_analyst_no_warning(self):
        """Technical (neutral+3) bekommt KEINE Warnung."""
        result = analyst_team(_MINIMAL_DATA, _FakeLLM())

        # Technical: neutral + score 3 → konsistent → kein Feld
        assert "konsistenz_warnung" not in result["technical"]

    def test_sentiment_inconsistent_gets_warning(self):
        """Sentiment (bearish+5) bekommt Warnung."""
        result = analyst_team(_MINIMAL_DATA, _FakeLLM())

        assert "konsistenz_warnung" in result["sentiment"]
        assert result["sentiment"]["konsistenz_warnung"] != ""

    def test_all_three_analysts_present(self):
        """Alle 3 Analysten-Keys sind vorhanden."""
        result = analyst_team(_MINIMAL_DATA, _FakeLLM())
        assert "fundamental" in result
        assert "technical" in result
        assert "sentiment" in result
        assert "technicals" in result
