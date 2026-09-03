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
# Roadmap C2: TECHNIK-Block als verbindlicher Markt-Snapshot (Ground-Truth)
# ===========================================================================


class TestBuildDataTextTechnikGroundTruth:
    """Der TECHNIK-Block ist als verbindliche Quelle (Ground-Truth) markiert."""

    def test_alle_contains_ground_truth_marker(self):
        """role='alle': Ground-Truth-Kennzeichnung im TECHNIK-Block."""
        text = _build_data_text(_FULL_DATA, role="alle")
        assert "=== TECHNIK ===" in text
        assert "VERBINDLICHE QUELLE" in text
        assert "verifizierte Markt-Snapshot" in text
        assert "erfinde keine abweichenden Werte" in text

    def test_technik_contains_ground_truth_marker(self):
        """role='technik': Ground-Truth-Kennzeichnung im TECHNIK-Block."""
        text = _build_data_text(_FULL_DATA, role="technik")
        assert "=== TECHNIK ===" in text
        assert "VERBINDLICHE QUELLE" in text
        assert "verifizierte Markt-Snapshot" in text

    def test_ground_truth_marker_right_after_header(self):
        """Die Kennzeichnung steht direkt nach dem '=== TECHNIK ==='-Header."""
        text = _build_data_text(_FULL_DATA, role="technik")
        header_pos = text.index("=== TECHNIK ===")
        marker_pos = text.index("VERBINDLICHE QUELLE")
        kurs_pos = text.index("Aktueller Kurs")
        assert header_pos < marker_pos < kurs_pos

    def test_no_ground_truth_marker_without_technik_role(self):
        """Rollen ohne TECHNIK-Sektion zeigen die Kennzeichnung nicht."""
        for role in ("fundamental", "sentiment", "macro_news"):
            text = _build_data_text(_FULL_DATA, role=role)
            assert "VERBINDLICHE QUELLE" not in text, f"role={role}"
            assert "=== TECHNIK ===" not in text, f"role={role}"


# ===========================================================================
# Roadmap C3: INSTRUMENT-KONTEXT-Block im Prolog (alle Rollen)
# ===========================================================================


_FULL_DATA_WITH_IDENTITY = {
    **_FULL_DATA,
    "fundamentals": {
        **_FULL_DATA["fundamentals"],
        "currency": "USD",
        "exchange": "NMS",
        "full_exchange_name": "NasdaqGS",
        "country": "United States",
        "quote_type": "EQUITY",
        "market": "us_market",
        "instrument_type": "Aktie",
    },
}


class TestBuildDataTextInstrumentContext:
    """Der INSTRUMENT-KONTEXT-Block erscheint rollenunabhängig im Prolog."""

    def test_instrument_context_in_alle(self):
        """role='alle': Block mit Typ, Börse, Land, Währung, Markt."""
        text = _build_data_text(_FULL_DATA_WITH_IDENTITY, role="alle")
        assert "=== INSTRUMENT-KONTEXT ===" in text
        assert "Typ: Aktie" in text
        assert "Börse: NasdaqGS" in text  # full_exchange_name hat Vorrang
        assert "Land: United States" in text
        assert "Währung: USD" in text
        assert "Markt: us_market" in text

    def test_instrument_context_in_all_roles(self):
        """Der Block erscheint für ALLE Rollen (Prolog, kein role-Guard)."""
        for role in ("alle", "fundamental", "technik", "sentiment", "macro_news"):
            text = _build_data_text(_FULL_DATA_WITH_IDENTITY, role=role)
            assert "=== INSTRUMENT-KONTEXT ===" in text, f"role={role}"
            assert "Land: United States" in text, f"role={role}"

    def test_exchange_fallback_when_full_name_missing(self):
        """Ohne full_exchange_name wird exchange angezeigt."""
        data = {
            **_FULL_DATA,
            "fundamentals": {
                **_FULL_DATA["fundamentals"],
                "exchange": "GER",
                "country": "Germany",
            },
        }
        text = _build_data_text(data, role="alle")
        assert "=== INSTRUMENT-KONTEXT ===" in text
        assert "Börse: GER" in text
        assert "Land: Germany" in text

    def test_missing_fields_no_crash_and_no_block(self):
        """Fehlende/None-Felder → kein Crash, Block wird weggelassen."""
        data = {
            **_FULL_DATA,
            "fundamentals": {k: v for k, v in _FULL_DATA["fundamentals"].items()},
        }
        # Keine Identity-Felder gesetzt
        text = _build_data_text(data, role="alle")
        assert "=== INSTRUMENT-KONTEXT ===" not in text

    def test_partial_fields_show_only_available(self):
        """Nur gesetzte Felder werden angezeigt (keine N/A-Reste)."""
        data = {
            **_FULL_DATA,
            "fundamentals": {
                **_FULL_DATA["fundamentals"],
                "instrument_type": "ETF",
            },
        }
        text = _build_data_text(data, role="alle")
        assert "=== INSTRUMENT-KONTEXT ===" in text
        assert "Typ: ETF" in text
        assert "Börse:" not in text
        assert "Land:" not in text

    def test_none_values_no_crash(self):
        """Explizit None-Werte → kein Crash, Block wird weggelassen."""
        data = {
            **_FULL_DATA,
            "fundamentals": {
                **_FULL_DATA["fundamentals"],
                "exchange": None,
                "full_exchange_name": None,
                "country": None,
                "currency": None,
                "market": None,
                "instrument_type": None,
            },
        }
        text = _build_data_text(data, role="alle")
        assert "=== INSTRUMENT-KONTEXT ===" not in text

    def test_block_does_not_duplicate_stock_identity(self):
        """Der Block fokussiert neue Felder und dupliziert nicht Name/Sektor."""
        text = _build_data_text(_FULL_DATA_WITH_IDENTITY, role="alle")
        instrument_part = text.split("=== INSTRUMENT-KONTEXT ===")[1].split("===")[0]
        assert "Sektor" not in instrument_part
        assert "Typ:" in instrument_part


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

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3, **kwargs) -> str | object:
        role = messages[0]["content"]
        if "Fundamental" in role:
            # Inkonsistent: bullish + score 1
            text = json.dumps({
                "rolle": "Fundamental-Analyst",
                "stimmung": "bullish",
                "score": 1,
                "zusammenfassung": "Gut aber Score inkonsistent",
            })
        elif "technisch" in role or "Technik" in role:
            # Konsistent: neutral + score 3
            text = json.dumps({
                "rolle": "Technik-Analyst",
                "stimmung": "neutral",
                "score": 3,
                "zusammenfassung": "Seitwärts",
            })
        elif "Sentiment" in role:
            # Inkonsistent: bearish + score 5
            text = json.dumps({
                "rolle": "Sentiment-Analyst",
                "stimmung": "bearish",
                "score": 5,
                "zusammenfassung": "Negativ aber Score inkonsistent",
            })
        else:
            text = json.dumps({"rolle": "Unknown", "stimmung": "neutral", "score": 3})
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult
            return StructuredChatResult(text=text, response_format_used=True)
        return text


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
        """Technical (neutral+3) bekommt KEINE Warnung (Default ist None)."""
        result = analyst_team(_MINIMAL_DATA, _FakeLLM())

        # Technical: neutral + score 3 → konsistent → konsistenz_warnung ist
        # None (Schema-Default, nicht vom Analysten gesetzt)
        assert result["technical"].get("konsistenz_warnung") is None

    def test_sentiment_inconsistent_gets_warning(self):
        """Sentiment (bearish+5) bekommt Warnung."""
        result = analyst_team(_MINIMAL_DATA, _FakeLLM())

        assert "konsistenz_warnung" in result["sentiment"]
        assert result["sentiment"]["konsistenz_warnung"] != ""

    def test_all_four_analysts_present(self):
        """Alle 4 Analysten-Keys sind vorhanden."""
        result = analyst_team(_MINIMAL_DATA, _FakeLLM())
        assert "fundamental" in result
        assert "technical" in result
        assert "sentiment" in result
        assert "macro_news" in result
        assert "technicals" in result
