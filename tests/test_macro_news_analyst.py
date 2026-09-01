"""Tests für den Makro/News-Analysten (Phase 3, 4. Analysten-Rolle).

Testet:
- analyst_team liefert 4 Analysten-Keys (fundamental, technical, sentiment,
  macro_news) + technicals.
- Der macro_news-Analyst bekommt die MAKRO-Sektion (vollständig) und die
  SENTIMENT-Sektion/Headlines, aber KEINE FUNDAMENTALS- oder TECHNIK-Sektion
  und keinen Währungsrisiko-Block.
- Ein Fehler beim macro_news-Analyst crasht analyst_team nicht (Fehlereintrag
  {"_raw": "", "fehler": "..."} analog zu den anderen Analysten).
- Der Report rendert die Makro/News-Sektion (Analysten-Tabelle + Makro/News-
  Einschätzung).
- _analyst_summary_text nimmt den 4. Analysten auf (für Debatte/Trader).
- ANALYST_MACRO_NEWS_SCHEMA funktioniert mit defaults_for_schema und
  validate_structured.

Alle Tests sind offline (kein Netzwerk) — der LLMClient wird gemockt.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import (  # noqa: E402
    _analyst_summary_text,
    _build_data_text,
    analyst_team,
)
from concilium.report import generate_report  # noqa: E402
from concilium.schemas import (  # noqa: E402
    ANALYST_MACRO_NEWS_SCHEMA,
    defaults_for_schema,
    validate_structured,
)

# --- Fixtures ---

_MACRO_NEWS_DATA = {
    "ticker": "TEST",
    "fundamentals": {
        "name": "Test Inc.",
        "sector": "Technology",
        "industry": "Semiconductors",
        "pe_ratio": 25.0,
        "eur_risiko": True,
        "currency": "USD",
    },
    "technicals": {
        "current_price": 50.0,
        "sma50": 48.0,
        "sma200": 45.0,
        "rsi14": 55.0,
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
        "eurusd": 1.08,
        "vix": 15.5,
        "oel_preis": 78.0,
        "sp500_trend": "aufwärts",
    },
    "peers": [
        {"ticker": "PEER1", "pe_ratio": 20.0, "market_cap": 800_000_000, "name": "Peer One"},
    ],
    "data_warnings": ["ADR-Risiko: EPS möglicherweise verzerrt"],
}


class _MacroNewsLLM:
    """Mock-LLM: dispatcht basierend auf dem System-Prompt-Inhalt.

    Gibt für jeden Analysten eine konsistente Antwort zurück und zeichnet
    alle User-Prompts auf, damit rollenspezifische Daten-Filter geprüft
    werden können.
    """

    def __init__(self):
        self.all_messages: list[list[dict]] = []

    def chat(self, messages, temperature=0.3, **kwargs):
        self.all_messages.append(messages)
        system = messages[0]["content"]
        if "Fundamental" in system:
            text = json.dumps({
                "rolle": "Fundamental-Analyst",
                "stimmung": "bullish",
                "score": 4,
                "zusammenfassung": "Gute Fundamentals",
            })
        elif "technisch" in system:
            text = json.dumps({
                "rolle": "Technik-Analyst",
                "stimmung": "neutral",
                "score": 3,
                "zusammenfassung": "Seitwärts",
            })
        elif "Makro" in system:
            text = json.dumps({
                "rolle": "Makro/News-Analyst",
                "stimmung": "neutral",
                "score": 3,
                "zusammenfassung": "Makro ruhig, News gemischt",
                "makro_einschaetzung": "Zinsen stabil, VIX niedrig",
                "relevante_headlines": "Headline 1 ist material",
            })
        elif "Sentiment" in system:
            text = json.dumps({
                "rolle": "Sentiment-Analyst",
                "stimmung": "bullish",
                "score": 4,
                "zusammenfassung": "Positiv",
                "dominant": "positiv",
            })
        else:
            text = json.dumps({"rolle": "Unknown", "stimmung": "neutral", "score": 3})
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult
            return StructuredChatResult(text=text, response_format_used=True)
        return text


# ===========================================================================
# (a) analyst_team liefert 4 Analysten-Keys
# ===========================================================================


class TestAnalystTeamFourKeys:
    """(a) analyst_team liefert 4 Analysten-Keys + technicals."""

    def test_four_analyst_keys_present(self):
        """fundamental, technical, sentiment, macro_news, technicals vorhanden."""
        result = analyst_team(_MACRO_NEWS_DATA, _MacroNewsLLM())
        assert "fundamental" in result
        assert "technical" in result
        assert "sentiment" in result
        assert "macro_news" in result
        assert "technicals" in result

    def test_macro_news_has_structured_keys(self):
        """macro_news-Ergebnis enthält die Schema-Keys inkl. Makro-Felder."""
        result = analyst_team(_MACRO_NEWS_DATA, _MacroNewsLLM())
        mn = result["macro_news"]
        assert mn["stimmung"] == "neutral"
        assert mn["score"] == 3
        assert mn["zusammenfassung"] == "Makro ruhig, News gemischt"
        assert mn["makro_einschaetzung"] == "Zinsen stabil, VIX niedrig"
        assert mn["relevante_headlines"] == "Headline 1 ist material"

    def test_macro_news_in_summary_for_debate(self):
        """_analyst_summary_text nimmt den Makro/News-Analysten auf."""
        result = analyst_team(_MACRO_NEWS_DATA, _MacroNewsLLM())
        summary = _analyst_summary_text(result)
        assert "Makro/News-Analyst:" in summary
        assert "Makro ruhig" in summary

    def test_four_calls_made(self):
        """4 LLM-Calls (einer pro Analyst)."""
        llm = _MacroNewsLLM()
        analyst_team(_MACRO_NEWS_DATA, llm)
        assert len(llm.all_messages) == 4


# ===========================================================================
# (b) macro_news-Analyst bekommt Makro+Sentiment, nicht Fundamentals/Technik
# ===========================================================================


class TestMacroNewsDataFilter:
    """(b) _build_data_text(role='macro_news') filtert rollenspezifisch."""

    def test_contains_stock_identity(self):
        text = _build_data_text(_MACRO_NEWS_DATA, role="macro_news")
        assert "Aktie: TEST" in text
        assert "Sektor: Technology" in text

    def test_contains_full_macro_section(self):
        text = _build_data_text(_MACRO_NEWS_DATA, role="macro_news")
        assert "=== MAKRO / ZINSEN ===" in text
        # Vollständig: auch die erweiterten Makro-Kennzahlen
        assert "10y US Treasury Yield" in text
        assert "10y Zinstrend" in text
        assert "EURUSD" in text
        assert "VIX" in text
        assert "Öl (WTI)" in text
        assert "S&P500-Trend" in text
        assert "S&P 500 KGV" in text

    def test_contains_sentiment_section(self):
        text = _build_data_text(_MACRO_NEWS_DATA, role="macro_news")
        assert "=== SENTIMENT ===" in text
        assert "Positive Headlines: 5" in text
        assert "Dominante Stimmung: positiv" in text

    def test_contains_headlines(self):
        text = _build_data_text(_MACRO_NEWS_DATA, role="macro_news")
        assert "Test headline 1" in text
        assert "Test headline 2" in text

    def test_no_fundamentals_section(self):
        text = _build_data_text(_MACRO_NEWS_DATA, role="macro_news")
        assert "=== FUNDAMENTALS ===" not in text
        # KGV aus den Fundamentals darf nicht auftauchen
        assert "KGV (trailing)" not in text

    def test_no_technik_section(self):
        text = _build_data_text(_MACRO_NEWS_DATA, role="macro_news")
        assert "=== TECHNIK ===" not in text
        assert "Aktueller Kurs" not in text

    def test_no_waehrungsrisiko_block(self):
        """Der Währungsrisiko-Block gehört zu fundamental/risk, nicht macro_news."""
        text = _build_data_text(_MACRO_NEWS_DATA, role="macro_news")
        assert "=== WÄHRUNGSRISIKO ===" not in text

    def test_no_datenqualitaet_warnings(self):
        text = _build_data_text(_MACRO_NEWS_DATA, role="macro_news")
        assert "=== DATENQUALITÄTS-WARNUNGEN ===" not in text

    def test_no_peer_vergleich(self):
        text = _build_data_text(_MACRO_NEWS_DATA, role="macro_news")
        assert "=== PEER-VERGLEICH ===" not in text

    def test_macro_news_analyst_receives_filtered_data(self):
        """End-to-End: Der macro_news-User-Prompt enthält Makro+Sentiment, nicht Fundamentals."""
        llm = _MacroNewsLLM()
        analyst_team(_MACRO_NEWS_DATA, llm, data_text=None)

        # Finde den macro_news-Call (System-Prompt enthält "Makro")
        user_texts = {
            msgs[0]["content"]: msgs[1]["content"] for msgs in llm.all_messages
        }
        mn_user = next(
            (u for s, u in user_texts.items() if "Makro" in s),
            None,
        )
        assert mn_user is not None, "Makro/News-Analyst-Call nicht gefunden"
        assert "=== MAKRO / ZINSEN ===" in mn_user
        assert "=== SENTIMENT ===" in mn_user
        assert "=== FUNDAMENTALS ===" not in mn_user
        assert "=== TECHNIK ===" not in mn_user

    def test_default_alle_unchanged(self):
        """Rückwärtskompatibilität: role='alle' zeigt weiterhin alle Sektionen."""
        text = _build_data_text(_MACRO_NEWS_DATA, role="alle")
        assert "=== FUNDAMENTALS ===" in text
        assert "=== TECHNIK ===" in text
        assert "=== MAKRO / ZINSEN ===" in text
        assert "=== SENTIMENT ===" in text
        # Währungsrisiko-Block bleibt in "alle" enthalten
        assert "=== WÄHRUNGSRISIKO ===" in text


# ===========================================================================
# (c) Fehler beim macro_news-Analyst crasht nicht
# ===========================================================================


class TestMacroNewsFailureResilience:
    """(c) macro_news-Fehler → Fehlereintrag, kein Crash, andere laufen weiter."""

    def test_macro_news_failure_does_not_crash(self):
        """macro_news-Call wirft → Fehlereintrag, andere Analysten normal."""

        class _PartialFailLLM:
            def __init__(self):
                self._lock = __import__("threading").Lock()

            def chat(self, messages, temperature=0.3, **kwargs):
                system = messages[0]["content"]
                if "Makro" in system:
                    raise RuntimeError("Makro-Datenquelle down")
                if "Fundamental" in system:
                    text = json.dumps({"rolle": "Fundamental-Analyst", "stimmung": "bullish",
                                        "score": 4, "zusammenfassung": "Gut"})
                elif "technisch" in system:
                    text = json.dumps({"rolle": "Technik-Analyst", "stimmung": "neutral",
                                        "score": 3, "zusammenfassung": "Ok"})
                else:
                    text = json.dumps({"rolle": "Sentiment-Analyst", "stimmung": "bullish",
                                        "score": 4, "zusammenfassung": "Positiv"})
                if kwargs.get("as_structured") and kwargs.get("response_format"):
                    from concilium.llm import StructuredChatResult
                    return StructuredChatResult(text=text, response_format_used=True)
                return text

        result = analyst_team(_MACRO_NEWS_DATA, _PartialFailLLM())

        # macro_news hat Fehlereintrag, kein Crash
        assert "fehler" in result["macro_news"]
        assert "Makro-Datenquelle down" in result["macro_news"]["fehler"]
        assert result["macro_news"]["_raw"] == ""
        # Andere Analysten normal
        assert result["fundamental"]["stimmung"] == "bullish"
        assert result["technical"]["stimmung"] == "neutral"
        assert result["sentiment"]["stimmung"] == "bullish"
        # technicals trotzdem durchgereicht
        assert result["technicals"]["current_price"] == 50.0

    def test_macro_news_failure_summary_is_empty_string(self):
        """_analyst_summary_text übersteht einen leeren/gestörten macro_news-Eintrag."""
        result = analyst_team(_MACRO_NEWS_DATA, _MacroNewsLLM())
        result["macro_news"] = {"_raw": "", "fehler": "boom"}
        summary = _analyst_summary_text(result)
        assert "Makro/News-Analyst:" in summary
        assert "N/A" in summary

    def test_all_fail_no_crash(self):
        """Alle Analysten (inkl. macro_news) werfen → kein Crash, alle Keys vorhanden."""

        class _AlwaysFailLLM:
            def chat(self, messages, temperature=0.3, **kwargs):
                raise RuntimeError("LLM komplett down")

        result = analyst_team(_MACRO_NEWS_DATA, _AlwaysFailLLM())
        for key in ("fundamental", "technical", "sentiment", "macro_news"):
            assert "fehler" in result[key]
            assert "LLM komplett down" in result[key]["fehler"]
        assert result["technicals"]["current_price"] == 50.0


# ===========================================================================
# (d) Report rendert die Makro/News-Sektion
# ===========================================================================


def _base_result() -> dict:
    """Minimaler Pipeline-Result-dict für Report-Tests."""
    return {
        "ticker": "TEST",
        "no_llm": False,
        "data": {
            "fundamentals": {"name": "Test Inc.", "sector": "Tech", "currency": "USD"},
            "technicals": {"current_price": 50.0},
            "sentiment": {},
            "news": [],
            "macro": {},
        },
        "analysts": {
            "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
            "technical": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
            "sentiment": {"stimmung": "bearish", "score": 2, "zusammenfassung": "Negativ", "_raw": ""},
            "macro_news": {
                "stimmung": "neutral",
                "score": 3,
                "zusammenfassung": "Makro gemischt",
                "makro_einschaetzung": "Steigende Zinsen belasten Tech",
                "relevante_headlines": "Zins-Hike-News ist material",
                "_raw": "",
            },
        },
        "debate": {"bull": {}, "bear": {}},
        "trade": {"aktion": "HALTEN", "begründung": "Test", "positionsanteil": 0},
        "risk": {"risiko_score": 3, "empfehlung": "GENEHMIGT"},
        "final": {"entscheidung": "GENEHMIGT", "confidence": 4, "begründung": "Ok"},
    }


class TestReportMacroNews:
    """(d) Report zeigt den Makro/News-Analysten."""

    def test_report_renders_macro_news_row(self):
        """Analysten-Tabelle enthält die Makro/News-Zeile."""
        report = generate_report(_base_result())
        assert "| Makro/News |" in report

    def test_report_renders_macro_news_section(self):
        """Makro/News-Einschätzung-Sektion mit Einschätzung + Headlines."""
        report = generate_report(_base_result())
        assert "### Makro/News-Einschätzung" in report
        assert "Steigende Zinsen belasten Tech" in report
        assert "Zins-Hike-News ist material" in report

    def test_report_without_macro_news_data_no_section(self):
        """Ohne makro_einschaetzung/relevante_headlines wird die Sektion weggelassen
        (z.B. alter Checkpoint oder Analyst-Ausfall)."""
        result = _base_result()
        result["analysts"]["macro_news"] = {"_raw": "", "fehler": "boom"}
        report = generate_report(result)
        assert "### Makro/News-Einschätzung" not in report

    def test_report_no_llm_no_section(self):
        """no_llm-Modus: keine Makro/News-Sektion."""
        result = _base_result()
        result["no_llm"] = True
        result.pop("analysts")
        report = generate_report(result)
        assert "### Makro/News-Einschätzung" not in report
        assert "| Makro/News |" not in report


# ===========================================================================
# Schema: ANALYST_MACRO_NEWS_SCHEMA funktioniert generisch
# ===========================================================================


class TestMacroNewsSchema:
    """ANALYST_MACRO_NEWS_SCHEMA mit defaults_for_schema / validate_structured."""

    def test_defaults_contain_all_keys(self):
        defaults = defaults_for_schema(ANALYST_MACRO_NEWS_SCHEMA)
        for key in ("stimmung", "score", "zusammenfassung",
                    "makro_einschaetzung", "relevante_headlines",
                    "konsistenz_warnung", "rolle"):
            assert key in defaults, f"Key '{key}' fehlt in defaults: {list(defaults.keys())}"

    def test_defaults_are_schema_conform(self):
        defaults = defaults_for_schema(ANALYST_MACRO_NEWS_SCHEMA)
        errors = validate_structured(defaults, ANALYST_MACRO_NEWS_SCHEMA)
        assert errors == [], f"Defaults nicht schema-konform: {errors}"

    def test_valid_result_validates_clean(self):
        valid = {
            "stimmung": "neutral",
            "score": 3,
            "zusammenfassung": "Makro ruhig",
            "makro_einschaetzung": "Zinsen stabil",
            "relevante_headlines": "keine material",
        }
        errors = validate_structured(valid, ANALYST_MACRO_NEWS_SCHEMA)
        assert errors == []

    def test_invalid_stimmung_flagged(self):
        invalid = {"stimmung": "launisch", "score": 3, "zusammenfassung": "X"}
        errors = validate_structured(invalid, ANALYST_MACRO_NEWS_SCHEMA)
        assert errors, "Ungültige Stimmung sollte gemeldet werden"

    def test_max_parallel_is_4(self):
        """Die 4 Analysten laufen parallel: _MAX_PARALLEL >= 4."""
        from concilium.agents import _MAX_PARALLEL
        assert _MAX_PARALLEL >= 4
