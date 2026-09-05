"""Tests für den Social-Media-Analysten (5. Analysten-Rolle).

Testet:
- analyst_team liefert 5 Analysten-Keys (fundamental, technical, sentiment,
  macro_news, social) + technicals.
- Der social-Analyst bekommt NUR die SOCIAL-MEDIA-Sektion (StockTwits/Reddit),
  keine Headlines, FUNDAMENTALS-, TECHNIK- oder MAKRO-Sektion.
- Ohne Social-Daten liefert die Rolle einen neutralen Hinweis (kein Crash).
- Ein Fehler beim social-Analyst crasht analyst_team nicht (Fehlereintrag
  {"_raw": "", "fehler": "..."} analog zu den anderen Analysten).
- Der Report rendert die Social-Media-Sektion (Analysten-Tabelle +
  Social-Media-Einschätzung).
- _analyst_summary_text nimmt den 5. Analysten auf (für Debatte/Trader).
- ANALYST_SOCIAL_SCHEMA funktioniert mit defaults_for_schema und
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
    ANALYST_SOCIAL_SCHEMA,
    defaults_for_schema,
    validate_structured,
)

# --- Fixtures ---

_SOCIAL_DATA = {
    "ticker": "TEST",
    "fundamentals": {
        "name": "Test Inc.",
        "sector": "Technology",
        "industry": "Semiconductors",
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
    "stocktwits_items": [
        {"title": "$TEST to the moon!", "published": None, "source": "web"},
        {"title": "$TEST bagholder here", "published": None, "source": "web"},
    ],
    "reddit_items": [
        {"title": "Why I like TEST", "published": None, "source": "reddit"},
    ],
    "macro": {
        "us_10y_yield": 4.2,
        "us_10y_trend": "steigend",
        "sp500_pe": 22.0,
        "sp500_source": "none",
    },
    "data_warnings": [],
}


class _SocialLLM:
    """Mock-LLM: dispatcht basierend auf dem System-Prompt-Inhalt.

    Zeichnet alle User-Prompts auf, damit rollenspezifische Daten-Filter
    geprüft werden können.
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
                "zusammenfassung": "Makro ruhig",
            })
        elif "Social-Media-Analyst" in system:
            # VOR dem Sentiment-Check: Der Social-Prompt enthält das Wort
            # "Sentiment" (Konträr-Indikator-Absatz) und würde sonst fälschlich
            # in den Sentiment-Branch laufen.
            text = json.dumps({
                "rolle": "Social-Media-Analyst",
                "stimmung": "bullish",
                "score": 4,
                "zusammenfassung": "Community euphorisch",
                "dominant": "positiv",
                "community_stimmung": "retail-bullish",
            })
        elif "Sentiment" in system:
            text = json.dumps({
                "rolle": "Sentiment-Analyst",
                "stimmung": "bullish",
                "score": 4,
                "zusammenfassung": "Positiv",
            })
        elif "Social-Media-Analyst" in system:
            text = json.dumps({
                "rolle": "Social-Media-Analyst",
                "stimmung": "bullish",
                "score": 4,
                "zusammenfassung": "Community euphorisch",
                "dominant": "positiv",
                "community_stimmung": "retail-bullish",
            })
        else:
            text = json.dumps({"rolle": "Unknown", "stimmung": "neutral", "score": 3})
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult
            return StructuredChatResult(text=text, response_format_used=True)
        return text


# ===========================================================================
# (a) analyst_team liefert 5 Analysten-Keys
# ===========================================================================


class TestAnalystTeamFiveKeys:
    """(a) analyst_team liefert 5 Analysten-Keys + technicals."""

    def test_five_analyst_keys_present(self):
        """fundamental, technical, sentiment, macro_news, social, technicals."""
        result = analyst_team(_SOCIAL_DATA, _SocialLLM())
        assert "fundamental" in result
        assert "technical" in result
        assert "sentiment" in result
        assert "macro_news" in result
        assert "social" in result
        assert "technicals" in result

    def test_social_has_structured_keys(self):
        """social-Ergebnis enthält die Schema-Keys inkl. Community-Felder."""
        result = analyst_team(_SOCIAL_DATA, _SocialLLM())
        so = result["social"]
        assert so["stimmung"] == "bullish"
        assert so["score"] == 4
        assert so["zusammenfassung"] == "Community euphorisch"
        assert so["dominant"] == "positiv"
        assert so["community_stimmung"] == "retail-bullish"

    def test_social_in_summary_for_debate(self):
        """_analyst_summary_text nimmt den Social-Media-Analysten auf."""
        result = analyst_team(_SOCIAL_DATA, _SocialLLM())
        summary = _analyst_summary_text(result)
        assert "Social-Media-Analyst:" in summary
        assert "Community euphorisch" in summary

    def test_five_calls_made(self):
        """5 LLM-Calls (einer pro Analyst)."""
        llm = _SocialLLM()
        analyst_team(_SOCIAL_DATA, llm)
        assert len(llm.all_messages) == 5


# ===========================================================================
# (b) social-Analyst bekommt NUR die Social-Media-Sektion
# ===========================================================================


class TestSocialDataFilter:
    """(b) _build_data_text(role='social') filtert rollenspezifisch."""

    def test_social_analyst_receives_filtered_data(self):
        """End-to-End: Der social-User-Prompt enthält Posts, keine Headlines."""
        llm = _SocialLLM()
        analyst_team(_SOCIAL_DATA, llm, data_text=None)

        user_texts = {
            msgs[0]["content"]: msgs[1]["content"] for msgs in llm.all_messages
        }
        social_user = next(
            (u for s, u in user_texts.items() if "Social-Media-Analyst" in s),
            None,
        )
        assert social_user is not None, "Social-Media-Analyst-Call nicht gefunden"
        assert "=== SOCIAL MEDIA (StockTwits/Reddit) ===" in social_user
        assert "$TEST to the moon!" in social_user
        assert "Why I like TEST" in social_user
        # Keine Nachrichten-Headlines, keine anderen Sektionen
        assert "Test headline 1" not in social_user
        assert "=== FUNDAMENTALS ===" not in social_user
        assert "=== TECHNIK ===" not in social_user
        assert "=== MAKRO" not in social_user
        assert "=== SENTIMENT ===" not in social_user

    def test_default_alle_includes_social(self):
        """Rückwärtskompatibilität: role='alle' zeigt weiterhin alle Sektionen."""
        text = _build_data_text(_SOCIAL_DATA, role="alle")
        assert "=== FUNDAMENTALS ===" in text
        assert "=== TECHNIK ===" in text
        assert "=== MAKRO / ZINSEN ===" in text
        assert "=== SENTIMENT ===" in text
        assert "=== SOCIAL MEDIA (StockTwits/Reddit) ===" in text


# ===========================================================================
# (c) Ohne Social-Daten: neutraler Hinweis, kein Crash
# ===========================================================================


class TestSocialNoDataResilience:
    """(c) Fehlende Social-Daten → Hinweis + neutrale Einschätzung möglich."""

    def test_no_social_data_hint(self):
        """Ohne stocktwits_items/reddit_items bekommt die Rolle einen Hinweis."""
        data = {k: v for k, v in _SOCIAL_DATA.items()
                if k not in ("stocktwits_items", "reddit_items")}
        llm = _SocialLLM()
        analyst_team(data, llm, data_text=None)

        user_texts = {
            msgs[0]["content"]: msgs[1]["content"] for msgs in llm.all_messages
        }
        social_user = next(
            (u for s, u in user_texts.items() if "Social-Media-Analyst" in s),
            None,
        )
        assert social_user is not None
        assert "Keine StockTwits- oder Reddit-Posts verfügbar" in social_user

    def test_social_failure_does_not_crash(self):
        """social-Call wirft → Fehlereintrag, andere Analysten normal."""

        class _PartialFailLLM:
            def chat(self, messages, temperature=0.3, **kwargs):
                system = messages[0]["content"]
                if "Social-Media-Analyst" in system:
                    raise RuntimeError("Social-Quelle down")
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

        result = analyst_team(_SOCIAL_DATA, _PartialFailLLM())

        # social hat Fehlereintrag, kein Crash
        assert "fehler" in result["social"]
        assert "Social-Quelle down" in result["social"]["fehler"]
        assert result["social"]["_raw"] == ""
        # Andere Analysten normal
        assert result["fundamental"]["stimmung"] == "bullish"
        assert result["technical"]["stimmung"] == "neutral"


# ===========================================================================
# (d) Report rendert die Social-Media-Sektion
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
            "macro_news": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Makro ruhig", "_raw": ""},
            "social": {
                "stimmung": "bullish",
                "score": 4,
                "zusammenfassung": "Community euphorisch",
                "dominant": "positiv",
                "community_stimmung": "retail-bullish",
                "_raw": "",
            },
        },
        "debate": {"bull": {}, "bear": {}},
        "trade": {"aktion": "HALTEN", "begründung": "Test", "positionsanteil": 0},
        "risk": {"risiko_score": 3, "empfehlung": "GENEHMIGT"},
        "final": {"entscheidung": "GENEHMIGT", "confidence": 4, "begründung": "Ok"},
    }


class TestReportSocial:
    """(d) Report zeigt den Social-Media-Analysten."""

    def test_report_renders_social_row(self):
        """Analysten-Tabelle enthält die Social-Media-Zeile."""
        report = generate_report(_base_result())
        assert "| Social-Media |" in report

    def test_report_renders_social_section(self):
        """Social-Media-Einschätzung-Sektion mit Community-Stimmung."""
        report = generate_report(_base_result())
        assert "### Social-Media-Einschätzung" in report
        assert "**Dominante Community-Stimmung:** retail-bullish" in report
        assert "**Dominante Stimmung (Posts):** positiv" in report

    def test_report_without_social_no_section(self):
        """Ohne social-Ergebnis wird die Sektion weggelassen (kein Crash)."""
        result = _base_result()
        result["analysts"].pop("social")
        report = generate_report(result)
        assert "### Social-Media-Einschätzung" not in report


# ===========================================================================
# Schema: ANALYST_SOCIAL_SCHEMA funktioniert generisch
# ===========================================================================


class TestSocialSchema:
    """ANALYST_SOCIAL_SCHEMA mit defaults_for_schema / validate_structured."""

    def test_defaults_contain_all_keys(self):
        defaults = defaults_for_schema(ANALYST_SOCIAL_SCHEMA)
        for key in ("stimmung", "score", "zusammenfassung",
                    "dominant", "community_stimmung",
                    "konsistenz_warnung", "rolle"):
            assert key in defaults, f"Key '{key}' fehlt in defaults: {list(defaults.keys())}"

    def test_defaults_are_schema_conform(self):
        defaults = defaults_for_schema(ANALYST_SOCIAL_SCHEMA)
        errors = validate_structured(defaults, ANALYST_SOCIAL_SCHEMA)
        assert errors == [], f"Defaults nicht schema-konform: {errors}"

    def test_valid_result_validates_clean(self):
        valid = {
            "stimmung": "neutral",
            "score": 3,
            "zusammenfassung": "Community ruhig",
            "dominant": "neutral",
            "community_stimmung": "retail-neutral",
        }
        errors = validate_structured(valid, ANALYST_SOCIAL_SCHEMA)
        assert errors == []

    def test_invalid_stimmung_flagged(self):
        invalid = {"stimmung": "launisch", "score": 3, "zusammenfassung": "X"}
        errors = validate_structured(invalid, ANALYST_SOCIAL_SCHEMA)
        assert errors, "Ungültige Stimmung sollte gemeldet werden"
