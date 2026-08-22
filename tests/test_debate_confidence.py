"""Tests für die Verbesserungen der Bull/Bear-Debatte.

Feature A: Differenzierte SYSTEM_BULL/SYSTEM_BEAR Prompts.
Feature B: _parse_debate_confidence, _debate_skew_text, debate() Konfidenz-Durchreichung, trader() Skew-Text.
Feature C: _clean_debate_text robuster Platzhalter, Report-Debatten-Konfidenz-Zeile.

Alle Tests sind offline (kein Netzwerk) — der LLMClient wird gemockt.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import (  # noqa: E402
    SYSTEM_BEAR,
    SYSTEM_BULL,
    _debate_skew_text,
    _parse_debate_confidence,
    debate,
    trader,
)
from concilium.report import _clean_debate_text, generate_report  # noqa: E402

# ===========================================================================
# Feature A: Differenzierte SYSTEM_BULL / SYSTEM_BEAR Prompts
# ===========================================================================


class TestSystemBullPrompt:
    """SYSTEM_BULL hat bullische Schwerpunkte."""

    def test_bull_mentions_wachstum(self):
        assert "Wachstum" in SYSTEM_BULL

    def test_bull_mentions_momentum(self):
        assert "Momentum" in SYSTEM_BULL

    def test_bull_mentions_sentiment(self):
        assert "Sentiment" in SYSTEM_BULL

    def test_bull_mentions_bewertung(self):
        assert "Bewertung" in SYSTEM_BULL

    def test_bull_mentions_peg(self):
        assert "PEG" in SYSTEM_BULL

    def test_bull_says_ignoriere_risiken(self):
        assert "Ignoriere die Risiken" in SYSTEM_BULL


class TestSystemBearPrompt:
    """SYSTEM_BEAR hat bearische Schwerpunkte."""

    def test_bear_mentions_bewertungsrisiken(self):
        assert "Bewertungsrisiken" in SYSTEM_BEAR

    def test_bear_mentions_konzentration(self):
        assert "Konzentration" in SYSTEM_BEAR

    def test_bear_mentions_zins(self):
        assert "Zins" in SYSTEM_BEAR

    def test_bear_mentions_margin(self):
        assert "Margin" in SYSTEM_BEAR

    def test_bear_mentions_technische_gegenanzeichen(self):
        assert "Technische Gegenanzeichen" in SYSTEM_BEAR or "technische" in SYSTEM_BEAR.lower()

    def test_bear_says_ignoriere_staerken(self):
        assert "Ignoriere die Stärken" in SYSTEM_BEAR


class TestPromptsDiffer:
    """SYSTEM_BULL und SYSTEM_BEAR sind unterschiedlich (nicht redundant)."""

    def test_prompts_are_different(self):
        assert SYSTEM_BULL != SYSTEM_BEAR

    def test_bull_has_wachstum_bear_has_bewertungsrisiken(self):
        """Bull hat 'Wachstum', Bear hat 'Bewertungsrisiken' — unterschiedliche Schwerpunkte."""
        assert "Wachstum" in SYSTEM_BULL
        assert "Bewertungsrisiken" in SYSTEM_BEAR

    def test_both_have_json_preamble_instruction(self):
        """Beide Prompts fordern einen JSON-Block mit confidence+name am Anfang."""
        assert "confidence" in SYSTEM_BULL
        assert "confidence" in SYSTEM_BEAR
        assert "name" in SYSTEM_BULL
        assert "name" in SYSTEM_BEAR

    def test_both_have_3_to_6_sentences_instruction(self):
        assert "3-6 Sätze" in SYSTEM_BULL
        assert "3-6 Sätze" in SYSTEM_BEAR


# ===========================================================================
# Feature B: _parse_debate_confidence
# ===========================================================================


class TestParseDebateConfidence:
    """Tests für _parse_debate_confidence."""

    def test_parses_json_preamble(self):
        """JSON-Block mit confidence am Anfang → parst 4."""
        agent = {"_raw": '{"confidence": 4, "name": "Bull-Argumentation"}\nFließtext.'}
        assert _parse_debate_confidence(agent) == 4

    def test_parses_json_codeblock(self):
        """```json {...}``` Block → parst confidence."""
        agent = {"_raw": '```json\n{"confidence": 5, "name": "Bull"}\n```\nText.'}
        assert _parse_debate_confidence(agent) == 5

    def test_direct_confidence_field(self):
        """Direktes confidence-Feld im Agent-Dict → parst."""
        agent = {"_raw": "kein json", "confidence": 3}
        assert _parse_debate_confidence(agent) == 3

    def test_direct_confidence_takes_priority(self):
        """Direktes confidence-Feld wird vor _raw verwendet."""
        agent = {"_raw": '{"confidence": 2}', "confidence": 5}
        assert _parse_debate_confidence(agent) == 5

    def test_no_json_returns_none(self):
        """Ohne JSON-Block und ohne direktes Feld → None."""
        agent = {"_raw": "Nur Fließtext ohne JSON"}
        assert _parse_debate_confidence(agent) is None

    def test_empty_raw_returns_none(self):
        """Leerer _raw → None."""
        agent = {"_raw": ""}
        assert _parse_debate_confidence(agent) is None

    def test_empty_dict_returns_none(self):
        """Leeres dict → None."""
        assert _parse_debate_confidence({}) is None

    def test_non_dict_returns_none(self):
        """Nicht-dict input → None, kein Crash."""
        assert _parse_debate_confidence("not a dict") is None  # type: ignore[arg-type]
        assert _parse_debate_confidence(None) is None  # type: ignore[arg-type]

    def test_invalid_confidence_returns_none(self):
        """confidence ist String 'abc' → None."""
        agent = {"_raw": '{"confidence": "abc"}'}
        assert _parse_debate_confidence(agent) is None

    def test_confidence_1_returns_1(self):
        """Confidence 1 wird korrekt geparsed."""
        agent = {"_raw": '{"confidence": 1, "name": "Bear"}\nText.'}
        assert _parse_debate_confidence(agent) == 1


# ===========================================================================
# Feature B: _debate_skew_text
# ===========================================================================


class TestDebateSkewText:
    """Tests für _debate_skew_text."""

    def test_bull_dominant(self):
        """Bull 5, Bear 1 → Nettoneigung 2.0 → bullischer Text."""
        text = _debate_skew_text(5, 1)
        assert "Bull 5/5 vs Bear 1/5" in text
        assert "Nettoneigung: +2.0" in text
        assert "Bull-Seite hat die Oberhand" in text

    def test_bear_dominant(self):
        """Bull 1, Bear 5 → Nettoneigung -2.0 → bearischer Text."""
        text = _debate_skew_text(1, 5)
        assert "Bull 1/5 vs Bear 5/5" in text
        assert "Nettoneigung: -2.0" in text
        assert "Bear-Seite hat die Oberhand" in text

    def test_ausgewogen(self):
        """Bull 3, Bear 3 → Nettoneigung 0.0 → ausgewogene Debatte."""
        text = _debate_skew_text(3, 3)
        assert "Bull 3/5 vs Bear 3/5" in text
        assert "Nettoneigung: +0.0" in text
        assert "ausgewogene Debatte" in text

    def test_near_equal_still_ausgewogen(self):
        """Bull 4, Bear 3 → Nettoneigung 0.5 → ausgewogen (nicht > 0.5)."""
        text = _debate_skew_text(4, 3)
        assert "ausgewogene Debatte" in text

    def test_both_none_empty(self):
        """Beide None → leerer String."""
        assert _debate_skew_text(None, None) == ""

    def test_only_bull(self):
        """Nur Bull verfügbar, Bear None → zeigt nur Bull."""
        text = _debate_skew_text(4, None)
        assert "Bull 4/5" in text
        assert "Bear-Seite nicht verfügbar" in text

    def test_only_bear(self):
        """Nur Bear verfügbar, Bull None → zeigt nur Bear."""
        text = _debate_skew_text(None, 3)
        assert "Bear 3/5" in text
        assert "Bull-Seite nicht verfügbar" in text

    def test_sachlich_no_alarmism(self):
        """Text ist sachlich — kein Ausrufezeichen-Alarmismus."""
        text = _debate_skew_text(5, 1)
        assert "!" not in text

    def test_bull_slight_edge(self):
        """Bull 4, Bear 2 → Nettoneigung 1.0 → Bull Oberhand."""
        text = _debate_skew_text(4, 2)
        assert "Bull-Seite hat die Oberhand" in text


# ===========================================================================
# Feature B: debate() setzt bull_confidence / bear_confidence
# ===========================================================================


class _DebateFakeLLM:
    """Mock-LLM, der für Bull/Bear JSON-Preamble + Fließtext zurückgibt.

    Dispatcht basierend auf dem System-Prompt-Inhalt.
    """

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3, **kwargs) -> str | object:
        system = messages[0]["content"]
        if "Bull" in system and "STÄRKEN" in system:
            text = '{"confidence": 4, "name": "Bull-Argumentation"}\nDie Aktie zeigt starkes Wachstum.'
        elif "Bear" in system and "RISIKEN" in system:
            text = '{"confidence": 2, "name": "Bear-Argumentation"}\nBewertung ist zu hoch.'
        else:
            text = '{"confidence": 3, "name": "Unknown"}\nUnbekannt.'
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult
            return StructuredChatResult(text=text, response_format_used=True)
        return text


_MINIMAL_ANALYSTS = {
    "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
}


class TestDebateConfidence:
    """debate() setzt bull_confidence und bear_confidence ins Ergebnis."""

    def test_bull_confidence_parsed(self):
        """debate() parst bull_confidence aus dem JSON-Block."""
        result = debate(_MINIMAL_ANALYSTS, _DebateFakeLLM())
        assert result["bull_confidence"] == 4

    def test_bear_confidence_parsed(self):
        """debate() parst bear_confidence aus dem JSON-Block."""
        result = debate(_MINIMAL_ANALYSTS, _DebateFakeLLM())
        assert result["bear_confidence"] == 2

    def test_bull_and_bear_keys_present(self):
        """bull und bear Keys sind weiterhin vorhanden."""
        result = debate(_MINIMAL_ANALYSTS, _DebateFakeLLM())
        assert "bull" in result
        assert "bear" in result

    def test_confidence_keys_present(self):
        """bull_confidence und bear_confidence Keys sind vorhanden."""
        result = debate(_MINIMAL_ANALYSTS, _DebateFakeLLM())
        assert "bull_confidence" in result
        assert "bear_confidence" in result

    def test_confidence_none_on_empty_raw(self):
        """Bei leerem _raw → confidence ist der Schema-Default (1)."""

        class _EmptyLLM:
            def chat(self, messages, temperature=0.3, **kwargs):
                if kwargs.get("as_structured") and kwargs.get("response_format"):
                    from concilium.llm import StructuredChatResult
                    return StructuredChatResult(text="", response_format_used=True)
                return ""

        result = debate(_MINIMAL_ANALYSTS, _EmptyLLM())
        # Durch Struktur-Garantie bekommt confidence den Default 1 (minimum)
        assert result["bull_confidence"] == 1
        assert result["bear_confidence"] == 1

    def test_confidence_none_on_no_json(self):
        """Bei Fließtext ohne JSON-Block → confidence ist der Schema-Default (1)."""

        class _NoJsonLLM:
            def chat(self, messages, temperature=0.3, **kwargs):
                text = "Nur Fließtext, kein JSON hier."
                if kwargs.get("as_structured") and kwargs.get("response_format"):
                    from concilium.llm import StructuredChatResult
                    return StructuredChatResult(text=text, response_format_used=True)
                return text

        result = debate(_MINIMAL_ANALYSTS, _NoJsonLLM())
        # Durch Struktur-Garantie bekommt confidence den Default 1 (minimum)
        assert result["bull_confidence"] == 1
        assert result["bear_confidence"] == 1


# ===========================================================================
# Feature B: trader() hängt Debatten-Konfidenz-Block an den Prompt
# ===========================================================================


class _CapturingTraderLLM:
    """Mock-LLM, der die messages aufzeichnet und JSON zurückgibt."""

    def __init__(self, response: str | None = None):
        if response is None:
            response = json.dumps({
                "rolle": "Trader",
                "aktion": "HALTEN",
                "zielkurs": None,
                "stop_loss": None,
                "positionsanteil": 0,
                "begründung": "Test",
                "zeithorizont": "Mittelfristig",
            })
        self._response = response
        self.captured_messages: list[list[dict]] = []

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3, **kwargs) -> str | object:
        self.captured_messages.append(messages)
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult
            return StructuredChatResult(text=self._response, response_format_used=True)
        return self._response


_TRADER_ANALYSTS = {
    "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
    "technicals": {"current_price": 100.0},
}


class TestTraderSkewText:
    """trader() hängt den Debatten-Konfidenz-Block an den user_text an."""

    def test_skew_text_appended_when_confidence_present(self):
        """Bei Konfidenz im debate_result wird der Skew-Block im Prompt angehängt."""
        debate_result = {
            "bull": {"_raw": '{"confidence": 5, "name": "Bull"}\nBull text'},
            "bear": {"_raw": '{"confidence": 1, "name": "Bear"}\nBear text'},
            "bull_confidence": 5,
            "bear_confidence": 1,
        }
        llm = _CapturingTraderLLM()
        trader(_TRADER_ANALYSTS, debate_result, llm)

        user_msg = llm.captured_messages[0][1]["content"]
        assert "Debatten-Konfidenz" in user_msg
        assert "Bull 5/5" in user_msg
        assert "Bear 1/5" in user_msg

    def test_skew_text_not_appended_when_no_confidence(self):
        """Ohne Konfidenz wird kein Skew-Block angehängt."""
        debate_result = {
            "bull": {"_raw": "Bull text ohne JSON"},
            "bear": {"_raw": "Bear text ohne JSON"},
        }
        llm = _CapturingTraderLLM()
        trader(_TRADER_ANALYSTS, debate_result, llm)

        user_msg = llm.captured_messages[0][1]["content"]
        assert "Debatten-Konfidenz" not in user_msg

    def test_skew_text_from_raw_parsing(self):
        """Auch ohne bull_confidence/bear_confidence Felder wird aus _raw geparsed."""
        debate_result = {
            "bull": {"_raw": '{"confidence": 4, "name": "Bull"}\nBull text'},
            "bear": {"_raw": '{"confidence": 2, "name": "Bear"}\nBear text'},
        }
        llm = _CapturingTraderLLM()
        trader(_TRADER_ANALYSTS, debate_result, llm)

        user_msg = llm.captured_messages[0][1]["content"]
        assert "Debatten-Konfidenz" in user_msg
        assert "Bull 4/5" in user_msg

    def test_skew_text_before_feedback_context(self):
        """Skew-Text kommt VOR feedback_context im user_text."""
        debate_result = {
            "bull": {"_raw": '{"confidence": 5, "name": "Bull"}\nBull'},
            "bear": {"_raw": '{"confidence": 2, "name": "Bear"}\nBear'},
            "bull_confidence": 5,
            "bear_confidence": 2,
        }
        llm = _CapturingTraderLLM()
        trader(_TRADER_ANALYSTS, debate_result, llm, feedback_context="FEEDBACK_MARKER_XYZ")

        user_msg = llm.captured_messages[0][1]["content"]
        skew_pos = user_msg.find("Debatten-Konfidenz")
        feedback_pos = user_msg.find("FEEDBACK_MARKER_XYZ")
        assert skew_pos != -1
        assert feedback_pos != -1
        assert skew_pos < feedback_pos, "Skew-Text muss VOR feedback_context stehen"

    def test_skew_text_before_reflection_context(self):
        """Skew-Text kommt VOR reflection_context im user_text."""
        debate_result = {
            "bull": {"_raw": '{"confidence": 4, "name": "Bull"}\nBull'},
            "bear": {"_raw": '{"confidence": 3, "name": "Bear"}\nBear'},
            "bull_confidence": 4,
            "bear_confidence": 3,
        }
        llm = _CapturingTraderLLM()
        trader(_TRADER_ANALYSTS, debate_result, llm, reflection_context="REFLECTION_MARKER_ABC")

        user_msg = llm.captured_messages[0][1]["content"]
        skew_pos = user_msg.find("Debatten-Konfidenz")
        reflection_pos = user_msg.find("REFLECTION_MARKER_ABC")
        assert skew_pos != -1
        assert reflection_pos != -1
        assert skew_pos < reflection_pos, "Skew-Text muss VOR reflection_context stehen"


# ===========================================================================
# Feature C: _clean_debate_text robuster Platzhalter
# ===========================================================================


class TestCleanDebateTextRobust:
    """_clean_debate_text gibt klaren Platzhalter bei fehlendem Text."""

    def test_empty_raw_gives_placeholder(self):
        """Leerer _raw → Platzhalter statt 'N/A'."""
        agent = {"_raw": ""}
        result = _clean_debate_text(agent)
        assert "N/A" not in result
        assert "nicht verfügbar" in result.lower()
        assert "⚠️" in result

    def test_missing_raw_key_gives_placeholder(self):
        """Fehlender _raw-Key → Platzhalter."""
        agent = {}
        result = _clean_debate_text(agent)
        assert "nicht verfügbar" in result.lower()

    def test_whitespace_only_gives_placeholder(self):
        """Nur Leerzeichen → Platzhalter."""
        agent = {"_raw": "   \n  \t  "}
        result = _clean_debate_text(agent)
        assert "nicht verfügbar" in result.lower()

    def test_json_only_gives_placeholder(self):
        """Nur JSON-Block ohne Fließtext → Platzhalter."""
        agent = {"_raw": '{"confidence": 3, "name": "Bull"}'}
        result = _clean_debate_text(agent)
        assert "nicht verfügbar" in result.lower()

    def test_normal_text_still_cleaned(self):
        """Normaler Text mit JSON-Preamble wird korrekt gesäubert."""
        agent = {"_raw": '{"confidence": 4, "name": "Bull"}\nDie Aktie ist stark.'}
        result = _clean_debate_text(agent)
        assert "Die Aktie ist stark." in result
        assert "confidence" not in result
        assert "nicht verfügbar" not in result.lower()

    def test_codeblock_text_cleaned(self):
        """Text mit ```json Codeblock-Preamble wird gesäubert."""
        agent = {"_raw": '```json\n{"confidence": 5, "name": "Bull"}\n```\nSehr bullisch.'}
        result = _clean_debate_text(agent)
        assert "Sehr bullisch." in result
        assert "nicht verfügbar" not in result.lower()

    def test_no_bare_NA(self):
        """Bei leerem _raw wird niemals nacktes 'N/A' zurückgegeben."""
        agent = {"_raw": ""}
        result = _clean_debate_text(agent)
        assert result.strip() != "N/A"


# ===========================================================================
# Feature C: Report zeigt Debatten-Konfidenz-Zeile
# ===========================================================================


class TestReportDebateConfidence:
    """Report-Debatten-Abschnitt zeigt Konfidenzen, wenn verfügbar."""

    _BASE_RESULT = {
        "ticker": "AAPL",
        "no_llm": False,
        "data": {
            "fundamentals": {"name": "Apple", "sector": "Tech"},
            "technicals": {"current_price": 150},
            "sentiment": {},
        },
        "analysts": {
            "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
            "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
            "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
        },
        "trade": {
            "aktion": "KAUFEN",
            "zielkurs": 180,
            "stop_loss": 130,
            "positionsanteil": 7,
            "begründung": "Test",
            "zeithorizont": "Mittelfristig",
        },
        "risk": {"risiko_score": 3, "empfehlung": "GENEHMIGT"},
        "final": {"entscheidung": "GENEHMIGT", "confidence": 4, "begründung": "Ok"},
    }

    def test_report_shows_confidence_line(self):
        """Report zeigt Debatten-Konfidenz-Zeile, wenn bull/bear_confidence gesetzt."""
        result = dict(self._BASE_RESULT)
        result["debate"] = {
            "bull": {"_raw": '{"confidence": 4, "name": "Bull"}\nBull text'},
            "bear": {"_raw": '{"confidence": 2, "name": "Bear"}\nBear text'},
            "bull_confidence": 4,
            "bear_confidence": 2,
        }
        report = generate_report(result)
        assert "Debatten-Konfidenz" in report
        assert "Bull 4/5" in report
        assert "Bear 2/5" in report

    def test_report_no_confidence_line_when_absent(self):
        """Report zeigt keine Konfidenz-Zeile, wenn confidence-Keys fehlen."""
        result = dict(self._BASE_RESULT)
        result["debate"] = {
            "bull": {"_raw": "Bull text"},
            "bear": {"_raw": "Bear text"},
        }
        report = generate_report(result)
        assert "Debatten-Konfidenz" not in report

    def test_report_shows_placeholder_for_empty_bear(self):
        """Bei leerem Bear wird der Platzhalter im Report angezeigt."""
        result = dict(self._BASE_RESULT)
        result["debate"] = {
            "bull": {"_raw": '{"confidence": 3, "name": "Bull"}\nBull text'},
            "bear": {"_raw": ""},
            "bull_confidence": 3,
            "bear_confidence": None,
        }
        report = generate_report(result)
        assert "nicht verfügbar" in report.lower()
        # Der Platzhalter steht im Bear-Abschnitt, nicht nacktes "N/A"
        bear_section = report.split("Bear-Argumentation")[1] if "Bear-Argumentation" in report else ""
        assert "nicht verfügbar" in bear_section.lower()

    def test_report_shows_only_bull_confidence(self):
        """Wenn nur bull_confidence vorhanden, zeigt Report nur Bull."""
        result = dict(self._BASE_RESULT)
        result["debate"] = {
            "bull": {"_raw": '{"confidence": 5, "name": "Bull"}\nBull text'},
            "bear": {"_raw": '{"confidence": 2, "name": "Bear"}\nBear text'},
            "bull_confidence": 5,
            "bear_confidence": None,
        }
        report = generate_report(result)
        assert "Debatten-Konfidenz" in report
        assert "Bull 5/5" in report
