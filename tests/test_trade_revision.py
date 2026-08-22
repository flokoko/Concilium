"""Tests für Feature A: Trade-Revision (2nd Pass), Feature B: PM MODIFIZIERT.

Alle Tests sind offline (kein Netzwerk) — der LLMClient wird gemockt.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import (  # noqa: E402
    SYSTEM_PM,
    SYSTEM_TRADE_REVISION,
    trade_revision,
)

# --------------------------------------------------------------------------- #
# Feature A: trade_revision
# --------------------------------------------------------------------------- #


class _CapturingLLM:
    """Mock-LLM, der die messages aufzeichnet und eine vordefinierte Antwort gibt."""

    def __init__(self, response: str):
        self._response = response
        self.captured_messages: list[list[dict]] = []
        self.captured_temperature: float | None = None

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3, **kwargs) -> str | object:
        self.captured_messages.append(messages)
        self.captured_temperature = temperature
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult
            return StructuredChatResult(text=self._response, response_format_used=True)
        return self._response


_REVISION_JSON = json.dumps({
    "rolle": "Trader",
    "aktion": "KAUFEN",
    "zielkurs": 115,
    "stop_loss": 92,
    "positionsanteil": 3,
    "begründung": "Risk-Manager empfiehlt kleinere Position, Portfolio-Fit zeigt Overlap.",
    "zeithorizont": "Mittelfristig",
})

_REVISION_JSON_5STUFIG = json.dumps({
    "rolle": "Trader",
    "aktion": "STARK KAUFEN",
    "zielkurs": 120,
    "stop_loss": 90,
    "positionsanteil": 5,
    "begründung": "Einwände nicht überzeugend, bleibe bei meiner Bewertung.",
    "zeithorizont": "Mittelfristig",
})

_ORIGINAL_TRADE = {
    "rolle": "Trader",
    "aktion": "KAUFEN",
    "rating": "KAUFEN",
    "zielkurs": 130,
    "stop_loss": 85,
    "positionsanteil": 8,
    "begründung": "Starke Fundamentals",
    "zeithorizont": "Mittelfristig",
}

_RISK = {
    "rolle": "Risk-Manager",
    "risiko_score": 4,
    "empfehlung": "MODIFIZIERT",
    "positionsgröße_empfohlen": "3",
    "auflagen": "Position auf 3% reduzieren",
}

_PORTFOLIO_FIT = {
    "rolle": "Portfolio-Fit-Analyst",
    "portfolio_fit_score": 2,
    "ziel_gewichtung_pct": 2.0,
    "konzentrationsrisiko_bewertung": "Hohe Konzentration im Tech-Sektor.",
    "sektor_overlap_bewertung": "Starke Überlagerung mit bestehenden Positionen.",
    "begründung": "Redundanz im Tech-Bereich.",
}


class TestTradeRevisionCall:
    """Testet dass trade_revision den LLM aufruft und trade+risk+portfolio_fit weiterreicht."""

    def test_calls_llm(self):
        """trade_revision ruft den LLM auf und liefert ein dict."""
        llm = _CapturingLLM(_REVISION_JSON)
        result = trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm)
        assert len(llm.captured_messages) == 1
        assert isinstance(result, dict)

    def test_system_prompt_is_trade_revision(self):
        """Der System-Prompt ist SYSTEM_TRADE_REVISION."""
        llm = _CapturingLLM(_REVISION_JSON)
        trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm)
        system_msg = llm.captured_messages[0][0]["content"]
        assert system_msg == SYSTEM_TRADE_REVISION

    def test_user_prompt_contains_trade(self):
        """Der User-Prompt enthält den Original-Trade."""
        llm = _CapturingLLM(_REVISION_JSON)
        trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm)
        user_msg = llm.captured_messages[0][1]["content"]
        assert "Ursprünglicher Trade-Vorschlag" in user_msg
        assert "130" in user_msg  # zielkurs aus _ORIGINAL_TRADE

    def test_user_prompt_contains_risk(self):
        """Der User-Prompt enthält die Risiko-Bewertung."""
        llm = _CapturingLLM(_REVISION_JSON)
        trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm)
        user_msg = llm.captured_messages[0][1]["content"]
        assert "Risiko-Bewertung" in user_msg
        assert "MODIFIZIERT" in user_msg

    def test_user_prompt_contains_portfolio_fit(self):
        """Der User-Prompt enthält die Portfolio-Fit-Einschätzung."""
        llm = _CapturingLLM(_REVISION_JSON)
        trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm)
        user_msg = llm.captured_messages[0][1]["content"]
        assert "Portfolio-Fit-Einschätzung" in user_msg
        assert "Redundanz" in user_msg

    def test_user_prompt_portfolio_fit_none(self):
        """Bei portfolio_fit=None wird 'Nicht verfügbar' im Prompt angezeigt."""
        llm = _CapturingLLM(_REVISION_JSON)
        trade_revision(_ORIGINAL_TRADE, _RISK, None, llm)
        user_msg = llm.captured_messages[0][1]["content"]
        assert "Nicht verfügbar" in user_msg

    def test_temperature_is_03(self):
        """trade_revision nutzt temp 0.3."""
        llm = _CapturingLLM(_REVISION_JSON)
        trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm)
        assert llm.captured_temperature == 0.3


class TestTradeRevisionNormalization:
    """Testet die 5-stufig/3-stufig Normalisierung im revidierten Trade."""

    def test_rating_set_from_raw_aktion(self):
        """rating = rohes 5-stufig (aus aktion-Feld)."""
        llm = _CapturingLLM(_REVISION_JSON_5STUFIG)
        result = trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm)
        assert result["rating"] == "STARK KAUFEN"

    def test_aktion_normalized_3step(self):
        """aktion = 3-stufig normalisiert (STARK KAUFEN → KAUFEN)."""
        llm = _CapturingLLM(_REVISION_JSON_5STUFIG)
        result = trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm)
        assert result["aktion"] == "KAUFEN"

    def test_kaufen_preserved(self):
        """KAUFEN bleibt KAUFEN in 3-stufig."""
        llm = _CapturingLLM(_REVISION_JSON)
        result = trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm)
        assert result["aktion"] == "KAUFEN"
        assert result["rating"] == "KAUFEN"

    def test_halten_preserved(self):
        """HALTEN bleibt HALTEN."""
        halten_json = json.dumps({
            "rolle": "Trader", "aktion": "HALTEN", "zielkurs": None,
            "stop_loss": None, "positionsanteil": 0,
            "begründung": "Risiko zu hoch", "zeithorizont": "Mittelfristig",
        })
        llm = _CapturingLLM(halten_json)
        result = trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm)
        assert result["aktion"] == "HALTEN"
        assert result["rating"] == "HALTEN"

    def test_stark_verkaufen_normalized(self):
        """STARK VERKAUFEN → aktion=VERKAUFEN, rating=STARK VERKAUFEN."""
        verkaufen_json = json.dumps({
            "rolle": "Trader", "aktion": "STARK VERKAUFEN", "zielkurs": None,
            "stop_loss": None, "positionsanteil": 0,
            "begründung": "Negativ", "zeithorizont": "Kurzfristig",
        })
        llm = _CapturingLLM(verkaufen_json)
        result = trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm)
        assert result["aktion"] == "VERKAUFEN"
        assert result["rating"] == "STARK VERKAUFEN"

    def test_fields_preserved(self):
        """zielkurs, stop_loss, positionsanteil, begründung bleiben erhalten."""
        llm = _CapturingLLM(_REVISION_JSON)
        result = trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm)
        assert result["zielkurs"] == 115
        assert result["stop_loss"] == 92
        assert result["positionsanteil"] == 3
        assert "Risk-Manager" in result["begründung"]


class TestTradeRevisionFeedbackReflection:
    """Testet dass feedback_context und reflection_context im Prompt landen."""

    def test_feedback_context_appended(self):
        """feedback_context wird am Ende des User-Prompts angehängt."""
        llm = _CapturingLLM(_REVISION_JSON)
        trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
                        feedback_context="TRACK_RECORD_MARKER_XYZ")
        user_msg = llm.captured_messages[0][1]["content"]
        assert "TRACK_RECORD_MARKER_XYZ" in user_msg

    def test_reflection_context_appended(self):
        """reflection_context wird am Ende des User-Prompts angehängt."""
        llm = _CapturingLLM(_REVISION_JSON)
        trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
                        reflection_context="REFLECTION_MARKER_ABC")
        user_msg = llm.captured_messages[0][1]["content"]
        assert "REFLECTION_MARKER_ABC" in user_msg

    def test_both_contexts_appended(self):
        """Beide Kontexte werden angehängt."""
        llm = _CapturingLLM(_REVISION_JSON)
        trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
                        feedback_context="FB_MARKER", reflection_context="RF_MARKER")
        user_msg = llm.captured_messages[0][1]["content"]
        assert "FB_MARKER" in user_msg
        assert "RF_MARKER" in user_msg


# --------------------------------------------------------------------------- #
# Feature A: Pipeline trade_revision Schritt
# --------------------------------------------------------------------------- #


class TestPipelineTradeRevision:
    """Testet dass die Pipeline den trade_revision Schritt zwischen PF und PM ausführt."""

    def test_trade_revised_flag_set(self):
        """Pipeline setzt trade_revised=True und trade_original nach Revision."""
        from unittest.mock import MagicMock, patch

        # Wir mocken die gesamten Agenten-Funktionen
        original_trade = {"rolle": "Trader", "aktion": "KAUFEN", "rating": "KAUFEN",
                          "zielkurs": 130, "stop_loss": 85, "positionsanteil": 8,
                          "begründung": "Stark", "zeithorizont": "Mittelfristig",
                          "_raw": ""}
        revised_trade = {"rolle": "Trader", "aktion": "KAUFEN", "rating": "KAUFEN",
                         "zielkurs": 115, "stop_loss": 92, "positionsanteil": 3,
                         "begründung": "Revidiert", "zeithorizont": "Mittelfristig",
                         "_raw": ""}

        with patch("concilium.pipeline.collect_ticker_data") as mock_data, \
             patch("concilium.pipeline.analyst_team") as mock_analysts, \
             patch("concilium.pipeline.debate") as mock_debate, \
             patch("concilium.pipeline.trader") as mock_trader, \
             patch("concilium.pipeline.risk_manager") as mock_risk, \
             patch("concilium.pipeline.fetch_portfolio_positions") as mock_positions, \
             patch("concilium.pipeline.portfolio_fit_agent") as mock_pf, \
             patch("concilium.pipeline.trade_revision") as mock_revision, \
             patch("concilium.pipeline.portfolio_manager") as mock_pm, \
             patch("concilium.pipeline.build_feedback_context", return_value=""), \
             patch("concilium.pipeline.build_reflection_context", return_value=""), \
             patch("concilium.journal.append_decision"):

            mock_data.return_value = {"ticker": "TEST", "fundamentals": {},
                                      "technicals": {}, "sentiment": {}, "news": []}
            mock_analysts.return_value = {}
            mock_debate.return_value = {}
            mock_trader.return_value = original_trade
            mock_risk.return_value = {"risiko_score": 4, "empfehlung": "MODIFIZIERT"}
            mock_positions.return_value = [{"name": "X", "ticker": "X", "depot_pct": 5}]
            mock_pf.return_value = {"portfolio_fit_score": 2}
            mock_revision.return_value = revised_trade
            mock_pm.return_value = {"entscheidung": "MODIFIZIERT", "confidence": 3}

            from concilium.pipeline import run_pipeline
            result = run_pipeline("TEST", llm=MagicMock(), ensemble=False)

            assert result["trade_revised"] is True
            assert result["trade_original"] == original_trade
            assert result["trade"] == revised_trade

    def test_trade_revised_false_on_exception(self):
        """Bei trade_revision Exception bleibt trade_revised=False, trade unverändert."""
        from unittest.mock import MagicMock, patch

        original_trade = {"rolle": "Trader", "aktion": "KAUFEN", "rating": "KAUFEN",
                          "zielkurs": 130, "_raw": ""}

        with patch("concilium.pipeline.collect_ticker_data") as mock_data, \
             patch("concilium.pipeline.analyst_team") as mock_analysts, \
             patch("concilium.pipeline.debate") as mock_debate, \
             patch("concilium.pipeline.trader") as mock_trader, \
             patch("concilium.pipeline.risk_manager") as mock_risk, \
             patch("concilium.pipeline.fetch_portfolio_positions") as mock_positions, \
             patch("concilium.pipeline.portfolio_fit_agent") as mock_pf, \
             patch("concilium.pipeline.trade_revision", side_effect=RuntimeError("LLM down")), \
             patch("concilium.pipeline.portfolio_manager") as mock_pm, \
             patch("concilium.pipeline.build_feedback_context", return_value=""), \
             patch("concilium.pipeline.build_reflection_context", return_value=""), \
             patch("concilium.journal.append_decision"):

            mock_data.return_value = {"ticker": "TEST", "fundamentals": {},
                                      "technicals": {}, "sentiment": {}, "news": []}
            mock_analysts.return_value = {}
            mock_debate.return_value = {}
            mock_trader.return_value = original_trade
            mock_risk.return_value = {"risiko_score": 4, "empfehlung": "ABGELEHNT"}
            mock_positions.return_value = [{"name": "X", "ticker": "X", "depot_pct": 5}]
            mock_pf.return_value = {"portfolio_fit_score": 2}
            mock_pm.return_value = {"entscheidung": "ABGELEHNT", "confidence": 2}

            from concilium.pipeline import run_pipeline
            result = run_pipeline("TEST", llm=MagicMock(), ensemble=False)

            assert result["trade_revised"] is False
            assert result["trade_original"] is None
            assert result["trade"] == original_trade

    def test_revision_passes_portfolio_fit(self):
        """trade_revision wird mit portfolio_fit aufgerufen (nicht None)."""
        from unittest.mock import MagicMock, patch

        pf_result = {"portfolio_fit_score": 3, "ziel_gewichtung_pct": 2}

        with patch("concilium.pipeline.collect_ticker_data") as mock_data, \
             patch("concilium.pipeline.analyst_team") as mock_analysts, \
             patch("concilium.pipeline.debate") as mock_debate, \
             patch("concilium.pipeline.trader") as mock_trader, \
             patch("concilium.pipeline.risk_manager") as mock_risk, \
             patch("concilium.pipeline.fetch_portfolio_positions") as mock_positions, \
             patch("concilium.pipeline.portfolio_fit_agent") as mock_pf, \
             patch("concilium.pipeline.trade_revision") as mock_revision, \
             patch("concilium.pipeline.portfolio_manager") as mock_pm, \
             patch("concilium.pipeline.build_feedback_context", return_value=""), \
             patch("concilium.pipeline.build_reflection_context", return_value=""), \
             patch("concilium.journal.append_decision"):

            mock_data.return_value = {"ticker": "TEST", "fundamentals": {},
                                      "technicals": {}, "sentiment": {}, "news": []}
            mock_analysts.return_value = {}
            mock_debate.return_value = {}
            mock_trader.return_value = {"aktion": "KAUFEN", "_raw": ""}
            mock_risk.return_value = {"risiko_score": 4}
            mock_positions.return_value = [{"name": "X", "ticker": "X", "depot_pct": 5}]
            mock_pf.return_value = pf_result
            mock_revision.return_value = {"aktion": "KAUFEN", "rating": "KAUFEN", "_raw": ""}
            mock_pm.return_value = {"entscheidung": "GENEHMIGT", "confidence": 4}

            from concilium.pipeline import run_pipeline
            run_pipeline("TEST", llm=MagicMock(), ensemble=False)

            # trade_revision wurde mit portfolio_fit=pf_result aufgerufen
            call_args = mock_revision.call_args
            assert call_args.args[2] == pf_result  # 3rd positional arg = portfolio_fit

    def test_revision_passes_none_when_no_portfolio_fit(self):
        """trade_revision wird mit portfolio_fit=None aufgerufen, wenn PF fehlgeschlagen."""
        from unittest.mock import MagicMock, patch

        with patch("concilium.pipeline.collect_ticker_data") as mock_data, \
             patch("concilium.pipeline.analyst_team") as mock_analysts, \
             patch("concilium.pipeline.debate") as mock_debate, \
             patch("concilium.pipeline.trader") as mock_trader, \
             patch("concilium.pipeline.risk_manager") as mock_risk, \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_fit_agent", side_effect=Exception("fail")), \
             patch("concilium.pipeline.trade_revision") as mock_revision, \
             patch("concilium.pipeline.portfolio_manager") as mock_pm, \
             patch("concilium.pipeline.build_feedback_context", return_value=""), \
             patch("concilium.pipeline.build_reflection_context", return_value=""), \
             patch("concilium.journal.append_decision"):

            mock_data.return_value = {"ticker": "TEST", "fundamentals": {},
                                      "technicals": {}, "sentiment": {}, "news": []}
            mock_analysts.return_value = {}
            mock_debate.return_value = {}
            mock_trader.return_value = {"aktion": "KAUFEN", "_raw": ""}
            mock_risk.return_value = {"risiko_score": 4}
            mock_revision.return_value = {"aktion": "KAUFEN", "rating": "KAUFEN", "_raw": ""}
            mock_pm.return_value = {"entscheidung": "GENEHMIGT", "confidence": 4}

            from concilium.pipeline import run_pipeline
            run_pipeline("TEST", llm=MagicMock(), ensemble=False)

            call_args = mock_revision.call_args
            assert call_args.args[2] is None  # portfolio_fit = None

    def test_revised_trade_goes_to_pm(self):
        """Der revidierte Trade (nicht der Original) wird an den PM übergeben."""
        from unittest.mock import MagicMock, patch

        original_trade = {"aktion": "KAUFEN", "zielkurs": 130, "_raw": "orig"}
        revised_trade = {"aktion": "KAUFEN", "zielkurs": 115, "_raw": "revised"}

        with patch("concilium.pipeline.collect_ticker_data") as mock_data, \
             patch("concilium.pipeline.analyst_team") as mock_analysts, \
             patch("concilium.pipeline.debate") as mock_debate, \
             patch("concilium.pipeline.trader") as mock_trader, \
             patch("concilium.pipeline.risk_manager") as mock_risk, \
             patch("concilium.pipeline.fetch_portfolio_positions") as mock_positions, \
             patch("concilium.pipeline.portfolio_fit_agent") as mock_pf, \
             patch("concilium.pipeline.trade_revision") as mock_revision, \
             patch("concilium.pipeline.portfolio_manager") as mock_pm, \
             patch("concilium.pipeline.build_feedback_context", return_value=""), \
             patch("concilium.pipeline.build_reflection_context", return_value=""), \
             patch("concilium.journal.append_decision"):

            mock_data.return_value = {"ticker": "TEST", "fundamentals": {},
                                      "technicals": {}, "sentiment": {}, "news": []}
            mock_analysts.return_value = {}
            mock_debate.return_value = {}
            mock_trader.return_value = original_trade
            mock_risk.return_value = {"risiko_score": 4}
            mock_positions.return_value = [{"name": "X", "ticker": "X", "depot_pct": 5}]
            mock_pf.return_value = {"portfolio_fit_score": 2}
            mock_revision.return_value = revised_trade
            mock_pm.return_value = {"entscheidung": "GENEHMIGT", "confidence": 4}

            from concilium.pipeline import run_pipeline
            run_pipeline("TEST", llm=MagicMock(), ensemble=False)

            # PM wurde mit revised_trade aufgerufen (1st positional arg = trade)
            pm_call_trade = mock_pm.call_args.args[0]
            assert pm_call_trade == revised_trade


# --------------------------------------------------------------------------- #
# Feature B: PM MODIFIZIERT im Report
# --------------------------------------------------------------------------- #


class TestReportModifiziert:
    """Testet dass der Report MODIFIZIERT als Entscheidung rendern kann."""

    def test_report_shows_modifiziert(self):
        """PM-Abschnitt im Report zeigt MODIFIZIERT mit ⚡ Emoji."""
        from concilium.report import generate_report

        result = {
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
            "debate": {
                "bull": {"_raw": "Bull"},
                "bear": {"_raw": "Bear"},
            },
            "trade": {
                "aktion": "KAUFEN",
                "zielkurs": 180,
                "stop_loss": 130,
                "positionsanteil": 7,
                "begründung": "Test",
                "zeithorizont": "Mittelfristig",
            },
            "risk": {"risiko_score": 3, "empfehlung": "MODIFIZIERT"},
            "final": {"entscheidung": "MODIFIZIERT", "confidence": 3,
                       "begründung": "Mit Auflagen genehmigt."},
        }

        report = generate_report(result)
        assert "MODIFIZIERT" in report
        assert "⚡" in report

    def test_report_genehmigt_still_works(self):
        """GENEHMIGT wird weiterhin mit ✅ gerendert."""
        from concilium.report import generate_report

        result = {
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
            "debate": {
                "bull": {"_raw": "Bull"},
                "bear": {"_raw": "Bear"},
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
            "final": {"entscheidung": "GENEHMIGT", "confidence": 4,
                       "begründung": "Genehmigt."},
        }

        report = generate_report(result)
        assert "GENEHMIGT" in report
        assert "✅" in report

    def test_report_abgelehnt_still_works(self):
        """ABGELEHNT wird weiterhin mit ❌ gerendert."""
        from concilium.report import generate_report

        result = {
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
            "debate": {
                "bull": {"_raw": "Bull"},
                "bear": {"_raw": "Bear"},
            },
            "trade": {
                "aktion": "HALTEN",
                "zielkurs": None,
                "stop_loss": None,
                "positionsanteil": 0,
                "begründung": "Neutral",
                "zeithorizont": "Mittelfristig",
            },
            "risk": {"risiko_score": 5, "empfehlung": "ABGELEHNT"},
            "final": {"entscheidung": "ABGELEHNT", "confidence": 1,
                       "begründung": "Abgelehnt."},
        }

        report = generate_report(result)
        assert "ABGELEHNT" in report
        assert "❌" in report

    def test_report_trade_revised_hinweis(self):
        """Report zeigt den Revisions-Hinweis wenn trade_revised=True."""
        from concilium.report import generate_report

        result = {
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
            "debate": {
                "bull": {"_raw": "Bull"},
                "bear": {"_raw": "Bear"},
            },
            "trade": {
                "aktion": "KAUFEN",
                "zielkurs": 115,
                "stop_loss": 92,
                "positionsanteil": 3,
                "begründung": "Revidiert",
                "zeithorizont": "Mittelfristig",
            },
            "trade_original": {
                "aktion": "KAUFEN",
                "zielkurs": 130,
                "positionsanteil": 8,
            },
            "trade_revised": True,
            "risk": {"risiko_score": 4, "empfehlung": "MODIFIZIERT"},
            "final": {"entscheidung": "MODIFIZIERT", "confidence": 3,
                       "begründung": "Mit Auflagen."},
        }

        report = generate_report(result)
        assert "revidiert" in report.lower()
        assert "Original-Trade" in report


# --------------------------------------------------------------------------- #
# Feature B: SYSTEM_PM Prompt enthält MODIFIZIERT
# --------------------------------------------------------------------------- #


class TestSystemPMModifiziert:
    """Testet dass der SYSTEM_PM Prompt MODIFIZIERT als Option enthält."""

    def test_system_pm_contains_modifiziert(self):
        """SYSTEM_PM enthält MODIFIZIERT als Entscheidungsoption."""
        assert "MODIFIZIERT" in SYSTEM_PM

    def test_system_pm_contains_three_options(self):
        """SYSTEM_PM enthält alle drei Entscheidungsoptionen."""
        assert "GENEHMIGT" in SYSTEM_PM
        assert "MODIFIZIERT" in SYSTEM_PM
        assert "ABGELEHNT" in SYSTEM_PM

    def test_system_pm_contains_modifiziert_description(self):
        """SYSTEM_PM enthält die Beschreibung für MODIFIZIERT."""
        assert "Auflagen" in SYSTEM_PM or "Bedingungen" in SYSTEM_PM
