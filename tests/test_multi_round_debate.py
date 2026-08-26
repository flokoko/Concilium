"""Tests für die Multi-Runden-Debatte (Bull/Bear Ping-Pong).

Testet:
- debate mit rounds=1 verhält sich wie bisher (2 Calls, bull/bear vorhanden).
- debate mit rounds=2 macht 4 Calls (2 Bull + 2 Bear) und die zweite Runde
  bekommt die Gegner-Argumentation im user_text.
- pipeline reicht debate_rounds an debate durch.
- cli parst --debate-rounds korrekt.
- report zeigt Rundenanzahl bei rounds > 1.

Alle Tests sind offline (kein Netzwerk) — _call_agent wird gemockt.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import debate  # noqa: E402

_MINIMAL_ANALYSTS = {
    "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
}


class _MockStructuredResult:
    """Mock StructuredChatResult für _call_agent structured path."""

    def __init__(self, text: str):
        self.text = text
        self.response_format_used = True


class TestDebateSingleRound:
    """debate(rounds=1) verhält sich wie bisher (2 Calls, bull/bear vorhanden)."""

    def test_rounds1_returns_bull_bear(self):
        """rounds=1 → bull und bear keys vorhanden."""
        with patch("concilium.agents._call_agent") as mock_call:
            mock_call.side_effect = [
                {"argumente": "Bull text", "confidence": 4, "_raw": "bull raw"},
                {"argumente": "Bear text", "confidence": 2, "_raw": "bear raw"},
            ]
            result = debate(_MINIMAL_ANALYSTS, MagicMock(), rounds=1)

        assert "bull" in result
        assert "bear" in result
        assert mock_call.call_count == 2

    def test_rounds1_has_rounds_key(self):
        """rounds=1 → 'rounds' key im Ergebnis."""
        with patch("concilium.agents._call_agent") as mock_call:
            mock_call.side_effect = [
                {"argumente": "Bull", "confidence": 4, "_raw": ""},
                {"argumente": "Bear", "confidence": 2, "_raw": ""},
            ]
            result = debate(_MINIMAL_ANALYSTS, MagicMock(), rounds=1)

        assert result["rounds"] == 1

    def test_rounds1_no_opponent_section(self):
        """rounds=1 → kein 'Argumentation der Gegenseite' im user_text."""
        with patch("concilium.agents._call_agent") as mock_call:
            mock_call.side_effect = [
                {"argumente": "Bull", "confidence": 4, "_raw": ""},
                {"argumente": "Bear", "confidence": 2, "_raw": ""},
            ]
            debate(_MINIMAL_ANALYSTS, MagicMock(), rounds=1)

        # Bei rounds=1 sollte kein "Argumentation der Gegenseite" im user_text sein
        for call in mock_call.call_args_list:
            user_text = call.args[2]  # 3. positional arg = user_text
            assert "Argumentation der Gegenseite" not in user_text


class TestDebateMultiRound:
    """debate(rounds=2) macht 4 Calls und zweite Runde bekommt Gegner-Argumentation."""

    def test_rounds2_makes_4_calls(self):
        """rounds=2 → 4 Calls (2 Bull + 2 Bear)."""
        with patch("concilium.agents._call_agent") as mock_call:
            mock_call.side_effect = [
                {"argumente": "Bull R1", "confidence": 4, "_raw": ""},
                {"argumente": "Bear R1", "confidence": 2, "_raw": ""},
                {"argumente": "Bull R2", "confidence": 4, "_raw": ""},
                {"argumente": "Bear R2", "confidence": 3, "_raw": ""},
            ]
            result = debate(_MINIMAL_ANALYSTS, MagicMock(), rounds=2)

        assert mock_call.call_count == 4
        assert result["rounds"] == 2

    def test_rounds2_second_bull_has_opponent_section(self):
        """rounds=2 → der 3. Call (Bull R2) enthält 'Argumentation der Gegenseite'."""
        with patch("concilium.agents._call_agent") as mock_call:
            mock_call.side_effect = [
                {"argumente": "Bull R1", "confidence": 4, "_raw": ""},
                {"argumente": "Bear R1 text", "confidence": 2, "_raw": ""},
                {"argumente": "Bull R2", "confidence": 4, "_raw": ""},
                {"argumente": "Bear R2", "confidence": 3, "_raw": ""},
            ]
            debate(_MINIMAL_ANALYSTS, MagicMock(), rounds=2)

        # 3. Call = Bull in Runde 2 → sollte Bear R1 Argumentation enthalten
        bull_r2_usertext = mock_call.call_args_list[2].args[2]
        assert "Argumentation der Gegenseite" in bull_r2_usertext
        assert "Bear R1 text" in bull_r2_usertext

    def test_rounds2_second_bear_has_opponent_section(self):
        """rounds=2 → der 4. Call (Bear R2) enthält 'Argumentation der Gegenseite'."""
        with patch("concilium.agents._call_agent") as mock_call:
            mock_call.side_effect = [
                {"argumente": "Bull R1 text", "confidence": 4, "_raw": ""},
                {"argumente": "Bear R1", "confidence": 2, "_raw": ""},
                {"argumente": "Bull R2 text", "confidence": 4, "_raw": ""},
                {"argumente": "Bear R2", "confidence": 3, "_raw": ""},
            ]
            debate(_MINIMAL_ANALYSTS, MagicMock(), rounds=2)

        # 4. Call = Bear in Runde 2 → sollte Bull R2 Argumentation enthalten
        bear_r2_usertext = mock_call.call_args_list[3].args[2]
        assert "Argumentation der Gegenseite" in bear_r2_usertext
        assert "Bull R2 text" in bear_r2_usertext

    def test_rounds2_first_round_no_opponent_section(self):
        """rounds=2 → erste Runde (Calls 1+2) hat KEINE 'Argumentation der Gegenseite'."""
        with patch("concilium.agents._call_agent") as mock_call:
            mock_call.side_effect = [
                {"argumente": "Bull R1", "confidence": 4, "_raw": ""},
                {"argumente": "Bear R1", "confidence": 2, "_raw": ""},
                {"argumente": "Bull R2", "confidence": 4, "_raw": ""},
                {"argumente": "Bear R2", "confidence": 3, "_raw": ""},
            ]
            debate(_MINIMAL_ANALYSTS, MagicMock(), rounds=2)

        # Erste 2 Calls sollten keinen Gegner-Block haben
        for i in range(2):
            user_text = mock_call.call_args_list[i].args[2]
            assert "Argumentation der Gegenseite" not in user_text

    def test_rounds2_returns_last_round_results(self):
        """rounds=2 → bull/bear sind die Dicts aus der letzten Runde."""
        with patch("concilium.agents._call_agent") as mock_call:
            mock_call.side_effect = [
                {"argumente": "Bull R1", "confidence": 3, "_raw": ""},
                {"argumente": "Bear R1", "confidence": 1, "_raw": ""},
                {"argumente": "Bull R2 final", "confidence": 5, "_raw": ""},
                {"argumente": "Bear R2 final", "confidence": 4, "_raw": ""},
            ]
            result = debate(_MINIMAL_ANALYSTS, MagicMock(), rounds=2)

        assert result["bull"]["argumente"] == "Bull R2 final"
        assert result["bear"]["argumente"] == "Bear R2 final"

    def test_rounds3_makes_6_calls(self):
        """rounds=3 → 6 Calls (3 Bull + 3 Bear)."""
        with patch("concilium.agents._call_agent") as mock_call:
            mock_call.side_effect = [
                {"argumente": f"Bull R{r+1}", "confidence": 4, "_raw": ""}
                for r in range(3)
            ] + [
                {"argumente": f"Bear R{r+1}", "confidence": 2, "_raw": ""}
                for r in range(3)
            ]
            # Reihenfolge: Bull1, Bear1, Bull2, Bear2, Bull3, Bear3
            mock_call.side_effect = [
                {"argumente": "Bull R1", "confidence": 4, "_raw": ""},
                {"argumente": "Bear R1", "confidence": 2, "_raw": ""},
                {"argumente": "Bull R2", "confidence": 4, "_raw": ""},
                {"argumente": "Bear R2", "confidence": 2, "_raw": ""},
                {"argumente": "Bull R3", "confidence": 4, "_raw": ""},
                {"argumente": "Bear R3", "confidence": 2, "_raw": ""},
            ]
            result = debate(_MINIMAL_ANALYSTS, MagicMock(), rounds=3)

        assert mock_call.call_count == 6
        assert result["rounds"] == 3


class TestPipelineDebateRounds:
    """pipeline reicht debate_rounds an debate durch."""

    def test_pipeline_passes_debate_rounds(self):
        """run_pipeline reicht debate_rounds an debate() weiter."""
        from concilium.pipeline import run_pipeline

        mock_data = {
            "ticker": "TEST",
            "fundamentals": {"name": "Test", "sector": "X"},
            "technicals": {"current_price": 100.0},
            "sentiment": {},
            "news": [],
            "macro": {},
            "peers": [],
            "history": [{"close": 100.0}],
            "data_warnings": [],
        }

        with patch("concilium.pipeline.collect_ticker_data", return_value=mock_data), \
             patch("concilium.pipeline.analyst_team", return_value=_MINIMAL_ANALYSTS), \
             patch("concilium.pipeline.debate", return_value={"bull": {}, "bear": {}}) as mock_debate, \
             patch("concilium.pipeline.trader", return_value={"aktion": "HALTEN"}), \
             patch("concilium.pipeline.risk_manager", return_value={"risiko_score": 3, "empfehlung": "GENEHMIGT"}), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_fit_agent", return_value=None), \
             patch("concilium.pipeline.trade_revision", return_value={"aktion": "HALTEN"}), \
             patch("concilium.pipeline.portfolio_manager", return_value={"entscheidung": "GENEHMIGT", "confidence": 4}), \
             patch("concilium.pipeline.build_feedback_context", return_value=""), \
             patch("concilium.pipeline.build_reflection_context", return_value=""), \
             patch("concilium.journal.append_decision"):
            run_pipeline("TEST", llm=MagicMock(), ensemble=False, debate_rounds=3)

        # debate sollte mit rounds=3 aufgerufen worden sein
        assert mock_debate.called
        kwargs = mock_debate.call_args
        assert kwargs.kwargs.get("rounds") == 3

    def test_pipeline_default_debate_rounds_is_1(self):
        """run_pipeline ohne debate_rounds → debate wird mit rounds=1 aufgerufen."""
        from concilium.pipeline import run_pipeline

        mock_data = {
            "ticker": "TEST",
            "fundamentals": {"name": "Test", "sector": "X"},
            "technicals": {"current_price": 100.0},
            "sentiment": {},
            "news": [],
            "macro": {},
            "peers": [],
            "history": [{"close": 100.0}],
            "data_warnings": [],
        }

        with patch("concilium.pipeline.collect_ticker_data", return_value=mock_data), \
             patch("concilium.pipeline.analyst_team", return_value=_MINIMAL_ANALYSTS), \
             patch("concilium.pipeline.debate", return_value={"bull": {}, "bear": {}}) as mock_debate, \
             patch("concilium.pipeline.trader", return_value={"aktion": "HALTEN"}), \
             patch("concilium.pipeline.risk_manager", return_value={"risiko_score": 3, "empfehlung": "GENEHMIGT"}), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_fit_agent", return_value=None), \
             patch("concilium.pipeline.trade_revision", return_value={"aktion": "HALTEN"}), \
             patch("concilium.pipeline.portfolio_manager", return_value={"entscheidung": "GENEHMIGT", "confidence": 4}), \
             patch("concilium.pipeline.build_feedback_context", return_value=""), \
             patch("concilium.pipeline.build_reflection_context", return_value=""), \
             patch("concilium.journal.append_decision"):
            run_pipeline("TEST", llm=MagicMock(), ensemble=False)

        assert mock_debate.called
        kwargs = mock_debate.call_args
        assert kwargs.kwargs.get("rounds") == 1


class TestCliDebateRounds:
    """cli parst --debate-rounds korrekt."""

    def test_cli_parses_debate_rounds_default(self):
        """--debate-rounds nicht gesetzt → default=1."""
        from concilium.cli import main

        # --no-llm um echtes LLM zu vermeiden, --ticker um Modus zu setzen
        with patch("concilium.cli.run_pipeline") as mock_pipeline, \
             patch("concilium.cli.generate_report", return_value="report"), \
             patch("builtins.print"):
            ret = main(["--ticker", "AAPL", "--no-llm"])
            assert ret == 0
            assert mock_pipeline.called
            assert mock_pipeline.call_args.kwargs.get("debate_rounds") == 1

    def test_cli_parses_debate_rounds_custom(self):
        """--debate-rounds 3 → debate_rounds=3."""
        from concilium.cli import main

        with patch("concilium.cli.run_pipeline") as mock_pipeline, \
             patch("concilium.cli.generate_report", return_value="report"), \
             patch("builtins.print"):
            ret = main(["--ticker", "AAPL", "--no-llm", "--debate-rounds", "3"])
            assert ret == 0
            assert mock_pipeline.called
            assert mock_pipeline.call_args.kwargs.get("debate_rounds") == 3


class TestReportDebateRounds:
    """Report zeigt Rundenanzahl bei rounds > 1."""

    def test_report_shows_rounds_when_gt_1(self):
        """Report zeigt 'Debatte über N Runden' bei rounds > 1."""
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
                "bull": {"_raw": '{"confidence": 4, "name": "Bull"}\nBull text'},
                "bear": {"_raw": '{"confidence": 2, "name": "Bear"}\nBear text'},
                "bull_confidence": 4,
                "bear_confidence": 2,
                "rounds": 3,
            },
            "trade": {"aktion": "KAUFEN", "zielkurs": 180, "stop_loss": 130, "positionsanteil": 7, "begründung": "Test", "zeithorizont": "Mittelfristig"},
            "risk": {"risiko_score": 3, "empfehlung": "GENEHMIGT"},
            "final": {"entscheidung": "GENEHMIGT", "confidence": 4, "begründung": "Ok"},
        }
        report = generate_report(result)
        assert "Debatte über 3 Runden" in report

    def test_report_no_rounds_when_1(self):
        """Report zeigt KEINE Runden-Zeile bei rounds=1."""
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
                "bull": {"_raw": '{"confidence": 4, "name": "Bull"}\nBull text'},
                "bear": {"_raw": '{"confidence": 2, "name": "Bear"}\nBear text'},
                "bull_confidence": 4,
                "bear_confidence": 2,
                "rounds": 1,
            },
            "trade": {"aktion": "KAUFEN", "zielkurs": 180, "stop_loss": 130, "positionsanteil": 7, "begründung": "Test", "zeithorizont": "Mittelfristig"},
            "risk": {"risiko_score": 3, "empfehlung": "GENEHMIGT"},
            "final": {"entscheidung": "GENEHMIGT", "confidence": 4, "begründung": "Ok"},
        }
        report = generate_report(result)
        assert "Debatte über 1 Runden" not in report
