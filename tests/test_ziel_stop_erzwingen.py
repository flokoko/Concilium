"""Tests für Feature 1: Ziel & Stop erzwingen in Trade-Revision.

Der LLM-Call wird gemockt — Tests prüfen den deterministischen Fallback.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import trade_revision  # noqa: E402

# --------------------------------------------------------------------------- #
# Hilfsklassen und Fixtures
# --------------------------------------------------------------------------- #


class _CapturingLLM:
    """Mock-LLM, der eine vordefinierte JSON-Antwort zurückgibt."""

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


_RISK = {"risiko_score": 4, "empfehlung": "MODIFIZIERT"}
_PORTFOLIO_FIT = {"portfolio_fit_score": 2}
_ORIGINAL_TRADE = {"aktion": "KAUFEN", "zielkurs": 130, "stop_loss": 85}


# --------------------------------------------------------------------------- #
# Tests: KAUFEN — deterministischer Fallback
# --------------------------------------------------------------------------- #


class TestKaufenFallback:
    """KAUFEN: zielkurs/stop_loss werden deterministisch gesetzt wenn fehlend."""

    def test_kaufen_missing_ziel_and_stop(self):
        """KAUFEN ohne zielkurs/stop_loss → Fallback auf 10% Up-/Downside."""
        llm_json = json.dumps({
            "rolle": "Trader", "aktion": "KAUFEN", "zielkurs": None,
            "stop_loss": None, "positionsanteil": 5,
            "begründung": "Test", "zeithorizont": "Mittelfristig",
        })
        llm = _CapturingLLM(llm_json)
        result = trade_revision(
            _ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
            current_price=100.0,
        )
        assert result["zielkurs"] is not None
        assert result["stop_loss"] is not None
        # zielkurs > current_price (10% Upside)
        assert float(result["zielkurs"]) == 110.0
        # stop_loss < current_price (10% Downside)
        assert float(result["stop_loss"]) == 90.0

    def test_kaufen_implausible_ziel_replaced(self):
        """KAUFEN mit zielkurs <= current_price → wird durch Fallback ersetzt."""
        llm_json = json.dumps({
            "rolle": "Trader", "aktion": "KAUFEN", "zielkurs": 80,  # <= 100
            "stop_loss": 90,  # plausibel (< current)
            "positionsanteil": 5,
            "begründung": "Test", "zeithorizont": "Mittelfristig",
        })
        llm = _CapturingLLM(llm_json)
        result = trade_revision(
            _ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
            current_price=100.0,
        )
        # zielkurs ersetzt durch 110.0
        assert float(result["zielkurs"]) == 110.0
        # stop_loss war plausibel → bleibt unverändert
        assert float(result["stop_loss"]) == 90.0

    def test_kaufen_implausible_stop_replaced(self):
        """KAUFEN mit stop_loss >= current_price → wird durch Fallback ersetzt."""
        llm_json = json.dumps({
            "rolle": "Trader", "aktion": "KAUFEN", "zielkurs": 120,  # plausibel
            "stop_loss": 110,  # >= 100 → unplausibel
            "positionsanteil": 5,
            "begründung": "Test", "zeithorizont": "Mittelfristig",
        })
        llm = _CapturingLLM(llm_json)
        result = trade_revision(
            _ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
            current_price=100.0,
        )
        # zielkurs war plausibel → bleibt
        assert float(result["zielkurs"]) == 120.0
        # stop_loss ersetzt durch 90.0
        assert float(result["stop_loss"]) == 90.0

    def test_kaufen_plausible_values_not_overwritten(self):
        """KAUFEN mit plausiblem zielkurs/stop_loss → Werte bleiben unverändert."""
        llm_json = json.dumps({
            "rolle": "Trader", "aktion": "KAUFEN", "zielkurs": 150,
            "stop_loss": 85, "positionsanteil": 5,
            "begründung": "Test", "zeithorizont": "Mittelfristig",
        })
        llm = _CapturingLLM(llm_json)
        result = trade_revision(
            _ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
            current_price=100.0,
        )
        assert result["zielkurs"] == 150
        assert result["stop_loss"] == 85


# --------------------------------------------------------------------------- #
# Tests: STARK KAUFEN — gleicher Fallback wie KAUFEN
# --------------------------------------------------------------------------- #


class TestStarkKaufenFallback:
    """STARK KAUFEN → aktion=KAUFEN, gleicher Fallback."""

    def test_stark_kaufen_missing_ziel_stop(self):
        """STARK KAUFEN ohne zielkurs/stop_loss → Fallback wie KAUFEN."""
        llm_json = json.dumps({
            "rolle": "Trader", "aktion": "STARK KAUFEN", "zielkurs": None,
            "stop_loss": None, "positionsanteil": 10,
            "begründung": "Sehr bullisch", "zeithorizont": "Langfristig",
        })
        llm = _CapturingLLM(llm_json)
        result = trade_revision(
            _ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
            current_price=50.0,
        )
        assert result["aktion"] == "KAUFEN"
        assert float(result["zielkurs"]) == 55.0  # 50 * 1.10
        assert float(result["stop_loss"]) == 45.0  # 50 * 0.90


# --------------------------------------------------------------------------- #
# Tests: VERKAUFEN — deterministischer Fallback (umgekehrt)
# --------------------------------------------------------------------------- #


class TestVerkaufenFallback:
    """VERKAUFEN: zielkurs < current, stop_loss > current."""

    def test_verkaufen_missing_ziel_and_stop(self):
        """VERKAUFEN ohne zielkurs/stop_loss → Fallback auf 0.9x / 1.1x."""
        llm_json = json.dumps({
            "rolle": "Trader", "aktion": "VERKAUFEN", "zielkurs": None,
            "stop_loss": None, "positionsanteil": 0,
            "begründung": "Bearish", "zeithorizont": "Kurzfristig",
        })
        llm = _CapturingLLM(llm_json)
        result = trade_revision(
            _ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
            current_price=100.0,
        )
        # zielkurs < current_price (10% Downside)
        assert float(result["zielkurs"]) == 90.0
        # stop_loss > current_price (10% Upside)
        assert float(result["stop_loss"]) == 110.0

    def test_verkaufen_implausible_ziel_replaced(self):
        """VERKAUFEN mit zielkurs >= current → wird ersetzt."""
        llm_json = json.dumps({
            "rolle": "Trader", "aktion": "VERKAUFEN", "zielkurs": 120,  # >= 100
            "stop_loss": 110,  # plausibel (> current)
            "positionsanteil": 0,
            "begründung": "Bearish", "zeithorizont": "Kurzfristig",
        })
        llm = _CapturingLLM(llm_json)
        result = trade_revision(
            _ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
            current_price=100.0,
        )
        assert float(result["zielkurs"]) == 90.0
        assert float(result["stop_loss"]) == 110.0

    def test_verkaufen_plausible_values_not_overwritten(self):
        """VERKAUFEN mit plausiblem zielkurs/stop_loss → Werte bleiben."""
        llm_json = json.dumps({
            "rolle": "Trader", "aktion": "VERKAUFEN", "zielkurs": 80,
            "stop_loss": 120, "positionsanteil": 0,
            "begründung": "Bearish", "zeithorizont": "Kurzfristig",
        })
        llm = _CapturingLLM(llm_json)
        result = trade_revision(
            _ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
            current_price=100.0,
        )
        assert result["zielkurs"] == 80
        assert result["stop_loss"] == 120


# --------------------------------------------------------------------------- #
# Tests: STARK VERKAUFEN — gleicher Fallback wie VERKAUFEN
# --------------------------------------------------------------------------- #


class TestStarkVerkaufenFallback:
    """STARK VERKAUFEN → aktion=VERKAUFEN, gleicher Fallback."""

    def test_stark_verkaufen_missing_ziel_stop(self):
        """STARK VERKAUFEN ohne zielkurs/stop_loss → Fallback wie VERKAUFEN."""
        llm_json = json.dumps({
            "rolle": "Trader", "aktion": "STARK VERKAUFEN", "zielkurs": None,
            "stop_loss": None, "positionsanteil": 0,
            "begründung": "Sehr bearish", "zeithorizont": "Kurzfristig",
        })
        llm = _CapturingLLM(llm_json)
        result = trade_revision(
            _ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
            current_price=200.0,
        )
        assert result["aktion"] == "VERKAUFEN"
        assert float(result["zielkurs"]) == 180.0  # 200 * 0.90
        assert float(result["stop_loss"]) == 220.0  # 200 * 1.10


# --------------------------------------------------------------------------- #
# Tests: HALTEN — keine Erzwingung
# --------------------------------------------------------------------------- #


class TestHaltenNoEnforcement:
    """HALTEN: zielkurs/stop_loss bleiben None wenn nicht vom Trader geliefert."""

    def test_halten_none_values_stay_none(self):
        """HALTEN mit None-Werten → keine Erzwingung, bleiben None."""
        llm_json = json.dumps({
            "rolle": "Trader", "aktion": "HALTEN", "zielkurs": None,
            "stop_loss": None, "positionsanteil": 0,
            "begründung": "Abwarten", "zeithorizont": "Mittelfristig",
        })
        llm = _CapturingLLM(llm_json)
        result = trade_revision(
            _ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
            current_price=100.0,
        )
        assert result["aktion"] == "HALTEN"
        assert result["zielkurs"] is None
        assert result["stop_loss"] is None

    def test_halten_trader_values_preserved(self):
        """HALTEN mit Trader-Werten → Werte bleiben (auch wenn 'unplausibel')."""
        llm_json = json.dumps({
            "rolle": "Trader", "aktion": "HALTEN", "zielkurs": 120,
            "stop_loss": 80, "positionsanteil": 0,
            "begründung": "Abwarten", "zeithorizont": "Mittelfristig",
        })
        llm = _CapturingLLM(llm_json)
        result = trade_revision(
            _ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
            current_price=100.0,
        )
        # HALTEN: Werte bleiben unverändert
        assert result["zielkurs"] == 120
        assert result["stop_loss"] == 80


# --------------------------------------------------------------------------- #
# Tests: current_price=None — kein Crash, kein Fallback
# --------------------------------------------------------------------------- #


class TestCurrentPriceNone:
    """current_price=None → kein Fallback, kein Crash."""

    def test_none_current_price_no_fallback(self):
        """Ohne current_price werden fehlende Werte nicht gesetzt."""
        llm_json = json.dumps({
            "rolle": "Trader", "aktion": "KAUFEN", "zielkurs": None,
            "stop_loss": None, "positionsanteil": 5,
            "begründung": "Test", "zeithorizont": "Mittelfristig",
        })
        llm = _CapturingLLM(llm_json)
        result = trade_revision(
            _ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
            current_price=None,
        )
        # Kein Fallback — Werte bleiben None
        assert result["zielkurs"] is None
        assert result["stop_loss"] is None
        assert result["aktion"] == "KAUFEN"

    def test_none_current_price_default_param(self):
        """Ohne current_price-Parameter (Default None) → ebenfalls kein Fallback."""
        llm_json = json.dumps({
            "rolle": "Trader", "aktion": "KAUFEN", "zielkurs": None,
            "stop_loss": None, "positionsanteil": 5,
            "begründung": "Test", "zeithorizont": "Mittelfristig",
        })
        llm = _CapturingLLM(llm_json)
        result = trade_revision(_ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm)
        assert result["zielkurs"] is None
        assert result["stop_loss"] is None


# --------------------------------------------------------------------------- #
# Tests: Rounding
# --------------------------------------------------------------------------- #


class TestRounding:
    """Prüft dass die Fallback-Werte auf 2 Nachkommastellen gerundet werden."""

    def test_kaufen_rounding(self):
        """current_price=33.33 → zielkurs=36.66, stop_loss=30.00."""
        llm_json = json.dumps({
            "rolle": "Trader", "aktion": "KAUFEN", "zielkurs": None,
            "stop_loss": None, "positionsanteil": 5,
            "begründung": "Test", "zeithorizont": "Mittelfristig",
        })
        llm = _CapturingLLM(llm_json)
        result = trade_revision(
            _ORIGINAL_TRADE, _RISK, _PORTFOLIO_FIT, llm,
            current_price=33.33,
        )
        assert float(result["zielkurs"]) == round(33.33 * 1.10, 2)
        assert float(result["stop_loss"]) == round(33.33 * 0.90, 2)
