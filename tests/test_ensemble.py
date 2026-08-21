"""Tests für ensemble_trader — Mehrheitsabstimmung, Plausibilitäts-Check, Single-Fallback.

Diese Tests benötigen KEIN Netzwerk — der LLMClient wird gemockt.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import (  # noqa: E402
    _fix_implausible_trade,
    _is_plausible_kauf,
    ensemble_trader,
)


class _FakeLLM:
    """Mock-LLM, der vordefinierte Antworten nacheinander zurückgibt.

    Simuliert LLMClient.chat() — gibt verschiedene Trader-Antworten zurück.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._call_idx = 0
        self.temperatures_seen: list[float] = []

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3, **kwargs) -> str:
        self.temperatures_seen.append(temperature)
        if self._call_idx < len(self._responses):
            resp = self._responses[self._call_idx]
            self._call_idx += 1
            return resp
        return self._responses[-1] if self._responses else ""


# Helper: JSON-String für Trader-Antwort bauen
def _trader_json(
    aktion: str = "KAUFEN",
    zielkurs: float | None = None,
    stop_loss: float | None = None,
    positionsanteil: int = 5,
) -> str:
    import json

    return json.dumps({
        "rolle": "Trader",
        "aktion": aktion,
        "zielkurs": zielkurs,
        "stop_loss": stop_loss,
        "positionsanteil": positionsanteil,
        "begründung": "Test-Begründung",
        "zeithorizont": "Mittelfristig",
    })


# Analysten-Mock mit current_price
_ANALYSTS = {
    "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Neutral", "_raw": ""},
    "technicals": {"current_price": 57.0},
}

_DEBATE = {
    "bull": {"_raw": "Bull-Argument"},
    "bear": {"_raw": "Bear-Argument"},
}


class TestEnsembleMajority:
    """Test Ensemble-Mehrheitsabstimmung."""

    def test_majority_kaufen_2_of_3(self):
        """2x KAUFEN, 1x HALTEN → aktion=KAUFEN, confidence=0.67."""
        llm = _FakeLLM([
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),
            _trader_json("HALTEN"),
            _trader_json("KAUFEN", zielkurs=62.0, stop_loss=52.0),
        ])

        result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        assert result["aktion"] == "KAUFEN"
        assert result["_ensemble"]["mehrheits_aktion"] == "KAUFEN"
        assert result["_ensemble"]["runs"] == 3
        assert result["_ensemble"]["ensemble_confidence"] == 0.67
        assert result["_ensemble"]["alle_aktionen"] == ["KAUFEN", "HALTEN", "KAUFEN"]

    def test_majority_halten(self):
        """2x HALTEN, 1x KAUFEN → aktion=HALTEN."""
        llm = _FakeLLM([
            _trader_json("HALTEN"),
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),
            _trader_json("HALTEN"),
        ])

        result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        assert result["aktion"] == "HALTEN"
        assert result["_ensemble"]["mehrheits_aktion"] == "HALTEN"
        assert result["_ensemble"]["ensemble_confidence"] == 0.67


class TestPlausibilityFix:
    """Test Plausibilitäts-Check für KAUFEN-Trades."""

    def test_implausible_zielkurs_fixed(self):
        """Zielkurs 32 bei current_price 57 → unplausibel, korrigiert."""
        llm = _FakeLLM([
            _trader_json("KAUFEN", zielkurs=32.0, stop_loss=50.0),  # unplausibel
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=52.0),  # plausibel
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=52.0),  # plausibel
        ])

        result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        # Mehrheit ist KAUFEN
        assert result["aktion"] == "KAUFEN"
        # Der erste Run (Basis) hat unplausiblen Zielkurs 32 → sollte korrigiert sein
        # Entweder aus plausiblen Runs übernommen oder None
        ziel = result.get("zielkurs")
        if ziel is not None:
            assert float(ziel) > 57.0, f"Zielkurs {ziel} sollte > current_price 57 sein"

    def test_implausible_stop_loss_fixed(self):
        """Stop-Loss über current_price bei KAUFEN → unplausibel, korrigiert."""
        llm = _FakeLLM([
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=60.0),  # stop > price → implausible
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),  # plausibel
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),  # plausibel
        ])

        result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        assert result["aktion"] == "KAUFEN"
        stop = result.get("stop_loss")
        if stop is not None:
            assert float(stop) < 57.0, f"Stop-Loss {stop} sollte < current_price 57 sein"

    def test_no_crash_without_current_price(self):
        """Ohne current_price im analysts-dict → kein Plausibilitäts-Check, kein Crash."""
        analysts_no_price = {
            "fundamental": {"stimmung": "bullish", "score": 4, "_raw": ""},
            "technical": {"stimmung": "bullish", "score": 4, "_raw": ""},
            "sentiment": {"stimmung": "neutral", "score": 3, "_raw": ""},
        }
        llm = _FakeLLM([
            _trader_json("KAUFEN", zielkurs=32.0, stop_loss=60.0),
            _trader_json("KAUFEN", zielkurs=32.0, stop_loss=60.0),
            _trader_json("KAUFEN", zielkurs=32.0, stop_loss=60.0),
        ])

        result = ensemble_trader(analysts_no_price, _DEBATE, llm, runs=3)

        # Sollte nicht crashen, aktion sollte KAUFEN sein
        assert result["aktion"] == "KAUFEN"
        assert result["_ensemble"]["mehrheits_aktion"] == "KAUFEN"


class TestSingleFallback:
    """Test Single-Fallback: nur 1 erfolgreicher Run."""

    def test_single_run_taken(self):
        """Wenn nur 1 Run erfolgreich → Ergebnis wird übernommen, confidence=1.0."""
        # 2 Runs schlagen fehl (leerer String → parse_json gibt {_raw: ""} zurück,
        # aber wir patchen trader() um Exception zu werfen für 2 von 3)
        llm = _FakeLLM([_trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0)])

        # Wir patchen trader so, dass es beim 2. und 3. Aufruf eine Exception wirft
        call_count = [0]
        original_trader = None

        import concilium.agents as agents_mod

        original_trader = agents_mod.trader

        def patched_trader(analysts, debate_result, llm_arg, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return original_trader(analysts, debate_result, llm_arg, **kwargs)
            raise RuntimeError("LLM-Aussetzer")

        with patch.object(agents_mod, "trader", patched_trader):
            result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        assert result["aktion"] == "KAUFEN"
        assert result["_ensemble"]["runs"] == 1
        assert result["_ensemble"]["ensemble_confidence"] == 1.0

    def test_all_runs_fail(self):
        """Wenn alle Runs fehlschlagen → HALTEN als Fallback, kein Crash."""
        import concilium.agents as agents_mod

        with patch.object(agents_mod, "trader", side_effect=RuntimeError("LLM down")):
            result = ensemble_trader(_ANALYSTS, _DEBATE, MagicMock(), runs=3)

        assert result["aktion"] == "HALTEN"
        assert result["_ensemble"]["ensemble_confidence"] == 0.0
        assert result["_ensemble"]["alle_aktionen"] == []


class TestPlausibilityHelpers:
    """Unit-Tests für die Plausibilitäts-Helper-Funktionen."""

    def test_is_plausible_kauf_ok(self):
        """KAUFEN mit ziel > price und stop < price → plausibel."""
        trade = {"aktion": "KAUFEN", "zielkurs": 65.0, "stop_loss": 50.0}
        assert _is_plausible_kauf(trade, current_price=57.0) is True

    def test_is_plausible_kauf_ziel_too_low(self):
        """KAUFEN mit ziel < price → unplausibel."""
        trade = {"aktion": "KAUFEN", "zielkurs": 32.0, "stop_loss": 50.0}
        assert _is_plausible_kauf(trade, current_price=57.0) is False

    def test_is_plausible_kauf_stop_too_high(self):
        """KAUFEN mit stop > price → unplausibel."""
        trade = {"aktion": "KAUFEN", "zielkurs": 65.0, "stop_loss": 60.0}
        assert _is_plausible_kauf(trade, current_price=57.0) is False

    def test_is_plausible_kauf_no_price(self):
        """Ohne current_price → immer plausibel."""
        trade = {"aktion": "KAUFEN", "zielkurs": 32.0, "stop_loss": 60.0}
        assert _is_plausible_kauf(trade, current_price=None) is True

    def test_fix_implausible_sets_ziel_none(self):
        """Fix setzt unplausiblen zielkurs auf None."""
        trade = {"aktion": "KAUFEN", "zielkurs": 32.0, "stop_loss": 50.0}
        fixed = _fix_implausible_trade(trade, current_price=57.0)
        assert fixed["zielkurs"] is None
        assert fixed["stop_loss"] == 50.0  # bleibt, da plausibel

    def test_fix_implausible_sets_stop_none(self):
        """Fix setzt unplausiblen stop_loss auf None."""
        trade = {"aktion": "KAUFEN", "zielkurs": 65.0, "stop_loss": 60.0}
        fixed = _fix_implausible_trade(trade, current_price=57.0)
        assert fixed["stop_loss"] is None
        assert fixed["zielkurs"] == 65.0  # bleibt, da plausibel

    def test_fix_does_not_touch_halten(self):
        """Bei HALTEN wird nichts geändert."""
        trade = {"aktion": "HALTEN", "zielkurs": 32.0, "stop_loss": 60.0}
        fixed = _fix_implausible_trade(trade, current_price=57.0)
        assert fixed == trade


class TestTemperaturePassThrough:
    """Test: ensemble_trader reicht verschiedene Temperaturen an den LLM-Call durch."""

    def test_temperatures_match_default_spread(self):
        """Bei runs=3 müssen die Temperaturen [0.3, 0.5, 0.7] sein."""
        # Je nach Temperatur unterschiedliche Aktion → Mehrheit KAUFEN
        llm = _FakeLLM([
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),   # temp=0.3
            _trader_json("HALTEN"),                                    # temp=0.5
            _trader_json("KAUFEN", zielkurs=62.0, stop_loss=52.0),   # temp=0.7
        ])

        result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        # Temperaturen wurden korrekt durchgereicht
        assert llm.temperatures_seen == [0.3, 0.5, 0.7]

        # Mehrheitsabstimmung funktioniert (2x KAUFEN, 1x HALTEN)
        assert result["aktion"] == "KAUFEN"
        assert result["_ensemble"]["mehrheits_aktion"] == "KAUFEN"
        assert result["_ensemble"]["alle_aktionen"] == ["KAUFEN", "HALTEN", "KAUFEN"]

    def test_temperature_varies_per_run_not_constant(self):
        """Stellt sicher, dass NICHT alle Runs mit 0.3 laufen (der Bug)."""
        llm = _FakeLLM([
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),
        ])

        ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        # Vor dem Fix waren alle Temperaturen 0.3 — jetzt müssen sie variieren
        assert len(set(llm.temperatures_seen)) > 1, (
            f"Temperaturen sollten variieren, war aber: {llm.temperatures_seen}"
        )

    def test_custom_temperature_range(self):
        """Eigener Temperatur-Range wird korrekt durchgereicht."""
        custom = [0.1, 0.9]
        llm = _FakeLLM([
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),
            _trader_json("HALTEN"),
        ])

        ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=2, temperature_range=custom)

        assert llm.temperatures_seen == [0.1, 0.9]
