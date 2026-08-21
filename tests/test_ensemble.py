"""Tests für ensemble_trader — Mehrheitsabstimmung, Plausibilitäts-Check, Single-Fallback.

Diese Tests benötigen KEIN Netzwerk — der LLMClient wird gemockt.
Der _FakeLLM ist thread-sicher und temperatur-keyed, damit die Tests
deterministisch bleiben auch wenn ensemble_trader/analyst_team parallel laufen.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import (  # noqa: E402
    _extract_current_price,
    _fix_implausible_trade,
    _is_plausible_kauf,
    analyst_team,
    ensemble_trader,
)


class _FakeLLM:
    """Thread-sicherer Mock-LLM, der Antworten nach Temperatur dispatcht.

    Da ensemble_trader parallel läuft, ist ein Index-basierter Mock nicht
    deterministisch. Stattdessen wird jede Temperatur auf eine feste Antwort
    gemappt: temp=0.3 → responses[0], temp=0.5 → responses[1], temp=0.7 →
    responses[2]. Bei eigenen Temperatur-Keys kann temp_keys übergeben werden.
    """

    _DEFAULT_TEMP_KEYS = [0.3, 0.5, 0.7]

    def __init__(self, responses: list[str], temp_keys: list[float] | None = None):
        keys = temp_keys if temp_keys is not None else self._DEFAULT_TEMP_KEYS
        self._temp_map: dict[float, str] = {}
        for i, resp in enumerate(responses):
            k = round(keys[i % len(keys)], 2)
            self._temp_map[k] = resp
        self.temperatures_seen: list[float] = []
        self._lock = threading.Lock()

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3, **kwargs) -> str:
        with self._lock:
            self.temperatures_seen.append(temperature)
        key = round(temperature, 2)
        if key in self._temp_map:
            return self._temp_map[key]
        # Fallback: erste verfügbare Antwort
        return list(self._temp_map.values())[0] if self._temp_map else ""


# Helper: JSON-String für Trader-Antwort bauen
def _trader_json(
    aktion: str = "KAUFEN",
    zielkurs: float | None = None,
    stop_loss: float | None = None,
    positionsanteil: int = 5,
) -> str:
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
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),  # temp=0.3
            _trader_json("HALTEN"),                                 # temp=0.5
            _trader_json("KAUFEN", zielkurs=62.0, stop_loss=52.0),  # temp=0.7
        ])

        result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        assert result["aktion"] == "KAUFEN"
        assert result["_ensemble"]["mehrheits_aktion"] == "KAUFEN"
        assert result["_ensemble"]["runs"] == 3
        assert result["_ensemble"]["ensemble_confidence"] == 0.67
        # Ergebnisse werden in Temp-Reihenfolge gesammelt → deterministisch
        assert result["_ensemble"]["alle_aktionen"] == ["KAUFEN", "HALTEN", "KAUFEN"]

    def test_majority_halten(self):
        """2x HALTEN, 1x KAUFEN → aktion=HALTEN."""
        llm = _FakeLLM([
            _trader_json("HALTEN"),                                 # temp=0.3
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),  # temp=0.5
            _trader_json("HALTEN"),                                 # temp=0.7
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
            _trader_json("KAUFEN", zielkurs=32.0, stop_loss=50.0),  # unplausibel  temp=0.3
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=52.0),  # plausibel    temp=0.5
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=52.0),  # plausibel    temp=0.7
        ])

        result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        assert result["aktion"] == "KAUFEN"
        ziel = result.get("zielkurs")
        if ziel is not None:
            assert float(ziel) > 57.0, f"Zielkurs {ziel} sollte > current_price 57 sein"

    def test_implausible_stop_loss_fixed(self):
        """Stop-Loss über current_price bei KAUFEN → unplausibel, korrigiert."""
        llm = _FakeLLM([
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=60.0),  # stop > price → implausible  temp=0.3
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),  # plausibel                  temp=0.5
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),  # plausibel                  temp=0.7
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

        assert result["aktion"] == "KAUFEN"
        assert result["_ensemble"]["mehrheits_aktion"] == "KAUFEN"


class TestSingleFallback:
    """Test Single-Fallback: nur 1 erfolgreicher Run."""

    def test_single_run_taken(self):
        """Wenn nur 1 Run erfolgreich → Ergebnis wird übernommen, confidence=1.0."""
        llm = _FakeLLM([_trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0)])

        import concilium.agents as agents_mod

        original_trader = agents_mod.trader

        def patched_trader(analysts, debate_result, llm_arg, temperature=0.3, **kwargs):
            # Nur temp=0.3 erfolgreich, 0.5 und 0.7 schlagen fehl
            if round(float(temperature), 2) == 0.3:
                return original_trader(analysts, debate_result, llm_arg, temperature=temperature)
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
        """Bei runs=3 müssen die Temperaturen {0.3, 0.5, 0.7} sein (Reihenfolge beliebig durch Parallelität)."""
        llm = _FakeLLM([
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),   # temp=0.3
            _trader_json("HALTEN"),                                    # temp=0.5
            _trader_json("KAUFEN", zielkurs=62.0, stop_loss=52.0),   # temp=0.7
        ])

        result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        # Temperaturen wurden korrekt durchgereicht (Reihenfolge durch Parallelität variierend)
        assert sorted(llm.temperatures_seen) == [0.3, 0.5, 0.7]

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
        ], temp_keys=custom)

        ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=2, temperature_range=custom)

        assert sorted(llm.temperatures_seen) == [0.1, 0.9]


# ---------------------------------------------------------------------------
# Neue Tests: Parallelität + current_price aus technicals
# ---------------------------------------------------------------------------


class TestAnalystTeamParallel:
    """Tests für analyst_team: Parallelität, Fehlerresilienz, technicals-Durchreichung."""

    def test_returns_dict_with_technicals(self):
        """analyst_team liefert fundamental/technical/sentiment + technicals."""
        llm = _FakeLLM([
            json.dumps({"rolle": "Fundamental-Analyst", "stimmung": "bullish", "score": 4, "zusammenfassung": "Gut"}),
            json.dumps({"rolle": "Technik-Analyst", "stimmung": "neutral", "score": 3, "zusammenfassung": "Ok"}),
            json.dumps({"rolle": "Sentiment-Analyst", "stimmung": "bullish", "score": 4, "zusammenfassung": "Positiv"}),
        ])

        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {"current_price": 123.45, "rsi14": 55.0},
            "sentiment": {},
        }
        result = analyst_team(data, llm)

        assert "fundamental" in result
        assert "technical" in result
        assert "sentiment" in result
        assert "technicals" in result
        assert result["technicals"] == {"current_price": 123.45, "rsi14": 55.0}

    def test_analysts_run_in_parallel(self):
        """Verifiziert, dass die 3 Analysten-Calls gleichzeitig starten.

        Nutzt eine threading.Barrier(3): jeder Analysten-Call betritt die
        Barrier. Wenn alle 3 Calls gestartet sind (bevor einer returned),
        wird die Barrier freigegeben. Dies ist die zuverlässigste Verifikation
        echter Parallelität — bei sequentieller Ausführung würde die Barrier
        timeouten.
        """
        barrier = threading.Barrier(3, timeout=5.0)
        call_threads: list[int] = []
        lock = threading.Lock()

        class _BarrierLLM:
            def chat(self, messages, temperature=0.3, **kwargs):
                tid = threading.get_ident()
                with lock:
                    call_threads.append(tid)
                # Warten bis alle 3 Threads hier sind — nur bei echter
                # Parallelität kommt die Barrier jemals frei
                barrier.wait()
                # Jeder Thread gibt eine andere Antwort
                role = messages[0]["content"]
                if "Fundamental" in role:
                    return json.dumps({"rolle": "Fundamental-Analyst", "stimmung": "bullish", "score": 4, "zusammenfassung": "Gut"})
                if "technisch" in role:
                    return json.dumps({"rolle": "Technik-Analyst", "stimmung": "neutral", "score": 3, "zusammenfassung": "Ok"})
                return json.dumps({"rolle": "Sentiment-Analyst", "stimmung": "bullish", "score": 4, "zusammenfassung": "Positiv"})

        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {"current_price": 50.0},
            "sentiment": {},
        }
        result = analyst_team(data, _BarrierLLM())

        # Wenn wir hier ankommen, haben alle 3 Threads die Barrier erreicht
        # → sie liefen parallel.
        assert len(call_threads) == 3, f"3 Threads sollten starten, war: {call_threads}"
        assert len(set(call_threads)) == 3, "3 verschiedene Thread-IDs"
        assert result["fundamental"]["stimmung"] == "bullish"
        assert result["technical"]["stimmung"] == "neutral"
        assert result["sentiment"]["stimmung"] == "bullish"
        assert result["technicals"]["current_price"] == 50.0

    def test_partial_failure_does_not_crash(self):
        """Ein Analysten-Call wirft → Fehlereintrag, andere Calls normal, kein Crash."""
        class _PartialFailLLM:
            def __init__(self):
                self._call_count = 0
                self._lock = threading.Lock()

            def chat(self, messages, temperature=0.3, **kwargs):
                with self._lock:
                    self._call_count += 1
                role = messages[0]["content"]
                if "Sentiment" in role:
                    raise RuntimeError("Sentiment-API down")
                if "Fundamental" in role:
                    return json.dumps({"rolle": "Fundamental-Analyst", "stimmung": "bullish", "score": 4, "zusammenfassung": "Gut"})
                return json.dumps({"rolle": "Technik-Analyst", "stimmung": "neutral", "score": 3, "zusammenfassung": "Ok"})

        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {"current_price": 50.0},
            "sentiment": {},
        }
        result = analyst_team(data, _PartialFailLLM())

        # fundamental und technical sind normal
        assert result["fundamental"]["stimmung"] == "bullish"
        assert result["technical"]["stimmung"] == "neutral"
        # sentiment hat Fehler-Eintrag, nicht gecrasht
        assert "fehler" in result["sentiment"]
        assert "Sentiment-API down" in result["sentiment"]["fehler"]
        # technicals trotzdem durchgereicht
        assert result["technicals"]["current_price"] == 50.0


class TestExtractCurrentPrice:
    """Tests für _extract_current_price: technicals primär, kein Regex mehr."""

    def test_price_from_technicals(self):
        """Aktueller Kurs wird aus analysts['technicals']['current_price'] extrahiert."""
        analysts = {
            "fundamental": {"_raw": "Aktueller Kurs: 999.99"},  # würde früher Regex matchen
            "technical": {"_raw": "Aktueller Kurs: 999.99"},
            "sentiment": {"_raw": ""},
            "technicals": {"current_price": 123.45},
        }
        price = _extract_current_price(analysts)
        assert price == 123.45

    def test_no_regex_from_raw_text(self):
        """Roher LLM-Text mit 'Aktueller Kurs: 999.99' wird NICHT mehr geparsed."""
        analysts = {
            "fundamental": {"_raw": "Aktueller Kurs: 999.99"},
            "technical": {"_raw": "Aktueller Kurs: 999.99"},
            "sentiment": {"_raw": "Aktueller Kurs: 999.99"},
            # KEIN technicals-Key!
        }
        price = _extract_current_price(analysts)
        # Fallback sucht in Subdicts nach 'current_price'-Feld, nicht nach Regex
        # → None, da kein current_price-Feld in den Subdicts
        assert price is None

    def test_fallback_to_subdict_current_price(self):
        """Fallback: current_price-Feld in Analysten-Subdicts wird gefunden."""
        analysts = {
            "fundamental": {"current_price": 88.8},
            "technical": {"_raw": ""},
            "sentiment": {"_raw": ""},
        }
        price = _extract_current_price(analysts)
        assert price == 88.8

    def test_empty_analysts(self):
        """Leeres dict → None, kein Crash."""
        assert _extract_current_price({}) is None

    def test_technicals_none(self):
        """technicals=None → None, kein Crash."""
        assert _extract_current_price({"technicals": None}) is None

    def test_technicals_not_dict(self):
        """technicals ist kein dict (z.B. list) → None, kein Crash."""
        assert _extract_current_price({"technicals": [1, 2, 3]}) is None

    def test_current_price_invalid(self):
        """current_price ist ein String der nicht zu float konvertierbar ist → None."""
        analysts = {"technicals": {"current_price": "N/A"}}
        assert _extract_current_price(analysts) is None

    def test_no_llm_path_safe(self):
        """Auch ohne LLM-Pfad (nur data ohne technicals) crasht es nicht."""
        assert _extract_current_price({"fundamental": {}}) is None
