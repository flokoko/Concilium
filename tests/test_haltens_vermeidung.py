"""Tests für Phase 2: HALTEN-Default entgegenwirken (Richtungs-Zwang bei klaren Signalen).

Abgedeckt:
(a) _force_direction_on_clear_signal: KAUFEN-Zwang (bullischer Score + Netto-Bull),
    VERKAUFEN-Zwang (bearischer Score + Netto-Bear), kein Zwang bei fehlenden
    Werten, kein Zwang bei KAUFEN/VERKAUFEN (nur HALTEN), kein Zwang bei
    gemischten Signalen, Robustheit (leere dicts/None → kein Crash).
(b) trader(): wendet den Zwang an (LLM liefert HALTEN, klare Bull-Signale →
    KAUFEN mit gesetztem zielkurs/stop_loss via _ensure_ziel_stop).
(c) trader_exit(): wendet den Zwang NICHT an (HALTEN bleibt trotz klarem Signal —
    Bestandsposition, Exit-Pfad).
(d) SYSTEM_TRADER-Prompt: HALTEN-Default-Absatz vorhanden (Exit-Prompt unverändert).

Die Tests benötigen KEIN Netzwerk — der LLMClient wird gemockt.
Der _FakeLLM ist thread-sicher und temperatur-keyed (Muster aus
tests/test_ensemble.py), damit ensemble_trader/analyst_team parallel laufen
können ohne deterministische Tests zu brechen.
"""

from __future__ import annotations

import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import (  # noqa: E402
    SYSTEM_TRADER,
    SYSTEM_TRADER_EXIT,
    _force_direction_on_clear_signal,
    trader,
    trader_exit,
)


# Autouse-Fixture: isoliert alle Tests von einer evtl. vorhandenen
# state/calibration.json (deterministisches Verhalten, wie in test_ensemble.py).
@pytest.fixture(autouse=True)
def _isolate_state_dir(monkeypatch):
    monkeypatch.setenv("CONCILIUM_STATE_DIR", "/nonexistent/test_halten_no_cal")


class _FakeLLM:
    """Thread-sicherer Mock-LLM, der Antworten nach Temperatur dispatcht.

    Muster aus tests/test_ensemble.py: temp=0.3 → responses[0], temp=0.5 →
    responses[1], temp=0.7 → responses[2]. Speichert zusätzlich System- und
    User-Prompts für Prompt-Assertions.
    """

    _DEFAULT_TEMP_KEYS = [0.3, 0.5, 0.7]

    def __init__(self, responses: list[str], temp_keys: list[float] | None = None):
        keys = temp_keys if temp_keys is not None else self._DEFAULT_TEMP_KEYS
        self._temp_map: dict[float, str] = {}
        for i, resp in enumerate(responses):
            k = round(keys[i % len(keys)], 2)
            self._temp_map[k] = resp
        self.temperatures_seen: list[float] = []
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []
        self._lock = threading.Lock()

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3, **kwargs) -> str | object:
        with self._lock:
            self.temperatures_seen.append(temperature)
            self.system_prompts.append(messages[0]["content"])
            self.user_prompts.append(messages[1]["content"])
        key = round(temperature, 2)
        if key in self._temp_map:
            text = self._temp_map[key]
        else:
            text = list(self._temp_map.values())[0] if self._temp_map else ""
        # Strukturierter Pfad: StructuredChatResult zurückgeben
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult
            return StructuredChatResult(text=text, response_format_used=True)
        return text


# Helper: JSON-String für Trader-Antwort bauen
def _trader_json(
    aktion: str = "HALTEN",
    zielkurs: float | None = None,
    stop_loss: float | None = None,
    positionsanteil: int = 5,
) -> str:
    return json.dumps({
        "rolle": "Trader",
        "aktion": aktion,
        "zielkurs": zielkurs,
        "stop_loss": stop_loss,
        "einstiegs_level": None,
        "positionsanteil": positionsanteil,
        "begründung": "Test-Begründung",
        "zeithorizont": "Mittelfristig",
    })


# Helper: Analysten-Dict (fundamental-Score einstellbar, wie test_ensemble.py)
def _analysts(f_score: int = 4, f_stimmung: str = "bullish", current_price: float = 57.0) -> dict:
    return {
        "fundamental": {"stimmung": f_stimmung, "score": f_score, "zusammenfassung": "Gut", "_raw": ""},
        "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
        "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Neutral", "_raw": ""},
        "technicals": {"current_price": current_price},
    }


# Helper: Debatte mit direkten Konfidenz-Feldern (strukturierter Pfad)
def _debate(bull_conf: int, bear_conf: int) -> dict:
    return {
        "bull": {"_raw": "Bull-Argument"},
        "bear": {"_raw": "Bear-Argument"},
        "bull_confidence": bull_conf,
        "bear_confidence": bear_conf,
    }


# ---------------------------------------------------------------------------
# (a) _force_direction_on_clear_signal — deterministischer Richtungs-Zwang
# ---------------------------------------------------------------------------


class TestKaufenZwang:
    """KAUFEN-Zwang: bullischer Fundamental-Score + Netto-Bull-Debatte."""

    def test_kaufen_zwang_bei_score_4_und_netto_bull(self):
        """HALTEN + Score 4 + Bull 4 > Bear 2 → aktion=KAUFEN, rating=KAUFEN."""
        trade = {"aktion": "HALTEN", "rating": "HALTEN"}

        result = _force_direction_on_clear_signal(trade, _analysts(f_score=4), _debate(4, 2))

        assert result["aktion"] == "KAUFEN"
        assert result["rating"] == "KAUFEN"  # NICHT STARK KAUFEN — kein Override der Stärke
        assert result["richtung_erzwungen"] is True
        assert "Richtungs-Zwang" in result["richtung_erzwungen_grund"]

    def test_kaufen_zwang_bei_score_5(self):
        """Score 5 (bullisch) + Netto-Bull → ebenfalls KAUFEN-Zwang."""
        trade = {"aktion": "HALTEN", "rating": "HALTEN"}

        result = _force_direction_on_clear_signal(trade, _analysts(f_score=5), _debate(3, 1))

        assert result["aktion"] == "KAUFEN"
        assert result["richtung_erzwungen"] is True

    def test_kaufen_zwang_gleicht_case_sensitivity_und_whitespace_aus(self):
        """aktion 'halten' (klein/Leerzeichen) wird erkannt und erzwungen."""
        trade = {"aktion": " halten ", "rating": "HALTEN"}

        result = _force_direction_on_clear_signal(trade, _analysts(f_score=4), _debate(4, 2))

        assert result["aktion"] == "KAUFEN"
        assert result["richtung_erzwungen"] is True


class TestVerkaufenZwang:
    """VERKAUFEN-Zwang: bearischer Fundamental-Score + Netto-Bear-Debatte."""

    def test_verkaufen_zwang_bei_score_2_und_netto_bear(self):
        """HALTEN + Score 2 + Bear 4 > Bull 2 → aktion=VERKAUFEN, rating=VERKAUFEN."""
        trade = {"aktion": "HALTEN", "rating": "HALTEN"}

        result = _force_direction_on_clear_signal(trade, _analysts(f_score=2, f_stimmung="bearish"), _debate(2, 4))

        assert result["aktion"] == "VERKAUFEN"
        assert result["rating"] == "VERKAUFEN"  # NICHT STARK VERKAUFEN
        assert result["richtung_erzwungen"] is True
        assert "Richtungs-Zwang" in result["richtung_erzwungen_grund"]

    def test_verkaufen_zwang_bei_score_1(self):
        """Score 1 (bearisch) + Netto-Bear → ebenfalls VERKAUFEN-Zwang."""
        trade = {"aktion": "HALTEN", "rating": "HALTEN"}

        result = _force_direction_on_clear_signal(trade, _analysts(f_score=1, f_stimmung="bearish"), _debate(1, 5))

        assert result["aktion"] == "VERKAUFEN"
        assert result["richtung_erzwungen"] is True


class TestKeinZwang:
    """Kein Zwang: fehlende Werte, gemischte Signale, andere Aktionen."""

    def test_kein_zwang_bei_fehlendem_fundamental_score(self):
        """fundamental.score fehlt → HALTEN bleibt (kein Zwang)."""
        trade = {"aktion": "HALTEN", "rating": "HALTEN"}
        analysts = _analysts()
        del analysts["fundamental"]["score"]

        result = _force_direction_on_clear_signal(trade, analysts, _debate(4, 2))

        assert result["aktion"] == "HALTEN"
        assert "richtung_erzwungen" not in result

    def test_kein_zwang_bei_none_score(self):
        """fundamental.score = None → HALTEN bleibt."""
        trade = {"aktion": "HALTEN", "rating": "HALTEN"}
        analysts = _analysts()
        analysts["fundamental"]["score"] = None

        result = _force_direction_on_clear_signal(trade, analysts, _debate(4, 2))

        assert result["aktion"] == "HALTEN"
        assert "richtung_erzwungen" not in result

    def test_kein_zwang_bei_fehlender_debatten_konfidenz(self):
        """bull/bear-Konfidenz fehlen (auch nicht in _raw parsbar) → HALTEN bleibt."""
        trade = {"aktion": "HALTEN", "rating": "HALTEN"}
        debate = {"bull": {"_raw": "Bull-Argument"}, "bear": {"_raw": "Bear-Argument"}}

        result = _force_direction_on_clear_signal(trade, _analysts(f_score=4), debate)

        assert result["aktion"] == "HALTEN"
        assert "richtung_erzwungen" not in result

    def test_kein_zwang_bei_score_3_gleichgewicht(self):
        """Score 3 (neutral) → HALTEN bleibt, auch bei Netto-Bull-Debatte."""
        trade = {"aktion": "HALTEN", "rating": "HALTEN"}

        result = _force_direction_on_clear_signal(trade, _analysts(f_score=3), _debate(4, 2))

        assert result["aktion"] == "HALTEN"
        assert "richtung_erzwungen" not in result

    def test_kein_zwang_bei_gemischten_signalen_bull_score_bear_debatte(self):
        """Score 4 (bullisch) ABER Bear-Mehrheit → gemischt, HALTEN bleibt."""
        trade = {"aktion": "HALTEN", "rating": "HALTEN"}

        result = _force_direction_on_clear_signal(trade, _analysts(f_score=4), _debate(1, 4))

        assert result["aktion"] == "HALTEN"
        assert "richtung_erzwungen" not in result

    def test_kein_zwang_bei_gemischten_signalen_bear_score_bull_debatte(self):
        """Score 2 (bearisch) ABER Bull-Mehrheit → gemischt, HALTEN bleibt."""
        trade = {"aktion": "HALTEN", "rating": "HALTEN"}

        result = _force_direction_on_clear_signal(trade, _analysts(f_score=2, f_stimmung="bearish"), _debate(4, 2))

        assert result["aktion"] == "HALTEN"
        assert "richtung_erzwungen" not in result

    def test_kauf_trades_werden_nie_ueberschrieben(self):
        """KAUFEN wird auch bei klarem Bear-Signal nicht überschrieben (nur HALTEN)."""
        trade = {"aktion": "KAUFEN", "rating": "KAUFEN"}

        result = _force_direction_on_clear_signal(trade, _analysts(f_score=2, f_stimmung="bearish"), _debate(1, 5))

        assert result["aktion"] == "KAUFEN"
        assert result["rating"] == "KAUFEN"
        assert "richtung_erzwungen" not in result

    def test_verkauf_trades_werden_nie_ueberschrieben(self):
        """VERKAUFEN wird auch bei klarem Bull-Signal nicht überschrieben."""
        trade = {"aktion": "VERKAUFEN", "rating": "VERKAUFEN"}

        result = _force_direction_on_clear_signal(trade, _analysts(f_score=4), _debate(4, 2))

        assert result["aktion"] == "VERKAUFEN"
        assert result["rating"] == "VERKAUFEN"
        assert "richtung_erzwungen" not in result

    def test_robust_leere_dicts_und_none_werte(self):
        """Leere dicts / None-Werte → kein Crash, HALTEN bleibt unverändert."""
        result = _force_direction_on_clear_signal({"aktion": "HALTEN"}, {}, {})
        assert result["aktion"] == "HALTEN"
        assert "richtung_erzwungen" not in result

        result = _force_direction_on_clear_signal({"aktion": "HALTEN"}, {"fundamental": None}, {"bull": None})
        assert result["aktion"] == "HALTEN"

        result = _force_direction_on_clear_signal(
            {"aktion": "HALTEN"}, {"fundamental": {"score": "kein-zahl"}}, {"bull": {}, "bear": {}}
        )
        assert result["aktion"] == "HALTEN"

    def test_robust_string_konfidenzen_werden_konvertiert(self):
        """Konfidenzen als Strings ('4') werden tolerant konvertiert (Zwang greift)."""
        trade = {"aktion": "HALTEN", "rating": "HALTEN"}
        debate = {"bull_confidence": "4", "bear_confidence": "2"}

        result = _force_direction_on_clear_signal(trade, _analysts(f_score=4), debate)

        assert result["aktion"] == "KAUFEN"
        assert result["richtung_erzwungen"] is True


# ---------------------------------------------------------------------------
# (b) trader() — Integration: Zwang + _ensure_ziel_stop in korrekter Reihenfolge
# ---------------------------------------------------------------------------


class TestTraderZwangIntegration:
    """trader() wendet den Zwang im Neukauf-Pfad an."""

    def test_trader_erzwingt_kaufen_mit_ziel_und_stop(self):
        """LLM sagt HALTEN, klare Bull-Signale → KAUFEN mit zielkurs/stop_loss gesetzt."""
        llm = _FakeLLM([_trader_json("HALTEN")])

        result = trader(_analysts(f_score=4), _debate(4, 2), llm)

        assert result["aktion"] == "KAUFEN"
        assert result["rating"] == "KAUFEN"  # Rating = KAUFEN (nicht STARK — kein Override der Stärke)
        assert result["richtung_erzwungen"] is True
        assert result["richtung_erzwungen_grund"]
        # _ensure_ziel_stop lief NACH dem Zwang → Ziel/Stop konsistent zur Aktion
        ziel = result.get("zielkurs")
        stop = result.get("stop_loss")
        assert ziel is not None and float(ziel) > 57.0
        assert stop is not None and float(stop) < 57.0

    def test_trader_halten_bleibt_bei_gemischten_signalen(self):
        """Gemischte Signale (bullischer Score, Bear-Debatte) → HALTEN bleibt."""
        llm = _FakeLLM([_trader_json("HALTEN")])

        result = trader(_analysts(f_score=4), _debate(1, 4), llm)

        assert result["aktion"] == "HALTEN"
        assert "richtung_erzwungen" not in result

    def test_trader_ohne_current_price_zwang_trotzdem_ohne_crash(self):
        """Kein current_price → Zwang greift, _ensure_ziel_stop-Fallback übersprungen."""
        llm = _FakeLLM([_trader_json("HALTEN")])
        analysts = _analysts(f_score=4)
        analysts["technicals"] = {}

        result = trader(analysts, _debate(4, 2), llm)

        assert result["aktion"] == "KAUFEN"
        assert result["richtung_erzwungen"] is True
        assert result.get("zielkurs") is None  # kein Fallback ohne current_price
        assert result.get("stop_loss") is None

    def test_trader_nimmt_system_trader_prompt(self):
        """Der Zwang läuft im Neukauf-Pfad mit dem SYSTEM_TRADER-Prompt."""
        llm = _FakeLLM([_trader_json("HALTEN")])

        trader(_analysts(f_score=4), _debate(4, 2), llm)

        assert llm.system_prompts[0] == SYSTEM_TRADER


# ---------------------------------------------------------------------------
# (c) trader_exit() — Exit-Pfad: Zwang wird NICHT angewendet
# ---------------------------------------------------------------------------


class TestTraderExitKeinZwang:
    """trader_exit wendet den Richtungs-Zwang NICHT an (HALTEN bleibt legitim)."""

    def test_exit_halten_bleibt_trotz_klarem_bull_signal(self):
        """HALTEN + klare Bull-Signale → im Exit-Pfad bleibt HALTEN, keine Metadaten."""
        llm = _FakeLLM([_trader_json("HALTEN")])

        result = trader_exit(_analysts(f_score=4), _debate(4, 2), llm)

        assert result["aktion"] == "HALTEN"
        assert result["rating"] == "HALTEN"
        assert "richtung_erzwungen" not in result
        # HALTEN: _ensure_ziel_stop erzwingt keine Werte
        assert result.get("zielkurs") is None
        assert result.get("stop_loss") is None

    def test_exit_halten_bleibt_trotz_klarem_bear_signal(self):
        """HALTEN + klare Bear-Signale → im Exit-Pfad bleibt ebenfalls HALTEN."""
        llm = _FakeLLM([_trader_json("HALTEN")])

        result = trader_exit(_analysts(f_score=2, f_stimmung="bearish"), _debate(1, 5), llm)

        assert result["aktion"] == "HALTEN"
        assert "richtung_erzwungen" not in result

    def test_exit_nutzt_exit_prompt(self):
        """Exit-Pfad nutzt weiterhin SYSTEM_TRADER_EXIT (nicht SYSTEM_TRADER)."""
        llm = _FakeLLM([_trader_json("HALTEN")])

        trader_exit(_analysts(f_score=4), _debate(4, 2), llm)

        assert llm.system_prompts[0] == SYSTEM_TRADER_EXIT


# ---------------------------------------------------------------------------
# (d) Prompt: SYSTEM_TRADER erzwingt begründetes HALTEN
# ---------------------------------------------------------------------------


class TestSystemTraderPrompt:
    """SYSTEM_TRADER enthält den HALTEN-Default-Absatz (Exit-Prompt unverändert)."""

    def test_system_trader_haelt_kein_default_absatz(self):
        assert "HALTEN ist KEIN Default-Ausweg" in SYSTEM_TRADER
        assert "warum weder gekauft noch verkauft werden sollte" in SYSTEM_TRADER
        assert "wage Halten ist die schlechteste Entscheidung" in SYSTEM_TRADER

    def test_exit_prompt_hat_eigenen_halten_absatz(self):
        """Exit-Prompt hat den Phase-1-Absatz (aufstocken/verkaufen) — bleibt unverändert."""
        assert "HALTEN ist KEIN Default-Ausweg" in SYSTEM_TRADER_EXIT
        assert "warum weder aufgestockt noch verkauft werden sollte" in SYSTEM_TRADER_EXIT

    def test_prompts_unterschiedlich(self):
        """Neukauf- und Exit-Prompt sind verschiedene Strings (Phase-1-Garantie)."""
        assert SYSTEM_TRADER != SYSTEM_TRADER_EXIT
