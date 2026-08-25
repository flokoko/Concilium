"""Tests für Entscheidungs-Disziplin — aggressive Ratings dämpfen.

Testet:
- _should_dampen_stark: liest calibration.json korrekt, True bei überkonfident,
  False bei fehlend/zu alt/ungültig/unterkonfident/wenig Daten
- _dampen_stark_rating: dämpft STARK KAUFEN → KAUFEN wenn aktiv,
  lässt KAUFEN unverändert, setzt rating_gedämpft
- trader: dämpft STARK KAUFEN → KAUFEN wenn aktiv, lässt KAUFEN unverändert
- ensemble_trader: dämpft finales Rating STARK KAUFEN → KAUFEN wenn aktiv
- Report: zeigt Hinweis nur wenn rating_gedämpft=True

Alle Tests sind OFFLINE-fähig: kein yfinance, kein Netzwerk.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import (  # noqa: E402
    _dampen_stark_rating,
    _should_dampen_stark,
    ensemble_trader,
    trader,
)
from concilium.report import generate_report  # noqa: E402

# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #


def _write_calibration_json(
    tmp_path,
    *,
    hit_rates: dict[str, float | None] | None = None,
    avg_confidences: dict[str, float] | None = None,
    hit_rate_gesamt: float = 0.4,
    anzahl_entscheidungen: int = 10,
    erstellt_am: datetime | None = None,
    invalid: bool = False,
    no_erstellt_am: bool = False,
) -> str:
    """Schreibt eine calibration.json in tmp_path/state/ und gibt den Pfad zurück."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    cal_path = state_dir / "calibration.json"

    if invalid:
        cal_path.write_text("{not valid json", encoding="utf-8")
        return str(cal_path)

    nach_aktion: dict[str, dict] = {}
    hr = hit_rates or {}
    ac = avg_confidences or {}
    for action in ("KAUFEN", "HALTEN", "VERKAUFEN"):
        rate = hr.get(action)
        if rate is None:
            continue
        nach_aktion[action] = {
            "n": 5,
            "hit_rate": rate,
            "avg_confidence": ac.get(action, 0.8),
        }

    payload: dict = {
        "anzahl_entscheidungen": anzahl_entscheidungen,
        "hit_rate_gesamt": hit_rate_gesamt,
        "nach_aktion": nach_aktion,
    }
    if not no_erstellt_am:
        payload["erstellt_am"] = (erstellt_am or datetime.now()).isoformat()

    cal_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(cal_path)


class _CapturingLLM:
    """Mock-LLM, der JSON zurückgibt."""

    def __init__(self, response: str = '{"aktion": "HALTEN"}'):
        self._response = response

    def chat(self, messages, temperature: float = 0.3, **kwargs):
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult
            return StructuredChatResult(text=self._response, response_format_used=True)
        return self._response


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


class _FakeLLM:
    """Thread-sicherer Mock-LLM, temperatur-keyed."""

    _DEFAULT_TEMP_KEYS = [0.3, 0.5, 0.7]

    def __init__(self, responses: list[str], temp_keys: list[float] | None = None):
        keys = temp_keys if temp_keys is not None else self._DEFAULT_TEMP_KEYS
        self._temp_map: dict[float, str] = {}
        for i, resp in enumerate(responses):
            k = round(keys[i % len(keys)], 2)
            self._temp_map[k] = resp
        self._lock = __import__("threading").Lock()

    def chat(self, messages, temperature: float = 0.3, **kwargs):
        key = round(temperature, 2)
        with self._lock:
            if key in self._temp_map:
                text = self._temp_map[key]
            else:
                text = list(self._temp_map.values())[0] if self._temp_map else ""
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult
            return StructuredChatResult(text=text, response_format_used=True)
        return text


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


# --------------------------------------------------------------------------- #
# Tests: _should_dampen_stark
# --------------------------------------------------------------------------- #


class TestShouldDampenStark:
    """Test _should_dampen_stark: True bei überkonfident, False sonst."""

    def test_true_bei_ueberkonfident_gesamt(self, tmp_path):
        """Gesamt-Gap > 0.15 → True."""
        # avg_confidence=0.8, hit_rate_gesamt=0.4 → gap=0.4 > 0.15
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.4, "HALTEN": 0.4},
            avg_confidences={"KAUFEN": 0.8, "HALTEN": 0.8},
            hit_rate_gesamt=0.4,
            anzahl_entscheidungen=10,
        )
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _should_dampen_stark() is True

    def test_true_bei_ueberkonfident_per_action(self, tmp_path):
        """Per-Action Gap > 0.15 für KAUFEN → True (Gesamt gut kalibriert)."""
        # Gesamt: KAUFEN avg_conf=0.9 hr=0.4, HALTEN avg_conf=0.5 hr=0.5
        # gewichtet: (0.9*5 + 0.5*5) / 10 = 0.7; hit_rate_gesamt=0.45 → gap=0.25 > 0.15
        # Um per-action isoliert zu testen: Gesamt gut kalibriert
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.4, "HALTEN": 0.7},
            avg_confidences={"KAUFEN": 0.9, "HALTEN": 0.5},
            hit_rate_gesamt=0.7,
            anzahl_entscheidungen=10,
        )
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            # KAUFEN: gap = 0.9 - 0.4 = 0.5 > 0.15 → True
            assert _should_dampen_stark("KAUFEN") is True

    def test_false_bei_gut_kalibriert(self, tmp_path):
        """Gap innerhalb ±0.15 → False."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.5, "HALTEN": 0.5},
            avg_confidences={"KAUFEN": 0.5, "HALTEN": 0.5},
            hit_rate_gesamt=0.5,
            anzahl_entscheidungen=10,
        )
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _should_dampen_stark() is False
            assert _should_dampen_stark("KAUFEN") is False

    def test_false_bei_unterkonfident(self, tmp_path):
        """Gap < -0.15 → False."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.9, "HALTEN": 0.9},
            avg_confidences={"KAUFEN": 0.3, "HALTEN": 0.3},
            hit_rate_gesamt=0.9,
            anzahl_entscheidungen=10,
        )
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _should_dampen_stark() is False

    def test_false_bei_fehlender_datei(self, tmp_path):
        """Keine Datei → False."""
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _should_dampen_stark() is False

    def test_false_bei_ungueltigem_json(self, tmp_path):
        """Kaputtes JSON → False."""
        _write_calibration_json(tmp_path, invalid=True)
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _should_dampen_stark() is False

    def test_false_bei_zu_alter_datei(self, tmp_path):
        """Datei älter als 7 Tage → False."""
        old_date = datetime.now() - timedelta(days=10)
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.4},
            avg_confidences={"KAUFEN": 0.8},
            hit_rate_gesamt=0.4,
            erstellt_am=old_date,
        )
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _should_dampen_stark() is False

    def test_false_bei_kein_erstellt_am(self, tmp_path):
        """Kein erstellt_am → False."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.4},
            avg_confidences={"KAUFEN": 0.8},
            hit_rate_gesamt=0.4,
            no_erstellt_am=True,
        )
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _should_dampen_stark() is False

    def test_false_bei_zu_wenig_entscheidungen(self, tmp_path):
        """anzahl_entscheidungen < 5 → False."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.4},
            avg_confidences={"KAUFEN": 0.8},
            hit_rate_gesamt=0.4,
            anzahl_entscheidungen=3,
        )
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _should_dampen_stark() is False

    def test_false_bei_garbage(self, tmp_path):
        """Beliebiger Müll → False."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        cal_path = state_dir / "calibration.json"
        cal_path.write_text("garbage", encoding="utf-8")
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _should_dampen_stark() is False

    def test_per_action_gut_kalibriert_trotz_gesamt_ueberkonfident(self, tmp_path):
        """Gesamt überkonfident → True auch wenn per-action gut."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.8, "HALTEN": 0.8},
            avg_confidences={"KAUFEN": 0.8, "HALTEN": 0.8},
            hit_rate_gesamt=0.4,
            anzahl_entscheidungen=10,
        )
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            # KAUFEN per-action gap = 0.0, aber Gesamt gap = 0.8-0.4=0.4 > 0.15 → True
            assert _should_dampen_stark("KAUFEN") is True


# --------------------------------------------------------------------------- #
# Tests: _dampen_stark_rating
# --------------------------------------------------------------------------- #


class TestDampenStarkRating:
    """Test _dampen_stark_rating: dämpft STARK → normal, setzt rating_gedämpft."""

    def test_stark_kaufen_gedämpft_wenn_ueberkonfident(self, tmp_path):
        """STARK KAUFEN → KAUFEN wenn überkonfident, rating_gedämpft=True."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.4},
            avg_confidences={"KAUFEN": 0.8},
            hit_rate_gesamt=0.4,
            anzahl_entscheidungen=10,
        )
        result: dict = {"rating": "STARK KAUFEN", "aktion": "KAUFEN"}
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            _dampen_stark_rating(result, "STARK KAUFEN")
        assert result["rating"] == "KAUFEN"
        assert result["aktion"] == "KAUFEN"
        assert result["rating_gedämpft"] is True
        assert result["rating_original"] == "STARK KAUFEN"

    def test_stark_verkaufen_gedämpft_wenn_ueberkonfident(self, tmp_path):
        """STARK VERKAUFEN → VERKAUFEN wenn überkonfident."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"VERKAUFEN": 0.3},
            avg_confidences={"VERKAUFEN": 0.8},
            hit_rate_gesamt=0.3,
            anzahl_entscheidungen=10,
        )
        result: dict = {"rating": "STARK VERKAUFEN", "aktion": "VERKAUFEN"}
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            _dampen_stark_rating(result, "STARK VERKAUFEN")
        assert result["rating"] == "VERKAUFEN"
        assert result["aktion"] == "VERKAUFEN"
        assert result["rating_gedämpft"] is True

    def test_kaufen_nicht_gedämpft(self, tmp_path):
        """KAUFEN bleibt unverändert, rating_gedämpft=False."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.4},
            avg_confidences={"KAUFEN": 0.8},
            hit_rate_gesamt=0.4,
            anzahl_entscheidungen=10,
        )
        result: dict = {"rating": "KAUFEN", "aktion": "KAUFEN"}
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            _dampen_stark_rating(result, "KAUFEN")
        assert result["rating"] == "KAUFEN"
        assert result["rating_gedämpft"] is False

    def test_stark_kaufen_nicht_gedämpft_wenn_gut_kalibriert(self, tmp_path):
        """STARK KAUFEN bleibt STARK KAUFEN wenn gut kalibriert."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.8},
            avg_confidences={"KAUFEN": 0.8},
            hit_rate_gesamt=0.8,
            anzahl_entscheidungen=10,
        )
        result: dict = {"rating": "STARK KAUFEN", "aktion": "KAUFEN"}
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            _dampen_stark_rating(result, "STARK KAUFEN")
        assert result["rating"] == "STARK KAUFEN"
        assert result["rating_gedämpft"] is False

    def test_stark_kaufen_nicht_gedämpft_ohne_datei(self, tmp_path):
        """Ohne calibration.json → kein Dämpfen."""
        result: dict = {"rating": "STARK KAUFEN", "aktion": "KAUFEN"}
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            _dampen_stark_rating(result, "STARK KAUFEN")
        assert result["rating"] == "STARK KAUFEN"
        assert result["rating_gedämpft"] is False


# --------------------------------------------------------------------------- #
# Tests: trader() Dämpfung
# --------------------------------------------------------------------------- #


class TestTraderDaempfung:
    """Test dass trader() STARK KAUFEN dämpft wenn überkonfident."""

    def _make_analysts(self):
        return {
            "fundamental": {"stimmung": "bullish", "score": 4, "_raw": "Gut"},
            "technical": {"stimmung": "bullish", "score": 4, "_raw": "Gut"},
            "sentiment": {"stimmung": "neutral", "score": 3, "_raw": "Neutral"},
            "technicals": {"current_price": 100.0},
        }

    def test_trader_daempft_stark_kaufen(self, tmp_path):
        """trader() mit STARK KAUFEN → KAUFEN wenn überkonfident."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.4},
            avg_confidences={"KAUFEN": 0.8},
            hit_rate_gesamt=0.4,
            anzahl_entscheidungen=10,
        )
        llm = _CapturingLLM(json.dumps({
            "rolle": "Trader",
            "aktion": "STARK KAUFEN",
            "zielkurs": 120,
            "stop_loss": 90,
            "positionsanteil": 8,
            "begründung": "Sehr bullish",
            "zeithorizont": "Mittelfristig",
        }))
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            result = trader(self._make_analysts(), {"bull": {"_raw": "B"}, "bear": {"_raw": "S"}}, llm)
        assert result["aktion"] == "KAUFEN"
        assert result["rating"] == "KAUFEN"
        assert result["rating_gedämpft"] is True
        assert result["rating_original"] == "STARK KAUFEN"

    def test_trader_laesst_kaufen_unveraendert(self, tmp_path):
        """trader() mit KAUFEN → bleibt KAUFEN, rating_gedämpft=False."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.4},
            avg_confidences={"KAUFEN": 0.8},
            hit_rate_gesamt=0.4,
            anzahl_entscheidungen=10,
        )
        llm = _CapturingLLM(json.dumps({
            "rolle": "Trader",
            "aktion": "KAUFEN",
            "zielkurs": 120,
            "stop_loss": 90,
            "positionsanteil": 8,
            "begründung": "Bullish",
            "zeithorizont": "Mittelfristig",
        }))
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            result = trader(self._make_analysts(), {"bull": {"_raw": "B"}, "bear": {"_raw": "S"}}, llm)
        assert result["aktion"] == "KAUFEN"
        assert result["rating"] == "KAUFEN"
        assert result["rating_gedämpft"] is False

    def test_trader_kein_dämpfen_ohne_datei(self, tmp_path):
        """Ohne calibration.json → STARK KAUFEN bleibt STARK KAUFEN."""
        llm = _CapturingLLM(json.dumps({
            "rolle": "Trader",
            "aktion": "STARK KAUFEN",
            "zielkurs": 120,
            "stop_loss": 90,
            "positionsanteil": 8,
            "begründung": "Sehr bullish",
            "zeithorizont": "Mittelfristig",
        }))
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            result = trader(self._make_analysts(), {"bull": {"_raw": "B"}, "bear": {"_raw": "S"}}, llm)
        assert result["aktion"] == "KAUFEN"
        assert result["rating"] == "STARK KAUFEN"
        assert result["rating_gedämpft"] is False


# --------------------------------------------------------------------------- #
# Tests: ensemble_trader() Dämpfung des finalen Ratings
# --------------------------------------------------------------------------- #


class TestEnsembleDaempfung:
    """Test dass ensemble_trader das finale Rating dämpft wenn überkonfident."""

    def test_ensemble_daempft_stark_kaufen(self, tmp_path):
        """Ensemble mit 3x STARK KAUFEN → finales Rating KAUFEN wenn überkonfident."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.4},
            avg_confidences={"KAUFEN": 0.8},
            hit_rate_gesamt=0.4,
            anzahl_entscheidungen=10,
        )
        llm = _FakeLLM([
            _trader_json("STARK KAUFEN", zielkurs=65.0, stop_loss=50.0),
            _trader_json("STARK KAUFEN", zielkurs=62.0, stop_loss=52.0),
            _trader_json("STARK KAUFEN", zielkurs=64.0, stop_loss=51.0),
        ])
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        assert result["aktion"] == "KAUFEN"
        assert result["rating"] == "KAUFEN"
        assert result["rating_gedämpft"] is True
        assert result.get("rating_original") == "STARK KAUFEN"

    def test_ensemble_kein_dämpfen_ohne_datei(self, tmp_path):
        """Ohne calibration.json → STARK KAUFEN bleibt STARK KAUFEN."""
        llm = _FakeLLM([
            _trader_json("STARK KAUFEN", zielkurs=65.0, stop_loss=50.0),
            _trader_json("STARK KAUFEN", zielkurs=62.0, stop_loss=52.0),
            _trader_json("STARK KAUFEN", zielkurs=64.0, stop_loss=51.0),
        ])
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "nonexistent")}):
            result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        assert result["aktion"] == "KAUFEN"
        assert result["rating"] == "STARK KAUFEN"
        assert result["rating_gedämpft"] is False

    def test_ensemble_daempft_stark_verkaufen(self, tmp_path):
        """Ensemble mit 3x STARK VERKAUFEN → finales Rating VERKAUFEN."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"VERKAUFEN": 0.3},
            avg_confidences={"VERKAUFEN": 0.8},
            hit_rate_gesamt=0.3,
            anzahl_entscheidungen=10,
        )
        llm = _FakeLLM([
            _trader_json("STARK VERKAUFEN"),
            _trader_json("STARK VERKAUFEN"),
            _trader_json("STARK VERKAUFEN"),
        ])
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        assert result["aktion"] == "VERKAUFEN"
        assert result["rating"] == "VERKAUFEN"
        assert result["rating_gedämpft"] is True
        assert result.get("rating_original") == "STARK VERKAUFEN"


# --------------------------------------------------------------------------- #
# Tests: Report zeigt Dämpfungs-Hinweis
# --------------------------------------------------------------------------- #


class TestReportDaempfungsHinweis:
    """Test dass der Report den Dämpfungs-Hinweis nur zeigt wenn rating_gedämpft=True."""

    def _make_result(self, rating_gedämpft: bool, rating: str = "KAUFEN", rating_original: str | None = None) -> dict:
        return {
            "ticker": "TEST",
            "data": {
                "fundamentals": {"name": "Test AG", "sector": "Tech"},
                "technicals": {"current_price": 100.0},
            },
            "analysts": {
                "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut"},
                "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut"},
                "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok"},
            },
            "trade": {
                "aktion": "KAUFEN" if "KAUFEN" in rating else "VERKAUFEN",
                "rating": rating,
                "rating_gedämpft": rating_gedämpft,
                "rating_original": rating_original or "STARK KAUFEN",
                "zielkurs": 120.0,
                "stop_loss": 90.0,
                "positionsanteil": 5,
                "zeithorizont": "Mittelfristig",
                "begründung": "Test",
            },
            "final": {"entscheidung": "GENEHMIGT", "begründung": "Ok"},
            "risk": {"risiko_score": 3},
        }

    def test_hinweis_wenn_gedämpft(self):
        """rating_gedämpft=True → 'Rating gedämpft' im Report."""
        result = self._make_result(True, "KAUFEN", "STARK KAUFEN")
        report = generate_report(result)
        assert "Rating gedämpft" in report
        assert "STARK KAUFEN → KAUFEN" in report

    def test_kein_hinweis_ohne_dämpfung(self):
        """rating_gedämpft=False → kein 'Rating gedämpft' im Report."""
        result = self._make_result(False, "STARK KAUFEN", None)
        report = generate_report(result)
        assert "Rating gedämpft" not in report

    def test_hinweis_auch_in_zusammenfassung(self):
        """Der Dämpfungs-Hinweis erscheint auch in der Zusammenfassung (Urteil-Zeile)."""
        result = self._make_result(True, "KAUFEN", "STARK KAUFEN")
        report = generate_report(result)
        # Der Hinweis sollte mindestens einmal im Report vorkommen
        assert report.count("Rating gedämpft") >= 1

    def test_hinweis_fuer_stark_verkaufen(self):
        """Dämpfung STARK VERKAUFEN → VERKAUFEN wird im Report gezeigt."""
        result = self._make_result(True, "VERKAUFEN", "STARK VERKAUFEN")
        report = generate_report(result)
        assert "Rating gedämpft" in report
        assert "STARK VERKAUFEN → VERKAUFEN" in report
