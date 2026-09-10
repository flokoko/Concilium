"""Tests für die kalibrierungs-gestützte PM-Genehmigungsschwelle (Phase 3).

Testet:
- _load_pm_calibration: liest gültige calibration.json korrekt, None bei
  fehlend/zu alt/ungültig/ohne erstellt_am; filtert falsch getypte Werte
- _apply_pm_genehmigungsschwelle: GENEHMIGT + niedrige Konfidenz + niedrige
  Hit-Rate → MODIFIZIERT (+ Flags); hohe Konfidenz oder hohe Hit-Rate →
  bleibt; MODIFIZIERT/ABGELEHNT unverändert; fehlende Werte → kein Crash
- portfolio_manager (Integration): gemockter LLM liefert GENEHMIGT mit
  confidence 3, calibration.json mit hit_rate_gesamt 0.3 → MODIFIZIERT mit
  genehmigung_herabgestuft=True; confidence 5 → bleibt GENEHMIGT

Mock-Muster: thread-sicherer, temperatur-keyed _FakeLLM (wie
tests/test_ensemble.py). calibration.json wird in ein temporäres
CONCILIUM_STATE_DIR geschrieben (Muster: tests/test_entscheidungs_disziplin.py).

Alle Tests sind OFFLINE-fähig: kein yfinance, kein Netzwerk.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import (  # noqa: E402
    _apply_pm_genehmigungsschwelle,
    _load_pm_calibration,
    portfolio_manager,
)


# --------------------------------------------------------------------------- #
# Autouse-Fixture: isoliert jeden Test von der echten state/calibration.json
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolate_state_dir(monkeypatch):
    monkeypatch.setenv("CONCILIUM_STATE_DIR", "/nonexistent/test_pm_genehmigungsschwelle")


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #


def _write_calibration_json(
    tmp_path,
    *,
    hit_rate_gesamt: float | None = 0.3,
    hit_rates: dict[str, float | None] | None = None,
    anzahl_entscheidungen: int = 10,
    age_days: float | None = None,
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
    for action, rate in (hit_rates or {}).items():
        if rate is None:
            continue
        nach_aktion[action] = {
            "n": 5,
            "hit_rate": rate,
            "avg_confidence": 0.8,
        }

    payload: dict = {
        "anzahl_entscheidungen": anzahl_entscheidungen,
        "nach_aktion": nach_aktion,
    }
    if hit_rate_gesamt is not None:
        payload["hit_rate_gesamt"] = hit_rate_gesamt
    if not no_erstellt_am:
        erstellt = datetime.now()
        if age_days is not None:
            erstellt = erstellt - timedelta(days=age_days)
        payload["erstellt_am"] = erstellt.isoformat()

    cal_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(cal_path)


def _set_state_dir(monkeypatch, cal_path: str):
    """Zeigt CONCILIUM_STATE_DIR auf das state/-Verzeichnis der cal_path."""
    monkeypatch.setenv("CONCILIUM_STATE_DIR", os.path.dirname(cal_path))


def _pm_json(entscheidung: str = "GENEHMIGT", confidence: int = 3) -> str:
    """PM-Antwort als JSON (FINAL_SCHEMA-konform)."""
    return json.dumps(
        {
            "entscheidung": entscheidung,
            "begründung": "Test-Begründung",
            "confidence": confidence,
        },
        ensure_ascii=False,
    )


class _FakeLLM:
    """Thread-sicherer Mock-LLM, temperatur-keyed (Muster: test_ensemble.py)."""

    _DEFAULT_TEMP_KEYS = [0.3, 0.5, 0.7]

    def __init__(self, responses: list[str], temp_keys: list[float] | None = None):
        keys = temp_keys if temp_keys is not None else self._DEFAULT_TEMP_KEYS
        self._temp_map: dict[float, str] = {}
        for i, resp in enumerate(responses):
            k = round(keys[i % len(keys)], 2)
            self._temp_map[k] = resp
        self._lock = threading.Lock()

    def chat(self, messages, temperature: float = 0.3, **kwargs):
        key = round(temperature, 2)
        with self._lock:
            text = self._temp_map.get(key, next(iter(self._temp_map.values()), ""))
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult

            return StructuredChatResult(text=text, response_format_used=True)
        return text


_TRADE = {"aktion": "KAUFEN", "zielkurs": 65.0, "stop_loss": 50.0, "positionsanteil": 5}
_RISK = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}


# --------------------------------------------------------------------------- #
# Tests: _load_pm_calibration
# --------------------------------------------------------------------------- #


class TestLoadPmCalibration:
    """Test _load_pm_calibration: dict bei gültiger JSON, None sonst."""

    def test_gueltige_json_liefert_dict(self, tmp_path, monkeypatch):
        """Gültige, frische JSON → dict mit allen Kennzahlen."""
        cal = _write_calibration_json(
            tmp_path,
            hit_rate_gesamt=0.3,
            hit_rates={"KAUFEN": 0.3, "HALTEN": 0.7},
            anzahl_entscheidungen=12,
        )
        _set_state_dir(monkeypatch, cal)
        result = _load_pm_calibration()

        assert isinstance(result, dict)
        assert result["hit_rate_gesamt"] == 0.3
        assert result["anzahl_entscheidungen"] == 12
        assert result["nach_aktion"]["KAUFEN"]["hit_rate"] == 0.3
        assert result["nach_aktion"]["HALTEN"]["hit_rate"] == 0.7

    def test_fehlende_datei_liefert_none(self, monkeypatch):
        """Keine calibration.json → None."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", "/nonexistent/pm_cal_leer")
        assert _load_pm_calibration() is None

    def test_zu_alte_datei_liefert_none(self, tmp_path, monkeypatch):
        """Erstellt_am > 7 Tage → None."""
        cal = _write_calibration_json(tmp_path, hit_rate_gesamt=0.3, age_days=8)
        _set_state_dir(monkeypatch, cal)
        assert _load_pm_calibration() is None

    def test_ungueltiges_json_liefert_none(self, tmp_path, monkeypatch):
        """Kaputtes JSON → None (kein Crash)."""
        cal = _write_calibration_json(tmp_path, invalid=True)
        _set_state_dir(monkeypatch, cal)
        assert _load_pm_calibration() is None

    def test_kein_erstellt_am_liefert_none(self, tmp_path, monkeypatch):
        """Fehlendes erstellt_am → None (Alters-Check scheitert)."""
        cal = _write_calibration_json(tmp_path, hit_rate_gesamt=0.3, no_erstellt_am=True)
        _set_state_dir(monkeypatch, cal)
        assert _load_pm_calibration() is None

    def test_ungueltiges_erstellt_am_liefert_none(self, tmp_path, monkeypatch):
        """Nicht-ISO erstellt_am → None."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        cal_path = state_dir / "calibration.json"
        cal_path.write_text(
            json.dumps({"hit_rate_gesamt": 0.3, "erstellt_am": "gestern"}),
            encoding="utf-8",
        )
        _set_state_dir(monkeypatch, str(cal_path))
        assert _load_pm_calibration() is None

    def test_falsch_getypte_werte_werden_gefiltert(self, tmp_path, monkeypatch):
        """Strings statt Zahlen → keine Kennzahlen → None."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        cal_path = state_dir / "calibration.json"
        cal_path.write_text(
            json.dumps(
                {
                    "hit_rate_gesamt": "hoch",
                    "anzahl_entscheidungen": "viele",
                    "erstellt_am": datetime.now().isoformat(),
                }
            ),
            encoding="utf-8",
        )
        _set_state_dir(monkeypatch, str(cal_path))
        assert _load_pm_calibration() is None

    def test_json_liste_liefert_none(self, tmp_path, monkeypatch):
        """JSON-Array statt Objekt → None."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        cal_path = state_dir / "calibration.json"
        cal_path.write_text("[1, 2, 3]", encoding="utf-8")
        _set_state_dir(monkeypatch, str(cal_path))
        assert _load_pm_calibration() is None


# --------------------------------------------------------------------------- #
# Tests: _apply_pm_genehmigungsschwelle
# --------------------------------------------------------------------------- #


class TestApplyPmGenehmigungsschwelleGesamt:
    """Bedingung 1: Gesamt-Überkonfidenz (hit_rate_gesamt < 0.5)."""

    def _cal(self, hit_rate_gesamt: float = 0.3) -> dict:
        # Aktionen bewusst HOCH, damit nur die Gesamt-Bedingung variiert
        return {
            "hit_rate_gesamt": hit_rate_gesamt,
            "anzahl_entscheidungen": 100,
            "nach_aktion": {
                "KAUFEN": {"n": 10, "hit_rate": 0.7, "avg_confidence": 0.8},
                "HALTEN": {"n": 80, "hit_rate": 0.9, "avg_confidence": 0.7},
            },
        }

    def test_genehmigt_niedrige_confidence_wird_herabgestuft(self):
        final = {"entscheidung": "GENEHMIGT", "confidence": 3, "begründung": "Ok"}
        assert _apply_pm_genehmigungsschwelle(final, _TRADE, self._cal()) is True
        assert final["entscheidung"] == "MODIFIZIERT"
        assert final["genehmigung_herabgestuft"] is True
        assert "MODIFIZIERT herabgestuft" in final["genehmigung_herabgestuft_grund"]
        assert "Gesamt-Hit-Rate 30%" in final["genehmigung_herabgestuft_grund"]
        assert "3/5" in final["genehmigung_herabgestuft_grund"]

    def test_genehmigt_hohe_confidence_bleibt(self):
        final = {"entscheidung": "GENEHMIGT", "confidence": 5, "begründung": "Ok"}
        original = dict(final)
        assert _apply_pm_genehmigungsschwelle(final, _TRADE, self._cal()) is False
        assert final == original

    def test_genehmigt_confidence_4_grenze_bleibt(self):
        """confidence == 4 ist NICHT niedrig (Schwelle: < 4) → bleibt GENEHMIGT."""
        final = {"entscheidung": "GENEHMIGT", "confidence": 4, "begründung": "Ok"}
        original = dict(final)
        assert _apply_pm_genehmigungsschwelle(final, _TRADE, self._cal()) is False
        assert final == original

    def test_genehmigt_hohe_hit_rate_bleibt(self):
        final = {"entscheidung": "GENEHMIGT", "confidence": 3, "begründung": "Ok"}
        original = dict(final)
        assert _apply_pm_genehmigungsschwelle(final, _TRADE, self._cal(0.8)) is False
        assert final == original

    def test_hit_rate_0_5_grenze_bleibt(self):
        """hit_rate_gesamt == 0.5 ist NICHT < 0.5 → keine Herabstufung."""
        final = {"entscheidung": "GENEHMIGT", "confidence": 3, "begründung": "Ok"}
        original = dict(final)
        assert _apply_pm_genehmigungsschwelle(final, _TRADE, self._cal(0.5)) is False
        assert final == original


class TestApplyPmGenehmigungsschwelleAktion:
    """Bedingung 2: Aktions-spezifische Hit-Rate (nach_aktion)."""

    def _cal(self, kaufen_hit_rate: float, hit_rate_gesamt: float = 0.8) -> dict:
        return {
            "hit_rate_gesamt": hit_rate_gesamt,
            "anzahl_entscheidungen": 100,
            "nach_aktion": {
                "KAUFEN": {"n": 10, "hit_rate": kaufen_hit_rate, "avg_confidence": 0.8},
                "HALTEN": {"n": 80, "hit_rate": 0.9, "avg_confidence": 0.7},
            },
        }

    def test_aktions_hit_rate_niedrig_herabgestuft(self):
        """Gesamt ok (0.8), aber KAUFEN 0.3 → Herabstufung mit Aktions-Grund."""
        final = {"entscheidung": "GENEHMIGT", "confidence": 2, "begründung": "Ok"}
        assert _apply_pm_genehmigungsschwelle(final, _TRADE, self._cal(0.3)) is True
        assert final["entscheidung"] == "MODIFIZIERT"
        assert final["genehmigung_herabgestuft"] is True
        assert "für KAUFEN" in final["genehmigung_herabgestuft_grund"]

    def test_aktions_hit_rate_hoch_bleibt(self):
        final = {"entscheidung": "GENEHMIGT", "confidence": 3, "begründung": "Ok"}
        original = dict(final)
        assert _apply_pm_genehmigungsschwelle(final, _TRADE, self._cal(0.7)) is False
        assert final == original

    def test_aktion_wird_normalisiert_stark_kaufen(self):
        """'STARK KAUFEN' wird auf KAUFEN normalisiert → Aktions-Bedingung greift."""
        final = {"entscheidung": "GENEHMIGT", "confidence": 3, "begründung": "Ok"}
        trade = {"aktion": "STARK KAUFEN"}
        assert _apply_pm_genehmigungsschwelle(final, trade, self._cal(0.3)) is True
        assert final["entscheidung"] == "MODIFIZIERT"
        assert "für KAUFEN" in final["genehmigung_herabgestuft_grund"]

    def test_unbekannte_aktion_ohne_gesamt_keine_herabstufung(self):
        """Unbekannte Aktion (→ HALTEN), HALTEN fehlt in nach_aktion, Gesamt ok."""
        final = {"entscheidung": "GENEHMIGT", "confidence": 3, "begründung": "Ok"}
        trade = {"aktion": "UNSINN"}
        assert _apply_pm_genehmigungsschwelle(final, trade, self._cal(0.3)) is False
        assert final["entscheidung"] == "GENEHMIGT"


class TestApplyPmGenehmigungsschwelleRobust:
    """Nur GENEHMIGT wird downgegradet; fehlende Werte → kein Crash."""

    def test_modifiziert_bleibt_unchanged(self):
        cal = {"hit_rate_gesamt": 0.3, "nach_aktion": {}}
        final = {"entscheidung": "MODIFIZIERT", "confidence": 1, "begründung": "Ok"}
        original = dict(final)
        assert _apply_pm_genehmigungsschwelle(final, _TRADE, cal) is False
        assert final == original

    def test_ablehnt_bleibt_unchanged(self):
        cal = {"hit_rate_gesamt": 0.3, "nach_aktion": {}}
        final = {"entscheidung": "ABGELEHNT", "confidence": 1, "begründung": "Ok"}
        original = dict(final)
        assert _apply_pm_genehmigungsschwelle(final, _TRADE, cal) is False
        assert final == original

    def test_fehlende_confidence_keine_herabstufung(self):
        cal = {"hit_rate_gesamt": 0.3, "nach_aktion": {}}
        final = {"entscheidung": "GENEHMIGT", "begründung": "Ok"}
        original = dict(final)
        assert _apply_pm_genehmigungsschwelle(final, _TRADE, cal) is False
        assert final == original

    def test_ungueltige_confidence_keine_herabstufung(self):
        cal = {"hit_rate_gesamt": 0.3, "nach_aktion": {}}
        final = {"entscheidung": "GENEHMIGT", "confidence": "hoch", "begründung": "Ok"}
        original = dict(final)
        assert _apply_pm_genehmigungsschwelle(final, _TRADE, cal) is False
        assert final == original

    def test_none_calibration_keine_herabstufung(self):
        final = {"entscheidung": "GENEHMIGT", "confidence": 3, "begründung": "Ok"}
        original = dict(final)
        assert _apply_pm_genehmigungsschwelle(final, _TRADE, None) is False
        assert final == original

    def test_leere_calibration_keine_herabstufung(self):
        final = {"entscheidung": "GENEHMIGT", "confidence": 3, "begründung": "Ok"}
        original = dict(final)
        assert _apply_pm_genehmigungsschwelle(final, _TRADE, {}) is False
        assert final == original

    def test_nicht_dict_inputs_kein_crash(self):
        # final=None → False ohne Crash
        assert _apply_pm_genehmigungsschwelle(None, _TRADE, {"hit_rate_gesamt": 0.3}) is False
        # trade=None ohne Gesamt-Kennzahl → keine Bedingung erfüllbar → False, kein Crash
        final = {"entscheidung": "GENEHMIGT", "confidence": 3, "begründung": "Ok"}
        original = dict(final)
        assert _apply_pm_genehmigungsschwelle(final, None, {"nach_aktion": {}}) is False
        assert final == original

    def test_kein_float_herabstufung_bei_nur_aktions_hit_rate(self):
        """hit_rate_gesamt fehlt komplett — nur Aktions-Bedingung treibt."""
        cal = {"nach_aktion": {"VERKAUFEN": {"n": 3, "hit_rate": 0.2, "avg_confidence": 0.9}}}
        final = {"entscheidung": "GENEHMIGT", "confidence": 3, "begründung": "Ok"}
        trade = {"aktion": "STARK VERKAUFEN"}
        assert _apply_pm_genehmigungsschwelle(final, trade, cal) is True
        assert final["entscheidung"] == "MODIFIZIERT"
        assert "für VERKAUFEN" in final["genehmigung_herabgestuft_grund"]


# --------------------------------------------------------------------------- #
# Tests: portfolio_manager (Integration)
# --------------------------------------------------------------------------- #


class TestPortfolioManagerIntegration:
    """Integration: Herabstufung greift im vollen PM-Call (gemockter LLM)."""

    def test_genehmigt_conf3_wird_modifiziert(self, tmp_path, monkeypatch):
        """GENEHMIGT + confidence 3 + hit_rate_gesamt 0.3 → MODIFIZIERT + Flag."""
        cal = _write_calibration_json(
            tmp_path,
            hit_rate_gesamt=0.3,
            hit_rates={"KAUFEN": 0.3},
            anzahl_entscheidungen=100,
        )
        _set_state_dir(monkeypatch, cal)

        llm = _FakeLLM([_pm_json("GENEHMIGT", confidence=3)])
        result = portfolio_manager(_TRADE, _RISK, llm)

        assert result["entscheidung"] == "MODIFIZIERT"
        assert result["genehmigung_herabgestuft"] is True
        assert "MODIFIZIERT herabgestuft" in result["genehmigung_herabgestuft_grund"]
        assert "Gesamt-Hit-Rate 30%" in result["genehmigung_herabgestuft_grund"]
        assert "3/5" in result["genehmigung_herabgestuft_grund"]

    def test_genehmigt_conf5_bleibt_genehmigt(self, tmp_path, monkeypatch):
        """GENEHMIGT + confidence 5 → bleibt GENEHMIGT (auch bei niedriger Hit-Rate)."""
        cal = _write_calibration_json(
            tmp_path,
            hit_rate_gesamt=0.3,
            hit_rates={"KAUFEN": 0.3},
            anzahl_entscheidungen=100,
        )
        _set_state_dir(monkeypatch, cal)

        llm = _FakeLLM([_pm_json("GENEHMIGT", confidence=5)])
        result = portfolio_manager(_TRADE, _RISK, llm)

        assert result["entscheidung"] == "GENEHMIGT"
        assert result.get("genehmigung_herabgestuft") is not True

    def test_ohne_kalibrierung_bleibt_genehmigt(self, monkeypatch):
        """Keine calibration.json → GENEHMIGT bleibt unverändert."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", "/nonexistent/pm_integration_leer")

        llm = _FakeLLM([_pm_json("GENEHMIGT", confidence=3)])
        result = portfolio_manager(_TRADE, _RISK, llm)

        assert result["entscheidung"] == "GENEHMIGT"
        assert result.get("genehmigung_herabgestuft") is not True

    def test_aktions_herabstufung_im_integration(self, tmp_path, monkeypatch):
        """Gesamt ok (0.8), KAUFEN 0.3 + confidence 3 → MODIFIZIERT mit Aktions-Grund."""
        cal = _write_calibration_json(
            tmp_path,
            hit_rate_gesamt=0.8,
            hit_rates={"KAUFEN": 0.3, "HALTEN": 0.9},
            anzahl_entscheidungen=100,
        )
        _set_state_dir(monkeypatch, cal)

        llm = _FakeLLM([_pm_json("GENEHMIGT", confidence=3)])
        result = portfolio_manager(_TRADE, _RISK, llm)

        assert result["entscheidung"] == "MODIFIZIERT"
        assert result["genehmigung_herabgestuft"] is True
        assert "für KAUFEN" in result["genehmigung_herabgestuft_grund"]
