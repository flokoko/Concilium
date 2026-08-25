"""Tests für kalibrierungs-gewichtete Ensemble-Abstimmung.

Testet:
- _load_ensemble_weights: liest korrekt, None bei fehlender/zu alter/ungültiger JSON
- _smooth_weight: geglättete Formel (0.5 + 0.5 * hit_rate)
- Gewichtete Abstimmung: KAUFEN mit hohem Gewicht schlägt HALTEN 2:1
- Fallback ohne weights: ungewichtete Mehrheit (unverändert)
- _ensemble enthält 'gewichtet' und 'aktion_gewichte'
- Report zeigt Gewichtungs-Hinweis nur wenn gewichtet=True

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
    _load_ensemble_weights,
    _smooth_weight,
    ensemble_trader,
)
from concilium.report import generate_report  # noqa: E402

# --------------------------------------------------------------------------- #
# Hilfsfunktionen (angepasst aus test_ensemble.py)
# --------------------------------------------------------------------------- #


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
    """Thread-sicherer Mock-LLM, temperatur-keyed (wie in test_ensemble.py)."""

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


def _write_calibration_json(
    tmp_path,
    *,
    hit_rates: dict[str, float | None] | None = None,
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
    for action in ("KAUFEN", "HALTEN", "VERKAUFEN"):
        rate = hr.get(action)
        if rate is None:
            continue
        nach_aktion[action] = {
            "n": 5,
            "hit_rate": rate,
            "avg_confidence": 0.6,
        }

    payload: dict = {
        "anzahl_entscheidungen": 10,
        "hit_rate_gesamt": 0.4,
        "nach_aktion": nach_aktion,
    }
    if not no_erstellt_am:
        payload["erstellt_am"] = (erstellt_am or datetime.now()).isoformat()

    cal_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(cal_path)


# --------------------------------------------------------------------------- #
# Tests: _load_ensemble_weights
# --------------------------------------------------------------------------- #


class TestLoadEnsembleWeights:
    """Test _load_ensemble_weights: liest korrekt, None bei Problemen."""

    def test_reads_valid_json(self, tmp_path):
        """Gültige JSON mit Hit-Raten → dict zurück."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.8, "HALTEN": 0.5, "VERKAUFEN": 0.0},
        )
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            weights = _load_ensemble_weights()
        assert weights is not None
        assert weights["KAUFEN"] == 0.8
        assert weights["HALTEN"] == 0.5
        assert weights["VERKAUFEN"] == 0.0

    def test_returns_none_missing_file(self, tmp_path):
        """Keine Datei → None."""
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _load_ensemble_weights() is None

    def test_returns_none_invalid_json(self, tmp_path):
        """Kaputtes JSON → None."""
        _write_calibration_json(tmp_path, invalid=True)
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _load_ensemble_weights() is None

    def test_returns_none_too_old(self, tmp_path):
        """Datei älter als 7 Tage → None."""
        old_date = datetime.now() - timedelta(days=10)
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.8},
            erstellt_am=old_date,
        )
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _load_ensemble_weights() is None

    def test_returns_none_no_erstellt_am(self, tmp_path):
        """Kein erstellt_am → None."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.8},
            no_erstellt_am=True,
        )
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _load_ensemble_weights() is None

    def test_returns_none_no_hit_rates(self, tmp_path):
        """JSON ohne nach_aktion oder ohne Hit-Raten → None."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        cal_path = state_dir / "calibration.json"
        payload = {
            "erstellt_am": datetime.now().isoformat(),
            "anzahl_entscheidungen": 0,
            "hit_rate_gesamt": None,
            "nach_aktion": {},
        }
        cal_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _load_ensemble_weights() is None

    def test_partial_hit_rates_ok(self, tmp_path):
        """Nur eine Aktion hat Hit-Rate → dict mit nur dieser Aktion."""
        _write_calibration_json(tmp_path, hit_rates={"KAUFEN": 0.36})
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            weights = _load_ensemble_weights()
        assert weights is not None
        assert "KAUFEN" in weights
        assert "HALTEN" not in weights

    def test_never_crashes_on_garbage(self, tmp_path):
        """Beliebiger Müll als Datei → None, kein Crash."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        cal_path = state_dir / "calibration.json"
        cal_path.write_text("garbage content", encoding="utf-8")
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            assert _load_ensemble_weights() is None


# --------------------------------------------------------------------------- #
# Tests: _smooth_weight
# --------------------------------------------------------------------------- #


class TestSmoothWeight:
    """Test geglättete Gewichts-Formel: 0.5 + 0.5 * hit_rate."""

    def test_hit_rate_0(self):
        """hit_rate 0.0 → gewicht 0.5."""
        assert _smooth_weight(0.0) == 0.5

    def test_hit_rate_1(self):
        """hit_rate 1.0 → gewicht 1.0."""
        assert _smooth_weight(1.0) == 1.0

    def test_hit_rate_0_5(self):
        """hit_rate 0.5 → gewicht 0.75."""
        assert _smooth_weight(0.5) == 0.75

    def test_hit_rate_0_36(self):
        """hit_rate 0.36 → gewicht 0.68."""
        assert round(_smooth_weight(0.36), 2) == 0.68

    def test_clamps_above_1(self):
        """hit_rate > 1.0 wird auf 1.0 geclamped → gewicht 1.0."""
        assert _smooth_weight(1.5) == 1.0

    def test_clamps_below_0(self):
        """hit_rate < 0.0 wird auf 0.0 geclamped → gewicht 0.5."""
        assert _smooth_weight(-0.5) == 0.5


# --------------------------------------------------------------------------- #
# Tests: Gewichtete Abstimmung
# --------------------------------------------------------------------------- #


class TestGewichteteAbstimmung:
    """Test kalibrierungs-gewichtete vs. ungewichtete Abstimmung."""

    def test_kaufen_gewinnt_gegen_2_halten_bei_gleichstand(self, tmp_path):
        """Bei Gleichstand (KAUFEN=1.0, HALTEN=1.0) gewinnt KAUFEN (first key).

        KAUFEN hit_rate=1.0 → gewicht 1.0
        HALTEN hit_rate=0.0 → gewicht 0.5
        Runs: [KAUFEN, HALTEN, HALTEN]
        Gewicht-Summen: KAUFEN=1.0, HALTEN=1.0 → Gleichstand → KAUFEN (first) gewinnt.
        Ungewichtet würde HALTEN 2:1 gewinnen.
        """
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 1.0, "HALTEN": 0.0},
        )
        llm = _FakeLLM([
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),  # temp=0.3
            _trader_json("HALTEN"),                                 # temp=0.5
            _trader_json("HALTEN"),                                 # temp=0.7
        ])
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        assert result["aktion"] == "KAUFEN"
        assert result["_ensemble"]["mehrheits_aktion"] == "KAUFEN"
        assert result["_ensemble"]["gewichtet"] is True

    def test_halten_gewinnt_ohne_gewichtung(self, tmp_path):
        """Ohne calibration.json → ungewichtete Mehrheit: HALTEN 2:1 gewinnt.

        Dies ist der Fallback-Verhalten (unverändert zur bisherigen Logik).
        """
        # KEINE calibration.json → _load_ensemble_weights() → None
        llm = _FakeLLM([
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),  # temp=0.3
            _trader_json("HALTEN"),                                 # temp=0.5
            _trader_json("HALTEN"),                                 # temp=0.7
        ])
        # CONCILIUM_STATE_DIR auf leeres Verzeichnis → keine JSON
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "nonexistent")}):
            result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        assert result["aktion"] == "HALTEN"
        assert result["_ensemble"]["mehrheits_aktion"] == "HALTEN"
        assert result["_ensemble"]["gewichtet"] is False
        assert result["_ensemble"]["aktion_gewichte"] == {}

    def test_gewichtung_aendert_ergebnis(self, tmp_path):
        """Ein klarer Flip: 4 Runs [KAUFEN, KAUFEN, HALTEN, HALTEN].

        Ungewichtet: 2:2 → Gleichstand → KAUFEN (first key) gewinnt.
        Gewichtet mit KAUFEN=0.0 (gewicht 0.5), HALTEN=1.0 (gewicht 1.0):
        KAUFEN=0.5+0.5=1.0, HALTEN=1.0+1.0=2.0 → HALTEN gewinnt (Flip!).
        """
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.0, "HALTEN": 1.0},
        )
        llm = _FakeLLM(
            [
                _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),  # temp=0.3
                _trader_json("KAUFEN", zielkurs=62.0, stop_loss=52.0),  # temp=0.5
                _trader_json("HALTEN"),                                 # temp=0.7
                _trader_json("HALTEN"),                                 # temp=0.9
            ],
            temp_keys=[0.3, 0.5, 0.7, 0.9],
        )
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            result = ensemble_trader(
                _ANALYSTS, _DEBATE, llm, runs=4,
                temperature_range=[0.3, 0.5, 0.7, 0.9],
            )

        assert result["aktion"] == "HALTEN"
        assert result["_ensemble"]["mehrheits_aktion"] == "HALTEN"
        assert result["_ensemble"]["gewichtet"] is True


# --------------------------------------------------------------------------- #
# Tests: _ensemble Metadaten
# --------------------------------------------------------------------------- #


class TestEnsembleMetadaten:
    """Test dass _ensemble 'gewichtet' und 'aktion_gewichte' enthält."""

    def test_gewichtet_true_mit_json(self, tmp_path):
        """Mit calibration.json → gewichtet=True, aktion_gewichte nicht leer."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.8, "HALTEN": 0.5, "VERKAUFEN": 0.0},
        )
        llm = _FakeLLM([
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),
            _trader_json("HALTEN"),
            _trader_json("KAUFEN", zielkurs=62.0, stop_loss=52.0),
        ])
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}):
            result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        ens = result["_ensemble"]
        assert "gewichtet" in ens
        assert ens["gewichtet"] is True
        assert "aktion_gewichte" in ens
        assert isinstance(ens["aktion_gewichte"], dict)
        assert ens["aktion_gewichte"]["KAUFEN"] == 0.9  # 0.5 + 0.5*0.8
        assert ens["aktion_gewichte"]["HALTEN"] == 0.75  # 0.5 + 0.5*0.5
        assert ens["aktion_gewichte"]["VERKAUFEN"] == 0.5  # 0.5 + 0.5*0.0

    def test_gewichtet_false_ohne_json(self, tmp_path):
        """Ohne calibration.json → gewichtet=False, aktion_gewichte={}."""
        llm = _FakeLLM([
            _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),
            _trader_json("HALTEN"),
            _trader_json("KAUFEN", zielkurs=62.0, stop_loss=52.0),
        ])
        with patch.dict(os.environ, {"CONCILIUM_STATE_DIR": str(tmp_path / "nonexistent")}):
            result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        ens = result["_ensemble"]
        assert ens["gewichtet"] is False
        assert ens["aktion_gewichte"] == {}


# --------------------------------------------------------------------------- #
# Tests: Report zeigt Gewichtungs-Hinweis
# --------------------------------------------------------------------------- #


class TestReportGewichtungsHinweis:
    """Test dass der Report den Gewichtungs-Hinweis nur zeigt wenn gewichtet=True."""

    def _make_result(self, gewichtet: bool, aktion_gewichte: dict | None = None) -> dict:
        """Baut ein minimales result-dict für generate_report."""
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
                "aktion": "KAUFEN",
                "rating": "KAUFEN",
                "zielkurs": 120.0,
                "stop_loss": 90.0,
                "positionsanteil": 5,
                "zeithorizont": "Mittelfristig",
                "begründung": "Test",
                "_ensemble": {
                    "runs": 3,
                    "mehrheits_aktion": "KAUFEN",
                    "ensemble_confidence": 0.67,
                    "alle_aktionen": ["KAUFEN", "HALTEN", "KAUFEN"],
                    "alle_ratings": ["KAUFEN", "HALTEN", "KAUFEN"],
                    "gewichtet": gewichtet,
                    "aktion_gewichte": aktion_gewichte or {},
                },
            },
            "final": {"entscheidung": "GENEHMIGT", "begründung": "Ok"},
            "risk": {"risiko_score": 3},
        }

    def test_hinweis_zeigt_sich_wenn_gewichtet(self):
        """gewichtet=True → 'Ensemble kalibrierungs-gewichtet' im Report."""
        result = self._make_result(
            True,
            {"KAUFEN": 0.9, "HALTEN": 0.75, "VERKAUFEN": 0.5},
        )
        report = generate_report(result)
        assert "kalibrierungs-gewichtet" in report
        assert "KAUFEN 0.90" in report
        assert "HALTEN 0.75" in report
        assert "VERKAUFEN 0.50" in report

    def test_kein_hinweis_ohne_gewichtung(self):
        """gewichtet=False → kein 'kalibrierungs-gewichtet' im Report."""
        result = self._make_result(False)
        report = generate_report(result)
        assert "kalibrierungs-gewichtet" not in report

    def test_kein_hinweis_bei_leeren_gewichten(self):
        """gewichtet=True aber aktion_gewichte leer → kein Hinweis (Guard)."""
        result = self._make_result(True, {})
        report = generate_report(result)
        assert "kalibrierungs-gewichtet" not in report
