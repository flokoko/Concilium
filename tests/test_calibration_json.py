"""Tests für Kalibrierungs-JSON (Feature 1 + Feature 2).

Feature 1: --evaluate schreibt state/calibration.json mit echten Hit-Raten.
Feature 2: feedback.py liest die JSON und nutzt die echte Hit-Rate statt des Proxys.

Alle Tests sind OFFLINE-fähig: kein yfinance, kein Netzwerk.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

# src zum Pfad hinzufügen
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.cli import _write_calibration_json  # noqa: E402
from concilium.feedback import (  # noqa: E402
    _compute_kalibrierung_echt,
    _compute_kalibrierung_echt_per_action,
    _compute_stats,
    _load_calibration_json,
    build_feedback_context,
)
from concilium.journal import JOURNAL_HEADER  # noqa: E402

# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #


def _write_journal(tmp_path, rows: list[dict]) -> str:
    """Schreibt eine Journal-CSV-Datei und gibt den Pfad zurück."""
    path = str(tmp_path / "decisions.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
        writer.writeheader()
        for row in rows:
            full_row = {k: row.get(k, "") for k in JOURNAL_HEADER}
            writer.writerow(full_row)
    return path


def _make_row(
    ticker: str = "AAPL",
    action: str = "KAUFEN",
    final_decision: str = "GENEHMIGT",
    confidence: str = "4",
    timestamp: str = "2026-01-01 10:00:00",
) -> dict:
    return {
        "ticker": ticker,
        "action": action,
        "final_decision": final_decision,
        "confidence": confidence,
        "timestamp": timestamp,
    }


def _make_cal_json(
    *,
    hit_rate_gesamt: float = 0.34,
    erstellt_am: str | None = None,
    nach_aktion: dict | None = None,
    anzahl: int = 32,
) -> dict:
    """Erzeugt eine Kalibrierungs-JSON-Struktur."""
    if erstellt_am is None:
        erstellt_am = datetime.now().isoformat()
    if nach_aktion is None:
        nach_aktion = {
            "KAUFEN": {"n": 22, "hit_rate": 0.364, "avg_confidence": 0.80},
            "HALTEN": {"n": 9, "hit_rate": 0.333, "avg_confidence": 0.91},
            "VERKAUFEN": {"n": 1, "hit_rate": 0.0, "avg_confidence": 1.0},
        }
    return {
        "erstellt_am": erstellt_am,
        "anzahl_entscheidungen": anzahl,
        "hit_rate_gesamt": hit_rate_gesamt,
        "nach_aktion": nach_aktion,
    }


def _write_cal_json(tmp_path, cal_data: dict) -> str:
    """Schreibt calibration.json in tmp_path/state/ und gibt den Pfad zurück."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    cal_path = state_dir / "calibration.json"
    with open(cal_path, "w", encoding="utf-8") as fh:
        json.dump(cal_data, fh, ensure_ascii=False)
    return str(cal_path)


# --------------------------------------------------------------------------- #
# Feature 1: --evaluate schreibt calibration.json
# --------------------------------------------------------------------------- #


class TestWriteCalibrationJSON:
    """Testet dass _write_calibration_json die JSON korrekt schreibt."""

    def test_writes_valid_json(self, tmp_path, monkeypatch):
        """--evaluate schreibt state/calibration.json mit korrekten Werten."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))
        eval_result = {
            "anzahl_entscheidungen": 32,
            "hit_rate_gesamt": 0.344,
            "nach_aktion": {
                "KAUFEN": {"n": 22, "hit_rate": 0.364, "avg_rendite": 2.5, "avg_confidence": 0.80},
                "HALTEN": {"n": 9, "hit_rate": 0.333, "avg_rendite": 0.1, "avg_confidence": 0.91},
                "VERKAUFEN": {"n": 1, "hit_rate": 0.0, "avg_rendite": -1.0, "avg_confidence": 1.0},
            },
        }
        _write_calibration_json(eval_result)

        cal_path = tmp_path / "state" / "calibration.json"
        assert cal_path.is_file()

        with open(cal_path, encoding="utf-8") as fh:
            data = json.load(fh)

        assert data["anzahl_entscheidungen"] == 32
        assert data["hit_rate_gesamt"] == 0.344
        assert "erstellt_am" in data
        # nach_aktion enthält nur n, hit_rate, avg_confidence (KEINE avg_rendite)
        assert data["nach_aktion"]["KAUFEN"]["n"] == 22
        assert data["nach_aktion"]["KAUFEN"]["hit_rate"] == 0.364
        assert data["nach_aktion"]["KAUFEN"]["avg_confidence"] == 0.80
        assert "avg_rendite" not in data["nach_aktion"]["KAUFEN"]

    def test_skips_actions_with_zero_n(self, tmp_path, monkeypatch):
        """Aktionen mit n=0 werden weggelassen."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))
        eval_result = {
            "anzahl_entscheidungen": 5,
            "hit_rate_gesamt": 0.4,
            "nach_aktion": {
                "KAUFEN": {"n": 5, "hit_rate": 0.4, "avg_rendite": 1.0, "avg_confidence": 0.8},
                "HALTEN": {"n": 0, "hit_rate": None, "avg_rendite": None, "avg_confidence": None},
                "VERKAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None, "avg_confidence": None},
            },
        }
        _write_calibration_json(eval_result)

        cal_path = tmp_path / "state" / "calibration.json"
        with open(cal_path, encoding="utf-8") as fh:
            data = json.load(fh)

        assert "KAUFEN" in data["nach_aktion"]
        assert "HALTEN" not in data["nach_aktion"]
        assert "VERKAUFEN" not in data["nach_aktion"]

    def test_atomic_write_no_tmp_file_left(self, tmp_path, monkeypatch):
        """Nach dem Schreiben gibt es keine .tmp-Datei mehr."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))
        eval_result = {
            "anzahl_entscheidungen": 1,
            "hit_rate_gesamt": 0.0,
            "nach_aktion": {
                "KAUFEN": {"n": 1, "hit_rate": 0.0, "avg_rendite": -1.0, "avg_confidence": 0.8},
                "HALTEN": {"n": 0, "hit_rate": None, "avg_rendite": None, "avg_confidence": None},
                "VERKAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None, "avg_confidence": None},
            },
        }
        _write_calibration_json(eval_result)

        state_dir = tmp_path / "state"
        files = list(state_dir.iterdir())
        # Nur calibration.json, keine .tmp
        assert all(not f.name.endswith(".tmp") for f in files)
        assert (state_dir / "calibration.json").is_file()

    def test_does_not_crash_on_invalid_state_dir(self, monkeypatch):
        """Bei ungültigem State-Dir crasht es nicht (nur Log-Warnung)."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", "/nonexistent/path/that/cannot/be/created")
        eval_result = {
            "anzahl_entscheidungen": 0,
            "hit_rate_gesamt": None,
            "nach_aktion": {},
        }
        # Sollte nicht raisen
        _write_calibration_json(eval_result)

    def test_conciliium_state_dir_override(self, tmp_path, monkeypatch):
        """CONCILIUM_STATE_DIR übersteuert das Verzeichnis."""
        custom_dir = str(tmp_path / "custom_state")
        monkeypatch.setenv("CONCILIUM_STATE_DIR", custom_dir)
        eval_result = {
            "anzahl_entscheidungen": 1,
            "hit_rate_gesamt": 1.0,
            "nach_aktion": {
                "KAUFEN": {"n": 1, "hit_rate": 1.0, "avg_rendite": 5.0, "avg_confidence": 0.8},
                "HALTEN": {"n": 0, "hit_rate": None, "avg_rendite": None, "avg_confidence": None},
                "VERKAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None, "avg_confidence": None},
            },
        }
        _write_calibration_json(eval_result)

        assert os.path.isfile(os.path.join(custom_dir, "calibration.json"))


# --------------------------------------------------------------------------- #
# Feature 2: _load_calibration_json
# --------------------------------------------------------------------------- #


class TestLoadCalibrationJSON:
    """Testet _load_calibration_json."""

    def test_reads_valid_json(self, tmp_path, monkeypatch):
        """Gültige JSON wird korrekt gelesen."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        cal_data = _make_cal_json()
        with open(state_dir / "calibration.json", "w", encoding="utf-8") as fh:
            json.dump(cal_data, fh)

        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(state_dir))
        result = _load_calibration_json()
        assert result is not None
        assert result["hit_rate_gesamt"] == 0.34
        assert result["anzahl_entscheidungen"] == 32

    def test_returns_none_on_missing_file(self, tmp_path, monkeypatch):
        """Fehlende Datei → None."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))
        result = _load_calibration_json()
        assert result is None

    def test_returns_none_on_corrupt_json(self, tmp_path, monkeypatch):
        """Kaputtes JSON → None."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        with open(state_dir / "calibration.json", "w") as fh:
            fh.write("{invalid json")

        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(state_dir))
        result = _load_calibration_json()
        assert result is None

    def test_returns_none_when_too_old(self, tmp_path, monkeypatch):
        """JSON älter als 7 Tage → None."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        old_date = (datetime.now() - timedelta(days=10)).isoformat()
        cal_data = _make_cal_json(erstellt_am=old_date)
        with open(state_dir / "calibration.json", "w", encoding="utf-8") as fh:
            json.dump(cal_data, fh)

        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(state_dir))
        result = _load_calibration_json()
        assert result is None

    def test_returns_none_when_no_erstellt_am(self, tmp_path, monkeypatch):
        """Kein erstellt_am → None."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        cal_data = {
            "anzahl_entscheidungen": 10,
            "hit_rate_gesamt": 0.5,
            "nach_aktion": {},
        }
        with open(state_dir / "calibration.json", "w", encoding="utf-8") as fh:
            json.dump(cal_data, fh)

        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(state_dir))
        result = _load_calibration_json()
        assert result is None


# --------------------------------------------------------------------------- #
# Feature 2: _compute_kalibrierung_echt
# --------------------------------------------------------------------------- #


class TestComputeKalibrierungEcht:
    """Testet die echte Kalibrierung aus JSON."""

    def test_overconfident(self):
        """Hohe Confidence, niedrige Hit-Rate → überkonfident."""
        cal = _make_cal_json(
            hit_rate_gesamt=0.34,
            nach_aktion={
                "KAUFEN": {"n": 22, "hit_rate": 0.36, "avg_confidence": 0.80},
                "HALTEN": {"n": 9, "hit_rate": 0.33, "avg_confidence": 0.91},
                "VERKAUFEN": {"n": 1, "hit_rate": 0.0, "avg_confidence": 1.0},
            },
        )
        result = _compute_kalibrierung_echt(cal)
        # Gewichtete avg_confidence:
        # (0.80*22 + 0.91*9 + 1.0*1) / 32 = (17.6 + 8.19 + 1.0) / 32 = 26.79/32 ≈ 0.837
        assert result["avg_confidence"] is not None
        assert abs(result["avg_confidence"] - (17.6 + 8.19 + 1.0) / 32) < 0.01
        assert result["hit_rate"] == 0.34
        assert result["gap"] > 0.15
        assert result["tendenz"] == "überkonfident"

    def test_well_calibrated(self):
        """Confidence ≈ Hit-Rate → gut kalibriert."""
        cal = _make_cal_json(
            hit_rate_gesamt=0.60,
            nach_aktion={
                "KAUFEN": {"n": 10, "hit_rate": 0.60, "avg_confidence": 0.60},
            },
        )
        result = _compute_kalibrierung_echt(cal)
        assert abs(result["gap"]) <= 0.15
        assert result["tendenz"] == "gut kalibriert"

    def test_returns_empty_on_missing_hit_rate(self):
        """Kein hit_rate_gesamt → leeres dict."""
        cal = {
            "anzahl_entscheidungen": 10,
            "hit_rate_gesamt": None,
            "nach_aktion": {"KAUFEN": {"n": 10, "hit_rate": 0.5, "avg_confidence": 0.8}},
        }
        result = _compute_kalibrierung_echt(cal)
        assert result["avg_confidence"] is None
        assert result["hit_rate"] is None

    def test_returns_empty_on_no_actions(self):
        """Keine Aktionen → leeres dict."""
        cal = {
            "anzahl_entscheidungen": 0,
            "hit_rate_gesamt": 0.5,
            "nach_aktion": {},
        }
        result = _compute_kalibrierung_echt(cal)
        assert result["avg_confidence"] is None


# --------------------------------------------------------------------------- #
# Feature 2: _compute_kalibrierung_echt_per_action
# --------------------------------------------------------------------------- #


class TestComputeKalibrierungEchtPerAction:
    """Testet die echte Kalibrierung pro Aktion aus JSON."""

    def test_filters_actions_below_n3(self):
        """Aktionen mit n < 3 werden weggelassen."""
        cal = _make_cal_json(
            nach_aktion={
                "KAUFEN": {"n": 22, "hit_rate": 0.36, "avg_confidence": 0.80},
                "HALTEN": {"n": 2, "hit_rate": 0.33, "avg_confidence": 0.91},
                "VERKAUFEN": {"n": 1, "hit_rate": 0.0, "avg_confidence": 1.0},
            },
        )
        result = _compute_kalibrierung_echt_per_action(cal)
        assert "KAUFEN" in result
        assert "HALTEN" not in result  # n=2 < 3
        assert "VERKAUFEN" not in result  # n=1 < 3

    def test_correct_values(self):
        """Werte werden korrekt berechnet."""
        cal = _make_cal_json(
            nach_aktion={
                "KAUFEN": {"n": 10, "hit_rate": 0.30, "avg_confidence": 0.80},
            },
        )
        result = _compute_kalibrierung_echt_per_action(cal)
        assert "KAUFEN" in result
        kaufen = result["KAUFEN"]
        assert kaufen["avg_confidence"] == 0.80
        assert kaufen["hit_rate"] == 0.30
        assert abs(kaufen["gap"] - 0.50) < 0.01
        assert kaufen["tendenz"] == "überkonfident"
        assert kaufen["n"] == 10

    def test_empty_dict_on_no_data(self):
        """Keine Aktionen → leeres dict."""
        cal = {"hit_rate_gesamt": 0.5, "nach_aktion": {}}
        result = _compute_kalibrierung_echt_per_action(cal)
        assert result == {}


# --------------------------------------------------------------------------- #
# Feature 2: _compute_stats nutzt echte Hit-Rate / Fallback
# --------------------------------------------------------------------------- #


class TestComputeStatsWithCalibration:
    """Testet dass _compute_stats die echte Hit-Rate nutzt wenn JSON vorhanden."""

    def test_uses_echte_hit_rate_when_json_present(self, tmp_path, monkeypatch):
        """Wenn calibration.json existiert → quelle='echte_hit_rate'."""
        cal_data = _make_cal_json(
            hit_rate_gesamt=0.34,
            nach_aktion={
                "KAUFEN": {"n": 22, "hit_rate": 0.36, "avg_confidence": 0.80},
                "HALTEN": {"n": 9, "hit_rate": 0.33, "avg_confidence": 0.91},
            },
        )
        _write_cal_json(tmp_path, cal_data)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        rows = [_make_row(ticker=f"T{i}") for i in range(5)]
        stats = _compute_stats(rows, min_decisions=5)

        assert stats["kalibrierung"]["quelle"] == "echte_hit_rate"
        assert "hit_rate" in stats["kalibrierung"]
        assert stats["kalibrierung"]["hit_rate"] == 0.34

    def test_falls_back_to_proxy_when_json_missing(self, tmp_path, monkeypatch):
        """Wenn keine calibration.json → quelle='proxy'."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        rows = [_make_row(ticker=f"T{i}", confidence="4") for i in range(5)]
        stats = _compute_stats(rows, min_decisions=5)

        assert stats["kalibrierung"]["quelle"] == "proxy"
        assert "genehmigungs_rate" in stats["kalibrierung"]

    def test_falls_back_to_proxy_when_json_too_old(self, tmp_path, monkeypatch):
        """Wenn calibration.json zu alt → Fallback auf proxy."""
        old_date = (datetime.now() - timedelta(days=10)).isoformat()
        cal_data = _make_cal_json(erstellt_am=old_date)
        _write_cal_json(tmp_path, cal_data)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        rows = [_make_row(ticker=f"T{i}", confidence="4") for i in range(5)]
        stats = _compute_stats(rows, min_decisions=5)

        assert stats["kalibrierung"]["quelle"] == "proxy"

    def test_falls_back_to_proxy_when_too_few_in_json(self, tmp_path, monkeypatch):
        """Wenn calibration.json anzahl < min_decisions → Fallback auf proxy."""
        cal_data = _make_cal_json(anzahl=3)
        _write_cal_json(tmp_path, cal_data)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        rows = [_make_row(ticker=f"T{i}", confidence="4") for i in range(5)]
        stats = _compute_stats(rows, min_decisions=5)

        assert stats["kalibrierung"]["quelle"] == "proxy"


# --------------------------------------------------------------------------- #
# Feature 2: build_feedback_context zeigt echte Hit-Rate vs Proxy
# --------------------------------------------------------------------------- #


class TestFeedbackContextEchteHitRate:
    """Testet dass build_feedback_context die richtige Beschriftung zeigt."""

    def test_shows_echte_hit_rate_when_json_present(self, tmp_path, monkeypatch):
        """Bei vorhandener JSON → 'echte Hit-Rate' in der Ausgabe."""
        cal_data = _make_cal_json(
            hit_rate_gesamt=0.34,
            nach_aktion={
                "KAUFEN": {"n": 22, "hit_rate": 0.36, "avg_confidence": 0.80},
                "HALTEN": {"n": 9, "hit_rate": 0.33, "avg_confidence": 0.91},
            },
        )
        _write_cal_json(tmp_path, cal_data)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        rows = [_make_row(ticker=f"T{i}", confidence="4") for i in range(5)]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        assert "echte Hit-Rate" in result
        assert "34%" in result

    def test_shows_genehmigungs_rate_when_proxy(self, tmp_path, monkeypatch):
        """Ohne JSON → 'Genehmigungs-Rate' (Proxy) in der Ausgabe."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        rows = [_make_row(ticker=f"T{i}", confidence="4", final_decision="GENEHMIGT") for i in range(5)]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        assert "Genehmigungs-Rate" in result
        assert "echte Hit-Rate" not in result

    def test_shows_proxy_marker_in_kalibrierung_line(self, tmp_path, monkeypatch):
        """Proxy-Fallback zeigt '(Proxy)' in der Kalibrierungs-Zeile."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        rows = [_make_row(ticker=f"T{i}", confidence="4", final_decision="GENEHMIGT") for i in range(5)]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        assert "Konfidenz-Kalibrierung (Proxy)" in result

    def test_pro_aktion_shows_hit_rate_when_json(self, tmp_path, monkeypatch):
        """Pro-Aktion-Block zeigt 'Hit-Rate' bei JSON, nicht 'Genehmigungs-Rate'."""
        cal_data = _make_cal_json(
            hit_rate_gesamt=0.34,
            nach_aktion={
                "KAUFEN": {"n": 22, "hit_rate": 0.36, "avg_confidence": 0.80},
                "HALTEN": {"n": 9, "hit_rate": 0.33, "avg_confidence": 0.91},
            },
        )
        _write_cal_json(tmp_path, cal_data)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        rows = [_make_row(ticker=f"T{i}", action="KAUFEN", confidence="4") for i in range(5)]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        # Pro-Aktion-Block sollte "Hit-Rate" enthalten, nicht "Genehmigungs-Rate"
        assert "Hit-Rate" in result

    def test_pro_aktion_shows_genehmigungs_rate_when_proxy(self, tmp_path, monkeypatch):
        """Pro-Aktion-Block zeigt 'Genehmigungs-Rate' beim Proxy."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="C", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="D", action="KAUFEN", confidence="4", final_decision="ABGELEHNT"),
            _make_row(ticker="E", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        assert "Genehmigungs-Rate" in result

    def test_no_yfinance_import_in_feedback(self, tmp_path, monkeypatch):
        """feedback.py lädt kein yfinance (netzfrei) — auch nicht mit JSON."""
        cal_data = _make_cal_json()
        _write_cal_json(tmp_path, cal_data)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        rows = [_make_row(ticker=f"T{i}", confidence="4") for i in range(5)]
        path = _write_journal(tmp_path, rows)

        # Wenn yfinance aufgerufen würde, würde der mock fehlschlagen
        with patch("concilium.feedback.realised_return_for_row") as mock_rr:
            result = build_feedback_context(path)
            assert mock_rr.call_count == 0

        assert "echte Hit-Rate" in result
