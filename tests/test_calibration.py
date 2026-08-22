"""Tests für Phase 4 — Konfidenz-Kalibrierung (Brier-Score, Reliability-Bänder).

Alle Tests sind OFFLINE-fähig: yfinance wird gemockt, kein Netzwerk.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

# src zum Pfad hinzufügen
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.evaluate import (  # noqa: E402
    _aggregate,
    _compute_konfidenz_kalibrierung,
    _compute_reliability_bins,
    _empty_result,
    evaluate_journal,
)
from concilium.feedback import build_feedback_context  # noqa: E402
from concilium.journal import JOURNAL_HEADER  # noqa: E402
from concilium.report import generate_track_record_report  # noqa: E402

# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #


def _make_eval(
    confidence: float | None,
    hit: bool | None,
    action: str = "KAUFEN",
) -> dict:
    """Erzeugt ein Einzel-Ergebnis-dict wie aus _evaluate_single."""
    return {
        "hit": hit,
        "rendite_pct": 1.0 if hit else -1.0,
        "ziel_erreicht": None,
        "stop_gerissen": None,
        "action": action,
        "rating": "",
        "rating_distance": None,
        "confidence": confidence,
        "portfolio_fit_score": None,
        "ticker": "TEST",
        "timestamp": "2026-01-01 10:00:00",
    }


def _make_prices(start_price: float, n_days: int, drift: float = 0.0) -> list[dict]:
    """Erzeugt eine Liste von Preis-Dicts für n_days Tage."""
    prices: list[dict] = []
    base_date = datetime.now() - timedelta(days=n_days + 5)
    price = start_price
    for i in range(n_days):
        d = base_date + timedelta(days=i)
        price = price * (1.0 + drift)
        prices.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "close": round(price, 2),
                "high": round(price * 1.01, 2),
                "low": round(price * 0.99, 2),
            }
        )
    return prices


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


def _make_journal_row(
    ticker: str = "AAPL",
    action: str = "KAUFEN",
    confidence: str = "4",
    final_decision: str = "GENEHMIGT",
    timestamp: str = "",
) -> dict:
    if not timestamp:
        timestamp = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "ticker": ticker,
        "action": action,
        "confidence": confidence,
        "final_decision": final_decision,
        "timestamp": timestamp,
    }


# --------------------------------------------------------------------------- #
# Tests: Brier-Score
# --------------------------------------------------------------------------- #


class TestBrierScore:
    """Testet den Brier-Score in _compute_konfidenz_kalibrierung."""

    def test_perfect_calibration_low_brier(self):
        """Confidence = 5 (p=1.0), alle hits → Brier ≈ 0."""
        evals = [
            _make_eval(confidence=5.0, hit=True),
            _make_eval(confidence=5.0, hit=True),
            _make_eval(confidence=5.0, hit=True),
        ]
        kal = _compute_konfidenz_kalibrierung(evals)
        assert kal["brier_score"] is not None
        assert kal["brier_score"] < 0.01  # nahe 0

    def test_miscalibrated_high_brier(self):
        """Confidence = 4 (p=0.8), aber alle misses → Brier hoch."""
        evals = [
            _make_eval(confidence=4.0, hit=False),
            _make_eval(confidence=4.0, hit=False),
            _make_eval(confidence=4.0, hit=False),
        ]
        kal = _compute_konfidenz_kalibrierung(evals)
        assert kal["brier_score"] is not None
        # p=0.8, hit=0 → brier_i = 0.64, Ø = 0.64
        assert kal["brier_score"] > 0.5

    def test_brier_perfect_is_lower_than_miscalibrated(self):
        """Perfekt kalibriert → niedrigerer Brier als unpassend."""
        perfect = _compute_konfidenz_kalibrierung(
            [_make_eval(confidence=5.0, hit=True) for _ in range(10)]
        )
        miscalibrated = _compute_konfidenz_kalibrierung(
            [_make_eval(confidence=5.0, hit=False) for _ in range(10)]
        )
        assert perfect["brier_score"] < miscalibrated["brier_score"]

    def test_brier_mixed(self):
        """Gemischte Werte → Brier zwischen 0 und max."""
        evals = [
            _make_eval(confidence=5.0, hit=True),   # (1.0 - 1)^2 = 0
            _make_eval(confidence=4.0, hit=False),   # (0.8 - 0)^2 = 0.64
            _make_eval(confidence=3.0, hit=True),   # (0.6 - 1)^2 = 0.16
            _make_eval(confidence=2.0, hit=False),   # (0.4 - 0)^2 = 0.16
        ]
        kal = _compute_konfidenz_kalibrierung(evals)
        expected = (0.0 + 0.64 + 0.16 + 0.16) / 4
        assert math.isclose(kal["brier_score"], expected, rel_tol=1e-6)

    def test_brier_none_when_no_valid(self):
        """Keine confidence/hit → Brier = None."""
        evals = [
            _make_eval(confidence=None, hit=True),
            _make_eval(confidence=4.0, hit=None),
        ]
        kal = _compute_konfidenz_kalibrierung(evals)
        assert kal["brier_score"] is None
        assert kal["n"] == 0


# --------------------------------------------------------------------------- #
# Tests: Kalibrierungs-Gap und Tendenz
# --------------------------------------------------------------------------- #


class TestKalibrierungsGap:
    """Testet Kalibrierungs-Gap und Tendenz-Klassifikation."""

    def test_gap_overconfident(self):
        """Hohe Confidence, niedrige Hit-Rate → überkonfident."""
        evals = [
            _make_eval(confidence=5.0, hit=False),
            _make_eval(confidence=5.0, hit=False),
            _make_eval(confidence=5.0, hit=False),
            _make_eval(confidence=5.0, hit=False),
            _make_eval(confidence=5.0, hit=True),  # 1 hit, 4 miss
        ]
        kal = _compute_konfidenz_kalibrierung(evals)
        # avg_conf = 1.0, avg_hit = 0.2, gap = 0.8
        assert kal["kalibrierungs_gap"] > 0.15
        assert kal["tendenz"] == "überkonfident"

    def test_gap_underconfident(self):
        """Niedrige Confidence, hohe Hit-Rate → unterkonfident."""
        evals = [
            _make_eval(confidence=1.0, hit=True),
            _make_eval(confidence=1.0, hit=True),
            _make_eval(confidence=1.0, hit=True),
            _make_eval(confidence=1.0, hit=True),
            _make_eval(confidence=1.0, hit=True),
        ]
        kal = _compute_konfidenz_kalibrierung(evals)
        # avg_conf = 0.2, avg_hit = 1.0, gap = -0.8
        assert kal["kalibrierungs_gap"] < -0.15
        assert kal["tendenz"] == "unterkonfident"

    def test_gap_well_calibrated(self):
        """Confidence ≈ Hit-Rate → gut kalibriert."""
        evals = [
            _make_eval(confidence=3.0, hit=True),   # p=0.6, hit=1
            _make_eval(confidence=3.0, hit=False),  # p=0.6, hit=0
            _make_eval(confidence=3.0, hit=True),   # p=0.6, hit=1
            _make_eval(confidence=3.0, hit=False),  # p=0.6, hit=0
        ]
        kal = _compute_konfidenz_kalibrierung(evals)
        # avg_conf = 0.6, avg_hit = 0.5, gap = 0.1 → gut kalibriert
        assert abs(kal["kalibrierungs_gap"]) <= 0.15
        assert kal["tendenz"] == "gut kalibriert"


# --------------------------------------------------------------------------- #
# Tests: Reliability-Bins
# --------------------------------------------------------------------------- #


class TestReliabilityBins:
    """Testet die Reliability-Bänder-Gruppierung."""

    def test_bins_grouped_correctly(self):
        """Zeilen werden in die richtigen Konfidenz-Intervalle gruppiert."""
        evals = [
            _make_eval(confidence=1.0, hit=True),   # p=0.2 → [0.2-0.4)
            _make_eval(confidence=2.0, hit=False),  # p=0.4 → [0.4-0.6)
            _make_eval(confidence=3.0, hit=True),   # p=0.6 → [0.6-0.8)
            _make_eval(confidence=4.0, hit=False),  # p=0.8 → [0.8-1.0]
            _make_eval(confidence=5.0, hit=True),   # p=1.0 → [0.8-1.0]
        ]
        bins = _compute_reliability_bins(evals)
        assert len(bins) == 4  # alle 4 Bins belegt

        # Erste Bin [0.2-0.4): 1 Zeile
        assert bins[0]["n"] == 1
        # Letzte Bin [0.8-1.0]: 2 Zeilen
        assert bins[3]["n"] == 2

    def test_bins_empty_not_included(self):
        """Leere Bins werden nicht in die Liste aufgenommen."""
        evals = [
            _make_eval(confidence=5.0, hit=True),
            _make_eval(confidence=5.0, hit=False),
        ]
        bins = _compute_reliability_bins(evals)
        # Nur die [0.8-1.0] Bin ist belegt
        assert len(bins) == 1
        assert bins[0]["n"] == 2

    def test_bins_hit_rate(self):
        """Hit-Rate pro Bin wird korrekt berechnet."""
        evals = [
            _make_eval(confidence=4.0, hit=True),
            _make_eval(confidence=4.0, hit=True),
            _make_eval(confidence=4.0, hit=False),
        ]
        bins = _compute_reliability_bins(evals)
        assert len(bins) == 1
        assert bins[0]["hit_rate"] is not None
        assert math.isclose(bins[0]["hit_rate"], 2 / 3, rel_tol=1e-6)

    def test_bins_mittlere_konfidenz(self):
        """Mittlere Konfidenz pro Bin wird korrekt berechnet."""
        evals = [
            _make_eval(confidence=4.0, hit=True),   # p=0.8
            _make_eval(confidence=5.0, hit=True),   # p=1.0
        ]
        bins = _compute_reliability_bins(evals)
        assert len(bins) == 1
        # mittlere_konfidenz = (0.8 + 1.0) / 2 = 0.9
        assert math.isclose(bins[0]["mittlere_konfidenz"], 0.9, rel_tol=1e-6)

    def test_bins_none_when_no_valid(self):
        """Keine gültigen Zeilen → leere Liste."""
        evals = [_make_eval(confidence=None, hit=True)]
        bins = _compute_reliability_bins(evals)
        assert bins == []


# --------------------------------------------------------------------------- #
# Tests: _aggregate-Keys
# --------------------------------------------------------------------------- #


class TestAggregateKeys:
    """Testet dass _aggregate die neuen Keys liefert."""

    def test_aggregate_has_kalibrierung_keys(self):
        """_aggregate enthält konfidenz_kalibrierung und reliability_bins."""
        evals = [
            _make_eval(confidence=4.0, hit=True),
            _make_eval(confidence=3.0, hit=False),
        ]
        result = _aggregate(evals)
        assert "konfidenz_kalibrierung" in result
        assert "reliability_bins" in result
        assert isinstance(result["konfidenz_kalibrierung"], dict)
        assert isinstance(result["reliability_bins"], list)

    def test_aggregate_kalibrierung_has_all_fields(self):
        """konfidenz_kalibrierung hat alle erwarteten Felder."""
        evals = [_make_eval(confidence=4.0, hit=True)]
        result = _aggregate(evals)
        kal = result["konfidenz_kalibrierung"]
        for key in (
            "brier_score",
            "n",
            "durchschnittliche_konfidenz",
            "durchschnittliche_tatsaechliche_hit_rate",
            "kalibrierungs_gap",
            "tendenz",
        ):
            assert key in kal

    def test_empty_result_has_new_fields(self):
        """_empty_result enthält die neuen Felder mit None/[]-Defaults."""
        result = _empty_result()
        assert "konfidenz_kalibrierung" in result
        assert "reliability_bins" in result
        assert result["konfidenz_kalibrierung"]["brier_score"] is None
        assert result["konfidenz_kalibrierung"]["n"] == 0
        assert result["reliability_bins"] == []

    def test_aggregate_no_valid_evals_returns_none(self):
        """_aggregate ohne gültige confidence/hit → None-Werte."""
        evals = [
            _make_eval(confidence=None, hit=True),
            _make_eval(confidence=4.0, hit=None),
        ]
        result = _aggregate(evals)
        assert result["konfidenz_kalibrierung"]["brier_score"] is None
        assert result["reliability_bins"] == []


# --------------------------------------------------------------------------- #
# Tests: evaluate_journal Integration
# --------------------------------------------------------------------------- #


class TestEvaluateJournalIntegration:
    """Testet dass evaluate_journal die neuen Felder liefert."""

    def test_evaluate_journal_has_kalibrierung(self, tmp_path):
        """evaluate_journal liefert konfidenz_kalibrierung im Ergebnis."""
        rows = [
            _make_journal_row(ticker="AAA", action="KAUFEN", confidence="4"),
            _make_journal_row(ticker="BBB", action="KAUFEN", confidence="3"),
        ]
        path = _write_journal(tmp_path, rows)

        with patch(
            "concilium.evaluate._load_price_history",
            side_effect=lambda t, **kw: _make_prices(100, 60, drift=0.01),
        ):
            result = evaluate_journal(path)

        assert "konfidenz_kalibrierung" in result
        assert "reliability_bins" in result
        assert result["konfidenz_kalibrierung"]["n"] > 0

    def test_evaluate_journal_empty_has_none(self, tmp_path):
        """evaluate_journal mit leerem Journal → None-Werte für Kalibrierung."""
        path = _write_journal(tmp_path, [])
        result = evaluate_journal(path)
        assert result["konfidenz_kalibrierung"]["brier_score"] is None
        assert result["reliability_bins"] == []


# --------------------------------------------------------------------------- #
# Tests: Robustheit
# --------------------------------------------------------------------------- #


class TestRobustness:
    """Testet Robustheit bei fehlenden Werten."""

    def test_no_confidence_no_crash(self):
        """Keine confidence → Brier = None, kein Crash."""
        evals = [
            _make_eval(confidence=None, hit=True),
            _make_eval(confidence=None, hit=False),
        ]
        kal = _compute_konfidenz_kalibrierung(evals)
        assert kal["brier_score"] is None
        assert kal["n"] == 0

    def test_no_hit_no_crash(self):
        """Kein hit → Brier = None, kein Crash."""
        evals = [
            _make_eval(confidence=4.0, hit=None),
            _make_eval(confidence=3.0, hit=None),
        ]
        kal = _compute_konfidenz_kalibrierung(evals)
        assert kal["brier_score"] is None

    def test_zero_confidence_skipped(self):
        """confidence = 0 → wird übersprungen (conf <= 0 Guard)."""
        evals = [_make_eval(confidence=0.0, hit=True)]
        kal = _compute_konfidenz_kalibrierung(evals)
        assert kal["brier_score"] is None
        assert kal["n"] == 0

    def test_nan_confidence_skipped(self):
        """NaN confidence → wird übersprungen."""
        evals = [_make_eval(confidence=float("nan"), hit=True)]
        kal = _compute_konfidenz_kalibrierung(evals)
        assert kal["brier_score"] is None

    def test_reliability_bins_no_hit_all_none(self):
        """Reliability-Bins mit nur hit=None → leere Liste."""
        evals = [
            _make_eval(confidence=4.0, hit=None),
            _make_eval(confidence=3.0, hit=None),
        ]
        bins = _compute_reliability_bins(evals)
        assert bins == []


# --------------------------------------------------------------------------- #
# Tests: build_feedback_context mit Kalibrierungs-Zeile
# --------------------------------------------------------------------------- #


class TestFeedbackKalibrierung:
    """Testet dass build_feedback_context die Kalibrierungs-Zeile injiziert."""

    def test_feedback_contains_kalibrierung_line(self, tmp_path):
        """Feedback-Kontext enthält eine Konfidenz-Kalibrierung-Zeile."""
        rows = [
            _make_journal_row(ticker="A", confidence="4", final_decision="GENEHMIGT"),
            _make_journal_row(ticker="B", confidence="4", final_decision="GENEHMIGT"),
            _make_journal_row(ticker="C", confidence="5", final_decision="GENEHMIGT"),
            _make_journal_row(ticker="D", confidence="4", final_decision="ABGELEHNT"),
            _make_journal_row(ticker="E", confidence="4", final_decision="GENEHMIGT"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)
        assert "Konfidenz-Kalibrierung" in result

    def test_feedback_kalibrierung_no_network(self, tmp_path):
        """build_feedback_context lädt kein yfinance (netzfrei)."""
        rows = [_make_journal_row(ticker=f"T{i}", confidence="4") for i in range(5)]
        path = _write_journal(tmp_path, rows)
        # Wenn yfinance aufgerufen würde, würde der mock fehlschlagen
        with patch("concilium.feedback.realised_return_for_row") as mock_rr:
            result = build_feedback_context(path)
            assert mock_rr.call_count == 0  # kein Netzwerk-Aufruf
        assert "Konfidenz-Kalibrierung" in result

    def test_feedback_overconfident_tendenz(self, tmp_path):
        """Hohe Confidence, alle ABGELEHNT → überkonfident."""
        rows = [
            _make_journal_row(ticker=f"T{i}", confidence="5", final_decision="ABGELEHNT")
            for i in range(5)
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)
        assert "Konfidenz-Kalibrierung" in result
        assert "überkonfident" in result

    def test_feedback_well_calibrated(self, tmp_path):
        """Confidence ≈ Genehmigungs-Rate → gut kalibriert."""
        rows = [
            _make_journal_row(ticker="A", confidence="3", final_decision="GENEHMIGT"),
            _make_journal_row(ticker="B", confidence="3", final_decision="ABGELEHNT"),
            _make_journal_row(ticker="C", confidence="3", final_decision="GENEHMIGT"),
            _make_journal_row(ticker="D", confidence="3", final_decision="ABGELEHNT"),
            _make_journal_row(ticker="E", confidence="3", final_decision="GENEHMIGT"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)
        # avg_conf = 0.6, genehmigungs_rate = 0.6, gap = 0 → gut kalibriert
        assert "gut kalibriert" in result

    def test_feedback_no_kalibrierung_data(self, tmp_path):
        """Keine final_decision → Kalibrierung sagt 'zu wenige Daten'."""
        rows = [
            _make_journal_row(ticker=f"T{i}", confidence="4", final_decision="")
            for i in range(5)
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)
        assert "Konfidenz-Kalibrierung" in result
        assert "zu wenige Daten" in result

    def test_feedback_does_not_crash_on_missing_confidence(self, tmp_path):
        """Fehlende confidence → kein Crash, Kalibrierung graceful."""
        rows = [
            _make_journal_row(ticker=f"T{i}", confidence="", final_decision="GENEHMIGT")
            for i in range(5)
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)
        assert isinstance(result, str)
        assert "Konfidenz-Kalibrierung" in result


# --------------------------------------------------------------------------- #
# Tests: Report-Sektion
# --------------------------------------------------------------------------- #


class TestReportKalibrierung:
    """Testet die '## Konfidenz-Kalibrierung'-Sektion im Report."""

    def test_report_has_kalibrierung_section(self):
        """Report enthält '## Konfidenz-Kalibrierung' mit Brier-Score."""
        eval_result = {
            "anzahl_entscheidungen": 10,
            "nach_aktion": {
                "KAUFEN": {"n": 6, "hit_rate": 0.667, "avg_rendite": 3.5},
                "HALTEN": {"n": 2, "hit_rate": 0.5, "avg_rendite": 0.0},
                "VERKAUFEN": {"n": 2, "hit_rate": 0.5, "avg_rendite": -1.0},
            },
            "hit_rate_gesamt": 0.6,
            "durchschnitt_rendite_gesamt": 2.0,
            "zielkurs_trefferquote": 0.5,
            "stop_verletzungsquote": 0.2,
            "konfidenz_baende": [
                {"band": "hoch", "hit_rate": 0.8, "n": 5},
                {"band": "mittel", "hit_rate": 0.4, "n": 3},
                {"band": "niedrig", "hit_rate": 0.2, "n": 2},
            ],
            "portfolio_fit_hoch": {"hit_rate": 0.75, "n": 4},
            "zusammenfassung": None,
            "fehler": [],
            "konfidenz_kalibrierung": {
                "brier_score": 0.18,
                "n": 10,
                "durchschnittliche_konfidenz": 0.72,
                "durchschnittliche_tatsaechliche_hit_rate": 0.6,
                "kalibrierungs_gap": 0.12,
                "tendenz": "gut kalibriert",
            },
            "reliability_bins": [
                {"bin": "[0.4-0.6)", "n": 3, "mittlere_konfidenz": 0.5, "hit_rate": 0.33},
                {"bin": "[0.6-0.8)", "n": 4, "mittlere_konfidenz": 0.7, "hit_rate": 0.5},
                {"bin": "[0.8-1.0]", "n": 3, "mittlere_konfidenz": 0.9, "hit_rate": 0.67},
            ],
        }
        report = generate_track_record_report(eval_result)
        assert "## Konfidenz-Kalibrierung" in report
        assert "Brier-Score" in report
        assert "0.18" in report
        assert "Kalibrierungs-Gap" in report
        assert "gut kalibriert" in report
        assert "Reliability-Bänder" in report
        assert "[0.4-0.6)" in report
        assert "[0.8-1.0]" in report

    def test_report_empty_result_no_kalibrierung_section(self):
        """Leeres Ergebnis → keine Konfidenz-Kalibrierung-Sektion."""
        from concilium.evaluate import _empty_result

        report = generate_track_record_report(_empty_result())
        assert "## Konfidenz-Kalibrierung" not in report

    def test_report_kalibrierung_no_nan(self):
        """Report mit Kalibrierung enthält kein 'nan'."""
        eval_result = {
            "anzahl_entscheidungen": 5,
            "nach_aktion": {
                "KAUFEN": {"n": 5, "hit_rate": 0.8, "avg_rendite": 2.0},
                "HALTEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
                "VERKAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
            },
            "hit_rate_gesamt": 0.8,
            "durchschnitt_rendite_gesamt": 2.0,
            "zielkurs_trefferquote": None,
            "stop_verletzungsquote": None,
            "konfidenz_baende": [{"band": "hoch", "hit_rate": 0.8, "n": 5}],
            "portfolio_fit_hoch": None,
            "zusammenfassung": None,
            "fehler": [],
            "konfidenz_kalibrierung": {
                "brier_score": 0.08,
                "n": 5,
                "durchschnittliche_konfidenz": 0.8,
                "durchschnittliche_tatsaechliche_hit_rate": 0.8,
                "kalibrierungs_gap": 0.0,
                "tendenz": "gut kalibriert",
            },
            "reliability_bins": [
                {"bin": "[0.8-1.0]", "n": 5, "mittlere_konfidenz": 0.8, "hit_rate": 0.8},
            ],
        }
        report = generate_track_record_report(eval_result)
        assert "nan" not in report.lower()

    def test_report_kalibrierung_only_brier_no_bins(self):
        """Report mit nur Brier-Score (keine reliability_bins) → Sektion ohne Tabelle."""
        eval_result = {
            "anzahl_entscheidungen": 3,
            "nach_aktion": {
                "KAUFEN": {"n": 3, "hit_rate": 0.667, "avg_rendite": 1.0},
                "HALTEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
                "VERKAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
            },
            "hit_rate_gesamt": 0.667,
            "durchschnitt_rendite_gesamt": 1.0,
            "zielkurs_trefferquote": None,
            "stop_verletzungsquote": None,
            "konfidenz_baende": [{"band": "hoch", "hit_rate": 0.667, "n": 3}],
            "portfolio_fit_hoch": None,
            "zusammenfassung": None,
            "fehler": [],
            "konfidenz_kalibrierung": {
                "brier_score": 0.12,
                "n": 3,
                "durchschnittliche_konfidenz": 0.7,
                "durchschnittliche_tatsaechliche_hit_rate": 0.667,
                "kalibrierungs_gap": 0.033,
                "tendenz": "gut kalibriert",
            },
            "reliability_bins": [],
        }
        report = generate_track_record_report(eval_result)
        assert "## Konfidenz-Kalibrierung" in report
        assert "Brier-Score" in report
        # Keine Reliability-Bänder-Tabelle
        assert "Reliability-Bänder" not in report

    def test_report_kalibrierung_none_values_show_na(self):
        """Fehlende Kalibrierungs-Werte → N/A im Report."""
        eval_result = _empty_result()
        eval_result["anzahl_entscheidungen"] = 5
        eval_result["konfidenz_baende"] = [{"band": "hoch", "hit_rate": 0.5, "n": 5}]
        eval_result["konfidenz_kalibrierung"] = {
            "brier_score": None,
            "n": 0,
            "durchschnittliche_konfidenz": None,
            "durchschnittliche_tatsaechliche_hit_rate": None,
            "kalibrierungs_gap": None,
            "tendenz": None,
        }
        # Da brier_score None ist und reliability_bins leer → Sektion erscheint nicht
        report = generate_track_record_report(eval_result)
        assert "## Konfidenz-Kalibrierung" not in report
