"""Tests für Track-Record-Transparenz + Retry und segmentierte Brier-Scores.

Feature 1: uebersprungen-Zählung und Retry bei fehlenden Kursdaten.
Feature 2: _compute_konfidenz_kalibrierung_segmentiert (pro Aktion / pro Rating).

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
    _compute_konfidenz_kalibrierung_segmentiert,
    _empty_result,
    evaluate_journal,
)
from concilium.report import generate_track_record_report  # noqa: E402

# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #


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
    from concilium.journal import JOURNAL_HEADER

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
    timestamp: str = "",
    rating: str = "",
) -> dict:
    """Erzeugt eine Journal-Zeile mit Defaults."""
    if not timestamp:
        timestamp = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "ticker": ticker,
        "action": action,
        "confidence": confidence,
        "timestamp": timestamp,
        "rating": rating,
    }


def _make_eval(
    confidence: float | None,
    hit: bool | None,
    action: str = "KAUFEN",
    rating: str = "",
) -> dict:
    """Erzeugt ein Einzel-Ergebnis-dict wie aus _evaluate_single."""
    return {
        "hit": hit,
        "rendite_pct": 1.0 if hit else -1.0,
        "ziel_erreicht": None,
        "stop_gerissen": None,
        "action": action,
        "rating": rating,
        "rating_distance": None,
        "confidence": confidence,
        "portfolio_fit_score": None,
        "ticker": "TEST",
        "timestamp": "2026-01-01 10:00:00",
    }


# --------------------------------------------------------------------------- #
# Feature 1: uebersprungen-Zählung
# --------------------------------------------------------------------------- #


class TestUebersprungenZaehlung:
    """Testet die uebersprungen-Kennzahl."""

    def test_uebersprungen_in_empty_result(self):
        """_empty_result enthält 'uebersprungen': 0."""
        result = _empty_result()
        assert "uebersprungen" in result
        assert result["uebersprungen"] == 0

    def test_uebersprungen_counted_when_no_price_data(self, tmp_path):
        """Ticker ohne Kursdaten (Retry schlägt fehl) → uebersprungen = 1."""
        rows = [
            _make_journal_row(ticker="NODATA", action="KAUFEN"),
        ]
        path = _write_journal(tmp_path, rows)

        with patch("concilium.evaluate._load_price_history", return_value=None):
            result = evaluate_journal(path)

        assert result["uebersprungen"] == 1
        assert result["anzahl_entscheidungen"] == 0
        assert len(result["fehler"]) == 1
        assert "NODATA" in result["fehler"][0]

    def test_uebersprungen_multiple_ticker(self, tmp_path):
        """Mehrere Ticker ohne Daten → uebersprungen zählt alle."""
        rows = [
            _make_journal_row(ticker="FAIL1", action="KAUFEN"),
            _make_journal_row(ticker="FAIL2", action="HALTEN"),
            _make_journal_row(ticker="FAIL3", action="VERKAUFEN"),
        ]
        path = _write_journal(tmp_path, rows)

        with patch("concilium.evaluate._load_price_history", return_value=None):
            result = evaluate_journal(path)

        assert result["uebersprungen"] == 3
        assert result["anzahl_entscheidungen"] == 0

    def test_uebersprungen_mixed(self, tmp_path):
        """Ein Ticker ok, zwei ohne Daten → uebersprungen = 2, anzahl = 1."""
        rows = [
            _make_journal_row(ticker="FAIL1", action="KAUFEN"),
            _make_journal_row(ticker="OK", action="KAUFEN", confidence="4"),
            _make_journal_row(ticker="FAIL2", action="KAUFEN"),
        ]
        path = _write_journal(tmp_path, rows)

        def mock_load(ticker, *, lookback_days=90):
            if ticker == "OK":
                return _make_prices(100, 60, drift=0.005)
            return None

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = evaluate_journal(path)

        assert result["uebersprungen"] == 2
        assert result["anzahl_entscheidungen"] == 1


# --------------------------------------------------------------------------- #
# Feature 1: Retry-Mechanismus
# --------------------------------------------------------------------------- #


class TestRetryMechanismus:
    """Testet den Retry bei fehlenden Kursdaten."""

    def test_retry_succeeds_second_call(self, tmp_path):
        """Erster Call None, zweiter Call (Retry) liefert Daten → nicht übersprungen."""
        rows = [
            _make_journal_row(ticker="AAPL", action="KAUFEN", confidence="4"),
        ]
        path = _write_journal(tmp_path, rows)

        call_count = {"n": 0}

        def mock_load(ticker, *, lookback_days=90):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return None  # Erster Call schlägt fehl
            return _make_prices(100, 60, drift=0.01)  # Retry liefert Daten

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            with patch("concilium.evaluate._delete_price_cache") as mock_del:
                result = evaluate_journal(path)

        assert result["uebersprungen"] == 0
        assert result["anzahl_entscheidungen"] == 1
        assert call_count["n"] == 2  # Zwei Calls: initial + retry
        # Cache-Löschung wurde für den Retry aufgerufen
        mock_del.assert_called_once_with("AAPL")

    def test_retry_fails_both_calls(self, tmp_path):
        """Erster und zweiter Call (Retry) schlagen fehl → übersprungen."""
        rows = [
            _make_journal_row(ticker="NOPE", action="KAUFEN"),
        ]
        path = _write_journal(tmp_path, rows)

        with patch("concilium.evaluate._load_price_history", return_value=None):
            with patch("concilium.evaluate._delete_price_cache"):
                result = evaluate_journal(path)

        assert result["uebersprungen"] == 1
        assert result["anzahl_entscheidungen"] == 0
        assert "auch nach Retry" in result["fehler"][0]

    def test_retry_not_triggered_when_first_succeeds(self, tmp_path):
        """Erster Call liefert Daten → kein Retry, nicht übersprungen."""
        rows = [
            _make_journal_row(ticker="AAPL", action="KAUFEN", confidence="4"),
        ]
        path = _write_journal(tmp_path, rows)

        call_count = {"n": 0}

        def mock_load(ticker, *, lookback_days=90):
            call_count["n"] += 1
            return _make_prices(100, 60, drift=0.01)

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            with patch("concilium.evaluate._delete_price_cache") as mock_del:
                result = evaluate_journal(path)

        assert result["uebersprungen"] == 0
        assert result["anzahl_entscheidungen"] == 1
        assert call_count["n"] == 1  # Nur ein Call — kein Retry nötig
        mock_del.assert_not_called()

    def test_retry_same_ticker_not_repeated(self, tmp_path):
        """Bei mehreren Zeilen mit gleichem Ticker wird der Retry nur 1× pro
        Ticker-Cache-Miss ausgelöst (price_cache wird gesetzt)."""
        rows = [
            _make_journal_row(ticker="AAPL", action="KAUFEN", confidence="4"),
            _make_journal_row(ticker="AAPL", action="HALTEN", confidence="3"),
        ]
        path = _write_journal(tmp_path, rows)

        call_count = {"n": 0}

        def mock_load(ticker, *, lookback_days=90):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return None  # Erster Call für AAPL schlägt fehl
            return _make_prices(100, 60, drift=0.01)  # Retry für AAPL

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            with patch("concilium.evaluate._delete_price_cache"):
                result = evaluate_journal(path)

        assert result["uebersprungen"] == 0
        assert result["anzahl_entscheidungen"] == 2
        # 2 Calls: 1× initial (fail) + 1× retry (success) → gecacht für 2. Zeile
        assert call_count["n"] == 2


# --------------------------------------------------------------------------- #
# Feature 2: Segmentierte Brier-Scores
# --------------------------------------------------------------------------- #


class TestSegmentierteBrierScores:
    """Testet _compute_konfidenz_kalibrierung_segmentiert."""

    def test_empty_result_has_segmentiert(self):
        """_empty_result enthält konfidenz_kalibrierung_segmentiert."""
        result = _empty_result()
        assert "konfidenz_kalibrierung_segmentiert" in result
        assert result["konfidenz_kalibrierung_segmentiert"] == {
            "nach_aktion": {},
            "nach_rating": {},
        }

    def test_aggregate_has_segmentiert(self):
        """_aggregate liefert konfidenz_kalibrierung_segmentiert."""
        evals = [
            _make_eval(confidence=4.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=3.0, hit=False, action="HALTEN"),
        ]
        result = _aggregate(evals)
        assert "konfidenz_kalibrierung_segmentiert" in result
        seg = result["konfidenz_kalibrierung_segmentiert"]
        assert "nach_aktion" in seg
        assert "nach_rating" in seg

    def test_segments_by_action(self):
        """Segmentierung nach Aktion liefert korrekte Werte."""
        evals = [
            _make_eval(confidence=5.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=5.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=4.0, hit=False, action="VERKAUFEN"),
            _make_eval(confidence=4.0, hit=False, action="VERKAUFEN"),
        ]
        seg = _compute_konfidenz_kalibrierung_segmentiert(evals)
        nach_aktion = seg["nach_aktion"]

        # KAUFEN: 2 Zeilen, beide hit, p=1.0 → brier=0
        assert "KAUFEN" in nach_aktion
        kauf = nach_aktion["KAUFEN"]
        assert kauf["n"] == 2
        assert math.isclose(kauf["brier_score"], 0.0, abs_tol=1e-6)
        assert math.isclose(kauf["durchschnittliche_konfidenz"], 1.0, abs_tol=1e-6)
        assert math.isclose(kauf["durchschnittliche_tatsaechliche_hit_rate"], 1.0, abs_tol=1e-6)

        # VERKAUFEN: 2 Zeilen, beide miss, p=0.8, hit=0 → brier=0.64
        assert "VERKAUFEN" in nach_aktion
        verk = nach_aktion["VERKAUFEN"]
        assert verk["n"] == 2
        assert math.isclose(verk["brier_score"], 0.64, abs_tol=1e-6)
        assert verk["tendenz"] == "überkonfident"

        # HALTEN hat keine Zeilen → nicht in nach_aktion
        assert "HALTEN" not in nach_aktion

    def test_segments_by_rating(self):
        """Segmentierung nach Rating liefert korrekte Werte."""
        evals = [
            _make_eval(confidence=5.0, hit=True, action="KAUFEN", rating="STARK KAUFEN"),
            _make_eval(confidence=5.0, hit=True, action="KAUFEN", rating="STARK KAUFEN"),
            _make_eval(confidence=3.0, hit=False, action="HALTEN", rating="HALTEN"),
        ]
        seg = _compute_konfidenz_kalibrierung_segmentiert(evals)
        nach_rating = seg["nach_rating"]

        assert "STARK KAUFEN" in nach_rating
        sk = nach_rating["STARK KAUFEN"]
        assert sk["n"] == 2
        assert math.isclose(sk["brier_score"], 0.0, abs_tol=1e-6)

        assert "HALTEN" in nach_rating
        ha = nach_rating["HALTEN"]
        assert ha["n"] == 1
        # p=0.6, hit=0 → brier=0.36
        assert math.isclose(ha["brier_score"], 0.36, abs_tol=1e-6)

        # Ratings ohne Zeilen → nicht vorhanden
        assert "KAUFEN" not in nach_rating
        assert "VERKAUFEN" not in nach_rating
        assert "STARK VERKAUFEN" not in nach_rating

    def test_empty_segments_omitted(self):
        """Leere Segmente (n=0) werden weggelassen."""
        evals = [
            _make_eval(confidence=4.0, hit=True, action="KAUFEN", rating="KAUFEN"),
        ]
        seg = _compute_konfidenz_kalibrierung_segmentiert(evals)
        nach_aktion = seg["nach_aktion"]
        nach_rating = seg["nach_rating"]

        # Nur KAUFEN hat Daten, HALTEN und VERKAUFEN fehlen
        assert "KAUFEN" in nach_aktion
        assert "HALTEN" not in nach_aktion
        assert "VERKAUFEN" not in nach_aktion

        # Nur Rating KAUFEN hat Daten
        assert "KAUFEN" in nach_rating
        assert "STARK KAUFEN" not in nach_rating

    def test_no_valid_evals_returns_empty(self):
        """Keine gültigen Evaluations → leere dicts."""
        evals = [
            _make_eval(confidence=None, hit=True, action="KAUFEN"),
            _make_eval(confidence=4.0, hit=None, action="KAUFEN"),
        ]
        seg = _compute_konfidenz_kalibrierung_segmentiert(evals)
        assert seg == {"nach_aktion": {}, "nach_rating": {}}

    def test_segment_brier_formula_matches_overall(self):
        """Segment-Brier für eine Aktion stimmt mit Gesamt-Brier für dieselbe
        Teilmenge überein."""
        evals = [
            _make_eval(confidence=5.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=4.0, hit=False, action="KAUFEN"),
            _make_eval(confidence=3.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=4.0, hit=False, action="VERKAUFEN"),
        ]
        seg = _compute_konfidenz_kalibrierung_segmentiert(evals)
        kauf_seg = seg["nach_aktion"]["KAUFEN"]

        # Handberechnung für KAUFEN: p = [1.0, 0.8, 0.6], hit = [1, 0, 1]
        # brier = [(1-1)^2, (0.8-0)^2, (0.6-1)^2] = [0, 0.64, 0.16] → Ø = 0.8/3
        expected_brier = (0.0 + 0.64 + 0.16) / 3
        assert math.isclose(kauf_seg["brier_score"], expected_brier, rel_tol=1e-6)
        assert kauf_seg["n"] == 3


# --------------------------------------------------------------------------- #
# Feature 1+2: Report-Warnung und Segment-Tabellen
# --------------------------------------------------------------------------- #


class TestReportTransparenz:
    """Testet die Report-Warnung bei uebersprungen > 0."""

    def test_report_warnung_bei_uebersprungen(self):
        """Report zeigt Warnung wenn uebersprungen > 0."""
        eval_result = _empty_result()
        eval_result["anzahl_entscheidungen"] = 1
        eval_result["uebersprungen"] = 32
        report = generate_track_record_report(eval_result)

        assert "⚠️ **Hinweis:**" in report
        assert "32 von 33" in report
        assert "übersprungen" in report
        assert "Teilmenge" in report

    def test_report_keine_warnung_bei_null_uebersprungen(self):
        """Report zeigt keine Warnung wenn uebersprungen = 0."""
        eval_result = _empty_result()
        eval_result["anzahl_entscheidungen"] = 10
        eval_result["uebersprungen"] = 0
        report = generate_track_record_report(eval_result)

        assert "Hinweis:" not in report
        assert "übersprungen" not in report.lower()

    def test_report_warnung_position_vor_uebersicht(self):
        """Warnung steht VOR der Übersichtstabelle."""
        eval_result = _empty_result()
        eval_result["anzahl_entscheidungen"] = 1
        eval_result["uebersprungen"] = 5
        report = generate_track_record_report(eval_result)

        warnung_pos = report.find("⚠️ **Hinweis:**")
        uebersicht_pos = report.find("## Übersicht")
        assert warnung_pos != -1
        assert uebersicht_pos != -1
        assert warnung_pos < uebersicht_pos

    def test_report_segment_tabellen_aktion(self):
        """Report zeigt '### Nach Aktion' Tabelle bei segmentierten Daten."""
        eval_result = _empty_result()
        eval_result["anzahl_entscheidungen"] = 3
        eval_result["konfidenz_kalibrierung"] = {
            "brier_score": 0.15,
            "n": 3,
            "durchschnittliche_konfidenz": 0.8,
            "durchschnittliche_tatsaechliche_hit_rate": 0.67,
            "kalibrierungs_gap": 0.13,
            "tendenz": "gut kalibriert",
        }
        eval_result["konfidenz_kalibrierung_segmentiert"] = {
            "nach_aktion": {
                "KAUFEN": {
                    "brier_score": 0.10,
                    "n": 2,
                    "durchschnittliche_konfidenz": 0.9,
                    "durchschnittliche_tatsaechliche_hit_rate": 0.5,
                    "kalibrierungs_gap": 0.4,
                    "tendenz": "überkonfident",
                },
            },
            "nach_rating": {},
        }
        report = generate_track_record_report(eval_result)

        assert "### Nach Aktion" in report
        assert "KAUFEN" in report
        assert "0.10" in report

    def test_report_segment_tabellen_rating(self):
        """Report zeigt '### Nach Rating' Tabelle bei segmentierten Daten."""
        eval_result = _empty_result()
        eval_result["anzahl_entscheidungen"] = 3
        eval_result["konfidenz_kalibrierung"] = {
            "brier_score": 0.15,
            "n": 3,
            "durchschnittliche_konfidenz": 0.8,
            "durchschnittliche_tatsaechliche_hit_rate": 0.67,
            "kalibrierungs_gap": 0.13,
            "tendenz": "gut kalibriert",
        }
        eval_result["konfidenz_kalibrierung_segmentiert"] = {
            "nach_aktion": {},
            "nach_rating": {
                "STARK KAUFEN": {
                    "brier_score": 0.05,
                    "n": 2,
                    "durchschnittliche_konfidenz": 1.0,
                    "durchschnittliche_tatsaechliche_hit_rate": 1.0,
                    "kalibrierungs_gap": 0.0,
                    "tendenz": "gut kalibriert",
                },
            },
        }
        report = generate_track_record_report(eval_result)

        assert "### Nach Rating" in report
        assert "STARK KAUFEN" in report
        assert "0.05" in report

    def test_report_keine_segment_tabellen_bei_leer(self):
        """Report zeigt keine Segment-Tabellen bei leeren Segmenten."""
        eval_result = _empty_result()
        eval_result["anzahl_entscheidungen"] = 3
        eval_result["konfidenz_kalibrierung"] = {
            "brier_score": 0.15,
            "n": 3,
            "durchschnittliche_konfidenz": 0.8,
            "durchschnittliche_tatsaechliche_hit_rate": 0.67,
            "kalibrierungs_gap": 0.13,
            "tendenz": "gut kalibriert",
        }
        eval_result["konfidenz_kalibrierung_segmentiert"] = {
            "nach_aktion": {},
            "nach_rating": {},
        }
        report = generate_track_record_report(eval_result)

        assert "### Nach Aktion" not in report
        assert "### Nach Rating" not in report

    def test_report_warnung_und_segmente_zusammen(self):
        """Report zeigt sowohl Warnung als auch Segment-Tabellen."""
        eval_result = _empty_result()
        eval_result["anzahl_entscheidungen"] = 1
        eval_result["uebersprungen"] = 32
        eval_result["konfidenz_kalibrierung"] = {
            "brier_score": 0.0,
            "n": 1,
            "durchschnittliche_konfidenz": 1.0,
            "durchschnittliche_tatsaechliche_hit_rate": 1.0,
            "kalibrierungs_gap": 0.0,
            "tendenz": "gut kalibriert",
        }
        eval_result["konfidenz_kalibrierung_segmentiert"] = {
            "nach_aktion": {
                "KAUFEN": {
                    "brier_score": 0.0,
                    "n": 1,
                    "durchschnittliche_konfidenz": 1.0,
                    "durchschnittliche_tatsaechliche_hit_rate": 1.0,
                    "kalibrierungs_gap": 0.0,
                    "tendenz": "gut kalibriert",
                },
            },
            "nach_rating": {
                "KAUFEN": {
                    "brier_score": 0.0,
                    "n": 1,
                    "durchschnittliche_konfidenz": 1.0,
                    "durchschnittliche_tatsaechliche_hit_rate": 1.0,
                    "kalibrierungs_gap": 0.0,
                    "tendenz": "gut kalibriert",
                },
            },
        }
        report = generate_track_record_report(eval_result)

        assert "32 von 33" in report
        assert "### Nach Aktion" in report
        assert "### Nach Rating" in report
