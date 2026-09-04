"""Tests für den Track-Record-Report-Guard bei 0 bewertbaren Zeilen.

Bug: generate_track_record_report() erzeugte bei 0 bewertbaren Zeilen
(z. B. weil yfinance keine Kurse liefert → evaluate_journal überspringt
alles) nur den Header — der Report war still leer, ohne jeden Hinweis.

Fix: Guard, der einen deutschen Warnblock schreibt. Der Report enthält
damit IMMER mindestens Header + erklärenden Hinweis.

Alle Tests laufen offline (kein yfinance, kein Netzwerk).
"""

from __future__ import annotations

import os
import sys

# src zum Pfad hinzufügen
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

from concilium.report import generate_track_record_report  # noqa: E402

WARN_SCHLUESSEL = "Keine bewertbaren Entscheidungen"


class TestTrackRecordGuardLeer:
    """Guard: 0 bewertbare Zeilen → Warnblock statt still leerem Report."""

    def test_leeres_dict_erzeugt_warnblock(self):
        """generate_track_record_report({}) → Warnhinweis, nicht nur Header."""
        report = generate_track_record_report({})
        assert report.startswith("# Concilium Track-Record-Evaluierung")
        assert WARN_SCHLUESSEL in report
        # Mehr als nur Header: Report muss die Erklärung enthalten
        assert "Kursdaten (yfinance)" in report
        assert "Die Kennzahlen sind daher leer" in report
        # Report endet weiterhin mit dem Footer
        assert "Keine Anlageberatung" in report

    def test_anzahl_0_mit_uebersprungen_erzeugt_warnblock(self):
        """anzahl_entscheidungen=0, uebersprungen=5 → Warnblock nennt 0 von 5."""
        report = generate_track_record_report(
            {"anzahl_entscheidungen": 0, "uebersprungen": 5}
        )
        assert WARN_SCHLUESSEL in report
        assert "0 von 5" in report
        assert "Kursdaten (yfinance)" in report

    def test_anzahl_0_ohne_uebersprungen_key(self):
        """{}-ähnliches dict ohne uebersprungen-Key → Guard greift trotzdem."""
        report = generate_track_record_report({"anzahl_entscheidungen": 0})
        assert WARN_SCHLUESSEL in report
        assert "0 von 0" in report

    def test_guard_nicht_bei_normalen_ergebnissen(self):
        """Mit bewertbaren Entscheidungen (n>0) erscheint der Guard NICHT."""
        eval_result = {
            "anzahl_entscheidungen": 2,
            "hit_rate_gesamt": 0.5,
            "nach_aktion": {
                "KAUFEN": {"n": 2, "hit_rate": 0.5, "avg_rendite": 1.0,
                           "avg_confidence": 4.0},
                "HALTEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
                "VERKAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
            },
            "uebersprungen": 0,
        }
        report = generate_track_record_report(eval_result)
        assert WARN_SCHLUESSEL not in report

    def test_guard_bei_n0_ohne_teilmenge_hinweis(self):
        """Bei 0 bewertbaren Zeilen steht der Teilmenge-Hinweis NICHT doppelt."""
        report = generate_track_record_report(
            {"anzahl_entscheidungen": 0, "uebersprungen": 3}
        )
        assert WARN_SCHLUESSEL in report
        # Der alte "Teilmenge"-Hinweis ist im Guard-Fall ersetzt (nicht doppelt)
        assert "basieren auf einer Teilmenge" not in report
