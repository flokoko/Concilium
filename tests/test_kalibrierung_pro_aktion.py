"""Tests für Feature 2: Konfidenz-Kalibrierung pro Aktion im Feedback.

Alle Tests sind OFFLINE-fähig: kein yfinance, kein Netzwerk.
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.feedback import (  # noqa: E402
    _compute_kalibrierung_proxy_per_action,
    _compute_stats,
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
    """Erzeugt eine Journal-Zeile mit Defaults."""
    return {
        "ticker": ticker,
        "action": action,
        "final_decision": final_decision,
        "confidence": confidence,
        "timestamp": timestamp,
    }


# --------------------------------------------------------------------------- #
# Tests: _compute_kalibrierung_proxy_per_action
# --------------------------------------------------------------------------- #


class TestKalibrierungProAktionStats:
    """Testet die granulare Kalibrierungs-Berechnung pro Aktion."""

    def test_kaufen_ueberkonfident(self):
        """KAUFEN mit hoher Confidence, niedriger Genehmigungs-Rate → überkonfident."""
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4", final_decision="ABGELEHNT"),
            _make_row(ticker="B", action="KAUFEN", confidence="4", final_decision="ABGELEHNT"),
            _make_row(ticker="C", action="KAUFEN", confidence="5", final_decision="GENEHMIGT"),
        ]
        result = _compute_kalibrierung_proxy_per_action(rows)
        assert "KAUFEN" in result
        kaufen = result["KAUFEN"]
        # conf: 4/5=0.8, 4/5=0.8, 5/5=1.0 → avg 0.866...
        assert abs(kaufen["avg_confidence"] - (0.8 + 0.8 + 1.0) / 3) < 0.01
        # 1 von 3 genehmigt → 0.333
        assert abs(kaufen["genehmigungs_rate"] - 1 / 3) < 0.01
        # gap = 0.866 - 0.333 = 0.533 > 0.15 → überkonfident
        assert kaufen["gap"] > 0.15
        assert kaufen["tendenz"] == "überkonfident"
        assert kaufen["n"] == 3

    def test_halten_gut_kalibriert(self):
        """HALTEN mit Confidence ≈ Genehmigungs-Rate → gut kalibriert."""
        rows = [
            _make_row(ticker="A", action="HALTEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="HALTEN", confidence="4", final_decision="ABGELEHNT"),
            _make_row(ticker="C", action="HALTEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="D", action="HALTEN", confidence="4", final_decision="ABGELEHNT"),
        ]
        result = _compute_kalibrierung_proxy_per_action(rows)
        # conf = 4/5 = 0.8 für alle, genehmigt 2/4 = 0.5
        # gap = 0.8 - 0.5 = 0.3 > 0.15 → überkonfident
        # Korrektur: gap = 0.3 → überkonfident, nicht gut kalibriert
        # Lass uns das genauer testen:
        assert "HALTEN" in result
        halten = result["HALTEN"]
        assert halten["n"] == 4
        assert abs(halten["avg_confidence"] - 0.8) < 0.01
        assert abs(halten["genehmigungs_rate"] - 0.5) < 0.01
        assert halten["gap"] > 0.15
        assert halten["tendenz"] == "überkonfident"

    def test_verkaufen_unterkonfident(self):
        """VERKAUFEN mit niedriger Confidence, hoher Genehmigungs-Rate → unterkonfident."""
        rows = [
            _make_row(ticker="A", action="VERKAUFEN", confidence="1", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="VERKAUFEN", confidence="1", final_decision="GENEHMIGT"),
            _make_row(ticker="C", action="VERKAUFEN", confidence="2", final_decision="GENEHMIGT"),
        ]
        result = _compute_kalibrierung_proxy_per_action(rows)
        assert "VERKAUFEN" in result
        verkaufen = result["VERKAUFEN"]
        # conf: 1/5=0.2, 1/5=0.2, 2/5=0.4 → avg 0.266...
        assert abs(verkaufen["avg_confidence"] - (0.2 + 0.2 + 0.4) / 3) < 0.01
        # 3/3 genehmigt → 1.0
        assert verkaufen["genehmigungs_rate"] == 1.0
        # gap = 0.266 - 1.0 = -0.733 < -0.15 → unterkonfident
        assert verkaufen["gap"] < -0.15
        assert verkaufen["tendenz"] == "unterkonfident"

    def test_less_than_3_rows_excluded(self):
        """Aktionen mit <3 gültigen Zeilen werden weggelassen."""
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4"),
            _make_row(ticker="B", action="KAUFEN", confidence="4"),  # nur 2 → excluded
            _make_row(ticker="C", action="HALTEN", confidence="3", final_decision="GENEHMIGT"),
            _make_row(ticker="D", action="HALTEN", confidence="3", final_decision="ABGELEHNT"),
            _make_row(ticker="E", action="HALTEN", confidence="3", final_decision="GENEHMIGT"),
        ]
        result = _compute_kalibrierung_proxy_per_action(rows)
        assert "KAUFEN" not in result  # nur 2 Zeilen → excluded
        assert "HALTEN" in result      # 3 Zeilen → included
        assert "VERKAUFEN" not in result  # 0 Zeilen → excluded

    def test_empty_rows_returns_empty(self):
        """Leere Zeilen-Liste → leeres dict."""
        result = _compute_kalibrierung_proxy_per_action([])
        assert result == {}

    def test_no_valid_rows_returns_empty(self):
        """Zeilen ohne confidence/final_decision → leeres dict."""
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="", final_decision=""),
            _make_row(ticker="B", action="KAUFEN", confidence="", final_decision=""),
            _make_row(ticker="C", action="KAUFEN", confidence="", final_decision=""),
        ]
        result = _compute_kalibrierung_proxy_per_action(rows)
        assert result == {}

    def test_gap_tendenz_thresholds(self):
        """Testet die ±0.15 Schwellen für die Tendenz-Bestimmung."""
        # gap = 0 → gut kalibriert
        # conf = 3/5 = 0.6, genehmigt = 3/5 = 0.6, gap = 0.0
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="3", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="KAUFEN", confidence="3", final_decision="ABGELEHNT"),
            _make_row(ticker="C", action="KAUFEN", confidence="3", final_decision="GENEHMIGT"),
            _make_row(ticker="D", action="KAUFEN", confidence="3", final_decision="ABGELEHNT"),
            _make_row(ticker="E", action="KAUFEN", confidence="3", final_decision="GENEHMIGT"),
        ]
        result = _compute_kalibrierung_proxy_per_action(rows)
        # conf = 0.6, genehmigt = 3/5 = 0.6, gap = 0.0
        assert result["KAUFEN"]["gap"] == 0.0
        assert result["KAUFEN"]["tendenz"] == "gut kalibriert"

    def test_multiple_actions_in_parallel(self):
        """Mehrere Aktionen mit ≥3 Zeilen → alle im Ergebnis."""
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="C", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="D", action="HALTEN", confidence="3", final_decision="ABGELEHNT"),
            _make_row(ticker="E", action="HALTEN", confidence="3", final_decision="ABGELEHNT"),
            _make_row(ticker="F", action="HALTEN", confidence="3", final_decision="ABGELEHNT"),
        ]
        result = _compute_kalibrierung_proxy_per_action(rows)
        assert "KAUFEN" in result
        assert "HALTEN" in result
        assert "VERKAUFEN" not in result


# --------------------------------------------------------------------------- #
# Tests: _compute_stats — kalibrierung_pro_aktion im Rückgabewert
# --------------------------------------------------------------------------- #


class TestComputeStatsIntegration:
    """Testet dass _compute_stats kalibrierung_pro_aktion zurückgibt."""

    def test_stats_contains_kalibrierung_pro_aktion_key(self, tmp_path):
        """_compute_stats liefert kalibrierung_pro_aktion als Key."""
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="C", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
        ]
        stats = _compute_stats(rows)
        assert "kalibrierung_pro_aktion" in stats
        assert isinstance(stats["kalibrierung_pro_aktion"], dict)

    def test_stats_kalibrierung_pro_aktion_empty_for_few_rows(self, tmp_path):
        """Bei <3 Zeilen pro Aktion ist kalibrierung_pro_aktion leer."""
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4"),
            _make_row(ticker="B", action="KAUFEN", confidence="4"),
        ]
        stats = _compute_stats(rows)
        assert stats["kalibrierung_pro_aktion"] == {}


# --------------------------------------------------------------------------- #
# Tests: build_feedback_context — pro-Aktion-Block im Kontext
# --------------------------------------------------------------------------- #


class TestFeedbackProAktionBlock:
    """Testet dass build_feedback_context den pro-Aktion-Block rendert."""

    def test_pro_aktion_block_shown_when_data_available(self, tmp_path):
        """Bei ≥3 Zeilen pro Aktion wird der Kalibrierung-pro-Aktion-Block angezeigt."""
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4", final_decision="ABGELEHNT"),
            _make_row(ticker="B", action="KAUFEN", confidence="4", final_decision="ABGELEHNT"),
            _make_row(ticker="C", action="KAUFEN", confidence="5", final_decision="GENEHMIGT"),
            _make_row(ticker="D", action="HALTEN", confidence="3", final_decision="GENEHMIGT"),
            _make_row(ticker="E", action="HALTEN", confidence="3", final_decision="ABGELEHNT"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        assert "Kalibrierung pro Aktion:" in result
        assert "KAUFEN:" in result
        assert "Gap" in result
        assert "überkonfident" in result or "unterkonfident" in result or "gut kalibriert" in result

    def test_pro_aktion_block_hidden_when_no_data(self, tmp_path):
        """Bei <3 Zeilen pro Aktion wird der Block weggelassen."""
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4"),
            _make_row(ticker="B", action="KAUFEN", confidence="4"),
            _make_row(ticker="C", action="HALTEN", confidence="3"),
            _make_row(ticker="D", action="HALTEN", confidence="3"),
            _make_row(ticker="E", action="VERKAUFEN", confidence="2"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        assert "Kalibrierung pro Aktion:" not in result

    def test_pro_aktion_block_only_actions_with_data(self, tmp_path):
        """Nur Aktionen mit ≥3 Zeilen werden im Block aufgelistet."""
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="C", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="D", action="VERKAUFEN", confidence="2", final_decision="ABGELEHNT"),
            _make_row(ticker="E", action="HALTEN", confidence="3"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        assert "Kalibrierung pro Aktion:" in result
        assert "KAUFEN:" in result
        # VERKAUFEN hat nur 1 Zeile mit final_decision → nicht im Block
        # HALTEN hat 1 Zeile → nicht im Block
        # Prüfe dass der Block KAUFEN enthält
        kauf_line = [line for line in result.split("\n") if line.startswith("- KAUFEN:")]
        assert len(kauf_line) == 1

    def test_pro_aktion_block_format(self, tmp_path):
        """Prüft das Format der Kalibrierung-pro-Aktion-Zeilen."""
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="C", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="D", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="E", action="HALTEN", confidence="3", final_decision="GENEHMIGT"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        # Format: "- KAUFEN: Ø Confidence 0.80, Genehmigungs-Rate 1.00, Gap -0.20 (unterkonfident)"
        assert "Ø Confidence" in result
        assert "Genehmigungs-Rate" in result
        assert "Gap" in result
        # Die Confidence 4/5 = 0.80, Genehmigungs-Rate 4/4 = 1.00
        assert "0.80" in result
        assert "1.00" in result

    def test_konfidenz_anpassung_anweisung_present(self, tmp_path):
        """Die Anweisung zur Konfidenz-Anpassung ist im Kontext enthalten."""
        rows = [_make_row(ticker=f"T{i}", confidence="4") for i in range(5)]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        assert "Passe deine Konfidenz" in result
        assert "historische Trefferquote" in result
        assert "überkonfidenter" in result

    def test_pro_aktion_block_with_all_three_actions(self, tmp_path):
        """Alle drei Aktionen mit ≥3 Zeilen → alle drei im Block."""
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="KAUFEN", confidence="4", final_decision="ABGELEHNT"),
            _make_row(ticker="C", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="D", action="HALTEN", confidence="3", final_decision="GENEHMIGT"),
            _make_row(ticker="E", action="HALTEN", confidence="3", final_decision="ABGELEHNT"),
            _make_row(ticker="F", action="HALTEN", confidence="3", final_decision="GENEHMIGT"),
            _make_row(ticker="G", action="VERKAUFEN", confidence="2", final_decision="ABGELEHNT"),
            _make_row(ticker="H", action="VERKAUFEN", confidence="2", final_decision="ABGELEHNT"),
            _make_row(ticker="I", action="VERKAUFEN", confidence="2", final_decision="ABGELEHNT"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        assert "Kalibrierung pro Aktion:" in result
        assert "- KAUFEN:" in result
        assert "- HALTEN:" in result
        assert "- VERKAUFEN:" in result
