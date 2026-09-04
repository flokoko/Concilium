"""Tests für die empfohlene Gewichtung in der Management-Summary.

Feature: Die empfohlene Gewichtung (Positionsanteil + Ziel-Gewichtung) wird
bereits in Sektion 6 (Trade-Vorschlag) und Sektion 8 (Portfolio-Fit) gezeigt,
aber nicht auf einen Blick in der Management-Summary. Diese Tests erzwingen
die Gewichtungs-Zeile zwischen Urteil und Score-Zeile.

Alle Tests sind offline (kein Netzwerk).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from concilium.report import generate_report  # noqa: E402

# --------------------------------------------------------------------------- #
# Hilfs-Builder für Fake-Result-Dicts
# --------------------------------------------------------------------------- #


def _base_result() -> dict:
    """Result mit GENEHMIGT + KAUFEN; Gewichtungsfelder pro Test überschreibbar."""
    return {
        "ticker": "AAPL",
        "no_llm": False,
        "data": {
            "fundamentals": {"name": "Apple", "sector": "Tech"},
            "technicals": {"current_price": 150},
            "sentiment": {},
        },
        "analysts": {
            "fundamental": {"stimmung": "bullish", "score": 4,
                            "zusammenfassung": "Gut", "_raw": ""},
            "technical": {"stimmung": "bullish", "score": 4,
                          "zusammenfassung": "Gut", "_raw": ""},
            "sentiment": {"stimmung": "neutral", "score": 3,
                          "zusammenfassung": "Ok", "_raw": ""},
        },
        "debate": {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}},
        "trade": {
            "aktion": "KAUFEN",
            "rating": "KAUFEN",
            "zielkurs": 340,
            "stop_loss": 285,
            "positionsanteil": 2.5,
            "begründung": "Test",
            "zeithorizont": "Mittelfristig",
        },
        "risk": {"risiko_score": 3, "empfehlung": "GENEHMIGT", "auflagen": "keine"},
        "portfolio_fit": {
            "portfolio_fit_score": 3,
            "ziel_gewichtung_pct": 1.2,
            "konzentrationsrisiko_bewertung": "Keine nennenswerte Konzentration.",
        },
        "final": {
            "entscheidung": "GENEHMIGT",
            "confidence": 4,
            "begründung": "Ok.",
        },
    }


def _summary(result: dict) -> str:
    """Extrahiert den Management-Summary-Abschnitt aus dem Report."""
    report = generate_report(result)
    return report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]


# --------------------------------------------------------------------------- #
# Gewichtungs-Zeile: Inhalt und Position
# --------------------------------------------------------------------------- #


class TestGewichtungZeile:
    """Die Gewichtungs-Zeile erscheint zwischen Urteil und Scores."""

    def test_beide_gewichtungen_angezeigt(self):
        """Positionsanteil + Ziel-Gewichtung → kombinierte Zeile."""
        summary = _summary(_base_result())
        assert (
            "**Empfohlene Gewichtung:** Positionsanteil 2.5 % · "
            "Ziel-Gewichtung 1.2 % des Portfolios" in summary
        )

    def test_nur_positionsanteil(self):
        """Nur positionsanteil gesetzt → nur der Positionsanteil-Teil."""
        result = _base_result()
        del result["portfolio_fit"]["ziel_gewichtung_pct"]
        summary = _summary(result)
        assert "**Empfohlene Gewichtung:** Positionsanteil 2.5 %" in summary
        assert "Ziel-Gewichtung" not in summary

    def test_nur_ziel_gewichtung(self):
        """Nur ziel_gewichtung_pct gesetzt → nur der Ziel-Gewichtung-Teil."""
        result = _base_result()
        result["trade"]["positionsanteil"] = None
        summary = _summary(result)
        assert (
            "**Empfohlene Gewichtung:** Ziel-Gewichtung 1.2 % des Portfolios"
            in summary
        )
        assert "Positionsanteil" not in summary

    def test_keine_gewichtungen_keine_zeile(self):
        """Kein Positionsanteil, keine Ziel-Gewichtung → Zeile bleibt weg."""
        result = _base_result()
        result["trade"]["positionsanteil"] = None
        del result["portfolio_fit"]["ziel_gewichtung_pct"]
        summary = _summary(result)
        assert "**Empfohlene Gewichtung:**" not in summary

    def test_gedaempft_hinweis_mit_original(self):
        """gedämpft=True + original → Hinweis wird an die Zeile angehängt."""
        result = _base_result()
        result["portfolio_fit"]["ziel_gewichtung_original"] = 2.0
        result["portfolio_fit"]["ziel_gewichtung_gedämpft"] = True
        summary = _summary(result)
        assert (
            "**Empfohlene Gewichtung:** Positionsanteil 2.5 % · "
            "Ziel-Gewichtung 1.2 % des Portfolios "
            "(nach Kalibrierung gedämpft, original 2.0)" in summary
        )

    def test_gedaempft_ohne_original_kein_hinweis(self):
        """gedämpft=True ohne original-Wert → kein Dämpfungs-Hinweis."""
        result = _base_result()
        result["portfolio_fit"]["ziel_gewichtung_gedämpft"] = True
        summary = _summary(result)
        assert "(nach Kalibrierung gedämpft" not in summary
        assert "Ziel-Gewichtung 1.2 % des Portfolios" in summary

    def test_position_nach_urteil_vor_scores(self):
        """Reihenfolge: Urteil → Empfohlene Gewichtung → Scores."""
        summary = _summary(_base_result())
        urteil_pos = summary.index("**Urteil:**")
        gewichtung_pos = summary.index("**Empfohlene Gewichtung:**")
        scores_pos = summary.index("**Scores:**")
        assert urteil_pos < gewichtung_pos < scores_pos


# --------------------------------------------------------------------------- #
# Robustheit: fehlende/leere/ungültige Werte
# --------------------------------------------------------------------------- #


class TestGewichtungRobustheit:
    """Crasht nie, auch bei fehlenden Sektionen oder NaN-Werten."""

    def test_none_sektionen_kein_crash(self):
        """Alle Sektionen None → kein Crash, keine Gewichtungs-Zeile."""
        result = {
            "ticker": "X",
            "no_llm": False,
            "data": {"fundamentals": {}, "technicals": {}, "sentiment": {}},
            "analysts": None,
            "debate": None,
            "trade": None,
            "risk": None,
            "portfolio_fit": None,
            "final": None,
        }
        report = generate_report(result)
        assert "## Management-Summary" in report
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "**Empfohlene Gewichtung:**" not in summary

    def test_leere_dicts_keine_zeile(self):
        """trade/portfolio_fit als leere dicts → keine Gewichtungs-Zeile."""
        result = _base_result()
        result["trade"] = {}
        result["portfolio_fit"] = {}
        summary = _summary(result)
        assert "**Empfohlene Gewichtung:**" not in summary

    def test_nan_werte_keine_zeile(self):
        """NaN-Werte → kein Crash, keine Gewichtungs-Zeile."""
        result = _base_result()
        result["trade"]["positionsanteil"] = float("nan")
        result["portfolio_fit"]["ziel_gewichtung_pct"] = float("nan")
        summary = _summary(result)
        assert "**Empfohlene Gewichtung:**" not in summary

    def test_nicht_numerische_werte_keine_zeile(self):
        """String-Werte (z.B. 'hoch' aus dem Portfolio-Fit) → Zeile bleibt weg."""
        result = _base_result()
        result["portfolio_fit"]["ziel_gewichtung_pct"] = "hoch"
        summary = _summary(result)
        assert "Ziel-Gewichtung" not in summary
