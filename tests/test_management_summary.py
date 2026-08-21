"""Tests für die Management-Summary als erste Report-Sektion.

Feature: Kompakte, deterministische Übersicht direkt nach dem Titel-Header
und vor dem Disclaimer. Kein LLM-Call — rein aus dem result-dict abgeleitet.

Alle Tests sind offline (kein Netzwerk).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.report import generate_report  # noqa: E402

# --------------------------------------------------------------------------- #
# Hilfs-Builder für Fake-Result-Dicts
# --------------------------------------------------------------------------- #

def _genehmigt_result() -> dict:
    """Vollständiges Result mit GENEHMIGT + KAUFEN."""
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
        "debate": {
            "bull": {"_raw": "Bull"},
            "bear": {"_raw": "Bear"},
            "bull_confidence": 5,
            "bear_confidence": 4,
        },
        "trade": {
            "aktion": "KAUFEN",
            "rating": "KAUFEN",
            "zielkurs": 340,
            "stop_loss": 285,
            "positionsanteil": 7,
            "begründung": "Test",
            "zeithorizont": "Mittelfristig",
        },
        "risk": {
            "risiko_score": 3,
            "empfehlung": "GENEHMIGT",
            "auflagen": "keine",
        },
        "portfolio_fit": {
            "portfolio_fit_score": 3,
            "konzentrationsrisiko_bewertung": "Keine nennenswerte Konzentration.",
        },
        "final": {
            "entscheidung": "GENEHMIGT",
            "confidence": 4,
            "begründung": "Solide Fundamentals und technisches Momentum.",
        },
    }


def _abgelehnt_result() -> dict:
    """Result mit ABGELEHNT."""
    return {
        "ticker": "XYZ",
        "no_llm": False,
        "data": {
            "fundamentals": {"name": "XYZ Corp", "sector": "Industrials"},
            "technicals": {"current_price": 50},
            "sentiment": {},
        },
        "analysts": {
            "fundamental": {"stimmung": "bearish", "score": 2, "_raw": ""},
            "technical": {"stimmung": "bearish", "score": 2, "_raw": ""},
            "sentiment": {"stimmung": "bearish", "score": 2, "_raw": ""},
        },
        "debate": {
            "bull": {"_raw": "Bull"},
            "bear": {"_raw": "Bear"},
        },
        "trade": {
            "aktion": "HALTEN",
            "zielkurs": None,
            "stop_loss": None,
            "positionsanteil": 0,
            "begründung": "Neutral",
            "zeithorizont": "Mittelfristig",
        },
        "risk": {"risiko_score": 5, "empfehlung": "ABGELEHNT"},
        "final": {
            "entscheidung": "ABGELEHNT",
            "confidence": 1,
            "begründung": "Risiko zu hoch.",
        },
    }


# --------------------------------------------------------------------------- #
# Position der Management-Summary im Report
# --------------------------------------------------------------------------- #

class TestManagementSummaryPosition:
    """Management-Summary erscheint nach Titel-Header, vor Disclaimer."""

    def test_summary_heading_present(self):
        report = generate_report(_genehmigt_result())
        assert "## Management-Summary" in report

    def test_summary_after_title_header(self):
        """Summary erscheint nach dem Titel-Header (Erstellt am / Sektor)."""
        report = generate_report(_genehmigt_result())
        header_pos = report.index("**Sektor:**")
        summary_pos = report.index("## Management-Summary")
        assert summary_pos > header_pos

    def test_summary_before_disclaimer(self):
        """Summary erscheint vor dem Disclaimer."""
        report = generate_report(_genehmigt_result())
        summary_pos = report.index("## Management-Summary")
        disclaimer_pos = report.index("Disclaimer")
        assert summary_pos < disclaimer_pos

    def test_summary_before_first_section(self):
        """Summary erscheint vor '## 1. Übersicht'."""
        report = generate_report(_genehmigt_result())
        summary_pos = report.index("## Management-Summary")
        uebersicht_pos = report.index("## 1. Übersicht")
        assert summary_pos < uebersicht_pos


# --------------------------------------------------------------------------- #
# 1. Gesamturteil (TL;DR)
# --------------------------------------------------------------------------- #

class TestManagementSummaryUrteil:
    """Gesamturteil-Zeile mit Emoji, Entscheidung und Trade-Aktion."""

    def test_genehmigt_shows_checkmark(self):
        report = generate_report(_genehmigt_result())
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "✅" in summary
        assert "GENEHMIGT" in summary

    def test_genehmigt_shows_trade_action_and_rating(self):
        """GENEHMIGT + KAUFEN → ✅, Trade: KAUFEN, Rating: KAUFEN."""
        report = generate_report(_genehmigt_result())
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Trade: KAUFEN" in summary
        assert "Rating: KAUFEN" in summary

    def test_genehmigt_shows_zielkurs_and_stop(self):
        """Zielkurs und Stop-Loss erscheinen in der Urteils-Zeile."""
        report = generate_report(_genehmigt_result())
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Zielkurs" in summary
        assert "Stop" in summary

    def test_abgelehnt_shows_x(self):
        report = generate_report(_abgelehnt_result())
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "❌" in summary
        assert "ABGELEHNT" in summary

    def test_modifiziert_shows_bolt(self):
        """MODIFIZIERT → ⚡."""
        result = _genehmigt_result()
        result["final"]["entscheidung"] = "MODIFIZIERT"
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "⚡" in summary
        assert "MODIFIZIERT" in summary

    def test_no_llm_shows_snapshot(self):
        """no_llm-Modus zeigt 'Datensnapshot (kein LLM)'."""
        result = {
            "ticker": "TEST",
            "no_llm": True,
            "data": {
                "fundamentals": {"name": "Test", "sector": "X"},
                "technicals": {"current_price": 100},
                "sentiment": {},
            },
        }
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Datensnapshot (kein LLM)" in summary


# --------------------------------------------------------------------------- #
# 2. Score-Zeile
# --------------------------------------------------------------------------- #

class TestManagementSummaryScores:
    """Score-Zeile mit Risiko, Portfolio-Fit, Debatte, Ensemble."""

    def test_scores_line_present(self):
        report = generate_report(_genehmigt_result())
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "**Scores:**" in summary

    def test_risiko_score_shown(self):
        report = generate_report(_genehmigt_result())
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Risiko 3/5" in summary

    def test_portfolio_fit_score_shown(self):
        report = generate_report(_genehmigt_result())
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Portfolio-Fit 3/5" in summary

    def test_debate_confidence_shown(self):
        report = generate_report(_genehmigt_result())
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Bull 5" in summary
        assert "Bear 4" in summary

    def test_ensemble_confidence_shown(self):
        """Ensemble-Konfidenz erscheint, wenn _ensemble im trade-dict."""
        result = _genehmigt_result()
        result["trade"]["_ensemble"] = {
            "runs": 5,
            "mehrheits_aktion": "KAUFEN",
            "ensemble_confidence": 0.67,
        }
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Ensemble-Konfidenz 67%" in summary

    def test_no_scores_when_missing(self):
        """Ohne risk/portfolio_fit/debate → keine Score-Zeile."""
        result = {
            "ticker": "TEST",
            "no_llm": False,
            "data": {"fundamentals": {}, "technicals": {}, "sentiment": {}},
            "analysts": {},
            "trade": {"aktion": "HALTEN"},
            "final": {"entscheidung": "GENEHMIGT", "begründung": "Ok"},
        }
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "**Scores:**" not in summary


# --------------------------------------------------------------------------- #
# 3. Kernrisiken
# --------------------------------------------------------------------------- #

class TestManagementSummaryKernrisiken:
    """Kernrisiken-Bullets: Auflagen, Portfolio-Fit, Konsistenz, Debatte."""

    def test_auflage_shown_when_present(self):
        """Risk-Auflagen erscheinen als Bullet, wenn nicht 'keine'."""
        result = _genehmigt_result()
        result["risk"]["auflagen"] = "Position auf 3% reduzieren"
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Position auf 3% reduzieren" in summary

    def test_auflage_keine_not_shown(self):
        """Auflagen='keine' → kein Bullet für die Auflagen selbst."""
        result = _genehmigt_result()
        result["risk"]["auflagen"] = "keine"
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        # Auflagen='keine' darf nicht als eigener Bullet auftauchen;
        # other risk bullets (z.B. Konzentrationsrisiko) sind erlaubt.
        bullets = [ln.strip() for ln in summary.split("\n") if ln.strip().startswith("- ")]
        auflagen_bullets = [b for b in bullets if b.lower() == "- keine"]
        assert len(auflagen_bullets) == 0

    def test_konzentrationsrisiko_shown(self):
        """portfolio_fit konzentrationsrisiko_bewertung erscheint als Bullet."""
        result = _genehmigt_result()
        result["portfolio_fit"]["konzentrationsrisiko_bewertung"] = (
            "Hohe Konzentration im Tech-Sektor."
        )
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Hohe Konzentration im Tech-Sektor." in summary

    def test_overlap_shown_when_no_konz(self):
        """Wenn konzentrationsrisiko=N/A aber sektor_overlap gesetzt → Overlap."""
        result = _genehmigt_result()
        result["portfolio_fit"]["konzentrationsrisiko_bewertung"] = "N/A"
        result["portfolio_fit"]["sektor_overlap_bewertung"] = (
            "Starke Überlagerung mit bestehenden Positionen."
        )
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Starke Überlagerung" in summary

    def test_konsistenz_warnung_shown(self):
        """Analysten-Konsistenz-Warnung erscheint als Bullet."""
        result = _genehmigt_result()
        result["analysts"]["fundamental"]["konsistenz_warnung"] = (
            "Konsistenz-Warnung: Stimmung='bullish' mit Score=1 ist inkonsistent."
        )
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Konsistenz-Warnung" in summary

    def test_bearish_debate_warning_shown(self):
        """Stark bearische Debatte (bear_conf >= bull_conf + 2) → Warnung."""
        result = _genehmigt_result()
        result["debate"]["bull_confidence"] = 2
        result["debate"]["bear_confidence"] = 5
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Debatte tendiert bearisch" in summary

    def test_no_bearish_warning_when_balanced(self):
        """Ausgewogene Debatte → keine bearische Warnung."""
        result = _genehmigt_result()
        result["debate"]["bull_confidence"] = 4
        result["debate"]["bear_confidence"] = 3
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Debatte tendiert bearisch" not in summary

    def test_no_risks_shows_placeholder(self):
        """Keine Risiken → 'Keine auffälligen Risiken erkannt.'"""
        result = {
            "ticker": "TEST",
            "no_llm": False,
            "data": {"fundamentals": {}, "technicals": {}, "sentiment": {}},
            "analysts": {},
            "trade": {"aktion": "KAUFEN"},
            "final": {"entscheidung": "GENEHMIGT", "begründung": "Ok"},
        }
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Keine auffälligen Risiken erkannt." in summary

    def test_max_four_risk_bullets(self):
        """Maximal 4 Risiko-Bullets."""
        result = _genehmigt_result()
        result["risk"]["auflagen"] = "Auflage 1"
        result["portfolio_fit"]["konzentrationsrisiko_bewertung"] = "Konz-Risiko 2"
        result["analysts"]["fundamental"]["konsistenz_warnung"] = "Warnung 3"
        result["analysts"]["technical"]["konsistenz_warnung"] = "Warnung 4"
        result["analysts"]["sentiment"]["konsistenz_warnung"] = "Warnung 5 (sollte fehlen)"
        result["debate"]["bull_confidence"] = 1
        result["debate"]["bear_confidence"] = 5
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        # Zähle Bullets (Zeilen die mit '- ' beginnen)
        bullets = [ln for ln in summary.split("\n") if ln.strip().startswith("- ")]
        assert len(bullets) <= 4


# --------------------------------------------------------------------------- #
# 4. Kurz-Begründung
# --------------------------------------------------------------------------- #

class TestManagementSummaryBegruendung:
    """Kurz-Begründung aus final['begründung']."""

    def test_begruendung_shown(self):
        report = generate_report(_genehmigt_result())
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "**Kurz-Begründung:**" in summary
        assert "Solide Fundamentals" in summary

    def test_begruendung_truncated(self):
        """Begründung wird auf ~200 Zeichen gekürzt."""
        result = _genehmigt_result()
        long_text = "A" * 300
        result["final"]["begründung"] = long_text
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        # Die Begründung darf nicht länger als ~210 Zeichen sein (200 + Präfix)
        begr_line = [ln for ln in summary.split("\n") if "Kurz-Begründung" in ln][0]
        assert len(begr_line) < 230

    def test_no_begruendung_when_empty(self):
        """Leere Begründung → Zeile fehlt."""
        result = _genehmigt_result()
        result["final"]["begründung"] = ""
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "**Kurz-Begründung:**" not in summary

    def test_no_begruendung_in_no_llm(self):
        """no_llm-Modus → keine Kurz-Begründung."""
        result = {
            "ticker": "TEST",
            "no_llm": True,
            "data": {"fundamentals": {}, "technicals": {}, "sentiment": {}},
        }
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "**Kurz-Begründung:**" not in summary


# --------------------------------------------------------------------------- #
# Robustheit: leeres Ergebnis
# --------------------------------------------------------------------------- #

class TestManagementSummaryRobustness:
    """Crasht nie, auch bei fehlenden/leeren Daten."""

    def test_empty_result_dict(self):
        """Leeres dict → crasht nicht, Summary erscheint."""
        report = generate_report({})
        assert "## Management-Summary" in report
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        # Sollte mindestens das Urteil zeigen (N/A oder Datensnapshot)
        assert "Urteil:" in summary or "Datensnapshot" in summary

    def test_result_with_none_values(self):
        """Result mit None-Werten → crasht nicht."""
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

    def test_no_portfolio_fit_no_risk_no_crash(self):
        """Kein risk, kein portfolio_fit → kompakte Summary ohne Crash."""
        result = {
            "ticker": "TEST",
            "no_llm": False,
            "data": {"fundamentals": {}, "technicals": {}, "sentiment": {}},
            "analysts": {},
            "trade": {"aktion": "KAUFEN"},
            "final": {"entscheidung": "GENEHMIGT", "begründung": "Ok"},
        }
        report = generate_report(result)
        summary = report.split("## Management-Summary")[1].split("## 1. Übersicht")[0]
        assert "Keine auffälligen Risiken erkannt." in summary
