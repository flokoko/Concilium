"""Tests für Phase 4: Entry-Timing via Limit-Orders (einstiegs_level).

Feature: Der Trader schlägt zusätzlich zum Zielkurs ein konkretes
Einstiegs-Level als Limit-Order-Preis vor (z.B. Support-Level wie SMA50 oder
Bollinger-Unterband). Das Feld ist optional (None wenn nicht gesetzt) und
ändert nichts an der Entscheidungs-/Positionslogik.

Testet:
1. schemas.py: einstiegs_level ist im TRADE_SCHEMA, optional (nicht required),
   Typ number|null, Schema-Validierung + Defaults.
2. agents.py: trader() reicht einstiegs_level durch (strukturierter Pfad);
   ohne Feld im LLM-Output bleibt es None (setdefault-Default); ensemble_trader()
   übernimmt es aus dem Basis-Run.
3. report.py: Trade-Sektion rendert "Einstiegs-Level (Limit-Order)" nur wenn
   gesetzt (None → keine Zeile).
4. journal.py: append_decision() schreibt die neue Spalte einstiegs_level
   (leer wenn nicht gesetzt, mit Header-Migration bestehender Dateien).

Alle Tests sind offline (kein Netzwerk).
"""

from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import ensemble_trader, trader  # noqa: E402
from concilium.journal import JOURNAL_HEADER, append_decision  # noqa: E402
from concilium.report import generate_report  # noqa: E402
from concilium.schemas import (  # noqa: E402
    TRADE_SCHEMA,
    _check_type,
    defaults_for_schema,
    validate_structured,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ANALYSTS = {
    "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
    "technicals": {"current_price": 100.0},
}

_DEBATE = {
    "bull": {"_raw": "Bull text", "argumente": "Bull argument"},
    "bear": {"_raw": "Bear text", "argumente": "Bear argument"},
    "bull_confidence": 4,
    "bear_confidence": 2,
}

_TRADE_JSON_MIT_LEVEL = json.dumps({
    "rolle": "Trader",
    "aktion": "KAUFEN",
    "zielkurs": 120,
    "stop_loss": 90,
    "einstiegs_level": 95.5,
    "positionsanteil": 5,
    "begründung": "Rücksetzer an SMA50 abwarten",
    "zeithorizont": "Mittelfristig",
})

_TRADE_JSON_OHNE_LEVEL = json.dumps({
    "rolle": "Trader",
    "aktion": "HALTEN",
    "zielkurs": None,
    "stop_loss": None,
    "positionsanteil": 0,
    "begründung": "Abwarten",
    "zeithorizont": "Kurzfristig",
})


class _StructuredMockLLM:
    """Mock-LLM: StructuredChatResult-Pfad (response_format_used=True)."""

    def __init__(self, response_json: str):
        self._response = response_json

    def chat(self, messages, temperature=0.3, **kwargs):
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult

            return StructuredChatResult(text=self._response, response_format_used=True)
        return self._response


# ===========================================================================
# 1. TRADE_SCHEMA
# ===========================================================================


class TestTradeSchemaEinstiegsLevel:
    """einstiegs_level im TRADE_SCHEMA: vorhanden, optional, korrekter Typ."""

    def test_feld_im_schema_vorhanden(self):
        """einstiegs_level ist als Property im TRADE_SCHEMA definiert."""
        props = TRADE_SCHEMA["json_schema"]["schema"]["properties"]
        assert "einstiegs_level" in props
        any_of_types = [sub.get("type") for sub in props["einstiegs_level"]["anyOf"]]
        assert any_of_types == ["number", "null"]

    def test_feld_nicht_required(self):
        """einstiegs_level ist optional — nicht in der required-Liste."""
        required = TRADE_SCHEMA["json_schema"]["schema"]["required"]
        assert "einstiegs_level" not in required

    def test_validierung_zahl_und_null_gueltig(self):
        """Schema-Validierung akzeptiert Zahl und null."""
        base = {
            "aktion": "KAUFEN", "zielkurs": 120, "stop_loss": 90,
            "positionsanteil": 5, "begründung": "Test",
        }
        assert validate_structured({**base, "einstiegs_level": 95.5}, TRADE_SCHEMA) == []
        assert validate_structured({**base, "einstiegs_level": None}, TRADE_SCHEMA) == []
        # Hinweis: validate_structured ist best-effort und lenient bei skalaren
        # anyOf-Feldern (bestehendes Verhalten, gilt auch für zielkurs/stop_loss).
        # Der Typ-Check selbst läuft über _check_type:
        assert not _check_type("95", "number")
        assert not _check_type(True, "number")
        assert _check_type(95.5, "number")
        assert _check_type(95, "number")

    def test_default_ist_none(self):
        """defaults_for_schema liefert für einstiegs_level den Default None."""
        defaults = defaults_for_schema(TRADE_SCHEMA)
        assert "einstiegs_level" in defaults
        assert defaults["einstiegs_level"] is None

    def test_backwards_compat_ohne_feld_gueltig(self):
        """Trade ohne einstiegs_level validiert weiterhin (Rückwärtskompatibilität)."""
        legacy = {
            "aktion": "KAUFEN", "zielkurs": 120, "stop_loss": 90,
            "positionsanteil": 5, "begründung": "Test",
        }
        assert validate_structured(legacy, TRADE_SCHEMA) == []


# ===========================================================================
# 2. agents.py — trader() / ensemble_trader()
# ===========================================================================


class TestTraderEinstiegsLevel:
    """trader() reicht einstiegs_level durch bzw. füllt es mit None auf."""

    def test_level_wird_durchgereicht(self):
        """trader() gibt einstiegs_level aus dem LLM-Output unverändert zurück."""
        llm = _StructuredMockLLM(_TRADE_JSON_MIT_LEVEL)
        result = trader(_ANALYSTS, _DEBATE, llm)
        assert result["einstiegs_level"] == 95.5
        assert result["aktion"] == "KAUFEN"

    def test_level_none_wenn_nicht_geliefert(self):
        """Ohne Feld im LLM-Output bleibt einstiegs_level None (setdefault-Default)."""
        llm = _StructuredMockLLM(_TRADE_JSON_OHNE_LEVEL)
        result = trader(_ANALYSTS, _DEBATE, llm)
        assert result["einstiegs_level"] is None
        assert result["aktion"] == "HALTEN"

    def test_ensemble_uebernimmt_level_aus_basis_run(self):
        """ensemble_trader() übernimmt einstiegs_level aus dem Mehrheits-Run."""
        llm = _StructuredMockLLM(_TRADE_JSON_MIT_LEVEL)
        result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)
        assert result["aktion"] == "KAUFEN"
        assert result["einstiegs_level"] == 95.5


# ===========================================================================
# 3. report.py — Trade-Sektion
# ===========================================================================


def _result_mit_level(einstiegs_level: float | None) -> dict:
    """Vollständiges Result-dict mit KAUFEN-Trade und optionalem Level."""
    trade: dict = {
        "aktion": "KAUFEN",
        "rating": "KAUFEN",
        "zielkurs": 120,
        "stop_loss": 90,
        "positionsanteil": 5,
        "begründung": "Test",
        "zeithorizont": "Mittelfristig",
    }
    if einstiegs_level is not None:
        trade["einstiegs_level"] = einstiegs_level
    return {
        "ticker": "AAPL",
        "no_llm": False,
        "data": {
            "fundamentals": {"name": "Apple", "sector": "Tech"},
            "technicals": {"current_price": 100},
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
            "bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"},
            "bull_confidence": 5, "bear_confidence": 4,
        },
        "trade": trade,
        "risk": {"risiko_score": 3, "empfehlung": "GENEHMIGT", "auflagen": "keine"},
        "final": {"entscheidung": "GENEHMIGT", "begründung": "Ok", "confidence": 4},
    }


class TestReportEinstiegsLevel:
    """generate_report rendert das Einstiegs-Level nur wenn gesetzt."""

    def test_level_wird_angezeigt(self):
        """Mit einstiegs_level erscheint 'Einstiegs-Level (Limit-Order)' im Report."""
        report = generate_report(_result_mit_level(95.5), reports_dir=None)
        assert "**Einstiegs-Level (Limit-Order):** 95.50" in report
        assert "**Positionsanteil:**" in report

    def test_kein_level_keine_zeile(self):
        """Ohne einstiegs_level (Feld fehlt oder None) erscheint keine Zeile."""
        report = generate_report(_result_mit_level(None), reports_dir=None)
        assert "Einstiegs-Level" not in report

    def test_level_auch_in_management_summary(self):
        """Die Management-Summary zeigt das Level im Urteil an."""
        report = generate_report(_result_mit_level(95.5), reports_dir=None)
        assert "Einstiegs-Level (Limit-Order): 95.50" in report


# ===========================================================================
# 4. journal.py — Journal-Spalte
# ===========================================================================


class TestJournalEinstiegsLevel:
    """append_decision() persistiert einstiegs_level als neue Spalte."""

    def test_spalte_im_header(self):
        """JOURNAL_HEADER enthält einstiegs_level (nach stop, vor position_pct)."""
        assert "einstiegs_level" in JOURNAL_HEADER
        assert JOURNAL_HEADER.index("einstiegs_level") == JOURNAL_HEADER.index("stop") + 1

    def test_level_wird_geschrieben(self):
        """Mit gesetztem Level landet der Wert in der CSV."""
        result = _result_mit_level(95.5)
        journal_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "_tmp_einstiegs_level_journal.csv")
        try:
            append_decision(result, journal_file=journal_file)
            with open(journal_file, encoding="utf-8") as fh:
                row = list(csv.DictReader(fh))[0]
            assert row["einstiegs_level"] == "95.5"
        finally:
            if os.path.isfile(journal_file):
                os.remove(journal_file)

    def test_leer_wenn_nicht_gesetzt(self):
        """Ohne Level bleibt die Spalte leer (Legacy-Verhalten unverändert)."""
        result = _result_mit_level(None)
        journal_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "_tmp_einstiegs_level_journal.csv")
        try:
            append_decision(result, journal_file=journal_file)
            with open(journal_file, encoding="utf-8") as fh:
                row = list(csv.DictReader(fh))[0]
            assert row["einstiegs_level"] == ""
        finally:
            if os.path.isfile(journal_file):
                os.remove(journal_file)
