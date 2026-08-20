"""Entscheidungs-Journal — protokolliert jede LLM-Entscheidung in einer CSV-Datei.

Schreibt Einträge an journal/decisions.csv (relativ zum Arbeitsverzeichnis).
Robust: crasht niemals, legt fehlende Ordner an, schreibt leer bei fehlenden Daten.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Spalten des CSV-Journals
JOURNAL_HEADER = [
    "timestamp",
    "ticker",
    "action",
    "target",
    "stop",
    "position_pct",
    "final_decision",
    "confidence",
    "bull_confidence",
    "bear_confidence",
]


def _parse_confidence_from_debate(agent: dict[str, Any]) -> str:
    """Versucht, die confidence aus dem JSON-Preamble eines Bull/Bear-Agenten zu extrahieren.

    Die Bull/Bear-Agenten geben einen JSON-Block wie {"confidence": 4, "name": "..."}
    gefolgt von Fließtext zurück.
    """
    raw = str(agent.get("_raw", ""))
    if not raw:
        return ""
    # JSON-Block am Anfang suchen
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            conf = data.get("confidence")
            if conf is not None:
                return str(conf)
        except (json.JSONDecodeError, TypeError):
            pass
    return ""


def append_decision(
    result: dict[str, Any],
    *,
    journal_dir: str | None = None,
    journal_file: str | None = None,
) -> None:
    """Schreibt einen Entscheidungs-Eintrag in die CSV-Journal-Datei.

    Args:
        result: Das Ergebnis-dict aus run_pipeline.
        journal_dir: Optionaler Pfad für das Journal-Verzeichnis (für Tests).
        journal_file: Optionaler vollständiger Pfad zur Journal-Datei (für Tests).

    Crasht niemals — bei Fehler wird nur eine Warnung geloggt.
    """
    try:
        # Pfad bestimmen
        if journal_file is None:
            if journal_dir is None:
                journal_dir = "journal"
            os.makedirs(journal_dir, exist_ok=True)
            journal_file = os.path.join(journal_dir, "decisions.csv")
        else:
            # Bei explizitem journal_file das übergeordnete Verzeichnis anlegen
            parent_dir = os.path.dirname(journal_file)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        # Felder extrahieren (leer lassen wenn nicht vorhanden)
        trade = result.get("trade", {}) or {}
        final = result.get("final", {}) or {}
        debate = result.get("debate", {}) or {}
        bull = debate.get("bull", {}) or {}
        bear = debate.get("bear", {}) or {}
        ticker = result.get("ticker", "")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = {
            "timestamp": timestamp,
            "ticker": ticker,
            "action": trade.get("aktion", ""),
            "target": trade.get("zielkurs", ""),
            "stop": trade.get("stop_loss", ""),
            "position_pct": trade.get("positionsanteil", ""),
            "final_decision": final.get("entscheidung", ""),
            "confidence": final.get("confidence", ""),
            "bull_confidence": _parse_confidence_from_debate(bull),
            "bear_confidence": _parse_confidence_from_debate(bear),
        }

        # Datei existiert? → Header nur schreiben wenn neu
        file_exists = os.path.isfile(journal_file)

        with open(journal_file, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.info("Entscheidung ins Journal geschrieben: %s", journal_file)
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Journal-Eintrag konnte nicht geschrieben werden: %s", exc)
