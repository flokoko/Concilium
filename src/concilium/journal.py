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

# Platform-Guard: fcntl ist Linux/Unix-only; auf anderen Plattformen None.
try:
    import fcntl
except ImportError:  # pragma: no cover — Windows hat kein fcntl
    fcntl = None

logger = logging.getLogger(__name__)

# Spalten des CSV-Journals
JOURNAL_HEADER = [
    "timestamp",
    "ticker",
    "action",
    "rating",
    "target",
    "stop",
    "position_pct",
    "final_decision",
    "confidence",
    "bull_confidence",
    "bear_confidence",
    "ensemble_confidence",
    "portfolio_fit_score",
    "ziel_gewichtung_pct",
]


def _parse_confidence_from_debate(agent: dict[str, Any]) -> str:
    """Extrahiert die confidence aus einem Bull/Bear-Agent-Dict.

    Im strukturierten Pfad hat das dict ein direktes ``confidence``-Feld —
    dieses wird zuerst geprüft. Als Fallback wird der JSON-Block aus
    ``agent["_raw"]`` via Regex extrahiert (nicht-strukturierter Pfad).

    Gibt einen leeren String zurück bei fehlender/ungültiger confidence.
    """
    if not isinstance(agent, dict):
        return ""

    # 1. Versuch: direktes confidence-Feld im Agent-Dict (strukturierter Pfad)
    direct = agent.get("confidence")
    if direct is not None:
        try:
            return str(int(direct))
        except (TypeError, ValueError):
            pass

    # 2. Versuch: JSON-Block aus _raw parsen (Fallback-Pfad)
    raw = str(agent.get("_raw", ""))
    if not raw:
        return ""
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


def _acquire_lock(fh: Any) -> None:
    """Holt ein exklusives Datei-Lock (best effort, crasht nie).

    Verwendet fcntl.flock (LOCK_EX) wenn fcntl verfügbar ist.
    Bei Fehler oder fehlendem fcntl → kein Lock (best effort).
    """
    if fcntl is None:
        return
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    except Exception as exc:  # noqa: BLE001 — best effort, Lock nicht kritisch
        logger.debug("Konnte kein File-Lock erwerben: %s", exc)


def _release_lock(fh: Any) -> None:
    """Gibt das File-Lock frei (best effort, crasht nie)."""
    if fcntl is None:
        return
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception as exc:  # noqa: BLE001 — best effort
        logger.debug("Konnte File-Lock nicht freigeben: %s", exc)


def _rewrite_journal_with_header(journal_file: str, existing_fields: list[str]) -> None:
    """Schreibt eine bestehende Journal-CSV neu mit dem erweiterten Header.

    Liest alle Zeilen der alten Datei, fügt fehlende Spalten (z. B.
    ensemble_confidence, portfolio_fit_score, ziel_gewichtung_pct) als leere
    Werte zu jeder Zeile hinzu und schreibt die Datei neu mit JOURNAL_HEADER.
    """
    try:
        with open(journal_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        with open(journal_file, "w", newline="", encoding="utf-8") as fh:
            _acquire_lock(fh)
            try:
                writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
                writer.writeheader()
                for row in rows:
                    # Fehlende Spalten als leer auffüllen
                    for field in JOURNAL_HEADER:
                        if field not in row:
                            row[field] = ""
                    writer.writerow({k: row.get(k, "") for k in JOURNAL_HEADER})
            finally:
                _release_lock(fh)
    except Exception as exc:  # noqa: BLE001 — best effort
        logger.warning("Journal-Header-Erweiterung fehlgeschlagen: %s", exc)


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

        # Ensemble-Konfidenz aus trade._ensemble extrahieren
        ensemble_info = trade.get("_ensemble", {}) or {}
        ensemble_confidence = ensemble_info.get("ensemble_confidence", "")
        if isinstance(ensemble_confidence, float):
            ensemble_confidence = f"{ensemble_confidence:.2f}"

        # Portfolio-Fit-Felder extrahieren (falls vorhanden)
        portfolio_fit = result.get("portfolio_fit") or {}
        portfolio_fit_score = portfolio_fit.get("portfolio_fit_score", "")
        ziel_gewichtung_pct = portfolio_fit.get("ziel_gewichtung_pct", "")

        row = {
            "timestamp": timestamp,
            "ticker": ticker,
            "action": trade.get("aktion", ""),
            "rating": trade.get("rating", ""),
            "target": trade.get("zielkurs", ""),
            "stop": trade.get("stop_loss", ""),
            "position_pct": trade.get("positionsanteil", ""),
            "final_decision": final.get("entscheidung", ""),
            "confidence": final.get("confidence", ""),
            "bull_confidence": _parse_confidence_from_debate(bull),
            "bear_confidence": _parse_confidence_from_debate(bear),
            "ensemble_confidence": ensemble_confidence,
            "portfolio_fit_score": portfolio_fit_score,
            "ziel_gewichtung_pct": ziel_gewichtung_pct,
        }

        # Datei existiert? → Header nur schreiben wenn neu
        file_exists = os.path.isfile(journal_file)

        # Bei bestehender Datei mit fehlenden Spalten:
        # Header ergänzen, indem wir die Datei neu schreiben mit erweitertem Header.
        if file_exists:
            try:
                with open(journal_file, encoding="utf-8") as fh_check:
                    reader = csv.DictReader(fh_check)
                    existing_fields = reader.fieldnames or []
                # Wenn Header nicht mit JOURNAL_HEADER übereinstimmt → Migration
                if set(JOURNAL_HEADER) - set(existing_fields):
                    _rewrite_journal_with_header(journal_file, list(existing_fields))
            except Exception as exc:  # noqa: BLE001 — best effort
                logger.warning("Journal-Header-Check fehlgeschlagen: %s", exc)

        with open(journal_file, "a", newline="", encoding="utf-8") as fh:
            _acquire_lock(fh)
            try:
                writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            finally:
                _release_lock(fh)

        logger.info("Entscheidung ins Journal geschrieben: %s", journal_file)
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Journal-Eintrag konnte nicht geschrieben werden: %s", exc)
