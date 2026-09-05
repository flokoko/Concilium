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

from . import config

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
    "ziel_gewichtung_original",
    # --- Roadmap C6: Deferred Reflection (Pending-Entries, look-ahead-frei) ---
    # reflection_status: "" (Legacy-Zeile vor C6) | "pending" (Ausgang unbekannt,
    # wird beim nächsten Lauf resolved) | "resolved" (Return + Lektion persistiert)
    "reflection_status",
    # resolved_at: ISO-Timestamp der Auflösung (leer solange pending)
    "resolved_at",
    # realised_return_pct / alpha_pct: persistierter realisierter Return inkl.
    # Alpha vs regionalem Benchmark (leer solange pending) — wird beim
    # Resolving einmalig berechnet und danach von build_reflection_context
    # wiederverwendet.
    "realised_return_pct",
    "alpha_pct",
    # lesson: persistierte Lektion (LLM- oder deterministischer Satz) — wird
    # beim Resolving einmalig generiert und danach wiederverwendet, damit die
    # Reflexion nicht bei jedem Lauf neu berechnet werden muss.
    "lesson",
]

# Statuswerte für reflection_status (C6)
REFLECTION_STATUS_PENDING = "pending"
REFLECTION_STATUS_RESOLVED = "resolved"


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


def _resolve_sort_key(row: dict[str, str]) -> str:
    """Sortier-Schlüssel für resolved-Zeilen (älteste zuerst prunen).

    Primär ``resolved_at`` (ISO-Format "%Y-%m-%d %H:%M:%S", lexikografisch
    sortierbar — analog feedback.py::_write_resolution). Leer oder
    unparseable → Fallback auf ``timestamp`` (dasselbe Format). Beide leer →
    leere String (sortiert als älteste Zeile, wird zuerst geprunt).
    """
    resolved_at = str(row.get("resolved_at") or "").strip()
    ts = str(row.get("timestamp") or "").strip()
    return resolved_at or ts


def _prune_resolved(journal_file: str, max_resolved: int) -> None:
    """Prunt die ältesten resolved-Einträge bis höchstens max_resolved übrig sind.

    Journal-Hygiene analog TradingAgents' TradingMemoryLog: Optionaler Cap
    auf aufgelöste (resolved) Einträge — die ältesten resolved-Zeilen (nach
    ``resolved_at``, Fallback ``timestamp``) werden entfernt, bis der Cap
    eingehalten wird. Pending- und Legacy-Zeilen (reflection_status "" oder
    "pending") werden NIE geprunt.

    Atomar + lock-sicher (tmpfile + os.replace, fcntl-Lock auf der
    Zieldatei — analog feedback.py::_write_resolution). Crasht nie: Bei
    Fehlern wird nur eine Warnung geloggt, das Journal bleibt unverändert.
    """
    if max_resolved <= 0:
        return  # Rotation deaktiviert
    tmp_path = ""
    try:
        # 1. Zeilen lesen + resolved-Zeilen mit Datei-Position sammeln
        with open(journal_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or JOURNAL_HEADER)
            rows = list(reader)

        resolved_indexed = [
            (idx, row)
            for idx, row in enumerate(rows)
            if str(row.get("reflection_status") or "").strip()
            == REFLECTION_STATUS_RESOLVED
        ]
        excess = len(resolved_indexed) - max_resolved
        if excess <= 0:
            return  # Cap eingehalten — nichts zu prunen

        # 2. Die excess ältesten resolved-Zeilen (nach resolved_at, Fallback
        #    timestamp) über ihre Datei-Position markieren — positions-basiert
        #    ist exakt auch bei Zeilen mit identischem (ticker, timestamp).
        resolved_indexed.sort(key=lambda item: _resolve_sort_key(item[1]))
        doomed_positions = {idx for idx, _ in resolved_indexed[:excess]}

        # 3. Restliche Zeilen atomar zurückschreiben (tmpfile + os.replace)
        tmp_path = f"{journal_file}.tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for idx, row in enumerate(rows):
                if idx in doomed_positions:
                    continue  # geprunt
                for field in fieldnames:
                    if field not in row:
                        row[field] = ""
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        try:
            with open(journal_file, encoding="utf-8") as lock_fh:
                _acquire_lock(lock_fh)
                try:
                    os.replace(tmp_path, journal_file)
                finally:
                    _release_lock(lock_fh)
        except OSError:
            # Lock auf der Zieldatei nicht möglich → replace ohne Lock (best effort)
            os.replace(tmp_path, journal_file)
        logger.info(
            "Journal-Rotation: %d resolved Einträge geprunt (Cap %d): %s",
            excess,
            max_resolved,
            journal_file,
        )
    except Exception as exc:  # noqa: BLE001 — crasht nie
        logger.warning("Journal-Rotation fehlgeschlagen: %s", exc)
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:  # noqa: BLE001
            pass


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
        ziel_gewichtung_original = portfolio_fit.get("ziel_gewichtung_original", "")

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
            "ziel_gewichtung_original": ziel_gewichtung_original,
            # C6: Neue Entscheidungen starten als "pending" — der Ausgang
            # existiert zum Entscheidungszeitpunkt noch gar nicht (kein
            # Look-ahead). Der nächste Lauf derselben Aktie resolved den
            # Eintrag, sobald decision_date + lookback_days vollständig
            # abgelaufen ist (resolve_pending_reflections in feedback.py).
            "reflection_status": REFLECTION_STATUS_PENDING,
            "resolved_at": "",
            "realised_return_pct": "",
            "alpha_pct": "",
            "lesson": "",
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

        # --- Idempotenz-Guard (Journal-Hygiene analog TradingAgents) -------
        # Vor dem Append prüfen, ob bereits eine Zeile mit demselben
        # (ticker, timestamp) existiert. Ein Checkpoint-Resume innerhalb
        # derselben Sekunde erzeugt denselben Timestamp — ohne Guard würde
        # doppelt geloggt. Best effort: Kann die Datei nicht gelesen werden,
        # schreiben wir normal (lieber ein Duplikat als ein verlorener Eintrag).
        try:
            if os.path.isfile(journal_file):
                with open(journal_file, encoding="utf-8") as fh_check:
                    for existing_row in csv.DictReader(fh_check):
                        existing_ticker = str(existing_row.get("ticker") or "").strip().lower()
                        existing_ts = str(existing_row.get("timestamp") or "").strip()
                        if (
                            existing_ticker == str(ticker).strip().lower()
                            and existing_ts == timestamp
                        ):
                            logger.warning(
                                "Journal-Duplikat verhindert: Eintrag (ticker=%s, timestamp=%s) "
                                "existiert bereits — kein Append",
                                ticker,
                                timestamp,
                            )
                            return
        except Exception as exc:  # noqa: BLE001 — best effort, Guard nicht kritisch
            logger.warning("Journal-Idempotenz-Check fehlgeschlagen — schreibe normal: %s", exc)

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

        # --- Journal-Hygiene: Rotation aufgelöster Einträge ----------------
        # Optionaler Cap (CONCILIUM_JOURNAL_MAX_RESOLVED, Default 0 = aus):
        # Älteste resolved-Zeilen prunen, bis der Cap eingehalten ist.
        # _prune_resolved crasht nie; auch die Config-Lesung (kann bei
        # Tippfehlern laut ValueError werfen) ist best effort.
        try:
            _prune_resolved(journal_file, config.journal_max_resolved())
        except Exception as exc:  # noqa: BLE001 — nie crashen
            logger.warning("Journal-Rotation konnte nicht ausgeführt werden: %s", exc)
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Journal-Eintrag konnte nicht geschrieben werden: %s", exc)
