"""Checkpoint-Persistenz — speichert Pipeline-Zwischenstände für Crash-Resume.

Analog zum Journal-Lock-Pattern: atomares Schreiben via tmpfile + os.replace,
fcntl-Lock (best effort), tolerante JSON-Serialisierung (default=str für numpy etc.).
Checkpoints landen unter state/ (relativ zum Arbeitsverzeichnis) oder unter
dem via CONCILIUM_STATE_DIR Übersteuerten Verzeichnis.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

# Platform-Guard: fcntl ist Linux/Unix-only; auf anderen Plattformen None.
try:
    import fcntl
except ImportError:  # pragma: no cover — Windows hat kein fcntl
    fcntl = None

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1


def _normalize_ticker(ticker: str) -> str:
    """Normalisiert einen Ticker für den Dateinamen (z. B. RWE.DE → RWE_DE)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", ticker)


def _state_dir(state_dir: str | None = None) -> str:
    """Löst das State-Verzeichnis auf.

    Priorität: expliziter Parameter > CONCILIUM_STATE_DIR-Env > 'state'.
    """
    if state_dir is not None:
        return state_dir
    env = os.environ.get("CONCILIUM_STATE_DIR")
    if env:
        return env
    return "state"


def _checkpoint_path(ticker: str, *, state_dir: str | None = None) -> str:
    """Gibt den vollständigen Pfad zur Checkpoint-Datei für den Ticker zurück."""
    base = _state_dir(state_dir)
    name = _normalize_ticker(ticker)
    return os.path.join(base, f"{name}_checkpoint.json")


def _acquire_lock(fh: Any) -> None:
    """Holt ein exklusives Datei-Lock (best effort, crasht nie)."""
    if fcntl is None:
        return
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    except Exception as exc:  # noqa: BLE001 — best effort
        logger.debug("Konnte kein File-Lock erwerben: %s", exc)


def _release_lock(fh: Any) -> None:
    """Gibt das File-Lock frei (best effort, crasht nie)."""
    if fcntl is None:
        return
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception as exc:  # noqa: BLE001 — best effort
        logger.debug("Konnte File-Lock nicht freigeben: %s", exc)


def save_checkpoint(
    result: dict[str, Any],
    ticker: str,
    *,
    state_dir: str | None = None,
) -> None:
    """Schreibt den aktuellen Pipeline-Zwischenstand als JSON atomar + unter Lock.

    Verwendet default=str für tolerante Serialisierung (numpy-Typen etc.).
    Crasht nie — bei Fehler wird nur gewarnt.
    """
    cp_path = _checkpoint_path(ticker, state_dir=state_dir)
    base_dir = os.path.dirname(cp_path)
    try:
        os.makedirs(base_dir, exist_ok=True)

        payload = dict(result)
        payload["_checkpoint_version"] = CHECKPOINT_VERSION

        # Atomar: in tmpfile im selben Verzeichnis schreiben, dann os.replace
        tmp_path = cp_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            _acquire_lock(fh)
            try:
                json.dump(payload, fh, default=str, ensure_ascii=False)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            finally:
                _release_lock(fh)
        os.replace(tmp_path, cp_path)
        logger.debug("Checkpoint gespeichert: %s", cp_path)
    except Exception as exc:  # noqa: BLE001 — nie crashen
        # tmpfile aufräumen falls übrig
        tmp_path = cp_path + ".tmp"
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        logger.warning("Checkpoint konnte nicht gespeichert werden: %s", exc)


def load_checkpoint(
    ticker: str,
    *,
    state_dir: str | None = None,
) -> dict[str, Any] | None:
    """Lädt den Checkpoint für den Ticker.

    Gibt None zurück, wenn keiner existiert oder das JSON kaputt ist.
    """
    cp_path = _checkpoint_path(ticker, state_dir=state_dir)
    if not os.path.isfile(cp_path):
        return None
    try:
        with open(cp_path, encoding="utf-8") as fh:
            _acquire_lock(fh)
            try:
                data = json.load(fh)
            finally:
                _release_lock(fh)
        if not isinstance(data, dict):
            logger.warning("Checkpoint ist kein dict — ignoriere: %s", cp_path)
            return None
        return data
    except json.JSONDecodeError as exc:
        logger.warning("Checkpoint JSON kaputt (%s): %s", cp_path, exc)
        return None
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Checkpoint konnte nicht geladen werden (%s): %s", cp_path, exc)
        return None


def clear_checkpoint(
    ticker: str,
    *,
    state_dir: str | None = None,
) -> None:
    """Entfernt den Checkpoint für den Ticker (best effort, crasht nie)."""
    cp_path = _checkpoint_path(ticker, state_dir=state_dir)
    try:
        if os.path.isfile(cp_path):
            os.remove(cp_path)
            logger.debug("Checkpoint aufgeräumt: %s", cp_path)
    except Exception as exc:  # noqa: BLE001 — best effort
        logger.warning("Checkpoint konnte nicht gelöscht werden: %s", exc)
