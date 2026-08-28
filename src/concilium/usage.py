"""Token-Usage-Logging — protokolliert den LLM-Token-Verbrauch pro Analyse.

Schreibt Einträge an usage/usage.csv (relativ zum Arbeitsverzeichnis),
analog journal/decisions.csv. Robust: crasht niemals, legt fehlende Ordner
an, schreibt nur bei vorhandenem usage.
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from typing import Any

# Platform-Guard: fcntl ist Linux/Unix-only; auf anderen Plattformen None.
try:
    import fcntl
except ImportError:  # pragma: no cover — Windows hat kein fcntl
    fcntl = None

logger = logging.getLogger(__name__)

# Spalten der Usage-CSV
USAGE_HEADER = [
    "timestamp",
    "ticker",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]

_DEFAULT_USAGE_FILE = os.path.join("usage", "usage.csv")


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


def record_usage(
    ticker: str,
    usage: dict | None,
    *,
    usage_file: str | None = None,
) -> None:
    """Schreibt einen Usage-Eintrag in die CSV-Datei.

    Args:
        ticker: Ticker-Symbol der Analyse.
        usage: usage-dict mit prompt_tokens/completion_tokens/total_tokens,
            oder None.
        usage_file: Optionaler Pfad zur Usage-CSV (für Tests).
            Default: usage/usage.csv.

    Wenn usage None ist oder keine total_tokens enthält → nichts tun.
    Crasht niemals — bei Fehler wird nur eine Warnung geloggt.
    """
    if usage is None or not usage.get("total_tokens"):
        return

    try:
        if usage_file is None:
            usage_file = _DEFAULT_USAGE_FILE

        parent_dir = os.path.dirname(usage_file)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "timestamp": timestamp,
            "ticker": ticker,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

        file_exists = os.path.isfile(usage_file)

        with open(usage_file, "a", newline="", encoding="utf-8") as fh:
            _acquire_lock(fh)
            try:
                writer = csv.DictWriter(fh, fieldnames=USAGE_HEADER)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            finally:
                _release_lock(fh)

        logger.info("Usage aufgezeichnet für %s: %s Tokens", ticker, row["total_tokens"])
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Usage-Eintrag konnte nicht geschrieben werden: %s", exc)


def summarize_usage(usage_file: str | None = None) -> dict:
    """Aggregiert die Usage-Daten aus der CSV-Datei.

    Args:
        usage_file: Optionaler Pfad zur Usage-CSV (für Tests).
            Default: usage/usage.csv.

    Returns:
        dict mit:
          - anzahl_calls: Anzahl der aufgezeichneten LLM-Aufrufe.
          - summe_prompt_tokens: Summe aller prompt_tokens.
          - summe_completion_tokens: Summe aller completion_tokens.
          - summe_total_tokens: Summe aller total_tokens.
          - anzahl_ticker: Anzahl eindeutiger Ticker.
          - ticker_tokens: {ticker: total_tokens} pro Ticker.

    Crasht niemals — bei fehlender/leerer Datei → leeres dict mit Nullen.
    """
    empty = {
        "anzahl_calls": 0,
        "summe_prompt_tokens": 0,
        "summe_completion_tokens": 0,
        "summe_total_tokens": 0,
        "anzahl_ticker": 0,
        "ticker_tokens": {},
    }

    if usage_file is None:
        usage_file = _DEFAULT_USAGE_FILE

    try:
        if not os.path.isfile(usage_file):
            return empty

        with open(usage_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        if not rows:
            return empty

        tickers: set[str] = set()
        ticker_tokens: dict[str, int] = {}
        sum_prompt = 0
        sum_completion = 0
        sum_total = 0

        for row in rows:
            try:
                prompt = int(row.get("prompt_tokens", 0) or 0)
                completion = int(row.get("completion_tokens", 0) or 0)
                total = int(row.get("total_tokens", 0) or 0)
            except (TypeError, ValueError):
                continue

            ticker = row.get("ticker", "")
            sum_prompt += prompt
            sum_completion += completion
            sum_total += total

            if ticker:
                tickers.add(ticker)
                ticker_tokens[ticker] = ticker_tokens.get(ticker, 0) + total

        return {
            "anzahl_calls": len(rows),
            "summe_prompt_tokens": sum_prompt,
            "summe_completion_tokens": sum_completion,
            "summe_total_tokens": sum_total,
            "anzahl_ticker": len(tickers),
            "ticker_tokens": ticker_tokens,
        }
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Usage-Aggregation fehlgeschlagen: %s", exc)
        return empty
