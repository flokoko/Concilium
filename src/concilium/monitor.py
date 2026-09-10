"""Stop-Monitor — täglicher Stop-/Ziel-Check der offenen Depot-Positionen.

Phase 4: Das Entscheidungs-Journal (journal/decisions.csv) hält ``stop`` und
``target`` pro Entscheidung, ausgewertet wurde es aber nur im Nachhinein
(--evaluate, nach lookback_days). Der Monitor ist der operative Teil: Er
lädt das reale Depot (Google-Sheet), liest pro Aktien-Position den jüngsten
Journal-Eintrag mit Stop/Ziel, prüft den aktuellen Kurs und markiert:
- STOP GERISSEN (Kurs hat das Stop-Level verletzt), bzw.
- ZIEL ERREICHT (Kurs hat das Kursziel erreicht).

Richtung: KAUFEN/HALTEN = Long (Stop unterhalb, Ziel oberhalb),
VERKAUFEN = Short (Stop oberhalb, Ziel unterhalb).

Crasht nie: Ein fehlgeschlagener Ticker wird gezählt (analog Review-/Batch-
Modus), wirft aber keinen Fehler — der Rest läuft weiter.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Any

from .data import collect_ticker_data
from .portfolio_fit import fetch_portfolio_positions

logger = logging.getLogger(__name__)

# Default-Journal (relativ zum Arbeitsverzeichnis — analog append_decision)
_DEFAULT_JOURNAL_FILE = os.path.join("journal", "decisions.csv")


def _parse_float(val: Any) -> float | None:
    """Parst tolerant einen positiven Zahlenwert (Stop/Target).

    Akzeptiert Zahlen (int/float), Strings mit Punkt- oder Komma-Dezimal-
    trenner ("165", "165.5", "165,5"). Leere/ungültige Werte und Werte <= 0
    werden als "nicht vorhanden" (None) behandelt — nie ein Crash.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return f if f > 0 else None
    text = str(val).strip().replace(",", ".")
    if not text:
        return None
    try:
        f = float(text)
    except ValueError:
        return None
    return f if f > 0 else None


def load_last_journal_entry(
    ticker: str, journal_file: str | None = None
) -> dict[str, str] | None:
    """Liest den jüngsten Journal-Eintrag für einen Ticker aus decisions.csv.

    "Jüngste" = letzte Zeile mit passendem Ticker (case-insensitive) — das
    Journal wird chronologisch appended, daher ist die letzte passende Zeile
    die aktuellste Entscheidung. Verglichen wird case-insensitive, da das
    Sheet/der LLM in der Schreibweise schwanken kann.

    Args:
        ticker: Ticker-Symbol (z. B. "AAPL", "BAS.DE").
        journal_file: Optionaler Pfad zur Journal-CSV (Default:
            journal/decisions.csv relativ zum Arbeitsverzeichnis — für Tests).

    Returns:
        Die letzte passende Zeile als dict oder None (kein Eintrag, fehlende
        Datei, Lesefehler — crasht nie).
    """
    if journal_file is None:
        journal_file = _DEFAULT_JOURNAL_FILE
    wanted = (ticker or "").strip().lower()
    if not wanted:
        return None
    try:
        last: dict[str, str] | None = None
        with open(journal_file, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("ticker") or "").strip().lower() == wanted:
                    last = row
        return last
    except FileNotFoundError:
        return None
    except OSError as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Journal konnte nicht gelesen werden (%s): %s", journal_file, exc)
        return None


def _depot_pct(pos: dict[str, Any]) -> float:
    """Liest depot_pct als Float (0.0 bei fehlendem/ungültigem Wert)."""
    try:
        return float(pos.get("depot_pct") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def run_monitor(
    llm=None,
    *,
    max_positions: int | None = None,
    as_of: str | None = None,
    journal_file: str | None = None,
) -> dict[str, Any]:
    """Stop-Monitor: prüft offene Aktien-Positionen gegen Journal-Stop/Ziel.

    Schritte:
    1. Depot laden via fetch_portfolio_positions() (Google-Sheet, Tages-Cache;
       crasht nie — trotzdem defensiv abgesichert, analog run_review).
    2. Filter auf type == "Aktie" (ETFs/Commodities = Buy-and-Hold, kein
       Stop-Monitor). Optional max_positions (größte zuerst nach depot_pct).
    3. Pro Aktie: jüngsten Journal-Eintrag mit stop/target laden, aktuellen
       Kurs via collect_ticker_data() (Tages-Cache) holen und Stop-/Ziel-Check
       ausführen.

    Args:
        llm: Unbenutzt (der Monitor ist deterministisch und netzfrei bzgl.
            LLM) — Parameter nur aus API-Symmetrie zu run_review/run_pipeline.
        max_positions: Wenn gesetzt, nur die N größten Aktien-Positionen
            (nach depot_pct absteigend) prüfen. None = alle.
        as_of: Optionales gepinntes Analysedatum (YYYY-MM-DD) — wird an
            collect_ticker_data durchgereicht (Kurs-Historie bis zu diesem
            Datum; für reproduzierbare Tests).
        journal_file: Optionaler Pfad zur Journal-CSV (Default:
            journal/decisions.csv — für Tests übersteuerbar).

    Returns:
        dict mit:
          - "positionen": {ticker: {name, depot_pct, current_price, stop,
            target, action, timestamp, journal_gefunden, stop_gerissen,
            ziel_erreicht, hinweis}} — eine Prüfung je Aktien-Position;
            stop_gerissen/ziel_erreicht sind bool oder None (nicht prüfbar).
          - "fehler": Anzahl fehlgeschlagener Ticker (crasht den Monitor nicht)
          - "gesamt_positionen": Anzahl aller Depot-Positionen (inkl. ETFs)
    """
    positionen: dict[str, dict[str, Any]] = {}
    fehler = 0

    # --- 1. Depot laden (fetch_portfolio_positions crasht nie — trotzdem
    #     defensiv absichern, damit der Monitor NIEMALS am Depot-Load scheitert).
    positions: list[dict[str, Any]] = []
    try:
        positions = fetch_portfolio_positions() or []
        if not isinstance(positions, list):
            positions = []
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Depot konnte nicht geladen werden: %s", exc)
        positions = []

    gesamt_positionen = len(positions)

    # --- 2. Filter: nur Aktien (ETFs/Commodities = Buy-and-Hold, kein Stop).
    aktien = [
        p
        for p in positions
        if isinstance(p, dict) and p.get("type") == "Aktie" and p.get("ticker")
    ]

    # Größte Positionen zuerst (nach depot_pct) — deterministische Reihenfolge
    # und Grundlage für max_positions.
    aktien.sort(key=_depot_pct, reverse=True)
    if max_positions is not None and max_positions >= 0:
        aktien = aktien[:max_positions]

    if not aktien:
        logger.info(
            "Monitor: keine Aktien-Positionen zu prüfen (%d Positionen gesamt).",
            gesamt_positionen,
        )
        return {
            "positionen": positionen,
            "fehler": fehler,
            "gesamt_positionen": gesamt_positionen,
        }

    # --- 3. Pro Aktie: Journal-Eintrag + Kurs + Stop-/Ziel-Check -------------
    for pos in aktien:
        ticker = str(pos.get("ticker") or "").strip()
        name = str(pos.get("name") or "")
        depot_pct = _depot_pct(pos)

        # Journal-Eintrag laden (None → kein Stop bekannt, kein Crash)
        entry = load_last_journal_entry(ticker, journal_file)
        if entry is None:
            positionen[ticker] = {
                "name": name,
                "depot_pct": depot_pct,
                "current_price": None,
                "stop": None,
                "target": None,
                "action": "",
                "timestamp": "",
                "journal_gefunden": False,
                "stop_gerissen": None,
                "ziel_erreicht": None,
                "hinweis": "Kein Journal-Eintrag (kein Stop bekannt)",
            }
            logger.info("Monitor: %s — kein Journal-Eintrag (kein Stop bekannt).", ticker)
            continue

        stop = _parse_float(entry.get("stop"))
        target = _parse_float(entry.get("target"))
        action = str(entry.get("action") or "").strip()
        timestamp = str(entry.get("timestamp") or "").strip()

        # Aktueller Kurs (Tages-Cache; Fehler → gezählt, kein Crash)
        try:
            data = collect_ticker_data(ticker, as_of=as_of)
            technicals = data.get("technicals", {}) or {}
            raw_price = technicals.get("current_price")
            current_price = float(raw_price) if raw_price is not None else None
        except Exception as exc:  # noqa: BLE001 — analog Review/Batch: zählen, nicht crashen
            logger.warning(
                "Monitor: Kursdaten für '%s' fehlgeschlagen: %s", ticker, exc
            )
            fehler += 1
            continue
        if current_price is None:
            logger.warning("Monitor: kein aktueller Kurs für '%s' verfügbar.", ticker)
            fehler += 1
            continue

        # Richtung: VERKAUFEN (auch "STARK VERKAUFEN") = Short, sonst Long.
        short = "VERKAUFEN" in action.upper()

        # --- Stop-Check ---
        # Long (KAUFEN/HALTEN): Stop gerissen wenn Kurs <= Stop.
        # Short (VERKAUFEN): Stop gerissen wenn Kurs >= Stop.
        stop_gerissen: bool | None = None
        if stop is not None:
            stop_gerissen = current_price >= stop if short else current_price <= stop
            if stop_gerissen:
                logger.warning(
                    "STOP GERISSEN: %s — Kurs %.2f %s Stop %.2f",
                    ticker,
                    current_price,
                    ">=" if short else "<=",
                    stop,
                )

        # --- Target-Check ---
        # Long: Ziel erreicht wenn Kurs >= Ziel. Short: Ziel erreicht wenn
        # Kurs <= Ziel.
        ziel_erreicht: bool | None = None
        if target is not None:
            ziel_erreicht = current_price <= target if short else current_price >= target

        # --- Hinweis (kompakte deutsche Beschreibung) ---
        if stop is None and target is None:
            hinweis = "Kein Stop/Ziel im Journal-Eintrag"
        elif stop_gerissen and ziel_erreicht:
            hinweis = "STOP GERISSEN und ZIEL ERREICHT"
        elif stop_gerissen:
            hinweis = "STOP GERISSEN"
        elif ziel_erreicht:
            hinweis = "ZIEL ERREICHT"
        else:
            hinweis = "OK"

        positionen[ticker] = {
            "name": name,
            "depot_pct": depot_pct,
            "current_price": current_price,
            "stop": stop,
            "target": target,
            "action": action,
            "timestamp": timestamp,
            "journal_gefunden": True,
            "stop_gerissen": stop_gerissen,
            "ziel_erreicht": ziel_erreicht,
            "hinweis": hinweis,
        }

    return {
        "positionen": positionen,
        "fehler": fehler,
        "gesamt_positionen": gesamt_positionen,
    }
