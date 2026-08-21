"""Track-Record-Evaluierung — gleicht Entscheidungs-Journal gegen tatsächliche Kurse ab.

Liest journal/decisions.csv und bewertet jede Entscheidung (KAUFEN/VERKAUFEN/HALTEN)
gegen die tatsächliche Kursentwicklung via yfinance. Aggregiert Hit-Rate,
Rendite, Zielkurs-/Stop-Quoten und Konfidenz-Korrelation.

Robust: crasht niemals — jede Zeile wird einzeln in try/except ausgewertet.
yfinance-Aufrufe sind über den Tages-Cache aus data.py gespeichert (Wiederverwendung
von _get_cache_dir / _get_today_key).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf

from .data import _get_cache_dir, _get_today_key
from .journal import JOURNAL_HEADER  # noqa: F401 — re-exportiert für Test-Zugriff
from .llm import LLMClient

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cache-Hilfsfunktionen (eigener Preis-Cache, nutzt Cache-Dir aus data.py)
# --------------------------------------------------------------------------- #


def _price_cache_path(cache_dir: str, today_key: str, ticker: str) -> str:
    """Dateipfad für den Preis-Cache eines Tickers."""
    safe_ticker = re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
    return os.path.join(cache_dir, f"prices_{today_key}_{safe_ticker}.json")


def _load_price_cache(
    ticker: str, today_key: str | None = None
) -> list[dict[str, Any]] | None:
    """Lädt gecachte Preisdaten für einen Ticker (Tages-Cache).

    Returns:
        Liste von {date, close, high, low}-Dicts oder None bei Cache-Miss/Fehler.
    """
    cache_dir = _get_cache_dir()
    if cache_dir is None:
        return None
    if today_key is None:
        today_key = _get_today_key()

    path = _price_cache_path(cache_dir, today_key, ticker)
    try:
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            entry = json.load(fh)
        if entry.get("cache_date") != today_key:
            return None
        data = entry.get("data")
        if not isinstance(data, list):
            return None
        logger.info("Price-Cache-Treffer für %s (%s)", ticker, today_key)
        return data
    except Exception as exc:  # noqa: BLE001 — Cache-Lesen crasht nie
        logger.debug("Price-Cache-Lesen fehlgeschlagen für %s: %s", ticker, exc)
        return None


def _save_price_cache(
    ticker: str,
    records: list[dict[str, Any]],
    today_key: str | None = None,
) -> None:
    """Speichert Preisdaten für einen Ticker im Tages-Cache (best effort)."""
    cache_dir = _get_cache_dir()
    if cache_dir is None:
        return
    if today_key is None:
        today_key = _get_today_key()

    path = _price_cache_path(cache_dir, today_key, ticker)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {"cache_date": today_key, "ticker": ticker, "data": records},
                fh,
                ensure_ascii=False,
            )
        logger.info("Price-Cache gespeichert für %s (%s)", ticker, today_key)
    except Exception as exc:  # noqa: BLE001 — Cache-Schreiben crasht nie
        logger.debug("Price-Cache-Schreiben fehlgeschlagen für %s: %s", ticker, exc)


# --------------------------------------------------------------------------- #
# Kursdaten laden (yfinance, mit Tages-Cache)
# --------------------------------------------------------------------------- #


def _load_price_history(
    ticker: str,
    *,
    lookback_days: int = 90,
) -> list[dict[str, Any]] | None:
    """Lädt OHLC-Kurse für einen Ticker via yfinance (mit Tages-Cache).

    Args:
        ticker: Yahoo-Ticker-Symbol (z. B. AAPL, RWE.DE).
        lookback_days: Lookback-Zeitraum in Tagen (bestimmt yfinance period).

    Returns:
        Liste von dicts: {date (YYYY-MM-DD), close, high, low}.
        None bei Fehler oder keinen Daten.
    """
    # Cache prüfen
    cached = _load_price_cache(ticker)
    if cached is not None:
        return cached

    # yfinance laden
    try:
        period_days = max(lookback_days + 60, 120)
        t = yf.Ticker(ticker)
        hist = t.history(period=f"{period_days}d", auto_adjust=False)
        if hist is None or hist.empty:
            return None

        records: list[dict[str, Any]] = []
        for date, row in hist.iterrows():
            records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "close": float(row["Close"]) if row.get("Close") is not None else None,
                    "high": float(row["High"]) if row.get("High") is not None else None,
                    "low": float(row["Low"]) if row.get("Low") is not None else None,
                }
            )

        if not records:
            return None

        _save_price_cache(ticker, records)
        return records
    except Exception as exc:  # noqa: BLE001 — best effort
        logger.warning("Kursdaten für '%s' konnten nicht geladen werden: %s", ticker, exc)
        return None


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #


def _parse_timestamp(ts: str) -> datetime | None:
    """Parst einen Journal-Timestamp 'YYYY-MM-DD HH:MM:SS' → datetime (naive)."""
    if not ts or not ts.strip():
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts.strip(), fmt)
        except ValueError:
            continue
    return None


def _safe_float(val: Any) -> float | None:
    """Konvertiert einen Wert sicher zu float oder None."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _find_price_on_or_before(
    prices: list[dict[str, Any]], target_date: datetime
) -> dict[str, Any] | None:
    """Findet den Kurs an oder vor dem Zieldatum (nächster Handelstag ≤ target)."""
    target_str = target_date.strftime("%Y-%m-%d")
    best: dict[str, Any] | None = None
    best_date = ""
    for p in prices:
        d = p.get("date", "")
        if d <= target_str and d > best_date:
            best = p
            best_date = d
    return best


def _find_price_on_or_after(
    prices: list[dict[str, Any]], target_date: datetime
) -> dict[str, Any] | None:
    """Findet den Kurs an oder nach dem Zieldatum (nächster Handelstag ≥ target)."""
    target_str = target_date.strftime("%Y-%m-%d")
    best: dict[str, Any] | None = None
    best_date = ""
    for p in prices:
        d = p.get("date", "")
        if d >= target_str and (best_date == "" or d < best_date):
            best = p
            best_date = d
    return best


# --------------------------------------------------------------------------- #
# Einzelentscheidung bewerten
# --------------------------------------------------------------------------- #


def _evaluate_single(
    row: dict[str, Any],
    prices: list[dict[str, Any]],
    lookback_days: int,
) -> dict[str, Any]:
    """Bewertet eine einzelne Journal-Zeile gegen die Kursdaten.

    Returns:
        dict mit: hit (bool|None), rendite_pct (float|None),
        ziel_erreicht (bool|None), stop_gerissen (bool|None),
        action (str), confidence (float|None),
        portfolio_fit_score (float|None), ticker (str), timestamp (str).
    """
    action = (row.get("action") or "").strip().upper()
    timestamp = row.get("timestamp", "")
    decision_date = _parse_timestamp(timestamp)

    # Leeres Ergebnis bei unbrauchbaren Daten
    empty = {
        "hit": None,
        "rendite_pct": None,
        "ziel_erreicht": None,
        "stop_gerissen": None,
        "action": action,
        "confidence": _safe_float(row.get("confidence")),
        "portfolio_fit_score": _safe_float(row.get("portfolio_fit_score")),
        "ticker": row.get("ticker", ""),
        "timestamp": timestamp,
    }

    if decision_date is None or not prices:
        return empty

    # Entry-Preis: Kurs am oder vor Entscheidungsdatum
    entry = _find_price_on_or_before(prices, decision_date)
    if entry is None:
        entry = prices[0]  # Fallback: erster verfügbarer Kurs
    entry_price = _safe_float(entry.get("close")) if entry else None
    if entry_price is None or entry_price <= 0:
        return empty

    # Exit-Preis: heute oder lookback_days nach Entscheidung, je nachdem was früher
    today = datetime.now()
    end_date = decision_date + timedelta(days=lookback_days)
    eval_end = min(end_date, today)

    exit_row = _find_price_on_or_before(prices, eval_end)
    if exit_row is None:
        exit_row = prices[-1]  # Fallback: letzter verfügbarer Kurs
    exit_price = _safe_float(exit_row.get("close")) if exit_row else None
    if exit_price is None:
        return empty

    # Rendite berechnen
    price_change_pct = (exit_price - entry_price) / entry_price * 100.0

    # Für VERKAUFEN: Rendite invertieren (Gewinn wenn Kurs fällt)
    if action == "VERKAUFEN":
        rendite_pct = -price_change_pct
    else:
        rendite_pct = price_change_pct

    # Perioden-Kurse für Stop/Target-Check
    entry_date_str = entry.get("date", "")
    exit_date_str = exit_row.get("date", "") if exit_row else ""
    period_prices = [
        p for p in prices if entry_date_str <= p.get("date", "") <= exit_date_str
    ]

    # Zielkurs-Check
    target = _safe_float(row.get("target"))
    ziel_erreicht: bool | None = None
    if target is not None and target > 0 and period_prices:
        if action == "VERKAUFEN":
            # Verkauf: Ziel liegt unterhalb → Treffer wenn Low ≤ target
            ziel_erreicht = any(
                p.get("low") is not None and float(p["low"]) <= target
                for p in period_prices
            )
        else:
            # Kauf/Halten: Ziel liegt oberhalb → Treffer wenn High ≥ target
            ziel_erreicht = any(
                p.get("high") is not None and float(p["high"]) >= target
                for p in period_prices
            )

    # Stop-Check
    stop = _safe_float(row.get("stop"))
    stop_gerissen: bool | None = None
    if stop is not None and stop > 0 and period_prices:
        if action == "VERKAUFEN":
            # Verkauf: Stop liegt oberhalb → gerissen wenn High ≥ stop
            stop_gerissen = any(
                p.get("high") is not None and float(p["high"]) >= stop
                for p in period_prices
            )
        else:
            # Kauf/Halten: Stop liegt unterhalb → gerissen wenn Low ≤ stop
            stop_gerissen = any(
                p.get("low") is not None and float(p["low"]) <= stop
                for p in period_prices
            )

    # Hit-Bestimmung
    hit: bool | None = None
    if action == "KAUFEN":
        hit = rendite_pct > 0
    elif action == "VERKAUFEN":
        hit = rendite_pct > 0  # bereits invertiert
    elif action == "HALTEN":
        # Halten ist "richtig" wenn Kurs ±5% stabil geblieben ist
        hit = abs(rendite_pct) <= 5.0

    return {
        "hit": hit,
        "rendite_pct": rendite_pct,
        "ziel_erreicht": ziel_erreicht,
        "stop_gerissen": stop_gerissen,
        "action": action,
        "confidence": _safe_float(row.get("confidence")),
        "portfolio_fit_score": _safe_float(row.get("portfolio_fit_score")),
        "ticker": row.get("ticker", ""),
        "timestamp": timestamp,
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _empty_result() -> dict[str, Any]:
    """Leeres Ergebnis-dict (für fehlende/leere Journal-Datei)."""
    return {
        "anzahl_entscheidungen": 0,
        "nach_aktion": {
            "KAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
            "HALTEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
            "VERKAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
        },
        "hit_rate_gesamt": None,
        "durchschnitt_rendite_gesamt": None,
        "zielkurs_trefferquote": None,
        "stop_verletzungsquote": None,
        "konfidenz_baende": [],
        "portfolio_fit_hoch": None,
        "zusammenfassung": None,
        "fehler": [],
    }


def _aggregate(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregiert die Einzel-Ergebnisse zu einem Ergebnis-dict."""
    result = _empty_result()
    result["anzahl_entscheidungen"] = len(evaluations)

    # --- Nach Aktion ---
    for action in ("KAUFEN", "HALTEN", "VERKAUFEN"):
        action_evals = [e for e in evaluations if e["action"] == action]
        n = len(action_evals)
        hits = [e for e in action_evals if e["hit"] is True]
        misses = [e for e in action_evals if e["hit"] is False]
        rated = len(hits) + len(misses)
        renditen = [e["rendite_pct"] for e in action_evals if e["rendite_pct"] is not None]

        result["nach_aktion"][action] = {
            "n": n,
            "hit_rate": len(hits) / rated if rated > 0 else None,
            "avg_rendite": sum(renditen) / len(renditen) if renditen else None,
        }

    # --- Gesamt ---
    all_hits = [e for e in evaluations if e["hit"] is True]
    all_misses = [e for e in evaluations if e["hit"] is False]
    all_rated = len(all_hits) + len(all_misses)
    result["hit_rate_gesamt"] = len(all_hits) / all_rated if all_rated > 0 else None

    all_renditen = [e["rendite_pct"] for e in evaluations if e["rendite_pct"] is not None]
    result["durchschnitt_rendite_gesamt"] = (
        sum(all_renditen) / len(all_renditen) if all_renditen else None
    )

    # --- Zielkurs-Trefferquote ---
    ziel_evals = [e for e in evaluations if e["ziel_erreicht"] is not None]
    ziel_treffer = [e for e in ziel_evals if e["ziel_erreicht"] is True]
    result["zielkurs_trefferquote"] = (
        len(ziel_treffer) / len(ziel_evals) if ziel_evals else None
    )

    # --- Stop-Verletzungsquote ---
    stop_evals = [e for e in evaluations if e["stop_gerissen"] is not None]
    stop_hits = [e for e in stop_evals if e["stop_gerissen"] is True]
    result["stop_verletzungsquote"] = (
        len(stop_hits) / len(stop_evals) if stop_evals else None
    )

    # --- Konfidenz-Bänder ---
    # hoch: confidence ≥ 4, mittel: 3, niedrig: ≤ 2
    bands = {"hoch": [], "mittel": [], "niedrig": []}
    for e in evaluations:
        conf = e.get("confidence")
        if conf is None:
            continue
        if conf >= 4:
            bands["hoch"].append(e)
        elif conf >= 3:
            bands["mittel"].append(e)
        else:
            bands["niedrig"].append(e)

    konfidenz_baende: list[dict[str, Any]] = []
    for band_name in ("hoch", "mittel", "niedrig"):
        band_evals = bands[band_name]
        n = len(band_evals)
        if n == 0:
            continue
        hits = [e for e in band_evals if e["hit"] is True]
        misses = [e for e in band_evals if e["hit"] is False]
        rated = len(hits) + len(misses)
        konfidenz_baende.append(
            {
                "band": band_name,
                "hit_rate": len(hits) / rated if rated > 0 else None,
                "n": n,
            }
        )
    result["konfidenz_baende"] = konfidenz_baende

    # --- Portfolio-Fit-Zusammenhang ---
    pf_evals = [e for e in evaluations if e.get("portfolio_fit_score") is not None]
    pf_hoch = [e for e in pf_evals if (e.get("portfolio_fit_score") or 0) >= 4]
    if pf_hoch:
        pf_hits = [e for e in pf_hoch if e["hit"] is True]
        pf_misses = [e for e in pf_hoch if e["hit"] is False]
        pf_rated = len(pf_hits) + len(pf_misses)
        result["portfolio_fit_hoch"] = {
            "hit_rate": len(pf_hits) / pf_rated if pf_rated > 0 else None,
            "n": len(pf_hoch),
        }

    return result


# --------------------------------------------------------------------------- #
# LLM-Zusammenfassung
# --------------------------------------------------------------------------- #


def _build_llm_summary(result: dict[str, Any], llm: LLMClient) -> str | None:
    """Erzeugt eine deutsche LLM-Zusammenfassung der Track-Record-Qualität."""
    try:
        n = result["anzahl_entscheidungen"]
        hr = result.get("hit_rate_gesamt")
        hr_str = f"{hr * 100:.1f}%" if hr is not None and not (isinstance(hr, float) and math.isnan(hr)) else "N/A"
        dr = result.get("durchschnitt_rendite_gesamt")
        dr_str = f"{dr:.2f}%" if dr is not None and not (isinstance(dr, float) and math.isnan(dr)) else "N/A"
        zt = result.get("zielkurs_trefferquote")
        zt_str = f"{zt * 100:.1f}%" if zt is not None and not (isinstance(zt, float) and math.isnan(zt)) else "N/A"

        kauf = result["nach_aktion"]["KAUFEN"]
        kauf_hr = f"{kauf['hit_rate'] * 100:.1f}%" if kauf["hit_rate"] and not (isinstance(kauf["hit_rate"], float) and math.isnan(kauf["hit_rate"])) else "N/A"

        bands_str = ", ".join(
            f"{b['band']} ({b['n']}): "
            f"{b['hit_rate'] * 100:.0f}%" if b["hit_rate"] is not None and not (isinstance(b["hit_rate"], float) and math.isnan(b["hit_rate"]))
            else f"{b['band']} ({b['n']}): N/A"
            for b in result.get("konfidenz_baende", [])
        )

        prompt = (
            f"Du bist ein Finanzanalyst. Erstelle eine kurze deutsche Zusammenfassung "
            f"(2-4 Sätze) über die Track-Record-Qualität eines Trading-Systems.\n\n"
            f"Daten:\n"
            f"- Anzahl Entscheidungen: {n}\n"
            f"- Hit-Rate gesamt: {hr_str}\n"
            f"- Durchschnittliche Rendite: {dr_str}\n"
            f"- Zielkurs-Trefferquote: {zt_str}\n"
            f"- KAUFEN Hit-Rate: {kauf_hr}\n"
            f"- Konfidenz-Bänder: {bands_str}\n\n"
            f"Bewerte: Ist die Trefferquote gut? Stimmt die Konfidenz mit dem Erfolg überein? "
            f"Gibt es Auffälligkeiten? Schreibe 2-4 Sätze auf Deutsch."
        )

        messages = [
            {"role": "system", "content": "Du bist ein Finanzanalyst-Assistent."},
            {"role": "user", "content": prompt},
        ]
        return llm.chat(messages, temperature=0.4)
    except Exception as exc:  # noqa: BLE001 — best effort
        logger.warning("LLM-Zusammenfassung konnte nicht erzeugt werden: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Hauptfunktion
# --------------------------------------------------------------------------- #


def evaluate_journal(
    journal_file: str | None = None,
    *,
    lookback_days: int = 90,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Wertet das Entscheidungs-Journal gegen tatsächliche Kurse aus.

    Liest die Journal-CSV, lädt für jeden Ticker die historischen Kurse via
    yfinance (mit Tages-Cache) und bewertet jede Entscheidung.

    Args:
        journal_file: Pfad zur Journal-CSV. Default: journal/decisions.csv.
        lookback_days: Bewertungszeitraum in Tagen (Default: 90).
        llm: Optionaler LLMClient für deutsche Zusammenfassung.

    Returns:
        dict mit aggregierten Kennzahlen (siehe _empty_result für Struktur).
        Crasht niemals — bei Fehlern werden einzelne Zeilen als 'fehler' notiert.
    """
    if journal_file is None:
        journal_file = os.path.join("journal", "decisions.csv")

    # Fehlende/leere Datei → leeres Ergebnis
    if not os.path.isfile(journal_file):
        logger.info("Journal-Datei nicht gefunden: %s", journal_file)
        return _empty_result()

    # Journal lesen
    try:
        with open(journal_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Journal konnte nicht gelesen werden: %s", exc)
        return _empty_result()

    if not rows:
        return _empty_result()

    # Jede Zeile auswerten
    evaluations: list[dict[str, Any]] = []
    fehler: list[str] = []
    price_cache: dict[str, list[dict[str, Any]] | None] = {}

    for row in rows:
        ticker = (row.get("ticker") or "").strip()
        if not ticker:
            continue

        try:
            # Preise für diesen Ticker laden (mit Caching pro Ticker)
            if ticker not in price_cache:
                price_cache[ticker] = _load_price_history(
                    ticker, lookback_days=lookback_days
                )

            prices = price_cache[ticker]
            if not prices:
                fehler.append(
                    f"{row.get('timestamp', '?')} {ticker}: "
                    f"Keine Kursdaten verfügbar."
                )
                continue

            eval_result = _evaluate_single(row, prices, lookback_days)
            evaluations.append(eval_result)
        except Exception as exc:  # noqa: BLE001 — jede Zeile einzeln
            fehler.append(f"{row.get('timestamp', '?')} {ticker}: {exc}")

    # Aggregieren
    result = _aggregate(evaluations)
    result["fehler"] = fehler

    # LLM-Zusammenfassung (falls llm gegeben)
    if llm is not None and result["anzahl_entscheidungen"] > 0:
        result["zusammenfassung"] = _build_llm_summary(result, llm)

    return result
