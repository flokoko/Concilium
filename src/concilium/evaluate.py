"""Track-Record-Evaluierung — gleicht Entscheidungs-Journal gegen tatsächliche Kurse ab.

Liest journal/decisions.csv und bewertet jede Entscheidung (KAUFEN/VERKAUFEN/HALTEN)
gegen die tatsächliche Kursentwicklung via yfinance. Aggregiert Hit-Rate,
Rendite, Zielkurs-/Stop-Quoten und Konfidenz-Korrelation.

Semantik (Phase 1): HALTEN ist KEIN Trade und KEINE Richtungsprognose
("kein Handlungsbedarf"). HALTEN-Zeilen werden weiterhin bewertet (hit/rendite
für Transparenz), gehen aber NICHT mehr in hit_rate_gesamt oder die
Konfidenz-Kalibrierung (Brier-Score, Gap, Reliability-Bänder) ein — diese
Kennzahlen messen nur echte Trades (KAUFEN/VERKAUFEN). HALTEN wird separat
als deskriptive Kennzahl ausgewiesen: halten_n (Anzahl) und halten_quote
(Anteil stabiler Verläufe, |rendite| <= 2 %).

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
from .llm import LLMClient, StructuredChatResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Invalidierungs-Prüfung (Stufe 1 — STANDALONE Transparenz-Metrik)
# --------------------------------------------------------------------------- #
#
# Jeder Analyst dokumentiert seit Stufe 1 im Journal-Feld ``invalidation``
# freitextlich, WAS seine These widerlegen würde (z. B. "KGV über 25",
# "Bruch unter SMA200"). Diese Bedingung ist menschlicher Prosa — sie
# mechanisch zu parsen (Regex auf "KGV über X") wäre brüchig und leicht zu
# fälschen. Deshalb: pro bewertbarer Journal-Zeile EIN LLM-Call, der mit der
# Invalidierungs-Bedingung + dem realisierten Kurskontext boolsch beantwortet,
# ob eine der Bedingungen verletzt wurde.
#
# WICHTIG (Flo): Diese Metrik ist ein STANDALONE Transparenz-Feature
# ("Invalidierungs-Trefferquote"). Sie verändert NICHT die bestehende
# Hit-Definition — hit/rendite/hit_rate_gesamt werden UNVERÄNDERT berechnet.
#
# Kosten-Gate: Der LLM-Check läuft nur für Zeilen mit
#   (a) nicht-leerer ``invalidation``-Spalte UND
#   (b) bewertbarem Kurs-Outcome (Preise geladen, Rendite berechnet).
# Ohne LLM (llm=None) oder bei Fehler → deterministischer Fallback None
# ("nicht bewertbar") — die Zeile zählt weder als Hit noch als Nicht-Hit
# und die Quote bleibt leer. Crasht nie.

_SYSTEM_INVALIDATION_CHECK = (
    "Du bist ein strenger Fakten-Prüfer für Trading-Entscheidungen. Du bekommst "
    "die Invalidierungs-Bedingung(en) einer Analysten-These und den "
    "realisierten Kursverlauf. Prüfe NUR, ob mindestens eine Bedingung "
    "laut Text verletzt wurde — bewerte NICHT, ob die These insgesamt gut war."
)

_USER_INVALIDATION_CHECK_TEMPLATE = (
    "Invalidierungs-Bedingung(en) des Analysten:\n{invalidation}\n\n"
    "Realisierter Kursverlauf im Bewertungszeitraum:\n{context}\n\n"
    "Frage: Wurde mindestens eine der Invalidierungs-Bedingungen verletzt?\n"
    "Antworte AUSSCHLIESSLICH mit JSON:\n"
    '{{"invalidiert": true|false, "begruendung": "kurze Begründung auf Deutsch"}}'
)


def _invalidation_price_context(
    eval_result: dict[str, Any],
) -> str:
    """Baut einen kompakten, deterministischen Kurskontext-Text für den LLM-Check.

    Enthält Aktion, Ticker, Zeitraum-Rendite, Ziel-/Stop-Ausgang. Alle Werte
    stammen aus dem bereits berechneten eval_result (keine zusätzlichen
    yfinance-Aufrufe) — fehlende Werte werden als "n/a" dargestellt.
    """
    parts: list[str] = [
        f"Ticker: {eval_result.get('ticker', 'n/a')}",
        f"Aktion: {eval_result.get('action', 'n/a')}",
        f"Entscheidungszeitpunkt: {eval_result.get('timestamp', 'n/a')}",
    ]
    rendite = eval_result.get("rendite_pct")
    parts.append(
        "Rendite im Bewertungszeitraum: "
        + (f"{rendite:.2f} %" if isinstance(rendite, int | float) else "n/a")
    )
    ziel = eval_result.get("ziel_erreicht")
    parts.append(
        "Zielkurs erreicht: "
        + ("ja" if ziel is True else "nein" if ziel is False else "n/a")
    )
    stop = eval_result.get("stop_gerissen")
    parts.append(
        "Stop gerissen: "
        + ("ja" if stop is True else "nein" if stop is False else "n/a")
    )
    return "\n".join(parts)


def _parse_invalidation_answer(text: str) -> tuple[bool, str] | None:
    """Parst die LLM-Antwort der Invalidierungs-Prüfung (tolerant).

    Erwartet JSON {"invalidiert": bool, "begruendung": str}. Akzeptiert auch
    Booleans als String ("true"/"false"/"ja"/"nein"). Gibt (bool, begründung)
    zurück oder None bei unlesbarer Antwort (→ Fallback, kein Crash).
    """
    from .agents import parse_json  # lokaler Import — vermeidet Zyklen beim Modul-Load

    data = parse_json(text or "")
    if not isinstance(data, dict):
        return None
    raw = data.get("invalidiert")
    if raw is None:
        return None
    if isinstance(raw, bool):
        verdict = raw
    else:
        s = str(raw).strip().lower()
        if s in ("true", "ja", "yes", "1"):
            verdict = True
        elif s in ("false", "nein", "no", "0"):
            verdict = False
        else:
            return None
    begr = str(data.get("begruendung") or "").strip()
    return verdict, begr


def check_invalidation_hit(
    invalidation: str,
    eval_result: dict[str, Any],
    llm: LLMClient | None,
) -> tuple[bool, str] | None:
    """Prüft per LLM, ob die Invalidierungs-Bedingung verletzt wurde.

    Ein Call pro bewertbarer Zeile (nur wenn ``invalidation`` nicht leer und
    ``llm`` gegeben). Deterministischer Fallback: None bei fehlendem LLM,
    Fehler oder unlesbarer Antwort — die Zeile wird dann als "nicht
    bewertbar" gezählt (weder Hit noch Miss).

    Returns:
        (invalidiert, begründung) oder None (Fallback / nicht bewertbar).
    """
    if llm is None:
        return None
    text = str(invalidation or "").strip()
    if not text:
        return None
    try:
        prompt = _USER_INVALIDATION_CHECK_TEMPLATE.format(
            invalidation=text,
            context=_invalidation_price_context(eval_result),
        )
        messages = [
            {"role": "system", "content": _SYSTEM_INVALIDATION_CHECK},
            {"role": "user", "content": prompt},
        ]
        answer = llm.chat(messages, temperature=0.0, max_tokens=2000)
        answer_text = str(answer) if not isinstance(answer, StructuredChatResult) else answer.text
        return _parse_invalidation_answer(answer_text)
    except Exception as exc:  # noqa: BLE001 — best effort, nie crashen
        logger.warning("Invalidierungs-Prüfung fehlgeschlagen: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Regionale Benchmark-Map (analog TradingAgents' benchmark_map)
# --------------------------------------------------------------------------- #

# Map: Börsen-Suffix (Upper-Case, ohne führenden Punkt) → Benchmark-Ticker.
# "" (kein Suffix) = US-Markt → SPY. Unbekannte Suffixe fallen auf SPY zurück.
_BENCHMARK_MAP: dict[str, str] = {
    "DE": "^GDAXI",     # DAX (Deutschland)
    "L": "^FTSE",       # FTSE 100 (UK)
    "T": "^N225",       # Nikkei 225 (Japan)
    "HK": "^HSI",       # Hang Seng (Hongkong)
    "NS": "^NSEI",      # Nifty 50 (Indien, NSE)
    "BO": "^BSESN",     # Sensex (Indien, BSE)
    "TO": "^GSPTSE",    # TSX Composite (Kanada)
    "AX": "^AXJO",      # ASX 200 (Australien)
    "SS": "000001.SS",  # SSE Composite (Shanghai)
    "SZ": "399001.SZ",  # SZSE Component (Shenzhen)
}


def benchmark_for_ticker(ticker: Any) -> str:
    """Leitet den Benchmark-Index deterministisch aus dem Börsen-Suffix ab.

    Analog TradingAgents' ``benchmark_map``: "RWE.DE" → ^GDAXI (DAX),
    "SHEL.L" → ^FTSE (FTSE 100), "7203.T" → ^N225 (Nikkei 225),
    "AAPL" (kein Suffix, US) → SPY. Unbekannte Suffixe → SPY (Fallback).

    Crasht nie — gibt immer einen String zurück.

    Args:
        ticker: Beliebiger Ticker-String (oder None/anderer Typ).

    Returns:
        Benchmark-Ticker-String ("^GDAXI", "^FTSE", "SPY", ...).
    """
    try:
        ticker_str = str(ticker or "").strip().upper()
        suffix = ticker_str.rsplit(".", 1)[-1] if "." in ticker_str else ""
        return _BENCHMARK_MAP.get(suffix, "SPY")
    except Exception:  # noqa: BLE001 — crasht nie
        return "SPY"


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


def _delete_price_cache(ticker: str, today_key: str | None = None) -> None:
    """Entfernt den Tages-Cache-Eintrag für einen Ticker (best effort).

    Wird für den Retry-Mechanismus verwendet: Wenn ein leerer oder korrupter
    Cache-Eintrag das Laden von Kursdaten blockiert, kann der Cache-Eintrag
    gelöscht und erneut von yfinance geladen werden.

    Crasht niemals — Löschen ist best effort.
    """
    cache_dir = _get_cache_dir()
    if cache_dir is None:
        return
    if today_key is None:
        today_key = _get_today_key()

    path = _price_cache_path(cache_dir, today_key, ticker)
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.info("Price-Cache gelöscht für %s (%s)", ticker, today_key)
    except Exception as exc:  # noqa: BLE001 — Cache-Löschen crasht nie
        logger.debug("Price-Cache-Löschen fehlgeschlagen für %s: %s", ticker, exc)


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
        # Fenster deckt auch den Evaluierungszeitraum ab: _evaluate_single
        # bewertet bis decision_date + 90d (eval_end). lookback_days allein
        # reicht daher nicht — bei älteren Entscheidungen fehlt sonst der
        # Kurs am eval_end und exit_row fällt auf den letzten verfügbaren
        # Kurs zurück (verfälschte Rendite). 2*lookback + 60 deckt
        # decision_date bis ~lookback+60 Tage zurück ab.
        period_days = max(lookback_days * 2 + 60, 120)
        t = yf.Ticker(ticker)
        hist = t.history(period=f"{period_days}d", auto_adjust=False)
        if hist is None or hist.empty:
            return None

        records: list[dict[str, Any]] = []
        for date, row in hist.iterrows():
            records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "close": _safe_float(row["Close"]),
                    "high": _safe_float(row["High"]),
                    "low": _safe_float(row["Low"]),
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
    """Konvertiert einen Wert sicher zu float oder None (NaN/±inf → None)."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        f = float(s)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None  # NaN/Inf (z. B. aus Kursdaten) darf nie weiterfließen
    return f


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

# 5-stufige Rating-Skala (Index-Mapping)
_RATING_INDEX_MAP = {
    "STARK KAUFEN": 0,
    "KAUFEN": 1,
    "HALTEN": 2,
    "VERKAUFEN": 3,
    "STARK VERKAUFEN": 4,
}


def _rating_index(rating: str) -> int | None:
    """Mapt eine 5-stufige Bewertung auf ihren Index 0..4.

    Unknown/leer -> None.
    """
    r = (rating or "").strip().upper()
    return _RATING_INDEX_MAP.get(r)


def _outcome_rating_index(rendite_pct: float | None) -> int | None:
    """Mapt die tatsächliche Rendite auf einen 5-stufigen Outcome-Index.

    Regel (kommentiert):
      rendite > +2%   -> STARK KAUFEN (0)
      0%..+2%         -> KAUFEN (1)
      -2%..0%         -> VERKAUFEN (3)  [leicht negativ = bearish]
      < -2%           -> STARK VERKAUFEN (4)
      |rendite| <= 2% aber rund um 0 -> HALTEN (2) nur bei |rendite| <= 2%

    Praktisch: >+2% -> 0, >0% -> 1, <=-2% -> 4, <0% -> 3, sonst -> 2.
    """
    if rendite_pct is None or not math.isfinite(rendite_pct):
        return None
    if rendite_pct > 2.0:
        return 0  # STARK KAUFEN
    if rendite_pct > 0.0:
        return 1  # KAUFEN
    if rendite_pct < -2.0:
        return 4  # STARK VERKAUFEN
    if rendite_pct < 0.0:
        return 3  # VERKAUFEN
    return 2  # HALTEN (rendite == 0 oder sehr kleine Schwankung)


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
        portfolio_fit_score (float|None), ticker (str), timestamp (str),
        ist_trade (bool — True nur für KAUFEN/VERKAUFEN; HALTEN ist kein Trade).
    """
    action = (row.get("action") or "").strip().upper()
    timestamp = row.get("timestamp", "")
    decision_date = _parse_timestamp(timestamp)
    rating = (row.get("rating") or "").strip().upper()
    ist_trade = action in ("KAUFEN", "VERKAUFEN")

    # Leeres Ergebnis bei unbrauchbaren Daten
    empty = {
        "hit": None,
        "rendite_pct": None,
        "ziel_erreicht": None,
        "stop_gerissen": None,
        "action": action,
        "rating": rating,
        "rating_distance": None,
        "confidence": _safe_float(row.get("confidence")),
        "portfolio_fit_score": _safe_float(row.get("portfolio_fit_score")),
        "ticker": row.get("ticker", ""),
        "timestamp": timestamp,
        "ist_trade": ist_trade,
    }

    if decision_date is None or not prices:
        return empty

    # Entry-Preis: Kurs am oder vor Entscheidungsdatum
    entry = _find_price_on_or_before(prices, decision_date)
    if entry is None:
        entry = prices[0]  # Fallback: erster verfügbarer Kurs
    entry_price = _safe_float(entry.get("close")) if entry else None
    if entry_price is None or not math.isfinite(entry_price) or entry_price <= 0:
        return empty

    # Exit-Preis: heute oder lookback_days nach Entscheidung, je nachdem was früher
    today = datetime.now()
    end_date = decision_date + timedelta(days=lookback_days)
    eval_end = min(end_date, today)

    exit_row = _find_price_on_or_before(prices, eval_end)
    if exit_row is None:
        exit_row = prices[-1]  # Fallback: letzter verfügbarer Kurs
    exit_price = _safe_float(exit_row.get("close")) if exit_row else None
    if exit_price is None or not math.isfinite(exit_price):
        return empty

    # Rendite berechnen
    price_change_pct = (exit_price - entry_price) / entry_price * 100.0

    # Für VERKAUFEN: Rendite invertieren (Gewinn wenn Kurs fällt)
    if action == "VERKAUFEN":
        rendite_pct = -price_change_pct
    else:
        rendite_pct = price_change_pct

    # NaN/Inf-Rendite (z. B. durch NaN-Kurse) → Zeile als leer werten
    if not math.isfinite(rendite_pct):
        return empty

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
                p.get("low") is not None
                and math.isfinite(float(p["low"]))
                and float(p["low"]) <= target
                for p in period_prices
            )
        else:
            # Kauf/Halten: Ziel liegt oberhalb → Treffer wenn High ≥ target
            ziel_erreicht = any(
                p.get("high") is not None
                and math.isfinite(float(p["high"]))
                and float(p["high"]) >= target
                for p in period_prices
            )

    # Stop-Check
    stop = _safe_float(row.get("stop"))
    stop_gerissen: bool | None = None
    if stop is not None and stop > 0 and period_prices:
        if action == "VERKAUFEN":
            # Verkauf: Stop liegt oberhalb → gerissen wenn High ≥ stop
            stop_gerissen = any(
                p.get("high") is not None
                and math.isfinite(float(p["high"]))
                and float(p["high"]) >= stop
                for p in period_prices
            )
        else:
            # Kauf/Halten: Stop liegt unterhalb → gerissen wenn Low ≤ stop
            stop_gerissen = any(
                p.get("low") is not None
                and math.isfinite(float(p["low"]))
                and float(p["low"]) <= stop
                for p in period_prices
            )

    # Hit-Bestimmung — ehrliche Logik (Option 3):
    #   1. Stop gerissen → Miss (Risikoregel verletzt, hat Vorrang vor allem)
    #   2. Endrendite ist die PRIMÄRE Hit-Bedingung: Nur ein am Ende profitabler
    #      Trade ist ein Hit. "Ziel erreicht" allein reicht nicht mehr, wenn der
    #      Trade am Ende negativ war (z. B. "Ziel erreicht, aber Position nicht
    #      geführt" — der Gewinn war unrealisiert und wurde zurückgegeben).
    #   ziel_erreicht bleibt als separates Feld im Rückgabedict (Transparenz),
    #   fließt aber nicht mehr direkt in hit ein.
    # stop_gerissen / ziel_erreicht is None (nicht angegeben) → Bedingung überspringen.
    hit: bool | None = None
    if action in ("KAUFEN", "VERKAUFEN"):
        if stop_gerissen is True:
            hit = False
        elif rendite_pct is not None and rendite_pct > 0:
            hit = True
        else:
            # Ziel erreicht, aber Trade am Ende nicht profitabel → kein Hit
            hit = False
    elif action == "HALTEN":
        if stop_gerissen is True:
            hit = False
        else:
            # Halten ist "richtig" wenn Kurs ±2% stabil blieb (verschärft von ±5%)
            hit = abs(rendite_pct) <= 2.0

    # Rating-Distanz: Abstand zwischen bewerteter Aktion und tatsächlichem Outcome
    rating_distance: int | None = None
    rating_idx = _rating_index(rating)
    outcome_idx = _outcome_rating_index(rendite_pct)
    if rating_idx is not None and outcome_idx is not None:
        rating_distance = abs(rating_idx - outcome_idx)

    return {
        "hit": hit,
        "rendite_pct": rendite_pct,
        "ziel_erreicht": ziel_erreicht,
        "stop_gerissen": stop_gerissen,
        "action": action,
        "rating": rating,
        "rating_distance": rating_distance,
        "confidence": _safe_float(row.get("confidence")),
        "portfolio_fit_score": _safe_float(row.get("portfolio_fit_score")),
        "ticker": row.get("ticker", ""),
        "timestamp": timestamp,
        "ist_trade": ist_trade,
    }


# --------------------------------------------------------------------------- #
# Statistische Signifikanz: DSR / MBL (Bailey & López de Prado 2014)
# --------------------------------------------------------------------------- #
#
# STANDALONE, strikt additive Metriken — reine Mathematik (nur `math`),
# netzfrei, kein LLM-Call, deterministisch. Sie verändern NICHT
# hit_rate_gesamt, den Brier-Score oder irgendeine bestehende Kennzahl.
#
#   DSR — Deflated Sharpe Ratio:
#     Bailey, D. H. & López de Prado, M. (2014): "The Deflated Sharpe Ratio:
#     Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
#     Journal of Portfolio Management 40(5).
#   MBL / MinTRL — Minimale Backtest-Länge:
#     Bailey, D. H. & López de Prado, M. (2012): "The Sharpe Ratio Efficient
#     Frontier", Journal of Risk 15(2) — im selben PSR-Rahmen wie der
#     DSR-Artikel von 2014.
#
# Implementierte Formeln (in den Docstrings zitiert und als Approximation
# gekennzeichnet):
#   σ(SR)    = sqrt( (1 − γ3·SR + (γ4−1)/4·SR²) / (n−1) )           [Mertens/Lo]
#   PSR(SR*) = Φ( (SR − SR*) / σ(SR) )                              [Bailey/LdP 2012]
#   SR0      = √V · ( (1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)) ),
#              γ = Euler-Mascheroni ≈ 0.5772                        [Bailey/LdP 2014, Gl. 1/6]
#   DSR      = PSR(SR0)                                             [Bailey/LdP 2014, Gl. 2]
#   MinTRL   = 1 + z_p² · (1 − γ3·SR_p + (γ4−1)/4·SR_p²) / SR_p²    [vereinfachte publizierte Variante]

DEFAULT_N_TRIALS: int = 30
"""Konservativer Default für die Anzahl unabhängiger Trials (Versuche).

Wird in compute_dsr verwendet, wenn kein konkreter Zähler übergeben wird.
In _aggregate gilt stattdessen: n_trials = Anzahl der bewerteten
Entscheidungen (jede Journal-Zeile ist ein "Versuch" des Systems),
mindestens 2. Beide Konventionen sind bewusst konservativ: mehr Trials →
höherer Schwellenwert E[max SR] → niedrigerer DSR.
"""

_MIN_TRADE_RETURNS_FUER_SIGNIFIKANZ: int = 10
"""Mindestanzahl Trade-Renditen (KAUFEN/VERKAUFEN) für DSR/MBL.

Darunter bleiben dsr/mbl None und es wird ein deutscher Hinweis
(siginifikanz_hinweis) gesetzt — nie crashen.
"""

_EULER_MASCHERONI: float = 0.5772156649015329
_SQRT_2PI: float = math.sqrt(2.0 * math.pi)
_JAHRESFAKTOR_SIGNIFIKANZ: float = 252.0
"""Annualisierungsfaktor der Trade-Return-Sharpe (Konvention wie backtest.py)."""


def _norm_cdf(x: float) -> float:
    """CDF der Standardnormalverteilung Φ(x) via math.erf (netzfrei, kein scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float | None:
    """Inverse Standardnormale Φ⁻¹(p) — deterministische Approximation, nur `math`.

    Startwert: Abramowitz & Stegun 26.2.23 (|ε| < 4.4e-4), dann vier
    Halley-Verfeinerungsschritte gegen die exakte erf-basierte Φ. Damit ist
    die Quantil-Approximation auf ~Maschinengenauigkeit genau — ohne scipy.
    Nicht-finite Eingaben → None (nie crashen).
    """
    try:
        p_f = float(p)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p_f):
        return None
    p_clamped = min(max(p_f, 1e-15), 1.0 - 1e-15)
    lower = p_clamped < 0.5
    q = p_clamped if lower else 1.0 - p_clamped
    t = math.sqrt(math.log(1.0 / (q * q)))
    x = t - (2.515517 + 0.802853 * t) / (1.0 + 1.432788 * t + 0.189269 * t * t)
    x = -x if lower else x
    for _ in range(4):
        fehler = _norm_cdf(x) - p_clamped
        dichte = math.exp(-0.5 * x * x) / _SQRT_2PI
        denom = 2.0 * dichte + x * fehler
        if abs(denom) < 1e-300:
            break
        x -= 2.0 * fehler / denom
    return x


def _sharpe_var_term(sharpe: float, skew: float, kurt: float) -> float | None:
    """Varianz-Term der PSR/DSR-Formel: 1 − γ3·SR + (γ4−1)/4·SR².

    γ3 = Schiefe, γ4 = Kurtosis (raw; Normalverteilung = 3).
    Nicht-positive oder nicht-finite Werte → None.
    """
    var_term = 1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe * sharpe
    if not math.isfinite(var_term) or var_term <= 0.0:
        return None
    return var_term


def _probst(
    sharpe: float | None,
    n: int | None,
    skew: float = 0.0,
    kurt: float = 3.0,
    sharpe_benchmark: float = 0.0,
) -> float | None:
    """Probabilistic Sharpe Ratio PSR(SR*): P[wahre Strategie-Schärfe > SR*].

    Formel (Bailey & López de Prado 2012, "The Sharpe Ratio Efficient
    Frontier", Journal of Risk 15(2); referenziert in Bailey & López de
    Prado 2014, "The Deflated Sharpe Ratio", Journal of Portfolio Management
    40(5), Gl. 2 mit SR* = 0):

        PSR(SR*) = Φ( (SR − SR*) · sqrt(n−1)
                      / sqrt(1 − γ3·SR + (γ4−1)/4·SR²) )

    mit SR = Sharpe pro Rendite-Beobachtung (hier: pro Trade), n = Anzahl
    Renditen, γ3 = Schiefe, γ4 = Kurtosis (raw, Normalfall 3).
    Φ ist die CDF der Standardnormalverteilung (via math.erf, netzfrei).

    Args:
        sharpe: Beobachteter Sharpe pro Beobachtung (NICHT annualisiert).
        n: Anzahl Rendite-Beobachtungen (n >= 2 nötig).
        skew: Schiefe der Renditen (Default 0 = normal).
        kurt: Kurtosis der Renditen, raw (Default 3 = normal).
        sharpe_benchmark: Schwellenwert SR* (0 für PSR; E[max SR] für DSR).

    Returns:
        PSR in (0, 1) oder None bei degenerate Eingaben (n < 2, None,
        nicht-finite Werte, Varianz-Term ≤ 0). Crasht nie.
    """
    try:
        if sharpe is None or n is None:
            return None
        n_i = int(n)
        if n_i < 2:
            return None
        sr = float(sharpe)
        g3 = float(skew) if skew is not None else 0.0
        g4 = float(kurt) if kurt is not None else 3.0
        sr0 = float(sharpe_benchmark) if sharpe_benchmark is not None else 0.0
        if not (math.isfinite(sr) and math.isfinite(g3)
                and math.isfinite(g4) and math.isfinite(sr0)):
            return None
        var_term = _sharpe_var_term(sr, g3, g4)
        if var_term is None:
            return None
        sigma = math.sqrt(var_term / (n_i - 1))
        return _norm_cdf((sr - sr0) / sigma)
    except (TypeError, ValueError, OverflowError):
        return None


def compute_dsr(
    sharpe: float | None,
    n_obs: int | None,
    skew: float = 0.0,
    kurt: float = 3.0,
    n_trials: int = DEFAULT_N_TRIALS,
    var_trials: float | None = None,
) -> float | None:
    """Deflated Sharpe Ratio (DSR) nach Bailey & López de Prado (2014).

    Antwortet die Frage: Wie wahrscheinlich ist es, dass die wahre
    Strategie-Schärfe > 0 ist, WENN man berücksichtigt, dass das Ergebnis
    aus N unabhängigen Trials ("Suchaufwand") ausgewählt wurde?

    Implementierte Approximation der Papier-Formeln (Bailey, D. H. & López
    de Prado, M. (2014): "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting and Non-Normality", Journal of
    Portfolio Management 40(5); siehe auch davidhbailey.com/dhbpapers/
    deflated-sharpe.pdf, Gl. 1/2/6):

        σ(SR)  = sqrt( (1 − γ3·SR + (γ4−1)/4·SR²) / (n−1) )
        SR0    = sqrt(V) · ( (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) )
        DSR    = Φ( (SR − SR0) / σ(SR) )

    mit γ ≈ 0.5772 (Euler-Mascheroni), e = Eulersche Zahl, N = Anzahl
    unabhängiger Trials, n = Anzahl Renditen, γ3/γ4 = Schiefe/Kurtosis der
    Renditen. SR0 ist der erwartete Maximal-Sharpe über N Trials unter der
    Nullhypothese (kein Können) — für großes N approximiert der
    Euler-Mascheroni-Term √(2·ln N)·√V, also genau die in der Literatur
    zitierte "expected best Sharpe from N independent trials"-Form.

    Achtung (Approximationen, bewusst dokumentiert):
      * V (Varianz der SR-Schätzungen über die Trials) liegt hier typischer-
        weise nicht empirisch vor. Default (var_trials=None): V wird
        KONSERVATIV durch die Sampling-Varianz des beobachteten SR-Schätzers
        genähert, V = σ(SR)² = (1 − γ3·SR + (γ4−1)/4·SR²)/(n−1). Das
        entspricht dem "unter der Null"-Fall und liefert den aus der Aufgabe
        bekannten √(2·ln(N)/n)-artigen Term. Mit var_trials kann die
        empirische Trial-Varianz (gleiche Einheit wie SR!) eingesetzt werden.
      * Die Trials werden als unabhängig angenommen (Papier, Appendix 3
        behandelt Korrelationen — hier ohne empirische Trial-Verteilung
        nicht verfügbar).
      * SR wird in PER-BEOBACHTUNG-Einheiten erwartet (hier: pro Trade). Der
        DSR-Test ist skaleninvariant, ein Annualisierungsfaktor kürzt sich.

    Args:
        sharpe: Sharpe pro Rendite-Beobachtung (per Trade, nicht annualisiert).
        n_obs: Anzahl Rendite-Beobachtungen (Trades).
        skew: Schiefe der Renditen (Default 0).
        kurt: Kurtosis der Renditen, raw (Default 3 = normal).
        n_trials: Anzahl unabhängiger Trials (Default DEFAULT_N_TRIALS = 30;
            in _aggregate: Anzahl bewerteter Entscheidungen, min. 2).
        var_trials: Optionale Varianz der Trial-SRs (gleiche Einheit wie
            sharpe). None → konservativer Proxy σ(SR)².

    Returns:
        DSR in (0, 1) oder None bei degenerate Eingaben (nie crashen).
    """
    try:
        if sharpe is None or n_obs is None:
            return None
        n_i = int(n_obs)
        if n_i < 2:
            return None
        sr = float(sharpe)
        g3 = float(skew) if skew is not None else 0.0
        g4 = float(kurt) if kurt is not None else 3.0
        if not (math.isfinite(sr) and math.isfinite(g3) and math.isfinite(g4)):
            return None
        var_term = _sharpe_var_term(sr, g3, g4)
        if var_term is None:
            return None
        sigma2 = var_term / (n_i - 1)  # Sampling-Varianz des SR-Schätzers

        if var_trials is not None:
            v = float(var_trials)
            if not math.isfinite(v) or v <= 0.0:
                return None
        else:
            v = sigma2  # konservativer Proxy (dokumentiert oben)

        n_trials_i = int(n_trials) if n_trials is not None else DEFAULT_N_TRIALS
        if n_trials_i < 2:
            n_trials_i = 2  # Φ⁻¹(1−1/N) für N=1 → Grenzfall, bewusst geklemmt

        z1 = _norm_ppf(1.0 - 1.0 / n_trials_i)
        z2 = _norm_ppf(1.0 - 1.0 / (n_trials_i * math.e))
        if z1 is None or z2 is None:
            return None
        sr0 = math.sqrt(v) * (
            (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
        )
        if not math.isfinite(sr0):
            return None
        return _probst(sr, n_i, g3, g4, sharpe_benchmark=sr0)
    except (TypeError, ValueError, OverflowError):
        return None


def compute_mbl(
    sharpe: float | None,
    annualization: float = 252.0,
    pval: float = 0.05,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float | None:
    """Minimale Backtest-Länge (MBL / MinTRL) nach Bailey & López de Prado.

    Mindestanzahl Rendite-Beobachtungen (Perioden; hier: Trades), damit ein
    gemessener Sharpe signifikant von 0 unterscheidbar ist (einseitig,
    Default p < 0.05).

    Implementierte Formel — vereinfachte, publizierte Variante aus dem
    PSR-Rahmen (Bailey & López de Prado 2012, "The Sharpe Ratio Efficient
    Frontier", Journal of Risk 15(2); Bailey & López de Prado 2014, "The
    Deflated Sharpe Ratio", Journal of Portfolio Management 40(5)):

        MinTRL = 1 + z_p² · (1 − γ3·SR_p + (γ4−1)/4·SR_p²) / SR_p²

    Herleitung: PSR(0) ≥ 1 − p ⇔ (SR_p − 0)·sqrt(T−1)/sqrt(1 − γ3·SR_p +
    (γ4−1)/4·SR_p²) ≥ z_p ⇒ T ≥ 1 + z_p²·(1 − γ3·SR_p + (γ4−1)/4·SR_p²)/SR_p².
    Normalfall (γ3 = 0, γ4 = 3): MinTRL = 1 + z_p²/SR_p².
    Der Moment-Term wird auf dem PERIODEN-Sharpe SR_p ausgewertet
    (so ist die Formel im Papier hergeleitet).

    Args:
        sharpe: ANNUALISIERTER Sharpe (Konvention wie im Backtest-Modul:
            Ø/std · sqrt(annualization)). Wird intern auf die Perioden-
            einheit zurückgerechnet (SR_p = sharpe / sqrt(annualization));
            für Trade-Renditen ist eine "Periode" ein Trade.
        annualization: Perioden pro Jahr (Default 252.0). None → 252.
        pval: Einseitiges Signifikanzniveau γ (Default 0.05).
        skew: Schiefe der Renditen (Default 0).
        kurt: Kurtosis der Renditen, raw (Default 3 = normal).

    Returns:
        Mindestanzahl Beobachtungen (float, ≥ 1) oder None, wenn der Sharpe
        fehlt, ≤ 0 ist (dann nie "signifikant > 0") oder die Eingaben
        degenieren. Crasht nie.
    """
    try:
        if sharpe is None:
            return None
        if pval is None or not (0.0 < float(pval) < 1.0):
            return None
        ann = 252 if annualization is None else float(annualization)
        if not math.isfinite(ann) or ann <= 0.0:
            return None
        sr_ann = float(sharpe)
        if not math.isfinite(sr_ann) or sr_ann <= 0.0:
            return None
        g3 = float(skew) if skew is not None else 0.0
        g4 = float(kurt) if kurt is not None else 3.0
        if not (math.isfinite(g3) and math.isfinite(g4)):
            return None
        z = _norm_ppf(1.0 - float(pval))
        if z is None:
            return None
        sr_p = sr_ann / math.sqrt(ann)
        adj = _sharpe_var_term(sr_p, g3, g4)
        if adj is None:
            return None
        return 1.0 + z * z * adj / (sr_p * sr_p)
    except (TypeError, ValueError, OverflowError):
        return None


def _moments_skew_kurt(
    renditen: list[float],
) -> tuple[float | None, float | None]:
    """Schiefe (γ3) und Kurtosis (γ4, raw) einer Rendite-Liste.

    Fisher-Pearson g1/g2 (konsistente, "bias-behaftete" Stichprobenmomente,
    analog scipy.stats.skew/kurtosis mit bias=True):
        g1 = m3 / m2^1.5
        g2 = m4 / m2² − 3   (exzessive Kurtosis)
        γ4 (raw) = g2 + 3
    Deterministisch, nur math. Bei < 3 Punkten oder Varianz ≤ 0 → (None, None).
    """
    try:
        werte = [float(r) for r in renditen if r is not None]
        n = len(werte)
        if n < 3:
            return (None, None)
        mittel = sum(werte) / n
        m2 = sum((v - mittel) ** 2 for v in werte) / n
        if m2 <= 0.0 or not math.isfinite(m2):
            return (None, None)
        m3 = sum((v - mittel) ** 3 for v in werte) / n
        m4 = sum((v - mittel) ** 4 for v in werte) / n
        g1 = m3 / (m2 ** 1.5)
        g2 = m4 / (m2 * m2) - 3.0
        skew = g1 if math.isfinite(g1) else None
        kurt = (g2 + 3.0) if math.isfinite(g2) else None
        return (skew, kurt)
    except (TypeError, ValueError, OverflowError):
        return (None, None)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _empty_result() -> dict[str, Any]:
    """Leeres Ergebnis-dict (für fehlende/leere Journal-Datei).

    HALTEN-Zeilen zählen NICHT zu hit_rate_gesamt / Konfidenz-Kalibrierung
    (nur echte Trades KAUFEN/VERKAUFEN). HALTEN wird deskriptiv über
    halten_n / halten_quote ausgewiesen (beide None/0 im Leerfall).
    """
    return {
        "anzahl_entscheidungen": 0,
        "nach_aktion": {
            "KAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None, "avg_confidence": None},
            "HALTEN": {"n": 0, "hit_rate": None, "avg_rendite": None, "avg_confidence": None},
            "VERKAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None, "avg_confidence": None},
        },
        "hit_rate_gesamt": None,
        "halten_n": 0,
        "halten_quote": None,
        "durchschnitt_rendite_gesamt": None,
        "durchschnitt_rating_distanz": None,
        "zielkurs_trefferquote": None,
        "stop_verletzungsquote": None,
        "konfidenz_baende": [],
        "portfolio_fit_hoch": None,
        "zusammenfassung": None,
        "fehler": [],
        "konfidenz_kalibrierung": {
            "brier_score": None,
            "n": 0,
            "durchschnittliche_konfidenz": None,
            "durchschnittliche_tatsaechliche_hit_rate": None,
            "kalibrierungs_gap": None,
            "tendenz": None,
        },
        "konfidenz_kalibrierung_segmentiert": {
            "nach_aktion": {},
            "nach_rating": {},
        },
        "reliability_bins": [],
        "uebersprungen": 0,
        # Stufe 1: Invalidierungs-Trefferquote — STANDALONE Transparenz-Metrik.
        # invalidation_hit_quote = Anteil der bewertbaren Zeilen (mit nicht-
        # leerer invalidation-Spalte + LLM-Check), bei denen mindestens eine
        # Bedingung verletzt wurde. Verändert NICHT die Hit-Definition.
        "invalidation_hit_quote": None,
        "invalidation_n": 0,
        "invalidation_hits": 0,
        "invalidation_nicht_bewertbar": 0,
        "invalidation_details": [],
        # Statistische Signifikanz (DSR/MBL, Bailey & López de Prado 2014).
        # Strikt additive, deterministische Metriken — verändern KEINE
        # bestehende Kennzahl. Im Leerfall alles None ("zu wenige Trades").
        "dsr": None,  # Deflated Sharpe Ratio in (0, 1) | None
        "dsr_ps": None,  # unaufgeblasener PSR (SR > 0) zur Referenz | None
        "mbl": None,  # Minimale Backtest-Länge (Trades) | None
        "siginifikanz_hinweis": "",  # deutscher Hinweis bei zu wenigen Trades
        "signifikanz_details": {  # Transparenz (Eingaben der Formeln)
            "n_trials": 0,
            "sharpe_trades_annualisiert": None,
            "n_trade_renditen": 0,
        },
    }


def _compute_konfidenz_kalibrierung(
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Berechnet die Konfidenz-Kalibrierung (Brier-Score, Gap, Tendenz).

    Brier-Score (binär): Für jede bewertete Zeile mit confidence und hit:
        p = confidence / 5  (normalisierte Wahrscheinlichkeit 0.2..1.0)
        hit_int = 1 wenn hit True, 0 wenn hit False
        brier_i = (p - hit_int) ** 2
    Brier-Score = Ø aller brier_i (niedriger = besser, 0 = perfekt).

    Kalibrierungs-Gap = Ø_Konfidenz - Ø_Hit-Rate (positiv = überkonfident).

    Tendenz:
        gap > +0.15 → "überkonfident"
        gap < -0.15 → "unterkonfident"
        sonst       → "gut kalibriert"

    Nur Zeilen mit confidence (nicht None, isfinite) und hit (nicht None)
    werden verwendet. Bei 0 gültigen Zeilen → None-Werte.

    Phase 1: Es gehen NUR echte Trades (KAUFEN/VERKAUFEN) ein — HALTEN
    ("kein Handlungsbedarf") ist keine Richtungsprognose und wird ausgeschlossen.
    """
    empty = {
        "brier_score": None,
        "n": 0,
        "durchschnittliche_konfidenz": None,
        "durchschnittliche_tatsaechliche_hit_rate": None,
        "kalibrierungs_gap": None,
        "tendenz": None,
    }

    # Nur echte Trades mit confidence und hit verwenden
    valid: list[dict[str, Any]] = []
    for e in evaluations:
        if not _is_trade_eval(e):
            continue  # HALTEN fließt nicht in die Kalibrierung ein
        conf = e.get("confidence")
        hit = e.get("hit")
        if conf is None or hit is None:
            continue
        conf_f = float(conf)
        if not math.isfinite(conf_f) or conf_f <= 0:
            continue
        valid.append(e)

    if not valid:
        return empty

    n = len(valid)
    brier_sum = 0.0
    conf_sum = 0.0
    hit_sum = 0.0

    for e in valid:
        conf_f = float(e["confidence"])
        p = conf_f / 5.0
        hit_int = 1 if e["hit"] is True else 0
        brier_sum += (p - hit_int) ** 2
        conf_sum += p
        hit_sum += hit_int

    brier_score = brier_sum / n
    avg_conf = conf_sum / n
    avg_hit = hit_sum / n
    gap = avg_conf - avg_hit

    if gap > 0.15:
        tendenz = "überkonfident"
    elif gap < -0.15:
        tendenz = "unterkonfident"
    else:
        tendenz = "gut kalibriert"

    return {
        "brier_score": brier_score,
        "n": n,
        "durchschnittliche_konfidenz": avg_conf,
        "durchschnittliche_tatsaechliche_hit_rate": avg_hit,
        "kalibrierungs_gap": gap,
        "tendenz": tendenz,
    }


def _compute_konfidenz_kalibrierung_segmentiert(
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Berechnet segmentierte Brier-Scores pro Aktion und pro Rating-Stufe.

    Verwendet dieselbe Brier-Formel wie _compute_konfidenz_kalibrierung:
        p = confidence / 5  (normalisierte Wahrscheinlichkeit 0.2..1.0)
        hit_int = 1 wenn hit True, 0 wenn hit False
        brier_i = (p - hit_int) ** 2

    Segmente:
        - nach_aktion: KAUFEN, VERKAUFEN (HALTEN ist kein Trade → kein Segment)
        - nach_rating: STARK KAUFEN, KAUFEN, HALTEN, VERKAUFEN, STARK VERKAUFEN

    Leere Segmente (n=0) werden weggelassen.

    Phase 1: Der nach_aktion-Loop enthält nur KAUFEN/VERKAUFEN (HALTEN ist
    kein Trade). Die nach_rating-Segmentierung bleibt UNVERÄNDERT über alle
    übergebenen Zeilen berechnet — das HALTEN-Rating ist ein legitimes
    Rating-Segment (auch bei HALTEN-Aktionen).

    Returns:
        dict: {"nach_aktion": {action: {brier_score, n, ...}}, "nach_rating": {...}}
    """
    _AKTIONEN = ("KAUFEN", "VERKAUFEN")
    _RATINGS = ("STARK KAUFEN", "KAUFEN", "HALTEN", "VERKAUFEN", "STARK VERKAUFEN")

    def _compute_segment(segment_evals: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Berechnet Brier-Kalibrierung für eine Teilmenge von Evaluations."""
        valid: list[dict[str, Any]] = []
        for e in segment_evals:
            conf = e.get("confidence")
            hit = e.get("hit")
            if conf is None or hit is None:
                continue
            conf_f = float(conf)
            if not math.isfinite(conf_f) or conf_f <= 0:
                continue
            valid.append(e)

        if not valid:
            return None

        n = len(valid)
        brier_sum = 0.0
        conf_sum = 0.0
        hit_sum = 0.0

        for e in valid:
            conf_f = float(e["confidence"])
            p = conf_f / 5.0
            hit_int = 1 if e["hit"] is True else 0
            brier_sum += (p - hit_int) ** 2
            conf_sum += p
            hit_sum += hit_int

        brier_score = brier_sum / n
        avg_conf = conf_sum / n
        avg_hit = hit_sum / n
        gap = avg_conf - avg_hit

        if gap > 0.15:
            tendenz = "überkonfident"
        elif gap < -0.15:
            tendenz = "unterkonfident"
        else:
            tendenz = "gut kalibriert"

        return {
            "brier_score": brier_score,
            "n": n,
            "durchschnittliche_konfidenz": avg_conf,
            "durchschnittliche_tatsaechliche_hit_rate": avg_hit,
            "kalibrierungs_gap": gap,
            "tendenz": tendenz,
        }

    nach_aktion: dict[str, Any] = {}
    for action in _AKTIONEN:
        seg = _compute_segment(
            [e for e in evaluations if e.get("action") == action]
        )
        if seg is not None:
            nach_aktion[action] = seg

    nach_rating: dict[str, Any] = {}
    for rating in _RATINGS:
        seg = _compute_segment(
            [e for e in evaluations if (e.get("rating") or "").strip().upper() == rating]
        )
        if seg is not None:
            nach_rating[rating] = seg

    return {"nach_aktion": nach_aktion, "nach_rating": nach_rating}


# Reliability-Bin-Grenzen: [untere, obere) Grenzen
# [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0+1)
_RELIABILITY_BIN_EDGES: list[tuple[float, float]] = [
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.01),  # 1.01 um 1.0 inklusiv zu erfassen
]


def _compute_reliability_bins(
    evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Gruppiert bewertete Zeilen in Konfidenz-Intervalle (Reliability-Bänder).

    Bins: [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
    Pro Bin: n, mittlere_konfidenz (Ø p), hit_rate.

    Nur Zeilen mit confidence (nicht None, isfinite, > 0) und hit (nicht None).
    Leere Bins werden nicht in die Liste aufgenommen.

    Phase 1: Es gehen NUR echte Trades (KAUFEN/VERKAUFEN) ein — HALTEN ist
    keine Richtungsprognose und verzerrt die Kalibrierung.
    """
    valid: list[dict[str, Any]] = []
    for e in evaluations:
        if not _is_trade_eval(e):
            continue  # HALTEN fließt nicht in die Kalibrierung ein (Phase 1)
        conf = e.get("confidence")
        hit = e.get("hit")
        if conf is None or hit is None:
            continue
        conf_f = float(conf)
        if not math.isfinite(conf_f) or conf_f <= 0:
            continue
        valid.append(e)

    if not valid:
        return []

    bins: list[dict[str, Any]] = []
    for lo, hi in _RELIABILITY_BIN_EDGES:
        bin_evals = [
            e for e in valid
            if lo <= float(e["confidence"]) / 5.0 < hi
        ]
        if not bin_evals:
            continue
        n = len(bin_evals)
        conf_vals = [float(e["confidence"]) / 5.0 for e in bin_evals]
        hits = [e for e in bin_evals if e["hit"] is True]
        rated = len(hits) + len([e for e in bin_evals if e["hit"] is False])
        mittlere_konfidenz = sum(conf_vals) / n
        hit_rate = len(hits) / rated if rated > 0 else None
        bins.append(
            {
                "bin": f"[{lo:.1f}-{hi:.1f})" if hi <= 1.0 else f"[{lo:.1f}-1.0]",
                "n": n,
                "mittlere_konfidenz": mittlere_konfidenz,
                "hit_rate": hit_rate,
            }
        )
    return bins


def _is_trade_eval(e: dict[str, Any]) -> bool:
    """Gibt zurück, ob eine Einzel-Evaluierung ein echter Trade ist.

    HALTEN ist kein Trade ("kein Handlungsbedarf") und fließt nicht in
    hit_rate_gesamt oder die Konfidenz-Kalibrierung ein.
    Nutzt das explizite ``ist_trade``-Feld, falls vorhanden; sonst Fallback
    auf die Aktion (nur KAUFEN/VERKAUFEN gelten als Trades).
    """
    ist_trade = e.get("ist_trade")
    if isinstance(ist_trade, bool):
        return ist_trade
    return (e.get("action") or "").strip().upper() in ("KAUFEN", "VERKAUFEN")


def _aggregate(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregiert die Einzel-Ergebnisse zu einem Ergebnis-dict.

    Semantik (Phase 1): ``hit_rate_gesamt`` und die Konfidenz-Kalibrierung
    (Brier-Score, Gap, Tendenz, Reliability-Bänder, Segmentierung nach Aktion)
    berücksichtigen NUR echte Trades (KAUFEN/VERKAUFEN). HALTEN wird als
    separate deskriptive Kennzahl ausgewiesen (``halten_n``, ``halten_quote``).
    """
    result = _empty_result()
    result["anzahl_entscheidungen"] = len(evaluations)

    # --- HALTEN als deskriptive Kennzahl (kein Trade) ---
    halten_evals = [e for e in evaluations if e.get("action") == "HALTEN"]
    result["halten_n"] = len(halten_evals)
    halten_rated = [e for e in halten_evals if e.get("hit") is not None]
    halten_hits = [e for e in halten_rated if e["hit"] is True]
    result["halten_quote"] = (
        len(halten_hits) / len(halten_rated) if halten_rated else None
    )

    # --- Nur echte Trades (KAUFEN/VERKAUFEN) für Gesamt-/Kalibrierungs-Kennzahlen ---
    trades = [e for e in evaluations if _is_trade_eval(e)]

    # --- Nach Aktion ---
    for action in ("KAUFEN", "HALTEN", "VERKAUFEN"):
        action_evals = [e for e in evaluations if e.get("action") == action]
        n = len(action_evals)
        hits = [e for e in action_evals if e.get("hit") is True]
        misses = [e for e in action_evals if e.get("hit") is False]
        rated = len(hits) + len(misses)
        renditen = [
            e["rendite_pct"]
            for e in action_evals
            if e.get("rendite_pct") is not None
            and math.isfinite(e["rendite_pct"])
        ]

        # Ø Confidence pro Aktion (normalisiert auf 0-1: conf/5)
        conf_vals = [
            float(e["confidence"]) / 5.0
            for e in action_evals
            if e.get("confidence") is not None
            and math.isfinite(float(e["confidence"]))
            and float(e["confidence"]) > 0
        ]
        avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else None

        result["nach_aktion"][action] = {
            "n": n,
            "hit_rate": len(hits) / rated if rated > 0 else None,
            "avg_rendite": sum(renditen) / len(renditen) if renditen else None,
            "avg_confidence": avg_conf,
        }

    # --- Gesamt (nur echte Trades KAUFEN/VERKAUFEN; HALTEN ist kein Trade) ---
    all_hits = [e for e in trades if e.get("hit") is True]
    all_misses = [e for e in trades if e.get("hit") is False]
    all_rated = len(all_hits) + len(all_misses)
    result["hit_rate_gesamt"] = len(all_hits) / all_rated if all_rated > 0 else None

    all_renditen = [
        e["rendite_pct"]
        for e in trades
        if e.get("rendite_pct") is not None
        and math.isfinite(e["rendite_pct"])
    ]
    result["durchschnitt_rendite_gesamt"] = (
        sum(all_renditen) / len(all_renditen) if all_renditen else None
    )

    # --- Zielkurs-Trefferquote ---
    ziel_evals = [e for e in evaluations if e.get("ziel_erreicht") is not None]
    ziel_treffer = [e for e in ziel_evals if e.get("ziel_erreicht") is True]
    result["zielkurs_trefferquote"] = (
        len(ziel_treffer) / len(ziel_evals) if ziel_evals else None
    )

    # --- Stop-Verletzungsquote ---
    stop_evals = [e for e in evaluations if e.get("stop_gerissen") is not None]
    stop_hits = [e for e in stop_evals if e.get("stop_gerissen") is True]
    result["stop_verletzungsquote"] = (
        len(stop_hits) / len(stop_evals) if stop_evals else None
    )

    # --- Konfidenz-Bänder ---
    # hoch: confidence ≥ 4, mittel: 3, niedrig: ≤ 2
    bands = {"hoch": [], "mittel": [], "niedrig": []}
    for e in trades:
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
        hits = [e for e in band_evals if e.get("hit") is True]
        misses = [e for e in band_evals if e.get("hit") is False]
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
    pf_evals = [e for e in trades if e.get("portfolio_fit_score") is not None]
    pf_hoch = [e for e in pf_evals if (e.get("portfolio_fit_score") or 0) >= 4]
    if pf_hoch:
        pf_hits = [e for e in pf_hoch if e["hit"] is True]
        pf_misses = [e for e in pf_hoch if e["hit"] is False]
        pf_rated = len(pf_hits) + len(pf_misses)
        result["portfolio_fit_hoch"] = {
            "hit_rate": len(pf_hits) / pf_rated if pf_rated > 0 else None,
            "n": len(pf_hoch),
        }

    # --- Durchschnittliche Rating-Distanz ---
    # Nur Zeilen mit gültigem rating_distance (int, nicht None)
    rating_distances = [
        e["rating_distance"]
        for e in trades
        if e.get("rating_distance") is not None
    ]
    result["durchschnitt_rating_distanz"] = (
        sum(rating_distances) / len(rating_distances) if rating_distances else None
    )

    # --- Konfidenz-Kalibrierung (Brier-Score, Gap, Tendenz) ---
    # Nur echte Trades (KAUFEN/VERKAUFEN) — HALTEN ist keine Richtungsprognose.
    result["konfidenz_kalibrierung"] = _compute_konfidenz_kalibrierung(trades)

    # --- Segmentierte Konfidenz-Kalibrierung (pro Aktion, pro Rating) ---
    # nach_aktion: nur KAUFEN/VERKAUFEN (HALTEN ist kein Trade → kein Segment);
    # nach_rating: unverändert über ALLE Zeilen (HALTEN-Rating = legitimes Segment).
    # Deshalb hier evaluations (nicht trades) übergeben.
    result["konfidenz_kalibrierung_segmentiert"] = (
        _compute_konfidenz_kalibrierung_segmentiert(evaluations)
    )

    # --- Reliability-Bänder (feinere Konfidenz-Intervalle) ---
    result["reliability_bins"] = _compute_reliability_bins(trades)

    # --- Invalidierungs-Trefferquote (Stufe 1, STANDALONE) ---------------------
    # Bewusst NICHT mit hit_rate_gesamt vermischt: invalidation_hit_quote misst
    # nur, wie oft die dokumentierten Thesen-Bedingungen faktisch verletzt
    # wurden. Zeilen ohne Bewertung (kein LLM-Check) fließen nicht ein.
    inv_evals = [
        e for e in evaluations if e.get("invalidation_hit") is not None
    ]
    inv_hits = [e for e in inv_evals if e.get("invalidation_hit") is True]
    result["invalidation_n"] = len(inv_evals)
    result["invalidation_hits"] = len(inv_hits)
    result["invalidation_hit_quote"] = (
        len(inv_hits) / len(inv_evals) if inv_evals else None
    )
    result["invalidation_details"] = [
        {
            "ticker": e.get("ticker", ""),
            "timestamp": e.get("timestamp", ""),
            "invalidiert": e.get("invalidation_hit"),
            "begruendung": e.get("invalidation_reason", ""),
            "invalidation": e.get("invalidation", ""),
        }
        for e in inv_evals
    ]

    # --- Statistische Signifikanz (DSR/MBL, Bailey & López de Prado 2014) ---
    # STANDALONE, strikt additiv, deterministisch und NETZFREI (reine
    # math-Formeln, kein LLM-Call, kein Netz). Bewusst VOR dem LLM-Summary-
    # Block (in evaluate_journal) berechnet — die Metriken hängen an den
    # Trade-Renditen, nicht am LLM. Basis: Renditen der echten Trades
    # (KAUFEN/VERKAUFEN) — konsistent mit hit_rate_gesamt/Brier-Filterung.
    # Bei zu wenigen Punkten → None + deutscher Hinweis (nie crashen).
    result["siginifikanz_hinweis"] = _compute_signifikanz(trades, result)

    return result


def _compute_signifikanz(
    trades: list[dict[str, Any]],
    result: dict[str, Any],
) -> str:
    """Berechnet DSR/MBL aus den Trade-Renditen und schreibt sie ins result.

    Hilfsfunktion für _aggregate (verändert NUR die neuen Keys):
      * result["dsr"]      — Deflated Sharpe Ratio (Bailey & LdP 2014), (0,1)|None
      * result["dsr_ps"]   — unaufgeblasener PSR (P[SR > 0]) zur Referenz, (0,1)|None
      * result["mbl"]      — Minimale Backtest-Länge (Anzahl Trades), float|None
      * result["signifikanz_details"] — n_trials, annualisierter Trade-Sharpe, n

    Konservativer n_trials-Default (dokumentiert!): Anzahl der bewerteten
    Entscheidungen (anzahl_entscheidungen) — jede Journal-Zeile ist ein
    "Versuch" des Systems; mindestens 2 (Φ⁻¹(1−1/N) braucht N ≥ 2).

    Bei < 10 Trade-Renditen: dsr/dsr_ps/mbl = None und deutscher Hinweis
    ("⚠️ zu wenige Trades für Signifikanz") als Rückgabewert.
    """
    try:
        renditen = [
            float(e["rendite_pct"])
            for e in trades
            if e.get("rendite_pct") is not None
            and math.isfinite(e["rendite_pct"])
        ]
        n_renditen = len(renditen)
        details = result.setdefault(
            "signifikanz_details",
            {"n_trials": 0, "sharpe_trades_annualisiert": None, "n_trade_renditen": 0},
        )
        details["n_trade_renditen"] = n_renditen
        details["n_trials"] = max(int(result.get("anzahl_entscheidungen") or 0), 2)

        if n_renditen < _MIN_TRADE_RETURNS_FUER_SIGNIFIKANZ:
            return (
                "⚠️ zu wenige Trades für Signifikanz: DSR/MBL brauchen "
                f"mindestens {_MIN_TRADE_RETURNS_FUER_SIGNIFIKANZ} "
                f"Trade-Renditen (KAUFEN/VERKAUFEN); vorhanden: {n_renditen}."
            )

        mittel = sum(renditen) / n_renditen
        varianz = sum((r - mittel) ** 2 for r in renditen) / (n_renditen - 1)
        if varianz <= 0.0 or not math.isfinite(varianz):
            return (
                "⚠️ Trade-Renditen ohne Streuung — Sharpe/DSR/MBL nicht "
                "berechenbar."
            )
        std = math.sqrt(varianz)
        # Konvention wie backtest.py: Ø/std · sqrt(252) (annualisiert).
        sharpe_ann = mittel / std * math.sqrt(_JAHRESFAKTOR_SIGNIFIKANZ)
        # Per-Trade-Einheiten für die PSR/DSR-Formeln (n = Trade-Anzahl).
        sharpe_per_trade = sharpe_ann / math.sqrt(_JAHRESFAKTOR_SIGNIFIKANZ)

        details["sharpe_trades_annualisiert"] = sharpe_ann
        skew, kurt = _moments_skew_kurt(renditen)

        # DSR (deflationiert) + unaufgeblasener PSR zur Referenz.
        result["dsr"] = compute_dsr(
            sharpe=sharpe_per_trade,
            n_obs=n_renditen,
            skew=skew if skew is not None else 0.0,
            kurt=kurt if kurt is not None else 3.0,
            n_trials=details["n_trials"],
        )
        result["dsr_ps"] = _probst(
            sharpe=sharpe_per_trade,
            n=n_renditen,
            skew=skew if skew is not None else 0.0,
            kurt=kurt if kurt is not None else 3.0,
            sharpe_benchmark=0.0,
        )

        # MBL: Mindestanzahl Trades für p < 0.05 (annualisierter Sharpe).
        result["mbl"] = compute_mbl(
            sharpe=sharpe_ann,
            annualization=_JAHRESFAKTOR_SIGNIFIKANZ,
            pval=0.05,
            skew=skew if skew is not None else 0.0,
            kurt=kurt if kurt is not None else 3.0,
        )
        return ""
    except Exception as exc:  # noqa: BLE001 — Signifikanz darf nie crashen
        logger.warning("Signifikanz-Berechnung (DSR/MBL) fehlgeschlagen: %s", exc)
        return "⚠️ Signifikanz-Metriken (DSR/MBL) konnten nicht berechnet werden."


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

        halten_n = result.get("halten_n", 0)
        halten_q = result.get("halten_quote")
        halten_str = (
            f"{halten_q * 100:.1f}%"
            if halten_q is not None and not (isinstance(halten_q, float) and math.isnan(halten_q))
            else "N/A"
        )

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
            f"- Hit-Rate (nur Trades KAUFEN/VERKAUFEN; HALTEN nicht enthalten): {hr_str}\n"
            f"- HALTEN: {halten_n} Entscheidungen, davon {halten_str} stabil (±2%)\n"
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
    uebersprungen = 0
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
                # Retry: Cache löschen und EINMAL erneut versuchen,
                # damit ein evtl. korrupter/leerer Cache-Eintrag nicht blockiert.
                _delete_price_cache(ticker)
                retry_prices = _load_price_history(
                    ticker, lookback_days=lookback_days
                )
                if retry_prices:
                    price_cache[ticker] = retry_prices
                    prices = retry_prices
                else:
                    price_cache[ticker] = None
                    uebersprungen += 1
                    fehler.append(
                        f"{row.get('timestamp', '?')} {ticker}: "
                        f"Keine Kursdaten verfügbar (auch nach Retry)."
                    )
                    continue

            eval_result = _evaluate_single(row, prices, lookback_days)
            # Roh-Zeile mitführen (interner Key, Underscore-Präfix analog
            # _data_text): die Invalidierungs-Prüfung liest daraus die
            # invalidation-Spalte. Wird von Report/Aggregation nicht gerendert.
            eval_result["_row"] = row
            evaluations.append(eval_result)
        except Exception as exc:  # noqa: BLE001 — jede Zeile einzeln
            uebersprungen += 1
            fehler.append(f"{row.get('timestamp', '?')} {ticker}: {exc}")

    # --- Invalidierungs-Prüfung (Stufe 1, STANDALONE Transparenz-Metrik) -------
    # Nur wenn ein LLM gegeben ist: Pro bewertbarer Zeile mit nicht-leerer
    # invalidation-Spalte EIN LLM-Call ("Wurde eine Bedingung verletzt?").
    # Ohne LLM (llm=None) wird der Check komplett übersprungen — alle
    # Kennzahlen bleiben None/0 (deterministischer Fallback, kein Crash).
    # Der Check verändert NICHT hit/rendite der Zeile (separate Felder).
    invalidation_nicht_bewertbar = 0
    if llm is not None:
        for eval_result in evaluations:
            try:
                invalidation_text = str(
                    (eval_result.get("_row") or {}).get("invalidation") or ""
                ).strip()
                if not invalidation_text:
                    continue  # Zeile ohne These → nicht bewertbar, zählt nicht
                check = check_invalidation_hit(
                    invalidation_text, eval_result, llm
                )
                if check is None:
                    invalidation_nicht_bewertbar += 1
                    continue  # Fallback: unlesbare Antwort/Fehler → überspringen
                verdict, begruendung = check
                eval_result["invalidation_hit"] = verdict
                eval_result["invalidation_reason"] = begruendung
                eval_result["invalidation"] = invalidation_text
            except Exception as exc:  # noqa: BLE001 — nie crashen
                invalidation_nicht_bewertbar += 1
                logger.warning(
                    "Invalidierungs-Check für Zeile übersprungen: %s", exc
                )

    # Aggregieren
    result = _aggregate(evaluations)
    result["fehler"] = fehler
    result["uebersprungen"] = uebersprungen
    result["invalidation_nicht_bewertbar"] = invalidation_nicht_bewertbar

    # LLM-Zusammenfassung (falls llm gegeben)
    if llm is not None and result["anzahl_entscheidungen"] > 0:
        result["zusammenfassung"] = _build_llm_summary(result, llm)

    return result


# --------------------------------------------------------------------------- #
# Realisierter Return für eine einzelne Journal-Zeile (für Reflexion)
# --------------------------------------------------------------------------- #


def realised_return_for_row(row: dict[str, Any], lookback_days: int = 30) -> dict[str, Any] | None:
    """Berechnet den realisierten Return für eine einzelne Journal-Zeile.

    Nutzt die vorhandenen Helper _load_price_history / _find_price_on_or_before /
    _find_price_on_or_after. Invertiert die Rendite für VERKAUFEN/STARK VERKAUFEN
    (analog _evaluate_single). Berechnet zusätzlich den Return des regionalen
    Benchmarks (via benchmark_for_ticker, z. B. ^GDAXI für *.DE) über das
    gleiche Zeitfenster und den Alpha (raw - benchmark).

    Args:
        row: Journal-Zeile (dict mit mindestens 'ticker', 'timestamp', 'action').
        lookback_days: Zeitfenster in Tagen (Default 30).

    Returns:
        dict mit ticker, entry_price, exit_price, raw_return_pct,
        benchmark_return_pct, alpha_pct, benchmark, timestamp, action —
        oder None bei irgendeinem Fehler (never raises).
    """
    try:
        ticker = (row.get("ticker") or "").strip()
        if not ticker:
            return None

        timestamp = row.get("timestamp", "")
        decision_date = _parse_timestamp(timestamp)
        if decision_date is None:
            return None

        action = (row.get("action") or "").strip().upper()

        # Preisgeschichte laden
        prices = _load_price_history(ticker, lookback_days=lookback_days)
        if not prices:
            return None

        # Entry: Kurs am oder vor Entscheidungsdatum
        entry_row = _find_price_on_or_before(prices, decision_date)
        if entry_row is None:
            entry_row = prices[0]
        entry_price = _safe_float(entry_row.get("close"))
        if entry_price is None or not math.isfinite(entry_price) or entry_price <= 0:
            return None

        # Exit: Kurs am oder nach decision_date + lookback_days, clamped to today
        today = datetime.now()
        end_date = decision_date + timedelta(days=lookback_days)
        eval_end = min(end_date, today)

        exit_row = _find_price_on_or_before(prices, eval_end)
        if exit_row is None:
            exit_row = prices[-1]
        exit_price = _safe_float(exit_row.get("close"))
        if exit_price is None or not math.isfinite(exit_price):
            return None

        # Rendite berechnen
        price_change_pct = (exit_price - entry_price) / entry_price * 100.0
        if not math.isfinite(price_change_pct):
            return None
        if action in ("VERKAUFEN", "STARK VERKAUFEN"):
            raw_return_pct = -price_change_pct
        else:
            raw_return_pct = price_change_pct

        # Benchmark-Return über das gleiche Fenster (regionaler Benchmark
        # via benchmark_for_ticker — z. B. ^GDAXI für *.DE, SPY für US)
        benchmark = benchmark_for_ticker(ticker)
        benchmark_return_pct: float | None = None
        alpha_pct: float | None = None
        try:
            benchmark_prices = _load_price_history(benchmark, lookback_days=lookback_days)
            if benchmark_prices:
                benchmark_entry = _find_price_on_or_before(benchmark_prices, decision_date)
                if benchmark_entry is None:
                    benchmark_entry = benchmark_prices[0]
                benchmark_exit = _find_price_on_or_before(benchmark_prices, eval_end)
                if benchmark_exit is None:
                    benchmark_exit = benchmark_prices[-1]
                benchmark_entry_price = _safe_float(benchmark_entry.get("close"))
                benchmark_exit_price = _safe_float(benchmark_exit.get("close"))
                if (benchmark_entry_price is not None and benchmark_exit_price is not None
                        and math.isfinite(benchmark_entry_price) and math.isfinite(benchmark_exit_price)
                        and benchmark_entry_price > 0):
                    benchmark_return_pct = (
                        (benchmark_exit_price - benchmark_entry_price)
                        / benchmark_entry_price * 100.0
                    )
                    if math.isfinite(benchmark_return_pct):
                        alpha_pct = raw_return_pct - benchmark_return_pct
                    else:
                        benchmark_return_pct = None
        except Exception as bench_exc:  # noqa: BLE001 — best effort
            logger.debug("Benchmark-Return konnte nicht berechnet werden: %s", bench_exc)

        return {
            "ticker": ticker,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "raw_return_pct": raw_return_pct,
            "benchmark": benchmark,
            "benchmark_return_pct": benchmark_return_pct,
            "alpha_pct": alpha_pct,
            "timestamp": timestamp,
            "action": action,
        }
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("realised_return fehlgeschlagen für Zeile: %s", exc)
        return None
