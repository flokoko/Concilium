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
    if rendite_pct is None:
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
        portfolio_fit_score (float|None), ticker (str), timestamp (str).
    """
    action = (row.get("action") or "").strip().upper()
    timestamp = row.get("timestamp", "")
    decision_date = _parse_timestamp(timestamp)
    rating = (row.get("rating") or "").strip().upper()

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

    # Hit-Bestimmung — fachlich priorisierte Logik:
    #   1. Stop gerissen → Miss (Risikoregel verletzt, hat Vorrang vor allem)
    #   2. Ziel erreicht → Hit (Kursprognose erfüllt)
    #   3. Sonst → Rendite-basiert (bisherige Logik)
    # stop_gerissen / ziel_erreicht is None (nicht angegeben) → Bedingung überspringen.
    hit: bool | None = None
    if action in ("KAUFEN", "VERKAUFEN"):
        if stop_gerissen is True:
            hit = False
        elif ziel_erreicht is True:
            hit = True
        else:
            hit = rendite_pct > 0  # VERKAUFEN: bereits invertiert
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
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _empty_result() -> dict[str, Any]:
    """Leeres Ergebnis-dict (für fehlende/leere Journal-Datei)."""
    return {
        "anzahl_entscheidungen": 0,
        "nach_aktion": {
            "KAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None, "avg_confidence": None},
            "HALTEN": {"n": 0, "hit_rate": None, "avg_rendite": None, "avg_confidence": None},
            "VERKAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None, "avg_confidence": None},
        },
        "hit_rate_gesamt": None,
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
    """
    empty = {
        "brier_score": None,
        "n": 0,
        "durchschnittliche_konfidenz": None,
        "durchschnittliche_tatsaechliche_hit_rate": None,
        "kalibrierungs_gap": None,
        "tendenz": None,
    }

    # Nur Zeilen mit confidence und hit verwenden
    valid: list[dict[str, Any]] = []
    for e in evaluations:
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
        - nach_aktion: KAUFEN, HALTEN, VERKAUFEN
        - nach_rating: STARK KAUFEN, KAUFEN, HALTEN, VERKAUFEN, STARK VERKAUFEN

    Leere Segmente (n=0) werden weggelassen.

    Returns:
        dict: {"nach_aktion": {action: {brier_score, n, ...}}, "nach_rating": {...}}
    """
    _AKTIONEN = ("KAUFEN", "HALTEN", "VERKAUFEN")
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
    """
    valid: list[dict[str, Any]] = []
    for e in evaluations:
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

    # --- Durchschnittliche Rating-Distanz ---
    # Nur Zeilen mit gültigem rating_distance (int, nicht None)
    rating_distances = [
        e["rating_distance"]
        for e in evaluations
        if e.get("rating_distance") is not None
    ]
    result["durchschnitt_rating_distanz"] = (
        sum(rating_distances) / len(rating_distances) if rating_distances else None
    )

    # --- Konfidenz-Kalibrierung (Brier-Score, Gap, Tendenz) ---
    result["konfidenz_kalibrierung"] = _compute_konfidenz_kalibrierung(evaluations)

    # --- Segmentierte Konfidenz-Kalibrierung (pro Aktion, pro Rating) ---
    result["konfidenz_kalibrierung_segmentiert"] = (
        _compute_konfidenz_kalibrierung_segmentiert(evaluations)
    )

    # --- Reliability-Bänder (feinere Konfidenz-Intervalle) ---
    result["reliability_bins"] = _compute_reliability_bins(evaluations)

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
            evaluations.append(eval_result)
        except Exception as exc:  # noqa: BLE001 — jede Zeile einzeln
            uebersprungen += 1
            fehler.append(f"{row.get('timestamp', '?')} {ticker}: {exc}")

    # Aggregieren
    result = _aggregate(evaluations)
    result["fehler"] = fehler
    result["uebersprungen"] = uebersprungen

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
    (analog _evaluate_single). Berechnet zusätzlich den SPY-Return über das
    gleiche Zeitfenster und den Alpha (raw - spy).

    Args:
        row: Journal-Zeile (dict mit mindestens 'ticker', 'timestamp', 'action').
        lookback_days: Zeitfenster in Tagen (Default 30).

    Returns:
        dict mit ticker, entry_price, exit_price, raw_return_pct,
        spy_return_pct, alpha_pct, timestamp, action — oder None bei
        irgendeinem Fehler (never raises).
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

        # SPY-Return über das gleiche Fenster
        spy_return_pct: float | None = None
        alpha_pct: float | None = None
        try:
            spy_prices = _load_price_history("SPY", lookback_days=lookback_days)
            if spy_prices:
                spy_entry = _find_price_on_or_before(spy_prices, decision_date)
                if spy_entry is None:
                    spy_entry = spy_prices[0]
                spy_exit = _find_price_on_or_before(spy_prices, eval_end)
                if spy_exit is None:
                    spy_exit = spy_prices[-1]
                spy_entry_price = _safe_float(spy_entry.get("close"))
                spy_exit_price = _safe_float(spy_exit.get("close"))
                if (spy_entry_price is not None and spy_exit_price is not None
                        and math.isfinite(spy_entry_price) and math.isfinite(spy_exit_price)
                        and spy_entry_price > 0):
                    spy_return_pct = (spy_exit_price - spy_entry_price) / spy_entry_price * 100.0
                    if math.isfinite(spy_return_pct):
                        alpha_pct = raw_return_pct - spy_return_pct
                    else:
                        spy_return_pct = None
        except Exception as spy_exc:  # noqa: BLE001 — best effort
            logger.debug("SPY-Return konnte nicht berechnet werden: %s", spy_exc)

        return {
            "ticker": ticker,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "raw_return_pct": raw_return_pct,
            "spy_return_pct": spy_return_pct,
            "alpha_pct": alpha_pct,
            "timestamp": timestamp,
            "action": action,
        }
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("realised_return fehlgeschlagen für Zeile: %s", exc)
        return None
