"""Portfolio-Analysis-Modul — deterministische Portfolio-Aggregation.

Berechnet Korrelations-Matrix, Overlap-Erkennung und Konzentrationswarnungen
über mehrere analysierte Ticker hinweg. Kein LLM, rein deterministisch.
Robust gegen fehlende/leere Daten (nie crashen, math.isfinite-Guards).
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# Minimum Anzahl überlappender Datenpunkte für Korrelations-Berechnung
MIN_OVERLAP_SAMPLES = 30

# Schwellenwert für hohe Korrelation (|r| > 0.7 → wenig Diversifikation)
HIGH_CORRELATION_THRESHOLD = 0.7

# Schwellenwert für Einzelposition-Konzentration (Depot-%)
SINGLE_POSITION_THRESHOLD = 5.0

# Schwellenwert für Sektor-/Region-Konzentration (kumuliertes Depot-%)
SECTOR_CONCENTRATION_THRESHOLD = 30.0


# ---------------------------------------------------------------------------
# Pearson-Korrelation (deterministisch, keine numpy-Abhängigkeit)
# ---------------------------------------------------------------------------


def _extract_close_series(
    history: list[dict[str, Any]],
) -> dict[str, float]:
    """Extrahiert {date: close} aus history_records.

    history_records sind dicts mit 'date' (YYYY-MM-DD String) und 'close' (float).
    Fallback: wenn kein 'date' vorhanden, wird 'time' versucht.
    Wenn keines von beiden vorhanden ist, wird ein Positions-Index als Key genutzt.
    """
    result: dict[str, float] = {}
    if not history:
        return result
    for i, record in enumerate(history):
        if not isinstance(record, dict):
            continue
        # Datum extrahieren (verschiedene mögliche Feldnamen)
        date_key = record.get("date") or record.get("time") or str(i)
        date_str = str(date_key)
        close = record.get("close")
        if close is None:
            continue
        try:
            close_float = float(close)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(close_float):
            continue
        result[date_str] = close_float
    return result


def _daily_returns(prices: list[float]) -> list[float]:
    """Berechnet Tagesrenditen (pct_change) aus einer Liste von Close-Preisen."""
    if len(prices) < 2:
        return []
    returns: list[float] = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        curr = prices[i]
        if prev == 0 or not math.isfinite(prev) or not math.isfinite(curr):
            continue
        r = (curr - prev) / prev
        if math.isfinite(r):
            returns.append(r)
    return returns


def _pearson_r(x: list[float], y: list[float]) -> float | None:
    """Berechnet den Pearson-Korrelationskoeffizienten zweier Reihen.

    Beide Reihen müssen gleich lang sein. Gibt None zurück bei zu wenigen
    Datenpunkten oder Null-Varianz.
    """
    n = len(x)
    if n != len(y) or n < 2:
        return None

    # Mittelwerte
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    # Abweichungen
    dx = [xi - mean_x for xi in x]
    dy = [yi - mean_y for yi in y]

    # Zähler und Nenner
    numerator = sum(dx[i] * dy[i] for i in range(n))
    sum_sq_x = sum(d * d for d in dx)
    sum_sq_y = sum(d * d for d in dy)
    denominator = math.sqrt(sum_sq_x * sum_sq_y)

    if denominator == 0 or not math.isfinite(denominator):
        return None

    r = numerator / denominator
    if not math.isfinite(r):
        return None
    # Clamp auf [-1.0, 1.0] (Floating-Point kann leicht überschreiten)
    r = max(-1.0, min(1.0, r))
    return r


def _aligned_returns(
    series_a: dict[str, float],
    series_b: dict[str, float],
) -> tuple[list[float], list[float], int]:
    """Alignet zwei {date: close}-Reihen auf gemeinsame Daten (inner join).

    Gibt (returns_a, returns_b, num_common_dates) zurück.
    num_common_dates ist die Anzahl der überlappenden Datenpunkte VOR
    pct_change (also eine mehr als die returns-Listen).
    """
    common_dates = sorted(set(series_a.keys()) & set(series_b.keys()))
    if len(common_dates) < 2:
        return [], [], len(common_dates)

    prices_a = [series_a[d] for d in common_dates]
    prices_b = [series_b[d] for d in common_dates]

    returns_a = _daily_returns(prices_a)
    returns_b = _daily_returns(prices_b)

    return returns_a, returns_b, len(common_dates)


def compute_correlations(
    history_map: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float | None]]:
    """Berechnet die Pearson-Korrelationsmatrix über Tagesrenditen.

    Args:
        history_map: Mapping {ticker: history_records} wobei history_records
            eine Liste von dicts mit 'date'/'time' und 'close' ist.

    Returns:
        Matrix {tickerA: {tickerB: korrelations_koeffizient | None}}.
        Diagonale ist 1.0. None bei zu wenigen überlappenden Daten (<30).
    """
    tickers = list(history_map.keys())
    matrix: dict[str, dict[str, float | None]] = {}

    # Pre-compute close series for each ticker
    close_series: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        close_series[ticker] = _extract_close_series(history_map[ticker])

    for t_a in tickers:
        matrix[t_a] = {}
        for t_b in tickers:
            if t_a == t_b:
                matrix[t_a][t_b] = 1.0
                continue
            returns_a, returns_b, n_common = _aligned_returns(
                close_series[t_a], close_series[t_b]
            )
            # n_common ist die Anzahl der überlappenden Preise;
            # returns haben einen weniger. Prüfe gegen MIN_OVERLAP_SAMPLES.
            if n_common < MIN_OVERLAP_SAMPLES:
                matrix[t_a][t_b] = None
                continue
            r = _pearson_r(returns_a, returns_b)
            matrix[t_a][t_b] = r

    return matrix


def correlation_sample_sizes(
    history_map: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    """Berechnet die Anzahl überlappender Datenpunkte für jedes Ticker-Paar.

    Returns:
        Matrix {tickerA: {tickerB: n_common_dates}}.
    """
    tickers = list(history_map.keys())
    sizes: dict[str, dict[str, int]] = {}

    close_series: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        close_series[ticker] = _extract_close_series(history_map[ticker])

    for t_a in tickers:
        sizes[t_a] = {}
        for t_b in tickers:
            if t_a == t_b:
                sizes[t_a][t_b] = len(close_series[t_a])
                continue
            _, _, n_common = _aligned_returns(close_series[t_a], close_series[t_b])
            sizes[t_a][t_b] = n_common

    return sizes


# ---------------------------------------------------------------------------
# Overlap-Erkennung (gegen bestehenden Bestand)
# ---------------------------------------------------------------------------


def _normalize_ticker(ticker: str) -> str:
    """Normalisiert einen Ticker für Vergleich (uppercase, stripped)."""
    return ticker.strip().upper() if ticker else ""


def portfolio_overlap(
    analysed_tickers: list[str],
    positions: list[dict[str, Any]],
    analysed_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Erkennt Overlap zwischen analysierten Tickern und bestehendem Bestand.

    Prüft:
    - Direkter Ticker-Match (Ticker oder Sheet-Symbol stimmt überein)
    - Name-Match (analysierter Name kommt im Depot-Namen vor oder umgekehrt)
    - Region-Overlap (analysierte Ticker in derselben Region wie Depot-Positionen)

    Args:
        analysed_tickers: Liste der analysierten Ticker-Symbole.
        positions: Depot-Positionen aus fetch_portfolio_positions().
        analysed_names: Optional {ticker: company_name} Mapping für Name-Match.

    Returns:
        dict mit:
          - direct_overlaps: Liste von {ticker, position, depot_pct}
          - region_overlaps: Liste von {ticker, region, depot_pct}
          - total_overlap_pct: kumulierter Depot-% der überlappten Positionen
          - warnings: Liste von Warnungs-Strings (deutsch)
    """
    if analysed_names is None:
        analysed_names = {}

    result: dict[str, Any] = {
        "direct_overlaps": [],
        "region_overlaps": [],
        "total_overlap_pct": 0.0,
        "warnings": [],
    }

    if not positions or not analysed_tickers:
        return result

    analysed_set = {_normalize_ticker(t) for t in analysed_tickers}
    analysed_name_map = {
        _normalize_ticker(t): (analysed_names.get(t, "") or "").lower()
        for t in analysed_tickers
    }

    overlapped_pcts: set[int] = set()  # Indices der überlappten Positionen

    for pos in positions:
        pos_ticker = _normalize_ticker(pos.get("ticker", ""))
        pos_sheet = _normalize_ticker(pos.get("sheet_symbol", ""))
        pos_name = (pos.get("name", "") or "").lower()
        depot_pct = pos.get("depot_pct", 0.0)
        idx = pos.get("_idx", 0)

        # Direkter Ticker-Match
        if pos_ticker in analysed_set or pos_sheet in analysed_set:
            result["direct_overlaps"].append({
                "ticker": pos.get("ticker", ""),
                "position_name": pos.get("name", ""),
                "depot_pct": depot_pct,
            })
            overlapped_pcts.add(idx)
            continue

        # Name-Match
        for t_norm, a_name in analysed_name_map.items():
            if a_name and pos_name and (a_name in pos_name or pos_name in a_name):
                result["direct_overlaps"].append({
                    "ticker": t_norm,
                    "position_name": pos.get("name", ""),
                    "depot_pct": depot_pct,
                })
                overlapped_pcts.add(idx)
                break

    # Total overlap percentage
    total_overlap = 0.0
    for pos in positions:
        idx = pos.get("_idx", 0)
        if idx in overlapped_pcts:
            total_overlap += pos.get("depot_pct", 0.0)
    result["total_overlap_pct"] = round(total_overlap, 2)

    # Warnings
    if result["direct_overlaps"]:
        for ov in result["direct_overlaps"]:
            result["warnings"].append(
                f"Overlap: {ov['ticker']} ({ov['position_name']}) "
                f"ist bereits mit {ov['depot_pct']:.1f}% im Depot."
            )

    if result["total_overlap_pct"] > 20.0:
        result["warnings"].append(
            f"Hoher Gesamt-Overlap: {result['total_overlap_pct']:.1f}% "
            f"des Depots überlappen mit analysierten Titeln."
        )

    return result


# ---------------------------------------------------------------------------
# Konzentrations-Warnungen
# ---------------------------------------------------------------------------


def portfolio_concentration(
    positions: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> list[str]:
    """Berechnet Konzentrationswarnungen über analysierte Titel + Bestand.

    Prüft:
    - Einzelposition > 5% (Bestand oder geplante Ziel-Gewichtung)
    - Kumulierte Ziel-Gewichtung der analysierten Titel
    - Sektor-/Region-Konzentration (>30% in einer Gruppe)

    Args:
        positions: Depot-Positionen aus fetch_portfolio_positions().
        weights: Optional {ticker: ziel_gewichtung_pct} der analysierten Titel.

    Returns:
        Liste von deutschen Warnungs-Strings.
    """
    warnings: list[str] = []

    if weights is None:
        weights = {}

    # --- Einzelposition > 5% im Bestand ---
    for pos in positions:
        pct = pos.get("depot_pct", 0.0)
        name = pos.get("name", "?")
        try:
            pct_float = float(pct)
        except (TypeError, ValueError):
            continue
        if pct_float > SINGLE_POSITION_THRESHOLD:
            warnings.append(
                f"Konzentration: '{name}' macht {pct_float:.1f}% des Depots aus "
                f"(>{SINGLE_POSITION_THRESHOLD:.0f}%-Regel)."
            )

    # --- Ziel-Gewichtungen der analysierten Titel ---
    total_target = 0.0
    for ticker, weight in weights.items():
        try:
            w = float(weight)
            if math.isfinite(w):
                total_target += w
                if w > SINGLE_POSITION_THRESHOLD:
                    warnings.append(
                        f"Ziel-Gewichtung: {ticker} bei {w:.1f}% "
                        f"(>{SINGLE_POSITION_THRESHOLD:.0f}%-Regel)."
                    )
        except (TypeError, ValueError):
            continue

    if total_target > 0:
        warnings.append(
            f"Kumulierte Ziel-Gewichtung der analysierten Titel: {total_target:.1f}%."
        )
        if total_target > 20.0:
            warnings.append(
                f"⚠️ Hohe kumulierte Ziel-Gewichtung ({total_target:.1f}%) "
                f"— Diversifikation prüfen."
            )

    # --- Sektor-/Region-Konzentration ---
    region_sums: dict[str, float] = {}
    for pos in positions:
        region = (pos.get("region", "") or "Unbekannt").strip()
        pct = pos.get("depot_pct", 0.0)
        try:
            pct_float = float(pct)
        except (TypeError, ValueError):
            pct_float = 0.0
        region_sums[region] = region_sums.get(region, 0.0) + pct_float

    for region, pct in region_sums.items():
        if pct > SECTOR_CONCENTRATION_THRESHOLD:
            warnings.append(
                f"Region-Konzentration: {region} bei {pct:.1f}% "
                f"(>{SECTOR_CONCENTRATION_THRESHOLD:.0f}%)."
            )

    return warnings


# ---------------------------------------------------------------------------
# Top-Level: run_portfolio_analysis
# ---------------------------------------------------------------------------


def run_portfolio_analysis(
    results: dict[str, dict[str, Any]],
    positions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Führt die vollständige Portfolio-Analyse über mehrere Ticker aus.

    Args:
        results: Mapping {ticker: pipeline_result} wobei jedes result ein
            "data"-dict mit "history" enthält, plus optional "portfolio_fit".
        positions: Depot-Positionen aus fetch_portfolio_positions().
            Wenn None, wird keine Overlap-Analyse durchgeführt.

    Returns:
        dict mit:
          - correlations: Korrelationsmatrix {tickerA: {tickerB: r|None}}
          - sample_sizes: {tickerA: {tickerB: n}}
          - overlap: Overlap-Ergebnis (oder None)
          - concentration_warnings: Liste von Warnungs-Strings
          - target_weights: {ticker: ziel_gewichtung_pct}
          - analysed_tickers: Liste der analysierten Ticker
    """
    analysed_tickers = list(results.keys())

    # History-Map für Korrelations-Berechnung extrahieren
    history_map: dict[str, list[dict[str, Any]]] = {}
    target_weights: dict[str, float] = {}
    analysed_names: dict[str, str] = {}

    for ticker, result in results.items():
        data = result.get("data", {})
        history = data.get("history", [])
        if isinstance(history, list):
            history_map[ticker] = history

        # Target weight aus portfolio_fit extrahieren
        pf = result.get("portfolio_fit")
        if isinstance(pf, dict):
            gw = pf.get("ziel_gewichtung_pct")
            if gw is not None:
                try:
                    target_weights[ticker] = float(gw)
                except (TypeError, ValueError):
                    pass

        # Company name für Name-Matching
        fundamentals = data.get("fundamentals", {})
        name = fundamentals.get("name", "")
        if name:
            analysed_names[ticker] = name

    # Korrelationsmatrix
    correlations = compute_correlations(history_map)
    sample_sizes = correlation_sample_sizes(history_map)

    # Overlap
    overlap = None
    if positions is not None:
        overlap = portfolio_overlap(analysed_tickers, positions, analysed_names)

    # Konzentrationswarnungen
    concentration_warnings = portfolio_concentration(
        positions or [], target_weights
    )

    return {
        "correlations": correlations,
        "sample_sizes": sample_sizes,
        "overlap": overlap,
        "concentration_warnings": concentration_warnings,
        "target_weights": target_weights,
        "analysed_tickers": analysed_tickers,
    }


def portfolio_context_to_text(context: dict[str, Any]) -> str:
    """Formatiert den Portfolio-Kontext als JSON-Text für den PM-Prompt.

    Args:
        context: Das Ergebnis aus run_portfolio_analysis().

    Returns:
        Kompakter JSON-String für den LLM-User-Prompt.
    """
    # Reduziere auf die wesentlichen Felder für den PM
    summary = {
        "analysed_tickers": context.get("analysed_tickers", []),
        "correlations": context.get("correlations", {}),
        "target_weights": context.get("target_weights", {}),
        "concentration_warnings": context.get("concentration_warnings", []),
        "overlap": context.get("overlap"),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2, default=str)
