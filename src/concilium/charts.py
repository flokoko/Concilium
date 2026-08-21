"""Chart-Modul — optional PNG-Chart-Generierung via matplotlib.

matplotlib ist eine OPTIONALE Dependency. Ist es nicht installiert,
geben alle Funktionen None zurück — der Report funktioniert ohne Charts weiter.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Optionaler matplotlib-Import — kein Crash bei fehlendem matplotlib
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive Backend, kein GUI nötig
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    plt = None  # type: ignore[assignment]


def is_chart_available() -> bool:
    """Gibt True zurück, wenn matplotlib importierbar ist."""
    return _MPL_AVAILABLE


def generate_chart(data: dict[str, Any], out_dir: str) -> str | None:
    """Erzeugt ein 2-Panel-PNG-Chart (Kurs+SMA+Bollinger / RSI) für einen Ticker.

    Panel 1 (oben): Kurs-Schlusskurse + SMA50 + SMA200 + Bollinger-Bänder.
    Panel 2 (unten): RSI(14) mit 30/70-Referenzlinien.

    Args:
        data: Das Daten-dict aus collect_ticker_data (enthält 'history',
              'technicals', 'ticker').
        out_dir: Verzeichnis, in das das Chart gespeichert wird (reports/).

    Returns:
        Relativer Bildpfad (relativ zu out_dir, z. B. 'charts/AAPL_20260821_1430.png'),
        oder None bei fehlendem matplotlib, fehlenden Daten oder Zeichenfehler.
    """
    if not _MPL_AVAILABLE:
        return None

    history = data.get("history")
    if not history or not isinstance(history, list) or len(history) < 2:
        return None

    try:
        dates = [r.get("date", "") for r in history]
        closes = [_safe_float(r.get("close")) for r in history]
        technicals = data.get("technicals", {})

        # SMA-Werte aus technicals (bereits von data.py berechnet)
        sma50_val = _safe_float(technicals.get("sma50"))
        sma200_val = _safe_float(technicals.get("sma200"))
        rsi_val = _safe_float(technicals.get("rsi14"))
        boll = technicals.get("bollinger", {})
        boll_upper = _safe_float(boll.get("upper"))
        boll_middle = _safe_float(boll.get("middle"))
        boll_lower = _safe_float(boll.get("lower"))

        # RSI-Serie aus der Historie nachberechnen, falls nicht vorhanden
        rsi_series = _compute_rsi_series(closes)

        ticker = data.get("ticker", "UNKNOWN")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Chart-Verzeichnis
        charts_dir = os.path.join(out_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)

        filename = f"{ticker}_{timestamp}.png"
        filepath = os.path.join(charts_dir, filename)
        rel_path = f"charts/{filename}"

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8), height_ratios=[3, 1], sharex=True
        )
        fig.suptitle(f"{ticker} — Kurs & RSI", fontsize=13)

        # --- Panel 1: Kurs + SMAs + Bollinger ---
        ax1.plot(dates, closes, label="Close", color="#1f77b4", linewidth=1.2)
        if sma50_val is not None:
            ax1.axhline(y=sma50_val, color="#ff7f0e", linestyle="--", linewidth=0.8, label="SMA50")
        if sma200_val is not None:
            ax1.axhline(y=sma200_val, color="#2ca02c", linestyle="--", linewidth=0.8, label="SMA200")
        if boll_upper is not None:
            ax1.axhline(y=boll_upper, color="#d62728", linestyle=":", linewidth=0.6, label="Boll upper")
        if boll_lower is not None:
            ax1.axhline(y=boll_lower, color="#d62728", linestyle=":", linewidth=0.6, label="Boll lower")
        if boll_middle is not None:
            ax1.axhline(y=boll_middle, color="#9467bd", linestyle=":", linewidth=0.6, label="Boll mid")
        ax1.set_ylabel("Kurs")
        ax1.legend(loc="best", fontsize=7)
        ax1.grid(True, alpha=0.3)

        # X-Ticks reduzieren (max ~10 Labels)
        n = len(dates)
        step = max(1, n // 10)
        ax1.set_xticks(range(0, n, step))
        ax1.set_xticklabels([dates[i] for i in range(0, n, step)], rotation=45, ha="right", fontsize=7)

        # --- Panel 2: RSI ---
        if rsi_series:
            ax2.plot(dates, rsi_series, color="#8c564b", linewidth=1.0, label="RSI(14)")
            ax2.axhline(y=70, color="red", linestyle="--", linewidth=0.6)
            ax2.axhline(y=30, color="green", linestyle="--", linewidth=0.6)
            ax2.fill_between(dates, 30, 70, alpha=0.05, color="gray")
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("RSI(14)")
            ax2.legend(loc="best", fontsize=7)
            ax2.grid(True, alpha=0.3)
        elif rsi_val is not None:
            # Nur Punkt, keine Serie
            ax2.axhline(y=rsi_val, color="#8c564b", linewidth=1.0, label=f"RSI(14)={rsi_val:.1f}")
            ax2.axhline(y=70, color="red", linestyle="--", linewidth=0.6)
            ax2.axhline(y=30, color="green", linestyle="--", linewidth=0.6)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("RSI(14)")
            ax2.legend(loc="best", fontsize=7)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.set_ylabel("RSI(14)")
            ax2.text(0.5, 0.5, "RSI nicht verfügbar", transform=ax2.transAxes,
                     ha="center", va="center", fontsize=10, color="gray")

        fig.tight_layout()
        fig.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close(fig)

        logger.info("Chart gespeichert: %s", filepath)
        return rel_path

    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Chart-Generierung fehlgeschlagen: %s", exc)
        return None


def _safe_float(val: Any) -> float | None:
    """Konvertiert einen Wert zu float oder None."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _compute_rsi_series(closes: list[float | None]) -> list[float | None]:
    """Berechnet eine RSI(14)-Serie aus einer Liste von Schlusskursen.

    Gibt eine Liste gleicher Länge zurück (mit None am Anfang bis genug
    Daten vorhanden sind). Verwendet die Wilder's-Smoothing-Methode.
    """
    # Nur gültige Werte extrahieren
    vals = [v for v in closes if v is not None]
    if len(vals) < 15:
        return []

    period = 14
    rsi_vals: list[float] = []

    # Preisänderungen
    deltas = [vals[i] - vals[i - 1] for i in range(1, len(vals))]

    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    # Erster Durchschnitt (SMA der ersten 'period' Werte)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder's Smoothing
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi_vals.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_vals.append(100.0 - (100.0 / (1.0 + rs)))

    # Auf gleiche Länge wie closes auffüllen (None am Anfang)
    padding = len(closes) - len(rsi_vals)
    return [None] * padding + rsi_vals
