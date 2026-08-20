"""Backtest-Modul — einfacher Signalproxy (SMA50 vs SMA200 Trend + RSI).

Berechnet kumulierte Rendite der Signal-Strategie vs Buy&Hold über die Historie.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def run_backtest(data: dict[str, Any]) -> dict[str, Any]:
    """Führt einen einfachen Backtest mit SMA-Crossover + RSI-Filter aus.

    Strategie:
      - SMA50 > SMA200 → Long-Signal (Trend aufwärts)
      - RSI < 70 → nicht überkauft (Filter)
      - Sonst: flat (keine Position)

    Returns:
        dict mit Strategie-Rendite, Buy&Hold-Rendite, Outperformance, Signale.
    """
    history = data.get("history", [])
    if len(history) < 200:
        logger.warning("Zu wenig Historie für Backtest (braucht >=200 Tage, hat %d)", len(history))
        return {
            "strategie_rendite": None,
            "buy_hold_rendite": None,
            "outperformance": None,
            "signale": [],
            "hinweis": f"Zu wenig Historie ({len(history)} Tage, min. 200 nötig).",
        }

    df = pd.DataFrame(history)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    df["sma50"] = df["close"].rolling(window=50).mean()
    df["sma200"] = df["close"].rolling(window=200).mean()

    # RSI(14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    avg_loss_safe = avg_loss.replace(0, 1e-10)
    rs = avg_gain / avg_loss_safe
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # Signal: 1 (long) wenn SMA50 > SMA200 und RSI < 75, sonst 0
    df["signal"] = 0
    mask_long = (df["sma50"] > df["sma200"]) & (df["rsi"] < 75)
    df.loc[mask_long, "signal"] = 1

    # Tagesrenditen
    df["daily_return"] = df["close"].pct_change()
    df["strategy_return"] = df["signal"].shift(1) * df["daily_return"]

    # Kumulierte Rendite
    df["cum_strategy"] = (1 + df["strategy_return"].fillna(0)).cumprod()
    df["cum_buyhold"] = (1 + df["daily_return"].fillna(0)).cumprod()

    # Nur ab dem Punkt, wo SMAs gültig sind
    valid_start = df["sma200"].first_valid_index()
    if valid_start is None:
        return {
            "strategie_rendite": None,
            "buy_hold_rendite": None,
            "outperformance": None,
            "signale": [],
            "hinweis": "SMA200 konnte nicht berechnet werden.",
        }

    strat_final = df["cum_strategy"].iloc[-1] / df["cum_strategy"].iloc[valid_start] - 1
    bh_final = df["cum_buyhold"].iloc[-1] / df["cum_buyhold"].iloc[valid_start] - 1

    # Signal-Übergänge
    df["signal_change"] = df["signal"].diff()
    signal_dates = []
    for idx, row in df.iterrows():
        if pd.notna(row.get("signal_change")) and row["signal_change"] != 0:
            signal_dates.append({
                "date": row["date"],
                "aktion": "LONG" if row["signal"] == 1 else "FLAT",
                "close": round(float(row["close"]), 2) if pd.notna(row["close"]) else None,
            })

    return {
        "strategie_rendite": round(float(strat_final) * 100, 2) if pd.notna(strat_final) else None,
        "buy_hold_rendite": round(float(bh_final) * 100, 2) if pd.notna(bh_final) else None,
        "outperformance": round(float(strat_final - bh_final) * 100, 2) if pd.notna(strat_final) and pd.notna(bh_final) else None,
        "signale": signal_dates[-10:],  # letzte 10 Signal-Wechsel
        "anzahl_signale": len(signal_dates),
        "startdatum": df.loc[valid_start, "date"],
        "enddatum": df.iloc[-1]["date"],
    }
