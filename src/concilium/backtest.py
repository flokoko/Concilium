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
            "sharpe_ratio": None,
            "max_drawdown_pct": None,
            "win_rate_pct": None,
            "anzahl_trades": 0,
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
            "sharpe_ratio": None,
            "max_drawdown_pct": None,
            "win_rate_pct": None,
            "anzahl_trades": 0,
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

    # --- Neue Kennzahlen: Sharpe, Max-Drawdown, Win-Rate ---
    # Nur ab valid_start berechnen (wo SMAs gültig sind)
    strat_returns = df["strategy_return"].iloc[valid_start:].dropna()
    sharpe_ratio: float | None = None
    max_drawdown_pct: float | None = None
    win_rate_pct: float | None = None
    anzahl_trades: int = 0

    # Sharpe Ratio (annualisiert, rf=0)
    if len(strat_returns) >= 2 and strat_returns.std() > 0:
        sharpe_ratio = round(float(strat_returns.mean() / strat_returns.std() * (252**0.5)), 4)

    # Max Drawdown der Strategie-Kurve
    cum = df["cum_strategy"].iloc[valid_start:]
    if len(cum) >= 2:
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_dd = drawdown.min()
        if pd.notna(max_dd):
            max_drawdown_pct = round(float(max_dd) * 100, 2)

    # Win-Rate: Long-Signal-Abschnitte bis zum nächsten Signalwechsel
    signal_series = df["signal"].iloc[valid_start:]
    trades: list[bool] = []  # True = positiver Trade
    in_long = False
    entry_idx = None

    for idx in range(len(signal_series)):
        sig = signal_series.iloc[idx]
        if sig == 1 and not in_long:
            in_long = True
            entry_idx = signal_series.index[idx]
        elif sig == 0 and in_long:
            # Long-Sektion endet — prüfe ob netto positiv
            if entry_idx is not None:
                entry_close = df.loc[entry_idx, "close"]
                exit_close = df.loc[signal_series.index[idx], "close"]
                if pd.notna(entry_close) and pd.notna(exit_close) and entry_close > 0:
                    trades.append(exit_close > entry_close)
                anzahl_trades += 1
            in_long = False
            entry_idx = None

    # Offene Position am Ende abschließen (falls noch in Long)
    if in_long and entry_idx is not None:
        entry_close = df.loc[entry_idx, "close"]
        exit_close = df["close"].iloc[-1]
        if pd.notna(entry_close) and pd.notna(exit_close) and entry_close > 0:
            trades.append(exit_close > entry_close)
        anzahl_trades += 1

    if len(trades) >= 1:
        win_rate_pct = round(sum(trades) / len(trades) * 100, 2)

    return {
        "strategie_rendite": round(float(strat_final) * 100, 2) if pd.notna(strat_final) else None,
        "buy_hold_rendite": round(float(bh_final) * 100, 2) if pd.notna(bh_final) else None,
        "outperformance": round(float(strat_final - bh_final) * 100, 2) if pd.notna(strat_final) and pd.notna(bh_final) else None,
        "signale": signal_dates[-10:],
        "anzahl_signale": len(signal_dates),
        "startdatum": df.loc[valid_start, "date"],
        "enddatum": df.iloc[-1]["date"],
        # --- Neue Kennzahlen ---
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate_pct": win_rate_pct,
        "anzahl_trades": anzahl_trades,
    }
