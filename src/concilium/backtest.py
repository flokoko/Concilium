"""Backtest-Modul — einfacher Signalproxy (SMA50 vs SMA200 Trend + RSI).

Berechnet kumulierte Rendite der Signal-Strategie vs Buy&Hold über die Historie.

Look-ahead-Konvention (Phase 5): Signale werden aus dem Close des Signal-Tags
berechnet und sind daher erst NACH Handelsschluss bekannt. Der realisierbare
Einstieg ist folgerichtig der Close des NÄCHSTEN Handelstags — sowohl in der
Win-Rate-Berechnung als auch im ``entry_close``-Feld der Signal-Übergänge.
Die Strategie-Rendite nutzt bereits ``signal.shift(1)`` und war damit korrekt.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _check_lookahead_bias(
    df: pd.DataFrame,
    signal_positions: list[int],
    entry_positions: list[int],
) -> bool:
    """Prüft deterministisch, dass kein Trade den Signal-Tag als Entry nutzt.

    Look-ahead-Bias-Check: Das Signal wird aus dem Close des Signal-Tags
    berechnet und ist erst nach Handelsschluss bekannt — der Einstiegskurs
    muss daher von einem Tag NACH dem Signal-Tag stammen.

    Args:
        df: Kurs-DataFrame (nur zur Längen-Validierung genutzt).
        signal_positions: Integer-Positionen der Signal-Tags (Long-Einstiege).
        entry_positions: Integer-Positionen der zugehörigen Entry-Tags.

    Returns:
        True, wenn für jeden Trade gilt: Entry-Position strikt nach Signal-
        Position (und beide im gültigen Bereich). False bei Verstoß,
        Längen-Mismatch oder ungültigen Positionen. Leere Trade-Listen
        gelten als bestanden (nichts zu prüfen).
    """
    try:
        if len(signal_positions) != len(entry_positions):
            return False
        n_rows = len(df)
        for sig_pos, entry_pos in zip(signal_positions, entry_positions, strict=True):
            if sig_pos is None or entry_pos is None:
                return False
            sig_i, entry_i = int(sig_pos), int(entry_pos)
            if not (0 <= sig_i < n_rows) or not (0 <= entry_i < n_rows):
                return False
            if entry_i <= sig_i:
                return False
        return True
    except (TypeError, ValueError) as exc:
        logger.warning("Look-ahead-Bias-Check konnte nicht ausgeführt werden: %s", exc)
        return False


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
            "lookahead_bias_geprueft": False,
            "lookahead_bias_hinweis": (
                f"Zu wenig Historie ({len(history)} Tage, min. 200 nötig) — "
                "Look-ahead-Check nicht durchgeführt."
            ),
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
            "lookahead_bias_geprueft": False,
            "lookahead_bias_hinweis": (
                "SMA200 konnte nicht berechnet werden — Look-ahead-Check nicht durchgeführt."
            ),
            "hinweis": "SMA200 konnte nicht berechnet werden.",
        }

    strat_final = df["cum_strategy"].iloc[-1] / df["cum_strategy"].iloc[valid_start] - 1
    bh_final = df["cum_buyhold"].iloc[-1] / df["cum_buyhold"].iloc[valid_start] - 1

    # Signal-Übergänge
    # Das Signal-Datum bleibt der Signal-Tag; ``close`` zeigt weiterhin den
    # Close des Signal-Tags (rückwärtskompatibel). Neu ist ``entry_close``:
    # der realisierbare Einstiegskurs — der Close des NÄCHSTEN Handelstags,
    # da das Signal (aus dem Close des Signal-Tags berechnet) erst nach
    # Handelsschluss bekannt ist. Bei FLAT-Übergängen ist entry_close None.
    df["signal_change"] = df["signal"].diff()
    signal_dates = []
    for idx, row in df.iterrows():
        if pd.notna(row.get("signal_change")) and row["signal_change"] != 0:
            entry_close_val: float | None = None
            if row["signal"] == 1 and idx + 1 < len(df):
                next_close = df["close"].iloc[idx + 1]
                if pd.notna(next_close):
                    entry_close_val = float(next_close)
            signal_dates.append({
                "date": row["date"],
                "aktion": "LONG" if row["signal"] == 1 else "FLAT",
                "close": round(float(row["close"]), 2) if pd.notna(row["close"]) else None,
                "entry_close": round(entry_close_val, 2) if entry_close_val is not None else None,
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
    # Look-ahead-frei: Das Signal beruht auf dem Close des Signal-Tags und ist
    # erst NACH Handelsschluss bekannt. Der Einstiegskurs ist daher der Close
    # des NÄCHSTEN Handelstags (nicht der unrealisierbare Close des Signal-Tags).
    # Der Ausstieg bleibt der Close des Signalwechsel-Tags bzw. der letzte Close.
    signal_series = df["signal"].iloc[valid_start:]
    trades: list[bool] = []  # True = positiver Trade
    signal_positions: list[int] = []  # Integer-Position der Signal-Tags
    entry_positions: list[int] = []  # Integer-Position der Entry-Tags (Folgetag)
    in_long = False
    entry_pos: int | None = None

    for rel_pos in range(len(signal_series)):
        sig = signal_series.iloc[rel_pos]
        # WICHTIG: signal_series ist ein Slice ab valid_start — die Loop-
        # Position ist relativ, die absolute df-Position kommt aus dem
        # (Range-)Index des Slices (so auch im Original via .index[idx]).
        pos = int(signal_series.index[rel_pos])
        if sig == 1 and not in_long:
            in_long = True
            signal_positions.append(pos)
            if pos + 1 < len(df):
                entry_pos = pos + 1
                entry_positions.append(entry_pos)
            else:
                # Signal am letzten Tag: kein realisierbarer Folgetags-Close,
                # die Position ist noch nicht eingestiegen → kein Trade.
                entry_pos = None
        elif sig == 0 and in_long:
            # Long-Sektion endet — prüfe ob netto positiv (Entry = Folgetags-Close)
            if entry_pos is not None:
                entry_close = df["close"].iloc[entry_pos]
                exit_close = df["close"].iloc[pos]
                if pd.notna(entry_close) and pd.notna(exit_close) and entry_close > 0:
                    trades.append(float(exit_close) > float(entry_close))
                anzahl_trades += 1
            in_long = False
            entry_pos = None

    # Offene Position am Ende abschließen (falls noch in Long und eingestiegen)
    if in_long and entry_pos is not None:
        entry_close = df["close"].iloc[entry_pos]
        exit_close = df["close"].iloc[-1]
        if pd.notna(entry_close) and pd.notna(exit_close) and entry_close > 0:
            trades.append(float(exit_close) > float(entry_close))
        anzahl_trades += 1

    if len(trades) >= 1:
        win_rate_pct = round(sum(trades) / len(trades) * 100, 2)

    # --- Deterministischer Look-ahead-Bias-Check (Phase 5) ---
    # Verifiziert, dass jeder Einstiegskurs von einem Tag NACH dem Signal-Tag
    # stammt (niemals der Close des Signal-Tags selbst).
    lookahead_bias_geprueft = _check_lookahead_bias(df, signal_positions, entry_positions)
    if not lookahead_bias_geprueft:
        logger.warning(
            "Look-ahead-Bias-Check fehlgeschlagen: Einstiegskurs stammt nicht "
            "von einem Tag nach dem Signal-Tag."
        )

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
        # --- Look-ahead-Bias-Check (Phase 5) ---
        "lookahead_bias_geprueft": lookahead_bias_geprueft,
        "lookahead_bias_hinweis": (
            "Look-ahead-Check bestanden: Einstiegskurse stammen vom Handelstag "
            "NACH dem Signal-Tag (das Signal ist erst nach Handelsschluss bekannt)."
            if lookahead_bias_geprueft
            else "Look-ahead-Check fehlgeschlagen: Einstiegskurs stammt nicht "
            "vom Folgetag des Signal-Tags."
        ),
    }
