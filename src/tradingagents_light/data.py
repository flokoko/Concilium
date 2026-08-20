"""Datenmodul — Sammelt Fundamentals, technische Indikatoren, Historie und Sentiment-Heuristik via yfinance."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentiment-Heuristik — einfache keyword-basierte Zählung
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "surge", "soar", "rally", "gain", "profit", "beat", "record", "high",
    "growth", "upgrade", "bullish", "strong", "breakthrough", "win", "deal",
    "launch", "innovation", "expand", "acquire", "approve", "rise", "jump",
    "boom", "optimistic", "outperform", "raise", "boost",
}

_NEGATIVE_WORDS = {
    "drop", "fall", "plunge", "loss", "miss", "cut", "downgrade", "bearish",
    "weak", "lawsuit", "fraud", "recall", "bankrupt", "crash", "decline",
    "fear", "risk", "warning", "halt", "suspend", "investigate", "sell-off",
    "slump", "dive", "tumble", "disappoint", "delay", "close", "fire",
}


def _count_sentiment(headlines: list[str]) -> dict[str, int]:
    """Zählt positive/negative/neutral-Schlagzeilen anhand von Keywords."""
    pos = neg = neu = 0
    for headline in headlines:
        text = headline.lower()
        found_pos = any(w in text for w in _POSITIVE_WORDS)
        found_neg = any(w in text for w in _NEGATIVE_WORDS)
        if found_pos and not found_neg:
            pos += 1
        elif found_neg and not found_pos:
            neg += 1
        elif found_pos and found_neg:
            # Gemischte Headline → neutral
            neu += 1
        else:
            neu += 1
    return {"positiv": pos, "negativ": neg, "neutral": neu}


def _extract_headlines(news_list: Any) -> list[str]:
    """Extrahiert Headlines aus yfinance news-Struktur (variiert nach Version)."""
    headlines: list[str] = []
    if not news_list:
        return headlines
    for item in news_list:
        if isinstance(item, dict):
            title = item.get("title") or item.get("headline") or ""
            if title:
                headlines.append(title)
        elif isinstance(item, str):
            headlines.append(item)
    return headlines


# ---------------------------------------------------------------------------
# Technische Indikatoren
# ---------------------------------------------------------------------------


def _compute_rsi(series: pd.Series, period: int = 14) -> float | None:
    """Berechnet den RSI (Relative Strength Index)."""
    if len(series) < period + 1:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    if avg_loss.iloc[-1] == 0:
        return 100.0
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    if pd.isna(rs):
        return None
    return float(100.0 - (100.0 / (1.0 + rs)))


def _compute_macd(series: pd.Series) -> dict[str, float | None]:
    """Berechnet MACD (12, 26, 9)."""
    if len(series) < 35:
        return {"macd": None, "signal": None, "histogram": None}
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    return {
        "macd": float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
        "signal": float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None,
        "histogram": float(hist.iloc[-1]) if not pd.isna(hist.iloc[-1]) else None,
    }


def _compute_bollinger(series: pd.Series, period: int = 20) -> dict[str, float | None]:
    """Berechnet Bollinger-Bänder."""
    if len(series) < period:
        return {"upper": None, "middle": None, "lower": None, "position": None}
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    last_close = float(series.iloc[-1])
    upper_val = float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else None
    lower_val = float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else None
    middle_val = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None

    position = None
    if upper_val is not None and lower_val is not None and (upper_val - lower_val) != 0:
        position = (last_close - lower_val) / (upper_val - lower_val)

    return {"upper": upper_val, "middle": middle_val, "lower": lower_val, "position": position}


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> float | None:
    """Konvertiert einen Wert sicher zu float oder None."""
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def collect_ticker_data(ticker: str) -> dict[str, Any]:
    """Sammelt alle Marktdaten für einen Ticker via yfinance.

    Returns:
        dict mit Schlüsseln: ticker, fundamentals, technicals, history, sentiment, news

    Raises:
        ValueError: bei ungültigem Ticker (keine Daten von yfinance).
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker darf nicht leer sein.")

    logger.info("Sammle Daten für %s …", ticker)
    t = yf.Ticker(ticker)

    # --- Historie (~250 Tage OHLCV) ---
    hist = t.history(period="1y", auto_adjust=False)
    if hist is None or hist.empty:
        raise ValueError(
            f"Ungültiger Ticker '{ticker}': Keine Kursdaten von yfinance erhalten. "
            "Bitte Ticker-Symbol prüfen (z. B. AAPL, MSFT, NVDA)."
        )

    # --- Info / Fundamentals ---
    info: dict[str, Any] = {}
    try:
        info = t.info or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Konnte .info nicht abrufen: %s", exc)

    market_cap = _safe_float(info.get("marketCap"))
    pe_ratio = _safe_float(info.get("trailingPE"))
    eps = _safe_float(info.get("trailingEps"))
    revenue = _safe_float(info.get("totalRevenue"))
    revenue_growth = _safe_float(info.get("revenueGrowth"))
    profit_margin = _safe_float(info.get("profitMargins"))
    peg_ratio = _safe_float(info.get("pegRatio"))
    fifty_two_week_high = _safe_float(info.get("fiftyTwoWeekHigh"))
    fifty_two_week_low = _safe_float(info.get("fiftyTwoWeekLow"))
    dividend_yield = _safe_float(info.get("dividendYield"))
    beta = _safe_float(info.get("beta"))
    currency = info.get("currency", "USD")
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    long_name = info.get("longName") or info.get("shortName") or ticker

    fundamentals = {
        "name": long_name,
        "sector": sector,
        "industry": industry,
        "currency": currency,
        "market_cap": market_cap,
        "pe_ratio": pe_ratio,
        "eps": eps,
        "revenue": revenue,
        "revenue_growth": revenue_growth,
        "profit_margin": profit_margin,
        "peg_ratio": peg_ratio,
        "dividend_yield": dividend_yield,
        "beta": beta,
        "fifty_two_week_high": fifty_two_week_high,
        "fifty_two_week_low": fifty_two_week_low,
    }

    # --- Technische Indikatoren aus Historie ---
    close = hist["Close"]
    volume = hist["Volume"]
    current_price = _safe_float(close.iloc[-1])

    sma50 = float(close.rolling(window=50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = float(close.rolling(window=200).mean().iloc[-1]) if len(close) >= 200 else None

    rsi = _compute_rsi(close)
    macd = _compute_macd(close)
    bollinger = _compute_bollinger(close)
    current_volume = _safe_float(volume.iloc[-1])
    avg_volume_30d = float(volume.tail(30).mean()) if len(volume) >= 30 else None

    technicals = {
        "current_price": current_price,
        "sma50": sma50,
        "sma200": sma200,
        "rsi14": rsi,
        "macd": macd,
        "bollinger": bollinger,
        "current_volume": current_volume,
        "avg_volume_30d": avg_volume_30d,
    }

    # --- Sentiment aus News ---
    news_list = None
    try:
        news_list = t.news
    except Exception as exc:  # noqa: BLE001
        logger.warning("Konnte .news nicht abrufen: %s", exc)

    headlines = _extract_headlines(news_list)
    sentiment = _count_sentiment(headlines)

    # --- Historie als Records (für Backtest / Report) ---
    history_records = []
    for date, row in hist.iterrows():
        history_records.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": _safe_float(row.get("Open")),
            "high": _safe_float(row.get("High")),
            "low": _safe_float(row.get("Low")),
            "close": _safe_float(row.get("Close")),
            "volume": _safe_float(row.get("Volume")),
        })

    return {
        "ticker": ticker,
        "fundamentals": fundamentals,
        "technicals": technicals,
        "history": history_records,
        "sentiment": sentiment,
        "news": headlines[:20],  # neueste 20 Headlines
    }
