"""Datenmodul — Sammelt Fundamentals, technische Indikatoren, Historie und Sentiment-Heuristik via yfinance.

Fallback-News-Quelle: Google News RSS (kein API-Key nötig, nur requests + xml.etree).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# Google News RSS — Fallback-Quelle für Headlines
_GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=de&gl=DE&ceid=DE:de"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

# Halbwertszeit für zeitgewichtete Sentiment-Zählung (Tage)
HALF_LIFE_DAYS = 7.0

# Dublin Core Namespace für <dc:date> in RSS-Feeds
_DC_NAMESPACE = "{http://purl.org/dc/elements/1.1/}"

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


def _classify_headline(headline: str) -> str:
    """Klassifiziert eine einzelne Headline als 'positiv', 'negativ' oder 'neutral'.

    Wird von _count_sentiment und _count_sentiment_weighted gemeinsam genutzt.
    """
    text = headline.lower()
    found_pos = any(w in text for w in _POSITIVE_WORDS)
    found_neg = any(w in text for w in _NEGATIVE_WORDS)
    if found_pos and not found_neg:
        return "positiv"
    if found_neg and not found_pos:
        return "negativ"
    # Gemischte Headline (pos+neg) oder gar keine Keywords → neutral
    return "neutral"


def _count_sentiment(headlines: list[str]) -> dict[str, int]:
    """Zählt positive/negative/neutral-Schlagzeilen anhand von Keywords."""
    pos = neg = neu = 0
    for headline in headlines:
        label = _classify_headline(headline)
        if label == "positiv":
            pos += 1
        elif label == "negativ":
            neg += 1
        else:
            neu += 1
    return {"positiv": pos, "negativ": neg, "neutral": neu}


def _count_sentiment_weighted(
    headlines_with_dates: list[dict[str, Any]],
    *,
    half_life_days: float = HALF_LIFE_DAYS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Zeitgewichtete Sentiment-Zählung — neuere Headlines zählen mehr.

    Gewicht pro Headline: 2 ** (-age_days / half_life_days)
    (exponentieller Zerfall, Halbwertszeit = half_life_days).

    Args:
        headlines_with_dates: Liste von dicts mit "title" (str) und
            "published" (timezone-aware datetime).
        half_life_days: Halbwertszeit in Tagen (Default: 7 Tage).
        now: Referenzzeitpunkt (Default: datetime.now(timezone.utc)).

    Returns:
        dict mit "positiv", "negativ", "neutral" (float, gewichtete Summen),
        "dominant" (str: positiv/negativ/neutral), "sample_size" (int),
        "weighted" (bool, immer True).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Sicherstellen, dass 'now' timezone-aware ist
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    weighted = {"positiv": 0.0, "negativ": 0.0, "neutral": 0.0}
    count = 0

    for item in headlines_with_dates:
        title = item.get("title", "") if isinstance(item, dict) else ""
        if not title:
            continue
        published = item.get("published") if isinstance(item, dict) else None
        if published is None:
            # Kein Datum → neutral mit Gewicht 1.0 (entspricht "jetzt")
            weight = 1.0
        else:
            # Sicherstellen, dass published timezone-aware ist
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            age_delta = now - published
            age_days = age_delta.total_seconds() / 86400.0
            # Negatives Alter (Zukunft) auf 0 clampen
            age_days = max(age_days, 0.0)
            weight = 2.0 ** (-age_days / half_life_days)

        label = _classify_headline(title)
        weighted[label] += weight
        count += 1

    # Dominante Stimmung ermitteln (bei Gleichstand → neutral)
    max_val = max(weighted["positiv"], weighted["negativ"], weighted["neutral"])
    if weighted["positiv"] == max_val and weighted["positiv"] > weighted["negativ"]:
        dominant = "positiv"
    elif weighted["negativ"] == max_val and weighted["negativ"] > weighted["positiv"]:
        dominant = "negativ"
    else:
        dominant = "neutral"

    return {
        "positiv": round(weighted["positiv"], 4),
        "negativ": round(weighted["negativ"], 4),
        "neutral": round(weighted["neutral"], 4),
        "dominant": dominant,
        "sample_size": count,
        "weighted": True,
    }


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


def _parse_rss_date(raw: str | None) -> datetime:
    """Parst ein RSS-Datum (pubDate RFC-822 oder dc:date ISO-8601).

    Fallback bei Fehler oder fehlendem Datum: aktuelle Zeit (timezone-aware).
    Crasht NIE.
    """
    if not raw:
        return datetime.now(timezone.utc)
    raw = raw.strip()
    try:
        # pubDate — RFC-822 Format (z.B. "Wed, 20 Aug 2026 08:30:00 GMT")
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
    except (TypeError, ValueError):
        pass
    try:
        # dc:date — ISO-8601 Format (z.B. "2026-08-20T08:30:00Z")
        clean = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        pass
    # Nicht parsebar → aktuelle Zeit als Fallback
    return datetime.now(timezone.utc)


def _fetch_google_news(
    ticker: str, company_name: str = "", limit: int = 10
) -> list[dict[str, Any]]:
    """Holt News-Headlines von Google News RSS als Fallback, wenn yfinance leer ist.

    Sucht nach Ticker und optional zusätzlich nach Firmennamen.
    Gibt NIE eine Exception weiter — bei Fehler/leer → leere Liste.

    Args:
        ticker: Ticker-Symbol (z. B. "NVDA").
        company_name: Optionaler Firmenname für erweiterte Suche (z. B. "NVIDIA").
        limit: Maximale Anzahl Headlines.

    Returns:
        Liste von dicts mit {"title": str, "published": datetime}.
        Leere Liste bei Fehler oder keinem Ergebnis.
    """
    # Suchbegriffe zusammenstellen: Ticker + optional Firmenname
    query_parts = [ticker]
    if company_name:
        query_parts.append(company_name)
    query = " OR ".join(query_parts)

    url = _GOOGLE_NEWS_RSS_URL.format(query=query)

    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — niemals crashen
        logger.warning("Google News RSS Anfrage fehlgeschlagen für '%s': %s", ticker, exc)
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        logger.warning("Google News RSS XML konnte nicht geparst werden für '%s': %s", ticker, exc)
        return []

    items: list[dict[str, Any]] = []
    # RSS-Struktur: <rss><channel><item><title>…</title><pubDate>…</pubDate></item>…
    for element in root.iter("item"):
        title_el = element.find("title")
        if title_el is None or not title_el.text:
            continue
        title = title_el.text.strip()
        # "Top Stories" als generischen Aggregator-Titel überspringen
        if not title or title.lower() == "top stories":
            continue

        # Zeitstempel: pubDate (RFC-822), Fallback auf dc:date (ISO-8601)
        pub_date_el = element.find("pubDate")
        raw_date: str | None = None
        if pub_date_el is not None and pub_date_el.text:
            raw_date = pub_date_el.text.strip()
        if not raw_date:
            # Fallback: Dublin Core <dc:date>
            dc_date_el = element.find(f"{_DC_NAMESPACE}date")
            if dc_date_el is not None and dc_date_el.text:
                raw_date = dc_date_el.text.strip()

        published = _parse_rss_date(raw_date)
        items.append({"title": title, "published": published})

        if len(items) >= limit:
            break

    if not items:
        logger.info("Google News RSS lieferte keine Headlines für '%s'.", ticker)
    else:
        logger.info("Google News RSS: %d Headlines für '%s' erhalten.", len(items), ticker)

    return items


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
    news_with_dates: list[dict[str, Any]] = []

    if headlines:
        news_source = "yfinance"
        # yfinance-News haben (meist) keine verlässlichen Zeitstempel → ungewichtet
    else:
        # Fallback: Google News RSS, wenn yfinance keine Headlines liefert
        logger.info("yfinance lieferte keine News für %s — versuche Google News RSS …", ticker)
        company = info.get("longName") or info.get("shortName") or ""
        news_with_dates = _fetch_google_news(ticker, company_name=company)
        headlines = [item["title"] for item in news_with_dates]
        news_source = "google_news" if headlines else "none"

    # Zeitgewichtete Sentiment-Zählung, wenn Zeitstempel verfügbar sind;
    # sonst ungewichtete Keyword-Zählung.
    if news_with_dates:
        sentiment = _count_sentiment_weighted(news_with_dates)
        # sample_size und dominant sind bereits enthalten
    else:
        base = _count_sentiment(headlines)
        # Dominante Stimmung auch im ungewichteten Fall ermitteln
        max_val = max(base["positiv"], base["negativ"], base["neutral"])
        if base["positiv"] == max_val and base["positiv"] > base["negativ"]:
            dominant = "positiv"
        elif base["negativ"] == max_val and base["negativ"] > base["positiv"]:
            dominant = "negativ"
        else:
            dominant = "neutral"
        sentiment = {
            **base,
            "dominant": dominant,
            "sample_size": len(headlines),
            "weighted": False,
        }

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
        "news_with_dates": news_with_dates[:20] if news_with_dates else [],
        "news_source": news_source,  # "yfinance" | "google_news" | "none"
    }
