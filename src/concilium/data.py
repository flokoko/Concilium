"""Datenmodul — Sammelt Fundamentals, technische Indikatoren, Historie und Sentiment-Heuristik via yfinance.

Fallback-News-Quelle: Google News RSS (kein API-Key nötig, nur requests + xml.etree).
"""

from __future__ import annotations

import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
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

# ---------------------------------------------------------------------------
# Tages-Cache für Marktdaten (yfinance-Ergebnisse)
# ---------------------------------------------------------------------------
# Vermeidet wiederholte yfinance-Aufrufe innerhalb desselben Kalendertags.
# Cache-Ort übersteuerbar via Env CONCILIUM_CACHE_DIR (Default: <repo>/cache/).
# Cache DEAKTIVIERBAR via CONCILIUM_CACHE_DIR="" (leerer String).
# Datei pro Ticker+Datum: market_{YYYY-MM-DD}_{ticker}.json.


def _get_cache_dir() -> str | None:
    """Bestimmt das Cache-Verzeichnis.

    Returns:
        Pfad zum Cache-Verzeichnis, oder None wenn Cache deaktiviert ist
        (CONCILIUM_CACHE_DIR="" → deaktiviert).

    Default: <repo>/cache/ (Repo-Root = Eltern von src/).
    """
    env = os.environ.get("CONCILIUM_CACHE_DIR")
    if env is not None:
        env = env.strip()
        if not env:
            return None  # leerer String → Cache deaktiviert
        return env
    # Default: <repo>/cache/ — Repo-Root ist Eltern von src/
    # __file__ = .../src/concilium/data.py → Repo-Root = .../
    repo_root = Path(__file__).resolve().parent.parent.parent
    return str(repo_root / "cache")


def _get_today_key() -> str:
    """Liefert das heutige UTC-Datum als YYYY-MM-DD String."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _cache_file_path(cache_dir: str, today_key: str, ticker: str) -> str:
    """Bestimmt den Dateipfad für einen Cache-Eintrag."""
    # Ticker kann / enthalten (z.B. nicht bereinigt) → sicher machen
    safe_ticker = re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
    return os.path.join(cache_dir, f"market_{today_key}_{safe_ticker}.json")


def _load_cache(ticker: str, today_key: str | None = None) -> dict[str, Any] | None:
    """Lädt gecachte Marktdaten für einen Ticker, wenn der Cache heute ist.

    Args:
        ticker: Aufgelöster Yahoo-Ticker (z.B. AAPL, RWE.DE).
        today_key: Optionales Datum YYYY-MM-DD (für Tests). Default: heute (UTC).

    Returns:
        Das data-dict (ohne isin/wkn) oder None bei Cache-Miss/Fehler/Deaktiviert.
    """
    cache_dir = _get_cache_dir()
    if cache_dir is None:
        return None  # Cache deaktiviert
    if today_key is None:
        today_key = _get_today_key()

    file_path = _cache_file_path(cache_dir, today_key, ticker)
    try:
        if not os.path.isfile(file_path):
            return None
        with open(file_path, encoding="utf-8") as fh:
            cached = json.load(fh, object_hook=_cache_json_object_hook)
        # Gültigkeit: cache_date muss mit today_key übereinstimmen
        if cached.get("cache_date") != today_key:
            return None
        # data-dict extrahieren
        data = cached.get("data")
        if not isinstance(data, dict):
            return None
        logger.info("Cache-Treffer für %s (%s)", ticker, today_key)
        return data
    except Exception as exc:  # noqa: BLE001 — Cache-Lesen crasht nie
        logger.debug("Cache-Lesen fehlgeschlagen für %s: %s", ticker, exc)
        return None


def _save_cache(ticker: str, data: dict[str, Any], today_key: str | None = None) -> None:
    """Speichert Marktdaten für einen Ticker im Tages-Cache.

    Crasht nie (best effort). Speichert nur die yfinance-abhängigen Daten,
    NICHT die Identifier-Metadaten (isin/wkn).

    Args:
        ticker: Aufgelöster Yahoo-Ticker.
        data: Das komplette data-dict (inkl. isin/wkn — diese werden beim
              Speichern entfernt).
        today_key: Optionales Datum YYYY-MM-DD (für Tests).
    """
    cache_dir = _get_cache_dir()
    if cache_dir is None:
        return  # Cache deaktiviert
    if today_key is None:
        today_key = _get_today_key()

    # Kopie ohne Identifier-Metadaten (isin/wkn gehören nicht in den Cache)
    cache_data = {k: v for k, v in data.items() if k not in ("isin", "wkn")}
    cache_entry = {
        "cache_date": today_key,
        "ticker": ticker,
        "data": cache_data,
    }
    file_path = _cache_file_path(cache_dir, today_key, ticker)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(cache_entry, fh, ensure_ascii=False, default=str)
        logger.info("Cache gespeichert für %s (%s)", ticker, today_key)
    except Exception as exc:  # noqa: BLE001 — Cache-Schreiben crasht nie
        logger.debug("Cache-Schreiben fehlgeschlagen für %s: %s", ticker, exc)


def _cache_json_object_hook(obj: dict[str, Any]) -> dict[str, Any]:
    """Reviver für json.load: konvertiert ISO-8601 datetime-Strings zurück.

    Wird in _load_cache verwendet, um serialisierte datetime-Objekte in
    news_with_dates wiederherzustellen.
    """
    for key, val in obj.items():
        if isinstance(val, str) and len(val) >= 10:
            # ISO-8601 datetime erkennen (z.B. "2026-08-20T08:30:00+00:00")
            if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", val):
                try:
                    dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    obj[key] = dt
                except (ValueError, TypeError):
                    pass
    return obj

# ---------------------------------------------------------------------------
# Identifier-Auflösung (ISIN / WKN → Yahoo-Ticker)
# ---------------------------------------------------------------------------

# ISIN: 2 Buchstaben + 9 alphanumerische Zeichen + 1 Prüfziffer
_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

# WKN: genau 6 alphanumerische Zeichen
_WKN_RE = re.compile(r"^[A-Z0-9]{6}$")

# ISIN-Extraktion aus wallstreet-online HTML
_ISIN_EXTRACT_RE = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")

# Yahoo Search API (kein API-Key nötig)
# WICHTIG: Yahoo blockiert den vollen Chrome-User-Agent mit HTTP 429 (Too Many
# Requests), obwohl der Endpoint eigentlich ohne Key funktioniert. Ein minimaler
# UA ("Mozilla/5.0") umgeht das Rate-Limit zuverlässig (verifiziert).
_YAHOO_SEARCH_URL = (
    "https://query1.finance.yahoo.com/v1/finance/search"
    "?q={query}&lang=de-DE&region=DE&quotesCount=5&newsCount=0"
)
_YAHOO_SEARCH_UA = "Mozilla/5.0"

# wallstreet-online Suche (für WKN → ISIN, Primärquelle)
_WSO_SEARCH_URL = "https://www.wallstreet-online.de/suche?q={query}"

# onvista Suche — Fallback-Quelle für WKN → ISIN, wenn wallstreet-online versagt
_ONVISTA_SEARCH_URL = (
    "https://www.onvista.de/aktien/suche.html"
    "?SEARCH_VALUE={query}&SEARCH_TYPE=0"
)


def _detect_identifier_type(identifier: str) -> str:
    """Klassifiziert einen Bezeichner als 'ISIN', 'WKN' oder 'TICKER'.

    Args:
        identifier: Eingabe (bereits uppercase, stripped).

    Returns:
        'ISIN', 'WKN' oder 'TICKER'.
    """
    if _ISIN_RE.match(identifier):
        return "ISIN"
    if _WKN_RE.match(identifier):
        return "WKN"
    return "TICKER"


def _isin_to_ticker(isin: str) -> str:
    """Löst eine ISIN über die Yahoo Search API zum Yahoo-Ticker auf.

    GET https://query1.finance.yahoo.com/v1/finance/search?q={ISIN}
    Parse quotes[0].symbol.

    Raises:
        ValueError: bei Netzwerk-/Parse-Fehler oder leerem Ergebnis.
    """
    url = _YAHOO_SEARCH_URL.format(query=isin)
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": _YAHOO_SEARCH_UA})
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 — best effort, nie crashen
        raise ValueError(
            f"Kann ISIN '{isin}' nicht zu einem Ticker auflösen "
            f"(Yahoo Search API nicht erreichbar). Bitte prüfen. "
            f"Fehler: {exc}"
        ) from exc

    quotes = data.get("quotes", [])
    if not quotes:
        raise ValueError(
            f"Kann ISIN '{isin}' nicht zu einem Ticker auflösen "
            "(keine Treffer bei Yahoo). Bitte prüfen."
        )

    symbol = quotes[0].get("symbol")
    if not symbol:
        raise ValueError(
            f"Kann ISIN '{isin}' nicht zu einem Ticker auflösen "
            "(Yahoo lieferte kein Symbol). Bitte prüfen."
        )

    logger.info("ISIN '%s' → Yahoo-Ticker '%s'", isin, symbol)
    return symbol


def _wkn_to_isin_wallstreet(wkn: str) -> str:
    """Löst eine WKN über wallstreet-online zur ISIN auf (Primärquelle).

    GET https://www.wallstreet-online.de/suche?q={WKN}
    Extrahiert die erste ISIN via Regex aus dem HTML.

    Raises:
        ValueError: bei Netzwerk-/Parse-Fehler oder leerem Ergebnis.
    """
    url = _WSO_SEARCH_URL.format(query=wkn)
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:  # noqa: BLE001 — best effort, nie crashen
        raise ValueError(
            f"Kann WKN '{wkn}' nicht zu einer ISIN auflösen "
            f"(wallstreet-online nicht erreichbar). Bitte prüfen. "
            f"Fehler: {exc}"
        ) from exc

    match = _ISIN_EXTRACT_RE.search(html)
    if not match:
        raise ValueError(
            f"Kann WKN '{wkn}' nicht zu einer ISIN auflösen "
            "(keine ISIN auf wallstreet-online gefunden). Bitte prüfen."
        )

    isin = match.group(0)
    logger.info("WKN '%s' → ISIN '%s' (wallstreet-online)", wkn, isin)
    return isin


def _wkn_to_isin_onvista(wkn: str) -> str:
    """Löst eine WKN über onvista zur ISIN auf (Fallback-Quelle).

    GET https://www.onvista.de/aktien/suche.html?SEARCH_VALUE={WKN}&SEARCH_TYPE=0
    Extrahiert die erste ISIN via Regex aus dem HTML.
    Verwendet den vollen Chrome-User-Agent (onvista blockiert minimale UAs).

    Raises:
        ValueError: bei Netzwerk-/Parse-Fehler oder leerem Ergebnis.
    """
    url = _ONVISTA_SEARCH_URL.format(query=wkn)
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:  # noqa: BLE001 — best effort, nie crashen
        raise ValueError(
            f"Kann WKN '{wkn}' nicht zu einer ISIN auflösen "
            f"(onvista nicht erreichbar). Bitte prüfen. "
            f"Fehler: {exc}"
        ) from exc

    match = _ISIN_EXTRACT_RE.search(html)
    if not match:
        raise ValueError(
            f"Kann WKN '{wkn}' nicht zu einer ISIN auflösen "
            "(keine ISIN auf onvista gefunden). Bitte prüfen."
        )

    isin = match.group(0)
    logger.info("WKN '%s' → ISIN '%s' (onvista)", wkn, isin)
    return isin


def _wkn_to_isin(wkn: str) -> str:
    """Löst eine WKN zur ISIN auf — mit Fallback über zwei Quellen.

    Reihenfolge:
      1. wallstreet-online (Primärquelle)
      2. onvista (Fallback)

    Schlägt eine Quelle fehl, wird ein Warning geloggt und die nächste
    Quelle versucht. Erst wenn beide versagen, wird ein ValueError geworfen.

    Raises:
        ValueError: wenn beide Quellen keinen Erfolg liefern.
    """
    errors: list[str] = []

    # Primärquelle: wallstreet-online
    try:
        return _wkn_to_isin_wallstreet(wkn)
    except ValueError as exc:
        logger.warning("wallstreet-online WKN→ISIN fehlgeschlagen für '%s': %s", wkn, exc)
        errors.append(f"wallstreet-online: {exc}")

    # Fallback: onvista
    try:
        return _wkn_to_isin_onvista(wkn)
    except ValueError as exc:
        logger.warning("onvista WKN→ISIN fehlgeschlagen für '%s': %s", wkn, exc)
        errors.append(f"onvista: {exc}")

    # Beide Quellen gescheitert
    raise ValueError(
        f"Kann WKN '{wkn}' nicht zu einer ISIN auflösen "
        f"(weder wallstreet-online noch onvista lieferten ein Ergebnis). "
        f"Fehler: {'; '.join(errors)}"
    )


def resolve_identifier(identifier: str) -> tuple[str, dict[str, Any]]:
    """Löst einen Bezeichner (ISIN, WKN oder Ticker) zum Yahoo-Ticker auf.

    Erkennt automatisch den Typ:
      - ISIN (z. B. DE000BASF111) → Yahoo Search API → Ticker
      - WKN  (z. B. 716460, BASF11) → wallstreet-online → ISIN → Yahoo → Ticker
      - Ticker (z. B. AAPL, RWE.DE) → unverändert

    Args:
        identifier: Ticker, ISIN oder WKN.

    Returns:
        Tuple (resolved_ticker, meta) wobei meta ein dict ist mit:
          - input_type: 'ISIN' | 'WKN' | 'TICKER'
          - isin: str | None
          - wkn: str | None

    Raises:
        ValueError: bei Auflösungsfehler (Netzwerk/Parse) mit deutscher Meldung.
    """
    identifier = identifier.strip().upper()
    if not identifier:
        raise ValueError("Bezeichner darf nicht leer sein.")

    id_type = _detect_identifier_type(identifier)

    if id_type == "TICKER":
        return identifier, {"input_type": "TICKER", "isin": None, "wkn": None}

    if id_type == "ISIN":
        ticker = _isin_to_ticker(identifier)
        return ticker, {"input_type": "ISIN", "isin": identifier, "wkn": None}

    # WKN → ISIN → Ticker
    isin = _wkn_to_isin(identifier)
    ticker = _isin_to_ticker(isin)
    return ticker, {"input_type": "WKN", "isin": isin, "wkn": identifier}

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
    # --- Erweitert (deutsch/englisch) ---
    "gewinn", "wachstum", "anstieg", "steigt", "plus", "ertrag",
    "turnaround", "rebound", "erholung", "aufschwung", "rekord",
    "uebertrifft", "uebererfuellt", "starke", "erhoehung",
    # Umlaut-Versionen (für direkten Token-Match)
    "übertrifft", "übererfüllt", "erhöhung",
}

_NEGATIVE_WORDS = {
    "drop", "fall", "plunge", "loss", "miss", "cut", "downgrade", "bearish",
    "weak", "lawsuit", "fraud", "recall", "bankrupt", "crash", "decline",
    "fear", "risk", "warning", "halt", "suspend", "investigate", "sell-off",
    "slump", "dive", "tumble", "disappoint", "delay", "close", "fire",
    # --- Erweitert (deutsch/englisch) ---
    "verlust", "ruecklaeufig", "rueckgang", "sturz", "krise", "sorgen",
    "druck", "verfehlt", "gewinnwarnung", "schwach", "minus",
    "absturz", "belastung", "einbruch",
    # Umlaut-Versionen
    "rückläufig", "rückgang", "druck", "unterdruck",
    "unterdruck", "fall",
}

# Negationswörter (deutsch/englisch). Wenn eines dieser Wörter unmittelbar
# (0-2 Wörter Abstand) vor einem positiven/negativen Keyword steht, wird
# die Bewertung dieses Keywords invertiert.
_NEGATION_WORDS = {
    "nicht", "kein", "keine", "keinen", "ohne", "weniger", "trotz",
    "mangelnd", "fehlend", "no", "not", "never", "none", "weder",
}


def _token_matches_keyword(token: str, keywords: set[str]) -> bool:
    """Prüft, ob ein Token ein Keyword aus der Menge matcht.

    Match-Strategie: Zuerst exakter Token-Match, dann Suffix-basierte
    Flexions-Erkennung (z.B. "surges" → "surge" + "s").
    Vermeidet falsche Substring-Matches wie "rise" in "krise".
    """
    # 1. Exakter Match
    if token in keywords:
        return True
    # 2. Flexionsformen: Keyword + bis zu 3 Zeichen Suffix (z.B. "surge" + "s")
    for kw in keywords:
        if len(kw) < 4:
            continue
        if token.startswith(kw) and len(token) - len(kw) <= 3:
            return True
    return False


def _classify_headline(headline: str) -> str:
    """Klassifiziert eine einzelne Headline als 'positiv', 'negativ' oder 'neutral'.

    Wird von _count_sentiment und _count_sentiment_weighted gemeinsam genutzt.

    Negations-Handling: Wenn ein Negationswort 0-2 Wörter vor einem positiven
    oder negativen Keyword steht, wird die Bewertung dieses Keywords invertiert
    (positiv→negativ, negativ→positiv).
    """
    text = headline.lower()
    tokens = text.split()

    pos_score = 0
    neg_score = 0

    for i, token in enumerate(tokens):
        # Token bereinigen (Satzzeichen am Anfang/Ende entfernen)
        clean = token.strip(".,!?;:\"'()[]{}")

        # Prüfen, ob das Token ein Keyword matcht
        matched_positive = _token_matches_keyword(clean, _POSITIVE_WORDS)
        matched_negative = _token_matches_keyword(clean, _NEGATIVE_WORDS)

        if not matched_positive and not matched_negative:
            continue

        # Prüfen, ob ein Negationswort in den vorangehenden 0-2 Wörtern steht
        negated = False
        for j in range(max(0, i - 2), i):
            prev_token = tokens[j].strip(".,!?;:\"'()[]{}")
            if prev_token in _NEGATION_WORDS:
                negated = True
                break

        # Bewertung anwenden (mit Invertierung bei Negation)
        if matched_positive:
            if negated:
                neg_score += 1
            else:
                pos_score += 1
        if matched_negative:
            if negated:
                pos_score += 1
            else:
                neg_score += 1

    if pos_score > 0 and neg_score == 0:
        return "positiv"
    if neg_score > 0 and pos_score == 0:
        return "negativ"
    if pos_score > neg_score:
        return "positiv"
    if neg_score > pos_score:
        return "negativ"
    # Gleichstand oder keine Keywords → neutral
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
# Social-Sentiment-Quellen (Phase 3): StockTwits + Reddit
# ---------------------------------------------------------------------------

# StockTwits öffentlicher Stream-Endpoint (kein API-Key nötig)
_STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

# Reddit Suche-Endpoint (öffentlich, kein OAuth — aber eigener UA Pflicht)
_REDDIT_SEARCH_URL = (
    "https://www.reddit.com/search.json?q={ticker}&sort=top&t=week&limit={limit}"
)
_REDDIT_USER_AGENT = "Concilium/1.0 (python-requests, contact=flokoko@googlemail.com)"


def _fetch_stocktwits(ticker: str, limit: int = 10) -> list[dict[str, Any]]:
    """Holt Nachrichten von StockTwits für einen Ticker (öffentlich, kein Key).

    URL: https://api.stocktwits.com/api/2/streams/symbol/{TICKER}.json
    Liefert ``messages[]`` mit ``body``, ``created_at`` und ``entities``.

    Args:
        ticker: Ticker-Symbol (z. B. "NVDA"). Bei Tickers mit Punkt-Suffix
            (z. B. "RWE.DE") wird der rohe Ticker best-effort verwendet —
            StockTwits nutzt Cashtags wie ``$RWE``; ein Fehlschlag → ``[]``.
        limit: Maximale Anzahl Posts, die zurückgegeben werden.

    Returns:
        Liste von dicts ``{text, date, source}`` (gleiche Struktur wie
        ``news_with_dates``). Bei Fehler/leer → leere Liste. Crasht NIE.
    """
    # Ticker normalisieren: Punkt-Suffix für StockTwits oft problematisch →
    # Best-effort: verwende den Teil vor dem Punkt (z. B. "RWE.DE" → "RWE")
    st_ticker = ticker.split(".")[0] if "." in ticker else ticker
    url = _STOCKTWITS_URL.format(ticker=st_ticker)

    try:
        resp = requests.get(
            url, timeout=10, headers={"User-Agent": _USER_AGENT}
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 — niemals crashen
        logger.warning("StockTwits-Anfrage fehlgeschlagen für '%s': %s", ticker, exc)
        return []

    messages = data.get("messages", [])
    if not messages:
        logger.info("StockTwits lieferte keine Messages für '%s'.", ticker)
        return []

    items: list[dict[str, Any]] = []
    for msg in messages:
        body = msg.get("body", "")
        if not body or not body.strip():
            continue
        created_at = msg.get("created_at", "")
        # ISO-datetime parsen (StockTwits liefert ISO-8601 mit Z)
        try:
            date_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if date_dt.tzinfo is None:
                date_dt = date_dt.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            date_dt = datetime.now(timezone.utc)

        # Quelle auf StockTwits-Plattform setzen (z. B. "web", oder direkt "StockTwits")
        source = msg.get("source", {})
        source_id = source.get("id", "StockTwits") if isinstance(source, dict) else "StockTwits"
        items.append({
            "title": body.strip(),
            "published": date_dt,
            "source": source_id,
        })

        if len(items) >= limit:
            break

    logger.info("StockTwits: %d Messages für '%s' erhalten.", len(items), ticker)
    return items


def _fetch_reddit(ticker: str, limit: int = 5) -> list[dict[str, Any]]:
    """Holt Reddit-Posts für einen Ticker (öffentliche JSON-Suche, kein OAuth).

    URL: https://www.reddit.com/search.json?q={ticker}&sort=top&t=week&limit={limit}

    **Pflicht:** Eigener User-Agent (Reddit blockt Default-UAs mit 403).
    Rate-limit-respektvoll: genau EIN Call, Timeout, bei 403/429 → ``[]``.

    Args:
        ticker: Ticker-Symbol (z. B. "NVDA").
        limit: Maximale Anzahl Posts, die zurückgegeben werden.

    Returns:
        Liste von dicts ``{text, date, source}``. Bei Fehler/leer → leere Liste.
        Crasht NIE.
    """
    url = _REDDIT_SEARCH_URL.format(ticker=ticker, limit=limit)

    try:
        resp = requests.get(
            url, timeout=10, headers={"User-Agent": _REDDIT_USER_AGENT}
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 — niemals crashen
        logger.warning("Reddit-Anfrage fehlgeschlagen für '%s': %s", ticker, exc)
        return []

    children = data.get("data", {}).get("children", [])
    if not children:
        logger.info("Reddit lieferte keine Posts für '%s'.", ticker)
        return []

    items: list[dict[str, Any]] = []
    for child in children:
        post_data = child.get("data", {})
        title = post_data.get("title", "")
        selftext = post_data.get("selftext", "")
        # title + selftext kombinieren
        combined = f"{title} {selftext}".strip() if selftext else title.strip()
        if not combined:
            continue

        created_utc = post_data.get("created_utc")
        try:
            date_dt = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            date_dt = datetime.now(timezone.utc)

        items.append({
            "title": combined,
            "published": date_dt,
            "source": "reddit",
        })

        if len(items) >= limit:
            break

    logger.info("Reddit: %d Posts für '%s' erhalten.", len(items), ticker)
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


def _get_dividend_yield(info: dict[str, Any]) -> float | None:
    """Bestimmt die Dividendenrendite robust aus dem yfinance-info-Dict.

    yfinance liefert zwei Felder:
      - 'trailingAnnualDividendYield' (TTM): verlässlich, korrekte Werte.
      - 'dividendYield': bei mehreren Tickern fehlerhaft (z.B. NVDA 46%,
        RWE.DE 210%, TSM 105%). Bekannter yfinance-Datenfehler.

    Strategie:
      1. Primär 'trailingAnnualDividendYield' verwenden (TTM, verlässlich).
      2. Fallback auf 'dividendYield', nur wenn Trailing fehlt/None ist
         UND der Wert plausibel ist (0 < wert < 0.5).
      3. Falls das Ergebnis ≥ 0.5 ist (unsinnig hoch), als None behandeln.
      4. Der Wert bleibt ein Anteil (0.021 = 2.1%).

    Crasht nie; liefert None bei unplausiblen oder fehlenden Werten.
    """
    # 1. Primär: trailingAnnualDividendYield (TTM, verlässlich)
    trailing = _safe_float(info.get("trailingAnnualDividendYield"))
    if trailing is not None:
        # Plausibilitäts-Sicherung: ≥ 0.5 ist unsinnig hoch
        if trailing < 0.5:
            return trailing
        return None

    # 2. Fallback: dividendYield, nur wenn plausibel (0 < wert < 0.5)
    fallback = _safe_float(info.get("dividendYield"))
    if fallback is not None and 0.0 < fallback < 0.5:
        return fallback

    # 3. Unplausibel oder fehlt → None
    return None


def _validate_fundamentals(fundamentals: dict[str, Any]) -> list[str]:
    """Prüft Fundamentals auf unplausible Werte und gibt deutsche Warn-Strings zurück.

    Best-effort: Crasht nie, überspringt None-Werte stumm.
    Leere Liste = alles plausibel.

    Geprüfte Regeln:
      - dividend_yield > 0.5 (50% Rendite) → ADR-/Datenfehler vermutet
      - profit_margin > 1.0 (100%) → unplausible Gewinnmarge
      - market_cap < 1e8 (100 Mio) → sehr geringe Marktkapitalisierung
      - market_cap/revenue > 50 → ungewöhnlich hohe Bewertung (nur Hinweis)
    """
    warnings: list[str] = []

    # --- Dividendenrendite > 50% (ADR-Fehler wie TSM: 1.05 = 105%) ---
    dy = fundamentals.get("dividend_yield")
    if dy is not None:
        try:
            dy_val = float(dy)
            if dy_val > 0.5:
                warnings.append(
                    f"Unplausible Dividendenrendite von {dy_val * 100:.1f}% erkannt "
                    "— vermutlich ADR-/Datenfehler."
                )
        except (TypeError, ValueError):
            pass

    # --- Gewinnmarge > 100% ---
    pm = fundamentals.get("profit_margin")
    if pm is not None:
        try:
            pm_val = float(pm)
            if pm_val > 1.0:
                warnings.append(
                    f"Unplausible Gewinnmarge von {pm_val * 100:.1f}%."
                )
        except (TypeError, ValueError):
            pass

    # --- Sehr geringe Marktkapitalisierung (< 100 Mio USD) ---
    mc = fundamentals.get("market_cap")
    if mc is not None:
        try:
            mc_val = float(mc)
            if mc_val < 1e8:
                warnings.append(
                    f"Sehr geringe Marktkapitalisierung ({mc_val:.0f}) — Daten prüfen."
                )
        except (TypeError, ValueError):
            pass

    # --- Ungewöhnlich hohe Bewertung: market_cap/revenue > 50 ---
    rev = fundamentals.get("revenue")
    if mc is not None and rev is not None:
        try:
            mc_val = float(mc)
            rev_val = float(rev)
            if rev_val > 0:
                ratio = mc_val / rev_val
                if ratio > 50:
                    warnings.append(
                        f"Ungewöhnlich hohe Bewertung (Marktkap./Umsatz = {ratio:.1f})."
                    )
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    return warnings


def _fetch_macro_data() -> dict[str, Any]:
    """Holt Makro/Zins-Daten (10y US Treasury, S&P 500) — best effort, nie crashen.

    Returns:
        dict mit us_10y_yield, us_10y_yield_1m_ago, us_10y_trend,
        sp500_pe, sp500_market_cap (alle None bei Fehler).
    """
    result: dict[str, Any] = {
        "us_10y_yield": None,
        "us_10y_yield_1m_ago": None,
        "us_10y_trend": None,
        "sp500_pe": None,
        "sp500_market_cap": None,
    }

    # --- 10y US Treasury Yield (^TNX) ---
    try:
        tnx = yf.Ticker("^TNX")
        tnx_hist = tnx.history(period="1mo")
        if tnx_hist is not None and not tnx_hist.empty:
            close_col = tnx_hist["Close"]
            current_yield = _safe_float(close_col.iloc[-1])
            old_yield = _safe_float(close_col.iloc[0]) if len(close_col) >= 1 else None
            result["us_10y_yield"] = current_yield
            result["us_10y_yield_1m_ago"] = old_yield
            if current_yield is not None and old_yield is not None:
                diff = current_yield - old_yield
                if abs(diff) < 0.05:
                    result["us_10y_trend"] = "flach"
                elif diff > 0:
                    result["us_10y_trend"] = "steigend"
                else:
                    result["us_10y_trend"] = "fallend"
    except Exception as exc:  # noqa: BLE001 — best effort
        logger.warning("Makrodaten ^TNX konnten nicht abgerufen werden: %s", exc)

    # --- S&P 500 Benchmark (^GSPC mit SPY-Fallback) ---
    sp500 = _get_sp500_benchmark()
    result["sp500_pe"] = sp500["sp500_pe"]
    result["sp500_market_cap"] = sp500["sp500_market_cap"]
    result["sp500_source"] = sp500["sp500_source"]

    return result


def _get_sp500_benchmark() -> dict[str, Any]:
    """Holt S&P 500 Benchmark-KGV — zuerst ^GSPC, bei None Fallback auf SPY (ETF).

    Liefert ein dict mit:
      sp500_pe: float | None — das KGV (trailingPE)
      sp500_market_cap: float | None — Marktkapitalisierung
      sp500_source: str — "GSPC" oder "SPY" (je nach Quelle) oder "none"

    Robust: bei Fehler/kein Netz → Werte None, sp500_source="none". Nie crashen.
    """
    result: dict[str, Any] = {
        "sp500_pe": None,
        "sp500_market_cap": None,
        "sp500_source": "none",
    }

    # 1. Versuch: ^GSPC (S&P 500 Index)
    try:
        sp500 = yf.Ticker("^GSPC")
        sp500_info = sp500.info or {}
        pe = _safe_float(sp500_info.get("trailingPE"))
        if pe is not None:
            result["sp500_pe"] = pe
            result["sp500_market_cap"] = _safe_float(sp500_info.get("marketCap"))
            result["sp500_source"] = "GSPC"
            return result
        logger.info("^GSPC lieferte kein trailingPE — versuche SPY-Fallback …")
    except Exception as exc:  # noqa: BLE001 — best effort
        logger.warning("Makrodaten ^GSPC konnten nicht abgerufen werden: %s", exc)

    # 2. Fallback: SPY (S&P 500 ETF) — liefert zuverlässig trailingPE
    try:
        spy = yf.Ticker("SPY")
        spy_info = spy.info or {}
        pe = _safe_float(spy_info.get("trailingPE"))
        if pe is not None:
            result["sp500_pe"] = pe
            # SPY hat keine S&P 500 Marktkap, aber ggf. eigene marketCap
            result["sp500_market_cap"] = _safe_float(spy_info.get("marketCap"))
            result["sp500_source"] = "SPY"
            logger.info("SPY-Fallback erfolgreich: trailingPE=%s", pe)
        else:
            logger.warning("SPY lieferte ebenfalls kein trailingPE — S&P 500 KGV bleibt None.")
    except Exception as exc:  # noqa: BLE001 — best effort
        logger.warning("SPY-Fallback fehlgeschlagen: %s", exc)

    return result


def _fetch_peer_data(peers: list[str]) -> list[dict[str, Any]]:
    """Holt KGV/Marktkapitalisierung für Peer-Ticker — best effort, nie crashen.

    Args:
        peers: Liste von Ticker-Symbolen.

    Returns:
        Liste von dicts mit ticker, pe_ratio, market_cap, name.
        Bei Fehler/leer → leere Liste.
    """
    result: list[dict[str, Any]] = []
    for peer_ticker in peers:
        peer_ticker = peer_ticker.strip().upper()
        if not peer_ticker:
            continue
        try:
            pt = yf.Ticker(peer_ticker)
            pinfo = pt.info or {}
            result.append({
                "ticker": peer_ticker,
                "pe_ratio": _safe_float(pinfo.get("trailingPE")),
                "market_cap": _safe_float(pinfo.get("marketCap")),
                "name": pinfo.get("longName") or pinfo.get("shortName") or peer_ticker,
            })
        except Exception as exc:  # noqa: BLE001 — best effort
            logger.warning("Peer-Daten für '%s' konnten nicht abgerufen werden: %s", peer_ticker, exc)
            result.append({
                "ticker": peer_ticker,
                "pe_ratio": None,
                "market_cap": None,
                "name": peer_ticker,
            })
    return result


def collect_ticker_data(
    ticker: str,
    peers: list[str] | None = None,
) -> dict[str, Any]:
    """Sammelt alle Marktdaten für einen Ticker via yfinance.

    Args:
        ticker: Ticker-Symbol (z. B. AAPL, RWE.DE).
        peers: Optionale Liste von Peer-Ticker-Symbolen für den Vergleich.

    Returns:
        dict mit Schlüsseln: ticker, fundamentals, technicals, history, sentiment,
        news, macro, peers

    Raises:
        ValueError: bei ungültigem Ticker (keine Daten von yfinance).
    """
    ticker = ticker.strip()
    if not ticker:
        raise ValueError("Ticker darf nicht leer sein.")

    # Identifier auflösen (ISIN/WKN → Yahoo-Ticker); bei reinem Ticker
    # bleibt alles unverändert — kein Netzwerkaufruf nötig.
    # Diese Auflösung gehört NICHT in den Cache.
    resolved_ticker, id_meta = resolve_identifier(ticker)
    ticker = resolved_ticker

    # --- Tages-Cache prüfen (yfinance-abhängige Daten) ---
    cached = _load_cache(ticker)
    if cached is not None:
        # Identifier-Metadaten wieder hinzufügen (nicht im Cache gespeichert)
        cached["isin"] = id_meta.get("isin")
        cached["wkn"] = id_meta.get("wkn")
        return cached

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
    dividend_yield = _get_dividend_yield(info)
    beta = _safe_float(info.get("beta"))
    currency = info.get("currency", "USD")
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    long_name = info.get("longName") or info.get("shortName") or ticker

    # --- Feature 1: Analysten-Erwartungen ---
    analyst_target_mean = _safe_float(info.get("targetMeanPrice"))
    analyst_target_high = _safe_float(info.get("targetHighPrice"))
    analyst_target_low = _safe_float(info.get("targetLowPrice"))
    recommendation_key = info.get("recommendationKey")  # String, kein float
    analyst_count = _safe_float(info.get("numberOfAnalystOpinions"))
    recommendation_mean = _safe_float(info.get("recommendationMean"))
    current_price_info = _safe_float(info.get("currentPrice"))

    # Upside berechnen (currentPrice als Basis)
    analyst_upside_pct: float | None = None
    if analyst_target_mean is not None and current_price_info is not None and current_price_info > 0:
        analyst_upside_pct = (analyst_target_mean - current_price_info) / current_price_info * 100.0

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
        # Feature 1: Analysten-Erwartungen
        "analyst_target_mean": analyst_target_mean,
        "analyst_target_high": analyst_target_high,
        "analyst_target_low": analyst_target_low,
        "recommendation_key": recommendation_key,
        "analyst_count": int(analyst_count) if analyst_count is not None else None,
        "recommendation_mean": recommendation_mean,
        "analyst_upside_pct": analyst_upside_pct,
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

    # --- Feature 2: Makro/Zins-Daten ---
    macro = _fetch_macro_data()

    # --- Feature 3: Peer-Vergleich ---
    peers_data: list[dict[str, Any]] = []
    if peers:
        peers_data = _fetch_peer_data(peers)

    # --- Sentiment aus News (Phase 3: yfinance + Google + StockTwits + Reddit) ---
    news_list = None
    try:
        news_list = t.news
    except Exception as exc:  # noqa: BLE001
        logger.warning("Konnte .news nicht abrufen: %s", exc)

    headlines: list[str] = []
    news_with_dates: list[dict[str, Any]] = []
    active_sources: list[str] = []  # welche Quellen lieferten Daten

    # 1. yfinance (Primärquelle)
    yf_headlines = _extract_headlines(news_list)
    if yf_headlines:
        headlines.extend(yf_headlines)
        # yfinance-News als news_with_dates ohne verlässliche Zeitstempel
        for h in yf_headlines:
            news_with_dates.append({
                "title": h,
                "published": None,
                "source": "yfinance",
            })
        active_sources.append("yfinance")

    # 2. Google News RSS (Fallback, wenn yfinance leer — aber immer versuchen
    #    als Ergänzung, um die Dichte zu erhöhen)
    if not yf_headlines:
        logger.info("yfinance lieferte keine News für %s — versuche Google News RSS …", ticker)
    company = info.get("longName") or info.get("shortName") or ""
    google_items = _fetch_google_news(ticker, company_name=company)
    if google_items:
        for item in google_items:
            entry = {"title": item["title"], "published": item["published"], "source": "google"}
            news_with_dates.append(entry)
            headlines.append(item["title"])
        if "google" not in active_sources:
            active_sources.append("google")

    # 3. StockTwits (ergänzend — nie ersetzend)
    try:
        stocktwits_items = _fetch_stocktwits(ticker)
    except Exception as exc:  # noqa: BLE001 — defensive, nie crashen
        logger.warning("StockTwits-Fetch fehlerhaft für '%s': %s", ticker, exc)
        stocktwits_items = []
    if stocktwits_items:
        for item in stocktwits_items:
            news_with_dates.append(item)
            headlines.append(item["title"])
        active_sources.append("stocktwits")

    # 4. Reddit (ergänzend — nie ersetzend)
    try:
        reddit_items = _fetch_reddit(ticker)
    except Exception as exc:  # noqa: BLE001 — defensive, nie crashen
        logger.warning("Reddit-Fetch fehlerhaft für '%s': %s", ticker, exc)
        reddit_items = []
    if reddit_items:
        for item in reddit_items:
            news_with_dates.append(item)
            headlines.append(item["title"])
        active_sources.append("reddit")

    # news_source: Fallback-Kaskade dokumentieren
    news_source = ", ".join(active_sources) if active_sources else "none"

    # Zeitgewichtete Sentiment-Zählung, wenn Zeitstempel verfügbar sind;
    # sonst ungewichtete Keyword-Zählung.
    # StockTwits/Reddit liefern immer Zeitstempel → gewichtete Zählung,
    # sobald mindestens ein Eintrag mit published != None existiert.
    has_dates = any(
        item.get("published") is not None
        for item in news_with_dates
        if isinstance(item, dict)
    )

    if has_dates:
        sentiment = _count_sentiment_weighted(news_with_dates)
        # sources im sentiment vermerken
        sentiment["sources"] = active_sources
    elif headlines:
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
            "sources": active_sources,
        }
    else:
        # Keine Headlines aus irgendeiner Quelle
        sentiment = {
            "positiv": 0,
            "negativ": 0,
            "neutral": 0,
            "dominant": "neutral",
            "sample_size": 0,
            "weighted": False,
            "sources": [],
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

    # --- Datenqualitäts-Validierung ---
    data_warnings = _validate_fundamentals(fundamentals)

    result = {
        "ticker": ticker,
        "fundamentals": fundamentals,
        "technicals": technicals,
        "history": history_records,
        "sentiment": sentiment,
        "news": headlines[:20],  # neueste 20 Headlines
        "news_with_dates": news_with_dates[:20] if news_with_dates else [],
        "news_source": news_source,  # "yfinance" | "google_news" | "none"
        # Feature 2: Makro/Zins-Daten
        "macro": macro,
        # Feature 3: Peer-Vergleich
        "peers": peers_data,
        # Datenqualitäts-Warnungen (immer eine Liste, auch leer)
        "data_warnings": data_warnings,
        # Identifier-Metadaten (ISIN/WKN falls als Eingabe verwendet)
        "isin": id_meta.get("isin"),
        "wkn": id_meta.get("wkn"),
    }

    # --- Tages-Cache speichern (yfinance-abhängige Daten) ---
    # Wichtig: Nur speichern, wenn KEIN Fehler auftrat (kein alter Cache-Stand
    # als Fallback bei yfinance-Fehlern). Da wir bis hier nur kommen, wenn
    # alle Daten erfolgreich geladen wurden, ist das hier sicher.
    _save_cache(ticker, result)

    return result
