"""Tests für Phase 3 — Sozial-Sentiment: StockTwits + Reddit (offline, gemockt).

Bereiche:
  1. _fetch_stocktwits mit gemocktem requests — Parsing + Fehlerfälle
  2. _fetch_reddit mit gemocktem requests — Parsing + Fehlerfälle
  3. Aggregation: collect_ticker_data zählt StockTwits/Reddit-Text mit
  4. Fallback: StockTwits/Reddit-Fehler → kein Crash, yfinance/Google bleibt
  5. Report: Quelle je Headline erscheint ([StockTwits]/[Reddit]/…)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# src zum Pfad hinzufügen
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.data import (  # noqa: E402
    _fetch_reddit,
    _fetch_stocktwits,
)
from concilium.report import _source_label, generate_report  # noqa: E402

# ---------------------------------------------------------------------------
# Mock-Daten
# ---------------------------------------------------------------------------

_MOCK_STOCKTWITS_JSON = {
    "messages": [
        {
            "body": "NVDA surges to record high on AI demand $NVDA",
            "created_at": "2026-08-20T08:30:00Z",
            "source": {"id": "web"},
        },
        {
            "body": "Bullish on NVDA earnings beat",
            "created_at": "2026-08-20T10:00:00Z",
            "source": {"id": "web"},
        },
        {
            "body": "",  # leerer Body → skip
            "created_at": "2026-08-20T11:00:00Z",
            "source": {"id": "web"},
        },
        {
            "body": "NVDA downgrade risk increasing",
            "created_at": "2026-08-20T12:00:00Z",
            "source": {},
        },
    ],
    "symbol": {"symbol": "NVDA"},
}

_MOCK_STOCKTWITS_EMPTY = {"messages": []}

_MOCK_REDDIT_JSON = {
    "data": {
        "children": [
            {
                "data": {
                    "title": "NVDA earnings beat expectations",
                    "selftext": "Strong growth in datacenter revenue",
                    "created_utc": 1724140800.0,  # 2024-08-20T12:00:00Z
                },
            },
            {
                "data": {
                    "title": "Is NVDA overvalued at these levels?",
                    "selftext": "",
                    "created_utc": 1724227200.0,  # 2024-08-21T12:00:00Z
                },
            },
            {
                "data": {
                    "title": "",  # leerer Titel, kein selftext → skip
                    "selftext": "",
                    "created_utc": 1724313600.0,
                },
            },
        ],
    },
}

_MOCK_REDDIT_EMPTY = {"data": {"children": []}}


def _mock_response(json_data):
    """Erstellt einen MagicMock-Response mit .json() und .raise_for_status()."""
    mock = MagicMock()
    mock.json.return_value = json_data
    mock.raise_for_status = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# 1. _fetch_stocktwits Tests
# ---------------------------------------------------------------------------


class TestFetchStocktwits:
    """Tests für _fetch_stocktwits — Parsing + Fehlerfälle."""

    def test_parses_messages(self):
        """Mock liefert 4 Messages (1 leer) → 3 dicts mit source=web."""
        with patch(
            "concilium.data.requests.get",
            return_value=_mock_response(_MOCK_STOCKTWITS_JSON),
        ):
            result = _fetch_stocktwits("NVDA", limit=10)

        assert len(result) == 3
        assert result[0]["title"] == "NVDA surges to record high on AI demand $NVDA"
        assert result[0]["source"] == "web"
        assert isinstance(result[0]["published"], datetime)
        assert result[0]["published"].tzinfo is not None

    def test_limit_respected(self):
        """limit=2 → nur 2 Items trotz 4 Messages."""
        with patch(
            "concilium.data.requests.get",
            return_value=_mock_response(_MOCK_STOCKTWITS_JSON),
        ):
            result = _fetch_stocktwits("NVDA", limit=2)

        assert len(result) == 2

    def test_empty_messages(self):
        """messages=[] → leere Liste."""
        with patch(
            "concilium.data.requests.get",
            return_value=_mock_response(_MOCK_STOCKTWITS_EMPTY),
        ):
            result = _fetch_stocktwits("NVDA")

        assert result == []

    def test_connection_error_returns_empty(self):
        """Bei requests.ConnectionError → leere Liste, kein Crash."""
        with patch("concilium.data.requests.get", side_effect=ConnectionError("net")):
            result = _fetch_stocktwits("NVDA")

        assert result == []

    def test_http_error_returns_empty(self):
        """Bei HTTPError (403/429) → leere Liste."""
        from requests.exceptions import HTTPError

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = HTTPError("403 Forbidden")

        with patch("concilium.data.requests.get", return_value=mock_resp):
            result = _fetch_stocktwits("NVDA")

        assert result == []

    def test_source_fallback_when_no_id(self):
        """Message mit source={} → source fällt auf 'StockTwits' zurück."""
        with patch(
            "concilium.data.requests.get",
            return_value=_mock_response(_MOCK_STOCKTWITS_JSON),
        ):
            result = _fetch_stocktwits("NVDA", limit=10)

        # Letzter Eintrag hat source={} → fallback "StockTwits"
        assert result[-1]["source"] == "StockTwits"

    def test_dot_suffix_ticker_handled(self):
        """RWE.DE → ruft StockTwits mit 'RWE' auf (Punkt-Suffix abgeschnitten)."""
        captured_url = []

        def fake_get(url, **kwargs):
            captured_url.append(url)
            return _mock_response(_MOCK_STOCKTWITS_EMPTY)

        with patch("concilium.data.requests.get", side_effect=fake_get):
            _fetch_stocktwits("RWE.DE")

        assert "RWE" in captured_url[0]
        assert "RWE.DE" not in captured_url[0]

    def test_invalid_json_returns_empty(self):
        """Bei invalid JSON → leere Liste."""
        mock_resp = MagicMock()
        mock_resp.json.side_effect = ValueError("not json")
        mock_resp.raise_for_status = MagicMock()

        with patch("concilium.data.requests.get", return_value=mock_resp):
            result = _fetch_stocktwits("NVDA")

        assert result == []


# ---------------------------------------------------------------------------
# 2. _fetch_reddit Tests
# ---------------------------------------------------------------------------


class TestFetchReddit:
    """Tests für _fetch_reddit — Parsing + Fehlerfälle."""

    def test_parses_posts(self):
        """Mock liefert 3 Posts (1 leer) → 2 dicts mit source=reddit."""
        with patch(
            "concilium.data.requests.get",
            return_value=_mock_response(_MOCK_REDDIT_JSON),
        ):
            result = _fetch_reddit("NVDA", limit=5)

        assert len(result) == 2
        assert result[0]["title"] == "NVDA earnings beat expectations Strong growth in datacenter revenue"
        assert result[0]["source"] == "reddit"
        assert isinstance(result[0]["published"], datetime)
        assert result[0]["published"].tzinfo is not None

    def test_limit_respected(self):
        """limit=1 → nur 1 Item."""
        with patch(
            "concilium.data.requests.get",
            return_value=_mock_response(_MOCK_REDDIT_JSON),
        ):
            result = _fetch_reddit("NVDA", limit=1)

        assert len(result) == 1

    def test_empty_children(self):
        """children=[] → leere Liste."""
        with patch(
            "concilium.data.requests.get",
            return_value=_mock_response(_MOCK_REDDIT_EMPTY),
        ):
            result = _fetch_reddit("NVDA")

        assert result == []

    def test_connection_error_returns_empty(self):
        """Bei requests.ConnectionError → leere Liste."""
        with patch("concilium.data.requests.get", side_effect=ConnectionError("net")):
            result = _fetch_reddit("NVDA")

        assert result == []

    def test_http_error_returns_empty(self):
        """Bei HTTPError (403) → leere Liste."""
        from requests.exceptions import HTTPError

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = HTTPError("403 Forbidden")

        with patch("concilium.data.requests.get", return_value=mock_resp):
            result = _fetch_reddit("NVDA")

        assert result == []

    def test_invalid_created_utc_falls_back_to_now(self):
        """Bei invalid created_utc → published wird auf aktuelle Zeit gesetzt."""
        reddit_data = {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "Test post",
                            "selftext": "",
                            "created_utc": "not-a-number",
                        },
                    },
                ],
            },
        }
        with patch(
            "concilium.data.requests.get",
            return_value=_mock_response(reddit_data),
        ):
            result = _fetch_reddit("NVDA")

        assert len(result) == 1
        pub = result[0]["published"]
        assert pub.tzinfo is not None

    def test_uses_custom_user_agent(self):
        """Reddit-Call verwendet eigenen User-Agent (nicht den generischen)."""
        captured_headers = []

        def fake_get(url, **kwargs):
            captured_headers.append(kwargs.get("headers", {}))
            return _mock_response(_MOCK_REDDIT_EMPTY)

        with patch("concilium.data.requests.get", side_effect=fake_get):
            _fetch_reddit("NVDA")

        assert "User-Agent" in captured_headers[0]
        ua = captured_headers[0]["User-Agent"]
        assert "Concilium" in ua


# ---------------------------------------------------------------------------
# 3. Aggregation: Sentiment zählt StockTwits/Reddit-Text mit
# ---------------------------------------------------------------------------


class TestSentimentAggregation:
    """Tests für die Aggregation über mehrere Quellen in collect_ticker_data."""

    def test_news_with_dates_have_source_field(self):
        """collect_ticker_data: news_with_dates-Einträge haben 'source'."""
        # Wir mocken yfinance + beide Social-Quellen, um offline zu testen
        from concilium.data import collect_ticker_data

        mock_yf_ticker = MagicMock()
        mock_hist = MagicMock()
        mock_hist.empty = False
        mock_hist.__iter__ = MagicMock(return_value=iter([]))
        mock_hist.__len__ = MagicMock(return_value=250)
        close_series = MagicMock()
        close_series.iloc.__getitem__ = MagicMock(return_value=150.0)
        close_series.rolling = MagicMock()
        close_series.tail = MagicMock()
        mock_hist.__getitem__ = MagicMock(return_value=close_series)
        volume_series = MagicMock()
        volume_series.iloc.__getitem__ = MagicMock(return_value=1e6)
        volume_series.tail = MagicMock(return_value=MagicMock(mean=MagicMock(return_value=1e6)))
        # dict-like Zugriff für hist["Close"], hist["Volume"]
        mock_hist.__getitem__ = MagicMock(
            side_effect=lambda key: close_series if key == "Close" else volume_series
        )
        mock_yf_ticker.history.return_value = mock_hist
        mock_yf_ticker.info = {
            "longName": "Test Corp",
            "sector": "Tech",
            "industry": "Semiconductors",
            "currency": "USD",
        }
        mock_yf_ticker.news = []  # keine yfinance-News

        with patch("concilium.data.yf.Ticker", return_value=mock_yf_ticker), \
             patch("concilium.data._fetch_google_news", return_value=[]), \
             patch("concilium.data._fetch_stocktwits", return_value=[
                 {"title": "Bullish NVDA surge", "published": datetime.now(timezone.utc), "source": "web"},
             ]), \
             patch("concilium.data._fetch_reddit", return_value=[
                 {"title": "NVDA earnings beat", "published": datetime.now(timezone.utc), "source": "reddit"},
             ]), \
             patch("concilium.data._fetch_macro_data", return_value={}), \
             patch("concilium.data._save_cache"):
            result = collect_ticker_data("NVDA")

        nwd = result["news_with_dates"]
        assert len(nwd) >= 2
        sources = {item["source"] for item in nwd if "source" in item}
        assert "web" in sources or "StockTwits" in sources
        assert "reddit" in sources

    def test_sentiment_includes_social_texts(self):
        """Sentiment-Zählung beinhaltet StockTwits/Reddit-Texte."""
        from concilium.data import collect_ticker_data

        mock_yf_ticker = MagicMock()
        mock_hist = MagicMock()
        mock_hist.empty = False
        close_series = MagicMock()
        close_series.iloc.__getitem__ = MagicMock(return_value=150.0)
        close_series.rolling = MagicMock(return_value=MagicMock(mean=MagicMock(return_value=MagicMock(__getitem__=MagicMock(return_value=145.0)))))
        volume_series = MagicMock()
        volume_series.iloc.__getitem__ = MagicMock(return_value=1e6)
        volume_series.tail = MagicMock(return_value=MagicMock(mean=MagicMock(return_value=1e6)))
        mock_hist.__getitem__ = MagicMock(
            side_effect=lambda key: close_series if key == "Close" else volume_series
        )
        mock_yf_ticker.history.return_value = mock_hist
        mock_yf_ticker.info = {"longName": "Test Corp", "currency": "USD", "sector": "Tech", "industry": "Semis"}
        mock_yf_ticker.news = []

        with patch("concilium.data.yf.Ticker", return_value=mock_yf_ticker), \
             patch("concilium.data._fetch_google_news", return_value=[]), \
             patch("concilium.data._fetch_stocktwits", return_value=[
                 {"title": "NVDA surges to record high", "published": datetime.now(timezone.utc), "source": "web"},
                 {"title": "NVDA plunges on weak data", "published": datetime.now(timezone.utc), "source": "web"},
             ]), \
             patch("concilium.data._fetch_reddit", return_value=[
                 {"title": "NVDA profit surge bullish", "published": datetime.now(timezone.utc), "source": "reddit"},
             ]), \
             patch("concilium.data._fetch_macro_data", return_value={}), \
             patch("concilium.data._save_cache"):
            result = collect_ticker_data("NVDA")

        sentiment = result["sentiment"]
        # 3 Headlines: "surges to record high" (positiv), "plunges on weak data" (negativ),
        # "profit surge bullish" (positiv) → positiv sollte > 0 sein
        assert sentiment["positiv"] > 0
        assert sentiment["negativ"] > 0
        assert sentiment["sample_size"] >= 3
        assert "stocktwits" in sentiment.get("sources", [])
        assert "reddit" in sentiment.get("sources", [])

    def test_news_list_still_returns_strings(self):
        """result['news'] bleibt list[str] für Rückwärtskompatibilität."""
        from concilium.data import collect_ticker_data

        mock_yf_ticker = MagicMock()
        mock_hist = MagicMock()
        mock_hist.empty = False
        close_series = MagicMock()
        close_series.iloc.__getitem__ = MagicMock(return_value=150.0)
        close_series.rolling = MagicMock(return_value=MagicMock(mean=MagicMock(return_value=MagicMock(__getitem__=MagicMock(return_value=145.0)))))
        volume_series = MagicMock()
        volume_series.iloc.__getitem__ = MagicMock(return_value=1e6)
        volume_series.tail = MagicMock(return_value=MagicMock(mean=MagicMock(return_value=1e6)))
        mock_hist.__getitem__ = MagicMock(
            side_effect=lambda key: close_series if key == "Close" else volume_series
        )
        mock_yf_ticker.history.return_value = mock_hist
        mock_yf_ticker.info = {"longName": "Test", "currency": "USD", "sector": "T", "industry": "I"}
        mock_yf_ticker.news = []

        with patch("concilium.data.yf.Ticker", return_value=mock_yf_ticker), \
             patch("concilium.data._fetch_google_news", return_value=[]), \
             patch("concilium.data._fetch_stocktwits", return_value=[
                 {"title": "Test headline", "published": datetime.now(timezone.utc), "source": "web"},
             ]), \
             patch("concilium.data._fetch_reddit", return_value=[]), \
             patch("concilium.data._fetch_macro_data", return_value={}), \
             patch("concilium.data._save_cache"):
            result = collect_ticker_data("TEST")

        news = result["news"]
        assert isinstance(news, list)
        assert all(isinstance(h, str) for h in news)


# ---------------------------------------------------------------------------
# 4. Fallback: StockTwits/Reddit-Fehler → kein Crash
# ---------------------------------------------------------------------------


class TestFallbackBehavior:
    """Bei Fehler der Zusatzquellen fällt sauber auf yfinance/Google zurück."""

    def test_stocktwits_error_does_not_crash(self):
        """StockTwits wirft Exception → [] → yfinance bleibt, kein Crash."""
        from concilium.data import collect_ticker_data

        mock_yf_ticker = MagicMock()
        mock_hist = MagicMock()
        mock_hist.empty = False
        close_series = MagicMock()
        close_series.iloc.__getitem__ = MagicMock(return_value=150.0)
        close_series.rolling = MagicMock(return_value=MagicMock(mean=MagicMock(return_value=MagicMock(__getitem__=MagicMock(return_value=145.0)))))
        volume_series = MagicMock()
        volume_series.iloc.__getitem__ = MagicMock(return_value=1e6)
        volume_series.tail = MagicMock(return_value=MagicMock(mean=MagicMock(return_value=1e6)))
        mock_hist.__getitem__ = MagicMock(
            side_effect=lambda key: close_series if key == "Close" else volume_series
        )
        mock_yf_ticker.history.return_value = mock_hist
        mock_yf_ticker.info = {"longName": "Test", "currency": "USD", "sector": "T", "industry": "I"}
        mock_yf_ticker.news = []  # yfinance leer

        with patch("concilium.data.yf.Ticker", return_value=mock_yf_ticker), \
             patch("concilium.data._fetch_google_news", return_value=[
                 {"title": "Google headline rally", "published": datetime.now(timezone.utc)},
             ]), \
             patch("concilium.data._fetch_stocktwits", side_effect=Exception("ST down")), \
             patch("concilium.data._fetch_reddit", side_effect=Exception("Reddit down")), \
             patch("concilium.data._fetch_macro_data", return_value={}), \
             patch("concilium.data._save_cache"):
            result = collect_ticker_data("NVDA")

        # yfinance/Google lieferten Headlines, kein Crash
        assert len(result["news"]) >= 1
        assert result["news_source"] != "none"

    def test_all_sources_fail_still_returns_result(self):
        """Alle Quellen leer → Result mit leeren Sentiment, kein Crash."""
        from concilium.data import collect_ticker_data

        mock_yf_ticker = MagicMock()
        mock_hist = MagicMock()
        mock_hist.empty = False
        close_series = MagicMock()
        close_series.iloc.__getitem__ = MagicMock(return_value=150.0)
        close_series.rolling = MagicMock(return_value=MagicMock(mean=MagicMock(return_value=MagicMock(__getitem__=MagicMock(return_value=145.0)))))
        volume_series = MagicMock()
        volume_series.iloc.__getitem__ = MagicMock(return_value=1e6)
        volume_series.tail = MagicMock(return_value=MagicMock(mean=MagicMock(return_value=1e6)))
        mock_hist.__getitem__ = MagicMock(
            side_effect=lambda key: close_series if key == "Close" else volume_series
        )
        mock_yf_ticker.history.return_value = mock_hist
        mock_yf_ticker.info = {"longName": "Test", "currency": "USD", "sector": "T", "industry": "I"}
        mock_yf_ticker.news = []

        with patch("concilium.data.yf.Ticker", return_value=mock_yf_ticker), \
             patch("concilium.data._fetch_google_news", return_value=[]), \
             patch("concilium.data._fetch_stocktwits", return_value=[]), \
             patch("concilium.data._fetch_reddit", return_value=[]), \
             patch("concilium.data._fetch_macro_data", return_value={}), \
             patch("concilium.data._save_cache"):
            result = collect_ticker_data("EMPTY")

        assert result["news"] == []
        assert result["news_source"] == "none"
        assert result["sentiment"]["sample_size"] == 0
        assert result["sentiment"]["dominant"] == "neutral"


# ---------------------------------------------------------------------------
# 5. Report: Quelle je Headline
# ---------------------------------------------------------------------------


class TestReportSourceLabel:
    """Tests für _source_label und Report-Generierung."""

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("yfinance", "yfinance"),
            ("google", "Google"),
            ("google_news", "Google"),
            ("stocktwits", "StockTwits"),
            ("web", "Web"),
            ("reddit", "Reddit"),
            ("StockTwits", "StockTwits"),
        ],
    )
    def test_source_label_mapping(self, source, expected):
        assert _source_label(source) == expected

    def test_source_label_empty(self):
        assert _source_label("") == "unknown"

    def test_report_shows_source_tag(self):
        """Report zeigt [StockTwits]/[Reddit] vor Headlines mit source."""
        result = {
            "ticker": "NVDA",
            "no_llm": True,
            "data": {
                "fundamentals": {"name": "NVIDIA", "sector": "Tech", "industry": "Semis", "currency": "USD"},
                "technicals": {},
                "sentiment": {"positiv": 1, "negativ": 0, "neutral": 0, "dominant": "positiv",
                              "sample_size": 3, "weighted": False,
                              "sources": ["yfinance", "stocktwits", "reddit"]},
                "news": [
                    "NVDA surges to record high",
                    "NVDA plunges on weak data",
                    "NVDA profit bullish",
                ],
                "news_with_dates": [
                    {"title": "NVDA surges to record high", "published": None, "source": "yfinance"},
                    {"title": "NVDA plunges on weak data", "published": None, "source": "StockTwits"},
                    {"title": "NVDA profit bullish", "published": None, "source": "reddit"},
                ],
                "news_source": "yfinance, stocktwits, reddit",
            },
        }

        report = generate_report(result)
        assert "[yfinance] NVDA surges to record high" in report
        assert "[StockTwits] NVDA plunges on weak data" in report
        assert "[Reddit] NVDA profit bullish" in report
        # Quellen-Zeile im Sentiment-Abschnitt
        assert "Quellen:" in report

    def test_report_robust_without_source(self):
        """Report ohne source-Feld → Headlines ohne Tag (rückwärtskompatibel)."""
        result = {
            "ticker": "AAPL",
            "no_llm": True,
            "data": {
                "fundamentals": {"name": "Apple", "sector": "Tech", "industry": "Consumer", "currency": "USD"},
                "technicals": {},
                "sentiment": {"positiv": 1, "negativ": 0, "neutral": 0, "dominant": "positiv",
                              "sample_size": 1, "weighted": False},
                "news": ["Apple announces new product"],
                "news_with_dates": [],
                "news_source": "yfinance",
            },
        }

        report = generate_report(result)
        assert "- Apple announces new product" in report
        # Kein Quellen-Tag in der Headline
        assert "[yfinance] Apple announces new product" not in report

    def test_report_shows_sources_line(self):
        """Sentiment-Sektion zeigt Quellen-Zeile, wenn sources vorhanden."""
        result = {
            "ticker": "NVDA",
            "no_llm": True,
            "data": {
                "fundamentals": {"name": "NVIDIA", "sector": "Tech", "industry": "Semis", "currency": "USD"},
                "technicals": {},
                "sentiment": {
                    "positiv": 2.0, "negativ": 1.0, "neutral": 0.5,
                    "dominant": "positiv", "sample_size": 3, "weighted": True,
                    "sources": ["yfinance", "stocktwits", "reddit"],
                },
                "news": [],
                "news_with_dates": [],
                "news_source": "yfinance, stocktwits, reddit",
            },
        }

        report = generate_report(result)
        assert "Quellen:" in report
        assert "yfinance" in report
        assert "StockTwits" in report
        assert "Reddit" in report

    def test_report_no_sources_line_when_empty(self):
        """Sentiment ohne sources → keine Quellen-Zeile."""
        result = {
            "ticker": "AAPL",
            "no_llm": True,
            "data": {
                "fundamentals": {"name": "Apple", "sector": "Tech", "industry": "Consumer", "currency": "USD"},
                "technicals": {},
                "sentiment": {"positiv": 1, "negativ": 0, "neutral": 0, "dominant": "positiv",
                              "sample_size": 1, "weighted": False},
                "news": [],
                "news_with_dates": [],
                "news_source": "yfinance",
            },
        }

        report = generate_report(result)
        assert "Quellen:" not in report
