"""Test für data.py — prüft collect_ticker_data('AAPL').

Benötigt Netzwerkzugriff für yfinance. Falls kein Netz verfügbar, wird der Test übersprungen.
"""

from __future__ import annotations

import os

# src zum Pfad hinzufügen
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tradingagents_light.data import (
    _fetch_google_news,  # noqa: E402
    collect_ticker_data,  # noqa: E402
)


def _has_network() -> bool:
    """Prüft, ob yfinance Daten abrufen kann."""
    try:
        import yfinance as yf

        t = yf.Ticker("AAPL")
        hist = t.history(period="5d")
        return hist is not None and not hist.empty
    except Exception:  # noqa: BLE001
        return False


# Skip entire module if no network
pytestmark = pytest.mark.skipif(not _has_network(), reason="Kein Netzwerkzugriff für yfinance")


class TestCollectTickerData:
    """Tests für collect_ticker_data."""

    @pytest.fixture(scope="class")
    def aapl_data(self):
        """Sammelt AAPL-Daten einmal für alle Tests."""
        return collect_ticker_data("AAPL")

    def test_ticker_present(self, aapl_data):
        assert aapl_data["ticker"] == "AAPL"

    def test_fundamentals_present(self, aapl_data):
        f = aapl_data["fundamentals"]
        assert "name" in f
        assert f["name"] is not None
        assert "market_cap" in f
        assert "pe_ratio" in f
        assert "eps" in f
        assert "fifty_two_week_high" in f
        assert "fifty_two_week_low" in f

    def test_technicals_present(self, aapl_data):
        t = aapl_data["technicals"]
        assert "current_price" in t
        assert t["current_price"] is not None
        assert t["current_price"] > 0
        assert "sma50" in t
        assert "sma200" in t
        assert "rsi14" in t

    def test_rsi_in_range(self, aapl_data):
        rsi = aapl_data["technicals"]["rsi14"]
        if rsi is not None:
            assert 0 <= rsi <= 100, f"RSI {rsi} nicht in [0, 100]"

    def test_history_present(self, aapl_data):
        hist = aapl_data["history"]
        assert len(hist) > 0
        # Mindestens 100 Tage sollten vorhanden sein
        assert len(hist) >= 100, f"Nur {len(hist)} Historie-Tage"

    def test_sentiment_present(self, aapl_data):
        s = aapl_data["sentiment"]
        assert "positiv" in s
        assert "negativ" in s
        assert "neutral" in s
        # Summe sollte >= 0 sein
        total = s["positiv"] + s["negativ"] + s["neutral"]
        assert total >= 0

    def test_invalid_ticker_raises(self):
        with pytest.raises(ValueError, match="Ungültiger Ticker"):
            collect_ticker_data("INVALIDTICKERXYZ123")


class TestSentimentHeuristic:
    """Tests für die Sentiment-Heuristik."""

    def test_positive_keyword(self):
        from tradingagents_light.data import _count_sentiment

        result = _count_sentiment(["Apple surges to record high"])
        assert result["positiv"] >= 1

    def test_negative_keyword(self):
        from tradingagents_light.data import _count_sentiment

        result = _count_sentiment(["Apple plunges on weak earnings"])
        assert result["negativ"] >= 1

    def test_neutral_no_keywords(self):
        from tradingagents_light.data import _count_sentiment

        result = _count_sentiment(["Apple announces quarterly results"])
        assert result["neutral"] >= 1


# Mini-RSS-XML für Mock-Tests (2 echte Items + 1 "Top Stories" zum Überspringen)
_MOCK_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News</title>
    <item><title>Top Stories</title></item>
    <item><title>NVIDIA surges to record high on AI demand</title></item>
    <item><title>Chip stocks rally as NVDA beats earnings</title></item>
  </channel>
</rss>
"""


class TestGoogleNewsFallback:
    """Tests für _fetch_google_news — ohne echtes Netzwerk."""

    def test_returns_headlines_from_mock_xml(self):
        """Mock liefert Mini-XML mit 2 echten Items → Funktion gibt 2 Headlines zurück."""
        mock_response = MagicMock()
        mock_response.text = _MOCK_RSS_XML
        mock_response.raise_for_status = MagicMock()

        with patch("tradingagents_light.data.requests.get", return_value=mock_response):
            result = _fetch_google_news("NVDA", company_name="NVIDIA")

        # "Top Stories" wird übersprungen → 2 echte Headlines
        assert len(result) == 2
        assert "NVIDIA surges to record high on AI demand" in result
        assert "Chip stocks rally as NVDA beats earnings" in result

    def test_returns_empty_on_connection_error(self):
        """Bei requests.ConnectionError → leere Liste, kein Crash."""
        with patch("tradingagents_light.data.requests.get", side_effect=ConnectionError("DNS failed")):
            result = _fetch_google_news("NVDA")

        assert result == []
