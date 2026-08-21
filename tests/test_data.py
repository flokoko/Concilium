"""Test für data.py — prüft collect_ticker_data('AAPL').

Benötigt Netzwerkzugriff für yfinance. Falls kein Netz verfügbar, wird der Test übersprungen.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

# src zum Pfad hinzufügen
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tradingagents_light.data import (  # noqa: E402
    _fetch_google_news,
    _get_dividend_yield,
    collect_ticker_data,
)
from tradingagents_light.journal import append_decision


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
    <item>
      <title>NVIDIA surges to record high on AI demand</title>
      <pubDate>Wed, 20 Aug 2026 08:30:00 GMT</pubDate>
    </item>
    <item>
      <title>Chip stocks rally as NVDA beats earnings</title>
      <pubDate>Tue, 19 Aug 2026 14:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

# Mini-RSS-XML mit dc:date statt pubDate
_MOCK_RSS_DC_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>Google News</title>
    <item>
      <title>Apple announces new product line</title>
      <dc:date>2026-08-18T10:00:00Z</dc:date>
    </item>
  </channel>
</rss>
"""


class TestGoogleNewsFallback:
    """Tests für _fetch_google_news — ohne echtes Netzwerk."""

    def test_returns_headlines_from_mock_xml(self):
        """Mock liefert Mini-XML mit 2 echten Items → Funktion gibt 2 dicts zurück."""
        mock_response = MagicMock()
        mock_response.text = _MOCK_RSS_XML
        mock_response.raise_for_status = MagicMock()

        with patch("tradingagents_light.data.requests.get", return_value=mock_response):
            result = _fetch_google_news("NVDA", company_name="NVIDIA")

        # "Top Stories" wird übersprungen → 2 echte Items
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert result[0]["title"] == "NVIDIA surges to record high on AI demand"
        assert result[1]["title"] == "Chip stocks rally as NVDA beats earnings"

    def test_returns_empty_on_connection_error(self):
        """Bei requests.ConnectionError → leere Liste, kein Crash."""
        with patch("tradingagents_light.data.requests.get", side_effect=ConnectionError("DNS failed")):
            result = _fetch_google_news("NVDA")

        assert result == []

    def test_pubdate_parsing(self):
        """pubDate (RFC-822) wird korrekt zu datetime geparst."""
        mock_response = MagicMock()
        mock_response.text = _MOCK_RSS_XML
        mock_response.raise_for_status = MagicMock()

        with patch("tradingagents_light.data.requests.get", return_value=mock_response):
            result = _fetch_google_news("NVDA")

        assert len(result) == 2
        # Erste Headline: "Wed, 20 Aug 2026 08:30:00 GMT"
        pub = result[0]["published"]
        assert isinstance(pub, datetime)
        assert pub.tzinfo is not None
        assert pub.year == 2026
        assert pub.month == 8
        assert pub.day == 20

    def test_dc_date_parsing(self):
        """dc:date (ISO-8601) wird korrekt geparst, wenn kein pubDate vorhanden."""
        mock_response = MagicMock()
        mock_response.text = _MOCK_RSS_DC_XML
        mock_response.raise_for_status = MagicMock()

        with patch("tradingagents_light.data.requests.get", return_value=mock_response):
            result = _fetch_google_news("AAPL")

        assert len(result) == 1
        pub = result[0]["published"]
        assert isinstance(pub, datetime)
        assert pub.tzinfo is not None
        assert pub.year == 2026
        assert pub.month == 8
        assert pub.day == 18

    def test_missing_date_fallbacks_to_now(self):
        """Items ohne pubDate/dc:date → published wird auf aktuelle Zeit gesetzt."""
        rss_no_date = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title>Some headline without date</title></item>
</channel></rss>
"""
        mock_response = MagicMock()
        mock_response.text = rss_no_date
        mock_response.raise_for_status = MagicMock()

        before = datetime.now(timezone.utc)
        with patch("tradingagents_light.data.requests.get", return_value=mock_response):
            result = _fetch_google_news("TEST")
        after = datetime.now(timezone.utc)

        assert len(result) == 1
        pub = result[0]["published"]
        assert pub.tzinfo is not None
        # Sollte etwa jetzt sein (innerhalb des before/after-Zeitfensters)
        assert before <= pub <= after


class TestSentimentWeighted:
    """Tests für zeitgewichtete Sentiment-Zählung."""

    def test_recent_headline_weights_more_than_old(self):
        """Eine heutige positive Headline soll mehr Gewicht haben als eine 14 Tage alte."""
        from tradingagents_light.data import _count_sentiment_weighted

        now = datetime.now(timezone.utc)
        headlines = [
            # Positive Headline von heute → Gewicht ≈ 1.0
            {"title": "Company surges to record high", "published": now},
            # Positive Headline von vor 14 Tagen → Gewicht ≈ 2^(-14/7) = 0.25
            {"title": "Company surges on strong growth", "published": now - timedelta(days=14)},
        ]

        result = _count_sentiment_weighted(headlines, now=now)

        # Beide sind positiv → gesamte gewichtete Summe sollte > 0 sein
        assert result["positiv"] > 0
        assert result["weighted"] is True
        assert result["sample_size"] == 2

        # Die heutige Headline wiegt ~1.0, die alte ~0.25 → Summe ~1.25
        # Die heutige allein (1.0) ist größer als der Beitrag der alten (0.25)
        recent_weight = 2.0 ** (0.0 / 7.0)  # age=0 → 1.0
        old_weight = 2.0 ** (-14.0 / 7.0)   # age=14 → 0.25
        assert recent_weight > old_weight
        # Die gewichtete Summe sollte etwa recent + old entsprechen
        assert abs(result["positiv"] - (recent_weight + old_weight)) < 0.01

    def test_dominant_sentiment(self):
        """Die dominante Stimmung wird korrekt ermittelt."""
        from tradingagents_light.data import _count_sentiment_weighted

        now = datetime.now(timezone.utc)
        headlines = [
            # Neu und positiv → hohes Gewicht
            {"title": "Stock surges on record profit", "published": now},
            # Alt und negativ → geringes Gewicht
            {"title": "Stock plunges on weak earnings", "published": now - timedelta(days=30)},
        ]

        result = _count_sentiment_weighted(headlines, now=now)

        # Positiv sollte dominieren (Gewicht 1.0 vs. 2^(-30/7) ≈ 0.052)
        assert result["dominant"] == "positiv"
        assert result["positiv"] > result["negativ"]

    def test_old_negative_can_outweigh_new_positive(self):
        """Wenn mehrere alte negative Headlines vs. eine neue positive → negativ dominiert."""
        from tradingagents_light.data import _count_sentiment_weighted

        now = datetime.now(timezone.utc)
        headlines = [
            # Eine neue positive Headline
            {"title": "Company surges", "published": now},
            # Eine neue negative Headline
            {"title": "Company plunges", "published": now},
            # Drei weitere neue negative Headlines
            {"title": "Company falls on weak data", "published": now - timedelta(hours=1)},
            {"title": "Company drops after lawsuit", "published": now - timedelta(hours=2)},
            {"title": "Company decline on risk warning", "published": now - timedelta(hours=3)},
        ]

        result = _count_sentiment_weighted(headlines, now=now)

        # Negativ sollte dominieren (4 negative vs. 1 positive, alle etwa gleich alt)
        assert result["dominant"] == "negativ"
        assert result["negativ"] > result["positiv"]
        assert result["sample_size"] == 5

    def test_empty_headlines(self):
        """Leere Liste → alle Werte 0, dominant neutral."""
        from tradingagents_light.data import _count_sentiment_weighted

        result = _count_sentiment_weighted([])
        assert result["positiv"] == 0.0
        assert result["negativ"] == 0.0
        assert result["neutral"] == 0.0
        assert result["dominant"] == "neutral"
        assert result["sample_size"] == 0
        assert result["weighted"] is True


# ---------------------------------------------------------------------------
# Feature 1-3: Neue Datenerfassungs-Tests
# ---------------------------------------------------------------------------


class TestAnalystFields:
    """Feature 1: Analysten-Erwartungen in fundamentals."""

    @pytest.fixture(scope="class")
    def rwe_data(self):
        """Sammelt RWE.DE-Daten einmal für alle Tests."""
        return collect_ticker_data("RWE.DE")

    def test_analyst_fields_present(self, rwe_data):
        """fundamentals hat analyst_target_mean (Float oder None) und analyst_upside_pct Schlüssel."""
        f = rwe_data["fundamentals"]
        assert "analyst_target_mean" in f
        assert "analyst_upside_pct" in f
        # analyst_target_mean sollte float oder None sein
        assert f["analyst_target_mean"] is None or isinstance(f["analyst_target_mean"], float)
        # analyst_upside_pct sollte float oder None sein
        assert f["analyst_upside_pct"] is None or isinstance(f["analyst_upside_pct"], float)

    def test_recommendation_fields_present(self, rwe_data):
        """recommendation_key und analyst_count sollten als Schlüssel existieren."""
        f = rwe_data["fundamentals"]
        assert "recommendation_key" in f
        assert "analyst_count" in f
        assert "recommendation_mean" in f


class TestMacroFields:
    """Feature 2: Makro/Zins-Daten."""

    @pytest.fixture(scope="class")
    def rwe_data(self):
        return collect_ticker_data("RWE.DE")

    def test_macro_keys_present(self, rwe_data):
        """data['macro'] hat die erwarteten Keys."""
        macro = rwe_data["macro"]
        assert "us_10y_yield" in macro
        assert "us_10y_yield_1m_ago" in macro
        assert "us_10y_trend" in macro
        assert "sp500_pe" in macro
        assert "sp500_market_cap" in macro
        assert "sp500_source" in macro


class TestPeersOptional:
    """Feature 3: Peer-Vergleich (optionaler Parameter)."""

    def test_peers_optional(self):
        """collect_ticker_data('RWE.DE', peers=['SHEL.L']) → data['peers'] ist eine Liste."""
        data = collect_ticker_data("RWE.DE", peers=["SHEL.L"])
        assert "peers" in data
        assert isinstance(data["peers"], list)
        # Best effort: wenn Netz verfügbar, sollte mindestens 1 Eintrag drin sein
        # Wenn Netz ausfällt → leere Liste (nicht failen)
        for entry in data["peers"]:
            assert "ticker" in entry
            assert "pe_ratio" in entry
            assert "market_cap" in entry
            assert "name" in entry

    def test_peers_default_empty(self):
        """Ohne peers-Parameter ist data['peers'] eine leere Liste."""
        data = collect_ticker_data("AAPL")
        assert "peers" in data
        assert isinstance(data["peers"], list)
        assert len(data["peers"]) == 0


# ---------------------------------------------------------------------------
# Feature 4: Entscheidungs-Journal Tests
# ---------------------------------------------------------------------------


class TestAppendDecision:
    """Feature 4: append_decision schreibt CSV-Journal."""

    def test_append_decision_creates_file(self, tmp_path):
        """append_decision mit Mock-result → Datei/Ordner entsteht + Header vorhanden."""
        mock_result = {
            "ticker": "TEST.DE",
            "trade": {
                "aktion": "KAUFEN",
                "zielkurs": 150.0,
                "stop_loss": 130.0,
                "positionsanteil": 5,
            },
            "final": {
                "entscheidung": "GENEHMIGT",
                "confidence": 4,
            },
            "debate": {
                "bull": {"_raw": '{"confidence": 4, "name": "Bull"}\nSteigt!'},
                "bear": {"_raw": '{"confidence": 2, "name": "Bear"}\nFällt!'},
            },
        }

        journal_file = str(tmp_path / "journal" / "decisions.csv")
        append_decision(mock_result, journal_file=journal_file)

        # Datei sollte existieren
        import os

        assert os.path.isfile(journal_file)

        # Header und Daten prüfen
        import csv as csv_mod

        with open(journal_file, encoding="utf-8") as fh:
            reader = csv_mod.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["ticker"] == "TEST.DE"
        assert row["action"] == "KAUFEN"
        assert row["target"] == "150.0"
        assert row["stop"] == "130.0"
        assert row["position_pct"] == "5"
        assert row["final_decision"] == "GENEHMIGT"
        assert row["confidence"] == "4"
        assert row["bull_confidence"] == "4"
        assert row["bear_confidence"] == "2"

    def test_append_decision_empty_fields(self, tmp_path):
        """Bei leerem result werden Felder leer gelassen (nicht crashen)."""
        mock_result: dict = {}

        journal_file = str(tmp_path / "journal_empty" / "decisions.csv")
        append_decision(mock_result, journal_file=journal_file)

        import os

        assert os.path.isfile(journal_file)

        import csv as csv_mod

        with open(journal_file, encoding="utf-8") as fh:
            reader = csv_mod.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["ticker"] == ""
        assert row["action"] == ""

    def test_append_decision_appends(self, tmp_path):
        """Mehrere Aufrufe anhängen an dieselbe Datei."""
        journal_file = str(tmp_path / "journal_append" / "decisions.csv")

        for i in range(3):
            mock_result = {
                "ticker": f"TEST{i}.DE",
                "trade": {"aktion": "KAUFEN"},
                "final": {"entscheidung": "GENEHMIGT", "confidence": 3},
            }
            append_decision(mock_result, journal_file=journal_file)

        import csv as csv_mod

        with open(journal_file, encoding="utf-8") as fh:
            reader = csv_mod.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["ticker"] == "TEST0.DE"
        assert rows[2]["ticker"] == "TEST2.DE"


class TestGetDividendYield:
    """Tests für _get_dividend_yield — robuste Dividendenrendite-Bestimmung."""

    def test_dividend_yield_uses_trailing(self):
        """Trailing (TTM) wird bevorzugt, auch wenn dividendYield kaputt ist."""
        info = {"trailingAnnualDividendYield": 0.021, "dividendYield": 2.1}
        assert _get_dividend_yield(info) == 0.021

    def test_dividend_yield_fallback_plausible(self):
        """Fallback auf dividendYield, wenn Trailing fehlt und Wert plausibel ist."""
        info = {"trailingAnnualDividendYield": None, "dividendYield": 0.03}
        assert _get_dividend_yield(info) == 0.03

    def test_dividend_yield_rejects_implausible(self):
        """Unplausibler dividendYield (≥0.5) wird verworfen → None."""
        info = {"trailingAnnualDividendYield": None, "dividendYield": 2.1}
        assert _get_dividend_yield(info) is None

    def test_dividend_yield_none(self):
        """Beide None → None, kein Crash."""
        info = {"trailingAnnualDividendYield": None, "dividendYield": None}
        assert _get_dividend_yield(info) is None
