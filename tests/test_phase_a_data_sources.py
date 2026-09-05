"""Tests für Phase A — Neue Datenquellen (offline, gemockt).

Bereiche:
  1. _fetch_insider_transactions — DataFrame-Parsing (yfinance-Style),
     Fehlerspalten → None, Exception → leere Liste, limit
  2. _fetch_polymarket — Parsing (outcomePrices als JSON-String + Liste),
     403/blocked → leere Liste, limit
  3. _fetch_global_macro_news — Deduplizierung, limit, Query-Fehler → leere Liste
  4. collect_ticker_data-Integration — neue Keys vorhanden (gemockte Fetches)
  5. _build_data_text — Rollen-Filterung (fundamental/macro_news/alle/sentiment)
  6. generate_report — Sektionen nur wenn Daten vorhanden, sonst keine Sektion
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# src zum Pfad hinzufügen
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import _build_data_text  # noqa: E402
from concilium.data import (  # noqa: E402
    _fetch_global_macro_news,
    _fetch_insider_transactions,
    _fetch_polymarket,
)
from concilium.report import generate_report  # noqa: E402

# ---------------------------------------------------------------------------
# Mock-Daten
# ---------------------------------------------------------------------------

# Insider-DataFrame im yfinance-Style (Spalten wie "Start Date", "Insider Name", …)
_INSIDER_DF = pd.DataFrame({
    "Start Date": ["2026-08-15", "2026-08-10", "2026-08-01"],
    "Insider Name": ["Cook Timothy D", "Gore Mark", "Kondo Susan"],
    "Transaction": ["Purchase", "Sale", "Sale"],
    "Shares": [2000, 10000, 500],
    "Price": [180.5, 175.0, 190.0],
    "Value": [361000.0, 1750000.0, 95000.0],
})

_INSIDER_DF_PARTIAL = pd.DataFrame({
    "Start Date": ["2026-08-15"],
    "Transaction": ["Purchase"],
    # "Insider Name", "Shares", "Price", "Value" fehlen → None
})

_POLYMARKET_LIST = [
    {
        "question": "Will NVDA beat earnings in Q3?",
        "outcomePrices": "[\"0.65\", \"0.35\"]",  # JSON-String
        "category": "Tech",
    },
    {
        "question": "Will the Fed cut rates in 2026?",
        "outcomePrices": ["0.40", "0.60"],  # bereits Liste
        "category": "Macro",
    },
    {
        "question": "Unparseable prices market",
        "outcomePrices": "not-a-json",
        "category": "Other",
    },
    {"title": "Title-fallback market", "outcomePrices": ["0.55"]},
    {"question": "", "outcomePrices": ["0.5"]},  # leerer Titel → skip
]


def _mock_response(json_data):
    """Erstellt einen MagicMock-Response mit .json() und .raise_for_status()."""
    mock = MagicMock()
    mock.json.return_value = json_data
    mock.raise_for_status = MagicMock()
    return mock


# RSS-Feed mit 2 Headlines (für Global-Macro-News-Tests)
_MACRO_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item><title>Fed holds rates steady amid inflation concerns</title></item>
  <item><title>Oil rises on supply chain disruptions</title></item>
</channel></rss>
"""


# ---------------------------------------------------------------------------
# 1. _fetch_insider_transactions
# ---------------------------------------------------------------------------


class TestFetchInsiderTransactions:
    """Tests für _fetch_insider_transactions — Parsing + Fehlerfälle."""

    def test_parses_dataframe(self):
        """Mock-DataFrame → Liste von dicts mit korrekten Werten."""
        with patch("concilium.data.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.insider_transactions = _INSIDER_DF
            result = _fetch_insider_transactions("AAPL")

        assert len(result) == 3
        assert result[0]["insider"] == "Cook Timothy D"
        assert result[0]["transaction"] == "Purchase"
        assert result[0]["shares"] == 2000
        assert result[0]["price"] == 180.5
        assert result[0]["value"] == 361000.0
        assert result[0]["date"] == "2026-08-15"

    def test_missing_columns_become_none(self):
        """Fehlende Spalten → None (kein Crash, robust gegen jede Struktur)."""
        with patch("concilium.data.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.insider_transactions = _INSIDER_DF_PARTIAL
            result = _fetch_insider_transactions("AAPL")

        assert len(result) == 1
        assert result[0]["insider"] is None
        assert result[0]["shares"] is None
        assert result[0]["price"] is None
        assert result[0]["value"] is None
        assert result[0]["transaction"] == "Purchase"

    def test_newer_yfinance_schema_insider_column(self):
        """Neuere yfinance-Version: Spalte 'Insider' statt 'Insider Name',
        Transaktionsart im 'Text'-Feld ("Sale at price …" → "Sale")."""
        df_new = pd.DataFrame({
            "Start Date": ["2026-08-25"],
            "Insider": ["NEWSTEAD JENNIFER"],
            "Transaction": [""],
            "Text": ["Sale at price 310.95 per share."],
            "Shares": [1439],
            "Value": [447457.0],
        })
        with patch("concilium.data.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.insider_transactions = df_new
            result = _fetch_insider_transactions("AAPL")

        assert len(result) == 1
        assert result[0]["insider"] == "NEWSTEAD JENNIFER"
        assert result[0]["transaction"] == "Sale"
        assert result[0]["shares"] == 1439
        assert result[0]["price"] is None
        assert result[0]["value"] == 447457.0

    def test_exception_returns_empty_list(self):
        """yf.Ticker wirft → leere Liste, kein Crash."""
        with patch("concilium.data.yf.Ticker", side_effect=RuntimeError("network down")):
            result = _fetch_insider_transactions("AAPL")
        assert result == []

    def test_property_raises_returns_empty_list(self):
        """insider_transactions-Property wirft beim Zugriff → leere Liste."""
        t = MagicMock()
        type(t).insider_transactions = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        with patch("concilium.data.yf.Ticker", return_value=t):
            result = _fetch_insider_transactions("AAPL")
        assert result == []

    def test_none_df_returns_empty_list(self):
        """DataFrame None → leere Liste."""
        with patch("concilium.data.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.insider_transactions = None
            result = _fetch_insider_transactions("AAPL")
        assert result == []

    def test_empty_df_returns_empty_list(self):
        """Leerer DataFrame → leere Liste."""
        with patch("concilium.data.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.insider_transactions = pd.DataFrame()
            result = _fetch_insider_transactions("AAPL")
        assert result == []

    def test_nan_values_become_none(self):
        """NaN in Shares/Price/Value → None (_safe_float)."""
        df_nan = pd.DataFrame({
            "Start Date": ["2026-08-15"],
            "Insider Name": ["Cook Timothy D"],
            "Transaction": ["Purchase"],
            "Shares": [float("nan")],
            "Price": [180.5],
            "Value": [None],
        })
        with patch("concilium.data.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.insider_transactions = df_nan
            result = _fetch_insider_transactions("AAPL")

        assert len(result) == 1
        assert result[0]["shares"] is None
        assert result[0]["price"] == 180.5
        assert result[0]["value"] is None

    def test_respects_limit(self):
        """limit=2 → nur 2 Einträge."""
        with patch("concilium.data.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.insider_transactions = _INSIDER_DF
            result = _fetch_insider_transactions("AAPL", limit=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 2. _fetch_polymarket
# ---------------------------------------------------------------------------


class TestFetchPolymarket:
    """Tests für _fetch_polymarket — Parsing + Fehlerfälle."""

    def test_parses_markets(self):
        """Mock-Antwort (Liste) → dicts mit Titel + Wahrscheinlichkeit."""
        with patch("concilium.data.requests.get", return_value=_mock_response(_POLYMARKET_LIST)):
            result = _fetch_polymarket("NVDA")

        # 5 Einträge, der leere-Titel-Eintrag wird übersprungen → 4
        assert len(result) == 4
        assert result[0]["title"] == "Will NVDA beat earnings in Q3?"
        assert result[0]["probability"] == pytest.approx(0.65)
        assert result[0]["category"] == "Tech"
        # Zweiter Eintrag: Liste statt JSON-String
        assert result[1]["probability"] == pytest.approx(0.40)
        # Unparsebare Preise → None (kein Crash)
        assert result[2]["probability"] is None
        # Titel-Fallback über "title"-Key
        assert result[3]["title"] == "Title-fallback market"

    def test_data_wrapper_response(self):
        """Antwort als {"data": [...]} wird ebenfalls verarbeitet (Fallback-Endpoint)."""
        with patch(
            "concilium.data.requests.get",
            return_value=_mock_response({"data": _POLYMARKET_LIST[:1]}),
        ):
            result = _fetch_polymarket("NVDA")
        assert len(result) == 1
        assert result[0]["probability"] == pytest.approx(0.65)

    def test_public_search_events_preferred(self):
        """public-search mit Events → verschachtelte Märkte werden extrahiert."""
        public_search_response = {
            "events": [
                {
                    "title": "NVIDIA (NVDA) closes above ___?",
                    "category": "Tech",
                    "markets": [
                        {
                            "question": "Will NVDA close above $180?",
                            "outcomePrices": ["1", "0"],
                        },
                        {
                            "question": "Will NVDA close above $185?",
                            "outcomePrices": ["0.25", "0.75"],
                        },
                    ],
                },
            ],
        }
        with patch(
            "concilium.data.requests.get",
            return_value=_mock_response(public_search_response),
        ):
            result = _fetch_polymarket("NVDA")
        assert len(result) == 2
        assert result[0]["title"] == "Will NVDA close above $180?"
        assert result[0]["probability"] == pytest.approx(1.0)
        assert result[0]["category"] == "Tech"
        assert result[1]["probability"] == pytest.approx(0.25)

    def test_public_search_empty_falls_back_to_markets(self):
        """public-search liefert keine Events → Fallback auf markets?search=."""
        public_search_response = {"events": [], "profiles": [], "tags": []}

        def _fake_get(url, timeout=15, headers=None):
            if "public-search" in url:
                return _mock_response(public_search_response)
            return _mock_response(_POLYMARKET_LIST)

        with patch("concilium.data.requests.get", side_effect=_fake_get):
            result = _fetch_polymarket("NVDA")
        assert len(result) == 4
        assert result[0]["title"] == "Will NVDA beat earnings in Q3?"

    def test_probability_out_of_range_ignored(self):
        """Wahrscheinlichkeit außerhalb [0, 1] → None."""
        markets = [{"question": "Weird market", "outcomePrices": ["1.5"]}]
        with patch("concilium.data.requests.get", return_value=_mock_response(markets)):
            result = _fetch_polymarket("XYZ")
        assert len(result) == 1
        assert result[0]["probability"] is None

    def test_403_returns_empty_list(self):
        """403/blocked (erwartet in Container-Umgebung) → leere Liste."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("403 Forbidden")
        with patch("concilium.data.requests.get", return_value=mock_resp):
            result = _fetch_polymarket("NVDA")
        assert result == []

    def test_connection_error_returns_empty_list(self):
        """ConnectionError → leere Liste, kein Crash."""
        with patch(
            "concilium.data.requests.get", side_effect=ConnectionError("DNS failed")
        ):
            result = _fetch_polymarket("NVDA")
        assert result == []

    def test_empty_response_returns_empty_list(self):
        """Leere Liste als Antwort → leere Liste."""
        with patch("concilium.data.requests.get", return_value=_mock_response([])):
            result = _fetch_polymarket("NVDA")
        assert result == []

    def test_respects_limit(self):
        """limit=2 → nur 2 Einträge."""
        with patch(
            "concilium.data.requests.get", return_value=_mock_response(_POLYMARKET_LIST)
        ):
            result = _fetch_polymarket("NVDA", limit=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 3. _fetch_global_macro_news
# ---------------------------------------------------------------------------


class TestFetchGlobalMacroNews:
    """Tests für _fetch_global_macro_news — Deduplizierung + Fehlerfälle."""

    def test_deduplicates_across_queries(self):
        """Gleiche Headline in zwei Queries → nur einmal."""
        with patch(
            "concilium.data._fetch_google_news",
            side_effect=[
                [{"title": "Fed holds rates", "published": None}],
                [{"title": "Fed holds rates", "published": None}],
                [],
                [],
                [],
            ],
        ):
            result = _fetch_global_macro_news(limit=10)
        assert len(result) == 1
        assert result[0]["source"] == "global_macro"

    def test_respects_limit(self):
        """limit=3 → maximal 3 Headlines."""
        items = [{"title": f"Macro headline {i}", "published": None} for i in range(5)]

        def _fake_fetch(query, limit=10):
            return items

        with patch("concilium.data._fetch_google_news", side_effect=_fake_fetch):
            result = _fetch_global_macro_news(limit=3)
        assert len(result) == 3
        assert all(r["source"] == "global_macro" for r in result)

    def test_query_exception_returns_empty_list(self):
        """_fetch_google_news wirft für alle Queries → leere Liste, kein Crash."""
        with patch(
            "concilium.data._fetch_google_news",
            side_effect=RuntimeError("network down"),
        ):
            result = _fetch_global_macro_news()
        assert result == []

    def test_partial_failure_still_collects(self):
        """Erste Query wirft, zweite liefert → Headlines werden trotzdem gesammelt."""

        def _fake_fetch(query, limit=10):
            if "Federal" in query:
                raise ConnectionError("blocked")
            return [{"title": f"Headline for {query[:20]}", "published": None}]

        with patch("concilium.data._fetch_google_news", side_effect=_fake_fetch):
            result = _fetch_global_macro_news(limit=10)
        assert len(result) == 4  # 4 verbleibende Queries liefern je 1 Headline

    def test_via_mocked_rss(self):
        """End-to-End über gemocktes RSS: _fetch_google_news wird wirklich genutzt."""
        mock_resp = MagicMock()
        mock_resp.text = _MACRO_RSS_XML
        mock_resp.raise_for_status = MagicMock()
        with patch("concilium.data.requests.get", return_value=mock_resp):
            result = _fetch_global_macro_news(limit=10)
        assert len(result) == 2
        assert result[0]["title"] == "Fed holds rates steady amid inflation concerns"
        assert result[0]["source"] == "global_macro"


# ---------------------------------------------------------------------------
# 4. collect_ticker_data-Integration (neue Keys)
# ---------------------------------------------------------------------------

_INFO_MINIMAL = {"longName": "Apple Inc.", "currency": "USD"}


def _make_mock_ticker(info: dict) -> MagicMock:
    """Mock-Ticker mit info-Dict + synthetischer Historie (offline)."""
    t = MagicMock()
    t.info = info
    dates = pd.date_range(end="2026-01-01", periods=250)
    hist = pd.DataFrame(
        {
            "Close": [100.0 + i * 0.1 for i in range(250)],
            "Volume": [1_000_000] * 250,
            "Open": [99.0] * 250,
            "High": [101.0] * 250,
            "Low": [98.0] * 250,
        },
        index=dates,
    )
    t.history.return_value = hist
    t.news = []
    return t


class TestCollectTickerDataPhaseAKeys:
    """collect_ticker_data setzt die drei neuen Keys (Fetches gemockt)."""

    @patch("concilium.data._fetch_polymarket", return_value=[])
    @patch("concilium.data._fetch_insider_transactions", return_value=[])
    @patch("concilium.data._fetch_global_macro_news", return_value=[])
    @patch("concilium.data._fetch_reddit", return_value=[])
    @patch("concilium.data._fetch_stocktwits", return_value=[])
    @patch("concilium.data._fetch_google_news", return_value=[])
    @patch("concilium.data._fetch_macro_data", return_value={})
    @patch("concilium.data._save_cache")
    @patch("concilium.data.yf.Ticker")
    def test_new_keys_present(
        self,
        mock_ticker_class,
        mock_save,
        mock_macro,
        _gnews,
        _twits,
        _reddit,
        _gm,
        _insider,
        _pm,
    ):
        mock_ticker_class.return_value = _make_mock_ticker(_INFO_MINIMAL)
        from concilium.data import collect_ticker_data

        result = collect_ticker_data("AAPL")
        assert "insider_transactions" in result
        assert "prediction_markets" in result
        assert "global_macro_news" in result
        assert result["insider_transactions"] == []
        assert result["prediction_markets"] == []
        assert result["global_macro_news"] == []

    @patch("concilium.data._fetch_polymarket", return_value=[{"title": "Fed?", "probability": 0.6, "category": "Macro"}])
    @patch("concilium.data._fetch_insider_transactions", return_value=[{"date": "2026-08-15", "insider": "Cook", "transaction": "Purchase", "shares": 2000.0, "price": 180.5, "value": 361000.0}])
    @patch("concilium.data._fetch_global_macro_news", return_value=[{"title": "Fed holds rates", "published": None, "source": "global_macro"}])
    @patch("concilium.data._fetch_reddit", return_value=[])
    @patch("concilium.data._fetch_stocktwits", return_value=[])
    @patch("concilium.data._fetch_google_news", return_value=[])
    @patch("concilium.data._fetch_macro_data", return_value={})
    @patch("concilium.data._save_cache")
    @patch("concilium.data.yf.Ticker")
    def test_new_keys_with_data(
        self,
        mock_ticker_class,
        mock_save,
        mock_macro,
        _gnews,
        _twits,
        _reddit,
        _gm,
        _insider,
        _pm,
    ):
        from concilium.data import collect_ticker_data

        mock_ticker_class.return_value = _make_mock_ticker(_INFO_MINIMAL)
        result = collect_ticker_data("AAPL")
        assert len(result["insider_transactions"]) == 1
        assert len(result["prediction_markets"]) == 1
        assert len(result["global_macro_news"]) == 1


# ---------------------------------------------------------------------------
# 5. _build_data_text — Rollen-Filterung
# ---------------------------------------------------------------------------

_PHASE_A_DATA = {
    "ticker": "AAPL",
    "fundamentals": {"name": "Apple Inc.", "sector": "Tech", "pe_ratio": 30.0},
    "technicals": {"current_price": 180.0},
    "sentiment": {"positiv": 1, "negativ": 0, "neutral": 1},
    "news": ["Apple headline"],
    "insider_transactions": [
        {"date": "2026-08-15", "insider": "Cook Timothy D", "transaction": "Purchase",
         "shares": 2000.0, "price": 180.5, "value": 361000.0},
    ],
    "prediction_markets": [
        {"title": "Fed rate cut?", "probability": 0.6, "category": "Macro"},
    ],
    "global_macro_news": [
        {"title": "Fed holds rates", "published": None, "source": "global_macro"},
    ],
}


class TestBuildDataTextPhaseA:
    """Die neuen Sektionen erscheinen rollen-abhängig im Daten-Text."""

    def test_fundamental_role_shows_insider(self):
        text = _build_data_text(_PHASE_A_DATA, role="fundamental")
        assert "=== INSIDER-TRANSAKTIONEN ===" in text
        assert "Cook Timothy D" in text
        # Makro/Polymarket/Global-Makro nicht für fundamental (ohne Makro-Daten)
        assert "=== PREDICTION MARKETS ===" not in text
        assert "=== GLOBAL-MAKRO-NEWS ===" not in text

    def test_macro_news_role_shows_markets_and_macro_news(self):
        text = _build_data_text(_PHASE_A_DATA, role="macro_news")
        assert "=== PREDICTION MARKETS ===" in text
        assert "Fed rate cut?" in text
        assert "=== GLOBAL-MAKRO-NEWS ===" in text
        assert "Fed holds rates" in text
        assert "=== INSIDER-TRANSAKTIONEN ===" not in text

    def test_alle_role_shows_all_three(self):
        text = _build_data_text(_PHASE_A_DATA, role="alle")
        assert "=== INSIDER-TRANSAKTIONEN ===" in text
        assert "=== PREDICTION MARKETS ===" in text
        assert "=== GLOBAL-MAKRO-NEWS ===" in text

    def test_sentiment_role_shows_none(self):
        text = _build_data_text(_PHASE_A_DATA, role="sentiment")
        assert "=== INSIDER-TRANSAKTIONEN ===" not in text
        assert "=== PREDICTION MARKETS ===" not in text
        assert "=== GLOBAL-MAKRO-NEWS ===" not in text

    def test_empty_lists_omit_blocks(self):
        """Leere Listen → Blöcke werden weggelassen (kein N/A-Rauschen)."""
        data_empty = {**_PHASE_A_DATA,
                      "insider_transactions": [],
                      "prediction_markets": [],
                      "global_macro_news": []}
        text = _build_data_text(data_empty, role="alle")
        assert "=== INSIDER-TRANSAKTIONEN ===" not in text
        assert "=== PREDICTION MARKETS ===" not in text
        assert "=== GLOBAL-MAKRO-NEWS ===" not in text


# ---------------------------------------------------------------------------
# 6. generate_report — Sektionen nur wenn Daten vorhanden
# ---------------------------------------------------------------------------


def _base_result(extra_data: dict) -> dict:
    """Minimaler result-dict für generate_report (no_llm)."""
    return {
        "ticker": "AAPL",
        "no_llm": True,
        "data": {
            "fundamentals": {"name": "Apple", "sector": "Tech", "industry": "HW", "currency": "USD"},
            "technicals": {},
            "sentiment": {"positiv": 1, "negativ": 0, "neutral": 1, "dominant": "neutral",
                          "sample_size": 2, "weighted": False},
            "news": [],
            "news_with_dates": [],
            "news_source": "yfinance",
            **extra_data,
        },
    }


class TestReportPhaseASections:
    """Report-Sektionen: nur mit Daten, sonst keine Sektion."""

    def test_sections_shown_when_data_present(self):
        result = _base_result({
            "insider_transactions": [
                {"date": "2026-08-15", "insider": "Cook Timothy D", "transaction": "Purchase",
                 "shares": 2000.0, "price": 180.5, "value": 361000.0},
            ],
            "prediction_markets": [
                {"title": "Fed rate cut?", "probability": 0.6, "category": "Macro"},
            ],
            "global_macro_news": [
                {"title": "Fed holds rates", "published": None, "source": "global_macro"},
            ],
        })
        report = generate_report(result)
        assert "## Insider-Transaktionen" in report
        assert "Cook Timothy D" in report
        assert "## Prediction Markets" in report
        assert "Fed rate cut?" in report
        assert "## Global-Makro-News" in report
        assert "Fed holds rates" in report

    def test_sections_omitted_when_empty(self):
        result = _base_result({
            "insider_transactions": [],
            "prediction_markets": [],
            "global_macro_news": [],
        })
        report = generate_report(result)
        assert "## Insider-Transaktionen" not in report
        assert "## Prediction Markets" not in report
        assert "## Global-Makro-News" not in report

    def test_report_robust_without_phase_a_keys(self):
        """Alte Results ohne die neuen Keys → kein Crash, keine Sektion."""
        result = _base_result({})
        report = generate_report(result)
        assert "## Insider-Transaktionen" not in report
        assert "## Prediction Markets" not in report
        assert "## Global-Makro-News" not in report
