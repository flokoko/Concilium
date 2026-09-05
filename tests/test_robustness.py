"""Tests für die drei Robustheits-/Effizienz-Verbesserungen:

1. Retry-Jitter in llm.py (_jittered_backoff)
2. Tages-Cache für Marktdaten in data.py (_load_cache / _save_cache)
3. Journal-CSV-Lock in journal.py (fcntl.flock)

Alle Tests sind offline (kein yfinance/Netzwerk nötig).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.data import (  # noqa: E402
    _cache_file_path,
    _cache_json_object_hook,
    _get_cache_dir,
    _get_today_key,
    _load_cache,
    _save_cache,
    collect_ticker_data,
)
from concilium.journal import append_decision  # noqa: E402
from concilium.llm import RETRY_BACKOFF, LLMClient, _jittered_backoff  # noqa: E402

# ---------------------------------------------------------------------------
# Aufgabe 1: Retry-Jitter
# ---------------------------------------------------------------------------


class TestJitteredBackoff:
    """Tests für _jittered_backoff — zufälliger Jitter im Backoff."""

    def test_backoff_in_jitter_range(self):
        """Backoff-Wert muss im Bereich ±30% des Basiswerts liegen."""
        for attempt in range(5):
            for _ in range(100):  # mehrfach prüfen wegen Zufälligkeit
                value = _jittered_backoff(attempt)
                base = RETRY_BACKOFF * (attempt + 1)
                low = base * 0.7
                high = base * 1.3
                assert low <= value <= high, (
                    f"attempt={attempt}: {value} nicht in [{low}, {high}]"
                )

    def test_backoff_is_not_constant(self):
        """Zwei Aufrufe mit gleichem attempt liefern unterschiedliche Werte (mit hoher Wahrscheinlichkeit)."""
        values = {_jittered_backoff(0) for _ in range(20)}
        assert len(values) > 1, "Jitter sollte verschiedene Werte produzieren"

    def test_backoff_increases_with_attempt(self):
        """Höhere attempt-Werte liefern im Mittel höhere Backoff-Werte."""
        avg_0 = sum(_jittered_backoff(0) for _ in range(100)) / 100
        avg_1 = sum(_jittered_backoff(1) for _ in range(100)) / 100
        avg_2 = sum(_jittered_backoff(2) for _ in range(100)) / 100
        assert avg_0 < avg_1 < avg_2

    def test_retry_429_uses_jittered_sleep(self):
        """Bei HTTP 429 wird time.sleep mit einem Jitter-Wert im erwarteten Bereich aufgerufen."""
        client = LLMClient(base_url="http://fake:8080/v1", api_key="test-key", model="test-model")

        # _MockResponse inline (ähnlich test_llm.py)
        class _MockResponse:
            def __init__(self, status_code, content=""):
                self.status_code = status_code
                self.text = content
                self._content = content

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise Exception(f"HTTP {self.status_code}")  # noqa: TRY002

            def json(self):
                return {"choices": [{"message": {"content": self._content}}]}

        responses = [
            _MockResponse(429, "Rate limited"),
            _MockResponse(200, "OK nach 429"),
        ]

        sleep_calls: list[float] = []

        def fake_sleep(seconds):
            sleep_calls.append(seconds)

        with patch("concilium.llm.requests.post", side_effect=responses):
            with patch("concilium.llm.time.sleep", side_effect=fake_sleep):
                result = client.chat([{"role": "user", "content": "Test"}])

        assert result == "OK nach 429"
        assert len(sleep_calls) == 1
        sleep_val = sleep_calls[0]
        # attempt=0 → base = RETRY_BACKOFF * 1 = 2, Jitter-Bereich [1.4, 2.6]
        expected_low = RETRY_BACKOFF * 1 * 0.7
        expected_high = RETRY_BACKOFF * 1 * 1.3
        assert expected_low <= sleep_val <= expected_high, (
            f"sleep({sleep_val}) nicht in [{expected_low}, {expected_high}]"
        )


# ---------------------------------------------------------------------------
# Aufgabe 2: Tages-Cache
# ---------------------------------------------------------------------------


class TestCacheConfig:
    """Tests für Cache-Konfiguration (_get_cache_dir, _get_today_key)."""

    def test_cache_disabled_via_empty_env(self, monkeypatch):
        """CONCILIUM_CACHE_DIR="" → Cache deaktiviert (None)."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", "")
        assert _get_cache_dir() is None

    def test_cache_dir_from_env(self, monkeypatch, tmp_path):
        """CONCILIUM_CACHE_DIR setzt das Cache-Verzeichnis."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", str(tmp_path))
        assert _get_cache_dir() == str(tmp_path)

    def test_cache_dir_default(self, monkeypatch):
        """Ohne Env-Variable → Default <repo>/cache/."""
        monkeypatch.delenv("CONCILIUM_CACHE_DIR", raising=False)
        result = _get_cache_dir()
        assert result is not None
        assert result.endswith("cache")

    def test_today_key_format(self):
        """_get_today_key liefert YYYY-MM-DD."""
        key = _get_today_key()
        assert len(key) == 10
        assert key[4] == "-" and key[7] == "-"

    def test_cache_file_path(self):
        """_cache_file_path generiert korrekten Dateinamen."""
        path = _cache_file_path("/tmp/cache", "2026-08-21", "AAPL")
        assert path == "/tmp/cache/market_2026-08-21_AAPL.json"

    def test_cache_file_path_sanitizes_ticker(self):
        """Ticker mit Sonderzeichen wird sicher gemacht."""
        path = _cache_file_path("/tmp/cache", "2026-08-21", "A/B/C")
        assert "/" not in path.split("/")[-1]  # kein / im Dateinamen
        assert path.endswith(".json")


class TestCacheLoadSave:
    """Tests für _load_cache / _save_cache."""

    def test_save_and_load_roundtrip(self, monkeypatch, tmp_path):
        """Speichern und Laden liefert dieselben Daten zurück."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", str(tmp_path))

        today = _get_today_key()
        data = {
            "ticker": "TEST.DE",
            "fundamentals": {"market_cap": 1e12, "pe_ratio": 15.5},
            "technicals": {"current_price": 100.0},
            "history": [{"date": "2026-08-20", "close": 100.0}],
            "sentiment": {"positiv": 3, "negativ": 1, "neutral": 2},
            "news": ["Headline 1", "Headline 2"],
            "macro": {"us_10y_yield": 4.5},
            "peers": [],
            "data_warnings": [],
            "isin": "DE0001234567",  # wird nicht gecacht
            "wkn": "123456",  # wird nicht gecacht
        }

        _save_cache("TEST.DE", data, today_key=today)

        loaded = _load_cache("TEST.DE", today_key=today)
        assert loaded is not None
        assert loaded["ticker"] == "TEST.DE"
        assert loaded["fundamentals"]["market_cap"] == 1e12
        assert loaded["fundamentals"]["pe_ratio"] == 15.5
        # isin/wkn dürfen NICHT im Cache stehen
        assert "isin" not in loaded
        assert "wkn" not in loaded

    def test_load_cache_miss_no_file(self, monkeypatch, tmp_path):
        """Bei nicht existierender Datei → None."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", str(tmp_path))
        assert _load_cache("NOTEXIST", today_key="2026-08-21") is None

    def test_load_cache_disabled(self, monkeypatch):
        """Cache deaktiviert → immer None."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", "")
        assert _load_cache("AAPL", today_key="2026-08-21") is None

    def test_load_cache_wrong_date(self, monkeypatch, tmp_path):
        """Cache mit altem Datum → None (ignorieren)."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", str(tmp_path))

        # Cache von gestern speichern
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        data = {"ticker": "AAPL", "fundamentals": {}}
        _save_cache("AAPL", data, today_key=yesterday)

        # Heute laden → None (falsches Datum)
        today = _get_today_key()
        assert _load_cache("AAPL", today_key=today) is None

    def test_save_cache_does_not_crash_on_error(self, monkeypatch):
        """Cache-Schreiben crasht nie, auch nicht bei ungültigem Pfad."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", "/nonexistent/path/that/does/not/exist")
        # Sollte nicht crashen
        _save_cache("AAPL", {"ticker": "AAPL"}, today_key="2026-08-21")

    def test_load_cache_does_not_crash_on_corrupt_file(self, monkeypatch, tmp_path):
        """Korrupte JSON-Datei → None, kein Crash."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", str(tmp_path))

        # Korrupte Datei schreiben
        path = _cache_file_path(str(tmp_path), "2026-08-21", "AAPL")
        with open(path, "w") as fh:
            fh.write("{invalid json!!!")

        assert _load_cache("AAPL", today_key="2026-08-21") is None


class TestCollectTickerDataCache:
    """Tests dass collect_ticker_data den Cache korrekt nutzt."""

    def test_two_calls_same_day_hit_yfinance_once(self, monkeypatch, tmp_path):
        """Zwei Aufrufe von collect_ticker_data am selben Tag → yfinance nur EINMAL aufgerufen."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", str(tmp_path))

        # yf.Ticker mocken
        mock_ticker_obj = MagicMock()
        # history: DataFrame-mock mit genug Daten
        import pandas as pd

        hist_df = pd.DataFrame(
            {
                "Open": [100.0] * 250,
                "High": [101.0] * 250,
                "Low": [99.0] * 250,
                "Close": [100.5] * 250,
                "Volume": [1000000] * 250,
            },
            index=pd.date_range("2025-01-01", periods=250, freq="B"),
        )
        mock_ticker_obj.history = MagicMock(return_value=hist_df)
        mock_ticker_obj.info = {
            "marketCap": 3e12,
            "trailingPE": 30.0,
            "trailingEps": 5.0,
            "totalRevenue": 4e11,
            "revenueGrowth": 0.1,
            "profitMargins": 0.25,
            "longName": "Test Corp",
            "currency": "USD",
            "sector": "Technology",
            "industry": "Software",
        }
        mock_ticker_obj.news = []

        call_count = [0]

        def mock_ticker_factory(ticker_symbol):
            call_count[0] += 1
            return mock_ticker_obj

        with patch("concilium.data.yf.Ticker", side_effect=mock_ticker_factory):
            with patch("concilium.data._fetch_macro_data", return_value={
                "us_10y_yield": 4.5,
                "us_10y_yield_1m_ago": 4.3,
                "us_10y_trend": "steigend",
                "sp500_pe": 22.0,
                "sp500_market_cap": None,
                "sp500_source": "none",
            }):
                with patch("concilium.data._fetch_google_news", return_value=[]):
                    # Phase-A-Fetches mocken: Insider nutzt eigenen yf.Ticker-Aufruf
                    # (soll hier nicht in die yfinance-Zählung eingehen — der
                    # Test prüft die Cache-Semantik für history/info).
                    with patch(
                        "concilium.data._fetch_insider_transactions", return_value=[]
                    ), patch(
                        "concilium.data._fetch_polymarket", return_value=[]
                    ), patch(
                        "concilium.data._fetch_global_macro_news", return_value=[]
                    ):
                        # Erster Aufruf → yfinance wird geladen
                        data1 = collect_ticker_data("TEST")
                        assert call_count[0] == 1

                        # Zweiter Aufruf → Cache-Treffer, yfinance NICHT erneut aufgerufen
                        data2 = collect_ticker_data("TEST")
                        assert call_count[0] == 1, "yfinance sollte beim 2. Aufruf nicht erneut aufgerufen werden"

        # Beide liefern dieselben Daten
        assert data1["ticker"] == data2["ticker"]
        assert data1["fundamentals"]["market_cap"] == data2["fundamentals"]["market_cap"]

    def test_different_date_reloads(self, monkeypatch, tmp_path):
        """Bei unterschiedlichem Datum (Cache ungültig) → neu laden."""
        # Cache von gestern existiert
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        old_data = {
            "ticker": "TEST",
            "fundamentals": {"market_cap": 1e10, "pe_ratio": 5.0},
            "technicals": {"current_price": 50.0},
            "history": [],
            "sentiment": {"positiv": 0, "negativ": 0, "neutral": 0},
            "news": [],
            "news_with_dates": [],
            "news_source": "none",
            "macro": {},
            "peers": [],
            "data_warnings": [],
        }
        # Cache von gestern speichern
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", str(tmp_path))
        _save_cache("TEST", old_data, today_key=yesterday)

        # Heutigen Cache-Laden mocken → sollte None liefern (falsches Datum)
        today = _get_today_key()
        loaded = _load_cache("TEST", today_key=today)
        assert loaded is None, "Cache mit gestrigem Datum sollte None liefern"

    def test_cache_disabled_no_persistence(self, monkeypatch, tmp_path):
        """Cache deaktiviert → keine Dateien, kein Caching."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", "")

        import pandas as pd

        mock_ticker_obj = MagicMock()
        hist_df = pd.DataFrame(
            {"Open": [100.0] * 250, "High": [101.0] * 250, "Low": [99.0] * 250,
             "Close": [100.5] * 250, "Volume": [1000000] * 250},
            index=pd.date_range("2025-01-01", periods=250, freq="B"),
        )
        mock_ticker_obj.history = MagicMock(return_value=hist_df)
        mock_ticker_obj.info = {"longName": "Test", "currency": "USD", "sector": "X", "industry": "Y"}
        mock_ticker_obj.news = []

        with patch("concilium.data.yf.Ticker", return_value=mock_ticker_obj):
            with patch("concilium.data._fetch_macro_data", return_value={}):
                with patch("concilium.data._fetch_google_news", return_value=[]):
                    collect_ticker_data("TEST")

        # Keine Cache-Dateien sollten existieren
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 0


class TestCacheJsonObjectHook:
    """Tests für _cache_json_object_hook (datetime-Deserialisierung)."""

    def test_datetime_string_revived(self):
        """ISO-8601 datetime-String wird zu datetime-Objekt."""
        obj = {"published": "2026-08-20T08:30:00+00:00", "title": "Test"}
        result = _cache_json_object_hook(obj)
        assert isinstance(result["published"], datetime)
        assert result["published"].year == 2026

    def test_non_datetime_string_untouched(self):
        """Normale Strings werden nicht angetastet."""
        obj = {"name": "Test Corp", "ticker": "AAPL"}
        result = _cache_json_object_hook(obj)
        assert result["name"] == "Test Corp"
        assert result["ticker"] == "AAPL"

    def test_datetime_with_z_suffix(self):
        """ISO-8601 mit Z-Suffix wird korrekt geparst."""
        obj = {"date": "2026-08-20T08:30:00Z"}
        result = _cache_json_object_hook(obj)
        assert isinstance(result["date"], datetime)
        assert result["date"].tzinfo is not None


# ---------------------------------------------------------------------------
# Aufgabe 3: Journal-Lock
# ---------------------------------------------------------------------------


class TestJournalLock:
    """Tests für Journal-CSV-Lock (fcntl.flock)."""

    def test_append_decision_smoke_single(self, tmp_path):
        """Ein einzelner append_decision funktioniert (Lock wird geholt/freigegeben)."""
        mock_result = {
            "ticker": "LOCK.DE",
            "trade": {"aktion": "KAUFEN", "zielkurs": 100.0},
            "final": {"entscheidung": "GENEHMIGT", "confidence": 4},
        }
        journal_file = str(tmp_path / "journal" / "decisions.csv")
        append_decision(mock_result, journal_file=journal_file)

        import csv as csv_mod

        with open(journal_file, encoding="utf-8") as fh:
            reader = csv_mod.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "LOCK.DE"

    def test_append_decision_smoke_double(self, tmp_path):
        """Zwei appends hintereinander funktionieren (Lock wird korrekt freigegeben)."""
        journal_file = str(tmp_path / "journal_lock" / "decisions.csv")

        for i in range(2):
            mock_result = {
                "ticker": f"LOCK{i}.DE",
                "trade": {"aktion": "KAUFEN"},
                "final": {"entscheidung": "GENEHMIGT", "confidence": 3},
            }
            append_decision(mock_result, journal_file=journal_file)

        import csv as csv_mod

        with open(journal_file, encoding="utf-8") as fh:
            reader = csv_mod.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["ticker"] == "LOCK0.DE"
        assert rows[1]["ticker"] == "LOCK1.DE"

    def test_append_decision_smoke_triple(self, tmp_path):
        """Drei appends hintereinander — Smoke-Test für Lock-Recycling."""
        journal_file = str(tmp_path / "journal_triple" / "decisions.csv")

        for i in range(3):
            mock_result = {
                "ticker": f"T{i}.DE",
                "trade": {"aktion": "VERKAUFEN"},
                "final": {"entscheidung": "ABGELEHNT", "confidence": 2},
            }
            append_decision(mock_result, journal_file=journal_file)

        import csv as csv_mod

        with open(journal_file, encoding="utf-8") as fh:
            reader = csv_mod.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 3

    def test_fcntl_imported_or_none(self):
        """fcntl ist entweder importiert (Linux) oder None (andere Plattformen)."""
        from concilium import journal

        assert hasattr(journal, "fcntl")
        assert journal.fcntl is None or hasattr(journal.fcntl, "flock")
