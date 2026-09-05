"""Tests für NaN-Robustheit der Kurs-/Indikator-Berechnung.

Hintergrund: yfinance liefert intermittierend NaN für den letzten Close.
Vor dem Fix ergab close.iloc[-1] → NaN → current_price=None, sma50/sma200=nan,
rsi14=None → keine fundierte Kaufentscheidung möglich (alle Ticker HALTEN).
Diese Tests pinpen das Verhalten "letzter gültiger (nicht-NaN) Wert".

Prüft:
  - _last_valid: normal, trailing-NaN, leer, all-NaN, None, Skalar
  - collect_ticker_data: letzter Close NaN → current_price/sma50/sma200/
    current_volume/avg_volume_30d aus letzten gültigen Werten
  - _fetch_macro_data: ^TNX mit NaN-Lastzeile → us_10y_yield aus gültigem Close
  - _compute_rsi: trailing-NaN → RSI aus letztem gültigen Fenster
  - _compute_bollinger: last_close NaN → position aus letztem gültigen Close
"""

from __future__ import annotations

import math
import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from concilium.data import (  # noqa: E402
    _compute_bollinger,
    _compute_rsi,
    _fetch_macro_data,
    _last_valid,
    collect_ticker_data,
)

# ---------------------------------------------------------------------------
# _last_valid — Unit-Tests
# ---------------------------------------------------------------------------


class TestLastValid:
    """_last_valid liefert den letzten gültigen (nicht-NaN) Wert einer Serie."""

    def test_normal_series_returns_last_value(self):
        """Normale Serie ohne NaN → letzter Wert."""
        s = pd.Series([1.0, 2.0, 3.0])
        assert _last_valid(s) == 3.0

    def test_trailing_nan_returns_last_valid(self):
        """Letzter Wert NaN → letzter gültiger Wert davor."""
        s = pd.Series([1.0, 2.0, float("nan")])
        assert _last_valid(s) == 2.0

    def test_nan_in_middle_returns_last_value(self):
        """NaN mitten in der Serie, letzter Wert gültig → letzter Wert."""
        s = pd.Series([1.0, float("nan"), 3.0])
        assert _last_valid(s) == 3.0

    def test_empty_series_returns_none(self):
        """Leere Serie → None (kein Crash)."""
        assert _last_valid(pd.Series([], dtype=float)) is None

    def test_none_returns_none(self):
        """None als Eingabe → None (kein Crash)."""
        assert _last_valid(None) is None

    def test_all_nan_returns_last_without_crash(self):
        """Alle Werte NaN → letzter (NaN) Wert, kein Crash; _safe_float macht None."""
        s = pd.Series([float("nan"), float("nan")])
        result = _last_valid(s)
        assert result is not None  # kein Crash — Rückgabe ist der letzte Wert
        assert math.isnan(result)

    def test_scalar_value_returns_value(self):
        """Skalar (z. B. volume.tail(30).mean()) → Wert direkt zurück."""
        assert _last_valid(42.5) == 42.5

    def test_scalar_nan_returns_none(self):
        """Skalar NaN → None (konsistent mit _safe_float)."""
        assert _last_valid(float("nan")) is None


# ---------------------------------------------------------------------------
# collect_ticker_data — gemockte Historie mit NaN als letztem Close
# ---------------------------------------------------------------------------

_INFO = {
    "marketCap": 1_000_000_000,
    "trailingPE": 20.0,
    "trailingEps": 5.0,
    "totalRevenue": 100_000_000_000,
    "revenueGrowth": 0.05,
    "profitMargins": 0.2,
    "fiftyTwoWeekHigh": 130.0,
    "fiftyTwoWeekLow": 90.0,
    "trailingAnnualDividendYield": 0.005,
    "beta": 1.0,
    "currency": "USD",
    "longName": "Test AG",
    "currentPrice": 124.8,
}


def _make_hist(last_close_nan: bool, n: int = 250) -> pd.DataFrame:
    """250 Tage Historie; optional letzter Close = NaN (yfinance-Fehlerfall)."""
    dates = pd.date_range(end="2026-01-01", periods=n, tz="UTC")
    closes = [100.0 + i * 0.1 for i in range(n)]
    if last_close_nan:
        closes[-1] = float("nan")
    return pd.DataFrame(
        {
            "Close": closes,
            "Volume": [1_000_000.0] * n,
            "Open": [99.0] * n,
            "High": [101.0] * n,
            "Low": [98.0] * n,
        },
        index=dates,
    )


def _make_ticker(hist: pd.DataFrame) -> MagicMock:
    """Mock-Ticker, der collect_ticker_data akzeptiert (Offline-Test)."""
    t = MagicMock()
    t.info = dict(_INFO)
    t.history.return_value = hist
    t.news = None
    return t


@patch("concilium.data._save_cache")
@patch("concilium.data._fetch_reddit", return_value=[])
@patch("concilium.data._fetch_stocktwits", return_value=[])
@patch("concilium.data._fetch_google_news", return_value=[])
@patch("concilium.data._fetch_macro_data", return_value={})
class TestCollectTickerDataNaNLastClose:
    """Letzter Close NaN → technische Werte aus dem letzten gültigen Close."""

    def test_current_price_is_last_valid_close(
        self, _macro, _google, _stw, _reddit, _cache
    ):
        """current_price ist der letzte gültige Close, nicht None."""
        hist = _make_hist(last_close_nan=True)
        with patch("concilium.data.yf.Ticker", return_value=_make_ticker(hist)):
            data = collect_ticker_data("TEST")
        t = data["technicals"]
        # letzter gültiger Close: 100.0 + 248 * 0.1 = 124.8
        assert t["current_price"] == pytest.approx(124.8)

    def test_sma200_is_valid_float_not_nan(
        self, _macro, _google, _stw, _reddit, _cache
    ):
        """sma200 ist ein gültiger float (nicht nan) trotz NaN-Lastzeile."""
        hist = _make_hist(last_close_nan=True)
        with patch("concilium.data.yf.Ticker", return_value=_make_ticker(hist)):
            data = collect_ticker_data("TEST")
        sma200 = data["technicals"]["sma200"]
        assert sma200 is not None
        assert not math.isnan(sma200)
        # identisch zum SMA200 der Serie ohne den NaN-Lastwert
        expected = float(hist["Close"].dropna().rolling(window=200).mean().iloc[-1])
        assert sma200 == pytest.approx(expected)

    def test_sma50_is_valid_float_not_nan(
        self, _macro, _google, _stw, _reddit, _cache
    ):
        """sma50 ist ein gültiger float (nicht nan) trotz NaN-Lastzeile."""
        hist = _make_hist(last_close_nan=True)
        with patch("concilium.data.yf.Ticker", return_value=_make_ticker(hist)):
            data = collect_ticker_data("TEST")
        sma50 = data["technicals"]["sma50"]
        assert sma50 is not None
        assert not math.isnan(sma50)

    def test_volume_values_from_valid_rows(
        self, _macro, _google, _stw, _reddit, _cache
    ):
        """current_volume/avg_volume_30d bleiben gültig (Fallback über _last_valid)."""
        hist = _make_hist(last_close_nan=True)
        with patch("concilium.data.yf.Ticker", return_value=_make_ticker(hist)):
            data = collect_ticker_data("TEST")
        t = data["technicals"]
        assert t["current_volume"] == pytest.approx(1_000_000.0)
        assert t["avg_volume_30d"] == pytest.approx(1_000_000.0)

    def test_normal_history_unchanged(self, _macro, _google, _stw, _reddit, _cache):
        """Ohne NaN bleibt das Verhalten identisch (kein Regression)."""
        hist = _make_hist(last_close_nan=False)
        with patch("concilium.data.yf.Ticker", return_value=_make_ticker(hist)):
            data = collect_ticker_data("TEST")
        t = data["technicals"]
        assert t["current_price"] == pytest.approx(124.9)  # 100 + 249 * 0.1
        assert not math.isnan(t["sma200"])
        assert not math.isnan(t["sma50"])


# ---------------------------------------------------------------------------
# _fetch_macro_data — ^TNX mit NaN als letztem Close
# ---------------------------------------------------------------------------


class TestMacroDataTrailingNaN:
    """_fetch_macro_data: NaN-Lastzeile in ^TNX-Historie → Yield aus gültigem Close."""

    def test_tnx_yield_from_last_valid_close(self):
        n = 20
        dates = pd.date_range(end="2026-01-01", periods=n, tz="UTC")
        closes = [4.0 + i * 0.01 for i in range(n)]
        closes[-1] = float("nan")
        tnx_hist = pd.DataFrame(
            {"Close": closes, "Volume": [1_000_000.0] * n}, index=dates
        )
        tnx = MagicMock()
        tnx.history.return_value = tnx_hist
        other = MagicMock()
        other.info = {}
        other.history.return_value = pd.DataFrame()

        def _ticker_factory(symbol):
            return tnx if symbol == "^TNX" else other

        with patch("concilium.data.yf.Ticker", side_effect=_ticker_factory):
            macro = _fetch_macro_data()

        # letzter gültiger Close: 4.0 + 18 * 0.01 = 4.18
        assert macro["us_10y_yield"] == pytest.approx(4.18)
        # Trend ist berechnet (beide Yields gültig), nicht None
        assert macro["us_10y_trend"] in {"flach", "steigend", "fallend"}


# ---------------------------------------------------------------------------
# _compute_rsi — trailing NaN
# ---------------------------------------------------------------------------


class TestComputeRsiTrailingNaN:
    """RSI nutzt das letzte gültige Fenster, wenn der letzte Wert NaN ist."""

    def test_rsi_uses_last_valid_window(self):
        closes = [100.0 + i * 0.5 for i in range(30)]
        s = pd.Series(closes)
        s.iloc[-1] = float("nan")
        rsi = _compute_rsi(s)
        expected = _compute_rsi(pd.Series(closes[:-1]))
        assert rsi is not None
        assert rsi == pytest.approx(expected)

    def test_rsi_normal_series_unchanged(self):
        """Ohne NaN unverändertes Verhalten."""
        closes = [100.0 + i * 0.5 for i in range(30)]
        rsi = _compute_rsi(pd.Series(closes))
        assert rsi is not None
        assert 0.0 <= rsi <= 100.0


# ---------------------------------------------------------------------------
# _compute_bollinger — last_close NaN
# ---------------------------------------------------------------------------


class TestComputeBollingerLastCloseNaN:
    """Bollinger-Position nutzt den letzten gültigen Close, wenn dieser NaN ist."""

    def test_position_uses_last_valid_close(self):
        closes = [100.0 + i for i in range(30)]
        s = pd.Series(closes)
        s.iloc[-1] = float("nan")
        bb = _compute_bollinger(s)
        expected = _compute_bollinger(pd.Series(closes[:-1]))
        assert bb["position"] is not None
        assert not math.isnan(bb["position"])
        assert bb["position"] == pytest.approx(expected["position"])
        assert bb["middle"] == pytest.approx(expected["middle"])
