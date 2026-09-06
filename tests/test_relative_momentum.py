"""Tests für Relatives Momentum (Phase 2, cross-sectional vs. S&P 500).

Diese Tests benötigen KEIN Netzwerk — alle yfinance-Aufrufe werden gemockt.
Geprüft wird:
  1. _compute_momentum_6m: korrekte 6M-Rendite aus einer gemockten Close-Serie
     (inkl. NaN-Robustheit und Zu-wenig-Daten-Fall).
  2. collect_ticker_data-Integration: relatives_momentum_6m = momentum_6m -
     sp500_momentum_6m wird korrekt berechnet; ist None, wenn die Benchmark
     fehlt.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.data import (  # noqa: E402
    _compute_momentum_6m,
    _get_sp500_momentum,
    collect_ticker_data,
)


def _make_close(start: float = 100.0, daily_pct: float = 0.0, n: int = 300) -> pd.Series:
    """Deterministische Close-Serie (exponentielles Wachstum, kein NaN)."""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    values = [start * (1.0 + daily_pct) ** i for i in range(n)]
    return pd.Series(values, index=dates, name="Close")


class TestComputeMomentum6m:
    """Tests für die 6M-Momentum-Berechnung aus einer Close-Serie."""

    def test_calculates_correct_6m_return(self):
        """300 Tage +1%/Tag: momentum_6m == (close[-1]/close[-126] - 1)*100."""
        close = _make_close(start=100.0, daily_pct=0.01, n=300)
        expected = (close.iloc[-1] / close.iloc[-126] - 1.0) * 100.0
        result = _compute_momentum_6m(close)
        assert result == pytest.approx(expected)

    def test_flat_series_gives_zero(self):
        """Konstante Serie → Momentum 0.0."""
        close = _make_close(start=100.0, daily_pct=0.0, n=300)
        assert _compute_momentum_6m(close) == pytest.approx(0.0)

    def test_too_few_data_points_returns_none(self):
        """125 Datenpunkte (< ~126) → None."""
        close = _make_close(n=125)
        assert _compute_momentum_6m(close) is None

    def test_exactly_126_points_returns_value(self):
        """Genau 126 Datenpunkte → berechenbar (untere Grenze)."""
        close = _make_close(n=126)
        expected = (close.iloc[-1] / close.iloc[0] - 1.0) * 100.0
        assert _compute_momentum_6m(close) == pytest.approx(expected)

    def test_nan_robust_last_value(self):
        """NaN am Ende der Serie → letzter gültiger Wert wird verwendet."""
        close = _make_close(start=100.0, daily_pct=0.01, n=300)
        close.iloc[-1] = float("nan")
        expected = (close.iloc[-2] / close.iloc[-126] - 1.0) * 100.0
        assert _compute_momentum_6m(close) == pytest.approx(expected)

    def test_none_series_returns_none(self):
        """None-Serie → None (nie crashen)."""
        assert _compute_momentum_6m(None) is None

    def test_zero_base_returns_none(self):
        """Basispreis 0 → None (Division unmöglich)."""
        close = _make_close(n=300)
        close.iloc[-126] = 0.0
        assert _compute_momentum_6m(close) is None


class TestCollectTickerDataMomentum:
    """Integration: collect_ticker_data füllt die drei Momentum-Felder."""

    def _make_hist(self, close: pd.Series) -> pd.DataFrame:
        """OHLCV-DataFrame-Rahmen mit der gegebenen Close-Serie."""
        n = len(close)
        return pd.DataFrame(
            {
                "Open": close.values,
                "High": close.values,
                "Low": close.values,
                "Close": close,
                "Volume": [1_000_000.0] * n,
            },
            index=close.index,
        )

    def _run_collect(self, close: pd.Series, sp500_momentum: float | None) -> dict:
        """Führt collect_ticker_data mit gemocktem Ticker + Benchmark aus.

        sp500_momentum=None simuliert eine nicht verfügbare Benchmark
        (_get_sp500_momentum wirft dann). Cache deaktiviert
        (CONCILIUM_CACHE_DIR="") — kein Zustand zwischen Tests.
        """
        t = MagicMock()
        t.history.return_value = self._make_hist(close)
        t.info = {"marketCap": 1e12, "trailingPE": 20.0, "currentPrice": float(close.iloc[-1])}
        t.news = []

        def fake_sp500_momentum(as_of=None):
            if sp500_momentum is None:
                raise RuntimeError("Benchmark nicht verfügbar")
            return sp500_momentum

        with patch("concilium.data.yf.Ticker", return_value=t), \
             patch("concilium.data._get_sp500_momentum", side_effect=fake_sp500_momentum), \
             patch("concilium.data._fetch_macro_data", return_value={
                 "eurusd": None, "sp500_pe": None, "sp500_market_cap": None,
                 "sp500_source": "none", "us_10y_yield": None,
             }), \
             patch("concilium.data._fetch_peer_data", return_value=[]), \
             patch("concilium.data._fetch_google_news", return_value=[]), \
             patch("concilium.data._fetch_stocktwits", return_value=[]), \
             patch("concilium.data._fetch_reddit", return_value=[]), \
             patch("concilium.data._fetch_insider_transactions", return_value=[]), \
             patch("concilium.data._fetch_polymarket", return_value=[]), \
             patch("concilium.data._fetch_global_macro_news", return_value=[]), \
             patch.dict(os.environ, {"CONCILIUM_CACHE_DIR": ""}):
            return collect_ticker_data("TEST")

    def test_relatives_momentum_is_difference(self):
        """relatives_momentum_6m == momentum_6m - sp500_momentum_6m."""
        close = _make_close(start=100.0, daily_pct=0.01, n=300)
        result = self._run_collect(close, sp500_momentum=5.0)

        tech = result["technicals"]
        expected_ticker = (close.iloc[-1] / close.iloc[-126] - 1.0) * 100.0
        assert tech["momentum_6m"] == pytest.approx(expected_ticker)
        assert tech["sp500_momentum_6m"] == pytest.approx(5.0)
        assert tech["relatives_momentum_6m"] == pytest.approx(expected_ticker - 5.0)

    def test_relatives_momentum_none_when_benchmark_missing(self):
        """Benchmark nicht verfügbar → sp500/relatives None, momentum_6m bleibt."""
        close = _make_close(start=100.0, daily_pct=0.01, n=300)
        result = self._run_collect(close, sp500_momentum=None)

        tech = result["technicals"]
        assert tech["momentum_6m"] is not None
        assert tech["sp500_momentum_6m"] is None
        assert tech["relatives_momentum_6m"] is None

    def test_momentum_absent_for_short_history(self):
        """Zu kurze Historie (<126 Punkte) → momentum_6m None, Benchmark ungefetcht."""
        close = _make_close(n=100)
        t = MagicMock()
        t.history.return_value = self._make_hist(close)
        t.info = {"marketCap": 1e12}
        t.news = []

        with patch("concilium.data.yf.Ticker", return_value=t), \
             patch("concilium.data._get_sp500_momentum") as mom_mock, \
             patch("concilium.data._fetch_macro_data", return_value={
                 "eurusd": None, "sp500_pe": None, "sp500_market_cap": None,
                 "sp500_source": "none", "us_10y_yield": None,
             }), \
             patch("concilium.data._fetch_peer_data", return_value=[]), \
             patch("concilium.data._fetch_google_news", return_value=[]), \
             patch("concilium.data._fetch_stocktwits", return_value=[]), \
             patch("concilium.data._fetch_reddit", return_value=[]), \
             patch("concilium.data._fetch_insider_transactions", return_value=[]), \
             patch("concilium.data._fetch_polymarket", return_value=[]), \
             patch("concilium.data._fetch_global_macro_news", return_value=[]), \
             patch.dict(os.environ, {"CONCILIUM_CACHE_DIR": ""}):
            result = collect_ticker_data("TEST")

        tech = result["technicals"]
        assert tech["momentum_6m"] is None
        assert tech["sp500_momentum_6m"] is None
        assert tech["relatives_momentum_6m"] is None
        mom_mock.assert_not_called()


class TestGetSp500Momentum:
    """Tests für _get_sp500_momentum — Fallback + Tages-Cache (gemockt)."""

    def _gspc_hist(self, daily_pct: float = 0.005, n: int = 250) -> pd.DataFrame:
        close = _make_close(start=5000.0, daily_pct=daily_pct, n=n)
        return pd.DataFrame({"Close": close}, index=close.index)

    def test_gspc_momentum_cached_second_call_hits_network_once(self, tmp_path):
        """Zweiter Aufruf kommt aus dem Tages-Cache — yf.Ticker nur 1x aufgerufen."""
        gspc = MagicMock()
        gspc.history.return_value = self._gspc_hist(daily_pct=0.005, n=250)

        with patch("concilium.data.yf.Ticker", return_value=gspc) as ticker_mock, \
             patch.dict(os.environ, {"CONCILIUM_CACHE_DIR": str(tmp_path)}):
            first = _get_sp500_momentum()
            second = _get_sp500_momentum()

        assert first is not None
        assert second == first
        assert ticker_mock.call_count == 1

    def test_gspc_empty_falls_back_to_spy(self):
        """Leere ^GSPC-Historie → Fallback auf SPY."""
        gspc = MagicMock()
        gspc.history.return_value = pd.DataFrame()
        spy = MagicMock()
        spy.history.return_value = self._gspc_hist(daily_pct=0.003, n=250)

        with patch("concilium.data.yf.Ticker", side_effect=[gspc, spy]), \
             patch.dict(os.environ, {"CONCILIUM_CACHE_DIR": ""}):
            result = _get_sp500_momentum()

        expected = _compute_momentum_6m(self._gspc_hist(daily_pct=0.003, n=250)["Close"])
        assert result == pytest.approx(expected)

    def test_both_empty_returns_none(self):
        """Beide Quellen ohne brauchbare Historie → None, nie crashen."""
        gspc = MagicMock()
        gspc.history.return_value = pd.DataFrame()
        spy = MagicMock()
        spy.history.return_value = None

        with patch("concilium.data.yf.Ticker", side_effect=[gspc, spy]), \
             patch.dict(os.environ, {"CONCILIUM_CACHE_DIR": ""}):
            assert _get_sp500_momentum() is None

    def test_network_error_returns_none(self):
        """yf.Ticker wirft Exception → None (best effort)."""
        with patch("concilium.data.yf.Ticker", side_effect=Exception("Network down")), \
             patch.dict(os.environ, {"CONCILIUM_CACHE_DIR": ""}):
            assert _get_sp500_momentum() is None


class TestGetSp500MomentumAsOf:
    """Regression (Bug D2): _get_sp500_momentum respektiert gepinntes as_of."""

    def _gspc_hist(self, daily_pct: float = 0.005, n: int = 250) -> pd.DataFrame:
        close = _make_close(start=5000.0, daily_pct=daily_pct, n=n)
        return pd.DataFrame({"Close": close.to_numpy()}, index=close.index.tz_localize("UTC"))

    @staticmethod
    def _two_phase_close(n: int = 250, k: int = 160) -> pd.Series:
        """Close-Serie: flach bis Index k, danach exponentiell +1 %/Tag (naiver Index).

        Das Momentum über 126 Punkte hängt vom Endpunkt ab — so unterscheidet
        sich der Wert auf der vollen Serie messbar vom Wert auf der
        beschnittenen (bis as_of = Index k) Serie.
        """
        flat = [100.0] * k
        ramp = [100.0 * (1.01 ** i) for i in range(1, n - k + 1)]
        dates = pd.date_range("2025-01-01", periods=n, freq="B")
        return pd.Series(flat + ramp, index=dates, name="Close")

    def test_as_of_uses_only_closes_up_to_as_of(self, tmp_path):
        """Momentum bei as_of entspricht der Berechnung auf der beschnittenen Serie.

        Die gemockte ^GSPC-Historie reicht über as_of hinaus — geprüft wird,
        dass nur Close-Werte <= as_of in die 6M-Berechnung eingehen und nicht
        die volle (bis-heute-)Serie. as_of = Index 160: davor liegen 161
        Punkte (berechenbar), danach 89 Punkte, die NICHT einfließen dürfen.
        """
        close = self._two_phase_close(n=250, k=160)
        as_of = close.index[160].strftime("%Y-%m-%d")
        hist = pd.DataFrame(
            {"Close": close.to_numpy()}, index=close.index.tz_localize("UTC")
        )
        gspc = MagicMock()
        gspc.history.return_value = hist

        with patch("concilium.data.yf.Ticker", return_value=gspc), \
             patch.dict(os.environ, {"CONCILIUM_CACHE_DIR": str(tmp_path)}):
            result = _get_sp500_momentum(as_of=as_of)

        expected = _compute_momentum_6m(
            close.loc[close.index <= pd.Timestamp(as_of)]
        )
        assert result == pytest.approx(expected)
        assert result != pytest.approx(_compute_momentum_6m(close))

    def test_as_of_passed_through_collect_ticker_data(self):
        """collect_ticker_data(as_of=...) reicht as_of an _get_sp500_momentum durch."""
        close = _make_close(start=100.0, daily_pct=0.01, n=300)
        n = len(close)
        hist = pd.DataFrame(
            {
                "Open": close.values,
                "High": close.values,
                "Low": close.values,
                "Close": close,
                "Volume": [1_000_000.0] * n,
            },
            index=close.index,
        )
        t = MagicMock()
        t.history.return_value = hist
        t.info = {"marketCap": 1e12, "trailingPE": 20.0}
        t.news = []
        # Spät genug, dass >=126 Punkte <= as_of bleiben (sonst momentum_6m
        # None und _get_sp500_momentum wird nie aufgerufen).
        as_of = close.index[200].strftime("%Y-%m-%d")

        with patch("concilium.data.yf.Ticker", return_value=t), \
             patch("concilium.data._get_sp500_momentum", return_value=3.0) as mom_mock, \
             patch("concilium.data._fetch_macro_data", return_value={
                 "eurusd": None, "sp500_pe": None, "sp500_market_cap": None,
                 "sp500_source": "none", "us_10y_yield": None,
             }), \
             patch("concilium.data._fetch_peer_data", return_value=[]), \
             patch("concilium.data._fetch_google_news", return_value=[]), \
             patch("concilium.data._fetch_stocktwits", return_value=[]), \
             patch("concilium.data._fetch_reddit", return_value=[]), \
             patch("concilium.data._fetch_insider_transactions", return_value=[]), \
             patch("concilium.data._fetch_polymarket", return_value=[]), \
             patch("concilium.data._fetch_global_macro_news", return_value=[]), \
             patch.dict(os.environ, {"CONCILIUM_CACHE_DIR": ""}):
            collect_ticker_data("TEST", as_of=as_of)

        mom_mock.assert_called_once_with(as_of=as_of)

    def test_as_of_cache_key_isolated_from_unpinned(self, tmp_path):
        """Gepinnter Lauf teilt den Tages-Cache nicht mit 'heute'-Läufen.

        Erst ein gepinnter Lauf (as_of gesetzt), dann ein un-gepinnter: Der
        zweite darf nicht aus dem as_of-Cache bedient werden (sonst würde ein
        falscher, bis-as_of-berechneter Benchmark-Wert in Live-Läufe gelangen)
        — yf.Ticker muss also zweimal aufgerufen werden.
        """
        hist = self._gspc_hist(daily_pct=0.005, n=250)
        as_of = hist.index[200].strftime("%Y-%m-%d")
        gspc = MagicMock()
        gspc.history.return_value = hist

        with patch("concilium.data.yf.Ticker", return_value=gspc) as ticker_mock, \
             patch.dict(os.environ, {"CONCILIUM_CACHE_DIR": str(tmp_path)}):
            first = _get_sp500_momentum(as_of=as_of)
            second = _get_sp500_momentum()

        assert first is not None
        assert second == pytest.approx(first)  # gleiche Serie → praktisch gleicher Wert
        assert ticker_mock.call_count == 2  # kein Cache-Sharing as_of ↔ heute

    def test_as_of_unpinned_cache_hits_and_still_works(self, tmp_path):
        """Un-gepinnter Fall unverändert: erster Aufruf fetcht, zweiter kommt aus Cache."""
        hist = self._gspc_hist(daily_pct=0.005, n=250)
        gspc = MagicMock()
        gspc.history.return_value = hist

        with patch("concilium.data.yf.Ticker", return_value=gspc) as ticker_mock, \
             patch.dict(os.environ, {"CONCILIUM_CACHE_DIR": str(tmp_path)}):
            first = _get_sp500_momentum()
            second = _get_sp500_momentum()

        assert first is not None
        assert second == first
        assert ticker_mock.call_count == 1
