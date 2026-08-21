"""Tests für _get_sp500_benchmark — SPY-Fallback-Logik mit gemockten Ticker.info-Werten.

Diese Tests benötigen KEIN Netzwerk — yfinance.Ticker wird gemockt.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.data import _get_sp500_benchmark  # noqa: E402


class TestGetSp500Benchmark:
    """Tests für _get_sp500_benchmark — Fallback-Logik."""

    def test_gspc_returns_pe(self):
        """Wenn ^GSPC trailingPE liefert → Quelle GSPC, kein SPY-Aufruf."""
        gspc_mock = MagicMock()
        gspc_mock.info = {"trailingPE": 22.5, "marketCap": 4.0e13}

        spy_mock = MagicMock()
        spy_mock.info = {"trailingPE": 25.69}

        with patch("concilium.data.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = [gspc_mock, spy_mock]
            result = _get_sp500_benchmark()

        assert result["sp500_pe"] == 22.5
        assert result["sp500_source"] == "GSPC"
        assert result["sp500_market_cap"] == 4.0e13

    def test_gspc_none_falls_back_to_spy(self):
        """Wenn ^GSPC trailingPE=None → Fallback auf SPY, Quelle SPY."""
        gspc_mock = MagicMock()
        gspc_mock.info = {"trailingPE": None, "marketCap": None}

        spy_mock = MagicMock()
        spy_mock.info = {"trailingPE": 25.69, "marketCap": 9.0e11}

        with patch("concilium.data.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = [gspc_mock, spy_mock]
            result = _get_sp500_benchmark()

        assert result["sp500_pe"] == 25.69
        assert result["sp500_source"] == "SPY"
        assert result["sp500_market_cap"] == 9.0e11

    def test_both_fail_returns_none(self):
        """Wenn beide Quellen None liefern → sp500_pe=None, source='none'."""
        gspc_mock = MagicMock()
        gspc_mock.info = {"trailingPE": None}

        spy_mock = MagicMock()
        spy_mock.info = {"trailingPE": None}

        with patch("concilium.data.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = [gspc_mock, spy_mock]
            result = _get_sp500_benchmark()

        assert result["sp500_pe"] is None
        assert result["sp500_source"] == "none"
        assert result["sp500_market_cap"] is None

    def test_gspc_exception_falls_back_to_spy(self):
        """Wenn ^GSPC eine Exception wirft → Fallback auf SPY."""
        spy_mock = MagicMock()
        spy_mock.info = {"trailingPE": 25.69}

        with patch("concilium.data.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = [Exception("Network error"), spy_mock]
            result = _get_sp500_benchmark()

        assert result["sp500_pe"] == 25.69
        assert result["sp500_source"] == "SPY"

    def test_result_dict_has_required_keys(self):
        """Das Result-dict hat immer die Keys sp500_pe, sp500_market_cap, sp500_source."""
        with patch("concilium.data.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = [Exception("fail"), Exception("fail")]
            result = _get_sp500_benchmark()

        assert "sp500_pe" in result
        assert "sp500_market_cap" in result
        assert "sp500_source" in result
