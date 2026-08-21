"""Tests für Backtest-Kennzahlen: Sharpe, Max-Drawdown, Win-Rate (Aufgabe 2)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.backtest import run_backtest  # noqa: E402


def _make_synthetic_history(n_days: int = 250, start_price: float = 100.0, trend: float = 0.0) -> list[dict]:
    """Erstellt eine synthetische Preisreihe mit genug Tagen für SMA200.

    Args:
        n_days: Anzahl Tage.
        start_price: Startkurs.
        trend: Tendenz pro Tag (z.B. 0.001 = +0.1%/Tag).
    """
    import random

    random.seed(42)
    history = []
    price = start_price
    for i in range(n_days):
        # Leichtes Rauschen + optioneller Trend
        daily_change = random.gauss(trend, 0.015)
        price *= (1 + daily_change)
        history.append({
            "date": f"2025-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}",
            "open": price * 0.99,
            "high": price * 1.01,
            "low": price * 0.98,
            "close": round(price, 4),
            "volume": 1000000,
        })
    return history


class TestBacktestNewMetrics:
    """Tests für die neuen Backtest-Kennzahlen."""

    def test_returns_new_metric_keys(self):
        """Backtest liefert sharpe_ratio, max_drawdown_pct, win_rate_pct, anzahl_trades."""
        history = _make_synthetic_history(250)
        result = run_backtest({"history": history})

        assert "sharpe_ratio" in result
        assert "max_drawdown_pct" in result
        assert "win_rate_pct" in result
        assert "anzahl_trades" in result

    def test_too_short_history_returns_none_metrics(self):
        """Bei zu kurzer Historie sind alle neuen Kennzahlen None/0."""
        history = _make_synthetic_history(50)
        result = run_backtest({"history": history})

        assert result["sharpe_ratio"] is None
        assert result["max_drawdown_pct"] is None
        assert result["win_rate_pct"] is None
        assert result["anzahl_trades"] == 0

    def test_sharpe_ratio_is_float_or_none(self):
        """Sharpe Ratio ist float oder None."""
        history = _make_synthetic_history(250)
        result = run_backtest({"history": history})

        assert result["sharpe_ratio"] is None or isinstance(result["sharpe_ratio"], float)

    def test_max_drawdown_is_negative_or_zero(self):
        """Max Drawdown sollte <= 0% sein (Drawdown ist negativ)."""
        history = _make_synthetic_history(250, trend=0.001)
        result = run_backtest({"history": history})

        if result["max_drawdown_pct"] is not None:
            assert result["max_drawdown_pct"] <= 0.01  # minimaler Toleranzwert

    def test_win_rate_in_range(self):
        """Win-Rate sollte zwischen 0 und 100 liegen."""
        history = _make_synthetic_history(250, trend=0.002)
        result = run_backtest({"history": history})

        if result["win_rate_pct"] is not None:
            assert 0 <= result["win_rate_pct"] <= 100

    def test_deterministic_sharpe_with_constant_returns(self):
        """Mit deterministischen konstanten Renditen lässt sich Sharpe deterministisch prüfen."""
        # Preise mit exakt +1% pro Tag → konstante Rendite → std=0 → Sharpe=None
        history = []
        price = 100.0
        for i in range(250):
            price *= 1.01
            history.append({
                "date": f"2025-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}",
                "close": round(price, 4),
                "volume": 1000000,
            })
        result = run_backtest({"history": history})

        # Bei konstanten Renditen ist std=0 → Sharpe=None
        assert result["sharpe_ratio"] is None

    def test_existing_fields_preserved(self):
        """Bestehende Felder sind weiterhin vorhanden."""
        history = _make_synthetic_history(250)
        result = run_backtest({"history": history})

        assert "strategie_rendite" in result
        assert "buy_hold_rendite" in result
        assert "outperformance" in result
        assert "signale" in result
        assert "anzahl_signale" in result

    def test_no_crash_on_empty_history(self):
        """Leere Historie → kein Crash, alle None."""
        result = run_backtest({"history": []})

        assert result["strategie_rendite"] is None
        assert result["sharpe_ratio"] is None
        assert result["max_drawdown_pct"] is None
        assert result["win_rate_pct"] is None
        assert result["anzahl_trades"] == 0

    def test_anzahl_trades_is_int(self):
        """anzahl_trades ist ein int."""
        history = _make_synthetic_history(250, trend=0.003)
        result = run_backtest({"history": history})

        assert isinstance(result["anzahl_trades"], int)
