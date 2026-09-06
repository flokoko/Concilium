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


class TestBacktestWindowCap:
    """Phase 5: Backtest wertet max. die letzten 500 Handelstage (~2y) aus.

    Hintergrund: Die Kurs-Historie wird mit period="2y" geladen (~500 Tage).
    Sollte das Fenster später verlängert werden (z. B. period="5y"), würde
    sonst der auswertbare Backtest-Zeitraum (startdatum) mitwachsen und die
    Kennzahlen wären über Läufe hinweg nicht mehr vergleichbar. Der Cap
    hält das Fenster konstant — konsistent mit der 2y-Historie.
    """

    @staticmethod
    def _make_osc_history(n_days: int, end_date: str = "2026-09-04") -> list[dict]:
        """Deterministische Serie: Aufwärtstrend + Sinus-Oszillation (Periode 120 Tage).

        Die Oszillation erzeugt wiederholt SMA50/SMA200-Crossovers → Signale.
        """
        import numpy as np
        import pandas as pd

        i = np.arange(n_days)
        closes = 100.0 + 0.02 * i + 20.0 * np.sin(2 * np.pi * i / 120.0)
        dates = pd.bdate_range(end=end_date, periods=n_days).strftime("%Y-%m-%d")
        return [
            {"date": d, "close": round(float(c), 4), "volume": 1_000_000}
            for d, c in zip(dates, closes)
        ]

    def test_longer_history_yields_more_signals(self):
        """Mehr Historie → mehr Signale (CEG-Bug-Fix): 500 Tage > 250 Tage.

        Mit 1y (~250 Tagen) war SMA200 erst ab Tag 199 gültig — es blieb kein
        Zeitraum für Crossovers übrig (0 Signale). Mit 2y (~500 Tagen) gibt
        es ~300 auswertbare Tage → Crossovers werden erkannt.
        """
        long_hist = self._make_osc_history(500)
        short_hist = long_hist[-250:]  # letztes 1y der gleichen Serie

        r_long = run_backtest({"history": long_hist})
        r_short = run_backtest({"history": short_hist})

        assert r_long["anzahl_signale"] > r_short["anzahl_signale"]
        assert r_short["anzahl_signale"] == 0  # exakt der alte CEG-Fehlerfall

    def test_backtest_capped_at_last_500_trading_days(self):
        """Bei >500 Tagen wird nur das letzte 500-Tage-Fenster ausgewertet.

        startdatum ist dann der SMA200-Start des zugeschnittenen Fensters
        (Position 499 im 700-Tage-Frame), nicht der SMA200-Start der vollen
        Serie (Position 199).
        """
        n = 700
        big_hist = self._make_osc_history(n)
        uncapped_start = big_hist[199]["date"]  # SMA200-Start der vollen Serie
        expected_capped_start = big_hist[n - 500 + 199]["date"]
        assert uncapped_start != expected_capped_start  # Cap macht einen Unterschied

        result = run_backtest({"history": big_hist})

        assert result["startdatum"] == expected_capped_start

    def test_cap_noop_at_500_days(self):
        """Bei exakt 500 Tagen ändert der Cap nichts (Fenster == Historie)."""
        hist = self._make_osc_history(500)
        expected_start = hist[199]["date"]

        result = run_backtest({"history": hist})

        assert result["startdatum"] == expected_start
