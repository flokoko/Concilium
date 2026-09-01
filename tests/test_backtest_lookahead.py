"""Tests für Look-ahead-Bias-Freiheit im Backtest (Phase 5).

Kernannahme: Ein Signal (SMA50 > SMA200, RSI < 75) wird aus dem Close des
Signal-Tags berechnet und ist daher erst NACH Handelsschluss bekannt. Der
realisierbare Einstieg ist der Close des NÄCHSTEN Handelstags — niemals der
Close des Signal-Tags selbst (klassischer Look-ahead-Bias).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.backtest import run_backtest  # noqa: E402
from concilium.report import generate_report  # noqa: E402


def _make_step_history() -> list[dict]:
    """Deterministische Kursreihe mit genau einem Long-Signal-Start an Tag 254.

    Konstruktion:
      - Tage 0-239: Close 100 → SMA50 == SMA200 (strikt '>' falsch) → kein Signal.
      - Tag 240: Sprung auf 110 → SMA50 > SMA200, aber RSI = 100 (+10-Sprung
        im 14-Tage-Fenster) → RSI-Filter blockt.
      - Tag 254: Der +10-Sprung (delta[240] = +10) fällt aus dem RSI-Fenster
        (Fenster deckt deltas 241..254 ab, alle 0) → RSI = 0 < 75 und
        SMA50 = (35*100 + 15*110)/50 = 103.0 > SMA200 = 100.75 → Long-Signal.
      - Tag 255: Einbruch auf 108 (RSI bleibt < 75, SMA50 > SMA200).
      - Tage 256-268: Close 110 (RSI = 50, Signal bleibt 1).
      - Tag 269 (letzter Tag): Das Delta +2 der Erholung (Tag 256) ist allein
        im RSI-Fenster (deltas 256..269) → RSI = 100 ≥ 75 → Signal endet
        (FLAT) und schließt den Trade exakt am Serienende.

    Damit ist handverifizierbar:
      - Signal-Tag 254: Close 110.0 (NICHT als Einstieg nutzbar)
      - Entry-Tag 255:  Close 108.0 (realisierbarer Einstieg)
      - Exit (Signalwechsel Tag 269): Close 110.0
    Alte Look-ahead-Logik: Entry 110.0 == Exit 110.0 → Verlust (0 % Win-Rate).
    Korrigierte Logik:      Entry 108.0 <  Exit 110.0 → Gewinn (100 % Win-Rate).
    """
    history = []
    for i in range(270):
        if i < 240:
            close = 100.0
        elif i == 255:
            close = 108.0
        else:
            close = 110.0
        history.append({
            "date": f"2025-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000000,
        })
    return history


class TestLookaheadFreeEntry:
    """Entry-Logik: Einstiegskurs muss vom Tag NACH dem Signal stammen."""

    def test_entry_is_next_day_close(self):
        """(a) Long-Signal an Tag i → Einstiegskurs ist der Close von Tag i+1."""
        result = run_backtest({"history": _make_step_history()})

        longs = [s for s in result["signale"] if s.get("aktion") == "LONG"]
        assert len(longs) == 1
        sig = longs[0]
        # Signal-Tag 254: angezeigter Close bleibt der Signal-Tags-Close …
        assert sig["close"] == 110.0
        # … aber der realisierbare Einstieg ist der Folgetags-Close (Tag 255).
        assert sig["entry_close"] == 108.0
        assert sig["entry_close"] != sig["close"]

    def test_win_rate_uses_next_day_entry(self):
        """(c) Win-Rate nutzt den Folgetags-Close als Entry (nicht Signal-Tags-Close)."""
        result = run_backtest({"history": _make_step_history()})

        # Ein Trade (geschlossen am Signalwechsel Tag 269):
        # Entry 108.0 → Exit 110.0 = Gewinn.
        # Die alte Look-ahead-Logik hätte Entry 110.0 == Exit 110.0 = Verlust
        # (Win-Rate 0.0) geliefert.
        assert result["anzahl_trades"] == 1
        assert result["win_rate_pct"] == 100.0

    def test_lookahead_check_true_with_enough_data(self):
        """(b) Ausreichend Daten → lookahead_bias_geprueft ist True + Hinweis."""
        result = run_backtest({"history": _make_step_history()})

        assert result["lookahead_bias_geprueft"] is True
        assert isinstance(result.get("lookahead_bias_hinweis"), str)
        assert result["lookahead_bias_hinweis"]

    def test_lookahead_check_true_without_trades(self):
        """Check läuft auch ohne Trades (leere Trade-Liste = vakuum wahr)."""
        history = [{"date": f"2025-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}", "close": 100.0}
                   for i in range(300)]
        result = run_backtest({"history": history})

        assert result["anzahl_trades"] == 0
        assert result["lookahead_bias_geprueft"] is True

    def test_too_little_data_no_crash_check_false(self):
        """(d) Zu wenig Daten → kein Crash, lookahead_bias_geprueft False."""
        result_empty = run_backtest({"history": []})
        assert result_empty["lookahead_bias_geprueft"] is False
        assert result_empty["win_rate_pct"] is None

        result_short = run_backtest({"history": [{"date": "2025-01-02", "close": 100.0}]})
        assert result_short["lookahead_bias_geprueft"] is False
        assert isinstance(result_short.get("lookahead_bias_hinweis"), str)


class TestCheckLookaheadBiasUnit:
    """Unit-Tests für die interne Check-Funktion."""

    def _mini_df(self):
        import pandas as pd

        return pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]})

    def test_entry_after_signal_passes(self):
        from concilium.backtest import _check_lookahead_bias

        assert _check_lookahead_bias(self._mini_df(), [1, 2], [2, 3]) is True

    def test_entry_same_day_fails(self):
        from concilium.backtest import _check_lookahead_bias

        assert _check_lookahead_bias(self._mini_df(), [1], [1]) is False

    def test_entry_before_signal_fails(self):
        from concilium.backtest import _check_lookahead_bias

        assert _check_lookahead_bias(self._mini_df(), [2], [1]) is False

    def test_length_mismatch_fails(self):
        from concilium.backtest import _check_lookahead_bias

        assert _check_lookahead_bias(self._mini_df(), [1], [1, 2]) is False

    def test_out_of_bounds_fails(self):
        from concilium.backtest import _check_lookahead_bias

        assert _check_lookahead_bias(self._mini_df(), [1], [99]) is False

    def test_no_trades_passes(self):
        from concilium.backtest import _check_lookahead_bias

        assert _check_lookahead_bias(self._mini_df(), [], []) is True


def _make_report_result(lookahead_geprueft: bool | None) -> dict:
    """Minimal-Result mit Backtest-Sektion für Report-Tests."""
    bt = {
        "strategie_rendite": 12.5,
        "buy_hold_rendite": 10.0,
        "outperformance": 2.5,
        "anzahl_signale": 1,
        "sharpe_ratio": 1.2,
        "max_drawdown_pct": -5.0,
        "win_rate_pct": 100.0,
        "anzahl_trades": 1,
        "startdatum": "2025-01-02",
        "enddatum": "2025-12-30",
        "signale": [
            {"date": "2025-09-25", "aktion": "LONG", "close": 110.0, "entry_close": 108.0},
        ],
    }
    if lookahead_geprueft is not None:
        bt["lookahead_bias_geprueft"] = lookahead_geprueft
    return {"ticker": "TEST", "data": {}, "backtest": bt}


class TestReportLookaheadLine:
    """Report zeigt die Look-ahead-Zeile nur bei bestandenem Check."""

    def test_report_shows_lookahead_free_when_checked(self):
        report = generate_report(_make_report_result(True))
        assert "Backtest look-ahead-frei geprüft: ja" in report

    def test_report_omits_line_when_not_checked(self):
        report = generate_report(_make_report_result(False))
        assert "Backtest look-ahead-frei geprüft: ja" not in report

    def test_report_omits_line_when_key_missing(self):
        report = generate_report(_make_report_result(None))
        assert "Backtest look-ahead-frei geprüft: ja" not in report
