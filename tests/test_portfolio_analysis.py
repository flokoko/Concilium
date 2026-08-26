"""Tests für portfolio_analysis.py — deterministische Portfolio-Aggregation.

Alle Tests sind OFFLINE-fähig (kein Netzwerk, konstruierte History-Reihen).
Testet Korrelations-Berechnung, Overlap-Erkennung, Konzentrationswarnungen,
CLI-Parsing und Report-Sektion.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.portfolio_analysis import (  # noqa: E402
    _daily_returns,
    _extract_close_series,
    _pearson_r,
    compute_correlations,
    correlation_sample_sizes,
    portfolio_concentration,
    portfolio_context_to_text,
    portfolio_overlap,
    run_portfolio_analysis,
)

# ---------------------------------------------------------------------------
# Helper: konstruierte History-Records
# ---------------------------------------------------------------------------


def _make_history(closes: list[float], start_date: str = "2024-01-01") -> list[dict]:
    """Erstellt history_records aus einer Liste von Close-Preisen."""
    from datetime import datetime, timedelta

    base = datetime.strptime(start_date, "%Y-%m-%d")
    records = []
    for i, close in enumerate(closes):
        date = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        records.append({"date": date, "close": close})
    return records


def _make_history_from_returns(
    returns: list[float], start_price: float = 100.0, start_date: str = "2024-01-01"
) -> list[dict]:
    """Erstellt history_records aus einer Liste von Tagesrenditen."""
    from datetime import datetime, timedelta

    base = datetime.strptime(start_date, "%Y-%m-%d")
    prices = [start_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    records = []
    for i, close in enumerate(prices):
        date = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        records.append({"date": date, "close": close})
    return records


def _linear_history(n: int = 60, start: float = 100.0, step: float = 1.0) -> list[dict]:
    """Lineare Aufwärts-Preise: 100, 101, 102, ... → perfekt positiv korreliert."""
    return _make_history([start + step * i for i in range(n)])


def _inverse_history(n: int = 60, start: float = 160.0, step: float = -1.0) -> list[dict]:
    """Lineare Abwärts-Preise: 160, 159, 158, ... → perfekt anti-korreliert
    mit einer steigenden Reihe bei gleichem Start-Datum."""
    return _make_history([start + step * i for i in range(n)])


def _randomish_history(n: int = 60, seed: int = 42) -> list[dict]:
    """Deterministisch pseudo-random Preise → ca. 0 Korrelation mit linear."""
    import random

    random.seed(seed)
    prices = [100.0]
    for _ in range(n - 1):
        change = random.uniform(-2.0, 2.0)
        prices.append(prices[-1] + change)
    return _make_history(prices)


# ---------------------------------------------------------------------------
# _extract_close_series
# ---------------------------------------------------------------------------


class TestExtractCloseSeries:
    def test_basic_extraction(self):
        history = [
            {"date": "2024-01-01", "close": 100.0},
            {"date": "2024-01-02", "close": 101.0},
        ]
        result = _extract_close_series(history)
        assert result == {"2024-01-01": 100.0, "2024-01-02": 101.0}

    def test_empty_history(self):
        assert _extract_close_series([]) == {}

    def test_missing_close(self):
        history = [{"date": "2024-01-01", "close": 100.0}, {"date": "2024-01-02"}]
        result = _extract_close_series(history)
        assert "2024-01-01" in result
        assert "2024-01-02" not in result

    def test_nan_close(self):
        history = [{"date": "2024-01-01", "close": float("nan")}]
        result = _extract_close_series(history)
        assert result == {}

    def test_time_field_fallback(self):
        history = [{"time": "2024-01-01", "close": 100.0}]
        result = _extract_close_series(history)
        assert "2024-01-01" in result

    def test_no_date_uses_index(self):
        history = [{"close": 100.0}, {"close": 101.0}]
        result = _extract_close_series(history)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _daily_returns
# ---------------------------------------------------------------------------


class TestDailyReturns:
    def test_basic(self):
        assert _daily_returns([100.0, 110.0]) == [0.1]

    def test_empty(self):
        assert _daily_returns([]) == []

    def test_single(self):
        assert _daily_returns([100.0]) == []

    def test_zero_prev(self):
        assert _daily_returns([0.0, 100.0]) == []

    def test_multiple(self):
        returns = _daily_returns([100.0, 110.0, 105.0, 115.0])
        assert len(returns) == 3
        assert returns[0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# _pearson_r
# ---------------------------------------------------------------------------


class TestPearsonR:
    def test_perfect_positive(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r = _pearson_r(x, y)
        assert r is not None
        assert r == pytest.approx(1.0, abs=1e-10)

    def test_perfect_negative(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        r = _pearson_r(x, y)
        assert r is not None
        assert r == pytest.approx(-1.0, abs=1e-10)

    def test_zero_correlation(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        y = [8, 7, 6, 5, 4, 3, 2, 1]
        # x steigt, y fällt → negative Korrelation
        r = _pearson_r(x, y)
        assert r is not None
        assert r == pytest.approx(-1.0, abs=1e-10)

    def test_uncorrelated(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        r = _pearson_r(x, y)
        # x steigt linear, y alterniert → Korrelation nahe 0
        assert r is not None
        assert abs(r) < 0.3

    def test_length_mismatch(self):
        assert _pearson_r([1, 2, 3], [1, 2]) is None

    def test_too_short(self):
        assert _pearson_r([1], [1]) is None

    def test_zero_variance(self):
        x = [5, 5, 5, 5]
        y = [1, 2, 3, 4]
        assert _pearson_r(x, y) is None

    def test_clamped(self):
        """Pearson r wird auf [-1, 1] geclamped."""
        x = list(range(100))
        y = [v * 1e15 for v in x]
        r = _pearson_r(x, y)
        assert r is not None
        assert r <= 1.0
        assert r >= -1.0


# ---------------------------------------------------------------------------
# compute_correlations
# ---------------------------------------------------------------------------


class TestComputeCorrelations:
    def test_perfect_positive_correlation(self):
        """Zwei identisch steigende Reihen → r = 1.0."""
        history_map = {
            "AAA": _linear_history(60, 100, 1),
            "BBB": _linear_history(60, 200, 2),
        }
        corr = compute_correlations(history_map)
        assert corr["AAA"]["BBB"] is not None
        assert corr["AAA"]["BBB"] == pytest.approx(1.0, abs=1e-6)
        assert corr["BBB"]["AAA"] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative_correlation(self):
        """Eine mit konstant positiven Returns, eine mit konstant negativen
        Returns → r = -1.0 (da Mittelwert-Abweichungen perfekt anti-korreliert)."""
        # Konstante positive Returns vs konstante negative Returns
        # Bei konstanten Returns ist die Varianz 0 → r wäre None.
        # Stattdessen: Returns die systematisch entgegengesetzt schwanken.
        # Reihe A: alternierende Returns [0.01, -0.01, 0.01, -0.01, ...]
        # Reihe B: entgegengesetzt [-0.01, 0.01, -0.01, 0.01, ...]
        returns_a = [0.01 if i % 2 == 0 else -0.01 for i in range(60)]
        returns_b = [-0.01 if i % 2 == 0 else 0.01 for i in range(60)]
        history_map = {
            "UP": _make_history_from_returns(returns_a),
            "DOWN": _make_history_from_returns(returns_b),
        }
        corr = compute_correlations(history_map)
        assert corr["UP"]["DOWN"] is not None
        assert corr["UP"]["DOWN"] == pytest.approx(-1.0, abs=1e-6)

    def test_approximately_zero_correlation(self):
        """Linear vs random → |r| sollte klein sein."""
        history_map = {
            "LIN": _linear_history(60, 100, 1),
            "RND": _randomish_history(60, seed=123),
        }
        corr = compute_correlations(history_map)
        r = corr["LIN"]["RND"]
        assert r is not None
        # Bei random Daten sollte |r| nicht extrem sein
        assert abs(r) < 0.7

    def test_diagonal_is_one(self):
        history_map = {"A": _linear_history(60), "B": _inverse_history(60)}
        corr = compute_correlations(history_map)
        assert corr["A"]["A"] == 1.0
        assert corr["B"]["B"] == 1.0

    def test_insufficient_overlap(self):
        """Zu wenige überlappende Tage → None."""
        history_map = {
            "A": _make_history([100, 101, 102], "2024-01-01"),
            "B": _make_history([100, 101, 102], "2024-01-01"),
        }
        corr = compute_correlations(history_map)
        assert corr["A"]["B"] is None

    def test_empty_history(self):
        history_map = {"A": [], "B": _linear_history(60)}
        corr = compute_correlations(history_map)
        assert corr["A"]["B"] is None

    def test_single_ticker(self):
        history_map = {"SOLO": _linear_history(60)}
        corr = compute_correlations(history_map)
        assert corr["SOLO"]["SOLO"] == 1.0

    def test_non_overlapping_dates(self):
        history_map = {
            "A": _make_history([100 + i for i in range(50)], "2024-01-01"),
            "B": _make_history([100 + i for i in range(50)], "2025-01-01"),
        }
        corr = compute_correlations(history_map)
        assert corr["A"]["B"] is None


# ---------------------------------------------------------------------------
# correlation_sample_sizes
# ---------------------------------------------------------------------------


class TestCorrelationSampleSizes:
    def test_basic(self):
        history_map = {
            "A": _linear_history(60),
            "B": _linear_history(60),
        }
        sizes = correlation_sample_sizes(history_map)
        assert sizes["A"]["B"] == 60
        assert sizes["A"]["A"] == 60

    def test_partial_overlap(self):
        history_map = {
            "A": _make_history([100 + i for i in range(40)], "2024-01-01"),
            "B": _make_history([100 + i for i in range(40)], "2024-01-20"),
        }
        sizes = correlation_sample_sizes(history_map)
        # A: 40 Tage ab 2024-01-01, B: 40 Tage ab 2024-01-20
        # Overlap: 2024-01-20 bis 2024-02-08 = 20 Tage
        assert sizes["A"]["B"] == 21  # 20 Tage ab 01-20 + 1


# ---------------------------------------------------------------------------
# portfolio_overlap
# ---------------------------------------------------------------------------


class TestPortfolioOverlap:
    def test_direct_ticker_match(self):
        positions = [
            {"ticker": "AAPL", "sheet_symbol": "AAPL", "name": "Apple", "depot_pct": 5.0, "_idx": 0},
        ]
        result = portfolio_overlap(["AAPL"], positions)
        assert len(result["direct_overlaps"]) == 1
        assert result["direct_overlaps"][0]["ticker"] == "AAPL"
        assert result["total_overlap_pct"] == 5.0

    def test_sheet_symbol_match(self):
        positions = [
            {"ticker": "IS3R.DE", "sheet_symbol": "IS3R.TG", "name": "iShares",
             "depot_pct": 3.0, "_idx": 0},
        ]
        result = portfolio_overlap(["IS3R.TG"], positions)
        assert len(result["direct_overlaps"]) == 1

    def test_name_match(self):
        positions = [
            {"ticker": "XYZ", "sheet_symbol": "XYZ", "name": "Apple Inc", "depot_pct": 4.0, "_idx": 0},
        ]
        result = portfolio_overlap(["AAPL"], positions, {"AAPL": "Apple"})
        assert len(result["direct_overlaps"]) == 1

    def test_no_overlap(self):
        positions = [
            {"ticker": "MSFT", "sheet_symbol": "MSFT", "name": "Microsoft", "depot_pct": 5.0, "_idx": 0},
        ]
        result = portfolio_overlap(["AAPL"], positions)
        assert len(result["direct_overlaps"]) == 0
        assert result["total_overlap_pct"] == 0.0

    def test_empty_positions(self):
        result = portfolio_overlap(["AAPL"], [])
        assert result["direct_overlaps"] == []
        assert result["total_overlap_pct"] == 0.0

    def test_empty_tickers(self):
        positions = [{"ticker": "AAPL", "depot_pct": 5.0, "_idx": 0}]
        result = portfolio_overlap([], positions)
        assert result["direct_overlaps"] == []

    def test_high_overlap_warning(self):
        positions = [
            {"ticker": "AAPL", "sheet_symbol": "AAPL", "name": "Apple", "depot_pct": 25.0, "_idx": 0},
        ]
        result = portfolio_overlap(["AAPL"], positions)
        assert any("Hoher Gesamt-Overlap" in w for w in result["warnings"])

    def test_two_positions_only_one_overlaps(self):
        """Bug 1: Zwei Depot-Positionen, nur eine überlappt →
        total_overlap_pct nur die überlappte, nicht die Summe beider."""
        positions = [
            {"ticker": "AAPL", "sheet_symbol": "AAPL", "name": "Apple", "depot_pct": 5.0},
            {"ticker": "MSFT", "sheet_symbol": "MSFT", "name": "Microsoft", "depot_pct": 30.0},
        ]
        # KEIN _idx in positions — das ist der Bug!
        result = portfolio_overlap(["AAPL"], positions)
        assert len(result["direct_overlaps"]) == 1
        assert result["direct_overlaps"][0]["ticker"] == "AAPL"
        # Nur AAPL (5%) überlappt, nicht AAPL+MSFT (35%)
        assert result["total_overlap_pct"] == 5.0

    def test_idx_based_dedup_with_explicit_idx(self):
        """Auch mit _idx-Flags wird korrekt dedupliziert (eindeutige Indices)."""
        positions = [
            {"ticker": "AAPL", "sheet_symbol": "AAPL", "name": "Apple", "depot_pct": 5.0, "_idx": 0},
            {"ticker": "MSFT", "sheet_symbol": "MSFT", "name": "Microsoft", "depot_pct": 30.0, "_idx": 1},
        ]
        result = portfolio_overlap(["AAPL"], positions)
        assert result["total_overlap_pct"] == 5.0


# ---------------------------------------------------------------------------
# portfolio_concentration
# ---------------------------------------------------------------------------


class TestPortfolioConcentration:
    def test_single_position_over_5pct(self):
        positions = [
            {"name": "Apple", "depot_pct": 8.0, "region": "USA"},
        ]
        warnings = portfolio_concentration(positions)
        assert any("Apple" in w and "8.0%" in w for w in warnings)

    def test_single_position_under_5pct(self):
        positions = [
            {"name": "Small", "depot_pct": 3.0, "region": "USA"},
        ]
        warnings = portfolio_concentration(positions)
        assert not any("Small" in w and "3.0%" in w for w in warnings)

    def test_target_weight_warning(self):
        positions = []
        weights = {"AAA": 7.0}
        warnings = portfolio_concentration(positions, weights)
        assert any("AAA" in w and "7.0%" in w for w in warnings)

    def test_cumulative_target_weight(self):
        positions = []
        weights = {"AAA": 3.0, "BBB": 4.0}
        warnings = portfolio_concentration(positions, weights)
        assert any("7.0%" in w for w in warnings)

    def test_high_cumulative_warning(self):
        positions = []
        weights = {"AAA": 12.0, "BBB": 10.0}
        warnings = portfolio_concentration(positions, weights)
        assert any("Hohe kumulierte" in w for w in warnings)

    def test_region_concentration(self):
        positions = [
            {"name": "A", "depot_pct": 15.0, "region": "USA"},
            {"name": "B", "depot_pct": 20.0, "region": "USA"},
        ]
        warnings = portfolio_concentration(positions)
        assert any("USA" in w and "35.0%" in w for w in warnings)

    def test_empty_positions(self):
        warnings = portfolio_concentration([])
        assert warnings == []


# ---------------------------------------------------------------------------
# run_portfolio_analysis (Integration)
# ---------------------------------------------------------------------------


class TestRunPortfolioAnalysis:
    def test_basic(self):
        results = {
            "AAA": {
                "data": {"history": _linear_history(60), "fundamentals": {"name": "Company A"}},
                "portfolio_fit": {"ziel_gewichtung_pct": 4.0},
            },
            "BBB": {
                "data": {"history": _inverse_history(60), "fundamentals": {"name": "Company B"}},
                "portfolio_fit": {"ziel_gewichtung_pct": 3.0},
            },
        }
        pa = run_portfolio_analysis(results, positions=None)
        assert "AAA" in pa["analysed_tickers"]
        assert "BBB" in pa["analysed_tickers"]
        assert pa["correlations"]["AAA"]["BBB"] is not None
        assert pa["target_weights"]["AAA"] == 4.0
        assert pa["target_weights"]["BBB"] == 3.0
        assert pa["overlap"] is None  # no positions passed

    def test_with_positions(self):
        positions = [
            {"ticker": "AAA", "sheet_symbol": "AAA", "name": "Company A",
             "depot_pct": 6.0, "region": "USA", "_idx": 0},
        ]
        results = {
            "AAA": {
                "data": {"history": _linear_history(60), "fundamentals": {"name": "Company A"}},
                "portfolio_fit": {"ziel_gewichtung_pct": 5.0},
            },
        }
        pa = run_portfolio_analysis(results, positions=positions)
        assert pa["overlap"] is not None
        assert len(pa["overlap"]["direct_overlaps"]) == 1

    def test_missing_history(self):
        results = {
            "AAA": {"data": {}, "portfolio_fit": None},
        }
        pa = run_portfolio_analysis(results)
        assert pa["correlations"]["AAA"]["AAA"] == 1.0

    def test_no_portfolio_fit(self):
        results = {
            "AAA": {"data": {"history": _linear_history(60)}, "portfolio_fit": None},
        }
        pa = run_portfolio_analysis(results)
        assert pa["target_weights"] == {}

    def test_empty_results(self):
        pa = run_portfolio_analysis({})
        assert pa["analysed_tickers"] == []
        assert pa["correlations"] == {}


# ---------------------------------------------------------------------------
# portfolio_context_to_text
# ---------------------------------------------------------------------------


class TestPortfolioContextToText:
    def test_produces_json(self):
        import json

        context = {
            "analysed_tickers": ["AAA", "BBB"],
            "correlations": {"AAA": {"BBB": 0.5}},
            "target_weights": {"AAA": 4.0},
            "concentration_warnings": ["warn"],
            "overlap": None,
        }
        text = portfolio_context_to_text(context)
        parsed = json.loads(text)
        assert "AAA" in parsed["analysed_tickers"]
        assert parsed["target_weights"]["AAA"] == 4.0

    def test_empty_context(self):
        text = portfolio_context_to_text({})
        assert "analysed_tickers" in text


# ---------------------------------------------------------------------------
# CLI: --portfolio Flag Parsing
# ---------------------------------------------------------------------------


class TestCLIPortfolioParsing:
    """Test CLI-Parsing: --portfolio und Mutual-Exclusion."""

    def test_portfolio_and_ticker_mutual_exclusion(self):
        from concilium.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--portfolio", "AAPL,NVDA", "--ticker", "MSFT"])
        assert exc_info.value.code == 2

    def test_portfolio_and_tickers_mutual_exclusion(self):
        from concilium.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--portfolio", "AAPL,NVDA", "--tickers", "MSFT,GOOG"])
        assert exc_info.value.code == 2

    def test_evaluate_and_portfolio_error(self):
        from concilium.cli import main

        result = main(["--evaluate", "--portfolio", "AAPL,NVDA"])
        assert result == 1

    def test_portfolio_only_no_other_mode(self):
        """--portfolio allein sollte nicht durch Mutual-Exclusion brechen."""
        from concilium.cli import main

        # Wir mocken run_portfolio, damit kein echter Lauf passiert
        with patch("concilium.cli.run_portfolio") as mock_run:
            with patch("concilium.cli.generate_report") as mock_report:
                with patch("concilium.cli.LLMClient"):
                    mock_run.return_value = {
                        "results": {},
                        "portfolio_analysis": {},
                        "tickers": [],
                    }
                    mock_report.return_value = "# dummy"
                    # --no-llm damit kein LLMClient erstellt wird
                    result = main(["--portfolio", "AAPL", "--no-llm"])
        # Sollte nicht mit SystemExit(2) brechen
        assert result in (0, 1)


# --------------------------------------------------------------------------- #
# Bug 6: --portfolio reicht --peers an run_pipeline durch
# --------------------------------------------------------------------------- #


class TestPortfolioPeersDurchreichung:
    """Bug 6: run_portfolio reicht peers an run_pipeline weiter."""

    def test_run_portfolio_passes_peers_to_run_pipeline(self):
        """run_portfolio reicht peers-Parameter an run_pipeline durch."""
        from unittest.mock import patch

        from concilium.pipeline import run_portfolio

        with patch("concilium.pipeline.run_pipeline") as mock_run, \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.portfolio_analysis.run_portfolio_analysis", return_value={}):

            mock_run.return_value = {
                "ticker": "AAPL",
                "data": {"fundamentals": {}, "technicals": {}, "history": []},
                "no_llm": True,
            }

            run_portfolio(["AAPL", "MSFT"], llm=None, peers=["GOOG", "AMZN"])

            # run_pipeline wurde für jeden Ticker aufgerufen
            assert mock_run.call_count == 2
            # Prüfe dass peers durchgereicht wurde
            for call in mock_run.call_args_list:
                assert call.kwargs.get("peers") == ["GOOG", "AMZN"]

    def test_run_portfolio_peers_default_none(self):
        """Default für peers ist None (rückwärtskompatibel)."""
        from unittest.mock import patch

        from concilium.pipeline import run_portfolio

        with patch("concilium.pipeline.run_pipeline") as mock_run, \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.portfolio_analysis.run_portfolio_analysis", return_value={}):

            mock_run.return_value = {
                "ticker": "AAPL",
                "data": {"fundamentals": {}, "technicals": {}, "history": []},
                "no_llm": True,
            }

            run_portfolio(["AAPL"], llm=None)

            assert mock_run.call_count == 1
            assert mock_run.call_args.kwargs.get("peers") is None


# ---------------------------------------------------------------------------
# Report: Portfolio-Blick-Sektion
# ---------------------------------------------------------------------------


class TestReportPortfolioBlick:
    """Test Report-Sektion 'Portfolio-Blick'."""

    def test_section_appears_with_portfolio_analysis(self):
        from concilium.report import generate_report

        result = {
            "ticker": "AAA",
            "data": {"ticker": "AAA", "fundamentals": {}, "technicals": {},
                     "sentiment": {}, "news": [], "history": []},
            "no_llm": True,
            "portfolio_analysis": {
                "analysed_tickers": ["AAA", "BBB"],
                "correlations": {"AAA": {"AAA": 1.0, "BBB": 0.85}, "BBB": {"AAA": 0.85, "BBB": 1.0}},
                "target_weights": {"AAA": 4.0, "BBB": 3.0},
                "overlap": None,
                "concentration_warnings": ["Kumulierte Ziel-Gewichtung: 7.0%."],
            },
        }
        report = generate_report(result)
        assert "Portfolio-Blick" in report
        assert "Korrelations-Matrix" in report
        assert "0.85" in report
        assert "Ziel-Gewichtungen" in report
        assert "Konzentrationswarnungen" in report

    def test_section_not_present_without_portfolio_analysis(self):
        from concilium.report import generate_report

        result = {
            "ticker": "AAA",
            "data": {"ticker": "AAA", "fundamentals": {}, "technicals": {},
                     "sentiment": {}, "news": [], "history": []},
            "no_llm": True,
        }
        report = generate_report(result)
        assert "Portfolio-Blick" not in report

    def test_high_correlation_highlighted(self):
        from concilium.report import generate_report

        result = {
            "ticker": "AAA",
            "data": {"ticker": "AAA", "fundamentals": {}, "technicals": {},
                     "sentiment": {}, "news": [], "history": []},
            "no_llm": True,
            "portfolio_analysis": {
                "analysed_tickers": ["AAA", "BBB"],
                "correlations": {"AAA": {"AAA": 1.0, "BBB": 0.9}, "BBB": {"AAA": 0.9, "BBB": 1.0}},
                "target_weights": {},
                "overlap": None,
                "concentration_warnings": [],
            },
        }
        report = generate_report(result)
        assert "⚠️" in report
        assert "0.90" in report

    def test_na_for_missing_correlation(self):
        from concilium.report import generate_report

        result = {
            "ticker": "AAA",
            "data": {"ticker": "AAA", "fundamentals": {}, "technicals": {},
                     "sentiment": {}, "news": [], "history": []},
            "no_llm": True,
            "portfolio_analysis": {
                "analysed_tickers": ["AAA", "BBB"],
                "correlations": {"AAA": {"AAA": 1.0, "BBB": None}, "BBB": {"AAA": None, "BBB": 1.0}},
                "target_weights": {},
                "overlap": None,
                "concentration_warnings": [],
            },
        }
        report = generate_report(result)
        assert "n/a" in report

    def test_overlap_section(self):
        from concilium.report import generate_report

        result = {
            "ticker": "AAA",
            "data": {"ticker": "AAA", "fundamentals": {}, "technicals": {},
                     "sentiment": {}, "news": [], "history": []},
            "no_llm": True,
            "portfolio_analysis": {
                "analysed_tickers": ["AAA"],
                "correlations": {},
                "target_weights": {"AAA": 4.0},
                "overlap": {
                    "direct_overlaps": [
                        {"ticker": "AAA", "position_name": "Apple", "depot_pct": 5.0},
                    ],
                    "total_overlap_pct": 5.0,
                    "warnings": ["Overlap: AAA (Apple) bereits mit 5.0% im Depot."],
                },
                "concentration_warnings": [],
            },
        }
        report = generate_report(result)
        assert "Overlap" in report
        assert "Apple" in report
        assert "5.0" in report

    def test_robust_empty_portfolio_analysis(self):
        from concilium.report import generate_report

        result = {
            "ticker": "AAA",
            "data": {"ticker": "AAA", "fundamentals": {}, "technicals": {},
                     "sentiment": {}, "news": [], "history": []},
            "no_llm": True,
            "portfolio_analysis": {},
        }
        report = generate_report(result)
        assert "Portfolio-Blick" in report
        # Sollte nicht crashen, keine Exception


# ---------------------------------------------------------------------------
# Pipeline: run_portfolio (mocked)
# ---------------------------------------------------------------------------


class TestRunPortfolio:
    """Test run_portfolio mit gemockter run_pipeline."""

    def test_no_llm_mode(self):
        """Im --no-llm Modus sollten nur Datensnapshots gesammelt werden."""
        from concilium.pipeline import run_portfolio

        def mock_run_pipeline(ticker, **kwargs):
            return {
                "ticker": ticker,
                "data": {
                    "ticker": ticker,
                    "fundamentals": {"name": f"Company {ticker}"},
                    "technicals": {},
                    "sentiment": {},
                    "news": [],
                    "history": _linear_history(60),
                },
                "no_llm": True,
                "portfolio_fit": None,
            }

        with patch("concilium.pipeline.run_pipeline", side_effect=mock_run_pipeline):
            with patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]):
                result = run_portfolio(["AAA", "BBB"], llm=None)

        assert "results" in result
        assert "portfolio_analysis" in result
        assert "AAA" in result["results"]
        assert "BBB" in result["results"]
        assert result["portfolio_analysis"]["analysed_tickers"] == ["AAA", "BBB"]

    def test_ticker_failure_doesnt_crash(self):
        """Ein fehlgeschlagener Ticker crasht nicht den Portfolio-Modus."""
        from concilium.pipeline import run_portfolio

        def mock_run_pipeline(ticker, **kwargs):
            if ticker == "BAD":
                raise ValueError("Ungültiger Ticker")
            return {
                "ticker": ticker,
                "data": {"ticker": ticker, "fundamentals": {}, "technicals": {},
                         "sentiment": {}, "news": [], "history": _linear_history(60)},
                "no_llm": True,
                "portfolio_fit": None,
            }

        with patch("concilium.pipeline.run_pipeline", side_effect=mock_run_pipeline):
            with patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]):
                result = run_portfolio(["AAA", "BAD"], llm=None)

        assert "AAA" in result["results"]
        assert "BAD" in result["results"]
        assert result["results"]["BAD"].get("error") is not None


# ---------------------------------------------------------------------------
# Pipeline: skip_final / run_portfolio PM-once (Phase-2 Fix)
# ---------------------------------------------------------------------------


class TestSkipFinal:
    """Testet run_pipeline(skip_final=True) — PM und Journal werden übersprungen."""

    def test_skip_final_no_pm_no_journal(self, tmp_path, monkeypatch):
        """skip_final=True: result['final'] ist None, _final_pending gesetzt,
        _completed_steps enthält NICHT 'final', Journal nicht geschrieben."""
        from unittest.mock import MagicMock, patch

        from concilium.pipeline import run_pipeline

        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        mock_trade = {"aktion": "KAUFEN", "rating": "KAUFEN", "_raw": ""}
        mock_risk = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}

        with patch("concilium.pipeline.collect_ticker_data") as mock_data, \
             patch("concilium.pipeline.analyst_team", return_value={}), \
             patch("concilium.pipeline.debate", return_value={}), \
             patch("concilium.pipeline.trader", return_value=mock_trade), \
             patch("concilium.pipeline.risk_manager", return_value=mock_risk), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_fit_agent", return_value={"portfolio_fit_score": 2}), \
             patch("concilium.pipeline.trade_revision", return_value=mock_trade), \
             patch("concilium.pipeline.portfolio_manager") as mock_pm, \
             patch("concilium.pipeline.build_feedback_context", return_value=""), \
             patch("concilium.pipeline.build_reflection_context", return_value=""), \
             patch("concilium.journal.append_decision") as mock_journal:

            mock_data.return_value = {
                "ticker": "TEST", "fundamentals": {}, "technicals": {},
                "sentiment": {}, "news": [],
            }

            result = run_pipeline("TEST", llm=MagicMock(), ensemble=False, skip_final=True)

        # PM wurde NICHT aufgerufen
        assert mock_pm.call_count == 0
        # Journal wurde NICHT aufgerufen
        assert mock_journal.call_count == 0
        # final ist None
        assert result.get("final") is None
        # _final_pending Marker gesetzt
        assert result.get("_final_pending") is True
        # 'final' nicht in _completed_steps
        assert "final" not in result.get("_completed_steps", [])
        # Vor-Schritte aber vorhanden
        assert "trade_revision" in result.get("_completed_steps", [])
        assert result.get("trade") is not None
        assert result.get("risk") is not None

    def test_skip_final_false_runs_pm_and_journal(self, tmp_path, monkeypatch):
        """skip_final=False (Default): PM wird aufgerufen, Journal geschrieben."""
        from unittest.mock import MagicMock, patch

        from concilium.pipeline import run_pipeline

        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        mock_trade = {"aktion": "KAUFEN", "rating": "KAUFEN", "_raw": ""}

        with patch("concilium.pipeline.collect_ticker_data") as mock_data, \
             patch("concilium.pipeline.analyst_team", return_value={}), \
             patch("concilium.pipeline.debate", return_value={}), \
             patch("concilium.pipeline.trader", return_value=mock_trade), \
             patch("concilium.pipeline.risk_manager", return_value={"risiko_score": 3}), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_fit_agent", return_value={"portfolio_fit_score": 2}), \
             patch("concilium.pipeline.trade_revision", return_value=mock_trade), \
             patch("concilium.pipeline.portfolio_manager", return_value={"entscheidung": "GENEHMIGT"}) as mock_pm, \
             patch("concilium.pipeline.build_feedback_context", return_value=""), \
             patch("concilium.pipeline.build_reflection_context", return_value=""), \
             patch("concilium.journal.append_decision") as mock_journal:

            mock_data.return_value = {
                "ticker": "TEST", "fundamentals": {}, "technicals": {},
                "sentiment": {}, "news": [],
            }

            result = run_pipeline("TEST", llm=MagicMock(), ensemble=False, skip_final=False)

        # PM wurde aufgerufen
        assert mock_pm.call_count == 1
        # Journal wurde aufgerufen
        assert mock_journal.call_count == 1
        # final ist gesetzt
        assert result["final"]["entscheidung"] == "GENEHMIGT"
        # 'final' in _completed_steps
        assert "final" in result["_completed_steps"]


class TestRunPortfolioPMOnce:
    """Testet dass run_portfolio den PM pro Ticker genau EINMAL aufruft."""

    def test_pm_called_once_per_ticker(self):
        """PM wird pro Ticker genau einmal aufgerufen (mit Portfolio-Kontext),
        Journal wird genau einmal pro Ticker geschrieben."""
        from unittest.mock import MagicMock, patch

        from concilium.pipeline import run_portfolio

        def mock_run_pipeline(ticker, **kwargs):
            # Simuliere skip_final=True Verhalten (PM übersprungen)
            return {
                "ticker": ticker,
                "data": {
                    "ticker": ticker,
                    "fundamentals": {"name": f"Company {ticker}"},
                    "technicals": {},
                    "sentiment": {},
                    "news": [],
                    "history": _linear_history(60),
                },
                "no_llm": False,
                "trade": {"aktion": "KAUFEN", "rating": "KAUFEN", "_raw": ""},
                "risk": {"risiko_score": 3, "empfehlung": "GENEHMIGT"},
                "portfolio_fit": {"portfolio_fit_score": 2},
                "final": None,
                "_final_pending": True,
                "_completed_steps": ["data", "analysts", "debate", "trade", "risk",
                                     "portfolio_fit", "trade_revision"],
                "_feedback_context": "",
                "_reflection_context": "",
            }

        with patch("concilium.pipeline.run_pipeline", side_effect=mock_run_pipeline), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_manager", return_value={"entscheidung": "GENEHMIGT", "confidence": 4}) as mock_pm, \
             patch("concilium.journal.append_decision") as mock_journal:

            result = run_portfolio(["AAA", "BBB"], llm=MagicMock(), ensemble=False)

        # PM wurde genau 2x aufgerufen (einmal pro Ticker)
        assert mock_pm.call_count == 2
        # Journal wurde genau 2x aufgerufen (einmal pro Ticker)
        assert mock_journal.call_count == 2

        # Beide Ticker haben final mit Entscheidung
        for t in ["AAA", "BBB"]:
            assert result["results"][t]["final"]["entscheidung"] == "GENEHMIGT"
            assert result["results"][t]["_final_pending"] is False
            assert result["results"][t]["_journal_written"] is True

    def test_pm_receives_portfolio_context(self):
        """PM wird mit portfolio_context aufgerufen (nicht None)."""
        from unittest.mock import MagicMock, patch

        from concilium.pipeline import run_portfolio

        def mock_run_pipeline(ticker, **kwargs):
            return {
                "ticker": ticker,
                "data": {
                    "ticker": ticker,
                    "fundamentals": {}, "technicals": {},
                    "sentiment": {}, "news": [],
                    "history": _linear_history(60),
                },
                "no_llm": False,
                "trade": {"aktion": "KAUFEN", "_raw": ""},
                "risk": {"risiko_score": 3},
                "portfolio_fit": None,
                "final": None,
                "_final_pending": True,
                "_completed_steps": ["data", "analysts", "debate", "trade", "risk",
                                     "portfolio_fit", "trade_revision"],
                "_feedback_context": "",
                "_reflection_context": "",
            }

        with patch("concilium.pipeline.run_pipeline", side_effect=mock_run_pipeline), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_manager", return_value={"entscheidung": "GENEHMIGT"}) as mock_pm, \
             patch("concilium.journal.append_decision"):

            run_portfolio(["AAA"], llm=MagicMock(), ensemble=False)

        # PM-Aufruf sollte portfolio_context als Keyword-Arg haben (nicht None)
        call_kwargs = mock_pm.call_args.kwargs
        assert call_kwargs.get("portfolio_context") is not None

    def test_pm_skipped_for_failed_ticker(self):
        """Ticker mit error oder ohne trade/risk → PM wird nicht aufgerufen."""
        from unittest.mock import MagicMock, patch

        from concilium.pipeline import run_portfolio

        def mock_run_pipeline(ticker, **kwargs):
            if ticker == "BAD":
                raise ValueError("Ungültiger Ticker")
            return {
                "ticker": ticker,
                "data": {
                    "ticker": ticker,
                    "fundamentals": {}, "technicals": {},
                    "sentiment": {}, "news": [],
                    "history": _linear_history(60),
                },
                "no_llm": False,
                "trade": {"aktion": "KAUFEN", "_raw": ""},
                "risk": {"risiko_score": 3},
                "portfolio_fit": None,
                "final": None,
                "_final_pending": True,
                "_completed_steps": ["data", "analysts", "debate", "trade", "risk",
                                     "portfolio_fit", "trade_revision"],
                "_feedback_context": "",
                "_reflection_context": "",
            }

        with patch("concilium.pipeline.run_pipeline", side_effect=mock_run_pipeline), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_manager", return_value={"entscheidung": "GENEHMIGT"}) as mock_pm, \
             patch("concilium.journal.append_decision") as mock_journal:

            result = run_portfolio(["AAA", "BAD"], llm=MagicMock(), ensemble=False)

        # PM wurde nur 1x aufgerufen (für AAA, nicht für BAD)
        assert mock_pm.call_count == 1
        # Journal wurde nur 1x aufgerufen
        assert mock_journal.call_count == 1
        # BAD hat error
        assert result["results"]["BAD"].get("error") is not None

    def test_no_llm_mode_no_pm(self):
        """Im --no-llm Modus wird kein PM aufgerufen."""
        from unittest.mock import patch

        from concilium.pipeline import run_portfolio

        def mock_run_pipeline(ticker, **kwargs):
            return {
                "ticker": ticker,
                "data": {
                    "ticker": ticker,
                    "fundamentals": {}, "technicals": {},
                    "sentiment": {}, "news": [],
                    "history": _linear_history(60),
                },
                "no_llm": True,
                "portfolio_fit": None,
            }

        with patch("concilium.pipeline.run_pipeline", side_effect=mock_run_pipeline), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_manager") as mock_pm, \
             patch("concilium.journal.append_decision") as mock_journal:

            result = run_portfolio(["AAA", "BBB"], llm=None)

        # Kein PM, kein Journal im no-llm Modus
        assert mock_pm.call_count == 0
        assert mock_journal.call_count == 0
        assert result["portfolio_analysis"]["analysed_tickers"] == ["AAA", "BBB"]
