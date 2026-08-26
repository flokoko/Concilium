"""Tests für factors.py — Multi-Faktor-Score (deterministischer Anker).

Alle Tests sind offline (kein Netzwerk, kein yfinance) — nur reine Berechnung.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.factors import compute_multi_factor_score  # noqa: E402

# ===========================================================================
# Hoch-Scoring: günstiges Value + starkes Momentum + starke Qualität
# ===========================================================================

class TestHighScoring:
    """Günstiges Value, hohes Momentum, starke Qualität → overall hoch."""

    def test_cheap_high_momentum_strong_quality(self):
        """KGV 8, PEG 0.7, Upside 30%, nahe Hoch, starker Konsens,
        hohe Marge, starkes Wachstum, Dividende → overall >= 4."""
        f = {
            "pe_ratio": 8.0,
            "peg_ratio": 0.7,
            "analyst_upside_pct": 30.0,
            "fifty_two_week_high": 100.0,
            "fifty_two_week_low": 50.0,
            "current_price": 95.0,
            "recommendation_mean": 1.5,
            "profit_margin": 0.30,
            "revenue_growth": 0.20,
            "dividend_yield": 0.05,
        }
        result = compute_multi_factor_score(f)
        assert result["overall_score"] is not None
        assert result["overall_score"] >= 4.0
        # Sub-Scores im Bereich 1-5
        assert 1 <= result["value_score"] <= 5
        assert 1 <= result["momentum_score"] <= 5
        assert 1 <= result["quality_score"] <= 5
        # Alle 3 Sub-Scores verfügbar
        assert result["subscores_available"] == 3

    def test_value_score_cheap_pe(self):
        """KGV <= 10 → Value-Komponente hoch."""
        result = compute_multi_factor_score({"pe_ratio": 9.0})
        assert result["value_score"] is not None
        assert result["value_score"] >= 4.0

    def test_momentum_near_high(self):
        """Kurs > 90% des Hochs → Momentum hoch."""
        f = {
            "fifty_two_week_high": 100.0,
            "current_price": 95.0,
            "recommendation_mean": 1.5,
        }
        result = compute_multi_factor_score(f)
        assert result["momentum_score"] is not None
        assert result["momentum_score"] >= 4.0

    def test_quality_high_margin_growth(self):
        """Hohe Marge + starkes Wachstum + Dividende → Quality hoch."""
        f = {
            "profit_margin": 0.30,
            "revenue_growth": 0.25,
            "dividend_yield": 0.045,
        }
        result = compute_multi_factor_score(f)
        assert result["quality_score"] is not None
        assert result["quality_score"] >= 4.0


# ===========================================================================
# Niedrig-Scoring: überbewertet + schwaches Momentum + negative Marge
# ===========================================================================

class TestLowScoring:
    """Überbewertet, schwaches Momentum, negative Marge → overall niedrig."""

    def test_overvalued_weak_momentum_negative_margin(self):
        """KGV 60, PEG 4, Upside -15%, weit vom Hoch, sell-Konsens,
        negative Marge, schrumpfender Umsatz, keine Dividende → overall <= 2.5."""
        f = {
            "pe_ratio": 60.0,
            "peg_ratio": 4.0,
            "analyst_upside_pct": -15.0,
            "fifty_two_week_high": 100.0,
            "fifty_two_week_low": 50.0,
            "current_price": 52.0,
            "recommendation_mean": 4.5,
            "profit_margin": -0.10,
            "revenue_growth": -0.10,
            "dividend_yield": 0.0,
        }
        result = compute_multi_factor_score(f)
        assert result["overall_score"] is not None
        assert result["overall_score"] <= 2.5
        assert 1 <= result["value_score"] <= 5
        assert 1 <= result["momentum_score"] <= 5
        assert 1 <= result["quality_score"] <= 5

    def test_value_score_expensive_pe(self):
        """KGV > 40 → Value-Komponente niedrig."""
        result = compute_multi_factor_score({"pe_ratio": 60.0})
        assert result["value_score"] is not None
        assert result["value_score"] <= 2.0

    def test_momentum_far_from_high(self):
        """Kurs weit vom Hoch, sell-Konsens → Momentum niedrig."""
        f = {
            "fifty_two_week_high": 100.0,
            "current_price": 40.0,
            "recommendation_mean": 4.5,
        }
        result = compute_multi_factor_score(f)
        assert result["momentum_score"] is not None
        assert result["momentum_score"] <= 2.0

    def test_quality_negative_margin_shrinking(self):
        """Negative Marge, schrumpfender Umsatz → Quality niedrig."""
        f = {
            "profit_margin": -0.15,
            "revenue_growth": -0.10,
        }
        result = compute_multi_factor_score(f)
        assert result["quality_score"] is not None
        assert result["quality_score"] <= 2.0


# ===========================================================================
# None-Handling: alle None → overall None
# ===========================================================================

class TestNoneHandling:
    """Alle Felder None oder leer → overall None, kein Crash."""

    def test_all_none(self):
        """Leeres Dict → overall None, subscores_available 0."""
        result = compute_multi_factor_score({})
        assert result["overall_score"] is None
        assert result["value_score"] is None
        assert result["momentum_score"] is None
        assert result["quality_score"] is None
        assert result["subscores_available"] == 0

    def test_all_values_none(self):
        """Alle Felder explizit None → overall None."""
        f = {
            "pe_ratio": None,
            "peg_ratio": None,
            "analyst_upside_pct": None,
            "profit_margin": None,
            "revenue_growth": None,
            "dividend_yield": None,
            "recommendation_mean": None,
            "fifty_two_week_high": None,
            "current_price": None,
        }
        result = compute_multi_factor_score(f)
        assert result["overall_score"] is None
        assert result["subscores_available"] == 0

    def test_partial_none(self):
        """Nur Value-Felder, Rest None → Value verfügbar, others None, overall = Value."""
        f = {"pe_ratio": 15.0}
        result = compute_multi_factor_score(f)
        assert result["value_score"] is not None
        assert result["momentum_score"] is None
        assert result["quality_score"] is None
        assert result["subscores_available"] == 1
        assert result["overall_score"] == result["value_score"]

    def test_non_dict_input(self):
        """None als Input → kein Crash, overall None."""
        result = compute_multi_factor_score(None)  # type: ignore[arg-type]
        assert result["overall_score"] is None
        assert result["subscores_available"] == 0


# ===========================================================================
# Wertebereich: alle Scores 1-5
# ===========================================================================

class TestScoreRange:
    """Sub-Scores sind immer im Bereich 1-5 (wenn nicht None)."""

    def test_extreme_high_values(self):
        """Alle Werte extrem bullisch → alle Sub-Scores <= 5."""
        f = {
            "pe_ratio": 1.0,
            "peg_ratio": 0.1,
            "analyst_upside_pct": 100.0,
            "fifty_two_week_high": 100.0,
            "current_price": 99.0,
            "recommendation_mean": 1.0,
            "profit_margin": 0.50,
            "revenue_growth": 0.50,
            "dividend_yield": 0.10,
        }
        result = compute_multi_factor_score(f)
        for key in ("value_score", "momentum_score", "quality_score", "overall_score"):
            val = result[key]
            assert val is not None
            assert 1 <= val <= 5, f"{key}={val} outside 1-5"

    def test_extreme_low_values(self):
        """Alle Werte extrem bearisch → alle Sub-Scores >= 1."""
        f = {
            "pe_ratio": 200.0,
            "peg_ratio": 10.0,
            "analyst_upside_pct": -50.0,
            "fifty_two_week_high": 100.0,
            "current_price": 10.0,
            "recommendation_mean": 5.0,
            "profit_margin": -0.50,
            "revenue_growth": -0.30,
            "dividend_yield": 0.0,
        }
        result = compute_multi_factor_score(f)
        for key in ("value_score", "momentum_score", "quality_score", "overall_score"):
            val = result[key]
            assert val is not None
            assert 1 <= val <= 5, f"{key}={val} outside 1-5"


# ===========================================================================
# Kurzeinschaetzung: deterministischer Text
# ===========================================================================

class TestKurzeinschaetzung:
    """Die kurzeinschaetzung ist ein deterministischer deutscher Text."""

    def test_all_available_text(self):
        """Alle 3 Sub-Scores verfügbar → Text enthält Value, Momentum, Qualität."""
        f = {
            "pe_ratio": 8.0,
            "fifty_two_week_high": 100.0,
            "current_price": 95.0,
            "recommendation_mean": 1.5,
            "profit_margin": 0.30,
            "revenue_growth": 0.20,
        }
        result = compute_multi_factor_score(f)
        text = result["kurzeinschaetzung"]
        assert "Value" in text
        assert "Momentum" in text
        assert "Qualität" in text

    def test_empty_text(self):
        """Leeres Dict → Fallback-Text."""
        result = compute_multi_factor_score({})
        text = result["kurzeinschaetzung"]
        assert text  # nicht leer
        assert "Keine" in text or "N/A" in text or "nicht" in text.lower()

    def test_only_value_text(self):
        """Nur Value → Text enthält Value, nicht Momentum."""
        result = compute_multi_factor_score({"pe_ratio": 15.0})
        text = result["kurzeinschaetzung"]
        assert "Value" in text


# ===========================================================================
# Bug 3: 52W-Nähe-Komponente mit current_price
# ===========================================================================

class TestMomentumCurrentPrice:
    """Bug 3: _momentum_score nutzt current_price für die 52W-Nähe-Komponente."""

    def test_momentum_with_current_price_near_high(self):
        """Mit current_price nahe 52W-Hoch → momentum_score höher."""
        f = {
            "fifty_two_week_high": 100.0,
            "fifty_two_week_low": 50.0,
            "current_price": 95.0,
            "recommendation_mean": 3.0,  # neutral
        }
        result = compute_multi_factor_score(f)
        # current_price/high = 0.95 → 5.0, recommendation_mean=3.0 → 3.0
        # Durchschnitt = 4.0
        assert result["momentum_score"] is not None
        assert result["momentum_score"] >= 4.0

    def test_momentum_without_current_price(self):
        """Ohne current_price → nur recommendation_mean-Komponente."""
        f = {
            "fifty_two_week_high": 100.0,
            "fifty_two_week_low": 50.0,
            "recommendation_mean": 3.0,
        }
        result = compute_multi_factor_score(f)
        assert result["momentum_score"] is not None
        # Nur eine Komponente (recommendation_mean=3.0) → momentum_score = 3.0
        assert result["momentum_score"] == 3.0

    def test_momentum_current_price_far_from_high(self):
        """Mit current_price weit vom Hoch → momentum_score niedriger."""
        f = {
            "fifty_two_week_high": 100.0,
            "fifty_two_week_low": 50.0,
            "current_price": 40.0,
            "recommendation_mean": 3.0,  # neutral
        }
        result = compute_multi_factor_score(f)
        # current_price/high = 0.40 → 1.0, recommendation_mean=3.0 → 3.0
        # Durchschnitt = 2.0
        assert result["momentum_score"] is not None
        assert result["momentum_score"] <= 2.0
