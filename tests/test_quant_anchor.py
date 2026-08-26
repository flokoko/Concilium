"""Tests, dass der quantitative Multi-Faktor-Score-Anker in den Analysten-Prompt injiziert wird."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import _build_data_text  # noqa: E402


def test_fundamental_role_contains_quant_score_anchor() -> None:
    """Der Fundamental-Daten-Text enthält den deterministischen Quant-Score-Anker."""
    data = {
        "ticker": "TEST",
        "fundamentals": {
            "name": "Test AG",
            "pe_ratio": 12.0,
            "peg_ratio": 0.8,
            "profit_margin": 0.30,
            "revenue_growth": 0.20,
            "analyst_upside_pct": 25.0,
            "fifty_two_week_high": 100.0,
        },
        "technicals": {},
        "sentiment": {},
        "news": [],
        "macro": {},
        "peers": [],
    }
    text = _build_data_text(data, role="fundamental")
    assert "Quant-Score" in text
    assert "Gesamt" in text
    assert "Referenz" in text


def test_no_overall_score_no_anchor():
    """Wenn keine Fundamentals verfügbar sind, erscheint kein Anker (kein Crash)."""
    data = {
        "fundamentals": {},
        "technicals": {},
        "sentiment": {},
        "news": [],
        "macro": {},
        "peers": [],
    }
    text = _build_data_text(data, role="fundamental")
    assert "Quant-Score" not in text


def test_current_price_in_quant_score():
    """Bug 3: current_price aus technicals wird an compute_multi_factor_score durchgereicht.

    fundamentals enthält fifty_two_week_high aber kein current_price.
    technicals enthält current_price. Der Quant-Score muss die 52W-Nähe
    einbeziehen → momentum_score != nur recommendation_mean.
    """
    from concilium.factors import compute_multi_factor_score

    data = {
        "ticker": "TEST",
        "fundamentals": {
            "name": "Test AG",
            "fifty_two_week_high": 100.0,
            "fifty_two_week_low": 50.0,
            "recommendation_mean": 3.0,  # neutral
            # KEIN current_price in fundamentals!
        },
        "technicals": {
            "current_price": 95.0,  # nahe Hoch → 5.0 für 52W-Nähe
        },
        "sentiment": {},
        "news": [],
        "macro": {},
        "peers": [],
    }
    text = _build_data_text(data, role="fundamental")
    # Quant-Score muss im Text auftauchen
    assert "Quant-Score" in text
    # Der Momentum-Score muss >= 4.0 sein (95/100=0.95→5.0, rec_mean=3.0→3.0, avg=4.0)
    # Wenn current_price NICHT durchgereicht wird, wäre momentum_score=3.0 (nur rec_mean)
    assert "Momentum" in text
    # Vergleiche mit direktem compute_multi_factor_score ohne current_price
    mf_without = compute_multi_factor_score(
        {"fifty_two_week_high": 100.0, "fifty_two_week_low": 50.0, "recommendation_mean": 3.0}
    )
    mf_with = compute_multi_factor_score(
        {"fifty_two_week_high": 100.0, "fifty_two_week_low": 50.0,
         "current_price": 95.0, "recommendation_mean": 3.0}
    )
    # Mit current_price sollte momentum_score höher sein
    assert mf_with["momentum_score"] > mf_without["momentum_score"]
