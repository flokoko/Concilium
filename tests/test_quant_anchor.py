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
