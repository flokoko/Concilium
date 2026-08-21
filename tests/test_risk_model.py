"""Tests für compute_position_size und risk_manager-Volatilität (Aufgabe 1)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import compute_position_size, risk_manager  # noqa: E402


class TestComputePositionSize:
    """Tests für die compute_position_size Formel."""

    def test_basic_volatility_targeting(self):
        """risk_budget 2%, vol 30% → 0.02/0.30 = 6.67%."""
        result = compute_position_size(0.30)
        assert result is not None
        assert abs(result - 6.67) < 0.01

    def test_cap_at_max_position(self):
        """Sehr niedrige Volatilität → Cap bei max_position_pct."""
        # vol=0.1 → 0.02/0.1 = 20%, capped bei 10%
        result = compute_position_size(0.10, max_position_pct=10.0)
        assert result == 10.0

    def test_custom_risk_budget(self):
        """Custom risk_budget_pct."""
        # risk_budget 3%, vol 30% → 0.03/0.30 = 10%
        result = compute_position_size(0.30, risk_budget_pct=3.0)
        assert result == 10.0

    def test_none_volatility(self):
        """None → None."""
        assert compute_position_size(None) is None

    def test_zero_volatility(self):
        """0 → None."""
        assert compute_position_size(0.0) is None

    def test_negative_volatility(self):
        """Negativ → None."""
        assert compute_position_size(-0.3) is None

    def test_non_float_volatility(self):
        """Nicht float-konvertierbar → None."""
        assert compute_position_size("not-a-number") is None

    def test_string_float_volatility(self):
        """String '0.30' → wird zu float konvertiert."""
        result = compute_position_size("0.30")
        assert result is not None
        assert abs(result - 6.67) < 0.01

    def test_very_high_volatility(self):
        """Sehr hohe Volatilität → kleine Position."""
        # vol=2.0 → 0.02/2.0 = 1%
        result = compute_position_size(2.0)
        assert result == 1.0


class TestRiskManagerVolatility:
    """Tests für risk_manager: rechnerische Positionsgröße wird ergänzt."""

    def test_risk_manager_adds_computational_fields(self):
        """risk_manager hängt positionsgröße_rechnerisch_pct und volatilität_annualisiert_pct an."""
        # Mock-LLM
        class _MockLLM:
            def chat(self, messages, temperature=0.3, **kwargs):
                return json.dumps({
                    "rolle": "Risk-Manager",
                    "risiko_score": 3,
                    "empfehlung": "GENEHMIGT",
                    "positionsgröße_empfohlen": "5",
                })

        # Synthetische Historie mit genug Daten für Volatilitätsberechnung
        history = []
        price = 100.0
        for i in range(30):
            # Abwechselnd +1% und -1% → konstante Volatilität
            if i % 2 == 0:
                price *= 1.01
            else:
                price *= 0.99
            history.append({"date": f"2026-01-{i + 1:02d}", "close": price})

        data = {
            "ticker": "TEST",
            "fundamentals": {"name": "Test Inc."},
            "technicals": {"current_price": 100.0},
            "history": history,
            "sentiment": {},
        }

        trade = {"aktion": "KAUFEN", "zielkurs": 120, "stop_loss": 90}
        result = risk_manager(trade, data, _MockLLM(), data_text="pre-computed")

        # LLM-Felder bleiben erhalten
        assert result.get("risiko_score") == 3
        assert result.get("empfehlung") == "GENEHMIGT"
        # Rechnerische Felder wurden ergänzt
        assert "positionsgröße_rechnerisch_pct" in result
        assert "volatilität_annualisiert_pct" in result
        # Volatilität sollte > 0 sein ( Preise schwanken)
        assert result["volatilität_annualisiert_pct"] is not None
        assert result["volatilität_annualisiert_pct"] > 0
        # Positionsgröße sollte berechnet worden sein
        assert result["positionsgröße_rechnerisch_pct"] is not None
        assert result["positionsgröße_rechnerisch_pct"] > 0

    def test_risk_manager_no_history(self):
        """Bei fehlender Historie → None für beide Felder."""
        class _MockLLM:
            def chat(self, messages, temperature=0.3, **kwargs):
                return json.dumps({"rolle": "Risk-Manager", "risiko_score": 2})

        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {},
            "history": [],
            "sentiment": {},
        }

        result = risk_manager({"aktion": "HALTEN"}, data, _MockLLM(), data_text="pre-computed")

        assert result["volatilität_annualisiert_pct"] is None
        assert result["positionsgröße_rechnerisch_pct"] is None
        # LLM-Feld bleibt erhalten
        assert result.get("risiko_score") == 2
