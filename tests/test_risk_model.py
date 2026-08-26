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


class TestRiskManagerRiskBlockInPrompt:
    """Tests: risk_manager reichert den LLM-Prompt mit dem rechnerischen Risiko-Modell an."""

    def test_prompt_contains_risk_block(self):
        """Der LLM-Prompt enthält 'RECHNERISCHES RISIKO-MODELL'."""
        class _CapturingLLM:
            def __init__(self):
                self.captured: list[list[dict]] = []

            def chat(self, messages, temperature=0.3, **kwargs):
                self.captured.append(messages)
                return json.dumps({"risiko_score": 3, "empfehlung": "GENEHMIGT"})

        llm = _CapturingLLM()
        history = []
        price = 100.0
        for i in range(30):
            if i % 2 == 0:
                price *= 1.01
            else:
                price *= 0.99
            history.append({"close": price})

        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {"current_price": 100.0},
            "history": history,
            "sentiment": {},
        }
        risk_manager({"aktion": "KAUFEN"}, data, llm, data_text="dummy")

        user_content = llm.captured[0][1]["content"]
        assert "RECHNERISCHES RISIKO-MODELL" in user_content

    def test_prompt_contains_position_size(self):
        """Der Prompt enthält die rechnerische Positionsgröße als Zahl."""
        class _CapturingLLM:
            def __init__(self):
                self.captured: list[list[dict]] = []

            def chat(self, messages, temperature=0.3, **kwargs):
                self.captured.append(messages)
                return json.dumps({"risiko_score": 3})

        llm = _CapturingLLM()
        history = []
        price = 100.0
        for i in range(30):
            if i % 2 == 0:
                price *= 1.01
            else:
                price *= 0.99
            history.append({"close": price})

        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {"current_price": 100.0},
            "history": history,
            "sentiment": {},
        }
        result = risk_manager({"aktion": "KAUFEN"}, data, llm, data_text="dummy")

        user_content = llm.captured[0][1]["content"]
        # Die rechnerische Positionsgröße muss im Prompt stehen (als Zahl, nicht N/A)
        assert "Rechnerische Positionsgröße" in user_content
        assert "N/A" not in user_content.split("Rechnerische Positionsgröße")[1].split("\n")[0]
        # Und im Rückgabedict
        assert result["positionsgröße_rechnerisch_pct"] is not None
        assert result["positionsgröße_rechnerisch_pct"] > 0

    def test_prompt_no_history_shows_na(self):
        """Bei fehlender Historie steht 'N/A' im Prompt für Volatilität und Position."""
        class _CapturingLLM:
            def __init__(self):
                self.captured: list[list[dict]] = []

            def chat(self, messages, temperature=0.3, **kwargs):
                self.captured.append(messages)
                return json.dumps({"risiko_score": 2})

        llm = _CapturingLLM()
        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {},
            "history": [],
            "sentiment": {},
        }
        risk_manager({"aktion": "HALTEN"}, data, llm, data_text="dummy")

        user_content = llm.captured[0][1]["content"]
        assert "RECHNERISCHES RISIKO-MODELL" in user_content
        assert "N/A" in user_content

    def test_volatility_in_prompt_matches_return_dict(self):
        """Der Volatilitätswert im Prompt stimmt mit dem im Rückgabedict überein."""
        class _CapturingLLM:
            def __init__(self):
                self.captured: list[list[dict]] = []

            def chat(self, messages, temperature=0.3, **kwargs):
                self.captured.append(messages)
                return json.dumps({"risiko_score": 3})

        llm = _CapturingLLM()
        history = []
        price = 100.0
        for i in range(30):
            if i % 2 == 0:
                price *= 1.01
            else:
                price *= 0.99
            history.append({"close": price})

        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {"current_price": 100.0},
            "history": history,
            "sentiment": {},
        }
        result = risk_manager({"aktion": "KAUFEN"}, data, llm, data_text="dummy")

        user_content = llm.captured[0][1]["content"]
        vol_pct = result["volatilität_annualisiert_pct"]
        # Der Wert aus dem Rückgabedict muss im Prompt auftauchen
        assert str(vol_pct) in user_content
        pos_pct = result["positionsgröße_rechnerisch_pct"]
        assert str(pos_pct) in user_content

    def test_prompt_contains_anweisung(self):
        """Der Prompt enthält die Anweisung zur positionsgröße_empfohlen."""
        class _CapturingLLM:
            def __init__(self):
                self.captured: list[list[dict]] = []

            def chat(self, messages, temperature=0.3, **kwargs):
                self.captured.append(messages)
                return json.dumps({"risiko_score": 3})

        llm = _CapturingLLM()
        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {},
            "history": [],
            "sentiment": {},
        }
        risk_manager({"aktion": "HALTEN"}, data, llm, data_text="dummy")

        user_content = llm.captured[0][1]["content"]
        assert "positionsgröße_empfohlen" in user_content
        assert "Abweichung" in user_content

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


# ===========================================================================
# Bug 5: _normalize_pct_string + risk_manager Normalisierung
# ===========================================================================

class TestNormalizePctString:
    """Bug 5: _normalize_pct_string extrahiert float aus number/string."""

    def test_number_passthrough(self):
        from concilium.agents import _normalize_pct_string
        assert _normalize_pct_string(5.0) == 5.0
        assert _normalize_pct_string(3) == 3.0

    def test_string_with_percent_and_space(self):
        from concilium.agents import _normalize_pct_string
        assert _normalize_pct_string("5 %") == 5.0

    def test_string_with_percent_no_space(self):
        from concilium.agents import _normalize_pct_string
        assert _normalize_pct_string("5%") == 5.0

    def test_string_with_comma_decimal(self):
        from concilium.agents import _normalize_pct_string
        assert _normalize_pct_string("5,5") == 5.5

    def test_string_with_dot_decimal(self):
        from concilium.agents import _normalize_pct_string
        assert _normalize_pct_string("5.5") == 5.5

    def test_none_returns_none(self):
        from concilium.agents import _normalize_pct_string
        assert _normalize_pct_string(None) is None

    def test_empty_string_returns_none(self):
        from concilium.agents import _normalize_pct_string
        assert _normalize_pct_string("") is None

    def test_invalid_string_returns_none(self):
        from concilium.agents import _normalize_pct_string
        assert _normalize_pct_string("abc") is None


class TestRiskManagerNormalizesPctStrings:
    """Bug 5: risk_manager normalisiert max_drawdown_schaetzung und positionsgröße_empfohlen."""

    def test_string_values_normalized_to_float(self):
        """LLM liefert Strings → risk_manager macht floats daraus."""
        class _MockLLM:
            def chat(self, messages, temperature=0.3, **kwargs):
                return json.dumps({
                    "rolle": "Risk-Manager",
                    "risiko_score": 3,
                    "empfehlung": "MODIFIZIERT",
                    "max_drawdown_schaetzung": "5 %",
                    "positionsgröße_empfohlen": "3,5",
                })

        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {},
            "history": [],
            "sentiment": {},
        }
        result = risk_manager({"aktion": "HALTEN"}, data, _MockLLM(), data_text="dummy")
        assert result["max_drawdown_schaetzung"] == 5.0
        assert result["positionsgröße_empfohlen"] == 3.5
        assert isinstance(result["max_drawdown_schaetzung"], float)
        assert isinstance(result["positionsgröße_empfohlen"], float)

    def test_number_values_preserved(self):
        """LLM liefert bereits Zahlen → bleiben Zahlen."""
        class _MockLLM:
            def chat(self, messages, temperature=0.3, **kwargs):
                return json.dumps({
                    "rolle": "Risk-Manager",
                    "risiko_score": 3,
                    "empfehlung": "GENEHMIGT",
                    "max_drawdown_schaetzung": 7.5,
                    "positionsgröße_empfohlen": 4.0,
                })

        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {},
            "history": [],
            "sentiment": {},
        }
        result = risk_manager({"aktion": "KAUFEN"}, data, _MockLLM(), data_text="dummy")
        assert result["max_drawdown_schaetzung"] == 7.5
        assert result["positionsgröße_empfohlen"] == 4.0

    def test_none_values_stay_none(self):
        """Fehlende Werte → None."""
        class _MockLLM:
            def chat(self, messages, temperature=0.3, **kwargs):
                return json.dumps({
                    "rolle": "Risk-Manager",
                    "risiko_score": 3,
                    "empfehlung": "GENEHMIGT",
                })

        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {},
            "history": [],
            "sentiment": {},
        }
        result = risk_manager({"aktion": "HALTEN"}, data, _MockLLM(), data_text="dummy")
        assert result["max_drawdown_schaetzung"] is None
        assert result["positionsgröße_empfohlen"] is None
