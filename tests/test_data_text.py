"""Tests für _build_data_text einmalig — data_text wird durchgereicht, nicht neu berechnet.

Verifiziert, dass analyst_team, risk_manager und portfolio_fit_agent ein
übergebenes data_text nutzen statt _build_data_text neu aufzurufen.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import analyst_team, risk_manager  # noqa: E402
from concilium.portfolio_fit import portfolio_fit_agent  # noqa: E402

# --- Mock-LLM, der den user-prompt aufzeichnet ---

class _RecordingLLM:
    """Mock-LLM, der die messages aufzeichnet und JSON zurückgibt."""

    def __init__(self, response: str = '{"rolle": "Test", "stimmung": "neutral", "score": 3}'):
        self._response = response
        self.last_messages: list[dict] = []

    def chat(self, messages, temperature=0.3, **kwargs):
        self.last_messages = messages
        return self._response


_MOCK_DATA = {
    "ticker": "TEST",
    "fundamentals": {"name": "Test Inc.", "sector": "Tech"},
    "technicals": {"current_price": 100.0, "rsi14": 50.0},
    "sentiment": {"positiv": 1, "negativ": 0, "neutral": 2},
    "news": ["Normal headline"],
}


class TestAnalystTeamUsesDataText:
    """Test: analyst_team nutzt übergebenes data_text statt neu zu berechnen."""

    def test_uses_provided_data_text(self):
        """Ein Marker-String im data_text taucht im LLM-User-Prompt auf."""
        marker = "UNIQUE_MARKER_STRING_XYZ123"
        data_text = f"Fake data text with {marker}"

        llm = _RecordingLLM(json.dumps({
            "rolle": "Fundamental-Analyst",
            "stimmung": "bullish",
            "score": 4,
            "zusammenfassung": "Gut",
        }))

        # analyst_team mit explizitem data_text aufrufen
        analyst_team(_MOCK_DATA, llm, data_text=data_text)

        # Der Marker muss im User-Prompt der messages auftauchen
        # (für jeden der 3 Analysten)
        # Da analyst_team parallel läuft, nehmen wir die letzten messages
        user_msg = llm.last_messages[1]["content"]
        assert marker in user_msg, f"Marker '{marker}' nicht im User-Prompt gefunden"

    def test_does_not_call_build_data_text_when_provided(self):
        """Wenn data_text gegeben ist, wird _build_data_text NICHT aufgerufen."""
        data_text = "Pre-computed data text"

        llm = _RecordingLLM(json.dumps({
            "rolle": "Fundamental-Analyst",
            "stimmung": "bullish",
            "score": 4,
            "zusammenfassung": "Gut",
        }))

        with patch("concilium.agents._build_data_text") as mock_build:
            analyst_team(_MOCK_DATA, llm, data_text=data_text)
            mock_build.assert_not_called()

    def test_calls_build_data_text_when_none(self):
        """Wenn data_text=None ist, wird _build_data_text intern aufgerufen (Abwärtskompatibilität)."""
        llm = _RecordingLLM(json.dumps({
            "rolle": "Fundamental-Analyst",
            "stimmung": "bullish",
            "score": 4,
            "zusammenfassung": "Gut",
        }))

        with patch("concilium.agents._build_data_text", return_value="computed text") as mock_build:
            analyst_team(_MOCK_DATA, llm, data_text=None)
            mock_build.assert_called_once_with(_MOCK_DATA)


class TestRiskManagerUsesDataText:
    """Test: risk_manager nutzt übergebenes data_text."""

    def test_uses_provided_data_text(self):
        """Ein Marker-String im data_text taucht im LLM-User-Prompt auf."""
        marker = "RISK_MARKER_ABC789"
        data_text = f"Risk data with {marker}"
        trade = {"aktion": "KAUFEN", "zielkurs": 120, "stop_loss": 90}

        llm = _RecordingLLM(json.dumps({"rolle": "Risk-Manager", "risiko_score": 3,
                                         "empfehlung": "GENEHMIGT"}))

        risk_manager(trade, _MOCK_DATA, llm, data_text=data_text)

        user_msg = llm.last_messages[1]["content"]
        assert marker in user_msg

    def test_does_not_call_build_data_text_when_provided(self):
        """Wenn data_text gegeben ist, wird _build_data_text NICHT aufgerufen."""
        trade = {"aktion": "HALTEN"}
        llm = _RecordingLLM()

        with patch("concilium.agents._build_data_text") as mock_build:
            risk_manager(trade, _MOCK_DATA, llm, data_text="pre-computed")
            mock_build.assert_not_called()

    def test_calls_build_data_text_when_none(self):
        """Wenn data_text=None, wird _build_data_text aufgerufen."""
        trade = {"aktion": "HALTEN"}
        llm = _RecordingLLM()

        with patch("concilium.agents._build_data_text", return_value="computed") as mock_build:
            risk_manager(trade, _MOCK_DATA, llm, data_text=None)
            mock_build.assert_called_once_with(_MOCK_DATA)


class TestPortfolioFitUsesDataText:
    """Test: portfolio_fit_agent nutzt übergebenes data_text."""

    def test_uses_provided_data_text(self):
        """Ein Marker-String im data_text taucht im LLM-User-Prompt auf."""
        marker = "PFIT_MARKER_QWE456"
        data_text = f"Portfolio data with {marker}"

        llm = _RecordingLLM(json.dumps({"rolle": "Portfolio-Fit-Analyst",
                                         "portfolio_fit_score": 3}))
        positions = [{"name": "Test", "ticker": "TST", "depot_pct": 5.0}]

        portfolio_fit_agent(_MOCK_DATA, llm, positions, data_text=data_text)

        user_msg = llm.last_messages[1]["content"]
        assert marker in user_msg

    def test_does_not_call_build_data_text_when_provided(self):
        """Wenn data_text gegeben ist, wird _build_data_text NICHT aufgerufen."""
        llm = _RecordingLLM()
        positions = []

        with patch("concilium.portfolio_fit._build_data_text") as mock_build:
            portfolio_fit_agent(_MOCK_DATA, llm, positions, data_text="pre-computed")
            mock_build.assert_not_called()

    def test_calls_build_data_text_when_none(self):
        """Wenn data_text=None, wird _build_data_text aufgerufen."""
        llm = _RecordingLLM()
        positions = []

        with patch("concilium.portfolio_fit._build_data_text", return_value="computed") as mock_build:
            portfolio_fit_agent(_MOCK_DATA, llm, positions, data_text=None)
            mock_build.assert_called_once_with(_MOCK_DATA)
