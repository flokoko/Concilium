"""Tests für portfolio_fit.py — Portfolio-Fit-Baustein.

Alle Tests sind OFFLINE-fähig (kein echtes Netzwerk, urllib wird gemockt).
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.portfolio_fit import (  # noqa: E402
    _parse_positions,
    _safe_float,
    fetch_portfolio_positions,
    portfolio_fit_agent,
)

# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    """Testet _safe_float mit deutschen Zahlen (Komma-Dezimaltrenner)."""

    def test_comma_decimal(self):
        assert _safe_float("34768,88607") == 34768.88607

    def test_simple_comma(self):
        assert _safe_float("4,5") == 4.5

    def test_integer_string(self):
        assert _safe_float("100") == 100.0

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_none(self):
        assert _safe_float(None) is None

    def test_thousands_separator(self):
        """Tausenderpunkte werden entfernt, Komma bleibt Dezimaltrenner."""
        assert _safe_float("1.234,56") == 1234.56

    def test_invalid(self):
        assert _safe_float("abc") is None


# ---------------------------------------------------------------------------
# _parse_positions (CSV-Parsing, offline)
# ---------------------------------------------------------------------------

# Dummy-CSV im Format des Google-Sheets (deutsches Locale, Komma als Feldtrenner,
# Werte mit Komma als Dezimaltrenner müssen gequotet sein)
_DUMMY_CSV = (
    'Bestand,Name,Symbol,Kurs,Marktwert,Anteil in %,Region\n'
    '10,Apple,AAPL,"150,00","1500,00","4,50",USA\n'
    '5,Microsoft,MSFT,"300,00","1500,00","4,50",USA\n'
    '8,iShares Core MSCI World,IUSQ.DE,"120,00","960,00","3,20",ETF\n'
    ',Summen,,"34000,00","100,00",\n'
    '2,Infineon,IFX.DE,"40,00","80,00","0,24",Deutschland\n'
    '3,Goldman Sachs,IS3R.TG,"200,00","600,00","1,80",USA\n'
)

# CSV mit Semikolon-Trenner (alternative Test-Daten)
_DUMMY_CSV_SEMI = (
    'Bestand;Name;Symbol;Kurs;Marktwert;Anteil in %;Region\n'
    '10;Apple;AAPL;"150,00";"1500,00";"4,50";USA\n'
    ';Summen;;"34000,00";"100,00";\n'
    '5;Microsoft;MSFT;"300,00";"1500,00";"4,50";USA\n'
)


class TestParsePositions:
    """Testet _parse_positions mit Dummy-CSV (offline)."""

    def test_parses_valid_positions(self):
        """Valid CSV → Positionen werden geparst, Gruppierungszeile übersprungen."""
        positions = _parse_positions(_DUMMY_CSV)
        # 5 Datenzeilen (Summenzeile mit leerem Bestand übersprungen)
        assert len(positions) == 5

    def test_grouping_row_skipped(self):
        """Zeile mit leerem Bestand (Summenzeile) wird übersprungen."""
        positions = _parse_positions(_DUMMY_CSV)
        names = [p["name"] for p in positions]
        assert "Summen" not in names

    def test_depot_pct_parsed(self):
        """Anteil in % wird als Float geparst (Komma-Dezimal)."""
        positions = _parse_positions(_DUMMY_CSV)
        apple = next(p for p in positions if p["name"] == "Apple")
        assert apple["depot_pct"] == 4.50

    def test_ticker_fix_applied(self):
        """TICKER_FIX wird angewendet: IS3R.TG → IS3R.DE."""
        positions = _parse_positions(_DUMMY_CSV)
        goldman = next(p for p in positions if p["sheet_symbol"] == "IS3R.TG")
        assert goldman["ticker"] == "IS3R.DE"
        assert goldman["sheet_symbol"] == "IS3R.TG"

    def test_ticker_fix_passthrough(self):
        """Normale Ticker bleiben unverändert."""
        positions = _parse_positions(_DUMMY_CSV)
        apple = next(p for p in positions if p["name"] == "Apple")
        assert apple["ticker"] == "AAPL"
        assert apple["sheet_symbol"] == "AAPL"

    def test_value_eur_parsed(self):
        """Marktwert wird als Float geparst (kann None sein)."""
        positions = _parse_positions(_DUMMY_CSV)
        apple = next(p for p in positions if p["name"] == "Apple")
        # Marktwert-Spalte: "1500,00" → 1500.0
        assert apple["value_eur"] == 1500.0

    def test_empty_csv(self):
        """Leerer CSV-Text → leere Liste."""
        assert _parse_positions("") == []

    def test_header_only(self):
        """Nur Header-Zeile → leere Liste."""
        csv_text = "Bestand,Name,Symbol,Kurs,Marktwert,Anteil in %,Region\n"
        assert _parse_positions(csv_text) == []

    def test_dict_fields_present(self):
        """Jede Position hat alle erwarteten Felder."""
        positions = _parse_positions(_DUMMY_CSV)
        for p in positions:
            assert "name" in p
            assert "ticker" in p
            assert "sheet_symbol" in p
            assert "type" in p
            assert "region" in p
            assert "depot_pct" in p
            assert "value_eur" in p


# ---------------------------------------------------------------------------
# fetch_portfolio_positions (Netzwerk-Fehlerfall, offline)
# ---------------------------------------------------------------------------


class TestFetchPortfolioPositions:
    """Testet fetch_portfolio_positions — Fehlerfälle (offline)."""

    def test_network_error_returns_empty(self):
        """Bei urllib-Fehler → leere Liste, kein Crash."""
        with patch(
            "concilium.portfolio_fit.urllib.request.urlopen",
            side_effect=ConnectionError("DNS failed"),
        ):
            result = fetch_portfolio_positions()
        assert result == []

    def test_timeout_returns_empty(self):
        """Bei Timeout → leere Liste, kein Crash."""
        with patch(
            "concilium.portfolio_fit.urllib.request.urlopen",
            side_effect=TimeoutError("Connection timed out"),
        ):
            result = fetch_portfolio_positions()
        assert result == []

    def test_valid_csv_returns_positions(self):
        """Bei gültigem CSV-Response → Positionen werden zurückgegeben."""
        mock_resp = _MockHTTPResponse(_DUMMY_CSV)
        with patch(
            "concilium.portfolio_fit.urllib.request.urlopen",
            return_value=mock_resp,
        ):
            result = fetch_portfolio_positions()
        assert len(result) == 5
        assert result[0]["name"] == "Apple"

    def test_empty_csv_returns_empty(self):
        """Leerer CSV-Body → leere Liste."""
        mock_resp = _MockHTTPResponse("Bestand,Name,Symbol\n")
        with patch(
            "concilium.portfolio_fit.urllib.request.urlopen",
            return_value=mock_resp,
        ):
            result = fetch_portfolio_positions()
        assert result == []


class _MockHTTPResponse:
    """Mock für urllib.request.urlopen Response-Objekt."""

    def __init__(self, text: str):
        self._data = text.encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# portfolio_fit_agent (mit Fake-LLM, offline)
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Mock-LLM, der eine vordefinierte JSON-Antwort zurückgibt."""

    def __init__(self, response: str):
        self._response = response
        self.last_messages: list[dict] = []

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3, **kwargs) -> str:
        self.last_messages = messages
        return self._response


_PORTFOLIO_FIT_JSON = json.dumps({
    "rolle": "Portfolio-Fit-Analyst",
    "konzentrationsrisiko_bewertung": "Apple ist mit 4,5% bereits stark gewichtet.",
    "sektor_overlap_bewertung": "Technologie-Sektor bereits mit 9% vertreten.",
    "portfolio_fit_score": 2,
    "ziel_gewichtung_pct": 2.0,
    "begründung": "Hoher Sektor-Overlap und Konzentrationsrisiko. Ziel-Gewichtung 2%.",
})

_MOCK_DATA = {
    "ticker": "AAPL",
    "fundamentals": {
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 3e12,
        "beta": 1.2,
    },
    "technicals": {"current_price": 150.0},
    "sentiment": {"positiv": 5, "negativ": 1, "neutral": 3},
    "news": [],
}

_MOCK_POSITIONS = [
    {"name": "Apple", "ticker": "AAPL", "sheet_symbol": "AAPL", "type": "Aktie",
     "region": "USA", "depot_pct": 4.5, "value_eur": 1500.0},
    {"name": "Microsoft", "ticker": "MSFT", "sheet_symbol": "MSFT", "type": "Aktie",
     "region": "USA", "depot_pct": 4.5, "value_eur": 1500.0},
    {"name": "iShares MSCI World", "ticker": "IUSQ.DE", "sheet_symbol": "IUSQ.DE",
     "type": "ETF", "region": "ETF", "depot_pct": 3.2, "value_eur": 960.0},
]


class TestPortfolioFitAgent:
    """Testet portfolio_fit_agent mit Fake-LLM (offline)."""

    def test_returns_parsed_json(self):
        """Agent gibt geparstes JSON mit korrekten Feldern zurück."""
        llm = _FakeLLM(_PORTFOLIO_FIT_JSON)
        result = portfolio_fit_agent(_MOCK_DATA, llm, _MOCK_POSITIONS)

        assert result["rolle"] == "Portfolio-Fit-Analyst"
        assert result["portfolio_fit_score"] == 2
        assert result["ziel_gewichtung_pct"] == 2.0
        assert "konzentrationsrisiko_bewertung" in result
        assert "sektor_overlap_bewertung" in result
        assert "begründung" in result

    def test_portfolio_daten_verfuegbar_true(self):
        """Bei nicht-leeren positions → portfolio_daten_verfuegbar = True."""
        llm = _FakeLLM(_PORTFOLIO_FIT_JSON)
        result = portfolio_fit_agent(_MOCK_DATA, llm, _MOCK_POSITIONS)
        assert result["portfolio_daten_verfuegbar"] is True

    def test_portfolio_daten_verfuegbar_false(self):
        """Bei leeren positions → portfolio_daten_verfuegbar = False."""
        llm = _FakeLLM(_PORTFOLIO_FIT_JSON)
        result = portfolio_fit_agent(_MOCK_DATA, llm, [])
        assert result["portfolio_daten_verfuegbar"] is False

    def test_prompt_contains_portfolio_text(self):
        """Der User-Prompt enthält den Portfolio-Text (Positionsnamen)."""
        llm = _FakeLLM(_PORTFOLIO_FIT_JSON)
        portfolio_fit_agent(_MOCK_DATA, llm, _MOCK_POSITIONS)

        # user message ist die zweite im messages-Array
        user_msg = llm.last_messages[1]["content"]
        assert "Apple" in user_msg
        assert "Microsoft" in user_msg
        assert "Depot:" in user_msg

    def test_prompt_contains_data_text(self):
        """Der User-Prompt enthält den Unternehmens-Text (Sektor/Ticker)."""
        llm = _FakeLLM(_PORTFOLIO_FIT_JSON)
        portfolio_fit_agent(_MOCK_DATA, llm, _MOCK_POSITIONS)

        user_msg = llm.last_messages[1]["content"]
        assert "AAPL" in user_msg
        assert "Technology" in user_msg

    def test_prompt_system_prompt_correct(self):
        """Der System-Prompt ist der Portfolio-Fit-Prompt."""
        llm = _FakeLLM(_PORTFOLIO_FIT_JSON)
        portfolio_fit_agent(_MOCK_DATA, llm, _MOCK_POSITIONS)

        system_msg = llm.last_messages[0]["content"]
        assert "Portfolio-Fit-Analyst" in system_msg
        assert "Konzentrationsrisiko" in system_msg
        assert "Sektor-/Branchen-Overlap" in system_msg

    def test_empty_positions_still_works(self):
        """Bei leeren positions crasht der Agent nicht und liefert ein Ergebnis."""
        llm = _FakeLLM(_PORTFOLIO_FIT_JSON)
        result = portfolio_fit_agent(_MOCK_DATA, llm, [])

        assert "rolle" in result
        assert result["portfolio_daten_verfuegbar"] is False

    def test_raw_in_result(self):
        """Das _raw-Feld ist vorhanden (von _call_agent)."""
        llm = _FakeLLM(_PORTFOLIO_FIT_JSON)
        result = portfolio_fit_agent(_MOCK_DATA, llm, _MOCK_POSITIONS)
        assert "_raw" in result
