"""Tests für Feature C: Sektor-Summary im Portfolio-Fit (_build_portfolio_summary).

Alle Tests sind offline (kein Netzwerk).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.portfolio_fit import (  # noqa: E402
    _build_portfolio_summary,
    _build_portfolio_text,
    portfolio_fit_agent,
)

# --------------------------------------------------------------------------- #
# Test-Positionen
# --------------------------------------------------------------------------- #

_POSITIONS = [
    {"name": "Apple", "ticker": "AAPL", "type": "Aktie", "region": "USA", "depot_pct": 4.5},
    {"name": "Microsoft", "ticker": "MSFT", "type": "Aktie", "region": "USA", "depot_pct": 4.5},
    {"name": "iShares Core MSCI World", "ticker": "IUSQ.DE", "type": "ETF", "region": "Welt", "depot_pct": 10.0},
    {"name": "Infineon", "ticker": "IFX.DE", "type": "Aktie", "region": "Deutschland", "depot_pct": 2.0},
    {"name": "Goldman Sachs", "ticker": "GS", "type": "Aktie", "region": "USA", "depot_pct": 1.8},
    {"name": "Physical Gold", "ticker": "SGLN.DE", "type": "Commodity", "region": "Global", "depot_pct": 3.0},
    {"name": "iShares MSCI EM", "ticker": "IS3R.DE", "type": "ETF", "region": "Emerging", "depot_pct": 2.5},
    {"name": "SAP", "ticker": "SAP.DE", "type": "Aktie", "region": "Deutschland", "depot_pct": 1.5},
    {"name": "iShares STOXX 600", "ticker": "EXSA.DE", "type": "ETF", "region": "EU", "depot_pct": 5.0},
    {"name": "Cocoa", "ticker": "COCO.L", "type": "Commodity", "region": "Global", "depot_pct": 0.8},
    {"name": "Nvidia", "ticker": "NVDA", "type": "Aktie", "region": "USA", "depot_pct": 3.2},
    {"name": "Allianz", "ticker": "ALV.DE", "type": "Aktie", "region": "Deutschland", "depot_pct": 1.0},
]


class TestBuildPortfolioSummaryTypes:
    """Testet die Typen-Aggregation (Aktie/ETF/Commodity)."""

    def test_returns_string(self):
        """_build_portfolio_summary liefert einen String."""
        result = _build_portfolio_summary(_POSITIONS)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_typen_mix_line(self):
        """Ergebnis enthält 'Portfolio-Typen-Mix'."""
        result = _build_portfolio_summary(_POSITIONS)
        assert "Portfolio-Typen-Mix" in result

    def test_aktie_summed(self):
        """Aktie-Anteil wird korrekt aggregiert."""
        result = _build_portfolio_summary(_POSITIONS)
        # Aktie: 4.5 + 4.5 + 2.0 + 1.8 + 1.5 + 3.2 + 1.0 = 18.5
        assert "Aktie 18.5%" in result

    def test_etf_summed(self):
        """ETF-Anteil wird korrekt aggregiert."""
        result = _build_portfolio_summary(_POSITIONS)
        # ETF: 10.0 + 2.5 + 5.0 = 17.5
        assert "ETF 17.5%" in result

    def test_commodity_summed(self):
        """Commodity-Anteil wird korrekt aggregiert."""
        result = _build_portfolio_summary(_POSITIONS)
        # Commodity: 3.0 + 0.8 = 3.8
        assert "Commodity 3.8%" in result

    def test_types_sorted_by_size(self):
        """Typen sind nach Anteil absteigend sortiert."""
        result = _build_portfolio_summary(_POSITIONS)
        # Aktie 18.5 > ETF 17.5 > Commodity 3.8
        aktie_pos = result.index("Aktie")
        etf_pos = result.index("ETF")
        commodity_pos = result.index("Commodity")
        assert aktie_pos < etf_pos < commodity_pos

    def test_only_aktie_etf_commodity_types(self):
        """Nur Aktie/ETF/Commodity-Typen erscheinen (keine unbekannten)."""
        result = _build_portfolio_summary(_POSITIONS)
        # Alle drei Typen müssen im Ergebnis sein
        assert "Aktie" in result
        assert "ETF" in result
        assert "Commodity" in result


class TestBuildPortfolioSummaryRegions:
    """Testet die Regionen-Aggregation (Top 5)."""

    def test_contains_regionen_mix_line(self):
        """Ergebnis enthält 'Regionen-Mix'."""
        result = _build_portfolio_summary(_POSITIONS)
        assert "Regionen-Mix" in result

    def test_usa_summed(self):
        """USA-Anteil wird korrekt aggregiert."""
        result = _build_portfolio_summary(_POSITIONS)
        # USA: 4.5 + 4.5 + 1.8 + 3.2 = 14.0
        assert "USA 14.0%" in result

    def test_deutschland_summed(self):
        """Deutschland-Anteil wird korrekt aggregiert."""
        result = _build_portfolio_summary(_POSITIONS)
        # Deutschland: 2.0 + 1.5 + 1.0 = 4.5
        assert "Deutschland 4.5%" in result

    def test_etf_region_summed(self):
        """ETF-Region wird korrekt aggregiert."""
        result = _build_portfolio_summary(_POSITIONS)
        # Welt: 10.0
        assert "Welt 10.0%" in result

    def test_top_5_regions(self):
        """Es werden maximal 5 Regionen gezeigt."""
        result = _build_portfolio_summary(_POSITIONS)
        # Zähle die Region-Einträge im Regionen-Mix-Teil
        regionen_line = result.split("Regionen-Mix:")[1].strip()
        # Split by ", " — jede Region ist ein Eintrag
        entries = [e for e in regionen_line.split(", ") if e.strip()]
        assert len(entries) <= 5

    def test_regions_sorted_by_size(self):
        """Regionen sind nach Anteil absteigend sortiert (Top 5)."""
        result = _build_portfolio_summary(_POSITIONS)
        regionen_line = result.split("Regionen-Mix:")[1].strip()
        entries = [e for e in regionen_line.split(", ") if e.strip()]
        # Extrahiere Prozentwerte
        pcts = []
        for e in entries:
            try:
                pct_str = e.split()[-1].replace("%", "")
                pcts.append(float(pct_str))
            except (ValueError, IndexError):
                pass
        if len(pcts) >= 2:
            assert pcts == sorted(pcts, reverse=True)

    def test_global_region_summed(self):
        """Global-Anteil (Commodity) wird korrekt aggregiert."""
        result = _build_portfolio_summary(_POSITIONS)
        # Global: 3.0 + 0.8 = 3.8
        assert "Global 3.8%" in result


class TestBuildPortfolioSummaryEdgeCases:
    """Testet Edge Cases für _build_portfolio_summary."""

    def test_empty_positions_returns_empty(self):
        """Bei leeren Positionen → leerer String."""
        assert _build_portfolio_summary([]) == ""

    def test_single_position(self):
        """Eine einzelne Position wird korrekt aggregiert."""
        positions = [{"name": "X", "type": "Aktie", "region": "DE", "depot_pct": 5.0}]
        result = _build_portfolio_summary(positions)
        assert "Aktie 5.0%" in result
        assert "DE 5.0%" in result

    def test_missing_type_defaults_to_unbekannt(self):
        """Fehlender type → 'Unbekannt'."""
        positions = [{"name": "X", "region": "DE", "depot_pct": 5.0}]
        result = _build_portfolio_summary(positions)
        assert "Unbekannt 5.0%" in result

    def test_missing_region_defaults_to_unbekannt(self):
        """Fehlender region → 'Unbekannt'."""
        positions = [{"name": "X", "type": "Aktie", "depot_pct": 5.0}]
        result = _build_portfolio_summary(positions)
        assert "Unbekannt 5.0%" in result

    def test_empty_region_defaults_to_unbekannt(self):
        """Leerer region-String → 'Unbekannt'."""
        positions = [{"name": "X", "type": "Aktie", "region": "", "depot_pct": 5.0}]
        result = _build_portfolio_summary(positions)
        assert "Unbekannt 5.0%" in result

    def test_missing_depot_pct_defaults_to_zero(self):
        """Fehlender depot_pct → 0.0."""
        positions = [{"name": "X", "type": "Aktie", "region": "DE"}]
        result = _build_portfolio_summary(positions)
        assert "Aktie 0.0%" in result

    def test_more_than_5_regions_shows_only_top5(self):
        """Bei >5 Regionen werden nur Top 5 gezeigt."""
        positions = [
            {"name": f"P{i}", "type": "Aktie", "region": f"R{i}", "depot_pct": float(10 - i)}
            for i in range(8)
        ]
        result = _build_portfolio_summary(positions)
        regionen_line = result.split("Regionen-Mix:")[1].strip()
        entries = [e for e in regionen_line.split(", ") if e.strip()]
        assert len(entries) == 5
        # R0 (10.0) ist die größte
        assert "R0" in entries[0]


class TestBuildPortfolioTextWithSummary:
    """Testet dass _build_portfolio_text die Summary enthält."""

    def test_text_contains_typen_mix(self):
        """_build_portfolio_text enthält die Typen-Mix-Zeile."""
        text = _build_portfolio_text(_POSITIONS)
        assert "Portfolio-Typen-Mix" in text

    def test_text_contains_regionen_mix(self):
        """_build_portfolio_text enthält die Regionen-Mix-Zeile."""
        text = _build_portfolio_text(_POSITIONS)
        assert "Regionen-Mix" in text

    def test_text_contains_top10(self):
        """_build_portfolio_text enthält weiterhin die Top-10-Liste."""
        text = _build_portfolio_text(_POSITIONS)
        assert "Größte Positionen (Top 10)" in text
        assert "iShares Core MSCI World" in text  # 10.0% = größte

    def test_text_contains_total_count(self):
        """_build_portfolio_text enthält die Gesamtanzahl."""
        text = _build_portfolio_text(_POSITIONS)
        assert "12 Positionen insgesamt" in text

    def test_summary_before_top10(self):
        """Summary erscheint VOR der Top-10-Liste."""
        text = _build_portfolio_text(_POSITIONS)
        summary_pos = text.index("Portfolio-Typen-Mix")
        top10_pos = text.index("Größte Positionen")
        assert summary_pos < top10_pos

    def test_empty_positions_returns_placeholder(self):
        """Bei leeren Positionen → 'Keine Portfolio-Daten verfügbar.'"""
        assert _build_portfolio_text([]) == "Keine Portfolio-Daten verfügbar."


# --------------------------------------------------------------------------- #
# Integration: portfolio_fit_agent Prompt enthält Summary
# --------------------------------------------------------------------------- #


class TestPortfolioFitAgentPromptWithSummary:
    """Testet dass der portfolio_fit_agent-Prompt die Summary enthält."""

    def test_prompt_contains_typen_mix(self):
        """Der User-Prompt des Portfolio-Fit-Agenten enthält den Typen-Mix."""
        import json

        class _FakeLLM:
            def __init__(self):
                self.last_messages: list[dict] = []

            def chat(self, messages, temperature=0.3, **kwargs):
                self.last_messages = messages
                return json.dumps({"rolle": "Portfolio-Fit-Analyst", "portfolio_fit_score": 3})

        llm = _FakeLLM()
        data = {
            "ticker": "TEST",
            "fundamentals": {"name": "Test", "sector": "Tech"},
            "technicals": {"current_price": 100},
            "sentiment": {},
        }
        portfolio_fit_agent(data, llm, _POSITIONS, data_text="dummy")

        user_msg = llm.last_messages[1]["content"]
        assert "Portfolio-Typen-Mix" in user_msg
        assert "Regionen-Mix" in user_msg
