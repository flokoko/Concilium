"""Tests für Phase 3: rechnerische Positionsgröße als harte Obergrenze.

Volatility-Targeting erzwingen: Der Risk-Manager berechnet eine
rechnerische Positionsgröße (``positionsgröße_rechnerisch_pct``), aber der
Trader/PM kann sie bisher ignorieren. Nach Risk-Parity-Praxis ist der
rechnerische Wert eine harte Obergrenze — der LLM darf nicht mehr Position
empfehlen, als das Volatility-Targeting erlaubt:

    _cap_position_by_volatility(trade, risk):
    - Nur KAUFEN/STARK KAUFEN (HALTEN/VERKAUFEN nie angetastet)
    - rechnerisch None/fehlt → Trade unverändert (rückwärtskompatibel)
    - positionsanteil > rechnerisch → gekappt (round 2), Metadaten _vol_cap
    - bereits kleiner (z. B. durch Technik-Signal) → kein Cap (gekappt=False)

Getestet werden:
- _cap_position_by_volatility: Kappen, Unterlauf, Grenzfall, fehlende Werte,
  ungültige Typen, NaN, Aktionen HALTEN/VERKAUFEN, Robustheit
- Pipeline-Wire-up: Cap läuft NACH dem Technik-Signal (5c' → 5c'') — das
  Signal skaliert zuerst, dann greift der Cap als Obergrenze; hat das
  Signal die Position bereits unter den Cap gesenkt, greift er nicht
- Report-Anzeige: Hinweis "⚠️ Positionsgröße auf rechnerisches
  Volatility-Targeting gekappt (X % statt Y %)" nur bei gekappt=True

Alle Tests sind OFFLINE-fähig: kein yfinance, kein Netzwerk, kein LLM.
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from concilium.agents import (  # noqa: E402
    _cap_position_by_volatility,
)
from concilium.pipeline import run_pipeline  # noqa: E402
from concilium.report import generate_report  # noqa: E402

# --------------------------------------------------------------------------- #
# Helfer
# --------------------------------------------------------------------------- #

_HINWEIS = (
    "⚠️ Positionsgröße auf rechnerisches Volatility-Targeting gekappt"
)


def _make_trade(
    aktion: str = "KAUFEN",
    positionsanteil: Any = 8.0,
) -> dict:
    return {
        "rolle": "Trader",
        "aktion": aktion,
        "rating": aktion,
        "zielkurs": 115,
        "stop_loss": 92,
        "positionsanteil": positionsanteil,
        "begründung": "Test",
        "zeithorizont": "Mittelfristig",
        "_raw": "",
    }


def _make_risk(rechnerisch: Any = 4.22) -> dict:
    return {
        "risiko_score": 3,
        "empfehlung": "GENEHMIGT",
        "positionsgröße_rechnerisch_pct": rechnerisch,
    }


def _analysts_mit_signal(
    *,
    current_price: float | None = 100.0,
    sma200: float | None = 90.0,
    sma50: float | None = 95.0,
    rsi: float | None = 45.0,
) -> dict:
    """Analysten-Dict mit technicals (Default: Kurs ÜBER SMA200 → kein Signal)."""
    technicals: dict = {}
    if current_price is not None:
        technicals["current_price"] = current_price
    if sma200 is not None:
        technicals["sma200"] = sma200
    if sma50 is not None:
        technicals["sma50"] = sma50
    if rsi is not None:
        technicals["rsi14"] = rsi
    return {
        "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
        "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
        "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
        "macro_news": {"_raw": ""},
        "technicals": technicals,
    }


# --------------------------------------------------------------------------- #
# Tests: _cap_position_by_volatility (Unit)
# --------------------------------------------------------------------------- #


class TestCapPositionByVolatility:
    """Der Cap greift nur für KAUFEN/STARK KAUFEN über dem rechnerischen Wert."""

    def test_kauf_wird_gekappt(self):
        """KAUFEN 8 % mit rechnerisch 4.22 % → auf 4.22 gekappt, gekappt=True."""
        trade = _make_trade("KAUFEN", positionsanteil=8)
        result = _cap_position_by_volatility(trade, _make_risk(4.22))
        assert result["positionsanteil"] == 4.22
        assert result["_vol_cap"]["gekappt"] is True
        assert result["_vol_cap"]["rechnerisch"] == 4.22
        assert result["_vol_cap"]["original"] == 8

    def test_stark_kaufen_wird_gekappt(self):
        """STARK KAUFEN wird genauso gekappt wie KAUFEN."""
        trade = _make_trade("STARK KAUFEN", positionsanteil=6.0)
        result = _cap_position_by_volatility(trade, _make_risk(2.5))
        assert result["positionsanteil"] == 2.5
        assert result["_vol_cap"]["gekappt"] is True
        assert result["_vol_cap"]["original"] == 6.0

    def test_kauf_unter_cap_unveraendert(self):
        """KAUFEN 3 % mit rechnerisch 4.22 % → unverändert (kein Cap nötig)."""
        trade = _make_trade("KAUFEN", positionsanteil=3)
        result = _cap_position_by_volatility(trade, _make_risk(4.22))
        assert result["positionsanteil"] == 3
        assert result["_vol_cap"]["gekappt"] is False
        assert result["_vol_cap"]["rechnerisch"] == 4.22
        assert result["_vol_cap"]["original"] == 3

    def test_kauf_genau_am_cap_unveraendert(self):
        """positionsanteil == rechnerisch → kein Cap (nur > wird gekappt)."""
        trade = _make_trade("KAUFEN", positionsanteil=4.22)
        result = _cap_position_by_volatility(trade, _make_risk(4.22))
        assert result["positionsanteil"] == 4.22
        assert result["_vol_cap"]["gekappt"] is False

    def test_halten_wird_nicht_gekappt(self):
        """HALTEN bleibt unangetastet — auch über dem rechnerischen Wert."""
        trade = _make_trade("HALTEN", positionsanteil=8)
        result = _cap_position_by_volatility(trade, _make_risk(4.22))
        assert result["positionsanteil"] == 8
        assert "_vol_cap" not in result

    def test_verkaufen_wird_nicht_gekappt(self):
        """VERKAUFEN bleibt unangetastet."""
        trade = _make_trade("VERKAUFEN", positionsanteil=8)
        result = _cap_position_by_volatility(trade, _make_risk(4.22))
        assert result["positionsanteil"] == 8
        assert "_vol_cap" not in result

    def test_aktion_gross_klein_leerzeichen_tolerant(self):
        """'Kaufen ' (Groß-/Kleinschreibung + Leerzeichen) wird erkannt."""
        trade = _make_trade("Kaufen ", positionsanteil=8)
        result = _cap_position_by_volatility(trade, _make_risk(4.22))
        assert result["positionsanteil"] == 4.22
        assert result["_vol_cap"]["gekappt"] is True

    def test_ohne_rechnerische_groesse_unveraendert(self):
        """Kein positionsgröße_rechnerisch_pct-Key → kein Cap, keine Metadaten."""
        trade = _make_trade("KAUFEN", positionsanteil=8)
        result = _cap_position_by_volatility(trade, {"risiko_score": 3})
        assert result["positionsanteil"] == 8
        assert "_vol_cap" not in result

    def test_rechnerisch_none_unveraendert(self):
        """positionsgröße_rechnerisch_pct=None → kein Cap (rückwärtskompatibel)."""
        trade = _make_trade("KAUFEN", positionsanteil=8)
        result = _cap_position_by_volatility(trade, _make_risk(None))
        assert result["positionsanteil"] == 8
        assert "_vol_cap" not in result

    def test_rechnerisch_ungueltig_unveraendert(self):
        """Nicht-numerischer rechnerischer Wert → kein Cap, kein Crash."""
        trade = _make_trade("KAUFEN", positionsanteil=8)
        result = _cap_position_by_volatility(trade, _make_risk("hoch"))
        assert result["positionsanteil"] == 8
        assert "_vol_cap" not in result

    def test_rechnerisch_nan_unveraendert(self):
        """NaN als rechnerischer Wert → kein Cap (NaN-Robustheit)."""
        trade = _make_trade("KAUFEN", positionsanteil=8)
        result = _cap_position_by_volatility(trade, _make_risk(float("nan")))
        assert result["positionsanteil"] == 8
        assert "_vol_cap" not in result

    def test_rechnerisch_als_int_tolerant(self):
        """Integer-Wert wird tolerant als float behandelt."""
        trade = _make_trade("KAUFEN", positionsanteil=8)
        result = _cap_position_by_volatility(trade, _make_risk(4))
        assert result["positionsanteil"] == 4.0
        assert result["_vol_cap"]["gekappt"] is True

    def test_rundung_auf_2_dezimalstellen(self):
        """Kapp-Wert wird auf 2 Dezimalstellen gerundet."""
        trade = _make_trade("KAUFEN", positionsanteil=8)
        result = _cap_position_by_volatility(trade, _make_risk(4.2266666))
        assert result["positionsanteil"] == 4.23
        assert result["_vol_cap"]["rechnerisch"] == 4.23

    def test_positionsanteil_none_kein_cap(self):
        """positionsanteil=None → nichts zu kappen, gekappt=False."""
        trade = _make_trade("KAUFEN", positionsanteil=None)
        result = _cap_position_by_volatility(trade, _make_risk(4.22))
        assert result["positionsanteil"] is None
        assert result["_vol_cap"]["gekappt"] is False
        assert result["_vol_cap"]["original"] is None

    def test_positionsanteil_ungueltig_kein_cap(self):
        """Nicht-numerischer positionsanteil → keine Datenverfälschung, kein Cap."""
        trade = _make_trade("KAUFEN", positionsanteil="hoch")
        result = _cap_position_by_volatility(trade, _make_risk(4.22))
        assert result["positionsanteil"] == "hoch"
        assert result["_vol_cap"]["gekappt"] is False
        assert result["_vol_cap"]["original"] is None

    def test_risk_none_crasht_nicht(self):
        """risk=None → kein Crash, Trade unverändert."""
        trade = _make_trade("KAUFEN", positionsanteil=8)
        result = _cap_position_by_volatility(trade, None)
        assert result["positionsanteil"] == 8
        assert "_vol_cap" not in result

    def test_leerer_trade_crasht_nicht(self):
        """Leeres Trade-dict → kein Crash, unverändert."""
        trade: dict = {}
        result = _cap_position_by_volatility(trade, _make_risk(4.22))
        assert result == {}
        assert "_vol_cap" not in result

    def test_risk_ohne_dict_crasht_nicht(self):
        """risk als nicht-dict → kein Crash, Trade unverändert."""
        trade = _make_trade("KAUFEN", positionsanteil=8)
        result = _cap_position_by_volatility(trade, "kaputt")
        assert result["positionsanteil"] == 8
        assert "_vol_cap" not in result


# --------------------------------------------------------------------------- #
# Mock-Daten für den Pipeline-Wire-up (analog test_pipeline_dampen_order.py)
# --------------------------------------------------------------------------- #

_MOCK_DATA = {
    "ticker": "TEST",
    "fundamentals": {"name": "TestCo", "sector": "Tech"},
    "technicals": {"current_price": 100},
    "sentiment": {},
    "news": [],
}

_MOCK_DEBATE = {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}}

_MOCK_FINAL = {"entscheidung": "GENEHMIGT", "confidence": 4, "begründung": "Ok."}


def _make_llm() -> MagicMock:
    """LLM-Mock mit leerem total_usage (verhindert usage/usage.csv-Einträge)."""
    llm = MagicMock()
    llm.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return llm


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Isoliertes state-Verzeichnis (CONCILIUM_STATE_DIR auf tmp_path)."""
    d = tmp_path / "state"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CONCILIUM_STATE_DIR", str(d))
    return str(d)


def _run_pipeline_mit_vol_cap(
    trade: dict,
    risk: dict,
    analysts: dict,
) -> dict:
    """Komplett gemockter Pipeline-Lauf (offline) mit echtem Technik-Signal + Vol-Cap."""
    patches = {
        "collect_ticker_data": MagicMock(return_value=_MOCK_DATA),
        "analyst_team": MagicMock(return_value=analysts),
        "debate": MagicMock(return_value=_MOCK_DEBATE),
        "trader": MagicMock(return_value=trade),
        "ensemble_trader": MagicMock(return_value=trade),
        "risk_manager": MagicMock(return_value=risk),
        "fetch_portfolio_positions": MagicMock(return_value=[]),
        "portfolio_fit_agent": MagicMock(return_value=None),
        "trade_revision": MagicMock(return_value=trade),
        "portfolio_manager": MagicMock(return_value=_MOCK_FINAL),
        "build_feedback_context": MagicMock(return_value=""),
        "build_reflection_context": MagicMock(return_value=""),
    }
    with patch.multiple("concilium.pipeline", **patches), patch(
        "concilium.journal.append_decision"
    ):
        return run_pipeline("TEST", llm=_make_llm(), ensemble=False, resume=False)


class TestVolCapPipelineWireUp:
    """Der Vol-Cap greift in der Pipeline NACH dem Technik-Signal (5c' → 5c'')."""

    def test_pipeline_kappt_kauf_ueber_rechnerischer_groesse(self, state_dir):
        """KAUFEN 8 %, rechnerisch 4.22 %, kein Technik-Signal → gekappt auf 4.22."""
        trade = _make_trade("KAUFEN", positionsanteil=8)
        result = _run_pipeline_mit_vol_cap(
            trade, _make_risk(4.22), _analysts_mit_signal()
        )
        assert result["trade"]["aktion"] == "KAUFEN"
        assert result["trade"]["positionsanteil"] == 4.22
        assert result["trade"]["_vol_cap"]["gekappt"] is True
        assert result["trade"]["_vol_cap"]["original"] == 8
        assert result["trade"]["_vol_cap"]["rechnerisch"] == 4.22

    def test_reihenfolge_technik_signal_dann_vol_cap(self, state_dir):
        """Technik-Signal skaliert zuerst (8 → 2.4), Vol-Cap (3.0) greift nicht mehr.

        Kurs 90 um 10 % unter SMA200 → Faktor 0.3 → Position 8 * 0.3 = 2.4.
        Wäre der Cap VOR dem Signal angewendet worden, wäre die Position auf
        3.0 gekappt und dann mit 0.3 auf 0.9 skaliert worden — nach dem Fix
        muss 2.4 übrig bleiben (Cap senkt nur, hebt aber nie an).
        """
        trade = _make_trade("KAUFEN", positionsanteil=8)
        result = _run_pipeline_mit_vol_cap(
            trade,
            _make_risk(3.0),
            _analysts_mit_signal(current_price=90.0, sma200=100.0, sma50=85.0),
        )
        assert result["trade"]["positionsanteil"] == 2.4
        assert result["trade"]["_technik_signal"]["faktor"] == 0.3
        assert result["trade"]["_vol_cap"]["gekappt"] is False
        assert result["trade"]["_vol_cap"]["original"] == 2.4

    def test_vol_cap_greift_nach_signal_wenn_noch_zu_gross(self, state_dir):
        """Signal (8 → 4.8) senkt, aber Cap (3.0) kappt trotzdem darunter."""
        trade = _make_trade("KAUFEN", positionsanteil=8)
        result = _run_pipeline_mit_vol_cap(
            trade,
            _make_risk(3.0),
            _analysts_mit_signal(current_price=95.0, sma200=100.0, sma50=85.0),
        )
        # Abstand 5 % unter SMA200 → Faktor 1.0 - 5/10 = 0.5 → 8 * 0.5 = 4.0;
        # Cap 3.0 < 4.0 → gekappt auf 3.0.
        assert result["trade"]["positionsanteil"] == 3.0
        assert result["trade"]["_vol_cap"]["gekappt"] is True
        assert result["trade"]["_vol_cap"]["original"] == 4.0

    def test_ohne_rechnerische_groesse_kein_cap(self, state_dir):
        """Rückwärtskompatibel: risk ohne rechnerische Größe → Position unverändert."""
        trade = _make_trade("KAUFEN", positionsanteil=8)
        result = _run_pipeline_mit_vol_cap(
            trade, _make_risk(None), _analysts_mit_signal()
        )
        assert result["trade"]["positionsanteil"] == 8
        assert "_vol_cap" not in result["trade"]

    def test_halten_wird_nie_gekappt(self, state_dir):
        """HALTEN läuft durch die Pipeline, ohne dass der Cap anfasst."""
        trade = _make_trade("HALTEN", positionsanteil=8)
        result = _run_pipeline_mit_vol_cap(
            trade, _make_risk(4.22), _analysts_mit_signal()
        )
        assert result["trade"]["aktion"] == "HALTEN"
        assert result["trade"]["positionsanteil"] == 8
        assert "_vol_cap" not in result["trade"]


# --------------------------------------------------------------------------- #
# Tests: Report-Anzeige (Kapp-Hinweis)
# --------------------------------------------------------------------------- #


def _full_result(trade: dict) -> dict:
    """Baut das vollständige result-dict für generate_report (offline)."""
    return {
        "ticker": "AAPL",
        "no_llm": False,
        "data": {
            "fundamentals": {"name": "Apple", "sector": "Tech"},
            "technicals": {"current_price": 150},
            "sentiment": {},
            "news": [],
        },
        "analysts": {
            "fundamental": {"stimmung": "bullish", "score": 4,
                            "zusammenfassung": "Gut", "_raw": ""},
            "technical": {"stimmung": "bullish", "score": 4,
                          "zusammenfassung": "Gut", "_raw": ""},
            "sentiment": {"stimmung": "neutral", "score": 3,
                          "zusammenfassung": "Ok", "_raw": ""},
        },
        "debate": {
            "bull": {"_raw": "Bull"},
            "bear": {"_raw": "Bear"},
            "bull_confidence": 5,
            "bear_confidence": 4,
        },
        "trade": trade,
        "risk": {
            "risiko_score": 3,
            "empfehlung": "GENEHMIGT",
            "auflagen": "keine",
            "positionsgröße_rechnerisch_pct": 4.22,
        },
        "portfolio_fit": None,
        "final": {
            "entscheidung": "GENEHMIGT",
            "confidence": 4,
            "begründung": "Solide Fundamentals.",
        },
    }


class TestVolCapReportAnzeige:
    """Der Report zeigt den Kapp-Hinweis nur bei tatsächlichem Kap."""

    def test_gekappt_zeigt_hinweis(self):
        """Bei gekappt=True: Hinweis mit neuem und originalen Wert."""
        trade = _make_trade("KAUFEN", positionsanteil=4.22)
        trade["_vol_cap"] = {
            "rechnerisch": 4.22,
            "original": 8.0,
            "gekappt": True,
        }
        report = generate_report(_full_result(trade))
        assert (
            f"> {_HINWEIS} (4.22 % statt 8.0 %)." in report
        )

    def test_ohne_cap_kein_hinweis(self):
        """Ohne _vol_cap-Metadaten: kein Hinweis im Report."""
        trade = _make_trade("KAUFEN", positionsanteil=8)
        report = generate_report(_full_result(trade))
        assert _HINWEIS not in report

    def test_gekappt_false_kein_hinweis(self):
        """Bei gekappt=False (Position schon unter dem Cap): kein Hinweis."""
        trade = _make_trade("KAUFEN", positionsanteil=3)
        trade["_vol_cap"] = {
            "rechnerisch": 4.22,
            "original": 3.0,
            "gekappt": False,
        }
        report = generate_report(_full_result(trade))
        assert _HINWEIS not in report

    def test_original_none_kein_hinweis(self):
        """Undarstellbare Werte (original=None) → kein Hinweis statt Crash."""
        trade = _make_trade("KAUFEN", positionsanteil=4.22)
        trade["_vol_cap"] = {
            "rechnerisch": 4.22,
            "original": None,
            "gekappt": True,
        }
        report = generate_report(_full_result(trade))
        assert _HINWEIS not in report

    def test_vol_cap_als_nicht_dict_ignoriert(self):
        """_vol_cap als nicht-dict → kein Hinweis, kein Crash."""
        trade = _make_trade("KAUFEN", positionsanteil=4.22)
        trade["_vol_cap"] = "kaputt"
        report = generate_report(_full_result(trade))
        assert _HINWEIS not in report
