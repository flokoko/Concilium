"""Tests für das Technik-Veto (SMA200) — Option 1.

Ein KAUFEN/STARK KAUFEN soll nicht mehr möglich sein, wenn der Kurs unter
dem SMA200 liegt (fallendes Messer). Ausnahme: RSI < 30 bei intaktem SMA200
(übergeordneter Aufwärtstrend) erlaubt eine kleine Position mit strengem Stop.

Alle Tests sind offline (kein Netzwerk) — der LLMClient wird gemockt.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from concilium.agents import (  # noqa: E402
    _technik_veto,
    ensemble_trader,
    trader,
)

# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #


def _analysts(
    current_price: float | None = 90.0,
    sma200: float | None = 100.0,
    sma50: float | None = 95.0,
    rsi: float | None = 45.0,
) -> dict:
    """Analysten-Dict mit technicals-Snapshot (Kurs UNTER SMA200 per Default)."""
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


def _trader_json(
    aktion: str = "KAUFEN",
    zielkurs: float | None = None,
    stop_loss: float | None = None,
    positionsanteil: float = 5,
) -> str:
    return json.dumps({
        "rolle": "Trader",
        "aktion": aktion,
        "zielkurs": zielkurs,
        "stop_loss": stop_loss,
        "positionsanteil": positionsanteil,
        "begründung": "Test-Begründung",
        "zeithorizont": "Mittelfristig",
    })


class _FakeLLM:
    """Mock-LLM: liefert für jede Temperatur dieselbe vordefinierte Antwort."""

    def __init__(self, response: str):
        self._response = response

    def chat(
        self, messages: list[dict[str, str]], temperature: float = 0.3, **kwargs
    ) -> str | object:
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult

            return StructuredChatResult(text=self._response, response_format_used=True)
        return self._response


_DEBATE = {
    "bull": {"_raw": "Bull-Argument"},
    "bear": {"_raw": "Bear-Argument"},
}


# --------------------------------------------------------------------------- #
# Tests: _technik_veto-Hilfsfunktion
# --------------------------------------------------------------------------- #


class TestTechnikVetoHelper:
    """Testet die reine _technik_veto-Hilfsfunktion."""

    def test_kurs_unter_sma200_vetoed(self):
        """Kurs unter SMA200, RSI neutral → vetoed=True."""
        result = _technik_veto(_analysts(current_price=90.0, sma200=100.0, rsi=45.0))
        assert result["vetoed"] is True
        assert result["ausnahme"] is False
        assert "SMA200" in result["grund"]

    def test_kurs_unter_sma200_stark_unterschied(self):
        """Deutlich unter SMA200 (10% darunter) → Veto greift."""
        result = _technik_veto(_analysts(current_price=80.0, sma200=100.0))
        assert result["vetoed"] is True

    def test_kurs_ueber_sma200_kein_veto(self):
        """Kurs über SMA200 → kein Veto."""
        result = _technik_veto(_analysts(current_price=110.0, sma200=100.0))
        assert result["vetoed"] is False
        assert result["ausnahme"] is False

    def test_kurs_gleich_sma200_kein_veto(self):
        """Kurs exakt gleich SMA200 → kein Veto (Veto nur bei Kurs < SMA200)."""
        result = _technik_veto(_analysts(current_price=100.0, sma200=100.0))
        assert result["vetoed"] is False

    def test_rsi_ausnahme_kurs_unter_sma200_ueber_sma50(self):
        """RSI < 30 bei Kurs > SMA50 → Ausnahme greift (vetoed=False)."""
        result = _technik_veto(
            _analysts(current_price=96.0, sma200=100.0, sma50=95.0, rsi=25.0)
        )
        assert result["vetoed"] is False
        assert result["ausnahme"] is True

    def test_rsi_30_grenze_ist_keine_ausnahme(self):
        """RSI exakt 30 ist keine Ausnahme (Voraussetzung rsi < 30)."""
        result = _technik_veto(
            _analysts(current_price=96.0, sma200=100.0, sma50=95.0, rsi=30.0)
        )
        assert result["vetoed"] is True
        assert result["ausnahme"] is False

    def test_rsi_unter_30_aber_kurs_unter_sma50_kein_veto_ersatz(self):
        """RSI < 30 aber Kurs AUCH unter SMA50 → Ausnahme greift NICHT (Veto)."""
        result = _technik_veto(
            _analysts(current_price=90.0, sma200=100.0, sma50=95.0, rsi=20.0)
        )
        assert result["vetoed"] is True
        assert result["ausnahme"] is False

    def test_fehlende_daten_kein_veto(self):
        """Fehlende technicals-Daten → kein Veto (konservativ)."""
        result = _technik_veto({})
        assert result["vetoed"] is False
        assert result["ausnahme"] is False

    def test_fehlender_sma200_kein_veto(self):
        """SMA200 None (zu kurze Historie) → kein Veto."""
        result = _technik_veto(_analysts(current_price=90.0, sma200=None))
        assert result["vetoed"] is False

    def test_fehlender_current_price_kein_veto(self):
        """current_price None → kein Veto."""
        result = _technik_veto(_analysts(current_price=None, sma200=100.0))
        assert result["vetoed"] is False

    def test_nan_werte_kein_veto(self):
        """NaN-Werte → kein Veto (konservativ, nicht blocken)."""
        nan = float("nan")
        result = _technik_veto(_analysts(current_price=nan, sma200=100.0))
        assert result["vetoed"] is False
        result = _technik_veto(_analysts(current_price=90.0, sma200=nan))
        assert result["vetoed"] is False

    def test_rsi_fehlt_bei_kurs_unter_sma200_vetoed(self):
        """Kurs unter SMA200, RSI fehlt → kein Ausnahmepfad, Veto greift."""
        result = _technik_veto(
            _analysts(current_price=90.0, sma200=100.0, sma50=95.0, rsi=None)
        )
        assert result["vetoed"] is True
        assert result["ausnahme"] is False

    def test_rsi_ausnahme_ohne_sma50_vetoed(self):
        """Kurs unter SMA200, RSI < 30, aber SMA50 fehlt → Ausnahme greift NICHT.

        Die Ausnahme verlangt einen intakten übergeordneten Trend (Kurs > SMA50);
        ohne SMA50-Wert kann das nicht verifiziert werden → kein Ausnahmepfad.
        """
        result = _technik_veto(
            _analysts(current_price=96.0, sma200=100.0, sma50=None, rsi=25.0)
        )
        assert result["vetoed"] is True
        assert result["ausnahme"] is False

    def test_sma50_string_wert_wird_geparst(self):
        """sma50 als String/Decimal-artiger Wert wird tolerant geparst."""
        technicals = {
            "current_price": 96.0,
            "sma200": 100.0,
            "sma50": "95.0",
            "rsi14": "25.0",
        }
        result = _technik_veto({"technicals": technicals})
        assert result["vetoed"] is False
        assert result["ausnahme"] is True


# --------------------------------------------------------------------------- #
# Tests: trader (Single-Modus) wendet das Veto an
# --------------------------------------------------------------------------- #


class TestTraderVeto:
    """Das Veto in trader() (Single-Modus)."""

    def test_kaufen_wird_zu_halten_bei_veto(self):
        """trader: KAUFEN bei Kurs unter SMA200 → HALTEN + Metadaten."""
        llm = _FakeLLM(_trader_json("KAUFEN", zielkurs=110.0, stop_loss=85.0))
        result = trader(_analysts(current_price=90.0, sma200=100.0), _DEBATE, llm)
        assert result["aktion"] == "HALTEN"
        assert result["rating"] == "HALTEN"
        assert result["_technik_veto"] == {
            "vetoed": True,
            "grund": "Kurs unter SMA200 — fallendes Messer, kein KAUFEN (Technik-Veto).",
            "ausnahme": False,
        }

    def test_stark_kaufen_wird_zu_halten_bei_veto(self):
        """trader: STARK KAUFEN bei Kurs unter SMA200 → HALTEN."""
        llm = _FakeLLM(_trader_json("STARK KAUFEN", zielkurs=120.0, stop_loss=80.0))
        result = trader(_analysts(current_price=90.0, sma200=100.0), _DEBATE, llm)
        assert result["aktion"] == "HALTEN"
        assert result["rating"] == "HALTEN"

    def test_kein_veto_bei_kurs_ueber_sma200(self):
        """trader: KAUFEN bleibt KAUFEN, wenn Kurs über SMA200."""
        llm = _FakeLLM(_trader_json("KAUFEN", zielkurs=130.0, stop_loss=100.0))
        result = trader(_analysts(current_price=110.0, sma200=100.0), _DEBATE, llm)
        assert result["aktion"] == "KAUFEN"
        assert "_technik_veto" not in result

    def test_kein_veto_key_bei_halten(self):
        """trader: HALTEN (auch unter SMA200) bleibt unangetastet."""
        llm = _FakeLLM(_trader_json("HALTEN"))
        result = trader(_analysts(current_price=90.0, sma200=100.0), _DEBATE, llm)
        assert result["aktion"] == "HALTEN"
        assert "_technik_veto" not in result

    def test_verkaufen_bleibt_unangetastet(self):
        """trader: VERKAUFEN wird vom Veto nicht berührt (auch unter SMA200)."""
        llm = _FakeLLM(_trader_json("VERKAUFEN", zielkurs=80.0, stop_loss=95.0))
        result = trader(_analysts(current_price=90.0, sma200=100.0), _DEBATE, llm)
        assert result["aktion"] == "VERKAUFEN"
        assert "_technik_veto" not in result

    def test_ausnahme_erzwingt_kleine_position_und_strengen_stop(self):
        """RSI-Ausnahme: KAUFEN bleibt, aber Position max 1.5% + Stop 5% unter Kurs."""
        llm = _FakeLLM(
            _trader_json("KAUFEN", zielkurs=110.0, stop_loss=90.0, positionsanteil=8)
        )
        result = trader(
            _analysts(current_price=96.0, sma200=100.0, sma50=95.0, rsi=25.0),
            _DEBATE,
            llm,
        )
        assert result["aktion"] == "KAUFEN"
        assert result["positionsanteil"] <= 1.5
        # Strenger Stop: 5% unter Kurs → 96 * 0.95 = 91.2
        assert result["stop_loss"] is not None
        assert result["stop_loss"] <= 96.0 * 0.95
        veto = result["_technik_veto"]
        assert veto["vetoed"] is False
        assert veto["ausnahme"] is True

    def test_fehlende_daten_kein_veto_im_trader(self):
        """Ohne technicals-Snapshot → trader verhält sich wie bisher."""
        analysts = {
            "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "G", "_raw": ""},
            "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "G", "_raw": ""},
            "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "N", "_raw": ""},
        }
        llm = _FakeLLM(_trader_json("KAUFEN", zielkurs=110.0, stop_loss=95.0))
        result = trader(analysts, _DEBATE, llm)
        assert result["aktion"] == "KAUFEN"
        assert "_technik_veto" not in result


# --------------------------------------------------------------------------- #
# Tests: ensemble_trader wendet das Veto an
# --------------------------------------------------------------------------- #


class TestEnsembleTraderVeto:
    """Das Veto nach der Mehrheitsabstimmung im ensemble_trader."""

    def test_mehrheit_kaufen_unter_sma200_wird_halten(self):
        """3x KAUFEN bei Kurs unter SMA200 → finale Aktion HALTEN, vetoed=True.

        trader() wendet das Veto bereits pro Run an — die Runs liefern also
        HALTEN, und die Mehrheit ist bereits HALTEN, bevor das Ensemble-Level-
        Veto greift. Der Basis-Run trägt die Veto-Metadaten.
        """
        llm = _FakeLLM(_trader_json("KAUFEN", zielkurs=110.0, stop_loss=85.0))
        result = ensemble_trader(
            _analysts(current_price=90.0, sma200=100.0), _DEBATE, llm, runs=3
        )
        assert result["aktion"] == "HALTEN"
        assert result["rating"] == "HALTEN"
        assert result["_technik_veto"]["vetoed"] is True
        assert result["_technik_veto"]["ausnahme"] is False
        # Die Runs selbst sind bereits HALTEN (Run-Level-Veto in trader())
        assert result["_ensemble"]["mehrheits_aktion"] == "HALTEN"
        assert result["_ensemble"]["alle_aktionen"] == ["HALTEN", "HALTEN", "HALTEN"]

    def test_ensemble_level_veto_safety_net_bei_kaufens_mehrheit(self):
        """Safety-Net: Mehrheit KAUFEN (Runs umgehen das Veto) → finale HALTEN.

        Deckt den Fall ab, dass die Ensemble-Runs trotzdem KAUFEN liefern
        (z. B. deterministisch erzeugte Trades in Tests oder zukünftige Pfade,
        die trader() umgehen). Das Ensemble-Level-Veto nach der Mehrheits-
        abstimmung muss den finalen Trade weiterhin auf HALTEN setzen.
        """
        kauf_trade = {
            "rolle": "Trader",
            "aktion": "KAUFEN",
            "rating": "KAUFEN",
            "zielkurs": 110.0,
            "stop_loss": 85.0,
            "positionsanteil": 5,
            "begründung": "Test",
            "zeithorizont": "Mittelfristig",
        }
        llm = _FakeLLM(_trader_json("KAUFEN"))
        with patch("concilium.agents.trader", return_value=dict(kauf_trade)):
            result = ensemble_trader(
                _analysts(current_price=90.0, sma200=100.0), _DEBATE, llm, runs=3
            )
        assert result["aktion"] == "HALTEN"
        assert result["rating"] == "HALTEN"
        assert result["_technik_veto"]["vetoed"] is True
        assert result["_technik_veto"]["ausnahme"] is False
        # Ensemble-Metadaten zeigen die (ungefilterte) KAUFEN-Mehrheit
        assert result["_ensemble"]["mehrheits_aktion"] == "KAUFEN"
        assert result["_ensemble"]["alle_aktionen"] == ["KAUFEN", "KAUFEN", "KAUFEN"]

    def test_ensemble_level_ausnahme_kappt_position(self):
        """Safety-Net RSI-Ausnahme: Mehrheit KAUFEN bleibt, aber Position ≤ 1.5%."""
        kauf_trade = {
            "rolle": "Trader",
            "aktion": "KAUFEN",
            "rating": "KAUFEN",
            "zielkurs": 110.0,
            "stop_loss": 90.0,
            "positionsanteil": 8,
            "begründung": "Test",
            "zeithorizont": "Mittelfristig",
        }
        llm = _FakeLLM(_trader_json("KAUFEN"))
        with patch("concilium.agents.trader", return_value=dict(kauf_trade)):
            result = ensemble_trader(
                _analysts(current_price=96.0, sma200=100.0, sma50=95.0, rsi=25.0),
                _DEBATE,
                llm,
                runs=3,
            )
        assert result["aktion"] == "KAUFEN"
        assert result["positionsanteil"] <= 1.5
        assert result["stop_loss"] <= 96.0 * 0.95
        assert result["_technik_veto"]["ausnahme"] is True

    def test_mehrheit_halten_kein_veto_key(self):
        """Mehrheit HALTEN → kein Veto-Eingriff, kein _technik_veto-Key."""
        llm = _FakeLLM(_trader_json("HALTEN"))
        result = ensemble_trader(
            _analysts(current_price=90.0, sma200=100.0), _DEBATE, llm, runs=3
        )
        assert result["aktion"] == "HALTEN"
        assert "_technik_veto" not in result

    def test_mehrheit_kaufen_ueber_sma200_unveraendert(self):
        """Kurs über SMA200 → Mehrheits-KAUFEN bleibt KAUFEN, kein Veto-Key."""
        llm = _FakeLLM(_trader_json("KAUFEN", zielkurs=130.0, stop_loss=100.0))
        result = ensemble_trader(
            _analysts(current_price=110.0, sma200=100.0), _DEBATE, llm, runs=3
        )
        assert result["aktion"] == "KAUFEN"
        assert "_technik_veto" not in result

    def test_mehrheit_kaufen_ohne_technicals_daten_unveraendert(self):
        """Fehlende SMA200-Daten → kein Veto (konservativ), KAUFEN bleibt."""
        analysts = {
            "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "G", "_raw": ""},
            "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "G", "_raw": ""},
            "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "N", "_raw": ""},
        }
        llm = _FakeLLM(_trader_json("KAUFEN", zielkurs=110.0, stop_loss=85.0))
        result = ensemble_trader(analysts, _DEBATE, llm, runs=3)
        assert result["aktion"] == "KAUFEN"
        assert "_technik_veto" not in result

    def test_mehrheit_kaufen_1_run_single_fallback(self):
        """Auch der Single-Fallback (1 Run) wird vom Veto erfasst."""
        llm = _FakeLLM(_trader_json("KAUFEN", zielkurs=110.0, stop_loss=None))
        result = ensemble_trader(
            _analysts(current_price=90.0, sma200=100.0), _DEBATE, llm, runs=1
        )
        assert result["aktion"] == "HALTEN"
        assert result["_technik_veto"]["vetoed"] is True


# --------------------------------------------------------------------------- #
# Tests: run_pipeline wendet das Veto nach trade_revision erneut an
# --------------------------------------------------------------------------- #

_MOCK_DATA = {
    "ticker": "TEST",
    "fundamentals": {"name": "TestCo", "sector": "Tech"},
    "technicals": {"current_price": 90.0, "sma50": 95.0, "sma200": 100.0, "rsi14": 45.0},
    "sentiment": {},
    "news": [],
}

_MOCK_ANALYSTS = {
    "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
    "technicals": {
        "current_price": 90.0,
        "sma50": 95.0,
        "sma200": 100.0,
        "rsi14": 45.0,
    },
}

_MOCK_DEBATE = {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}}

_MOCK_RISK = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}

_MOCK_FINAL = {"entscheidung": "GENEHMIGT", "confidence": 4, "begründung": "Ok."}


def _make_trade(aktion: str = "KAUFEN") -> dict:
    return {
        "rolle": "Trader",
        "aktion": aktion,
        "rating": aktion,
        "zielkurs": 110 if aktion == "KAUFEN" else None,
        "stop_loss": 85 if aktion == "KAUFEN" else None,
        "positionsanteil": 3 if aktion == "KAUFEN" else 0,
        "_raw": "",
    }


def _make_llm() -> MagicMock:
    """LLM-Mock mit leerem total_usage (verhindert usage/usage.csv-Einträge)."""
    return MagicMock(total_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


class TestPipelineVetoNachRevision:
    """Schritt 5c (Trade-Revision) darf das Veto nicht umgehen."""

    def _run(self, tmp_path, *, revision_liefert: dict):
        """Gemockter Pipeline-Lauf: Trader KAUFEN unter SMA200, Revision patcht."""
        from datetime import datetime

        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "calibration.json").write_text(
            json.dumps({
                "erstellt_am": datetime.now().isoformat(),
                "anzahl_entscheidungen": 10,
                "hit_rate_gesamt": 0.4,
                "nach_aktion": {
                    "KAUFEN": {"n": 5, "hit_rate": 0.5, "avg_confidence": 0.6},
                    "HALTEN": {"n": 5, "hit_rate": 0.5, "avg_confidence": 0.6},
                    "VERKAUFEN": {"n": 5, "hit_rate": 0.5, "avg_confidence": 0.6},
                },
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        pf = {"portfolio_fit_score": 3, "ziel_gewichtung_pct": 2.0}

        patches = {
            "collect_ticker_data": MagicMock(return_value=_MOCK_DATA),
            "analyst_team": MagicMock(return_value=_MOCK_ANALYSTS),
            "debate": MagicMock(return_value=_MOCK_DEBATE),
            "trader": MagicMock(return_value=_make_trade("KAUFEN")),
            "ensemble_trader": MagicMock(return_value=_make_trade("KAUFEN")),
            "risk_manager": MagicMock(return_value=_MOCK_RISK),
            "fetch_portfolio_positions": MagicMock(return_value=[]),
            "portfolio_fit_agent": MagicMock(return_value=pf),
            "trade_revision": MagicMock(return_value=revision_liefert),
            "portfolio_manager": MagicMock(return_value=_MOCK_FINAL),
            "build_feedback_context": MagicMock(return_value=""),
            "build_reflection_context": MagicMock(return_value=""),
        }
        with patch.multiple("concilium.pipeline", **patches), patch(
            "concilium.journal.append_decision"
        ):
            from concilium.pipeline import run_pipeline

            return run_pipeline("TEST", llm=_make_llm(), ensemble=False, resume=False)

    def test_revision_kaufent_wieder_veto_setzt_halten_zurueck(self, tmp_path, monkeypatch):
        """Revision macht aus HALTEN wieder KAUFEN → Veto setzt zurück auf HALTEN."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))
        revised = _make_trade("KAUFEN")
        revised["_technik_veto"] = {"vetoed": False, "grund": "weggedrückt", "ausnahme": False}

        result = self._run(tmp_path, revision_liefert=revised)

        assert result["trade_revised"] is True
        assert result["trade"]["aktion"] == "HALTEN"
        assert result["trade"]["rating"] == "HALTEN"
        assert result["trade"]["_technik_veto"]["vetoed"] is True

    def test_revision_bleibt_halten_kein_doppeltes_veto(self, tmp_path, monkeypatch):
        """Revision respektiert HALTEN → kein Veto-Eingriff nötig, keine Duplikate."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))
        revised = _make_trade("HALTEN")
        revised["_technik_veto"] = {"vetoed": True, "grund": "Kurs unter SMA200", "ausnahme": False}

        result = self._run(tmp_path, revision_liefert=revised)

        assert result["trade"]["aktion"] == "HALTEN"
        # Metadaten aus der Revision bleiben unangetastet (kein Überschreiben)
        assert result["trade"]["_technik_veto"]["grund"] == "Kurs unter SMA200"

    def test_revision_ausnahme_kappt_position(self, tmp_path, monkeypatch):
        """Revision liefert KAUFEN im Ausnahme-Umfeld → Position auf 1.5% gekappt."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))
        revised = _make_trade("KAUFEN")
        revised["positionsanteil"] = 8

        # Ausnahme-Umfeld: RSI < 30, Kurs über SMA50, aber unter SMA200
        analysts = dict(_MOCK_ANALYSTS)
        analysts["technicals"] = {
            "current_price": 96.0,
            "sma50": 95.0,
            "sma200": 100.0,
            "rsi14": 25.0,
        }
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime

        (state_dir / "calibration.json").write_text(
            json.dumps({
                "erstellt_am": datetime.now().isoformat(),
                "anzahl_entscheidungen": 10,
                "hit_rate_gesamt": 0.4,
                "nach_aktion": {
                    "KAUFEN": {"n": 5, "hit_rate": 0.5, "avg_confidence": 0.6},
                    "HALTEN": {"n": 5, "hit_rate": 0.5, "avg_confidence": 0.6},
                    "VERKAUFEN": {"n": 5, "hit_rate": 0.5, "avg_confidence": 0.6},
                },
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        pf = {"portfolio_fit_score": 3, "ziel_gewichtung_pct": 2.0}
        patches = {
            "collect_ticker_data": MagicMock(return_value=_MOCK_DATA),
            "analyst_team": MagicMock(return_value=analysts),
            "debate": MagicMock(return_value=_MOCK_DEBATE),
            "trader": MagicMock(return_value=_make_trade("KAUFEN")),
            "ensemble_trader": MagicMock(return_value=_make_trade("KAUFEN")),
            "risk_manager": MagicMock(return_value=_MOCK_RISK),
            "fetch_portfolio_positions": MagicMock(return_value=[]),
            "portfolio_fit_agent": MagicMock(return_value=pf),
            "trade_revision": MagicMock(return_value=revised),
            "portfolio_manager": MagicMock(return_value=_MOCK_FINAL),
            "build_feedback_context": MagicMock(return_value=""),
            "build_reflection_context": MagicMock(return_value=""),
        }
        with patch.multiple("concilium.pipeline", **patches), patch(
            "concilium.journal.append_decision"
        ):
            from concilium.pipeline import run_pipeline

            result = run_pipeline("TEST", llm=_make_llm(), ensemble=False, resume=False)

        assert result["trade"]["aktion"] == "KAUFEN"
        assert result["trade"]["positionsanteil"] <= 1.5
        assert result["trade"]["_technik_veto"]["ausnahme"] is True


# --------------------------------------------------------------------------- #
# Tests: Report-Zeile für Technik-Veto / -Ausnahme
# --------------------------------------------------------------------------- #

_VETO_TRADE = {
    "rolle": "Trader",
    "aktion": "HALTEN",
    "rating": "HALTEN",
    "zielkurs": None,
    "stop_loss": None,
    "positionsanteil": 0,
    "begründung": "Veto-Test",
    "zeithorizont": "Mittelfristig",
    "_technik_veto": {
        "vetoed": True,
        "grund": "Kurs unter SMA200 — fallendes Messer, kein KAUFEN (Technik-Veto).",
        "ausnahme": False,
    },
}

_AUSNAHME_TRADE = {
    "rolle": "Trader",
    "aktion": "KAUFEN",
    "rating": "KAUFEN",
    "zielkurs": 110.0,
    "stop_loss": 91.2,
    "positionsanteil": 1.5,
    "begründung": "Ausnahme-Test",
    "zeithorizont": "Mittelfristig",
    "_technik_veto": {
        "vetoed": False,
        "grund": "RSI < 30 bei intaktem SMA200-Umfeld — kleine Position (max 1.5%) "
                 "mit strengem Stop erlaubt (Technik-Ausnahme).",
        "ausnahme": True,
    },
}


def _report_result(trade: dict) -> dict:
    """Vollständiges Result für generate_report mit vorbereitetem Trade."""
    return {
        "ticker": "TEST",
        "no_llm": False,
        "data": {
            "fundamentals": {"name": "TestCo", "sector": "Tech"},
            "technicals": {"current_price": 90.0, "sma200": 100.0},
            "sentiment": {},
        },
        "analysts": {
            "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "G", "_raw": ""},
            "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "G", "_raw": ""},
            "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "N", "_raw": ""},
        },
        "debate": {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}},
        "trade": trade,
        "risk": {"risiko_score": 3, "empfehlung": "GENEHMIGT", "auflagen": "keine"},
        "final": {"entscheidung": "GENEHMIGT", "confidence": 4, "begründung": "Ok."},
    }


class TestReportTechnikVeto:
    """Report zeigt Veto/Ausnahme im Trade-Abschnitt."""

    def test_veto_zeile_im_trade_abschnitt(self):
        """vetoed=True → Veto-Warnzeile im Trade-Vorschlag-Abschnitt."""
        from concilium.report import generate_report

        report = generate_report(_report_result(dict(_VETO_TRADE)))
        assert "> ⚠️ **Technik-Veto:** Kurs unter SMA200 — KAUFEN auf HALTEN reduziert." in report
        # Veto-Zeile vor der Aktion-Zeile (im Trade-Abschnitt)
        veto_pos = report.index("Technik-Veto")
        aktion_pos = report.index("**Aktion:** HALTEN")
        assert veto_pos < aktion_pos

    def test_veto_zeile_im_management_summary(self):
        """Veto-Warnzeile erscheint auch in der Management-Summary."""
        from concilium.report import generate_report

        report = generate_report(_report_result(dict(_VETO_TRADE)))
        assert "Kurs unter SMA200 — KAUFEN auf HALTEN reduziert" in report
        trade_pos = report.index("## 4. Trade-Vorschlag") if "## 4. Trade-Vorschlag" in report else report.index("Trade-Vorschlag")
        veto_first = report.index("Kurs unter SMA200 — KAUFEN auf HALTEN reduziert")
        # Mindestens eine Instanz innerhalb oder vor der Summary — wir erwarten zwei
        assert veto_first < trade_pos  # erste Instanz in der Summary

    def test_ausnahme_zeile_im_trade_abschnitt(self):
        """ausnahme=True → Ausnahme-Warnzeile im Trade-Vorschlag-Abschnitt."""
        from concilium.report import generate_report

        report = generate_report(_report_result(dict(_AUSNAHME_TRADE)))
        assert (
            "> ⚠️ **Technik-Ausnahme:** RSI < 30 bei intaktem SMA200 — kleine Position "
            "(max 1.5%) mit strengem Stop erlaubt." in report
        )
        veto_pos = report.index("Technik-Ausnahme")
        aktion_pos = report.index("**Aktion:** KAUFEN")
        assert veto_pos < aktion_pos

    def test_kein_veto_keine_zeile(self):
        """Ohne _technik_veto-Metadaten → keine Veto-Zeile im Report."""
        from concilium.report import generate_report

        trade = dict(_VETO_TRADE)
        trade.pop("_technik_veto")
        report = generate_report(_report_result(trade))
        assert "Technik-Veto:" not in report
        assert "Technik-Ausnahme:" not in report

    def test_ausnahme_und_veto_exklusiv(self):
        """vetoed=True → keine Ausnahme-Zeile; ausnahme=True → keine Veto-Zeile."""
        from concilium.report import generate_report

        report = generate_report(_report_result(dict(_VETO_TRADE)))
        assert "Technik-Ausnahme:" not in report
        report = generate_report(_report_result(dict(_AUSNAHME_TRADE)))
        assert "Kurs unter SMA200 — KAUFEN auf HALTEN reduziert" not in report
