"""Tests für das graduelle Technik-Signal (SMA200) — Refactor Veto → Signal.

Ein KAUFEN unter SMA200 blockt nicht mehr komplett: KAUFEN bleibt KAUFEN,
aber die Positionsgröße wird graduell skaliert (Faktor 0.3–1.0, je nach
Abstand unter dem SMA200). Ausnahme: RSI < 30 bei intaktem SMA50-Umfeld →
Faktor 0.5 mit strengem Stop (Technik-Ausnahme).

Alle Tests sind offline (kein Netzwerk) — der LLMClient wird gemockt.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from concilium.agents import (  # noqa: E402
    _apply_technik_signal,
    _technik_signal,
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
# Tests: _technik_signal-Hilfsfunktion
# --------------------------------------------------------------------------- #


class TestTechnikSignalHelper:
    """Testet die reine _technik_signal-Hilfsfunktion."""

    def test_kurs_unter_sma200_gradueller_faktor(self):
        """Kurs 10% unter SMA200, RSI neutral → Faktor 0.3 (Untergrenze)."""
        result = _technik_signal(_analysts(current_price=90.0, sma200=100.0, rsi=45.0))
        assert result["faktor"] == pytest.approx(0.3)
        assert result["ausnahme"] is False
        assert "SMA200" in result["grund"]
        assert "fallendes Messer" in result["grund"]

    def test_kurs_deutlich_unter_sma200_faktor_untengrenze(self):
        """Deutlich unter SMA200 (20% darunter) → Faktor clamped auf 0.3."""
        result = _technik_signal(_analysts(current_price=80.0, sma200=100.0))
        assert result["faktor"] == pytest.approx(0.3)

    def test_kurs_kaum_unter_sma200_faktor_nahe_1(self):
        """3% unter SMA200 → Faktor 0.7 (1.0 - 3/10)."""
        result = _technik_signal(_analysts(current_price=97.0, sma200=100.0))
        assert result["faktor"] == pytest.approx(0.7)
        assert "3.0%" in result["grund"]

    def test_kurs_ueber_sma200_kein_abschlag(self):
        """Kurs über SMA200 → Faktor 1.0, leerer Grund."""
        result = _technik_signal(_analysts(current_price=110.0, sma200=100.0))
        assert result["faktor"] == 1.0
        assert result["grund"] == ""
        assert result["ausnahme"] is False

    def test_kurs_gleich_sma200_kein_abschlag(self):
        """Kurs exakt gleich SMA200 → kein Abschlag (nur bei Kurs < SMA200)."""
        result = _technik_signal(_analysts(current_price=100.0, sma200=100.0))
        assert result["faktor"] == 1.0
        assert result["grund"] == ""

    def test_fehlende_daten_faktor_1(self):
        """Fehlende technicals-Daten → Faktor 1.0 (konservativ, kein Abschlag)."""
        result = _technik_signal({})
        assert result["faktor"] == 1.0
        assert result["grund"] == ""
        assert result["ausnahme"] is False

    def test_fehlender_sma200_faktor_1(self):
        """SMA200 None (zu kurze Historie) → kein Abschlag."""
        result = _technik_signal(_analysts(current_price=90.0, sma200=None))
        assert result["faktor"] == 1.0

    def test_fehlender_current_price_faktor_1(self):
        """current_price None → kein Abschlag."""
        result = _technik_signal(_analysts(current_price=None, sma200=100.0))
        assert result["faktor"] == 1.0

    def test_nan_werte_faktor_1(self):
        """NaN-Werte → kein Abschlag (konservativ)."""
        nan = float("nan")
        result = _technik_signal(_analysts(current_price=nan, sma200=100.0))
        assert result["faktor"] == 1.0
        result = _technik_signal(_analysts(current_price=90.0, sma200=nan))
        assert result["faktor"] == 1.0

    def test_rsi_ausnahme_faktor_05(self):
        """RSI < 30 bei Kurs > SMA50 → Ausnahme greift (Faktor 0.5)."""
        result = _technik_signal(
            _analysts(current_price=96.0, sma200=100.0, sma50=95.0, rsi=25.0)
        )
        assert result["faktor"] == 0.5
        assert result["ausnahme"] is True
        assert "Technik-Ausnahme" in result["grund"]

    def test_rsi_30_grenze_ist_keine_ausnahme(self):
        """RSI exakt 30 ist keine Ausnahme (Voraussetzung rsi < 30)."""
        result = _technik_signal(
            _analysts(current_price=96.0, sma200=100.0, sma50=95.0, rsi=30.0)
        )
        assert result["ausnahme"] is False
        # 4% unter SMA200 → gradueller Faktor 0.6 statt Ausnahme-0.5
        assert result["faktor"] == pytest.approx(0.6)

    def test_rsi_unter_30_aber_kurs_unter_sma50_gradueller_faktor(self):
        """RSI < 30 aber Kurs AUCH unter SMA50 → Ausnahme greift NICHT."""
        result = _technik_signal(
            _analysts(current_price=90.0, sma200=100.0, sma50=95.0, rsi=20.0)
        )
        assert result["ausnahme"] is False
        assert result["faktor"] == pytest.approx(0.3)

    def test_rsi_fehlt_gradueller_faktor(self):
        """Kurs unter SMA200, RSI fehlt → gradueller Faktor, keine Ausnahme."""
        result = _technik_signal(
            _analysts(current_price=90.0, sma200=100.0, sma50=95.0, rsi=None)
        )
        assert result["ausnahme"] is False
        assert result["faktor"] == pytest.approx(0.3)

    def test_ausnahme_ohne_sma50_gradueller_faktor(self):
        """Kurs unter SMA200, RSI < 30, aber SMA50 fehlt → kein Ausnahmepfad.

        Die Ausnahme verlangt einen intakten übergeordneten Trend (Kurs > SMA50);
        ohne SMA50-Wert kann das nicht verifiziert werden → gradueller Faktor.
        """
        result = _technik_signal(
            _analysts(current_price=96.0, sma200=100.0, sma50=None, rsi=25.0)
        )
        assert result["ausnahme"] is False
        assert result["faktor"] == pytest.approx(0.6)

    def test_sma50_string_wert_wird_geparst(self):
        """sma50/rsi als String-artige Werte werden tolerant geparst."""
        technicals = {
            "current_price": 96.0,
            "sma200": 100.0,
            "sma50": "95.0",
            "rsi14": "25.0",
        }
        result = _technik_signal({"technicals": technicals})
        assert result["faktor"] == 0.5
        assert result["ausnahme"] is True


# --------------------------------------------------------------------------- #
# Tests: _apply_technik_signal (Skalierung am Trade-dict)
# --------------------------------------------------------------------------- #


class TestApplyTechnikSignal:
    """Testet _apply_technik_signal direkt am Trade-dict."""

    def test_kaufen_bleibt_kaufen_position_skaliert(self):
        """KAUFEN unter SMA200 → Aktion bleibt KAUFEN, Position skaliert, Ziel/Stop intakt."""
        trade = {
            "rolle": "Trader",
            "aktion": "KAUFEN",
            "rating": "KAUFEN",
            "zielkurs": 110.0,
            "stop_loss": 85.0,
            "positionsanteil": 5.0,
        }
        result = _apply_technik_signal(trade, _analysts(current_price=90.0, sma200=100.0))
        assert result["aktion"] == "KAUFEN"
        assert result["rating"] == "KAUFEN"
        # 10% unter SMA200 → Faktor 0.3 → 5.0 * 0.3 = 1.5
        assert result["positionsanteil"] == pytest.approx(1.5)
        # Ziel-/Stop-Werte werden NICHT angetastet (nur Skalierung)
        assert result["zielkurs"] == 110.0
        assert result["stop_loss"] == 85.0
        assert result["_technik_signal"]["faktor"] == pytest.approx(0.3)
        assert result["_technik_signal"]["ausnahme"] is False
        assert "SMA200" in result["_technik_signal"]["grund"]

    def test_stark_kaufen_wird_auch_skaliert(self):
        """STARK KAUFEN unter SMA200 → ebenfalls skaliert, Aktion bleibt."""
        trade = {
            "rolle": "Trader",
            "aktion": "STARK KAUFEN",
            "rating": "STARK KAUFEN",
            "zielkurs": 120.0,
            "stop_loss": 80.0,
            "positionsanteil": 4.0,
        }
        result = _apply_technik_signal(trade, _analysts(current_price=80.0, sma200=100.0))
        assert result["aktion"] == "STARK KAUFEN"
        assert result["positionsanteil"] == pytest.approx(1.2)  # 4.0 * 0.3

    def test_position_fehlt_default_3_skaliert(self):
        """Fehlender positionsanteil → Basis 3.0, skaliert auf 3.0 * Faktor."""
        trade = {"rolle": "Trader", "aktion": "KAUFEN", "zielkurs": 110.0, "stop_loss": 85.0}
        result = _apply_technik_signal(trade, _analysts(current_price=94.0, sma200=100.0))
        # 6% unter SMA200 → Faktor 0.4 → 3.0 * 0.4 = 1.2
        assert result["positionsanteil"] == pytest.approx(1.2)

    def test_skalierte_position_untergrenze_05(self):
        """Untergrenze 0.5: kleines Ergebnis wird auf 0.5 angehoben."""
        trade = {"rolle": "Trader", "aktion": "KAUFEN", "positionsanteil": 1.0}
        result = _apply_technik_signal(trade, _analysts(current_price=90.0, sma200=100.0))
        # 1.0 * 0.3 = 0.3 → Untergrenze 0.5
        assert result["positionsanteil"] == pytest.approx(0.5)

    def test_faktor_1_kein_metadaten_key(self):
        """Kurs über SMA200 → Trade unverändert, kein _technik_signal-Key."""
        trade = {"rolle": "Trader", "aktion": "KAUFEN", "positionsanteil": 5.0}
        result = _apply_technik_signal(trade, _analysts(current_price=110.0, sma200=100.0))
        assert result["positionsanteil"] == 5.0
        assert "_technik_signal" not in result

    def test_halten_wird_nicht_angeruehrt(self):
        """HALTEN (auch unter SMA200) bleibt unangetastet, kein Key."""
        trade = {"rolle": "Trader", "aktion": "HALTEN", "positionsanteil": 3.0}
        result = _apply_technik_signal(trade, _analysts(current_price=90.0, sma200=100.0))
        assert result["aktion"] == "HALTEN"
        assert result["positionsanteil"] == 3.0
        assert "_technik_signal" not in result

    def test_verkaufen_wird_nicht_angeruehrt(self):
        """VERKAUFEN wird vom Signal nicht berührt (auch unter SMA200)."""
        trade = {"rolle": "Trader", "aktion": "VERKAUFEN", "positionsanteil": 2.0}
        result = _apply_technik_signal(trade, _analysts(current_price=90.0, sma200=100.0))
        assert result["aktion"] == "VERKAUFEN"
        assert result["positionsanteil"] == 2.0
        assert "_technik_signal" not in result

    def test_ausnahme_position_und_strenger_stop(self):
        """RSI-Ausnahme: Faktor 0.5 + Stop deterministisch 5% unter Kurs."""
        trade = {
            "rolle": "Trader",
            "aktion": "KAUFEN",
            "zielkurs": 110.0,
            "stop_loss": 95.0,  # zu locker (> 5%-Niveau) → wird auf 91.2 gezogen
            "positionsanteil": 8.0,
        }
        result = _apply_technik_signal(
            trade,
            _analysts(current_price=96.0, sma200=100.0, sma50=95.0, rsi=25.0),
        )
        assert result["aktion"] == "KAUFEN"
        assert result["positionsanteil"] == pytest.approx(4.0)  # 8.0 * 0.5
        # Strenger Stop: 96 * 0.95 = 91.2
        assert result["stop_loss"] == pytest.approx(91.2)
        assert result["_technik_signal"]["faktor"] == 0.5
        assert result["_technik_signal"]["ausnahme"] is True

    def test_ausnahme_default_basis_3_skaliert(self):
        """Ausnahme mit fehlendem positionsanteil: Default-Basis 3.0 → 3.0 * 0.5 = 1.5."""
        trade = {"rolle": "Trader", "aktion": "KAUFEN", "zielkurs": 110.0}
        result = _apply_technik_signal(
            trade,
            _analysts(current_price=96.0, sma200=100.0, sma50=95.0, rsi=25.0),
        )
        assert result["positionsanteil"] == pytest.approx(1.5)
        assert result["stop_loss"] == pytest.approx(91.2)
        assert result["_technik_signal"]["ausnahme"] is True

    def test_ausnahme_stop_bleibt_wenn_schon_streng(self):
        """Ausnahme: vorhandener Stop unter dem 5%-Niveau bleibt stehen."""
        trade = {
            "rolle": "Trader",
            "aktion": "KAUFEN",
            "positionsanteil": 8.0,
            "stop_loss": 91.0,  # strenger als 91.2
        }
        result = _apply_technik_signal(
            trade,
            _analysts(current_price=96.0, sma200=100.0, sma50=95.0, rsi=25.0),
        )
        assert result["stop_loss"] == pytest.approx(91.0)

    def test_idempotenz_doppeltes_anwenden_skaliert_nicht_zweimal(self):
        """Doppelter Apply (Ensemble/Pipeline) skaliert NICHT kumulativ."""
        trade = {"rolle": "Trader", "aktion": "KAUFEN", "positionsanteil": 5.0}
        analysts = _analysts(current_price=90.0, sma200=100.0)
        result = _apply_technik_signal(trade, analysts)
        assert result["positionsanteil"] == pytest.approx(1.5)
        result = _apply_technik_signal(result, analysts)
        assert result["positionsanteil"] == pytest.approx(1.5)  # nicht 0.45

    def test_crash_sicherheit_unpassender_positionsanteil(self):
        """Unpassender positionsanteil (dict) → kein Crash, Trade unverändert."""
        trade = {"rolle": "Trader", "aktion": "KAUFEN", "positionsanteil": {"x": 1}}
        result = _apply_technik_signal(trade, _analysts(current_price=90.0, sma200=100.0))
        assert result["aktion"] == "KAUFEN"
        assert result["positionsanteil"] == {"x": 1}  # unverändert
        assert "_technik_signal" not in result

    def test_fehlende_daten_kein_eingriff(self):
        """Ohne technicals-Snapshot → kein Eingriff, kein Key."""
        trade = {"rolle": "Trader", "aktion": "KAUFEN", "positionsanteil": 5.0}
        result = _apply_technik_signal(trade, {})
        assert result["positionsanteil"] == 5.0
        assert "_technik_signal" not in result


# --------------------------------------------------------------------------- #
# Tests: trader (Single-Modus) wendet das Signal an
# --------------------------------------------------------------------------- #


class TestTraderSignal:
    """Das graduelle Signal in trader() (Single-Modus)."""

    def test_kaufen_bei_signal_bleibt_kaufen(self):
        """trader: KAUFEN bei Kurs unter SMA200 → bleibt KAUFEN, Position skaliert."""
        llm = _FakeLLM(_trader_json("KAUFEN", zielkurs=110.0, stop_loss=85.0))
        result = trader(_analysts(current_price=90.0, sma200=100.0), _DEBATE, llm)
        assert result["aktion"] == "KAUFEN"  # NICHT HALTEN
        assert result["rating"] == "KAUFEN"
        assert result["positionsanteil"] == pytest.approx(1.5)  # 5 * 0.3
        assert result["_technik_signal"]["faktor"] == pytest.approx(0.3)
        assert result["_technik_signal"]["ausnahme"] is False

    def test_kein_signal_key_bei_kurs_ueber_sma200(self):
        """trader: KAUFEN über SMA200 → unverändert, kein Metadaten-Key."""
        llm = _FakeLLM(_trader_json("KAUFEN", zielkurs=130.0, stop_loss=100.0))
        result = trader(_analysts(current_price=110.0, sma200=100.0), _DEBATE, llm)
        assert result["aktion"] == "KAUFEN"
        assert "_technik_signal" not in result

    def test_kein_signal_key_bei_halten(self):
        """trader: HALTEN (auch unter SMA200) bleibt unangetastet."""
        llm = _FakeLLM(_trader_json("HALTEN"))
        result = trader(_analysts(current_price=90.0, sma200=100.0), _DEBATE, llm)
        assert result["aktion"] == "HALTEN"
        assert "_technik_signal" not in result

    def test_verkaufen_bleibt_unangetastet(self):
        """trader: VERKAUFEN wird vom Signal nicht berührt (auch unter SMA200)."""
        llm = _FakeLLM(_trader_json("VERKAUFEN", zielkurs=80.0, stop_loss=95.0))
        result = trader(_analysts(current_price=90.0, sma200=100.0), _DEBATE, llm)
        assert result["aktion"] == "VERKAUFEN"
        assert "_technik_signal" not in result

    def test_ausnahme_kleine_position_und_strenger_stop(self):
        """RSI-Ausnahme: KAUFEN bleibt, Faktor 0.5 + Stop 5% unter Kurs."""
        llm = _FakeLLM(
            _trader_json("KAUFEN", zielkurs=110.0, stop_loss=95.0, positionsanteil=8)
        )
        result = trader(
            _analysts(current_price=96.0, sma200=100.0, sma50=95.0, rsi=25.0),
            _DEBATE,
            llm,
        )
        assert result["aktion"] == "KAUFEN"
        assert result["positionsanteil"] == pytest.approx(4.0)
        assert result["stop_loss"] == pytest.approx(91.2)
        assert result["_technik_signal"]["ausnahme"] is True

    def test_fehlende_daten_kein_eingriff_im_trader(self):
        """Ohne technicals-Snapshot → trader verhält sich wie bisher."""
        analysts = {
            "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "G", "_raw": ""},
            "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "G", "_raw": ""},
            "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "N", "_raw": ""},
        }
        llm = _FakeLLM(_trader_json("KAUFEN", zielkurs=110.0, stop_loss=95.0))
        result = trader(analysts, _DEBATE, llm)
        assert result["aktion"] == "KAUFEN"
        assert "_technik_signal" not in result


# --------------------------------------------------------------------------- #
# Tests: ensemble_trader wendet das Signal an (Safety-Net, nicht-blockend)
# --------------------------------------------------------------------------- #


class TestEnsembleTraderSignal:
    """Das graduelle Signal nach der Mehrheitsabstimmung im ensemble_trader."""

    def test_mehrheit_kaufen_unter_sma200_bleibt_kaufen_skaliert(self):
        """3x KAUFEN bei Kurs unter SMA200 → finale Aktion KAUFEN, skaliert.

        trader() wendet das Signal bereits pro Run an (idempotent) — das
        Ensemble-Level-Signal darf NICHT ein zweites Mal kumulativ skalieren.
        """
        llm = _FakeLLM(_trader_json("KAUFEN", zielkurs=110.0, stop_loss=85.0))
        result = ensemble_trader(
            _analysts(current_price=90.0, sma200=100.0), _DEBATE, llm, runs=3
        )
        assert result["aktion"] == "KAUFEN"  # NICHT HALTEN
        assert result["rating"] == "KAUFEN"
        assert result["positionsanteil"] == pytest.approx(1.5)  # idempotent: 5 * 0.3
        assert result["_technik_signal"]["faktor"] == pytest.approx(0.3)
        assert result["_technik_signal"]["ausnahme"] is False
        assert result["_ensemble"]["mehrheits_aktion"] == "KAUFEN"
        assert result["_ensemble"]["alle_aktionen"] == ["KAUFEN", "KAUFEN", "KAUFEN"]

    def test_safety_net_skaliert_gemockten_kauf_trade(self):
        """Safety-Net: Runs umgehen trader() (KAUFEN, unskaliert) → Ensemblesignal skaliert."""
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
        assert result["aktion"] == "KAUFEN"  # NICHT auf HALTEN gesetzt
        assert result["rating"] == "KAUFEN"
        assert result["positionsanteil"] == pytest.approx(1.5)  # 5 * 0.3
        assert result["_technik_signal"]["faktor"] == pytest.approx(0.3)
        assert result["_ensemble"]["mehrheits_aktion"] == "KAUFEN"

    def test_ensemble_ausnahme_kleine_position(self):
        """Safety-Net RSI-Ausnahme: Mehrheit KAUFEN bleibt, Faktor 0.5 + strenger Stop."""
        kauf_trade = {
            "rolle": "Trader",
            "aktion": "KAUFEN",
            "rating": "KAUFEN",
            "zielkurs": 110.0,
            "stop_loss": 95.0,
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
        assert result["positionsanteil"] == pytest.approx(4.0)
        assert result["stop_loss"] == pytest.approx(91.2)
        assert result["_technik_signal"]["ausnahme"] is True

    def test_mehrheit_halten_kein_signal_key(self):
        """Mehrheit HALTEN → kein Signal-Eingriff, kein _technik_signal-Key."""
        llm = _FakeLLM(_trader_json("HALTEN"))
        result = ensemble_trader(
            _analysts(current_price=90.0, sma200=100.0), _DEBATE, llm, runs=3
        )
        assert result["aktion"] == "HALTEN"
        assert "_technik_signal" not in result

    def test_mehrheit_kaufen_ueber_sma200_unveraendert(self):
        """Kurs über SMA200 → Mehrheits-KAUFEN bleibt KAUFEN, kein Signal-Key."""
        llm = _FakeLLM(_trader_json("KAUFEN", zielkurs=130.0, stop_loss=100.0))
        result = ensemble_trader(
            _analysts(current_price=110.0, sma200=100.0), _DEBATE, llm, runs=3
        )
        assert result["aktion"] == "KAUFEN"
        assert "_technik_signal" not in result

    def test_mehrheit_kaufen_ohne_technicals_daten_unveraendert(self):
        """Fehlende SMA200-Daten → kein Abschlag (konservativ), KAUFEN bleibt."""
        analysts = {
            "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "G", "_raw": ""},
            "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "G", "_raw": ""},
            "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "N", "_raw": ""},
        }
        llm = _FakeLLM(_trader_json("KAUFEN", zielkurs=110.0, stop_loss=85.0))
        result = ensemble_trader(analysts, _DEBATE, llm, runs=3)
        assert result["aktion"] == "KAUFEN"
        assert "_technik_signal" not in result

    def test_mehrheit_kaufen_1_run_single_fallback_skaliert(self):
        """Auch der Single-Fallback (1 Run) wird vom Signal erfasst."""
        llm = _FakeLLM(_trader_json("KAUFEN", zielkurs=110.0, stop_loss=None))
        result = ensemble_trader(
            _analysts(current_price=90.0, sma200=100.0), _DEBATE, llm, runs=1
        )
        assert result["aktion"] == "KAUFEN"
        assert result["positionsanteil"] == pytest.approx(1.5)
        assert result["_technik_signal"]["faktor"] == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# Tests: run_pipeline wendet das Signal nach trade_revision erneut an
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


class TestPipelineSignalNachRevision:
    """Schritt 5c (Trade-Revision) darf das Signal nicht umgehen (skaliert nur)."""

    def _run(self, tmp_path, *, revision_liefert: dict, analysts: dict | None = None):
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
            "analyst_team": MagicMock(return_value=analysts or _MOCK_ANALYSTS),
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

    def test_revision_kaufent_signal_skaliert_erneut(self, tmp_path, monkeypatch):
        """Revision macht KAUFEN → Signal skaliert erneut, Aktion bleibt KAUFEN."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))
        revised = _make_trade("KAUFEN")
        revised["positionsanteil"] = 5
        # Stale/Junk-Metadaten der Revision — der erneute Apply überschreibt sie.
        revised["_technik_signal"] = {"faktor": 1.0, "grund": "weggedrückt", "ausnahme": False}

        result = self._run(tmp_path, revision_liefert=revised)

        assert result["trade_revised"] is True
        assert result["trade"]["aktion"] == "KAUFEN"  # NICHT HALTEN
        assert result["trade"]["rating"] == "KAUFEN"
        # 5 * 0.3 = 1.5 (Basis aus _technik_signal_basis, kein kumulativer Abschlag)
        assert result["trade"]["positionsanteil"] == pytest.approx(1.5)
        assert result["trade"]["_technik_signal"]["faktor"] == pytest.approx(0.3)
        assert result["trade"]["_technik_signal"]["ausnahme"] is False

    def test_revision_halten_kein_eingriff(self, tmp_path, monkeypatch):
        """Revision respektiert HALTEN → kein Signal-Eingriff, Metadaten unangetastet."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))
        revised = _make_trade("HALTEN")
        revised["_technik_signal"] = {"faktor": 0.3, "grund": "Kurs unter SMA200", "ausnahme": False}

        result = self._run(tmp_path, revision_liefert=revised)

        assert result["trade"]["aktion"] == "HALTEN"
        # Metadaten aus der Revision bleiben unangetastet (HALTEN wird nie angerührt)
        assert result["trade"]["_technik_signal"]["grund"] == "Kurs unter SMA200"

    def test_revision_ausnahme_position_und_stop(self, tmp_path, monkeypatch):
        """Revision liefert KAUFEN im Ausnahme-Umfeld → Faktor 0.5 + strenger Stop."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))
        revised = _make_trade("KAUFEN")
        revised["positionsanteil"] = 8
        revised["stop_loss"] = 95.0  # zu locker (> 5%-Niveau) → wird auf 91.2 gezogen

        # Ausnahme-Umfeld: RSI < 30, Kurs über SMA50, aber unter SMA200
        analysts = dict(_MOCK_ANALYSTS)
        analysts["technicals"] = {
            "current_price": 96.0,
            "sma50": 95.0,
            "sma200": 100.0,
            "rsi14": 25.0,
        }

        result = self._run(tmp_path, revision_liefert=revised, analysts=analysts)

        assert result["trade"]["aktion"] == "KAUFEN"
        assert result["trade"]["positionsanteil"] == pytest.approx(4.0)  # 8 * 0.5
        assert result["trade"]["stop_loss"] == pytest.approx(91.2)
        assert result["trade"]["_technik_signal"]["ausnahme"] is True


# --------------------------------------------------------------------------- #
# Tests: Report-Zeile für Technik-Signal / -Ausnahme
# --------------------------------------------------------------------------- #

_SIGNAL_TRADE = {
    "rolle": "Trader",
    "aktion": "KAUFEN",
    "rating": "KAUFEN",
    "zielkurs": 110.0,
    "stop_loss": 85.0,
    "positionsanteil": 1.5,
    "begründung": "Signal-Test",
    "zeithorizont": "Mittelfristig",
    "_technik_signal": {
        "faktor": 0.3,
        "grund": (
            "Kurs 10.0% unter SMA200 — Positionsgröße um Faktor 0.30 "
            "reduziert (fallendes Messer)."
        ),
        "ausnahme": False,
    },
}

_AUSNAHME_SIGNAL_TRADE = {
    "rolle": "Trader",
    "aktion": "KAUFEN",
    "rating": "KAUFEN",
    "zielkurs": 110.0,
    "stop_loss": 91.2,
    "positionsanteil": 4.0,
    "begründung": "Ausnahme-Test",
    "zeithorizont": "Mittelfristig",
    "_technik_signal": {
        "faktor": 0.5,
        "grund": "RSI < 30 bei intaktem SMA50-Umfeld — kleine Position erlaubt (Technik-Ausnahme).",
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


class TestReportTechnikSignal:
    """Report zeigt das Signal/die Ausnahme im Trade-Abschnitt und der Summary."""

    def test_signal_zeile_im_trade_abschnitt(self):
        """faktor < 1.0 → Signal-Warnzeile im Trade-Vorschlag-Abschnitt."""
        from concilium.report import generate_report

        report = generate_report(_report_result(dict(_SIGNAL_TRADE)))
        assert (
            "> ⚠️ **Technik-Signal:** Kurs unter SMA200 — "
            "Positionsgröße um Faktor 0.3 reduziert (fallendes Messer)." in report
        )
        # Signal-Zeile vor der Aktion-Zeile (im Trade-Abschnitt)
        signal_pos = report.index("Technik-Signal")
        aktion_pos = report.index("**Aktion:** KAUFEN")
        assert signal_pos < aktion_pos

    def test_signal_zeile_im_management_summary(self):
        """Signal-Warnzeile erscheint auch in der Management-Summary."""
        from concilium.report import generate_report

        report = generate_report(_report_result(dict(_SIGNAL_TRADE)))
        needle = "Positionsgröße um Faktor 0.3 reduziert (fallendes Messer)"
        assert needle in report
        trade_pos = report.index("Trade-Vorschlag")
        first_pos = report.index(needle)
        # Erste Instanz in der Management-Summary (vor der Trade-Sektion)
        assert first_pos < trade_pos

    def test_ausnahme_zeile_im_trade_abschnitt(self):
        """ausnahme=True → Ausnahme-Warnzeile im Trade-Vorschlag-Abschnitt."""
        from concilium.report import generate_report

        report = generate_report(_report_result(dict(_AUSNAHME_SIGNAL_TRADE)))
        assert (
            "> ⚠️ **Technik-Ausnahme:** RSI < 30 bei intaktem SMA50 — "
            "kleine Position (Faktor 0.5) mit strengem Stop." in report
        )
        signal_pos = report.index("Technik-Ausnahme")
        aktion_pos = report.index("**Aktion:** KAUFEN")
        assert signal_pos < aktion_pos

    def test_kein_signal_keine_zeile(self):
        """Ohne _technik_signal-Metadaten → keine Signal-Zeile im Report."""
        from concilium.report import generate_report

        trade = dict(_SIGNAL_TRADE)
        trade.pop("_technik_signal")
        report = generate_report(_report_result(trade))
        assert "Technik-Signal:" not in report
        assert "Technik-Ausnahme:" not in report

    def test_faktor_1_keine_zeile(self):
        """faktor == 1.0 → keine Warnzeile (nur Abschlag wird gerendert)."""
        from concilium.report import generate_report

        trade = dict(_SIGNAL_TRADE)
        trade["_technik_signal"] = {"faktor": 1.0, "grund": "", "ausnahme": False}
        report = generate_report(_report_result(trade))
        assert "Technik-Signal:" not in report
        assert "Technik-Ausnahme:" not in report

    def test_signal_und_ausnahme_exklusiv(self):
        """ausnahme=True → keine Signal-Zeile; faktor<1 ohne Ausnahme → keine Ausnahme-Zeile."""
        from concilium.report import generate_report

        report = generate_report(_report_result(dict(_SIGNAL_TRADE)))
        assert "Technik-Ausnahme:" not in report
        report = generate_report(_report_result(dict(_AUSNAHME_SIGNAL_TRADE)))
        assert "Technik-Signal:" not in report
