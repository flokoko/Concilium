"""Tests für den deterministischen Final-Position-Guard (Punkt 4).

Feature: Nach dem Portfolio-Manager (Schritt 6) dürfen KEINE LLM-Outputs mehr
die Ziel-Gewichtung (ziel_gewichtung_pct) über das harte Maximum heben.
Der Final-Guard kappt portfolio_fit["ziel_gewichtung_pct"] deterministisch am
CONCILIUM_MAX_POSITION_PCT (Default 15.0 %) — hart, nicht durch LLM
übersteuerbar — und schreibt Metadaten portfolio_fit["_final_guard"].

Getestet werden:
- config.max_position_pct: Env-Wert, Default 15.0, ValueError bei ungültig
- _apply_final_position_guard: Kap 25 → 15.0, Original/Flag-Metadaten,
  unverändert bei 8 %, Idempotenz (Resume), nie-crashen (None-Inputs,
  nicht-numerische Werte, fehlende Config), Konsistenz-Hinweis trade
- Pipeline-Wire-up: Guard läuft NACH dem PM (PM-Modifiziert-Wert wird
  gekappt), skip_final-Modus crasht nicht
- Report: Final-Guard-Hinweis nur bei gekappt=True

Alle Tests sind OFFLINE-fähig: kein yfinance, kein Netzwerk, kein LLM.
"""

from __future__ import annotations

import math
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium import config  # noqa: E402
from concilium.pipeline import (  # noqa: E402
    _apply_final_position_guard,
    run_pipeline,
)
from concilium.report import generate_report  # noqa: E402

# --------------------------------------------------------------------------- #
# Tests: config.max_position_pct
# --------------------------------------------------------------------------- #


class TestMaxPositionPct:
    """Test die Config-Funktion max_position_pct (lazy, float, laute ValueError)."""

    def test_default_15_0(self, monkeypatch):
        """Ohne Env-Var → 15.0 (Default)."""
        monkeypatch.delenv("CONCILIUM_MAX_POSITION_PCT", raising=False)
        assert config.max_position_pct() == 15.0
        assert isinstance(config.max_position_pct(), float)

    def test_env_value_float(self, monkeypatch):
        """Env-Var '8.5' → 8.5 (float-Koerzion)."""
        with patch.dict("os.environ", {"CONCILIUM_MAX_POSITION_PCT": "8.5"}):
            assert config.max_position_pct() == 8.5

    def test_env_value_int_string_coerced(self, monkeypatch):
        """Env-Var '20' (int-String) → 20.0 (float)."""
        with patch.dict("os.environ", {"CONCILIUM_MAX_POSITION_PCT": "20"}):
            assert config.max_position_pct() == 20.0

    def test_invalid_value_raises_value_error(self, monkeypatch):
        """Tippfehler 'fünfzehn' → laute ValueError (kein stiller Fallback)."""
        with patch.dict("os.environ", {"CONCILIUM_MAX_POSITION_PCT": "fünfzehn"}):
            with pytest.raises(ValueError, match="CONCILIUM_MAX_POSITION_PCT"):
                config.max_position_pct()

    def test_zero_raises_value_error(self, monkeypatch):
        """Wert 0.0 ist ungültig (> 0 erforderlich) → ValueError."""
        with patch.dict("os.environ", {"CONCILIUM_MAX_POSITION_PCT": "0"}):
            with pytest.raises(ValueError, match="CONCILIUM_MAX_POSITION_PCT"):
                config.max_position_pct()

    def test_negative_raises_value_error(self, monkeypatch):
        """Negativer Wert → ValueError."""
        with patch.dict("os.environ", {"CONCILIUM_MAX_POSITION_PCT": "-5"}):
            with pytest.raises(ValueError, match="CONCILIUM_MAX_POSITION_PCT"):
                config.max_position_pct()

    def test_lazy_fresh_read_each_call(self, monkeypatch):
        """Lazy: Wert wird bei jedem Aufruf frisch gelesen (kein Import-Cache)."""
        monkeypatch.delenv("CONCILIUM_MAX_POSITION_PCT", raising=False)
        assert config.max_position_pct() == 15.0
        with patch.dict("os.environ", {"CONCILIUM_MAX_POSITION_PCT": "12"}):
            assert config.max_position_pct() == 12.0
        assert config.max_position_pct() == 15.0


# --------------------------------------------------------------------------- #
# Tests: _apply_final_position_guard
# --------------------------------------------------------------------------- #


class TestApplyFinalPositionGuard:
    """Test den Guard: Clamp, Metadaten, Idempotenz, nie-crashen."""

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch):
        """Env-Var pro Test isolieren (Default 15.0 unless anders gesetzt)."""
        monkeypatch.delenv("CONCILIUM_MAX_POSITION_PCT", raising=False)

    def test_clamps_25_to_max_and_records_metadata(self):
        """25 % Ziel-Gewichtung → auf 15.0 gekappt, Metadaten korrekt."""
        pf = {"ziel_gewichtung_pct": 25.0}
        trade = {"aktion": "KAUFEN"}
        result = _apply_final_position_guard(pf, trade)
        assert result["ziel_gewichtung_pct"] == 15.0
        guard = result["_final_guard"]
        assert guard["max_pct"] == 15.0
        assert guard["original"] == 25.0
        assert guard["gekappt"] is True

    def test_leaves_8_pct_untouched_gekappt_false(self):
        """8 % < Max → unverändert, Metadaten mit gekappt=False."""
        pf = {"ziel_gewichtung_pct": 8.0}
        trade = {"aktion": "KAUFEN"}
        result = _apply_final_position_guard(pf, trade)
        assert result["ziel_gewichtung_pct"] == 8.0
        guard = result["_final_guard"]
        assert guard["max_pct"] == 15.0
        assert guard["original"] == 8.0
        assert guard["gekappt"] is False

    def test_idempotent_already_clamped_value_not_double_shifted(self):
        """Resume: bereits gekappter Wert bleibt unverändert, original erhalten."""
        pf = {
            "ziel_gewichtung_pct": 15.0,
            "_final_guard": {"max_pct": 15.0, "original": 25.0, "gekappt": True},
        }
        trade = {"aktion": "KAUFEN"}
        result = _apply_final_position_guard(pf, trade)
        assert result["ziel_gewichtung_pct"] == 15.0  # NICHT erneut verschoben
        guard = result["_final_guard"]
        assert guard["original"] == 25.0  # Provenance erhalten
        assert guard["gekappt"] is True

    def test_idempotent_reapply_after_reweight_reclamps(self):
        """Erneutes Anheben (PM-Re-Weight) wird erneut gekappt, original bleibt."""
        pf = {
            "ziel_gewichtung_pct": 30.0,  # PM hat wieder hochgesetzt
            "_final_guard": {"max_pct": 15.0, "original": 25.0, "gekappt": True},
        }
        result = _apply_final_position_guard(pf, {"aktion": "KAUFEN"})
        assert result["ziel_gewichtung_pct"] == 15.0
        guard = result["_final_guard"]
        assert guard["original"] == 25.0  # NICHT 30.0 überschrieben
        assert guard["gekappt"] is True

    def test_does_not_touch_ziel_gewichtung_original(self):
        """Provenance-Feld ziel_gewichtung_original (Dämpfung) bleibt intakt."""
        pf = {
            "ziel_gewichtung_pct": 25.0,
            "ziel_gewichtung_original": 40.0,
            "ziel_gewichtung_gedämpft": True,
        }
        result = _apply_final_position_guard(pf, {"aktion": "KAUFEN"})
        assert result["ziel_gewichtung_pct"] == 15.0
        assert result["ziel_gewichtung_original"] == 40.0
        assert result["ziel_gewichtung_gedämpft"] is True

    def test_consistency_hint_kaufen(self):
        """KAUFEN: Konsistenz-Hinweis bool (True nach Clamp, False bei Grenzverletzung)."""
        trade = {"aktion": "KAUFEN"}
        _apply_final_position_guard({"ziel_gewichtung_pct": 25.0}, trade)
        assert trade["_final_guard_consistency"] is True  # 15.0 <= 15.0

        trade2 = {"aktion": "KAUFEN"}
        _apply_final_position_guard({"ziel_gewichtung_pct": 8.0}, trade2)
        assert trade2["_final_guard_consistency"] is True  # 8.0 <= 15.0

    def test_consistency_hint_stark_kaufen(self):
        """STARK KAUFEN: Hinweis wird ebenfalls gesetzt."""
        trade = {"aktion": "STARK KAUFEN"}
        _apply_final_position_guard({"ziel_gewichtung_pct": 3.0}, trade)
        assert trade["_final_guard_consistency"] is True

    def test_consistency_hint_not_set_for_halten_verkaufen(self):
        """HALTEN/VERKAUFEN: kein Konsistenz-Hinweis (nur KAUFEN/STARK KAUFEN)."""
        for aktion in ("HALTEN", "VERKAUFEN", "STARK VERKAUFEN"):
            trade = {"aktion": aktion}
            _apply_final_position_guard({"ziel_gewichtung_pct": 25.0}, trade)
            assert "_final_guard_consistency" not in trade, aktion

    def test_positionsanteil_not_forced_to_match(self):
        """Kein Zwangs-Match: positionsanteil bleibt unabhängig (andere Semantik)."""
        trade = {"aktion": "KAUFEN", "positionsanteil": 3}
        result = _apply_final_position_guard(
            {"ziel_gewichtung_pct": 25.0}, trade
        )
        assert trade["positionsanteil"] == 3  # unverändert
        assert result["ziel_gewichtung_pct"] == 15.0

    def test_none_portfolio_fit_returns_none(self):
        """portfolio_fit None → None, kein Crash."""
        assert _apply_final_position_guard(None, {"aktion": "KAUFEN"}) is None

    def test_none_trade_no_crash_no_consistency(self):
        """trade None → kein Crash, kein Konsistenz-Feld."""
        pf = {"ziel_gewichtung_pct": 25.0}
        result = _apply_final_position_guard(pf, None)
        assert result["ziel_gewichtung_pct"] == 15.0
        assert result["_final_guard"]["gekappt"] is True

    def test_non_numeric_ziel_records_metadata_gekappt_false(self):
        """Nicht-numerisches Ziel → kein Crash, gekappt=False, Wert unverändert."""
        pf = {"ziel_gewichtung_pct": "hoch"}
        result = _apply_final_position_guard(pf, {"aktion": "KAUFEN"})
        assert result["ziel_gewichtung_pct"] == "hoch"
        guard = result["_final_guard"]
        assert guard["gekappt"] is False
        assert guard["original"] == "hoch"

    def test_nan_ziel_no_clamp_gekappt_false(self):
        """NaN-Ziel → kein Clamp (nur endlich numerische Werte), gekappt=False."""
        pf = {"ziel_gewichtung_pct": float("nan")}
        result = _apply_final_position_guard(pf, {"aktion": "KAUFEN"})
        assert math.isnan(result["ziel_gewichtung_pct"])
        assert result["_final_guard"]["gekappt"] is False

    def test_inf_ziel_no_clamp_gekappt_false(self):
        """±Inf-Ziel → kein Clamp, gekappt=False (kein Crash)."""
        pf = {"ziel_gewichtung_pct": float("inf")}
        result = _apply_final_position_guard(pf, {"aktion": "KAUFEN"})
        assert math.isinf(result["ziel_gewichtung_pct"])
        assert result["_final_guard"]["gekappt"] is False

    def test_bool_ziel_treated_as_non_numeric(self):
        """Bool-Ziel (True) ist kein numerischer Zielwert → kein Clamp."""
        pf = {"ziel_gewichtung_pct": True}
        result = _apply_final_position_guard(pf, {"aktion": "KAUFEN"})
        assert result["ziel_gewichtung_pct"] is True
        assert result["_final_guard"]["gekappt"] is False

    def test_zero_and_negative_ziel_not_clamped(self):
        """Ziel <= 0: kein Clamp (Guard senkt nur, hebt nie an)."""
        for ziel in (0, 0.0, -3.0):
            pf = {"ziel_gewichtung_pct": ziel}
            result = _apply_final_position_guard(pf, {"aktion": "KAUFEN"})
            assert result["ziel_gewichtung_pct"] == ziel, ziel
            assert result["_final_guard"]["gekappt"] is False

    def test_env_override_changes_max(self, monkeypatch):
        """Env-Override: Max 8.0 → 25 % wird auf 8.0 gekappt."""
        monkeypatch.setenv("CONCILIUM_MAX_POSITION_PCT", "8.0")
        pf = {"ziel_gewichtung_pct": 25.0}
        result = _apply_final_position_guard(pf, {"aktion": "KAUFEN"})
        assert result["ziel_gewichtung_pct"] == 8.0
        assert result["_final_guard"]["max_pct"] == 8.0

    def test_invalid_config_leaves_input_unchanged(self, monkeypatch):
        """Ungültige Config (ValueError) → kein Crash, Input unverändert."""
        monkeypatch.setenv("CONCILIUM_MAX_POSITION_PCT", "fünfzehn")
        pf = {"ziel_gewichtung_pct": 25.0}
        result = _apply_final_position_guard(pf, {"aktion": "KAUFEN"})
        assert result == pf  # unverändert, kein "_final_guard" geschrieben

    def test_exactly_at_max_not_clamped(self):
        """Ziel == Max → kein Clamp nötig, gekappt=False."""
        pf = {"ziel_gewichtung_pct": 15.0}
        result = _apply_final_position_guard(pf, {"aktion": "KAUFEN"})
        assert result["ziel_gewichtung_pct"] == 15.0
        assert result["_final_guard"]["gekappt"] is False

    def test_positionsanteil_below_floor_left_untouched(self):
        """positionsanteil < 0.1 wird NICHT angefasst (informational only)."""
        trade = {"aktion": "KAUFEN", "positionsanteil": 0.05}
        result = _apply_final_position_guard(
            {"ziel_gewichtung_pct": 8.0}, trade
        )
        assert trade["positionsanteil"] == 0.05  # unverändert
        assert result["ziel_gewichtung_pct"] == 8.0


# --------------------------------------------------------------------------- #
# Tests: Pipeline-Wire-up (NACH dem PM, Schritt 6)
# --------------------------------------------------------------------------- #

_MOCK_DATA = {
    "ticker": "TEST",
    "fundamentals": {"name": "TestCo", "sector": "Tech"},
    "technicals": {"current_price": 100},
    "sentiment": {},
    "news": [],
}

_MOCK_ANALYSTS = {
    "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
}

_MOCK_DEBATE = {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}}

_MOCK_RISK = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}


def _make_trade(aktion: str = "KAUFEN") -> dict:
    return {
        "rolle": "Trader",
        "aktion": aktion,
        "rating": aktion,
        "zielkurs": 115,
        "stop_loss": 92,
        "positionsanteil": 3,
        "_raw": "",
    }


def _run_pipeline_with_mocks(
    trade: dict,
    portfolio_fit: dict | None,
    *,
    skip_final: bool = False,
    pm_side_effect=None,
    ensemble: bool = False,
) -> dict:
    """Führt run_pipeline mit komplett gemockten Agenten aus (offline).

    pm_side_effect: optionaler side_effect für den portfolio_manager-Mock
    (z. B. MODIFIZIERT, das die Ziel-Gewichtung hochsetzt). Default:
    GENEHMIGT ohne Seiteneffekt.

    CONCILIUM_STATE_DIR muss vom Test gesetzt sein (state_dir-Fixture),
    damit die Dämpfung nicht die echte state/calibration.json liest
    (ohne calibration.json → keine Dämpfung, Ziel-Gewichtung bleibt wie
    vom PM gesetzt).
    """
    if pm_side_effect is None:
        pm_side_effect = MagicMock(
            return_value={"entscheidung": "GENEHMIGT", "confidence": 4}
        )
    patches = {
        "collect_ticker_data": MagicMock(return_value=_MOCK_DATA),
        "analyst_team": MagicMock(return_value=_MOCK_ANALYSTS),
        "debate": MagicMock(return_value=_MOCK_DEBATE),
        "trader": MagicMock(return_value=trade),
        "ensemble_trader": MagicMock(return_value=trade),
        "risk_manager": MagicMock(return_value=_MOCK_RISK),
        "fetch_portfolio_positions": MagicMock(return_value=[]),
        "portfolio_fit_agent": MagicMock(return_value=portfolio_fit),
        "trade_revision": MagicMock(return_value=trade),
        "portfolio_manager": MagicMock(side_effect=pm_side_effect),
        "build_feedback_context": MagicMock(return_value=""),
        "build_reflection_context": MagicMock(return_value=""),
    }
    with patch.multiple("concilium.pipeline", **patches), patch(
        "concilium.journal.append_decision"
    ):
        return run_pipeline(
            "TEST",
            llm=MagicMock(),
            ensemble=ensemble,
            skip_final=skip_final,
            resume=False,
        )


class TestPipelineFinalGuardWireUp:
    """Test das Wire-up: Guard läuft NACH dem Portfolio-Manager."""

    @pytest.fixture
    def state_dir(self, tmp_path, monkeypatch):
        """Isoliertes state-Verzeichnis (CONCILIUM_STATE_DIR auf tmp_path)."""
        d = tmp_path / "state"
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(d))
        return str(d)

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch):
        """Max-Position-Env-Var pro Test isolieren."""
        monkeypatch.delenv("CONCILIUM_MAX_POSITION_PCT", raising=False)

    @staticmethod
    def _pm_reweight_30(*args, **kwargs):
        """Simuliert PM-MODIFIZIERT, das die Ziel-Gewichtung auf 30 % hebt."""
        portfolio_fit = kwargs.get("portfolio_fit")
        if isinstance(portfolio_fit, dict):
            portfolio_fit["ziel_gewichtung_pct"] = 30.0
        return {
            "entscheidung": "MODIFIZIERT",
            "confidence": 4,
            "auflagen": "Ziel-Gewichtung auf 30 % erhöht",
        }

    def test_pm_modifiziert_reweight_gets_clamped(self, tmp_path, state_dir):
        """PM-MODIFIZIERT hebt auf 30 % → Final-Guard kappt auf 15.0 (nach PM)."""
        pf = {"portfolio_fit_score": 3, "ziel_gewichtung_pct": 5.0}
        trade = _make_trade("KAUFEN")
        result = _run_pipeline_with_mocks(
            trade, pf, pm_side_effect=self._pm_reweight_30
        )
        pf_result = result["portfolio_fit"]
        assert pf_result["ziel_gewichtung_pct"] == 15.0
        guard = pf_result["_final_guard"]
        assert guard["max_pct"] == 15.0
        assert guard["original"] == 30.0
        assert guard["gekappt"] is True
        # Konsistenz-Hinweis am finalen Trade
        assert result["trade"]["_final_guard_consistency"] is True

    def test_guard_runs_after_pm_call_order(self, tmp_path, state_dir):
        """Aufruf-Reihenfolge: erst PM, dann Guard (post-PM-Wert wird gekappt)."""
        import concilium.pipeline as pipeline_mod

        events: list[str] = []
        real_guard = pipeline_mod._apply_final_position_guard

        def pm_spy(*args, **kwargs):
            events.append("pm")
            return self._pm_reweight_30(*args, **kwargs)

        def guard_spy(portfolio_fit, trade=None):
            events.append("guard")
            return real_guard(portfolio_fit, trade)

        pf = {"portfolio_fit_score": 3, "ziel_gewichtung_pct": 5.0}
        with patch.object(
            pipeline_mod,
            "_apply_final_position_guard",
            side_effect=guard_spy,
        ):
            result = _run_pipeline_with_mocks(
                _make_trade("KAUFEN"), pf, pm_side_effect=pm_spy
            )
        assert events == ["pm", "guard"]  # PM VOR dem Guard
        assert result["portfolio_fit"]["ziel_gewichtung_pct"] == 15.0

    def test_skip_final_mode_guards_and_no_crash(self, tmp_path, state_dir):
        """skip_final=True: Guard läuft gegen vorhandenes portfolio_fit, kein Crash."""
        pf = {"portfolio_fit_score": 3, "ziel_gewichtung_pct": 25.0}
        result = _run_pipeline_with_mocks(
            _make_trade("KAUFEN"), pf, skip_final=True
        )
        assert result["_final_pending"] is True
        assert result["final"] is None
        assert result["portfolio_fit"]["ziel_gewichtung_pct"] == 15.0
        assert result["portfolio_fit"]["_final_guard"]["gekappt"] is True

    def test_portfolio_fit_none_no_crash(self, state_dir):
        """portfolio_fit None → kein Crash, PM/Genehmigung laufen normal."""
        result = _run_pipeline_with_mocks(_make_trade("KAUFEN"), None)
        assert result["portfolio_fit"] is None

    def test_genehmigt_ohne_reweight_bleibt_ungekappt(self, state_dir):
        """PM-GENEHMIGT ohne Re-Weight: 8 % bleibt, gekappt=False."""
        pf = {"portfolio_fit_score": 3, "ziel_gewichtung_pct": 8.0}
        result = _run_pipeline_with_mocks(_make_trade("KAUFEN"), pf)
        pf_result = result["portfolio_fit"]
        assert pf_result["ziel_gewichtung_pct"] == 8.0
        assert pf_result["_final_guard"]["gekappt"] is False


# --------------------------------------------------------------------------- #
# Tests: Report-Anzeige (Final-Guard-Hinweis)
# --------------------------------------------------------------------------- #


def _full_result(portfolio_fit: dict) -> dict:
    """Baut das vollständige result-dict für generate_report (offline)."""
    return {
        "ticker": "AAPL",
        "data": {
            "fundamentals": {"name": "Apple Inc.", "sector": "Technology"},
            "technicals": {"current_price": 230},
        },
        "analysts": {
            "fundamental": {"stimmung": "bullish", "score": 4,
                            "zusammenfassung": "Gut", "_raw": ""},
        },
        "debate": {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}},
        "trade": {
            "aktion": "KAUFEN",
            "rating": "KAUFEN",
            "zielkurs": 340,
            "stop_loss": 285,
            "positionsanteil": 7,
            "begründung": "Test",
            "zeithorizont": "Mittelfristig",
        },
        "risk": {
            "risiko_score": 3,
            "empfehlung": "GENEHMIGT",
            "auflagen": "keine",
        },
        "portfolio_fit": portfolio_fit,
        "final": {
            "entscheidung": "GENEHMIGT",
            "confidence": 4,
            "begründung": "Solide Fundamentals.",
        },
    }


class TestReportFinalGuardHinweis:
    """Test den Final-Guard-Hinweis im Portfolio-Fit-Report-Abschnitt."""

    def test_gekappt_zeigt_hinweis(self):
        """Bei gekappt=True: deutscher Final-Guard-Hinweis mit max/original."""
        result = _full_result({
            "portfolio_fit_score": 3,
            "ziel_gewichtung_pct": 15.0,
            "_final_guard": {
                "max_pct": 15.0,
                "original": 25.0,
                "gekappt": True,
            },
        })
        report = generate_report(result)
        assert (
            "**Final-Guard:** Ziel-Gewichtung auf max. 15.0 % begrenzt "
            "(hart, nicht durch LLM übersteuerbar; original 25.0)" in report
        )

    def test_gekappt_false_kein_hinweis(self):
        """Bei gekappt=False: kein Final-Guard-Hinweis."""
        result = _full_result({
            "portfolio_fit_score": 3,
            "ziel_gewichtung_pct": 8.0,
            "_final_guard": {
                "max_pct": 15.0,
                "original": 8.0,
                "gekappt": False,
            },
        })
        report = generate_report(result)
        assert "Final-Guard:" not in report

    def test_ohne_guard_metadaten_kein_hinweis(self):
        """Ohne _final_guard-Metadaten: kein Hinweis (Legacy-Verhalten)."""
        result = _full_result({
            "portfolio_fit_score": 3,
            "ziel_gewichtung_pct": 7.2,
        })
        report = generate_report(result)
        assert "Final-Guard:" not in report
