"""Tests für den Deep-Think/Quick-Think Modell-Split (analog TradingAgents).

Abgedeckte Ebenen:
1. ``LLMClient.chat(model=...)`` — Override landet im Payload, Default unverändert,
   Fallback-Modell bleibt vom Override unberührt.
2. ``_call_agent`` — reicht ``model`` an ``llm.chat`` durch (structured + plain).
3. Agenten-Funktionen — reichen ``model`` an alle ihre LLM-Calls durch.
4. ``config`` — LLM_DEEP_THINK_MODEL / LLM_QUICK_THINK_MODEL (Default leer).
5. ``run_pipeline`` / ``run_portfolio`` — Verdrahtung quick→Analysten/Debatte/Trader,
   deep→Risiko-Debatte/Trade-Revision/PM; param > env; leer/None = kein Split.
6. Konfigurations-Fingerprint — nimmt Split-Modelle nur bei aktivem Split auf.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium import config  # noqa: E402
from concilium.agents import (  # noqa: E402
    _call_agent,
    analyst_team,
    debate,
    ensemble_trader,
    portfolio_manager,
    risk_debate,
    risk_manager,
    trade_revision,
    trader,
)
from concilium.llm import LLMClient  # noqa: E402
from concilium.pipeline import (  # noqa: E402
    _pipeline_fingerprint,
    run_pipeline,
    run_portfolio,
)


@pytest.fixture(autouse=True)
def _clean_model_split_env(monkeypatch):
    """Entfernt Split-Env-Vars — Tests steuern die Konfiguration explizit."""
    for var in ("LLM_DEEP_THINK_MODEL", "LLM_QUICK_THINK_MODEL", "LLM_FALLBACK_MODEL"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Hilfs-Mocks
# ---------------------------------------------------------------------------


class _CapturingChatLLM:
    """Mock-LLM, das alle chat()-KwArgs aufzeichnet (wie _CapturingLLM in test_llm.py)."""

    def __init__(self, raw: str = '{"rolle": "Test", "score": 3}'):
        self.calls: list[dict] = []
        self._raw = raw

    def chat(self, messages, temperature: float = 0.3, **kwargs):
        self.calls.append({"temperature": temperature, **kwargs})
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult

            return StructuredChatResult(text=self._raw, response_format_used=True)
        return self._raw


class _MockResponse:
    """Mock für requests.Response (gleiche Strategie wie test_llm_fallback.py)."""

    def __init__(self, status_code: int, content: str = "Test-Antwort"):
        self.status_code = status_code
        self._content = content
        self.text = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")  # noqa: TRY002

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


_MINIMAL_ANALYSTS = {
    "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
    "macro_news": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
}

_MINIMAL_TRADE = {"rolle": "Trader", "aktion": "KAUFEN", "rating": "KAUFEN", "_raw": ""}
_MINIMAL_RISK = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}
_MINIMAL_DATA = {"ticker": "TEST", "technicals": {}, "history": []}

_TRADE_JSON = '{"aktion": "KAUFEN", "zielkurs": 110, "stop_loss": 90, "positionsanteil": 2}'
_DEBATE_JSON = '{"argumente": "Bull-Argument", "confidence": 4}'
_RISK_JSON = '{"risiko_score": 3, "empfehlung": "GENEHMIGT"}'
_FINAL_JSON = '{"entscheidung": "GENEHMIGT", "confidence": 4, "begründung": "Ok."}'


def _assert_all_calls_have_model(llm: _CapturingChatLLM, expected: str | None) -> None:
    """Jeder aufgezeichnete chat()-Call muss model=expected geführt haben."""
    assert llm.calls, "Es wurden keine LLM-Calls aufgezeichnet"
    for i, call in enumerate(llm.calls):
        assert call.get("model") == expected, f"Call {i}: model={call.get('model')!r} != {expected!r}"


# ---------------------------------------------------------------------------
# 1. _call_agent — model-Durchreichung an llm.chat
# ---------------------------------------------------------------------------


class TestCallAgentModelPassthrough:
    """_call_agent reicht model an llm.chat durch (beide Pfade)."""

    def test_structured_path_passes_model(self):
        llm = _CapturingChatLLM(raw='{"rolle": "Test", "score": 3}')
        _call_agent(
            llm, "system prompt", "user text",
            response_format={"type": "json_object"},
            structured=True,
            model="deep-model-x",
        )
        assert len(llm.calls) == 1
        assert llm.calls[0]["model"] == "deep-model-x"

    def test_plain_path_passes_model(self):
        llm = _CapturingChatLLM()
        _call_agent(llm, "system prompt", "user text", model="quick-model-y")
        assert len(llm.calls) == 1
        assert llm.calls[0]["model"] == "quick-model-y"
        # Plain-Pfad: kein as_structured-Kwarg (Default False)
        assert llm.calls[0].get("as_structured") is None

    def test_default_model_is_none(self):
        """Ohne model-Parameter wird model=None übergeben (bisheriges Verhalten)."""
        llm = _CapturingChatLLM(raw='{"rolle": "Test", "score": 3}')
        _call_agent(
            llm, "system prompt", "user text",
            response_format={"type": "json_object"},
            structured=True,
        )
        assert llm.calls[0]["model"] is None


# ---------------------------------------------------------------------------
# 2. Agenten-Funktionen — model-Durchreichung
# ---------------------------------------------------------------------------


class TestAgentFunctionsModelPassthrough:
    """Jede Agenten-Funktion reicht model an ALLE ihre LLM-Calls durch."""

    def test_analyst_team_passes_model_to_all_four(self):
        llm = _CapturingChatLLM(raw='{"stimmung": "bullish", "score": 4}')
        result = analyst_team(
            {"technicals": {}}, llm, data_text="DATEN", model="quick-x",
        )
        assert len(llm.calls) == 4
        _assert_all_calls_have_model(llm, "quick-x")
        assert set(result.keys()) >= {"fundamental", "technical", "sentiment", "macro_news"}

    def test_analyst_team_default_none(self):
        llm = _CapturingChatLLM(raw='{"stimmung": "bullish", "score": 4}')
        analyst_team({"technicals": {}}, llm, data_text="DATEN")
        _assert_all_calls_have_model(llm, None)

    def test_debate_passes_model_to_bull_and_bear(self):
        llm = _CapturingChatLLM(raw=_DEBATE_JSON)
        result = debate(_MINIMAL_ANALYSTS, llm, rounds=1, model="quick-x")
        assert len(llm.calls) == 2
        _assert_all_calls_have_model(llm, "quick-x")
        assert result["rounds"] == 1

    def test_debate_default_none(self):
        llm = _CapturingChatLLM(raw=_DEBATE_JSON)
        debate(_MINIMAL_ANALYSTS, llm)
        _assert_all_calls_have_model(llm, None)

    def test_trader_passes_model(self):
        llm = _CapturingChatLLM(raw=_TRADE_JSON)
        trader(
            _MINIMAL_ANALYSTS,
            {"bull": {"_raw": "b"}, "bear": {"_raw": "br"}},
            llm,
            model="quick-x",
        )
        _assert_all_calls_have_model(llm, "quick-x")

    def test_trader_default_none(self):
        llm = _CapturingChatLLM(raw=_TRADE_JSON)
        trader(_MINIMAL_ANALYSTS, {"bull": {}, "bear": {}}, llm)
        _assert_all_calls_have_model(llm, None)

    def test_ensemble_trader_passes_model_to_every_run(self):
        llm = _CapturingChatLLM(raw=_TRADE_JSON)
        ensemble_trader(
            _MINIMAL_ANALYSTS,
            {"bull": {"_raw": "b"}, "bear": {"_raw": "br"}},
            llm,
            runs=2,
            model="quick-x",
        )
        assert len(llm.calls) == 2
        _assert_all_calls_have_model(llm, "quick-x")

    def test_risk_debate_passes_model_to_perspectives_and_synthesis(self):
        """3 Perspektiven (Runde 1) + Synthese — alle mit demselben Override."""
        llm = _CapturingChatLLM(raw=_RISK_JSON)
        risk_debate(
            _MINIMAL_TRADE, _MINIMAL_DATA, llm,
            data_text="DATEN", rounds=1, model="deep-x",
        )
        # Runde 1: 3 Perspektiven + 1 Synthese
        assert len(llm.calls) == 4
        _assert_all_calls_have_model(llm, "deep-x")

    def test_risk_debate_round2_also_gets_model(self):
        """Runde 2 (Reaktionen) läuft ebenfalls mit dem Override-Modell."""
        llm = _CapturingChatLLM(raw=_DEBATE_JSON)
        risk_debate(
            _MINIMAL_TRADE, _MINIMAL_DATA, llm,
            data_text="DATEN", rounds=2, model="deep-x",
        )
        # 3 Perspektiven × 2 Runden + 1 Synthese
        assert len(llm.calls) == 7
        _assert_all_calls_have_model(llm, "deep-x")

    def test_risk_manager_passes_model(self):
        llm = _CapturingChatLLM(raw=_RISK_JSON)
        risk_manager(_MINIMAL_TRADE, _MINIMAL_DATA, llm, data_text="DATEN", model="deep-x")
        # Default-Runden (2) → 3 + 3 + 1 Calls
        assert len(llm.calls) == 7
        _assert_all_calls_have_model(llm, "deep-x")

    def test_portfolio_manager_passes_model(self):
        llm = _CapturingChatLLM(raw=_FINAL_JSON)
        portfolio_manager(_MINIMAL_TRADE, _MINIMAL_RISK, llm, model="deep-x")
        _assert_all_calls_have_model(llm, "deep-x")

    def test_portfolio_manager_default_none(self):
        llm = _CapturingChatLLM(raw=_FINAL_JSON)
        portfolio_manager(_MINIMAL_TRADE, _MINIMAL_RISK, llm)
        _assert_all_calls_have_model(llm, None)

    def test_trade_revision_passes_model(self):
        llm = _CapturingChatLLM(raw=_TRADE_JSON)
        trade_revision(_MINIMAL_TRADE, _MINIMAL_RISK, None, llm, model="deep-x")
        _assert_all_calls_have_model(llm, "deep-x")

    def test_trade_revision_default_none(self):
        llm = _CapturingChatLLM(raw=_TRADE_JSON)
        trade_revision(_MINIMAL_TRADE, _MINIMAL_RISK, None, llm)
        _assert_all_calls_have_model(llm, None)


# ---------------------------------------------------------------------------
# 3. LLMClient.chat — Override im Payload
# ---------------------------------------------------------------------------


class TestLLMClientChatModelOverride:
    """chat(model=...) setzt das model-Feld im Payload; Default bleibt unverändert."""

    def _make_client(self, **kwargs) -> LLMClient:
        defaults = {
            "base_url": "http://fake:8080/v1",
            "api_key": "test-key",
            "model": "primary-model",
        }
        defaults.update(kwargs)
        return LLMClient(**defaults)

    def test_model_override_lands_in_payload(self):
        client = self._make_client()
        with patch("concilium.llm.requests.post", return_value=_MockResponse(200)) as mock_post:
            client.chat([{"role": "user", "content": "Test"}], model="override-model")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "override-model"

    def test_default_uses_primary_model(self):
        client = self._make_client()
        with patch("concilium.llm.requests.post", return_value=_MockResponse(200)) as mock_post:
            client.chat([{"role": "user", "content": "Test"}])
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "primary-model"

    def test_none_model_uses_primary_model(self):
        client = self._make_client()
        with patch("concilium.llm.requests.post", return_value=_MockResponse(200)) as mock_post:
            client.chat([{"role": "user", "content": "Test"}], model=None)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "primary-model"

    def test_empty_model_string_uses_primary_model(self):
        """Leerer String (z. B. aus einer leeren Env-Var) = kein Split."""
        client = self._make_client()
        with patch("concilium.llm.requests.post", return_value=_MockResponse(200)) as mock_post:
            client.chat([{"role": "user", "content": "Test"}], model="")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "primary-model"

    def test_fallback_uses_configured_fallback_model_not_override(self):
        """429-Erschöpfung → Fallback-Modell, unabhängig vom Override."""
        client = self._make_client(fallback_model="fallback-model")
        responses = [
            _MockResponse(429, "Rate limited"),
            _MockResponse(429, "Rate limited"),
            _MockResponse(429, "Rate limited"),
            _MockResponse(200, "Erfolg vom Fallback"),
        ]
        with patch("concilium.llm.requests.post", side_effect=responses) as mock_post:
            with patch("concilium.llm.time.sleep"):
                result = client.chat(
                    [{"role": "user", "content": "Test"}], model="override-model",
                )
        assert result == "Erfolg vom Fallback"
        assert mock_post.call_count == 4
        models = [c.kwargs["json"]["model"] for c in mock_post.call_args_list]
        assert models[:3] == ["override-model"] * 3  # Primär: Override
        assert models[3] == "fallback-model"  # Fallback: konfiguriertes Fallback-Modell


# ---------------------------------------------------------------------------
# 4. config — LLM_DEEP_THINK_MODEL / LLM_QUICK_THINK_MODEL
# ---------------------------------------------------------------------------


class TestConfigModelSplit:
    """Env-Vars: Default leer, Lazy-Loading, Typ str."""

    def test_defaults_empty(self):
        assert config.llm_deep_think_model() == ""
        assert config.llm_quick_think_model() == ""

    def test_env_set_after_import(self, monkeypatch):
        monkeypatch.setenv("LLM_DEEP_THINK_MODEL", "glm-5.3-pro")
        monkeypatch.setenv("LLM_QUICK_THINK_MODEL", "glm-5.3-flash")
        assert config.llm_deep_think_model() == "glm-5.3-pro"
        assert config.llm_quick_think_model() == "glm-5.3-flash"

    def test_lazy_reload_between_calls(self, monkeypatch):
        monkeypatch.setenv("LLM_DEEP_THINK_MODEL", "erstes-modell")
        first = config.llm_deep_think_model()
        monkeypatch.setenv("LLM_DEEP_THINK_MODEL", "zweites-modell")
        second = config.llm_deep_think_model()
        assert first == "erstes-modell"
        assert second == "zweites-modell"


# ---------------------------------------------------------------------------
# 5. Pipeline-Verdrahtung
# ---------------------------------------------------------------------------


_MOCK_DATA = {
    "ticker": "TEST",
    "fundamentals": {"name": "TestCo", "sector": "Tech"},
    "technicals": {"current_price": 100},
    "sentiment": {},
    "news": [],
}

_MOCK_ANALYSTS = dict(_MINIMAL_ANALYSTS)
_MOCK_DEBATE = {"bull": {"_raw": "Bull argument"}, "bear": {"_raw": "Bear argument"}}
_MOCK_TRADE = {
    "rolle": "Trader",
    "aktion": "KAUFEN",
    "rating": "KAUFEN",
    "zielkurs": 115,
    "stop_loss": 92,
    "positionsanteil": 3,
    "_raw": "",
}
_MOCK_REVISED_TRADE = {
    "rolle": "Trader",
    "aktion": "HALTEN",
    "rating": "HALTEN",
    "zielkurs": None,
    "stop_loss": None,
    "positionsanteil": 0,
    "_raw": "revised",
}
_MOCK_RISK = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}
_MOCK_PORTFOLIO_FIT = {"portfolio_fit_score": 2}
_MOCK_FINAL = {"entscheidung": "GENEHMIGT", "confidence": 4, "begründung": "Ok."}


def _patch_pipeline_agents(**overrides):
    """Patcht alle Agenten-Funktionen + Daten + Journal (Konvention test_pipeline_resume)."""
    defaults = {
        "collect_ticker_data": MagicMock(return_value=_MOCK_DATA),
        "analyst_team": MagicMock(return_value=dict(_MOCK_ANALYSTS)),
        "debate": MagicMock(return_value=dict(_MOCK_DEBATE)),
        "trader": MagicMock(return_value=dict(_MOCK_TRADE)),
        "ensemble_trader": MagicMock(return_value=dict(_MOCK_TRADE)),
        "risk_manager": MagicMock(return_value=dict(_MOCK_RISK)),
        "fetch_portfolio_positions": MagicMock(return_value=[]),
        "portfolio_fit_agent": MagicMock(return_value=dict(_MOCK_PORTFOLIO_FIT)),
        "trade_revision": MagicMock(return_value=dict(_MOCK_REVISED_TRADE)),
        "portfolio_manager": MagicMock(return_value=dict(_MOCK_FINAL)),
        "build_feedback_context": MagicMock(return_value=""),
        "build_reflection_context": MagicMock(return_value=""),
    }
    defaults.update(overrides)
    return defaults


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Isoliertes state-Verzeichnis für Pipeline-Tests."""
    d = tmp_path / "state"
    d.mkdir()
    monkeypatch.setenv("CONCILIUM_STATE_DIR", str(d))
    return str(d)


class TestRunPipelineModelSplit:
    """run_pipeline reicht deep/quick_think_model an die richtigen Agenten durch."""

    def test_explicit_models_routed_to_correct_agents(self, state_dir):
        agents = _patch_pipeline_agents()
        with patch.multiple("concilium.pipeline", **agents), \
             patch("concilium.journal.append_decision"):
            run_pipeline(
                "TEST",
                llm=MagicMock(),
                ensemble=False,
                deep_think_model="deep-x",
                quick_think_model="quick-y",
            )

        # Quick-Think: Analysten, Bull/Bear-Debatte, Trader
        assert agents["analyst_team"].call_args.kwargs.get("model") == "quick-y"
        assert agents["debate"].call_args.kwargs.get("model") == "quick-y"
        assert agents["trader"].call_args.kwargs.get("model") == "quick-y"
        # Deep-Think: Risiko-Debatte, Trade-Revision, Portfolio-Manager
        assert agents["risk_manager"].call_args.kwargs.get("model") == "deep-x"
        assert agents["trade_revision"].call_args.kwargs.get("model") == "deep-x"
        assert agents["portfolio_manager"].call_args.kwargs.get("model") == "deep-x"

    def test_explicit_models_routed_in_ensemble_mode(self, state_dir):
        """Ensemble-Pfad (Default): ensemble_trader bekommt das Quick-Think-Modell."""
        agents = _patch_pipeline_agents()
        with patch.multiple("concilium.pipeline", **agents), \
             patch("concilium.journal.append_decision"):
            run_pipeline(
                "TEST",
                llm=MagicMock(),
                ensemble=True,
                deep_think_model="deep-x",
                quick_think_model="quick-y",
            )

        assert agents["ensemble_trader"].call_args.kwargs.get("model") == "quick-y"
        assert agents["trader"].call_count == 0  # Single-Pfad nicht genutzt
        assert agents["risk_manager"].call_args.kwargs.get("model") == "deep-x"

    def test_env_fallback_when_params_none(self, state_dir, monkeypatch):
        """Parameter None → Modelle aus der Env lesen."""
        monkeypatch.setenv("LLM_DEEP_THINK_MODEL", "env-deep")
        monkeypatch.setenv("LLM_QUICK_THINK_MODEL", "env-quick")
        agents = _patch_pipeline_agents()
        with patch.multiple("concilium.pipeline", **agents), \
             patch("concilium.journal.append_decision"):
            run_pipeline("TEST", llm=MagicMock(), ensemble=False)

        assert agents["analyst_team"].call_args.kwargs.get("model") == "env-quick"
        assert agents["debate"].call_args.kwargs.get("model") == "env-quick"
        assert agents["trader"].call_args.kwargs.get("model") == "env-quick"
        assert agents["risk_manager"].call_args.kwargs.get("model") == "env-deep"
        assert agents["trade_revision"].call_args.kwargs.get("model") == "env-deep"
        assert agents["portfolio_manager"].call_args.kwargs.get("model") == "env-deep"

    def test_params_beat_env(self, state_dir, monkeypatch):
        monkeypatch.setenv("LLM_DEEP_THINK_MODEL", "env-deep")
        monkeypatch.setenv("LLM_QUICK_THINK_MODEL", "env-quick")
        agents = _patch_pipeline_agents()
        with patch.multiple("concilium.pipeline", **agents), \
             patch("concilium.journal.append_decision"):
            run_pipeline(
                "TEST",
                llm=MagicMock(),
                ensemble=False,
                deep_think_model="param-deep",
                quick_think_model="param-quick",
            )

        assert agents["analyst_team"].call_args.kwargs.get("model") == "param-quick"
        assert agents["risk_manager"].call_args.kwargs.get("model") == "param-deep"

    def test_no_split_passes_none_backward_compatible(self, state_dir):
        """Ohne Split (Env leer, Parameter None) → model=None überall (bisheriges Verhalten)."""
        agents = _patch_pipeline_agents()
        with patch.multiple("concilium.pipeline", **agents), \
             patch("concilium.journal.append_decision"):
            run_pipeline("TEST", llm=MagicMock(), ensemble=False)

        for name in ("analyst_team", "debate", "trader", "risk_manager",
                     "trade_revision", "portfolio_manager"):
            assert agents[name].call_args.kwargs.get("model") is None, name

    def test_env_empty_string_means_no_split(self, state_dir, monkeypatch):
        monkeypatch.setenv("LLM_DEEP_THINK_MODEL", "")
        monkeypatch.setenv("LLM_QUICK_THINK_MODEL", "")
        agents = _patch_pipeline_agents()
        with patch.multiple("concilium.pipeline", **agents), \
             patch("concilium.journal.append_decision"):
            run_pipeline("TEST", llm=MagicMock(), ensemble=False)

        for name in ("analyst_team", "debate", "trader", "risk_manager",
                     "trade_revision", "portfolio_manager"):
            assert agents[name].call_args.kwargs.get("model") is None, name


class TestRunPortfolioModelSplit:
    """run_portfolio reicht die Modelle an Einzel-Pipelines UND den Phase-2-PM durch."""

    def _mock_pipeline_result(self) -> dict:
        return {
            "ticker": "TEST",
            "data": {"fundamentals": {}, "technicals": {}, "history": []},
            "no_llm": False,
            "_feedback_context": "",
            "_reflection_context": "",
            "trade": dict(_MOCK_TRADE),
            "risk": dict(_MOCK_RISK),
            "portfolio_fit": None,
        }

    def test_explicit_models_forwarded(self):
        with patch("concilium.pipeline.run_pipeline") as mock_run, \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.portfolio_analysis.run_portfolio_analysis", return_value={}), \
             patch("concilium.pipeline.portfolio_manager") as mock_pm, \
             patch("concilium.journal.append_decision"):
            mock_run.return_value = self._mock_pipeline_result()
            mock_pm.return_value = dict(_MOCK_FINAL)

            run_portfolio(
                ["TEST"],
                llm=MagicMock(),
                deep_think_model="deep-x",
                quick_think_model="quick-y",
            )

        assert mock_run.call_args.kwargs.get("deep_think_model") == "deep-x"
        assert mock_run.call_args.kwargs.get("quick_think_model") == "quick-y"
        # Phase 2: direkter PM-Call bekommt das Deep-Think-Modell
        assert mock_pm.call_args.kwargs.get("model") == "deep-x"

    def test_env_resolved_once_for_phase2_pm(self, monkeypatch):
        """Phase-2-PM (direkter Call!) nutzt dasselbe Deep-Think-Modell wie die Pipelines."""
        monkeypatch.setenv("LLM_DEEP_THINK_MODEL", "env-deep")
        with patch("concilium.pipeline.run_pipeline") as mock_run, \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.portfolio_analysis.run_portfolio_analysis", return_value={}), \
             patch("concilium.pipeline.portfolio_manager") as mock_pm, \
             patch("concilium.journal.append_decision"):
            mock_run.return_value = self._mock_pipeline_result()
            mock_pm.return_value = dict(_MOCK_FINAL)

            run_portfolio(["TEST"], llm=MagicMock())

        assert mock_run.call_args.kwargs.get("deep_think_model") == "env-deep"
        assert mock_pm.call_args.kwargs.get("model") == "env-deep"


# ---------------------------------------------------------------------------
# 6. Konfigurations-Fingerprint
# ---------------------------------------------------------------------------


class TestPipelineFingerprintModelSplit:
    """Split-Modelle gehören in den Resume-Fingerprint — aber nur bei aktivem Split."""

    _BASE_KWARGS = {
        "ensemble": True,
        "ensemble_runs": 3,
        "peers": None,
        "debate_rounds": 1,
        "backtest": False,
    }

    def test_no_split_matches_legacy_fingerprint(self):
        """Ohne Split → identischer Fingerprint wie vor der Änderung (Altdaten-Kompatibilität)."""
        legacy = _pipeline_fingerprint(**self._BASE_KWARGS)
        no_split = _pipeline_fingerprint(
            deep_think_model=None, quick_think_model=None, **self._BASE_KWARGS,
        )
        empty_split = _pipeline_fingerprint(
            deep_think_model="", quick_think_model="", **self._BASE_KWARGS,
        )
        assert legacy == no_split == empty_split
        assert "deep_think_model" not in legacy
        assert "quick_think_model" not in legacy

    def test_active_split_changes_fingerprint(self):
        legacy = _pipeline_fingerprint(**self._BASE_KWARGS)
        with_split = _pipeline_fingerprint(
            deep_think_model="deep-x", quick_think_model="quick-y", **self._BASE_KWARGS,
        )
        assert with_split != legacy

    def test_different_models_produce_different_fingerprints(self):
        a = _pipeline_fingerprint(
            deep_think_model="model-a", quick_think_model=None, **self._BASE_KWARGS,
        )
        b = _pipeline_fingerprint(
            deep_think_model="model-b", quick_think_model=None, **self._BASE_KWARGS,
        )
        c = _pipeline_fingerprint(
            deep_think_model="model-a", quick_think_model="quick-y", **self._BASE_KWARGS,
        )
        assert a != b
        assert a != c

    def test_same_models_produce_same_fingerprint(self):
        a = _pipeline_fingerprint(
            deep_think_model="deep-x", quick_think_model="quick-y", **self._BASE_KWARGS,
        )
        b = _pipeline_fingerprint(
            deep_think_model="deep-x", quick_think_model="quick-y", **self._BASE_KWARGS,
        )
        assert a == b
