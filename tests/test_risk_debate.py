"""Tests für die 3-Perspektiven-Risiko-Debatte (Phase B).

Testet risk_debate() und die risk_manager-Dünne-Hülle:
1. Ablauf: 3 Perspektiven × 2 Runden + 1 Synthese = 7 LLM-Calls.
2. Runde 2 enthält die Runde-1-Argumente der anderen Perspektiven.
3. Synthese nutzt RISK_SCHEMA und liefert das vollständige risk-dict.
4. Best-effort: Perspektiven-Ausfall → leerer String, Synthese-Ausfall →
   defaults_for_schema(RISK_SCHEMA)-Fallback. Es wird nie gecrasht.
5. Rechnerische Felder (volatilität_annualisiert_pct,
   positionsgröße_rechnerisch_pct) und PCT-Normalisierung wie bisher.
6. Report: optionale Risiko-Debatte-Zeilen (nur wenn _risk_debate vorhanden).

Alle Tests sind offline (kein Netzwerk).
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import (  # noqa: E402
    risk_debate,
    risk_manager,
)
from concilium.llm import StructuredChatResult  # noqa: E402
from concilium.report import generate_report  # noqa: E402
from concilium.schemas import RISK_SCHEMA, defaults_for_schema  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MOCK_DATA = {
    "ticker": "TEST",
    "fundamentals": {},
    "technicals": {},
    "history": [],
    "sentiment": {},
}

_TRADE = {"aktion": "KAUFEN", "zielkurs": 120, "stop_loss": 90}

_RISK_JSON = json.dumps({
    "risiko_score": 3,
    "volatilität_bewertung": "moderat",
    "max_drawdown_schaetzung": "10 %",
    "positionsgröße_empfohlen": "5",
    "auflagen": "Stop-Loss bei 90 setzen",
    "empfehlung": "MODIFIZIERT",
})

_DEBATE_AGGRESSIVE_R1 = json.dumps({
    "confidence": 4,
    "name": "Aggressiv",
    "argumente": "AGGRESSIV-R1: Die Upside rechtfertigt das Risiko.",
})
_DEBATE_NEUTRAL_R1 = json.dumps({
    "confidence": 3,
    "name": "Neutral",
    "argumente": "NEUTRAL-R1: Volatilität und Drawdown sind moderat.",
})
_DEBATE_CONSERVATIVE_R1 = json.dumps({
    "confidence": 2,
    "name": "Konservativ",
    "argumente": "KONSERVATIV-R1: Kapitalerhalt hat Vorrang, Position verkleinern.",
})
_DEBATE_AGGRESSIVE_R2 = json.dumps({
    "confidence": 4,
    "name": "Aggressiv",
    "argumente": "AGGRESSIV-R2: Widerrufe die übermäßige Vorsicht mit Daten.",
})
_DEBATE_NEUTRAL_R2 = json.dumps({
    "confidence": 3,
    "name": "Neutral",
    "argumente": "NEUTRAL-R2: Halte an den Auflagen fest.",
})
_DEBATE_CONSERVATIVE_R2 = json.dumps({
    "confidence": 2,
    "name": "Konservativ",
    "argumente": "KONSERVATIV-R2: Enge Stops bleiben Pflicht.",
})


class _DebateMockLLM:
    """Mock-LLM: Debatten-JSON für Perspektiven, Risiko-JSON für die Synthese.

    Dispatch über response_format-Namen ("risk" = Synthese, "debate" =
    Perspektive) und den System-Prompt (Perspektiven-Unterscheidung).
    """

    def __init__(
        self,
        risk_json: str = _RISK_JSON,
        perspective_error: bool = False,
        synthesis_error: bool = False,
    ):
        self.captured: list[tuple[str, str, dict]] = []
        self._risk_json = risk_json
        self._perspective_error = perspective_error
        self._synthesis_error = synthesis_error

    def chat(self, messages, temperature=0.3, **kwargs):
        system = messages[0]["content"]
        user = messages[1]["content"]
        self.captured.append((system, user, {**dict(kwargs), "temperature": temperature}))
        rf_name = (
            kwargs.get("response_format", {}).get("json_schema", {}).get("name")
        )
        if rf_name == "risk":
            if self._synthesis_error:
                raise RuntimeError("Synthese-Ausfall")
            return StructuredChatResult(text=self._risk_json, response_format_used=True)
        # Perspektiven-Call (debate)
        if self._perspective_error:
            raise RuntimeError("Perspektiven-Ausfall")
        if "Aggressive Risk-Analyst" in system:
            if "Argumentation der anderen Risiko-Perspektiven" in user:
                return StructuredChatResult(
                    text=_DEBATE_AGGRESSIVE_R2, response_format_used=True
                )
            return StructuredChatResult(
                text=_DEBATE_AGGRESSIVE_R1, response_format_used=True
            )
        if "Neutrale Risk-Analyst" in system:
            if "Argumentation der anderen Risiko-Perspektiven" in user:
                return StructuredChatResult(
                    text=_DEBATE_NEUTRAL_R2, response_format_used=True
                )
            return StructuredChatResult(
                text=_DEBATE_NEUTRAL_R1, response_format_used=True
            )
        if "Konservative Risk-Analyst" in system:
            if "Argumentation der anderen Risiko-Perspektiven" in user:
                return StructuredChatResult(
                    text=_DEBATE_CONSERVATIVE_R2, response_format_used=True
                )
            return StructuredChatResult(
                text=_DEBATE_CONSERVATIVE_R1, response_format_used=True
            )
        raise AssertionError(f"Unbekannter System-Prompt: {system[:60]}")


def _calls_by_marker(llm: _DebateMockLLM, marker: str) -> list[tuple[str, str, dict]]:
    return [c for c in llm.captured if marker in c[0]]


# ---------------------------------------------------------------------------
# Tests: risk_debate Ablauf
# ---------------------------------------------------------------------------


class TestRiskDebateFlow:
    """Debatten-Ablauf: 7 Calls, Runde-2-Kontext, Synthese-Format."""

    def test_returns_full_risk_dict(self):
        """Ergebnis enthält alle risk-Felder + rechnerische Felder + _risk_debate."""
        llm = _DebateMockLLM()
        result = risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy")

        assert result["risiko_score"] == 3
        assert result["volatilität_bewertung"] == "moderat"
        assert result["max_drawdown_schaetzung"] == 10.0
        assert result["positionsgröße_empfohlen"] == 5.0
        assert result["auflagen"] == "Stop-Loss bei 90 setzen"
        assert result["empfehlung"] == "MODIFIZIERT"
        # Rechnerische Felder (keine Historie → None)
        assert result["volatilität_annualisiert_pct"] is None
        assert result["positionsgröße_rechnerisch_pct"] is None
        # Debatten-Argumente für den Report
        assert "_risk_debate" in result
        assert "Aggressiv" in result["_risk_debate"]["runde1"]
        assert "Konservativ" in result["_risk_debate"]["runde2"]

    def test_seven_llm_calls(self):
        """6 Perspektiven-Calls (3 × 2 Runden) + 1 Synthese-Call."""
        llm = _DebateMockLLM()
        risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy")

        assert len(llm.captured) == 7
        # 6 Perspektiven-Calls: je Perspektive genau 2 (Runde 1 + Runde 2)
        assert len(_calls_by_marker(llm, "Aggressive Risk-Analyst")) == 2
        assert len(_calls_by_marker(llm, "Neutrale Risk-Analyst")) == 2
        assert len(_calls_by_marker(llm, "Konservative Risk-Analyst")) == 2
        # 1 Synthese-Call
        assert len(_calls_by_marker(llm, "Risk-Manager und fasst die Risiko-Debatte")) == 1

    def test_perspectives_use_debate_schema_synthesis_risk_schema(self):
        """Perspektiven: DEBATE_SCHEMA + temperature 0.5; Synthese: RISK_SCHEMA."""
        llm = _DebateMockLLM()
        risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy")

        for system, _, kwargs in llm.captured:
            rf_name = kwargs.get("response_format", {}).get("json_schema", {}).get("name")
            if "Risk-Manager und fasst die Risiko-Debatte" in system:
                assert rf_name == "risk"
            else:
                assert rf_name == "debate"
                assert kwargs.get("temperature") == 0.5

    def test_risk_block_in_all_prompts(self):
        """Der rechnerische Risiko-Block steht in ALLEN 7 Prompts."""
        llm = _DebateMockLLM()
        risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy")

        for _system, user, _kwargs in llm.captured:
            assert "RECHNERISCHES RISIKO-MODELL" in user

    def test_runde2_contains_other_perspectives_arguments(self):
        """Runde 2: jede Perspektive sieht die Runde-1-Argumente der anderen beiden."""
        llm = _DebateMockLLM()
        risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy")

        # Aggressive Perspektive, Runde 2
        agg_r2 = [
            c for c in _calls_by_marker(llm, "Aggressive Risk-Analyst")
            if "Argumentation der anderen Risiko-Perspektiven" in c[1]
        ]
        assert len(agg_r2) == 1
        user = agg_r2[0][1]
        assert "--- Neutral (Runde 1) ---" in user
        assert "NEUTRAL-R1" in user
        assert "--- Konservativ (Runde 1) ---" in user
        assert "KONSERVATIV-R1" in user
        # Eigenes Runde-1-Argument ist NICHT im Kontext
        assert "AGGRESSIV-R1" not in user

    def test_synthesis_receives_all_six_arguments(self):
        """Die Synthese bekommt alle 6 Argumente (3 Perspektiven × 2 Runden)."""
        llm = _DebateMockLLM()
        risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy")

        synth = _calls_by_marker(llm, "Risk-Manager und fasst die Risiko-Debatte")
        assert len(synth) == 1
        user = synth[0][1]
        for marker in (
            "AGGRESSIV-R1", "NEUTRAL-R1", "KONSERVATIV-R1",
            "AGGRESSIV-R2", "NEUTRAL-R2", "KONSERVATIV-R2",
        ):
            assert marker in user, f"Marker '{marker}' fehlt im Synthese-Prompt"

    def test_feedback_context_in_all_prompts(self):
        """feedback_context hängt an Perspektiven- UND Synthese-Prompts."""
        llm = _DebateMockLLM()
        feedback = "=== DEIN TRACK-RECORD (letzte 10 Entscheidungen) ==="
        risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy", feedback_context=feedback)

        for _system, user, _kwargs in llm.captured:
            assert "TRACK-RECORD" in user

    def test_data_text_reused_not_rebuilt(self):
        """Mit data_text wird _build_data_text NICHT aufgerufen."""
        llm = _DebateMockLLM()
        with patch("concilium.agents._build_data_text") as mock_build:
            risk_debate(_TRADE, _MOCK_DATA, llm, data_text="pre-computed")
            mock_build.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Best-effort / Fallback
# ---------------------------------------------------------------------------


class TestRiskDebateBestEffort:
    """Fehlerfälle: nie crashen, Fallbacks greifen."""

    def test_perspective_failure_continues_with_empty_argument(self):
        """Perspektiven-Calls schlagen fehl → Debatte läuft weiter, Synthese kommt."""
        llm = _DebateMockLLM(perspective_error=True)
        result = risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy")

        # Synthese wurde trotzdem ausgeführt und liefert das risk-dict
        assert result["risiko_score"] == 3
        assert result["empfehlung"] == "MODIFIZIERT"
        # Debatten-Argumente sind leer, Synthese-Prompt enthält Platzhalter
        assert result["_risk_debate"]["runde1"]["Aggressiv"] == ""
        synth = _calls_by_marker(llm, "Risk-Manager und fasst die Risiko-Debatte")
        assert "(kein Argument geliefert)" in synth[0][1]

    def test_synthesis_failure_falls_back_to_schema_defaults(self):
        """Synthese schlägt fehl → defaults_for_schema(RISK_SCHEMA)-Fallback."""
        llm = _DebateMockLLM(synthesis_error=True)
        result = risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy")

        defaults = defaults_for_schema(RISK_SCHEMA)
        assert result["risiko_score"] == defaults["risiko_score"]
        assert result["empfehlung"] == "GENEHMIGT"
        # Normalisierung läuft auch im Fallback-Pfad
        assert result["max_drawdown_schaetzung"] is None
        assert result["positionsgröße_empfohlen"] is None
        # Rechnerische Felder werden trotzdem gesetzt
        assert result["volatilität_annualisiert_pct"] is None
        assert result["positionsgröße_rechnerisch_pct"] is None
        # Debatten-Argumente bleiben erhalten
        assert "KONSERVATIV-R1" in result["_risk_debate"]["runde1"]["Konservativ"]

    def test_string_pct_values_normalized(self):
        """Synthese liefert Strings ("10 %", "5") → floats im Ergebnis."""
        llm = _DebateMockLLM(risk_json=json.dumps({
            "risiko_score": 4,
            "max_drawdown_schaetzung": "12,5 %",
            "positionsgröße_empfohlen": "3,5",
            "empfehlung": "ABGELEHNT",
        }))
        result = risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy")
        assert result["max_drawdown_schaetzung"] == 12.5
        assert result["positionsgröße_empfohlen"] == 3.5
        assert isinstance(result["max_drawdown_schaetzung"], float)
        assert isinstance(result["positionsgröße_empfohlen"], float)


# ---------------------------------------------------------------------------
# Tests: risk_manager dünne Hülle
# ---------------------------------------------------------------------------


class TestRiskManagerThinWrapper:
    """risk_manager bleibt mit identischer Signatur bestehen und delegiert."""

    def test_delegates_to_risk_debate(self):
        """risk_manager ruft risk_debate mit identischen Argumenten auf
        und reicht die konfigurierten Runden explizit durch (Default 2)."""
        with patch("concilium.agents.risk_debate") as mock_rd:
            mock_rd.return_value = {"risiko_score": 2}
            result = risk_manager(
                _TRADE, _MOCK_DATA, "llm",
                data_text="dt", feedback_context="fb",
            )
        mock_rd.assert_called_once_with(
            _TRADE, _MOCK_DATA, "llm", data_text="dt", feedback_context="fb",
            rounds=2, model=None,
        )
        assert result == {"risiko_score": 2}

    def test_end_to_end_same_shape(self):
        """risk_manager liefert dasselbe dict wie risk_debate (alle Felder)."""
        llm = _DebateMockLLM()
        result = risk_manager(_TRADE, _MOCK_DATA, llm, data_text="dummy")
        for key in (
            "risiko_score", "volatilität_bewertung", "max_drawdown_schaetzung",
            "positionsgröße_empfohlen", "auflagen", "empfehlung",
            "volatilität_annualisiert_pct", "positionsgröße_rechnerisch_pct",
        ):
            assert key in result, f"Key '{key}' fehlt im risk_manager-Ergebnis"


# ---------------------------------------------------------------------------
# Tests: Runden-Konfiguration (CONCILIUM_RISK_DEBATE_ROUNDS)
# ---------------------------------------------------------------------------


class TestRiskDebateRounds:
    """rounds-Parameter: 1 Runde spart Runde 2, Default bleibt 2 Runden."""

    def test_rounds_one_skips_round_two(self):
        """rounds=1 → nur 3 Perspektiven-Calls + 1 Synthese = 4 LLM-Calls."""
        llm = _DebateMockLLM()
        result = risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy", rounds=1)

        assert len(llm.captured) == 4
        # Je Perspektive nur 1 Call (Runde 2 läuft nicht)
        assert len(_calls_by_marker(llm, "Aggressive Risk-Analyst")) == 1
        assert len(_calls_by_marker(llm, "Neutrale Risk-Analyst")) == 1
        assert len(_calls_by_marker(llm, "Konservative Risk-Analyst")) == 1
        assert len(_calls_by_marker(llm, "Risk-Manager und fasst die Risiko-Debatte")) == 1
        # Ergebnis bleibt vollständig (nie crashen)
        assert result["risiko_score"] == 3
        assert result["empfehlung"] == "MODIFIZIERT"

    def test_rounds_two_explicit_seven_calls(self):
        """rounds=2 (explizit) → wie bisher 7 Calls."""
        llm = _DebateMockLLM()
        risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy", rounds=2)
        assert len(llm.captured) == 7

    def test_rounds_none_reads_config_one(self, monkeypatch):
        """rounds=None + CONCILIUM_RISK_DEBATE_ROUNDS=1 → 4 Calls (Config-Fallback)."""
        monkeypatch.setenv("CONCILIUM_RISK_DEBATE_ROUNDS", "1")
        llm = _DebateMockLLM()
        risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy", rounds=None)
        assert len(llm.captured) == 4

    def test_rounds_none_reads_config_two(self, monkeypatch):
        """rounds=None + CONCILIUM_RISK_DEBATE_ROUNDS=2 → 7 Calls (Config-Fallback)."""
        monkeypatch.setenv("CONCILIUM_RISK_DEBATE_ROUNDS", "2")
        llm = _DebateMockLLM()
        risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy", rounds=None)
        assert len(llm.captured) == 7

    def test_synthesis_header_one_round(self):
        """rounds=1 → Synthese-Header sagt '1 Runde' statt '2 Runden'."""
        llm = _DebateMockLLM()
        risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy", rounds=1)
        synth = _calls_by_marker(llm, "Risk-Manager und fasst die Risiko-Debatte")
        assert "3 Perspektiven × 1 Runde" in synth[0][1]
        assert "2 Runden" not in synth[0][1]

    def test_synthesis_header_two_rounds(self):
        """rounds=2 → Synthese-Header sagt '2 Runden' (wie bisher)."""
        llm = _DebateMockLLM()
        risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy", rounds=2)
        synth = _calls_by_marker(llm, "Risk-Manager und fasst die Risiko-Debatte")
        assert "3 Perspektiven × 2 Runden" in synth[0][1]

    def test_synthesis_one_round_gets_round1_arguments(self):
        """rounds=1 → Synthese bekommt die 3 Runde-1-Argumente, keine Runde-2-Marker."""
        llm = _DebateMockLLM()
        risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy", rounds=1)
        synth = _calls_by_marker(llm, "Risk-Manager und fasst die Risiko-Debatte")
        user = synth[0][1]
        assert "AGGRESSIV-R1" in user
        assert "NEUTRAL-R1" in user
        assert "KONSERVATIV-R1" in user
        assert "AGGRESSIV-R2" not in user

    def test_risk_debate_key_rounds_one_only_runde1(self):
        """rounds=1 → _risk_debate enthält nur runde1, kein (leeres) runde2."""
        llm = _DebateMockLLM()
        result = risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy", rounds=1)
        assert set(result["_risk_debate"].keys()) == {"runde1"}
        assert "AGGRESSIV-R1" in result["_risk_debate"]["runde1"]["Aggressiv"]

    def test_risk_manager_passes_env_rounds(self, monkeypatch):
        """risk_manager liest die Config und reicht rounds explizit durch."""
        monkeypatch.setenv("CONCILIUM_RISK_DEBATE_ROUNDS", "1")
        llm = _DebateMockLLM()
        result = risk_manager(_TRADE, _MOCK_DATA, llm, data_text="dummy")
        assert len(llm.captured) == 4
        assert result["risiko_score"] == 3

    def test_risk_manager_default_rounds_two(self, monkeypatch):
        """Ohne Env: risk_manager → 7 Calls (Default bleibt 2 Runden)."""
        monkeypatch.delenv("CONCILIUM_RISK_DEBATE_ROUNDS", raising=False)
        llm = _DebateMockLLM()
        risk_manager(_TRADE, _MOCK_DATA, llm, data_text="dummy")
        assert len(llm.captured) == 7

    def test_report_compatible_with_rounds_one_result(self):
        """Integration: rounds=1-Ergebnis (ohne runde2) bleibt report-kompatibel."""
        llm = _DebateMockLLM()
        risk = risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy", rounds=1)
        report = generate_report(TestRiskDebateReport._base_result(risk))
        assert "**Risiko-Debatte:** Aggressiv vs Neutral vs Konservativ" in report
        assert "**Aggressiv:** AGGRESSIV-R1" in report


# ---------------------------------------------------------------------------
# Tests: Report-Anzeige
# ---------------------------------------------------------------------------


class TestRiskDebateReport:
    """Optionale Risiko-Debatte-Zeilen im Report."""

    @staticmethod
    def _base_result(risk: dict) -> dict:
        return {
            "ticker": "AAA",
            "data": {"ticker": "AAA", "fundamentals": {}, "technicals": {},
                     "sentiment": {}, "news": [], "history": []},
            "analysts": {
                "fundamental": {"stimmung": "bullish", "score": 4, "_raw": ""},
                "technical": {"stimmung": "bullish", "score": 4, "_raw": ""},
                "sentiment": {"stimmung": "neutral", "score": 3, "_raw": ""},
                "macro_news": {"stimmung": "neutral", "score": 3, "_raw": ""},
            },
            "debate": {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}},
            "trade": {"aktion": "KAUFEN", "positionsanteil": 5},
            "risk": risk,
            "final": {"entscheidung": "GENEHMIGT", "confidence": 4},
        }

    def test_report_shows_debate_when_present(self):
        """Mit _risk_debate werden die Kernargumente angezeigt."""
        result = self._base_result({
            "risiko_score": 3,
            "empfehlung": "MODIFIZIERT",
            "_risk_debate": {
                "runde1": {"Aggressiv": "", "Neutral": "", "Konservativ": ""},
                "runde2": {
                    "Aggressiv": "AGGRESSIV-R2: Upside rechtfertigt das Risiko.",
                    "Neutral": "NEUTRAL-R2: Auflagen einhalten.",
                    "Konservativ": "KONSERVATIV-R2: Enge Stops bleiben Pflicht.",
                },
            },
        })
        report = generate_report(result)
        assert "**Risiko-Debatte:** Aggressiv vs Neutral vs Konservativ" in report
        assert "**Aggressiv:** AGGRESSIV-R2" in report
        assert "**Neutral:** NEUTRAL-R2" in report
        assert "**Konservativ:** KONSERVATIV-R2" in report

    def test_report_no_debate_section_without_key(self):
        """Altes risk-dict ohne _risk_debate → Report unverändert."""
        result = self._base_result({"risiko_score": 3, "empfehlung": "GENEHMIGT"})
        report = generate_report(result)
        assert "Aggressiv vs Neutral vs Konservativ" not in report

    def test_report_falls_back_to_runde1(self):
        """Wenn nur Runde 1 vorhanden ist, werden deren Argumente gezeigt."""
        result = self._base_result({
            "risiko_score": 2,
            "empfehlung": "GENEHMIGT",
            "_risk_debate": {
                "runde1": {
                    "Aggressiv": "AGGRESSIV-R1: Wachstum trägt.",
                    "Neutral": "",
                    "Konservativ": "",
                },
                "runde2": {"Aggressiv": "", "Neutral": "", "Konservativ": ""},
            },
        })
        report = generate_report(result)
        assert "**Risiko-Debatte:** Aggressiv vs Neutral vs Konservativ" in report
        assert "**Aggressiv:** AGGRESSIV-R1" in report
        # Leere Argumente werden nicht angezeigt
        assert "**Neutral:**" not in report

    def test_report_shows_result_of_risk_debate_end_to_end(self):
        """Integration: Das risk_debate-Ergebnis ist report-kompatibel."""
        llm = _DebateMockLLM()
        risk = risk_debate(_TRADE, _MOCK_DATA, llm, data_text="dummy")
        report = generate_report(self._base_result(risk))
        assert "**Risiko-Debatte:** Aggressiv vs Neutral vs Konservativ" in report
