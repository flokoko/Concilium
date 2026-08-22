"""Tests für strukturierte LLM-Outputs (Phase 0).

Testet:
1. llm.py: chat(as_structured=True, response_format=...) → StructuredChatResult
2. llm.py: 400-Fallback (Provider lehnt response_format ab → erneut ohne, response_format_used=False)
3. llm.py: chat() ohne as_structured → weiterhin str (Rückwärtskompatibilität)
4. Agenten mit strukturierten Mock-LLMs (response_format_used=True): dict mit Schema-Keys
5. Agenten mit Fallback-Mock-LLMs (response_format_used=False): parse_json-Verhalten wie bisher

Alle Tests sind offline (kein Netzwerk).
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import (  # noqa: E402
    debate,
    portfolio_manager,
    risk_manager,
    trade_revision,
    trader,
)
from concilium.llm import LLMClient, StructuredChatResult  # noqa: E402
from concilium.schemas import (  # noqa: E402
    ANALYST_FUNDAMENTAL_SCHEMA,
    ANALYST_SENTIMENT_SCHEMA,
    ANALYST_TECHNICAL_SCHEMA,
    DEBATE_SCHEMA,
    FINAL_SCHEMA,
    RISK_SCHEMA,
    TRADE_SCHEMA,
    validate_structured,
)

# ---------------------------------------------------------------------------
# Helper: Validierung
# ---------------------------------------------------------------------------

def _assert_has_keys(result: dict, keys: list[str]) -> None:
    """Prüft, dass das dict alle erwarteten Keys enthält."""
    for key in keys:
        assert key in result, f"Key '{key}' fehlt im Ergebnis: {list(result.keys())}"


# ---------------------------------------------------------------------------
# Mock-LLMs
# ---------------------------------------------------------------------------

class _StructuredMockLLM:
    """Mock-LLM, der StructuredChatResult zurückgibt (strukturierter Pfad).

    Gibt eine vordefinierte JSON-Antwort als StructuredChatResult mit
    response_format_used=True zurück.
    """

    def __init__(self, response_json: str, response_format_used: bool = True):
        self._response = response_json
        self._rfu = response_format_used
        self.captured_messages: list[list[dict]] = []
        self.captured_kwargs: list[dict] = []

    def chat(self, messages, temperature=0.3, **kwargs):
        self.captured_messages.append(messages)
        self.captured_kwargs.append(kwargs)
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            return StructuredChatResult(text=self._response, response_format_used=self._rfu)
        return self._response


class _FallbackMockLLM:
    """Mock-LLM, der str zurückgibt (Fallback-Pfad, kein StructuredChatResult).

    Simuliert, dass der Provider response_format nicht unterstützt:
    _call_agent bekommt ein str und nutzt parse_json.
    """

    def __init__(self, response: str):
        self._response = response
        self.captured_messages: list[list[dict]] = []

    def chat(self, messages, temperature=0.3, **kwargs):
        self.captured_messages.append(messages)
        # Wenn as_structured=True: gib str zurück (kein StructuredChatResult)
        # → _call_agent fällt in den else-Zweig und nutzt parse_json
        return self._response


# ---------------------------------------------------------------------------
# Fixtures: Testdaten
# ---------------------------------------------------------------------------

_ANALYSTS = {
    "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
    "technicals": {"current_price": 100.0},
}

_DEBATE = {
    "bull": {"_raw": "Bull text", "argumente": "Bull argument"},
    "bear": {"_raw": "Bear text", "argumente": "Bear argument"},
    "bull_confidence": 4,
    "bear_confidence": 2,
}

_TRADE_JSON = json.dumps({
    "rolle": "Trader",
    "aktion": "STARK KAUFEN",
    "zielkurs": 120,
    "stop_loss": 90,
    "positionsanteil": 5,
    "begründung": "Starke Fundamentals",
    "zeithorizont": "Mittelfristig",
})

_RISK_JSON = json.dumps({
    "rolle": "Risk-Manager",
    "risiko_score": 3,
    "volatilität_bewertung": "moderat",
    "max_drawdown_schaetzung": "10%",
    "positionsgröße_empfohlen": "5",
    "auflagen": "keine",
    "empfehlung": "GENEHMIGT",
})

_FINAL_JSON = json.dumps({
    "rolle": "Portfolio-Manager",
    "entscheidung": "GENEHMIGT",
    "begründung": "Trade ist gerechtfertigt",
    "confidence": 4,
})

_DEBATE_BULL_JSON = json.dumps({
    "confidence": 4,
    "name": "Bull-Argumentation",
    "argumente": "Die Aktie zeigt starkes Wachstum und gute Fundamentals.",
})

_DEBATE_BEAR_JSON = json.dumps({
    "confidence": 2,
    "name": "Bear-Argumentation",
    "argumente": "Bewertung ist zu hoch, Risiko steigend.",
})

_MOCK_DATA = {
    "ticker": "TEST",
    "fundamentals": {"name": "Test Inc.", "sector": "Tech"},
    "technicals": {"current_price": 100.0, "rsi14": 50.0},
    "sentiment": {"positiv": 1, "negativ": 0, "neutral": 2},
    "history": [{"close": 100.0}, {"close": 101.0}],
}


# ===========================================================================
# 1. llm.py: chat(as_structured=True) → StructuredChatResult
# ===========================================================================

class _MockResponse:
    """Mock für requests.Response."""

    def __init__(self, status_code: int, content: str = "Test"):
        self.status_code = status_code
        self._content = content
        self.text = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")  # noqa: TRY002

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class TestLLMStructuredChat:
    """Tests für LLMClient.chat mit as_structured und response_format."""

    def test_structured_returns_structured_result(self):
        """chat(as_structured=True, response_format=...) → StructuredChatResult."""
        client = LLMClient(base_url="http://fake:8080/v1", api_key="key", model="m")
        resp = _MockResponse(200, '{"test": true}')
        with patch("concilium.llm.requests.post", return_value=resp):
            result = client.chat(
                [{"role": "user", "content": "test"}],
                response_format=TRADE_SCHEMA,
                as_structured=True,
            )
        assert isinstance(result, StructuredChatResult)
        assert result.text == '{"test": true}'
        assert result.response_format_used is True

    def test_structured_without_as_structured_returns_str(self):
        """chat(response_format=...) ohne as_structured → str (Rückwärtskompatibilität)."""
        client = LLMClient(base_url="http://fake:8080/v1", api_key="key", model="m")
        resp = _MockResponse(200, "Hallo")
        with patch("concilium.llm.requests.post", return_value=resp):
            result = client.chat(
                [{"role": "user", "content": "test"}],
                response_format=TRADE_SCHEMA,
            )
        assert isinstance(result, str)
        assert result == "Hallo"

    def test_no_response_format_returns_str(self):
        """chat() ohne response_format → str (kompatibel)."""
        client = LLMClient(base_url="http://fake:8080/v1", api_key="key", model="m")
        resp = _MockResponse(200, "Hallo")
        with patch("concilium.llm.requests.post", return_value=resp):
            result = client.chat([{"role": "user", "content": "test"}])
        assert isinstance(result, str)
        assert result == "Hallo"

    def test_as_structured_without_response_format_returns_str(self):
        """chat(as_structured=True) ohne response_format → str (kein structured ohne schema)."""
        client = LLMClient(base_url="http://fake:8080/v1", api_key="key", model="m")
        resp = _MockResponse(200, "Hallo")
        with patch("concilium.llm.requests.post", return_value=resp):
            result = client.chat(
                [{"role": "user", "content": "test"}],
                as_structured=True,
            )
        assert isinstance(result, str)
        assert result == "Hallo"

    def test_400_fallback_response_format_used_false(self):
        """Bei 400 (invalid response_format) → Fallback ohne response_format."""
        client = LLMClient(base_url="http://fake:8080/v1", api_key="key", model="m")
        # Erste Antwort: 400 (response_format abgelehnt), zweite: 200 (ohne rf)
        responses = [
            _MockResponse(400, "Bad Request"),
            _MockResponse(200, "Fallback-Antwort"),
        ]
        with patch("concilium.llm.requests.post", side_effect=responses):
            result = client.chat(
                [{"role": "user", "content": "test"}],
                response_format=TRADE_SCHEMA,
                as_structured=True,
            )
        assert isinstance(result, StructuredChatResult)
        assert result.text == "Fallback-Antwort"
        assert result.response_format_used is False

    def test_400_fallback_payload_has_no_response_format(self):
        """Der Fallback-Request (2. Versuch) hat kein response_format im Payload."""
        client = LLMClient(base_url="http://fake:8080/v1", api_key="key", model="m")
        responses = [
            _MockResponse(400, "Bad"),
            _MockResponse(200, "OK"),
        ]
        with patch("concilium.llm.requests.post", side_effect=responses) as mock_post:
            client.chat(
                [{"role": "user", "content": "test"}],
                response_format=TRADE_SCHEMA,
                as_structured=True,
            )
        # Erster Call: hat response_format
        first_payload = mock_post.call_args_list[0][1]["json"]
        assert "response_format" in first_payload
        # Zweiter Call: kein response_format
        second_payload = mock_post.call_args_list[1][1]["json"]
        assert "response_format" not in second_payload

    def test_200_response_format_in_payload(self):
        """Bei 200 wird response_format im Payload gesendet."""
        client = LLMClient(base_url="http://fake:8080/v1", api_key="key", model="m")
        resp = _MockResponse(200, "OK")
        with patch("concilium.llm.requests.post", return_value=resp) as mock_post:
            client.chat(
                [{"role": "user", "content": "test"}],
                response_format=TRADE_SCHEMA,
            )
        payload = mock_post.call_args[1]["json"]
        assert "response_format" in payload
        assert payload["response_format"] == TRADE_SCHEMA

    def test_no_response_format_not_in_payload(self):
        """Ohne response_format ist es nicht im Payload (kompatibel)."""
        client = LLMClient(base_url="http://fake:8080/v1", api_key="key", model="m")
        resp = _MockResponse(200, "OK")
        with patch("concilium.llm.requests.post", return_value=resp) as mock_post:
            client.chat([{"role": "user", "content": "test"}])
        payload = mock_post.call_args[1]["json"]
        assert "response_format" not in payload


# ===========================================================================
# 2. Agenten mit strukturierten Mock-LLMs (response_format_used=True)
# ===========================================================================

class TestTraderStructured:
    """Trader mit strukturiertem LLM-Output."""

    def test_structured_result_has_trade_keys(self):
        """trader() mit StructuredChatResult → dict mit trade-Keys."""
        llm = _StructuredMockLLM(_TRADE_JSON)
        result = trader(_ANALYSTS, _DEBATE, llm)
        _assert_has_keys(result, ["aktion", "rating", "zielkurs", "stop_loss", "positionsanteil", "begründung"])

    def test_structured_rating_is_5stufig(self):
        """rating enthält das 5-stufige Rating (STARK KAUFEN)."""
        llm = _StructuredMockLLM(_TRADE_JSON)
        result = trader(_ANALYSTS, _DEBATE, llm)
        assert result["rating"] == "STARK KAUFEN"

    def test_structured_aktion_is_3stufig(self):
        """aktion ist 3-stufig normalisiert (KAUFEN)."""
        llm = _StructuredMockLLM(_TRADE_JSON)
        result = trader(_ANALYSTS, _DEBATE, llm)
        assert result["aktion"] == "KAUFEN"

    def test_structured_has_raw(self):
        """_raw ist gesetzt (für Kompatibilität)."""
        llm = _StructuredMockLLM(_TRADE_JSON)
        result = trader(_ANALYSTS, _DEBATE, llm)
        assert "_raw" in result


class TestTradeRevisionStructured:
    """trade_revision mit strukturiertem LLM-Output."""

    def test_structured_result_has_trade_keys(self):
        """trade_revision() mit StructuredChatResult → dict mit trade-Keys."""
        llm = _StructuredMockLLM(_TRADE_JSON)
        original = {"aktion": "KAUFEN", "rating": "KAUFEN", "zielkurs": 130}
        risk = {"risiko_score": 3, "empfehlung": "MODIFIZIERT"}
        result = trade_revision(original, risk, None, llm)
        _assert_has_keys(result, ["aktion", "rating", "zielkurs", "stop_loss"])

    def test_structured_rating_normalization(self):
        """rating wird normalisiert (STARK KAUFEN → rating=STARK KAUFEN, aktion=KAUFEN)."""
        llm = _StructuredMockLLM(_TRADE_JSON)
        original = {"aktion": "KAUFEN", "rating": "KAUFEN"}
        risk = {"risiko_score": 3, "empfehlung": "MODIFIZIERT"}
        result = trade_revision(original, risk, None, llm)
        assert result["rating"] == "STARK KAUFEN"
        assert result["aktion"] == "KAUFEN"


class TestRiskManagerStructured:
    """risk_manager mit strukturiertem LLM-Output."""

    def test_structured_result_has_risk_keys(self):
        """risk_manager() mit StructuredChatResult → dict mit risk-Keys."""
        llm = _StructuredMockLLM(_RISK_JSON)
        trade = {"aktion": "KAUFEN", "zielkurs": 120, "stop_loss": 90}
        result = risk_manager(trade, _MOCK_DATA, llm, data_text="dummy")
        _assert_has_keys(result, ["risiko_score", "auflagen", "empfehlung"])

    def test_structured_has_computed_fields(self):
        """risk_manager ergänzt volatilität_annualisiert_pct und positionsgröße_rechnerisch_pct."""
        llm = _StructuredMockLLM(_RISK_JSON)
        trade = {"aktion": "KAUFEN", "zielkurs": 120, "stop_loss": 90}
        result = risk_manager(trade, _MOCK_DATA, llm, data_text="dummy")
        assert "volatilität_annualisiert_pct" in result
        assert "positionsgröße_rechnerisch_pct" in result


class TestPortfolioManagerStructured:
    """portfolio_manager mit strukturiertem LLM-Output."""

    def test_structured_result_has_final_keys(self):
        """portfolio_manager() mit StructuredChatResult → dict mit final-Keys."""
        llm = _StructuredMockLLM(_FINAL_JSON)
        trade = {"aktion": "KAUFEN", "zielkurs": 120, "stop_loss": 90}
        risk = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}
        result = portfolio_manager(trade, risk, llm)
        _assert_has_keys(result, ["entscheidung", "begründung", "confidence"])

    def test_structured_confidence_is_int(self):
        """confidence ist ein Integer."""
        llm = _StructuredMockLLM(_FINAL_JSON)
        trade = {"aktion": "KAUFEN"}
        risk = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}
        result = portfolio_manager(trade, risk, llm)
        assert result["confidence"] == 4


class TestDebateStructured:
    """debate() mit strukturiertem LLM-Output."""

    def test_structured_bull_has_confidence(self):
        """bull-dict hat confidence-Feld direkt."""
        responses = [_DEBATE_BULL_JSON, _DEBATE_BEAR_JSON]
        llm = _DebateStructuredLLM(responses)
        result = debate(_ANALYSTS, llm)
        assert result["bull"]["confidence"] == 4

    def test_structured_bear_has_confidence(self):
        """bear-dict hat confidence-Feld direkt."""
        responses = [_DEBATE_BULL_JSON, _DEBATE_BEAR_JSON]
        llm = _DebateStructuredLLM(responses)
        result = debate(_ANALYSTS, llm)
        assert result["bear"]["confidence"] == 2

    def test_structured_bull_confidence_extracted(self):
        """bull_confidence wird aus dem direkten confidence-Feld extrahiert."""
        responses = [_DEBATE_BULL_JSON, _DEBATE_BEAR_JSON]
        llm = _DebateStructuredLLM(responses)
        result = debate(_ANALYSTS, llm)
        assert result["bull_confidence"] == 4

    def test_structured_bear_confidence_extracted(self):
        """bear_confidence wird aus dem direkten confidence-Feld extrahiert."""
        responses = [_DEBATE_BULL_JSON, _DEBATE_BEAR_JSON]
        llm = _DebateStructuredLLM(responses)
        result = debate(_ANALYSTS, llm)
        assert result["bear_confidence"] == 2

    def test_structured_bull_has_argumente(self):
        """bull-dict hat 'argumente'-Feld (Fließtext)."""
        responses = [_DEBATE_BULL_JSON, _DEBATE_BEAR_JSON]
        llm = _DebateStructuredLLM(responses)
        result = debate(_ANALYSTS, llm)
        assert "argumente" in result["bull"]
        assert "Wachstum" in result["bull"]["argumente"]


class _DebateStructuredLLM:
    """Mock-LLM für debate(): gibt verschiedene Antworten für Bull/Bear."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._idx = 0

    def chat(self, messages, temperature=0.5, **kwargs):
        text = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            return StructuredChatResult(text=text, response_format_used=True)
        return text


class TestAnalystTeamStructured:
    """analyst_team mit strukturiertem LLM-Output."""

    def test_structured_analysts_have_keys(self):
        """analyst_team mit StructuredChatResult → dicts mit analyst-Keys."""
        from concilium.agents import analyst_team

        responses = [
            json.dumps({"rolle": "Fundamental-Analyst", "stimmung": "bullish", "score": 4, "zusammenfassung": "Gut"}),
            json.dumps({"rolle": "Technik-Analyst", "stimmung": "neutral", "score": 3, "zusammenfassung": "Ok",
                         "trend": "seitwärts", "signale": "RSI normal"}),
            json.dumps({"rolle": "Sentiment-Analyst", "stimmung": "bullish", "score": 4, "zusammenfassung": "Positiv",
                         "dominant": "positiv"}),
        ]
        llm = _AnalystStructuredLLM(responses)
        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {"current_price": 123.45, "rsi14": 55.0},
            "sentiment": {},
        }
        result = analyst_team(data, llm)

        _assert_has_keys(result["fundamental"], ["stimmung", "score", "zusammenfassung"])
        _assert_has_keys(result["technical"], ["stimmung", "score", "zusammenfassung"])
        _assert_has_keys(result["sentiment"], ["stimmung", "score", "zusammenfassung"])
        assert result["fundamental"]["stimmung"] == "bullish"
        assert result["technical"]["stimmung"] == "neutral"
        assert result["sentiment"]["stimmung"] == "bullish"

    def test_structured_consistency_warning_appended(self):
        """Bei inkonsistenter Stimmung/Score wird konsistenz_warnung angehängt."""
        from concilium.agents import analyst_team

        responses = [
            json.dumps({"rolle": "Fundamental-Analyst", "stimmung": "bullish", "score": 1, "zusammenfassung": "Inkonsistent"}),
            json.dumps({"rolle": "Technik-Analyst", "stimmung": "neutral", "score": 3, "zusammenfassung": "Ok"}),
            json.dumps({"rolle": "Sentiment-Analyst", "stimmung": "bearish", "score": 5, "zusammenfassung": "Inkonsistent"}),
        ]
        llm = _AnalystStructuredLLM(responses)
        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {"current_price": 100.0},
            "sentiment": {},
        }
        result = analyst_team(data, llm)

        # bullish + score 1 → inkonsistent
        assert "konsistenz_warnung" in result["fundamental"]
        assert result["fundamental"]["konsistenz_warnung"] != ""
        # bearish + score 5 → inkonsistent
        assert "konsistenz_warnung" in result["sentiment"]
        assert result["sentiment"]["konsistenz_warnung"] != ""
        # neutral + score 3 → konsistent → kein Feld
        assert "konsistenz_warnung" not in result["technical"]


class _AnalystStructuredLLM:
    """Mock-LLM für analyst_team: dispatcht basierend auf System-Prompt."""

    def __init__(self, responses: list[str]):
        # responses: [fundamental, technical, sentiment]
        self._responses = responses

    def chat(self, messages, temperature=0.3, **kwargs):
        system = messages[0]["content"]
        if "Fundamental" in system:
            text = self._responses[0]
        elif "technisch" in system:
            text = self._responses[1]
        elif "Sentiment" in system:
            text = self._responses[2]
        else:
            text = self._responses[0]
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            return StructuredChatResult(text=text, response_format_used=True)
        return text


# ===========================================================================
# 3. Fallback-Pfad (response_format_used=False → parse_json)
# ===========================================================================

class TestFallbackPath:
    """Agenten mit Fallback-Mock (str-Rückgabe → parse_json)."""

    def test_trader_fallback_uses_parse_json(self):
        """trader() mit str-Rückgabe → parse_json auf Fließtext mit JSON-Block."""
        # Fließtext mit eingebettetem JSON
        text = f'Hier ist meine Analyse:\n{_TRADE_JSON}\nDas war es.'
        llm = _FallbackMockLLM(text)
        result = trader(_ANALYSTS, _DEBATE, llm)
        _assert_has_keys(result, ["aktion", "rating", "zielkurs"])
        assert result["rating"] == "STARK KAUFEN"
        assert result["aktion"] == "KAUFEN"

    def test_risk_manager_fallback(self):
        """risk_manager() mit str-Rückgabe → parse_json."""
        llm = _FallbackMockLLM(_RISK_JSON)
        trade = {"aktion": "KAUFEN", "zielkurs": 120, "stop_loss": 90}
        result = risk_manager(trade, _MOCK_DATA, llm, data_text="dummy")
        _assert_has_keys(result, ["risiko_score", "auflagen", "empfehlung"])
        assert result["risiko_score"] == 3

    def test_portfolio_manager_fallback(self):
        """portfolio_manager() mit str-Rückgabe → parse_json."""
        llm = _FallbackMockLLM(_FINAL_JSON)
        trade = {"aktion": "KAUFEN"}
        risk = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}
        result = portfolio_manager(trade, risk, llm)
        _assert_has_keys(result, ["entscheidung", "begründung", "confidence"])
        assert result["entscheidung"] == "GENEHMIGT"

    def test_debate_fallback_confidence_from_raw(self):
        """debate() mit str-Rückgabe → confidence aus _raw via parse_json."""
        bull_text = '{"confidence": 4, "name": "Bull"}\nDie Aktie ist stark.'
        bear_text = '{"confidence": 2, "name": "Bear"}\nBewertung zu hoch.'

        class _DebateFallbackLLM:
            def __init__(self):
                self._idx = 0

            def chat(self, messages, temperature=0.5, **kwargs):
                if self._idx == 0:
                    text = bull_text
                else:
                    text = bear_text
                self._idx += 1
                # Gibt str zurück → _call_agent nutzt parse_json
                return text

        result = debate(_ANALYSTS, _DebateFallbackLLM())
        assert result["bull_confidence"] == 4
        assert result["bear_confidence"] == 2


# ===========================================================================
# 4. Schema-Validierung (selbst-enthaltene Validierung)
# ===========================================================================

class TestSchemaValidation:
    """Tests für die selbst-enthaltene Schema-Validierung."""

    def test_trade_schema_valid(self):
        """Ein gültiges trade-dict hat keine Validierungsfehler."""
        valid = {
            "aktion": "KAUFEN",
            "zielkurs": 120,
            "stop_loss": 90,
            "positionsanteil": 5,
            "begründung": "Test",
        }
        errors = validate_structured(valid, TRADE_SCHEMA)
        assert errors == []

    def test_trade_schema_missing_required(self):
        """Fehlendes required-Feld → Fehler."""
        invalid = {"aktion": "KAUFEN", "zielkurs": 120}
        errors = validate_structured(invalid, TRADE_SCHEMA)
        assert len(errors) > 0
        assert any("stop_loss" in e for e in errors)

    def test_trade_schema_invalid_enum(self):
        """Ungültige aktion → Fehler."""
        invalid = {
            "aktion": "KAUFEN!",
            "zielkurs": 120,
            "stop_loss": 90,
            "positionsanteil": 5,
            "begründung": "Test",
        }
        errors = validate_structured(invalid, TRADE_SCHEMA)
        assert len(errors) > 0

    def test_risk_schema_valid(self):
        """Gültiges risk-dict."""
        valid = {"risiko_score": 3, "auflagen": "keine", "empfehlung": "GENEHMIGT"}
        errors = validate_structured(valid, RISK_SCHEMA)
        assert errors == []

    def test_risk_schema_score_out_of_range(self):
        """risiko_score 0 → Fehler (minimum 1)."""
        invalid = {"risiko_score": 0, "auflagen": "keine", "empfehlung": "GENEHMIGT"}
        errors = validate_structured(invalid, RISK_SCHEMA)
        assert any("minimum" in e for e in errors)

    def test_final_schema_valid(self):
        """Gültiges final-dict."""
        valid = {"entscheidung": "GENEHMIGT", "begründung": "Test", "confidence": 4}
        errors = validate_structured(valid, FINAL_SCHEMA)
        assert errors == []

    def test_final_schema_confidence_out_of_range(self):
        """confidence 6 → Fehler (maximum 5)."""
        invalid = {"entscheidung": "GENEHMIGT", "begründung": "Test", "confidence": 6}
        errors = validate_structured(invalid, FINAL_SCHEMA)
        assert any("maximum" in e for e in errors)

    def test_debate_schema_valid(self):
        """Gültiges debate-dict."""
        valid = {"confidence": 4, "name": "Bull", "argumente": "Text"}
        errors = validate_structured(valid, DEBATE_SCHEMA)
        assert errors == []

    def test_analyst_schema_valid(self):
        """Gültiges analyst-dict."""
        valid = {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut"}
        errors = validate_structured(valid, ANALYST_FUNDAMENTAL_SCHEMA)
        assert errors == []

    def test_analyst_schema_invalid_stimmung(self):
        """Ungültige stimmung → Fehler."""
        invalid = {"stimmung": "very_bullish", "score": 4, "zusammenfassung": "Gut"}
        errors = validate_structured(invalid, ANALYST_FUNDAMENTAL_SCHEMA)
        assert len(errors) > 0

    def test_analyst_technical_has_extra_fields(self):
        """Technik-Analyst kann trend und signale haben."""
        valid = {
            "stimmung": "bullish", "score": 4, "zusammenfassung": "Gut",
            "trend": "aufwärts", "signale": "RSI normal",
        }
        errors = validate_structured(valid, ANALYST_TECHNICAL_SCHEMA)
        assert errors == []

    def test_analyst_sentiment_has_dominant(self):
        """Sentiment-Analyst kann dominant haben."""
        valid = {
            "stimmung": "bullish", "score": 4, "zusammenfassung": "Gut",
            "dominant": "positiv",
        }
        errors = validate_structured(valid, ANALYST_SENTIMENT_SCHEMA)
        assert errors == []

    def test_additional_properties_false_rejects_unknown(self):
        """additionalProperties: false → unbekanntes Feld → Fehler."""
        invalid = {
            "stimmung": "bullish", "score": 4, "zusammenfassung": "Gut",
            "unknown_field": "should_fail",
        }
        errors = validate_structured(invalid, ANALYST_FUNDAMENTAL_SCHEMA)
        assert any("Unexpected" in e for e in errors)

    def test_null_zielkurs_valid(self):
        """zielkurs null ist gültig (anyOf number/string/null)."""
        valid = {
            "aktion": "HALTEN",
            "zielkurs": None,
            "stop_loss": None,
            "positionsanteil": None,
            "begründung": "Keine Aktion",
        }
        errors = validate_structured(valid, TRADE_SCHEMA)
        assert errors == []

    def test_string_zielkurs_valid(self):
        """zielkurs als String ist gültig (anyOf number/string/null)."""
        valid = {
            "aktion": "KAUFEN",
            "zielkurs": "120.50",
            "stop_loss": "90",
            "positionsanteil": 5,
            "begründung": "Test",
        }
        errors = validate_structured(valid, TRADE_SCHEMA)
        assert errors == []
