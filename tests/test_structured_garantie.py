"""Tests für die Struktur-Garantie in _call_agent (Phase 0 Bug-Fix).

Testet, dass _call_agent bei structured=True IMMER ein dict mit allen
Schema-Keys zurückgibt — gefüllt mit sicheren Defaults, wenn das Modell
leer oder unvollständig antwortet.

Testfälle:
1. Leere Antwort ("") → alle Schema-Keys mit Defaults
2. Leeres dict ("{}") → alle Schema-Keys mit Defaults
3. Teilweise Antwort → fehlende Felder Default-gefüllt, vorhandene erhalten
4. defaults_for_schema liefert für jedes Schema ein dict mit allen required-Keys
5. validate_structured(defaults_for_schema(SCHEMA), SCHEMA) == [] (schema-konform)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import _call_agent  # noqa: E402
from concilium.llm import StructuredChatResult  # noqa: E402
from concilium.schemas import (  # noqa: E402
    ANALYST_FUNDAMENTAL_SCHEMA,
    ANALYST_SENTIMENT_SCHEMA,
    ANALYST_TECHNICAL_SCHEMA,
    DEBATE_SCHEMA,
    FINAL_SCHEMA,
    RISK_SCHEMA,
    TRADE_SCHEMA,
    defaults_for_schema,
    validate_structured,
)

ALL_SCHEMAS = [
    ("TRADE_SCHEMA", TRADE_SCHEMA),
    ("RISK_SCHEMA", RISK_SCHEMA),
    ("FINAL_SCHEMA", FINAL_SCHEMA),
    ("DEBATE_SCHEMA", DEBATE_SCHEMA),
    ("ANALYST_FUNDAMENTAL_SCHEMA", ANALYST_FUNDAMENTAL_SCHEMA),
    ("ANALYST_TECHNICAL_SCHEMA", ANALYST_TECHNICAL_SCHEMA),
    ("ANALYST_SENTIMENT_SCHEMA", ANALYST_SENTIMENT_SCHEMA),
]


# ---------------------------------------------------------------------------
# Mock-LLM
# ---------------------------------------------------------------------------


class _MockLLM:
    """Einfacher Mock-LLM für _call_agent mit structured=True.

    Gibt einen vordefinierten Text als StructuredChatResult zurück
    (response_format_used=True).
    """

    def __init__(self, text: str, response_format_used: bool = True):
        self._text = text
        self._rfu = response_format_used

    def chat(self, messages, temperature=0.3, **kwargs):
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            return StructuredChatResult(text=self._text, response_format_used=self._rfu)
        return self._text


# ---------------------------------------------------------------------------
# 1. Leere Antwort → alle Schema-Keys mit Defaults
# ---------------------------------------------------------------------------


class TestEmptyResponseDefaults:
    """_call_agent mit leerer Antwort → dict mit allen Schema-Keys."""

    def test_empty_string_final_schema(self):
        """Leere Antwort ('') bei FINAL_SCHEMA → alle required-Keys mit Defaults."""
        llm = _MockLLM("")
        result = _call_agent(
            llm, "system", "user",
            response_format=FINAL_SCHEMA, structured=True,
        )
        assert isinstance(result, dict)
        for key in ("entscheidung", "begründung", "confidence"):
            assert key in result, f"Key '{key}' fehlt: {list(result.keys())}"
        # Defaults sind schema-konform
        assert result["entscheidung"] == "GENEHMIGT"  # neutraler enum-Default
        assert result["begründung"] == ""
        assert result["confidence"] == 1  # minimum
        # _raw bleibt erhalten
        assert result["_raw"] == ""

    def test_empty_dict_final_schema(self):
        """Leeres dict ('{}') bei FINAL_SCHEMA → alle required-Keys mit Defaults."""
        llm = _MockLLM("{}")
        result = _call_agent(
            llm, "system", "user",
            response_format=FINAL_SCHEMA, structured=True,
        )
        for key in ("entscheidung", "begründung", "confidence"):
            assert key in result, f"Key '{key}' fehlt: {list(result.keys())}"

    def test_empty_string_trade_schema(self):
        """Leere Antwort bei TRADE_SCHEMA → alle Schema-Keys mit Defaults."""
        llm = _MockLLM("")
        result = _call_agent(
            llm, "system", "user",
            response_format=TRADE_SCHEMA, structured=True,
        )
        expected_keys = {"aktion", "zielkurs", "stop_loss", "positionsanteil", "begründung",
                         "zeithorizont", "rolle", "_raw"}
        assert expected_keys.issubset(set(result.keys()))
        # anyOf mit null → None
        assert result["zielkurs"] is None
        assert result["stop_loss"] is None
        assert result["positionsanteil"] is None
        # string → ""
        assert result["begründung"] == ""
        # string mit enum → neutraler Wert
        assert result["aktion"] == "HALTEN"

    def test_empty_string_risk_schema(self):
        """Leere Antwort bei RISK_SCHEMA → alle Schema-Keys mit Defaults."""
        llm = _MockLLM("")
        result = _call_agent(
            llm, "system", "user",
            response_format=RISK_SCHEMA, structured=True,
        )
        expected_keys = {"risiko_score", "volatilität_bewertung", "max_drawdown_schaetzung",
                         "positionsgröße_empfohlen", "auflagen", "empfehlung"}
        assert expected_keys.issubset(set(result.keys()))
        assert result["risiko_score"] == 1  # minimum
        assert result["auflagen"] == ""
        assert result["empfehlung"] == "GENEHMIGT"  # neutraler enum-Default

    def test_empty_string_debate_schema(self):
        """Leere Antwort bei DEBATE_SCHEMA → alle Schema-Keys mit Defaults."""
        llm = _MockLLM("")
        result = _call_agent(
            llm, "system", "user",
            response_format=DEBATE_SCHEMA, structured=True,
        )
        expected_keys = {"confidence", "name", "argumente"}
        assert expected_keys.issubset(set(result.keys()))
        assert result["confidence"] == 1  # minimum
        assert result["argumente"] == ""

    def test_empty_string_analyst_schemas(self):
        """Leere Antwort bei Analysten-Schemas → alle Schema-Keys mit Defaults."""
        for schema in (ANALYST_FUNDAMENTAL_SCHEMA, ANALYST_TECHNICAL_SCHEMA, ANALYST_SENTIMENT_SCHEMA):
            llm = _MockLLM("")
            result = _call_agent(
                llm, "system", "user",
                response_format=schema, structured=True,
            )
            assert "stimmung" in result
            assert "score" in result
            assert "zusammenfassung" in result
            assert result["stimmung"] == "neutral"  # neutraler enum-Default
            assert result["score"] == 1  # minimum
            assert result["zusammenfassung"] == ""


# ---------------------------------------------------------------------------
# 2. Teilweise Antwort → fehlende Felder Default-gefüllt, vorhandene erhalten
# ---------------------------------------------------------------------------


class TestPartialResponseDefaults:
    """_call_agent mit teilweiser Antwort → setdefault-Semantik."""

    def test_partial_final_keeps_provided(self):
        """Teilweise Antwort: vorhandene Werte bleiben erhalten."""
        partial = json.dumps({"entscheidung": "ABGELEHNT", "confidence": 5})
        llm = _MockLLM(partial)
        result = _call_agent(
            llm, "system", "user",
            response_format=FINAL_SCHEMA, structured=True,
        )
        # Vom Modell gelieferte Werte
        assert result["entscheidung"] == "ABGELEHNT"
        assert result["confidence"] == 5
        # Fehlendes Feld wird Default-gefüllt
        assert result["begründung"] == ""

    def test_partial_trade_keeps_provided(self):
        """Teilweise Antwort bei TRADE_SCHEMA: vorhandene Werte bleiben."""
        partial = json.dumps({
            "aktion": "KAUFEN",
            "zielkurs": 150,
            "begründung": "Starke Fundamentals",
        })
        llm = _MockLLM(partial)
        result = _call_agent(
            llm, "system", "user",
            response_format=TRADE_SCHEMA, structured=True,
        )
        # Vom Modell geliefert
        assert result["aktion"] == "KAUFEN"
        assert result["zielkurs"] == 150
        assert result["begründung"] == "Starke Fundamentals"
        # Default-gefüllt (fehlend im Modell-Output)
        assert result["stop_loss"] is None
        assert result["positionsanteil"] is None
        assert result["zeithorizont"] is None

    def test_partial_risk_keeps_provided(self):
        """Teilweise Antwort bei RISK_SCHEMA: vorhandene Werte bleiben."""
        partial = json.dumps({"risiko_score": 4, "empfehlung": "ABGELEHNT"})
        llm = _MockLLM(partial)
        result = _call_agent(
            llm, "system", "user",
            response_format=RISK_SCHEMA, structured=True,
        )
        assert result["risiko_score"] == 4
        assert result["empfehlung"] == "ABGELEHNT"
        # Default-gefüllt
        assert result["auflagen"] == ""

    def test_raw_fallback_gets_defaults(self):
        """Bei nicht-JSON-Antwort im strukturierten Pfad: _raw + Defaults."""
        llm = _MockLLM("Das ist kein JSON.")
        result = _call_agent(
            llm, "system", "user",
            response_format=FINAL_SCHEMA, structured=True,
        )
        # _raw ist gesetzt
        assert result["_raw"] == "Das ist kein JSON."
        # Alle Schema-Keys sind vorhanden (durch Defaults)
        for key in ("entscheidung", "begründung", "confidence"):
            assert key in result, f"Key '{key}' fehlt im _raw-Fallback: {list(result.keys())}"


# ---------------------------------------------------------------------------
# 3. defaults_for_schema liefert für jedes Schema ein dict mit allen required-Keys
# ---------------------------------------------------------------------------


class TestDefaultsForSchema:
    """defaults_for_schema liefert korrekte Defaults für alle Schemas."""

    def test_defaults_contain_all_required_keys(self):
        """Für jedes Schema: defaults_for_schema enthält alle required-Keys."""
        for name, schema in ALL_SCHEMAS:
            inner = schema.get("json_schema", {}).get("schema", schema)
            required = inner.get("required", [])
            defaults = defaults_for_schema(schema)
            for req in required:
                assert req in defaults, (
                    f"{name}: required Key '{req}' fehlt in defaults_for_schema"
                )

    def test_defaults_contain_all_properties(self):
        """defaults_for_schema setzt Defaults für ALLE properties (nicht nur required)."""
        for name, schema in ALL_SCHEMAS:
            inner = schema.get("json_schema", {}).get("schema", schema)
            properties = inner.get("properties", {})
            defaults = defaults_for_schema(schema)
            assert set(defaults.keys()) == set(properties.keys()), (
                f"{name}: defaults keys {set(defaults.keys())} != properties keys {set(properties.keys())}"
            )

    def test_defaults_no_extra_keys(self):
        """defaults_for_schema setzt KEINE extra Keys (additionalProperties:false)."""
        for name, schema in ALL_SCHEMAS:
            inner = schema.get("json_schema", {}).get("schema", schema)
            properties = inner.get("properties", {})
            defaults = defaults_for_schema(schema)
            # Keine Keys außerhalb der properties
            extra = set(defaults.keys()) - set(properties.keys())
            assert not extra, f"{name}: extra Keys in defaults: {extra}"

    def test_string_defaults_are_empty_or_neutral(self):
        """String-Defaults sind '' oder ein neutraler enum-Wert."""
        for name, schema in ALL_SCHEMAS:
            inner = schema.get("json_schema", {}).get("schema", schema)
            for key, prop in inner.get("properties", {}).items():
                val = defaults_for_schema(schema)[key]
                if prop.get("type") == "string" and "enum" not in prop:
                    assert val == "", f"{name}.{key}: String-Default sollte '' sein, ist {val!r}"

    def test_anyof_with_null_defaults_to_none(self):
        """anyOf mit null → None."""
        trade_defaults = defaults_for_schema(TRADE_SCHEMA)
        assert trade_defaults["zielkurs"] is None
        assert trade_defaults["stop_loss"] is None
        assert trade_defaults["positionsanteil"] is None
        assert trade_defaults["zeithorizont"] is None

    def test_integer_defaults_use_minimum(self):
        """Integer-Defaults mit minimum → minimum-Wert."""
        final_defaults = defaults_for_schema(FINAL_SCHEMA)
        assert final_defaults["confidence"] == 1  # minimum: 1

        risk_defaults = defaults_for_schema(RISK_SCHEMA)
        assert risk_defaults["risiko_score"] == 1  # minimum: 1

        debate_defaults = defaults_for_schema(DEBATE_SCHEMA)
        assert debate_defaults["confidence"] == 1  # minimum: 1


# ---------------------------------------------------------------------------
# 4. validate_structured(defaults_for_schema(SCHEMA), SCHEMA) == []
# ---------------------------------------------------------------------------


class TestDefaultsAreSchemaConform:
    """Defaults sind selbst schema-konform (validate_structured gibt [] zurück)."""

    def test_defaults_validate_clean(self):
        """validate_structured(defaults_for_schema(SCHEMA), SCHEMA) == [] für alle Schemas."""
        for name, schema in ALL_SCHEMAS:
            defaults = defaults_for_schema(schema)
            errors = validate_structured(defaults, schema)
            assert errors == [], (
                f"{name}: defaults_for_schema ist nicht schema-konform: {errors}"
            )

    def test_defaults_no_unexpected_field_errors(self):
        """Keine 'Unexpected field' Fehler von validate_structured."""
        for name, schema in ALL_SCHEMAS:
            defaults = defaults_for_schema(schema)
            errors = validate_structured(defaults, schema)
            unexpected = [e for e in errors if "Unexpected" in e]
            assert not unexpected, (
                f"{name}: 'Unexpected field' Fehler in defaults: {unexpected}"
            )
