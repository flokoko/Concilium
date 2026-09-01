"""JSON-Schema-Definitionen für strukturierte LLM-Outputs (OpenAI json_schema-Stil).

Jede Konstante ist ein dict im Format:
    {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}

Die Schema-Feldnamen sind deutsch und entsprechen den bisherigen Keys der
Downstream-Consumer (report.py, journal.py, pipeline.py), sodass keine
Anpassungen an den Renderern nötig sind.

Zusätzlich: eine selbst-enthaltene Schema-Validierungsfunktion
(:func:`validate_structured`), die KEIN externes Paket (jsonschema) benötigt.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Schema-Definitionen
# ---------------------------------------------------------------------------

TRADE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "trade",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["aktion", "zielkurs", "stop_loss", "positionsanteil", "begründung"],
            "properties": {
                "aktion": {
                    "type": "string",
                    "enum": ["STARK KAUFEN", "KAUFEN", "HALTEN", "VERKAUFEN", "STARK VERKAUFEN"],
                },
                "zielkurs": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "string"},
                        {"type": "null"},
                    ],
                },
                "stop_loss": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "string"},
                        {"type": "null"},
                    ],
                },
                "positionsanteil": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "null"},
                    ],
                },
                "begründung": {"type": "string"},
                "zeithorizont": {
                    "anyOf": [
                        {"type": "string", "enum": ["Kurzfristig", "Mittelfristig", "Langfristig"]},
                        {"type": "null"},
                    ],
                },
                "rolle": {"type": "string"},
            },
        },
    },
}

RISK_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "risk",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["risiko_score", "auflagen", "empfehlung"],
            "properties": {
                "risiko_score": {"type": "integer", "minimum": 1, "maximum": 5},
                "volatilität_bewertung": {"type": "string"},
                "max_drawdown_schaetzung": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "string"},
                    ],
                },
                "positionsgröße_empfohlen": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "string"},
                    ],
                },
                "auflagen": {"type": "string"},
                "empfehlung": {
                    "type": "string",
                    "enum": ["GENEHMIGT", "MODIFIZIERT", "ABGELEHNT"],
                },
            },
        },
    },
}

FINAL_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "final",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["entscheidung", "begründung", "confidence"],
            "properties": {
                "entscheidung": {
                    "type": "string",
                    "enum": ["GENEHMIGT", "MODIFIZIERT", "ABGELEHNT"],
                },
                "begründung": {"type": "string"},
                "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
            },
        },
    },
}

DEBATE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "debate",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["confidence", "argumente"],
            "properties": {
                "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                "name": {"type": "string"},
                "argumente": {"type": "string"},
            },
        },
    },
}

# Analysten-Schemas (fundamental/technical/sentiment) — gemeinsame Basis
_ANALYST_BASE: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["stimmung", "score", "zusammenfassung"],
    "properties": {
        "stimmung": {
            "type": "string",
            "enum": ["bullish", "neutral", "bearish"],
        },
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "zusammenfassung": {"type": "string"},
        "konsistenz_warnung": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ],
        },
        "rolle": {"type": "string"},
    },
}

ANALYST_FUNDAMENTAL_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "analyst_fundamental",
        "schema": _ANALYST_BASE,
    },
}

ANALYST_TECHNICAL_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "analyst_technical",
        "schema": {
            **_ANALYST_BASE,
            "properties": {
                **_ANALYST_BASE["properties"],
                "trend": {"type": "string"},
                "signale": {"type": "string"},
            },
        },
    },
}

ANALYST_SENTIMENT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "analyst_sentiment",
        "schema": {
            **_ANALYST_BASE,
            "properties": {
                **_ANALYST_BASE["properties"],
                "dominant": {"type": "string"},
            },
        },
    },
}

ANALYST_MACRO_NEWS_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "analyst_macro_news",
        "schema": {
            **_ANALYST_BASE,
            "properties": {
                **_ANALYST_BASE["properties"],
                "makro_einschaetzung": {"type": "string"},
                "relevante_headlines": {"type": "string"},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Selbst-enthaltene Schema-Validierung (kein externes Paket)
# ---------------------------------------------------------------------------

def _check_type(value: Any, type_name: str) -> bool:
    """Prüft, ob value dem JSON-Schema-Typ entspricht."""
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    return False


def _validate_against_schema(value: Any, schema: dict[str, Any]) -> list[str]:
    """Validiert value gegen ein JSON-Schema-Sub-Schema (kein externes Paket).

    Unterstützt: type, enum, required, properties, additionalProperties,
    anyOf, minimum, maximum.

    Gibt eine Liste von Fehler-Strings zurück (leer = gültig).
    """
    errors: list[str] = []

    # anyOf
    if "anyOf" in schema:
        for sub in schema["anyOf"]:
            if not _validate_against_schema(value, sub):
                return []
        return [f"Value {value!r} matched none of anyOf options"]

    if not isinstance(value, dict):
        if "type" in schema and schema["type"] == "object":
            return [f"Expected object, got {type(value).__name__}"]
        return []

    obj_schema = schema.get("properties", {})
    required = schema.get("required", [])

    # required-Felder
    for field in required:
        if field not in value:
            errors.append(f"Missing required field: {field}")

    # additionalProperties: false → keine undefinierten Keys
    if schema.get("additionalProperties") is False:
        for key in value:
            if key not in obj_schema:
                errors.append(f"Unexpected field: {key}")

    # Feld-Typen prüfen
    for field, field_schema in obj_schema.items():
        if field not in value:
            continue
        val = value[field]
        if "enum" in field_schema and val not in field_schema["enum"]:
            errors.append(f"Field '{field}': {val!r} not in enum {field_schema['enum']}")
            continue
        if "type" in field_schema:
            if not _check_type(val, field_schema["type"]):
                errors.append(
                    f"Field '{field}': expected {field_schema['type']}, got {type(val).__name__}"
                )
                continue
            if field_schema["type"] == "integer":
                if "minimum" in field_schema and val < field_schema["minimum"]:
                    errors.append(f"Field '{field}': {val} < minimum {field_schema['minimum']}")
                if "maximum" in field_schema and val > field_schema["maximum"]:
                    errors.append(f"Field '{field}': {val} > maximum {field_schema['maximum']}")
        if "anyOf" in field_schema:
            matched = False
            for sub in field_schema["anyOf"]:
                if not _validate_against_schema(val, sub):
                    matched = True
                    break
            if not matched:
                errors.append(f"Field '{field}': {val!r} matched none of anyOf")

    return errors


def _default_for_field(field_schema: dict[str, Any]) -> Any:
    """Bestimmt einen sicheren, schema-konformen Default-Wert für ein Feld.

    Regeln:
      - anyOf mit null → None (null ist explizit erlaubt)
      - anyOf ohne null → erster sinnvoller non-null Typ (bevorzugt string → "")
      - string mit enum → neutrales enum-Wert oder erstes enum-Wert
      - string ohne enum → ""
      - integer mit minimum → minimum (sicherster gültiger Wert)
      - integer ohne minimum → 0
      - number → 0.0
      - boolean → False
      - null → None
    """
    # anyOf-Handling
    if "anyOf" in field_schema:
        options = field_schema["anyOf"]
        # Wenn null explizit erlaubt ist → None
        for sub in options:
            if sub.get("type") == "null":
                return None
        # Sonst: bevorzuge string (leerer String ist sicher für Rendering)
        for sub in options:
            if sub.get("type") == "string":
                return ""
        for sub in options:
            if sub.get("type") == "number":
                return 0.0
            if sub.get("type") == "integer":
                return sub.get("minimum", 0)
        return None

    type_name = field_schema.get("type")
    if type_name == "string":
        # Bei enum: neutrales/erstes enum-Wert wählen (schema-konform)
        if "enum" in field_schema:
            enum = field_schema["enum"]
            for neutral in ("HALTEN", "neutral", "GENEHMIGT"):
                if neutral in enum:
                    return neutral
            return enum[0]
        return ""
    if type_name == "integer":
        if "minimum" in field_schema:
            return field_schema["minimum"]
        return 0
    if type_name == "number":
        return 0.0
    if type_name == "boolean":
        return False
    if type_name == "null":
        return None
    return None


def defaults_for_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Baut ein dict mit sicheren Defaults für JEDE property eines response_format-Schemas.

    Nur Top-Level properties werden berücksichtigt (kein deep-nesting).
    Die Defaults sind so gewählt, dass sie schema-konform sind —
    ``validate_structured(defaults_for_schema(schema), schema)`` gibt ``[]`` zurück.

    Args:
        schema: Ein response_format-dict wie TRADE_SCHEMA, RISK_SCHEMA etc.

    Returns:
        dict mit allen property-Keys und ihren sicheren Default-Werten.
    """
    inner = schema.get("json_schema", {}).get("schema", schema)
    properties = inner.get("properties", {})
    return {key: _default_for_field(prop) for key, prop in properties.items()}


def validate_structured(result: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validiert ein dict gegen ein Top-Level response_format-Schema.

    Args:
        result: Das geparste dict aus der LLM-Antwort.
        schema: Ein response_format-dict wie TRADE_SCHEMA etc.

    Returns:
        Liste von Fehler-Strings (leer = gültig). Validierung ist best-effort
        und bricht nie — fehlende/extra Felder werden gemeldet, aber die
        Funktion wirft keine Exception.
    """
    inner = schema.get("json_schema", {}).get("schema", schema)
    return _validate_against_schema(result, inner)
