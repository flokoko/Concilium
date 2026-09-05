"""Zentrale Konfiguration — ALLE Umgebungsvariablen an einem Ort.

Jede Konfig-Funktion liest ihre Env-Variable FRISCH bei jedem Aufruf
(Lazy-Loading, KEIN Import-Cache). Das ist zwingend: Die Tests setzen
CONCILIUM_CACHE_DIR / CONCILIUM_STATE_DIR / CONCILIUM_REPORTS_DIR per
pytest-Fixture NACH dem Modul-Import via os.environ[...] = ...

WICHTIG: Deshalb hier KEINE ``CONFIG = {...}``-Konstante beim Import
aufbauen und cachen — das würde die Tests brechen.

Typ-Koerzion analog TradingAgents' ``_coerce``: Env-Var-Strings werden auf
den Typ des Default-Werts koerziert (bool/int/float/str). Bei ungültigem
Wert (z. B. Tippfehler ``CONCILIUM_CACHE_DIR=treu``) gibt es eine LAUTE
ValueError-Meldung statt stillschweigendem Fallback auf den Default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

# ---------------------------------------------------------------------------
# Pfad-Helper
# ---------------------------------------------------------------------------


def _repo_root() -> str:
    """Repo-Root = Eltern von src/concilium/ (dort liegen cache/, reports/, watchlist.txt).

    __file__ = <repo>/src/concilium/config.py → parent.parent.parent = <repo>.
    """
    return str(Path(__file__).resolve().parent.parent.parent)


# ---------------------------------------------------------------------------
# Typ-Koerzion (analog TradingAgents _coerce) — LAUT bei Tippfehlern
# ---------------------------------------------------------------------------

_TRUTHY = ("true", "1", "yes", "on")
_FALSY = ("false", "0", "no", "off")


def _coerce(value: str, reference: object, key: str | None = None) -> object:
    """Koerziert einen Env-Var-String auf den Typ des Default-Werts (reference).

    Typ kommt vom Default-Wert: bool → ("true","1","yes","on") / ("false","0","no","off"),
    int/float → numerische Konvertierung, str → unverändert.

    Args:
        value: Roher Env-Var-String.
        reference: Default-Wert, dessen Typ die Koerzion bestimmt.
        key: Optionaler Env-Variablen-Name für die Fehlermeldung.

    Raises:
        ValueError: Bei ungültigem Wert — mit klarer Meldung im Stil
            "Invalid value for CONCILIUM_CACHE_DIR: expected a boolean
            (true/1/yes/on), got 'treu'" (LAUT statt stiller Fallback).
    """
    prefix = f"Invalid value for {key}: " if key else "Invalid value: "
    # bool: explizite Wahrheits-Listen (case-insensitive), sonst lauter Fehler.
    if isinstance(reference, bool):
        v = value.strip().lower()
        if v in _TRUTHY:
            return True
        if v in _FALSY:
            return False
        raise ValueError(
            f"{prefix}expected a boolean (true/1/yes/on or false/0/no/off), got {value!r}"
        )
    # int / float: numerische Konvertierung, sonst lauter Fehler.
    if isinstance(reference, int):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"{prefix}expected an integer, got {value!r}"
            ) from exc
    if isinstance(reference, float):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"{prefix}expected a float, got {value!r}"
            ) from exc
    # str: nichts zu tun.
    return value


def _env(key: str, default: str) -> str:
    """Liest eine Env-Variable frisch und koerziert sie auf den Default-Typ.

    Bei Koerzion-Fehlern ValueError mit env-var-spezifischer Meldung
    (siehe _coerce) — LAUT statt stiller Fallback auf den Default.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    return str(_coerce(raw, default, key=key))


# ---------------------------------------------------------------------------
# Pfad-Konfiguration (state / reports / cache / watchlist)
# ---------------------------------------------------------------------------


def state_dir(state_dir: str | None = None) -> str:
    """Löst das State-Verzeichnis auf.

    Priorität: expliziter Parameter > CONCILIUM_STATE_DIR-Env > 'state'
    (relativer Pfad, wie bisher).
    """
    if state_dir is not None:
        return state_dir
    return _env("CONCILIUM_STATE_DIR", "state")


def reports_dir() -> str:
    """Löst das Reports-Verzeichnis auf.

    Priorität: CONCILIUM_REPORTS_DIR-Env > <repo>/reports (absoluter Pfad).
    """
    return _env("CONCILIUM_REPORTS_DIR", os.path.join(_repo_root(), "reports"))


def cache_dir() -> str | None:
    """Löst das Cache-Verzeichnis auf.

    Priorität: CONCILIUM_CACHE_DIR-Env > <repo>/cache.

    Spezialfall (Rückwärtskompatibilität, VOR der Koerzion geprüft):
    CONCILIUM_CACHE_DIR="" (leerer String) → None = Cache DEAKTIVIERT.
    """
    env = os.environ.get("CONCILIUM_CACHE_DIR")
    if env is not None:
        env = env.strip()
        if not env:
            return None  # leerer String → Cache deaktiviert
        return env
    return os.path.join(_repo_root(), "cache")


def watchlist_path() -> str:
    """Löst den Watchlist-Pfad auf.

    Priorität: CONCILIUM_WATCHLIST-Env > <repo>/watchlist.txt.
    """
    return _env("CONCILIUM_WATCHLIST", os.path.join(_repo_root(), "watchlist.txt"))


# ---------------------------------------------------------------------------
# LLM-Konfiguration
# ---------------------------------------------------------------------------


def llm_base_url() -> str:
    """Base-URL des OpenAI-kompatiblen Endpunkts.

    Priorität: LLM_BASE_URL-Env > 'https://ollama.com/v1'.
    """
    return _env("LLM_BASE_URL", "https://ollama.com/v1")


def llm_api_key() -> str:
    """API-Key für den LLM-Endpunkt.

    Priorität: LLM_API_KEY-Env > OLLAMA_API_KEY-Env > ''.
    """
    api_key = os.environ.get("LLM_API_KEY")
    if api_key:
        return api_key
    return os.environ.get("OLLAMA_API_KEY", "")


def llm_model() -> str:
    """Primäres LLM-Modell.

    Priorität: LLM_MODEL-Env > 'glm-5.3-flash'.
    """
    return _env("LLM_MODEL", "glm-5.3-flash")


def llm_fallback_model() -> str:
    """Fallback-LLM-Modell ('' = kein Fallback).

    Priorität: LLM_FALLBACK_MODEL-Env > ''.
    """
    return _env("LLM_FALLBACK_MODEL", "")


def llm_deep_think_model() -> str:
    """Deep-Think-Modell für komplexe Reasoning-Agenten ('' = kein Split).

    Wird von Risiko-Debatte, Trade-Revision und Portfolio-Manager genutzt.
    Priorität: LLM_DEEP_THINK_MODEL-Env > '' (leer = primäres Modell,
    bisheriges Verhalten).
    """
    return _env("LLM_DEEP_THINK_MODEL", "")


def llm_quick_think_model() -> str:
    """Quick-Think-Modell für schnelle Agenten ('' = kein Split).

    Wird von Analysten, Bull/Bear-Debatte und Trader genutzt.
    Priorität: LLM_QUICK_THINK_MODEL-Env > '' (leer = primäres Modell,
    bisheriges Verhalten).
    """
    return _env("LLM_QUICK_THINK_MODEL", "")


# ---------------------------------------------------------------------------
# Risiko-Debatte
# ---------------------------------------------------------------------------


def risk_debate_rounds() -> int:
    """Anzahl der Risiko-Debatten-Runden (1 oder 2).

    Priorität: CONCILIUM_RISK_DEBATE_ROUNDS-Env > 2 (Default).
    Typ-koerziert (int) via _coerce — bei Tippfehler LAUTE ValueError
    mit Env-Variablen-Namen (gleiche Koerzion wie in _env, nur ohne
    den dortigen str()-Wrap, damit der Rückgabetyp int bleibt).
    """
    raw = os.environ.get("CONCILIUM_RISK_DEBATE_ROUNDS")
    if raw is None:
        return 2
    return cast(int, _coerce(raw, 2, key="CONCILIUM_RISK_DEBATE_ROUNDS"))


# ---------------------------------------------------------------------------
# Journal-Hygiene (Idempotenz-Guard + Rotation aufgelöster Einträge)
# ---------------------------------------------------------------------------


def journal_max_resolved() -> int:
    """Cap auf aufgelöste (resolved) Journal-Einträge (0 = Rotation aus).

    Priorität: CONCILIUM_JOURNAL_MAX_RESOLVED-Env > 0 (Default).
    Default 0 = Rotation DEAKTIVIERT (Rückwärtskompatibilität: exakt
    bisheriges Verhalten, das Journal wächst unbegrenzt).
    Wert > 0 = nach jedem Append werden die ältesten resolved-Einträge
    (nach resolved_at, Fallback timestamp) geprunt, bis höchstens N
    resolved-Einträge übrig sind. Pending- und Legacy-Einträge werden NIE
    geprunt (Journal-Hygiene analog TradingAgents' TradingMemoryLog).
    """
    raw = os.environ.get("CONCILIUM_JOURNAL_MAX_RESOLVED")
    if raw is None:
        return 0
    return cast(int, _coerce(raw, 0, key="CONCILIUM_JOURNAL_MAX_RESOLVED"))
