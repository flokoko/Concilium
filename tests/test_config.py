"""Tests für die zentrale config.py — Lazy-Loading und Typ-Koerzion.

Die kritische Eigenschaft: config.py liest Env-Variablen FRISCH bei jedem
Aufruf (kein Import-Cache). Die Tests setzen Env-Vars per monkeypatch NACH
dem Import — genau wie die autouse-Fixtures in conftest.py.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import pytest  # noqa: E402

from concilium import config  # noqa: E402

# ---------------------------------------------------------------------------
# Lazy-Loading: Env-Var wird bei JEDEM Zugriff frisch gelesen
# ---------------------------------------------------------------------------


class TestLazyLoading:
    """Env-Vars, die NACH dem Import gesetzt werden, müssen wirken."""

    def test_state_dir_env_set_after_import(self, monkeypatch):
        """CONCILIUM_STATE_DIR nach Import → config.state_dir() folgt sofort."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", "/tmp/nach_import_state")
        assert config.state_dir() == "/tmp/nach_import_state"

    def test_state_dir_env_change_between_calls(self, monkeypatch):
        """Zwei Aufrufe mit dazwischen geändertem Env → zwei verschiedene Werte."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", "/tmp/erster")
        first = config.state_dir()
        monkeypatch.setenv("CONCILIUM_STATE_DIR", "/tmp/zweiter")
        second = config.state_dir()
        assert first == "/tmp/erster"
        assert second == "/tmp/zweiter"

    def test_cache_dir_env_set_after_import(self, monkeypatch, tmp_path):
        """CONCILIUM_CACHE_DIR nach Import → config.cache_dir() folgt sofort."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", str(tmp_path))
        assert config.cache_dir() == str(tmp_path)

    def test_cache_dir_disabled_after_import(self, monkeypatch):
        """CONCILIUM_CACHE_DIR="" nach Import → None (Cache deaktiviert)."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", "")
        assert config.cache_dir() is None

    def test_reports_dir_env_set_after_import(self, monkeypatch):
        """CONCILIUM_REPORTS_DIR nach Import → config.reports_dir() folgt sofort."""
        monkeypatch.setenv("CONCILIUM_REPORTS_DIR", "/tmp/nach_import_reports")
        assert config.reports_dir() == "/tmp/nach_import_reports"

    def test_watchlist_path_env_set_after_import(self, monkeypatch, tmp_path):
        """CONCILIUM_WATCHLIST nach Import → config.watchlist_path() folgt sofort."""
        monkeypatch.setenv("CONCILIUM_WATCHLIST", str(tmp_path / "wl.txt"))
        assert config.watchlist_path() == str(tmp_path / "wl.txt")


# ---------------------------------------------------------------------------
# Defaults (Env unset)
# ---------------------------------------------------------------------------


class TestDefaults:
    """Ohne Env-Variable greifen die bisherigen Defaults."""

    def test_state_dir_default_relative(self, monkeypatch):
        """Ohne Env → 'state' (relativ, wie bisher)."""
        monkeypatch.delenv("CONCILIUM_STATE_DIR", raising=False)
        assert config.state_dir() == "state"

    def test_state_dir_explicit_param_wins(self, monkeypatch):
        """Expliziter Parameter schlägt Env."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", "/tmp/env_state")
        assert config.state_dir("/tmp/explizit") == "/tmp/explizit"

    def test_cache_dir_default_repo_root(self, monkeypatch):
        """Ohne Env → <repo>/cache (Repo-Root = Eltern von src/concilium/)."""
        monkeypatch.delenv("CONCILIUM_CACHE_DIR", raising=False)
        result = config.cache_dir()
        assert result is not None
        assert result.endswith("cache")

    def test_reports_dir_default_repo_root(self, monkeypatch):
        """Ohne Env → <repo>/reports (absolut)."""
        monkeypatch.delenv("CONCILIUM_REPORTS_DIR", raising=False)
        result = config.reports_dir()
        assert result.endswith("reports")
        assert os.path.isabs(result)

    def test_watchlist_path_default_repo_root(self, monkeypatch):
        """Ohne Env → <repo>/watchlist.txt."""
        monkeypatch.delenv("CONCILIUM_WATCHLIST", raising=False)
        result = config.watchlist_path()
        assert result.endswith("watchlist.txt")


# ---------------------------------------------------------------------------
# LLM-Konfiguration
# ---------------------------------------------------------------------------


class TestLLMConfig:
    """LLM-Env-Vars: Prioritäten und Defaults."""

    def test_llm_base_url_default(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        assert config.llm_base_url() == "https://ollama.com/v1"

    def test_llm_base_url_env(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
        assert config.llm_base_url() == "http://localhost:11434/v1"

    def test_llm_api_key_fallback_chain(self, monkeypatch):
        """LLM_API_KEY > OLLAMA_API_KEY > ''."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        assert config.llm_api_key() == ""
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
        assert config.llm_api_key() == "ollama-key"
        monkeypatch.setenv("LLM_API_KEY", "llm-key")
        assert config.llm_api_key() == "llm-key"

    def test_llm_model_default(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        assert config.llm_model() == "glm-5.3-flash"

    def test_llm_model_env(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        assert config.llm_model() == "gpt-4o"

    def test_llm_fallback_model_default_empty(self, monkeypatch):
        monkeypatch.delenv("LLM_FALLBACK_MODEL", raising=False)
        assert config.llm_fallback_model() == ""


# ---------------------------------------------------------------------------
# Typ-Koerzion (_coerce) — inkl. LAUTER Fehlermeldung
# ---------------------------------------------------------------------------


class TestCoerce:
    """_coerce: Typ vom Default-Wert, lauter ValueError bei Tippfehlern."""

    def test_bool_truthy_values(self):
        for raw in ("true", "1", "yes", "on", "TRUE", "On", "  yes  "):
            assert config._coerce(raw, False) is True

    def test_bool_falsy_values(self):
        for raw in ("false", "0", "no", "off", "FALSE", "Off", "  no  "):
            assert config._coerce(raw, True) is False

    def test_bool_typo_raises_loudly(self):
        """Tippfehler 'treu' → ValueError MIT Env-Variablen-Namen."""
        with pytest.raises(ValueError) as exc_info:
            config._coerce("treu", False, key="CONCILIUM_CACHE_DIR")
        msg = str(exc_info.value)
        assert "CONCILIUM_CACHE_DIR" in msg
        assert "treu" in msg
        assert "boolean" in msg

    def test_int_valid(self):
        assert config._coerce("42", 0) == 42

    def test_int_invalid_raises_with_key(self):
        with pytest.raises(ValueError) as exc_info:
            config._coerce("zwei", 0, key="LLM_MAX_RETRIES")
        msg = str(exc_info.value)
        assert "LLM_MAX_RETRIES" in msg
        assert "integer" in msg

    def test_float_valid(self):
        assert config._coerce("0.5", 0.0) == pytest.approx(0.5)

    def test_float_invalid_raises_with_key(self):
        with pytest.raises(ValueError) as exc_info:
            config._coerce("half", 0.0, key="X")
        assert "float" in str(exc_info.value)

    def test_str_passthrough(self):
        assert config._coerce("/tmp/pfad", "default") == "/tmp/pfad"

    def test_invalid_value_message_format(self):
        """Spec-Format: 'Invalid value for <KEY>: expected a boolean ...'."""
        with pytest.raises(ValueError, match=r"Invalid value for TEST_VAR: expected"):
            config._coerce("treu", False, key="TEST_VAR")


# ---------------------------------------------------------------------------
# Delegation: Die bestehenden Module lesen über config.py
# ---------------------------------------------------------------------------


class TestDelegation:
    """Die Modul-Helper delegieren an config.py (gleiche Werte)."""

    def test_checkpoint_state_dir_matches_config(self, monkeypatch):
        from concilium import checkpoint

        monkeypatch.setenv("CONCILIUM_STATE_DIR", "/tmp/deleg_state")
        assert checkpoint._state_dir() == config.state_dir()
        assert checkpoint._state_dir("/tmp/x") == "/tmp/x"

    def test_feedback_state_dir_matches_config(self, monkeypatch):
        from concilium import feedback

        monkeypatch.setenv("CONCILIUM_STATE_DIR", "/tmp/deleg_state")
        assert feedback._state_dir() == config.state_dir()
        assert feedback._state_dir("/tmp/x") == "/tmp/x"

    def test_agents_state_dir_matches_config(self, monkeypatch):
        from concilium import agents

        monkeypatch.setenv("CONCILIUM_STATE_DIR", "/tmp/deleg_state")
        assert agents._ensemble_state_dir() == config.state_dir()

    def test_data_cache_dir_matches_config(self, monkeypatch, tmp_path):
        from concilium import data

        monkeypatch.setenv("CONCILIUM_CACHE_DIR", str(tmp_path))
        assert data._get_cache_dir() == config.cache_dir()
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", "")
        assert data._get_cache_dir() is None

    def test_cli_dirs_match_config(self, monkeypatch):
        from concilium import cli

        monkeypatch.setenv("CONCILIUM_STATE_DIR", "/tmp/deleg_state")
        monkeypatch.setenv("CONCILIUM_REPORTS_DIR", "/tmp/deleg_reports")
        assert cli._state_dir() == config.state_dir()
        assert cli._reports_dir() == config.reports_dir()
        assert cli._state_dir("/tmp/x") == "/tmp/x"

    def test_cli_watchlist_default_from_config(self, monkeypatch, tmp_path):
        """_read_watchlist nutzt config.watchlist_path() als Default."""
        from concilium import cli

        wl = tmp_path / "watchlist.txt"
        wl.write_text("TEST\n", encoding="utf-8")
        monkeypatch.setenv("CONCILIUM_WATCHLIST", str(wl))
        assert cli._read_watchlist() == ["TEST"]
