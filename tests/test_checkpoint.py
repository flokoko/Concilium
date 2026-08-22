"""Tests für das Checkpoint-Persistenz-Modul."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.checkpoint import (  # noqa: E402
    _normalize_ticker,
    clear_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Isoliertes state-Verzeichnis via CONCILIUM_STATE_DIR."""
    d = tmp_path / "state"
    d.mkdir()
    monkeypatch.setenv("CONCILIUM_STATE_DIR", str(d))
    return str(d)


# --------------------------------------------------------------------------- #
# _normalize_ticker
# --------------------------------------------------------------------------- #


class TestNormalizeTicker:
    def test_dot_replaced(self):
        assert _normalize_ticker("RWE.DE") == "RWE_DE"

    def test_colon_replaced(self):
        assert _normalize_ticker("SHEL.L") == "SHEL_L"

    def test_alphanumeric_preserved(self):
        assert _normalize_ticker("AAPL") == "AAPL"

    def test_dash_preserved(self):
        assert _normalize_ticker("BRK-B") == "BRK-B"

    def test_multiple_special_chars(self):
        assert _normalize_ticker("A.B.C") == "A_B_C"


# --------------------------------------------------------------------------- #
# save / load Roundtrip
# --------------------------------------------------------------------------- #


class TestSaveLoad:
    def test_save_load_roundtrip(self, state_dir):
        """save → load liefert die gleichen Daten zurück."""
        result = {
            "ticker": "AAPL",
            "data": {"ticker": "AAPL", "price": 150},
            "analysts": {"fundamental": {"score": 4}},
            "_completed_steps": ["data", "analysts"],
        }
        save_checkpoint(result, "AAPL")
        loaded = load_checkpoint("AAPL")
        assert loaded is not None
        assert loaded["ticker"] == "AAPL"
        assert loaded["analysts"]["fundamental"]["score"] == 4
        assert loaded["_completed_steps"] == ["data", "analysts"]

    def test_save_creates_file(self, state_dir):
        """save legt eine Datei an."""
        save_checkpoint({"ticker": "TEST"}, "TEST")
        # Datei sollte existieren
        files = os.listdir(state_dir)
        assert any("TEST" in f for f in files)

    def test_save_overwrites_existing(self, state_dir):
        """Ein zweites save überschreibt den ersten Checkpoint."""
        save_checkpoint({"ticker": "X", "val": 1}, "X")
        save_checkpoint({"ticker": "X", "val": 2}, "X")
        loaded = load_checkpoint("X")
        assert loaded["val"] == 2

    def test_normalized_filename_for_dot_ticker(self, state_dir):
        """RWE.DE → Dateiname RWE_DE_checkpoint.json."""
        save_checkpoint({"ticker": "RWE.DE"}, "RWE.DE")
        assert os.path.isfile(os.path.join(state_dir, "RWE_DE_checkpoint.json"))

    def test_numpy_like_value_serialized(self, state_dir):
        """default=str serialisiert nicht-JSON-native Typen tolerant."""
        # Simuliere ein numpy-int64-ähnliches Objekt
        class FakeInt:
            def __str__(self):
                return "42"

        result = {"ticker": "NP", "score": FakeInt()}
        save_checkpoint(result, "NP")
        loaded = load_checkpoint("NP")
        assert loaded is not None
        assert loaded["score"] == "42"

    def test_checkpoint_version_added(self, state_dir):
        """save fügt _checkpoint_version hinzu."""
        save_checkpoint({"ticker": "V"}, "V")
        loaded = load_checkpoint("V")
        assert loaded["_checkpoint_version"] == 1


# --------------------------------------------------------------------------- #
# load bei kaputter / fehlender Datei
# --------------------------------------------------------------------------- #


class TestLoadCorrupt:
    def test_load_missing_file_returns_none(self, state_dir):
        """load gibt None zurück, wenn keine Datei existiert."""
        assert load_checkpoint("NONEXIST") is None

    def test_load_corrupt_json_returns_none(self, state_dir):
        """load gibt None zurück, wenn das JSON kaputt ist."""
        # Schreibe kaputtes JSON direkt in die Datei
        path = os.path.join(state_dir, "CORRUPT_checkpoint.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{ broken json !!! ")

        assert load_checkpoint("CORRUPT") is None

    def test_load_non_dict_returns_none(self, state_dir):
        """load gibt None zurück, wenn das JSON ein Array ist."""
        path = os.path.join(state_dir, "ARR_checkpoint.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([1, 2, 3], fh)

        assert load_checkpoint("ARR") is None


# --------------------------------------------------------------------------- #
# clear
# --------------------------------------------------------------------------- #


class TestClear:
    def test_clear_removes_file(self, state_dir):
        """clear entfernt die Checkpoint-Datei."""
        save_checkpoint({"ticker": "CLR"}, "CLR")
        assert load_checkpoint("CLR") is not None
        clear_checkpoint("CLR")
        assert load_checkpoint("CLR") is None

    def test_clear_missing_file_no_error(self, state_dir):
        """clear auf nicht-existente Datei crasht nicht."""
        clear_checkpoint("GHOST")

    def test_clear_after_save(self, state_dir):
        """Speichern, löschen, laden → None."""
        save_checkpoint({"ticker": "C2"}, "C2")
        clear_checkpoint("C2")
        assert load_checkpoint("C2") is None


# --------------------------------------------------------------------------- #
# CONCILIUM_STATE_DIR Isolation
# --------------------------------------------------------------------------- #


class TestStateDirOverride:
    def test_explicit_state_dir_param(self, tmp_path):
        """state_dir-Parameter wird korrekt genutzt."""
        d = str(tmp_path / "custom")
        save_checkpoint({"ticker": "DIR"}, "DIR", state_dir=d)
        assert os.path.isfile(os.path.join(d, "DIR_checkpoint.json"))
        loaded = load_checkpoint("DIR", state_dir=d)
        assert loaded["ticker"] == "DIR"

    def test_env_override(self, tmp_path, monkeypatch):
        """CONCILIUM_STATE_DIR-Env steuert das Verzeichnis."""
        d = str(tmp_path / "envstate")
        monkeypatch.setenv("CONCILIUM_STATE_DIR", d)
        save_checkpoint({"ticker": "ENV"}, "ENV")
        assert os.path.isfile(os.path.join(d, "ENV_checkpoint.json"))

    def test_no_env_uses_default_state(self, tmp_path, monkeypatch):
        """Ohne Env wird 'state' relativ zum CWD verwendet."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CONCILIUM_STATE_DIR", raising=False)
        save_checkpoint({"ticker": "DEF"}, "DEF")
        assert os.path.isfile(os.path.join(str(tmp_path), "state", "DEF_checkpoint.json"))


# --------------------------------------------------------------------------- #
# Atomarität
# --------------------------------------------------------------------------- #


class TestAtomicity:
    def test_no_tmpfile_left_after_save(self, state_dir):
        """Nach save gibt es keine .tmp-Datei."""
        save_checkpoint({"ticker": "ATOM"}, "ATOM")
        files = os.listdir(state_dir)
        assert not any(f.endswith(".tmp") for f in files)
