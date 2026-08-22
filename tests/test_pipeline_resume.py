"""Tests für Pipeline-Resume (Checkpoint-Crash-Resilienz).

Simuliert einen Crash nach Schritt N (z. B. debate) und verifiziert, dass
ein zweiter Lauf mit resume=True nur die fehlenden Schritte ausführt.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Isoliertes state-Verzeichnis für Pipeline-Resume-Tests."""
    d = tmp_path / "state"
    d.mkdir()
    monkeypatch.setenv("CONCILIUM_STATE_DIR", str(d))
    return str(d)


# Hilfsdaten für gemockte Agenten-Rückgaben
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

_MOCK_DEBATE = {
    "bull": {"_raw": "Bull argument"},
    "bear": {"_raw": "Bear argument"},
}

_MOCK_TRADE = {
    "rolle": "Trader",
    "aktion": "KAUFEN",
    "rating": "KAUFEN",
    "zielkurs": 115,
    "stop_loss": 92,
    "positionsanteil": 3,
    "_raw": "",
}

_MOCK_RISK = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}

_MOCK_PORTFOLIO_FIT = {"portfolio_fit_score": 2}

_MOCK_REVISED_TRADE = {
    "rolle": "Trader",
    "aktion": "KAUFEN",
    "rating": "KAUFEN",
    "zielkurs": 110,
    "stop_loss": 95,
    "positionsanteil": 2,
    "_raw": "revised",
}

_MOCK_FINAL = {"entscheidung": "GENEHMIGT", "confidence": 4, "begründung": "Sieht gut aus."}


def _patch_all_agents(**overrides):
    """Patcht alle Agenten-Funktionen + Daten + Journal.

    kwargs in overrides können side_effect/setReturn für einzelne Agenten überschreiben.
    Gibt das dict der patcher-Objekte zurück (für .start()/.stop() oder context).
    """
    defaults = {
        "collect_ticker_data": MagicMock(return_value=_MOCK_DATA),
        "analyst_team": MagicMock(return_value=_MOCK_ANALYSTS),
        "debate": MagicMock(return_value=_MOCK_DEBATE),
        "trader": MagicMock(return_value=_MOCK_TRADE),
        "ensemble_trader": MagicMock(return_value=_MOCK_TRADE),
        "risk_manager": MagicMock(return_value=_MOCK_RISK),
        "fetch_portfolio_positions": MagicMock(return_value=[]),
        "portfolio_fit_agent": MagicMock(return_value=_MOCK_PORTFOLIO_FIT),
        "trade_revision": MagicMock(return_value=_MOCK_REVISED_TRADE),
        "portfolio_manager": MagicMock(return_value=_MOCK_FINAL),
        "build_feedback_context": MagicMock(return_value=""),
        "build_reflection_context": MagicMock(return_value=""),
    }
    defaults.update(overrides)
    return defaults


# --------------------------------------------------------------------------- #
# Resume-Szenario: Crash nach debate, Resume führt nur trade..final aus
# --------------------------------------------------------------------------- #


class TestResumeAfterCrash:
    """Crash nach Schritt 3 (debate) → Resume führt nur Schritte 4-6 aus."""

    def test_crash_leaves_checkpoint(self, state_dir):
        """Erster Lauf crasht bei risk_manager → Checkpoint bis debate vorhanden."""
        from concilium.checkpoint import load_checkpoint
        from concilium.pipeline import run_pipeline

        agents = _patch_all_agents(
            risk_manager=MagicMock(side_effect=RuntimeError("HTTP 429 — Rate Limited")),
        )

        with patch.multiple(
            "concilium.pipeline",
            **agents,
        ), patch("concilium.journal.append_decision"):
            with pytest.raises(RuntimeError, match="HTTP 429"):
                run_pipeline("TEST", llm=MagicMock(), ensemble=False, resume=False)

        # Checkpoint sollte existieren mit _completed_steps bis "debate"
        cp = load_checkpoint("TEST")
        assert cp is not None
        assert "data" in cp["_completed_steps"]
        assert "analysts" in cp["_completed_steps"]
        assert "debate" in cp["_completed_steps"]
        # risk wurde nicht abgeschlossen
        assert "risk" not in cp["_completed_steps"]

    def test_resume_completes_remaining_steps(self, state_dir):
        """Nach Crash: resume=True führt nur fehlende Schritte aus,Pipeline komplett."""
        from concilium.pipeline import run_pipeline

        # Phase 1: Crash bei risk_manager
        agents_crash = _patch_all_agents(
            risk_manager=MagicMock(side_effect=RuntimeError("HTTP 429")),
        )
        with patch.multiple("concilium.pipeline", **agents_crash), \
             patch("concilium.journal.append_decision"):
            with pytest.raises(RuntimeError):
                run_pipeline("TEST", llm=MagicMock(), ensemble=False, resume=False)

        # Phase 2: Resume — risk_manager funktioniert jetzt
        agents_ok = _patch_all_agents()  # alle funktionieren
        with patch.multiple("concilium.pipeline", **agents_ok), \
             patch("concilium.journal.append_decision"):
            result = run_pipeline("TEST", llm=MagicMock(), ensemble=False, resume=True)

        # Pipeline sollte komplett sein
        assert result["final"]["entscheidung"] == "GENEHMIGT"
        assert "final" in result["_completed_steps"]

    def test_resume_does_not_recompute_completed_steps(self, state_dir):
        """Bei Resume werden bereits fertige Schritte NICHT neu aufgerufen."""
        from concilium.pipeline import run_pipeline

        # Phase 1: Crash bei risk_manager (nach analysts + debate)
        agents_crash = _patch_all_agents(
            risk_manager=MagicMock(side_effect=RuntimeError("Crash")),
        )
        with patch.multiple("concilium.pipeline", **agents_crash), \
             patch("concilium.journal.append_decision"):
            with pytest.raises(RuntimeError):
                run_pipeline("TEST", llm=MagicMock(), ensemble=False, resume=False)

        # Phase 2: Resume — prüfe dass analyst_team + debate NICHT erneut aufgerufen werden
        agents_resume = _patch_all_agents()
        with patch.multiple("concilium.pipeline", **agents_resume), \
             patch("concilium.journal.append_decision"):
            run_pipeline("TEST", llm=MagicMock(), ensemble=False, resume=True)

            # analyst_team und debate sollten beim Resume NICHT aufgerufen worden sein
            agents_resume["analyst_team"].assert_not_called()
            agents_resume["debate"].assert_not_called()
            # Aber risk_manager, trade_revision, portfolio_manager SOLLTEN aufgerufen worden sein
            agents_resume["risk_manager"].assert_called_once()
            agents_resume["portfolio_manager"].assert_called_once()

    def test_successful_run_clears_checkpoint(self, state_dir):
        """Ein erfolgreicher Lauf (ohne Crash) räumt den Checkpoint auf."""
        from concilium.checkpoint import load_checkpoint
        from concilium.pipeline import run_pipeline

        agents = _patch_all_agents()
        with patch.multiple("concilium.pipeline", **agents), \
             patch("concilium.journal.append_decision"):
            result = run_pipeline("TEST", llm=MagicMock(), ensemble=False, resume=False)

        assert result["final"]["entscheidung"] == "GENEHMIGT"
        # Checkpoint sollte weg sein
        assert load_checkpoint("TEST") is None

    def test_successful_resume_clears_checkpoint(self, state_dir):
        """Nach Resume+Abschluss wird der Checkpoint aufgeräumt."""
        from concilium.checkpoint import load_checkpoint
        from concilium.pipeline import run_pipeline

        # Phase 1: Crash
        agents_crash = _patch_all_agents(
            risk_manager=MagicMock(side_effect=RuntimeError("Crash")),
        )
        with patch.multiple("concilium.pipeline", **agents_crash), \
             patch("concilium.journal.append_decision"):
            with pytest.raises(RuntimeError):
                run_pipeline("TEST", llm=MagicMock(), ensemble=False, resume=False)

        assert load_checkpoint("TEST") is not None

        # Phase 2: Resume erfolgreich
        agents_ok = _patch_all_agents()
        with patch.multiple("concilium.pipeline", **agents_ok), \
             patch("concilium.journal.append_decision"):
            run_pipeline("TEST", llm=MagicMock(), ensemble=False, resume=True)

        assert load_checkpoint("TEST") is None


# --------------------------------------------------------------------------- #
# resume=False ignoriert bestehenden Checkpoint
# --------------------------------------------------------------------------- #


class TestNoResumeIgnoresCheckpoint:
    """resume=False startet immer von vorn, selbst wenn ein Checkpoint existiert."""

    def test_no_resume_recomputes_everything(self, state_dir):
        """Bei resume=False werden alle Schritte neu ausgeführt."""
        from concilium.pipeline import run_pipeline

        # Phase 1: Erstelle einen Checkpoint durch einen erfolgreichen Lauf
        agents1 = _patch_all_agents()
        with patch.multiple("concilium.pipeline", **agents1), \
             patch("concilium.journal.append_decision"):
            run_pipeline("TEST", llm=MagicMock(), ensemble=False, resume=False)

        # Jetzt existiert KEIN Checkpoint mehr (erfolgreicher Lauf räumt auf).
        # Aber wir können manuell einen hinterlegen:
        from concilium.checkpoint import save_checkpoint
        save_checkpoint(
            {
                "ticker": "TEST",
                "data": _MOCK_DATA,
                "_data_text": None,
                "_feedback_context": "",
                "_reflection_context": "",
                "analysts": _MOCK_ANALYSTS,
                "debate": _MOCK_DEBATE,
                "_completed_steps": ["data", "analysts", "debate"],
            },
            "TEST",
        )

        # Phase 2: resume=False → alle Schritte neu
        agents2 = _patch_all_agents()
        with patch.multiple("concilium.pipeline", **agents2), \
             patch("concilium.journal.append_decision"):
            run_pipeline("TEST", llm=MagicMock(), ensemble=False, resume=False)

            # analyst_team sollte aufgerufen worden sein (resume=False)
            agents2["analyst_team"].assert_called_once()
            agents2["debate"].assert_called_once()


# --------------------------------------------------------------------------- #
# Resume ohne bestehenden Checkpoint → startet von vorn
# --------------------------------------------------------------------------- #


class TestResumeNoCheckpoint:
    """resume=True ohne Checkpoint → normaler Lauf von vorn."""

    def test_resume_without_checkpoint_starts_fresh(self, state_dir):
        """Wenn kein Checkpoint da ist, verhält sich resume=True wie resume=False."""
        from concilium.pipeline import run_pipeline

        agents = _patch_all_agents()
        with patch.multiple("concilium.pipeline", **agents), \
             patch("concilium.journal.append_decision"):
            result = run_pipeline("TEST", llm=MagicMock(), ensemble=False, resume=True)

        assert result["final"]["entscheidung"] == "GENEHMIGT"
        agents["analyst_team"].assert_called_once()


# --------------------------------------------------------------------------- #
# Crash bei verschiedenen Schritten
# --------------------------------------------------------------------------- #


class TestCrashAtDifferentSteps:
    """Crash bei verschiedenen Schritten — Checkpoint enthält richtige Daten."""

    def test_crash_at_trader(self, state_dir):
        """Crash bei Schritt 4 (trader) → Checkpoint bis debate."""
        from concilium.checkpoint import load_checkpoint
        from concilium.pipeline import run_pipeline

        agents = _patch_all_agents(
            trader=MagicMock(side_effect=RuntimeError("429")),
            ensemble_trader=MagicMock(side_effect=RuntimeError("429")),
        )
        with patch.multiple("concilium.pipeline", **agents), \
             patch("concilium.journal.append_decision"):
            with pytest.raises(RuntimeError):
                run_pipeline("TEST", llm=MagicMock(), ensemble=False, resume=False)

        cp = load_checkpoint("TEST")
        assert cp is not None
        assert "debate" in cp["_completed_steps"]
        assert "trade" not in cp["_completed_steps"]

    def test_crash_at_portfolio_manager(self, state_dir):
        """Crash bei Schritt 6 (portfolio_manager) → Checkpoint bis trade_revision."""
        from concilium.checkpoint import load_checkpoint
        from concilium.pipeline import run_pipeline

        agents = _patch_all_agents(
            portfolio_manager=MagicMock(side_effect=RuntimeError("LLM down")),
        )
        with patch.multiple("concilium.pipeline", **agents), \
             patch("concilium.journal.append_decision"):
            with pytest.raises(RuntimeError):
                run_pipeline("TEST", llm=MagicMock(), ensemble=False, resume=False)

        cp = load_checkpoint("TEST")
        assert cp is not None
        assert "trade_revision" in cp["_completed_steps"]
        assert "final" not in cp["_completed_steps"]


# --------------------------------------------------------------------------- #
# No-LLM-Modus: Checkpoint wird aufgeräumt
# --------------------------------------------------------------------------- #


class TestNoLlmCheckpoint:
    """Im No-LLM-Modus wird ein eventuell vorhandener Checkpoint aufgeräumt."""

    def test_no_llm_clears_checkpoint(self, state_dir):
        """run_pipeline(llm=None) räumt den Checkpoint auf."""
        from concilium.checkpoint import load_checkpoint, save_checkpoint
        from concilium.pipeline import run_pipeline

        # Checkpoint manuell anlegen (mit data-Key, realistisch)
        save_checkpoint(
            {
                "ticker": "TEST",
                "data": _MOCK_DATA,
                "_data_text": None,
                "_feedback_context": "",
                "_reflection_context": "",
                "_completed_steps": ["data"],
            },
            "TEST",
        )
        assert load_checkpoint("TEST") is not None

        with patch("concilium.pipeline.collect_ticker_data", return_value=_MOCK_DATA):
            run_pipeline("TEST", llm=None, resume=True)

        # Checkpoint sollte weg sein
        assert load_checkpoint("TEST") is None


# --------------------------------------------------------------------------- #
# Ensemble + Resume
# --------------------------------------------------------------------------- #


class TestEnsembleResume:
    """Ensemble-Trader mit Resume —Crash nach debate, Resume mit ensemble=True."""

    def test_resume_with_ensemble(self, state_dir):
        """Resume funktioniert auch mit ensemble=True."""
        from concilium.pipeline import run_pipeline

        # Phase 1: Crash bei ensemble_trader
        agents_crash = _patch_all_agents(
            ensemble_trader=MagicMock(side_effect=RuntimeError("429")),
        )
        with patch.multiple("concilium.pipeline", **agents_crash), \
             patch("concilium.journal.append_decision"):
            with pytest.raises(RuntimeError):
                run_pipeline("TEST", llm=MagicMock(), ensemble=True, resume=False)

        # Phase 2: Resume mit ensemble=True
        agents_ok = _patch_all_agents()
        with patch.multiple("concilium.pipeline", **agents_ok), \
             patch("concilium.journal.append_decision"):
            result = run_pipeline("TEST", llm=MagicMock(), ensemble=True, resume=True)

        assert result["final"]["entscheidung"] == "GENEHMIGT"
        # ensemble_trader sollte beim Resume aufgerufen worden sein
        agents_ok["ensemble_trader"].assert_called_once()
