"""Tests für Bug 1: Ziel-Gewichtungs-Dämpfung NACH der Trade-Revision.

Die kalibrierungs-gestützte Dämpfung der Ziel-Gewichtung (Schritt 5b') muss
auf der FINALEN (revidierten) Trade-Aktion basieren. Die Trade-Revision
(Schritt 5c) kann die Aktion ändern (z. B. KAUFEN→HALTEN wegen Risk-/
Portfolio-Einwand) — die Dämpfung läuft daher NACH Schritt 5c.

Getestet werden:
- Reihenfolge: bei KAUFEN→HALTEN-Revision wird mit der revidierten Aktion
  gedämpft (nicht mit der Original-Aktion)
- Ohne Revisions-Änderung: identisches Verhalten wie bisher (KAUFEN bleibt
  maßgeblich)
- Resume-Idempotenz: bereits gedämpfte Werte werden bei Resume nicht
  erneut gedämpft (keine Doppel-Skalierung 10.0 → 5.2 → 2.7)
- Resume mit abgeschlossenem trade_revision: Dämpfung läuft trotzdem
  (der Block steht bewusst NICHT unter dem trade_revision-Guard)

Alle Tests sind OFFLINE-fähig: kein yfinance, kein Netzwerk, kein LLM.
Die echte state/calibration.json wird NIEMALS angefasst — alle Tests
schreiben in tmp_path und setzen CONCILIUM_STATE_DIR.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from concilium.pipeline import _pipeline_fingerprint, run_pipeline  # noqa: E402

# --------------------------------------------------------------------------- #
# Mock-Daten (analog test_ziel_gewichtung_daempfung.py / test_pipeline_resume.py)
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

_MOCK_FINAL = {"entscheidung": "GENEHMIGT", "confidence": 4, "begründung": "Ok."}


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


def _write_calibration_json(tmp_path, hit_rates: dict[str, float]) -> None:
    """Schreibt eine frische calibration.json in tmp_path/state/."""
    import json
    from datetime import datetime

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "erstellt_am": datetime.now().isoformat(),
        "anzahl_entscheidungen": 10,
        "hit_rate_gesamt": 0.4,
        "nach_aktion": {
            action: {"n": 5, "hit_rate": rate, "avg_confidence": 0.6}
            for action, rate in hit_rates.items()
        },
    }
    (state_dir / "calibration.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def _patched_pipeline(trade: dict, portfolio_fit: dict | None, *, revised_trade: dict | None = None):
    """Gibt den patch.multiple-Kontext für einen komplett gemockten Pipeline-Lauf zurück.

    revised_trade: Rückgabe von trade_revision (None = Original-Trade, keine Änderung).
    """
    patches = {
        "collect_ticker_data": MagicMock(return_value=_MOCK_DATA),
        "analyst_team": MagicMock(return_value=_MOCK_ANALYSTS),
        "debate": MagicMock(return_value=_MOCK_DEBATE),
        "trader": MagicMock(return_value=trade),
        "ensemble_trader": MagicMock(return_value=trade),
        "risk_manager": MagicMock(return_value=_MOCK_RISK),
        "fetch_portfolio_positions": MagicMock(return_value=[]),
        "portfolio_fit_agent": MagicMock(return_value=portfolio_fit),
        "trade_revision": MagicMock(return_value=revised_trade if revised_trade is not None else trade),
        "portfolio_manager": MagicMock(return_value=_MOCK_FINAL),
        "build_feedback_context": MagicMock(return_value=""),
        "build_reflection_context": MagicMock(return_value=""),
        # _dampen_ziel_gewichtung wird NICHT gepatcht — läuft real gegen das
        # tmp-CONCILIUM_STATE_DIR des Tests.
    }
    return patches


def _make_llm() -> MagicMock:
    """LLM-Mock mit leerem total_usage (verhindert usage/usage.csv-Einträge)."""
    llm = MagicMock()
    llm.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return llm


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Isoliertes state-Verzeichnis (CONCILIUM_STATE_DIR auf tmp_path)."""
    d = tmp_path / "state"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CONCILIUM_STATE_DIR", str(d))
    return str(d)


# --------------------------------------------------------------------------- #
# Reihenfolge: Dämpfung basiert auf der revidierten Aktion
# --------------------------------------------------------------------------- #


class TestDaempfungNachTradeRevision:
    """Der 5b'-Block läuft NACH Schritt 5c und nutzt die revidierte Aktion."""

    def test_revision_aendert_aktion_daempfung_nutzt_revidierte(self, tmp_path, state_dir):
        """KAUFEN→HALTEN-Revision: Dämpfung mit HALTEN-Faktor (0.1429 → clamp 0.3).

        Wäre die Dämpfung (wie vor dem Fix) VOR der Revision mit KAUFEN
        (Faktor 0.52) passiert, ergäbe sich 10.0 → 5.2 — nach dem Fix muss
        der revidierte HALTEN-Wert maßgeblich sein: 10.0 → 3.0.
        """
        _write_calibration_json(
            tmp_path, {"KAUFEN": 0.52, "HALTEN": 0.1429, "VERKAUFEN": 0.0}
        )
        pf = {"portfolio_fit_score": 2, "ziel_gewichtung_pct": 10.0}
        original = _make_trade("KAUFEN")
        revised = _make_trade("HALTEN")

        patches = _patched_pipeline(original, pf, revised_trade=revised)
        with patch.multiple("concilium.pipeline", **patches), patch(
            "concilium.journal.append_decision"
        ):
            result = run_pipeline("TEST", llm=_make_llm(), ensemble=False, resume=False)

        assert result["trade"]["aktion"] == "HALTEN"  # Revision hat gegriffen
        assert result["trade_revised"] is True
        pf_result = result["portfolio_fit"]
        # Mit der REVIDIERTEN Aktion (HALTEN, Faktor 0.3) gedämpft
        assert pf_result["ziel_gewichtung_pct"] == 3.0
        assert pf_result["ziel_gewichtung_original"] == 10.0
        assert pf_result["ziel_gewichtung_gedämpft"] is True

    def test_revision_ohne_aktionsaenderung_kaufens_faktor(self, tmp_path, state_dir):
        """Revision ändert nichts (KAUFEN bleibt KAUFEN) → KAUFEN-Faktor 0.52."""
        _write_calibration_json(
            tmp_path, {"KAUFEN": 0.52, "HALTEN": 0.1429, "VERKAUFEN": 0.0}
        )
        pf = {"portfolio_fit_score": 2, "ziel_gewichtung_pct": 10.0}
        trade = _make_trade("KAUFEN")

        patches = _patched_pipeline(trade, pf)
        with patch.multiple("concilium.pipeline", **patches), patch(
            "concilium.journal.append_decision"
        ):
            result = run_pipeline("TEST", llm=_make_llm(), ensemble=False, resume=False)

        assert result["trade"]["aktion"] == "KAUFEN"
        pf_result = result["portfolio_fit"]
        assert pf_result["ziel_gewichtung_pct"] == 5.2  # KAUFEN-Faktor 0.52
        assert pf_result["ziel_gewichtung_original"] == 10.0
        assert pf_result["ziel_gewichtung_gedämpft"] is True

    def test_daempfung_liefert_wert_fuer_pm_nach_revision(self, tmp_path, state_dir):
        """Der PM sieht den bereits gedämpften Wert (Reihenfolge 5c → 5b' → 6)."""
        _write_calibration_json(
            tmp_path, {"KAUFEN": 0.52, "HALTEN": 0.1429, "VERKAUFEN": 0.0}
        )
        pf = {"portfolio_fit_score": 2, "ziel_gewichtung_pct": 10.0}
        original = _make_trade("KAUFEN")
        revised = _make_trade("HALTEN")

        patches = _patched_pipeline(original, pf, revised_trade=revised)
        with patch.multiple("concilium.pipeline", **patches), patch(
            "concilium.journal.append_decision"
        ):
            result = run_pipeline("TEST", llm=_make_llm(), ensemble=False, resume=False)

        # Der gedämpfte Wert muss im result landen (und damit für Journal + PM
        # sichtbar sein), konsistent zur revidierten Aktion.
        assert result["portfolio_fit"]["ziel_gewichtung_pct"] == 3.0
        assert result["trade"]["aktion"] == "HALTEN"


# --------------------------------------------------------------------------- #
# Resume-Idempotenz: keine Doppel-Dämpfung
# --------------------------------------------------------------------------- #


class TestDaempfungResumeIdempotenz:
    """Resume darf einen bereits gedämpften Wert nicht erneut dämpfen."""

    def test_checkpoint_mit_gedaempfter_gewichtung_nicht_nochmal_gedaempft(
        self, tmp_path, state_dir
    ):
        """Checkpoint enthält bereits gedämpfte portfolio_fit → Resume dämpft NICHT erneut.

        Ohne Idempotenz-Guard würde 10.0 (Original) → 5.2 → 2.7 doppelt
        skaliert. Mit Guard bleibt es bei 5.2.
        """
        _write_calibration_json(
            tmp_path, {"KAUFEN": 0.52, "HALTEN": 0.1429, "VERKAUFEN": 0.0}
        )
        # Manueller Checkpoint: trade_revision + final abgeschlossen,
        # portfolio_fit bereits gedämpft (10.0 → 5.2, Original erhalten).
        pf = {
            "portfolio_fit_score": 2,
            "ziel_gewichtung_pct": 5.2,
            "ziel_gewichtung_original": 10.0,
            "ziel_gewichtung_gedämpft": True,
        }
        save_checkpoint(
            {
                "ticker": "TEST",
                "data": _MOCK_DATA,
                "_data_text": None,
                "_feedback_context": "",
                "_reflection_context": "",
                "analysts": _MOCK_ANALYSTS,
                "debate": _MOCK_DEBATE,
                "trade": _make_trade("HALTEN"),
                "risk": _MOCK_RISK,
                "portfolio_fit": pf,
                "_completed_steps": [
                    "data",
                    "analysts",
                    "debate",
                    "trade",
                    "risk",
                    "portfolio_fit",
                    "trade_revision",
                ],
                # C5: Fingerprint der identischen Resume-Konfiguration —
                # ohne ihn würde der Checkpoint als Altdaten ignoriert.
                "_pipeline_fingerprint": _pipeline_fingerprint(
                    ensemble=False,
                    ensemble_runs=3,
                    peers=None,
                    debate_rounds=1,
                    backtest=False,
                ),
            },
            "TEST",
        )

        patches = _patched_pipeline(_make_trade("HALTEN"), pf)
        with patch.multiple("concilium.pipeline", **patches), patch(
            "concilium.journal.append_decision"
        ):
            result = run_pipeline("TEST", llm=_make_llm(), ensemble=False, resume=True)

        pf_result = result["portfolio_fit"]
        # NICHT erneut gedämpft: 5.2 bleibt 5.2 (keine 5.2*0.3=1.6-Skalierung)
        assert pf_result["ziel_gewichtung_pct"] == 5.2
        assert pf_result["ziel_gewichtung_original"] == 10.0  # Original unverändert
        assert pf_result["ziel_gewichtung_gedämpft"] is True

    def test_resume_laeuft_trotz_abgeschlossenem_trade_revision(self, tmp_path, state_dir):
        """Der Dämpfungsblock steht NICHT unter dem trade_revision-Guard.

        Bei Resume mit abgeschlossenem trade_revision (trade = revidierter
        Trade aus dem Checkpoint, noch NICHT gedämpft) läuft die Dämpfung
        trotzdem — mit der revidierten Aktion des Checkpoint-Trades.
        """
        _write_calibration_json(
            tmp_path, {"KAUFEN": 0.52, "HALTEN": 0.1429, "VERKAUFEN": 0.0}
        )
        pf = {"portfolio_fit_score": 2, "ziel_gewichtung_pct": 10.0}
        revised = _make_trade("HALTEN")
        save_checkpoint(
            {
                "ticker": "TEST",
                "data": _MOCK_DATA,
                "_data_text": None,
                "_feedback_context": "",
                "_reflection_context": "",
                "analysts": _MOCK_ANALYSTS,
                "debate": _MOCK_DEBATE,
                "trade": revised,
                "risk": _MOCK_RISK,
                "portfolio_fit": pf,
                "_completed_steps": [
                    "data",
                    "analysts",
                    "debate",
                    "trade",
                    "risk",
                    "portfolio_fit",
                    "trade_revision",
                ],
                # C5: Fingerprint der identischen Resume-Konfiguration —
                # ohne ihn würde der Checkpoint als Altdaten ignoriert.
                "_pipeline_fingerprint": _pipeline_fingerprint(
                    ensemble=False,
                    ensemble_runs=3,
                    peers=None,
                    debate_rounds=1,
                    backtest=False,
                ),
            },
            "TEST",
        )

        patches = _patched_pipeline(revised, pf)
        with patch.multiple("concilium.pipeline", **patches), patch(
            "concilium.journal.append_decision"
        ) as mock_journal:
            result = run_pipeline("TEST", llm=_make_llm(), ensemble=False, resume=True)

        # Dämpfung TROTZ abgeschlossenem trade_revision ausgeführt —
        # mit der revidierten Aktion (HALTEN, Faktor 0.3): 10.0 → 3.0
        pf_result = result["portfolio_fit"]
        assert pf_result["ziel_gewichtung_pct"] == 3.0
        assert pf_result["ziel_gewichtung_original"] == 10.0
        assert pf_result["ziel_gewichtung_gedämpft"] is True
        # Pipeline lief normal weiter (final + Journal)
        assert result["final"]["entscheidung"] == "GENEHMIGT"
        mock_journal.assert_called_once()
        # Checkpoint wurde aufgeräumt
        assert load_checkpoint("TEST") is None

    def test_double_run_im_gleichen_prozess_idempotent(self, tmp_path, state_dir):
        """Zweiter Lauf auf demselben result-dict dämpft nicht erneut (direkt aufgerufen).

        Simuliert die Situation „Checkpoint geladen, portfolio_fit schon
        gedämpft" — der Guard (ziel_gewichtung_gedämpft=True) verhindert die
        zweite Anwendung unabhängig vom Checkpoint-Mechanismus.
        """
        _write_calibration_json(
            tmp_path, {"KAUFEN": 0.52, "HALTEN": 0.1429, "VERKAUFEN": 0.0}
        )
        pf = {
            "portfolio_fit_score": 2,
            "ziel_gewichtung_pct": 5.2,
            "ziel_gewichtung_original": 10.0,
            "ziel_gewichtung_gedämpft": True,
        }
        patches = _patched_pipeline(_make_trade("KAUFEN"), pf)
        with patch.multiple("concilium.pipeline", **patches), patch(
            "concilium.journal.append_decision"
        ):
            run_pipeline("TEST", llm=_make_llm(), ensemble=False, resume=True)

        # Der zweite Lauf (z. B. durch Resume auf denselben Stand) ändert nichts:
        assert pf["ziel_gewichtung_pct"] == 5.2
        assert pf["ziel_gewichtung_original"] == 10.0
