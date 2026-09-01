"""Tests für kalibrierungs-gestützte Dämpfung der Ziel-Gewichtung.

Feature: Der Portfolio-Fit empfiehlt eine Ziel-Gewichtung (ziel_gewichtung_pct).
Der Track-Record zeigt Überkonfidenz (Konfidenz 4-5, aber ~29-34% Hit-Rate).
Die empfohlene Ziel-Gewichtung wird daher deterministisch an die historische
Trefferquote der Aktion skaliert:

    faktor = clamp(hit_rate, 0.3, 1.0)
    gedämpft = round(ziel_gewichtung * faktor, 1)

Getestet werden:
- _load_action_hit_rate: korrektes Lesen, None bei fehlender/zu alter/
  ungültiger JSON, None bei fehlender Aktion
- _dampen_ziel_gewichtung: KAUFEN (0.52 → Faktor 0.52), HALTEN (0.143 →
  Faktor 0.3 Untergrenze), VERKAUFEN (0.0 → Faktor 0.3), None ohne Daten,
  Obergrenze (hit_rate 1.0 → unverändert)
- Pipeline-Wire-up: ziel_gewichtung_original + ziel_gewichtung_gedämpft
- Journal-Spalte ziel_gewichtung_original
- Report-Zeile mit gedämpft-Hinweis

Alle Tests sind OFFLINE-fähig: kein yfinance, kein Netzwerk, kein LLM.
Die echte state/calibration.json wird NIEMALS angefasst — alle Tests
schreiben in tmp_path und setzen CONCILIUM_STATE_DIR.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.journal import JOURNAL_HEADER, append_decision  # noqa: E402
from concilium.pipeline import run_pipeline  # noqa: E402
from concilium.portfolio_fit import (  # noqa: E402
    _dampen_ziel_gewichtung,
    _load_action_hit_rate,
)
from concilium.report import generate_report  # noqa: E402

# --------------------------------------------------------------------------- #
# Helfer: Kalibrierungs-JSON in tmp_path schreiben
# --------------------------------------------------------------------------- #


_HIT_RATE_UNSET = object()


def _write_calibration_json(
    tmp_path,
    *,
    hit_rates: dict[str, float] | None = None,
    erstellt_am=None,
    invalid: bool = False,
    no_erstellt_am: bool = False,
    hit_rate_raw=_HIT_RATE_UNSET,
) -> str:
    """Schreibt eine calibration.json in tmp_path/state/ und gibt den Pfad zurück.

    hit_rate_raw: Optionaler Roh-Wert für die KAUFEN-hit_rate (z. B. "abc",
    True, [0.5]), um nicht-numerische Werte zu testen. Wird der Parameter
    übergeben, überschreibt er die KAUFEN-hit_rate (auch mit None).
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    cal_path = state_dir / "calibration.json"

    if invalid:
        cal_path.write_text("{not valid json", encoding="utf-8")
        return str(cal_path)

    nach_aktion: dict[str, dict] = {}
    hr = hit_rates or {}
    for action in ("KAUFEN", "HALTEN", "VERKAUFEN"):
        rate = hr.get(action)
        if rate is None:
            continue
        nach_aktion[action] = {
            "n": 5,
            "hit_rate": rate,
            "avg_confidence": 0.6,
        }
    # Spezialfall: Roh-Wert für KAUFEN (nicht-numerisch etc.)
    if hit_rate_raw is not _HIT_RATE_UNSET:
        if "KAUFEN" not in nach_aktion:
            nach_aktion["KAUFEN"] = {"n": 5, "avg_confidence": 0.6}
        nach_aktion["KAUFEN"]["hit_rate"] = hit_rate_raw

    payload: dict = {
        "anzahl_entscheidungen": 10,
        "hit_rate_gesamt": 0.4,
        "nach_aktion": nach_aktion,
    }
    if not no_erstellt_am:
        payload["erstellt_am"] = (erstellt_am or datetime.now()).isoformat()

    cal_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(cal_path)


# --------------------------------------------------------------------------- #
# Tests: _load_action_hit_rate
# --------------------------------------------------------------------------- #


class TestLoadActionHitRate:
    """Test _load_action_hit_rate: liest korrekt, None bei Problemen."""

    def test_reads_kaufen_hit_rate(self, tmp_path):
        """Gültige JSON → Hit-Rate der Aktion als float."""
        _write_calibration_json(tmp_path, hit_rates={"KAUFEN": 0.52})
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
        ):
            assert _load_action_hit_rate("KAUFEN") == 0.52

    def test_reads_halten_hit_rate(self, tmp_path):
        """HALTEN-Hit-Rate aus nach_aktion gelesen."""
        _write_calibration_json(tmp_path, hit_rates={"HALTEN": 0.1429})
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
        ):
            assert _load_action_hit_rate("HALTEN") == 0.1429

    def test_missing_file_returns_none(self, tmp_path):
        """Keine calibration.json im State-Dir → None."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(state_dir)}
        ):
            assert _load_action_hit_rate("KAUFEN") is None

    def test_invalid_json_returns_none(self, tmp_path):
        """Kaputtes JSON → None (kein Crash)."""
        _write_calibration_json(tmp_path, invalid=True)
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
        ):
            assert _load_action_hit_rate("KAUFEN") is None

    def test_too_old_file_returns_none(self, tmp_path):
        """erstellt_am älter als 7 Tage → None."""
        old_date = datetime.now() - timedelta(days=8)
        _write_calibration_json(
            tmp_path, hit_rates={"KAUFEN": 0.52}, erstellt_am=old_date
        )
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
        ):
            assert _load_action_hit_rate("KAUFEN") is None

    def test_fresh_file_ok(self, tmp_path):
        """erstellt_am genau 6 Tage alt (unter der Grenze) → Hit-Rate ok."""
        fresh = datetime.now() - timedelta(days=6)
        _write_calibration_json(
            tmp_path, hit_rates={"KAUFEN": 0.52}, erstellt_am=fresh
        )
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
        ):
            assert _load_action_hit_rate("KAUFEN") == 0.52

    def test_no_erstellt_am_returns_none(self, tmp_path):
        """Fehlendes erstellt_am → None."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        cal_path = state_dir / "calibration.json"
        payload = {
            "anzahl_entscheidungen": 61,
            "nach_aktion": {"KAUFEN": {"n": 25, "hit_rate": 0.52}},
        }
        cal_path.write_text(json.dumps(payload), encoding="utf-8")
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(state_dir)}
        ):
            assert _load_action_hit_rate("KAUFEN") is None

    def test_missing_action_returns_none(self, tmp_path):
        """Aktion fehlt in nach_aktion → None (andere Aktionen stören nicht)."""
        _write_calibration_json(tmp_path, hit_rates={"KAUFEN": 0.52})
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
        ):
            assert _load_action_hit_rate("VERKAUFEN") is None

    def test_missing_hit_rate_returns_none(self, tmp_path):
        """Aktion vorhanden, aber ohne hit_rate → None."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        cal_path = state_dir / "calibration.json"
        payload = {
            "erstellt_am": datetime.now().isoformat(),
            "nach_aktion": {"KAUFEN": {"n": 25, "avg_confidence": 0.792}},
        }
        cal_path.write_text(json.dumps(payload), encoding="utf-8")
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(state_dir)}
        ):
            assert _load_action_hit_rate("KAUFEN") is None

    def test_non_numeric_hit_rate_returns_none(self, tmp_path):
        """hit_rate als String/Bool → None (nicht numerisch)."""
        for raw in ("0.5", True, [0.5]):
            _write_calibration_json(tmp_path, hit_rate_raw=raw)
            with patch.dict(
                "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
            ):
                assert _load_action_hit_rate("KAUFEN") is None, f"raw={raw!r}"

    def test_garbage_file_never_crashes(self, tmp_path):
        """Beliebiger Müll als Datei → None, kein Crash."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        cal_path = state_dir / "calibration.json"
        cal_path.write_text("garbage content !!!", encoding="utf-8")
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(state_dir)}
        ):
            assert _load_action_hit_rate("KAUFEN") is None

    def test_no_state_dir_env_uses_state_relative(self, tmp_path, monkeypatch):
        """Ohne CONCILIUM_STATE_DIR wird 'state/calibration.json' gelesen."""
        monkeypatch.delenv("CONCILIUM_STATE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)  # 'state' relativ zu tmp_path
        assert _load_action_hit_rate("KAUFEN") is None  # kein state/ vorhanden


# --------------------------------------------------------------------------- #
# Tests: _dampen_ziel_gewichtung
# --------------------------------------------------------------------------- #


class TestDampenZielGewichtung:
    """Test Dämpfungs-Logik: faktor = clamp(hit_rate, 0.3, 1.0)."""

    def test_kaufen_faktor_052(self, tmp_path):
        """KAUFEN hit_rate 0.52 → Faktor 0.52 (10 → 5.2)."""
        _write_calibration_json(tmp_path, hit_rates={"KAUFEN": 0.52})
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
        ):
            assert _dampen_ziel_gewichtung(10.0, "KAUFEN") == 5.2

    def test_halten_untergrenze_03(self, tmp_path):
        """HALTEN hit_rate 0.143 → Faktor 0.3 (Untergrenze, 10 → 3.0)."""
        _write_calibration_json(tmp_path, hit_rates={"HALTEN": 0.14285714285714285})
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
        ):
            assert _dampen_ziel_gewichtung(10.0, "HALTEN") == 3.0

    def test_verkaufen_null_hit_rate_untergrenze_03(self, tmp_path):
        """VERKAUFEN hit_rate 0.0 → Faktor 0.3 (nicht 0, 10 → 3.0)."""
        _write_calibration_json(tmp_path, hit_rates={"VERKAUFEN": 0.0})
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
        ):
            assert _dampen_ziel_gewichtung(10.0, "VERKAUFEN") == 3.0

    def test_no_calibration_returns_none(self, tmp_path):
        """Keine Kalibrierungsdaten → None (Signal: nichts gedämpft)."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(state_dir)}
        ):
            assert _dampen_ziel_gewichtung(10.0, "KAUFEN") is None

    def test_oberegrenze_hit_rate_1(self, tmp_path):
        """hit_rate 1.0 → Faktor 1.0, Gewichtung unverändert."""
        _write_calibration_json(tmp_path, hit_rates={"KAUFEN": 1.0})
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
        ):
            assert _dampen_ziel_gewichtung(7.2, "KAUFEN") == 7.2

    def test_too_old_calibration_returns_none(self, tmp_path):
        """Zu alte Kalibrierung → None (nichts gedämpft)."""
        old_date = datetime.now() - timedelta(days=10)
        _write_calibration_json(
            tmp_path, hit_rates={"KAUFEN": 0.52}, erstellt_am=old_date
        )
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
        ):
            assert _dampen_ziel_gewichtung(10.0, "KAUFEN") is None

    def test_rundung_auf_1_dezimalstelle(self, tmp_path):
        """Ergebnis wird auf 1 Dezimalstelle gerundet (7.2*0.52=3.744 → 3.7)."""
        _write_calibration_json(tmp_path, hit_rates={"KAUFEN": 0.52})
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
        ):
            assert _dampen_ziel_gewichtung(7.2, "KAUFEN") == 3.7

    def test_grenze_genau_03_bleibt_03(self, tmp_path):
        """hit_rate exakt 0.3 → Faktor 0.3 (Untergrenze inklusiv)."""
        _write_calibration_json(tmp_path, hit_rates={"KAUFEN": 0.3})
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
        ):
            assert _dampen_ziel_gewichtung(10.0, "KAUFEN") == 3.0

    def test_direkt_unter_grenze_wird_angehoben(self, tmp_path):
        """hit_rate 0.1 → Faktor 0.3 (Untergrenze greift, 10 → 3.0)."""
        _write_calibration_json(tmp_path, hit_rates={"HALTEN": 0.1})
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(tmp_path / "state")}
        ):
            assert _dampen_ziel_gewichtung(10.0, "HALTEN") == 3.0

    def test_real_calibration_values_kaufen(self, tmp_path):
        """Echte Werte vom 01.09.: KAUFEN 0.52, 7.2% → 3.7%."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "erstellt_am": "2026-09-01T09:00:50.667675",
            "anzahl_entscheidungen": 61,
            "hit_rate_gesamt": 0.29508196721311475,
            "nach_aktion": {
                "KAUFEN": {"n": 25, "hit_rate": 0.52, "avg_confidence": 0.792},
                "HALTEN": {
                    "n": 35,
                    "hit_rate": 0.14285714285714285,
                    "avg_confidence": 0.6285714285714286,
                },
                "VERKAUFEN": {"n": 1, "hit_rate": 0.0, "avg_confidence": 1.0},
            },
        }
        # erstellt_am frisch schreiben, damit der Alters-Check besteht
        payload["erstellt_am"] = datetime.now().isoformat()
        (state_dir / "calibration.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        with patch.dict(
            "os.environ", {"CONCILIUM_STATE_DIR": str(state_dir)}
        ):
            assert _dampen_ziel_gewichtung(7.2, "KAUFEN") == 3.7
            assert _dampen_ziel_gewichtung(7.2, "HALTEN") == 2.2
            assert _dampen_ziel_gewichtung(7.2, "VERKAUFEN") == 2.2


# --------------------------------------------------------------------------- #
# Tests: Pipeline-Wire-up (Schritt 5b')
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


def _run_pipeline_with_mocks(
    trade: dict,
    portfolio_fit: dict | None,
    *,
    ensemble: bool = False,
) -> dict:
    """Führt run_pipeline mit komplett gemockten Agenten aus (offline).

    CONCILIUM_STATE_DIR muss vom Test gesetzt sein (state_dir-Fixture),
    damit die Dämpfung nicht die echte state/calibration.json liest.
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
        "trade_revision": MagicMock(return_value=trade),
        "portfolio_manager": MagicMock(
            return_value={"entscheidung": "GENEHMIGT", "confidence": 4}
        ),
        "build_feedback_context": MagicMock(return_value=""),
        "build_reflection_context": MagicMock(return_value=""),
        # _dampen_ziel_gewichtung wird NICHT gepatcht — läuft real gegen das
        # tmp-CONCILIUM_STATE_DIR des Tests.
    }
    with patch.multiple("concilium.pipeline", **patches), patch(
        "concilium.journal.append_decision"
    ):
        return run_pipeline(
            "TEST", llm=MagicMock(), ensemble=ensemble, resume=False
        )


class TestPipelineDampenWireUp:
    """Test das Wire-up nach Schritt 5b: Original + gedämpft-Feld."""

    @pytest.fixture
    def state_dir(self, tmp_path, monkeypatch):
        """Isoliertes state-Verzeichnis (CONCILIUM_STATE_DIR auf tmp_path)."""
        d = tmp_path / "state"
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(d))
        return str(d)

    def test_kaufen_gedämpft_original_erhalten(self, tmp_path, state_dir):
        """KAUFEN mit Kalibrierung: pct gedämpft, Original + Flag gesetzt."""
        _write_calibration_json(
            tmp_path,
            hit_rates={"KAUFEN": 0.52, "HALTEN": 0.1429, "VERKAUFEN": 0.0},
        )
        pf = {"portfolio_fit_score": 2, "ziel_gewichtung_pct": 10.0}
        result = _run_pipeline_with_mocks(_make_trade("KAUFEN"), pf)
        pf_result = result["portfolio_fit"]
        assert pf_result["ziel_gewichtung_original"] == 10.0
        assert pf_result["ziel_gewichtung_pct"] == 5.2
        assert pf_result["ziel_gewichtung_gedämpft"] is True

    def test_ohne_kalibrierung_nichts_gedämpft(self, state_dir):
        """Ohne calibration.json: Original gesetzt, aber kein Flag/Wert geändert."""
        pf = {"portfolio_fit_score": 2, "ziel_gewichtung_pct": 7.2}
        result = _run_pipeline_with_mocks(_make_trade("KAUFEN"), pf)
        pf_result = result["portfolio_fit"]
        assert pf_result["ziel_gewichtung_pct"] == 7.2
        assert "ziel_gewichtung_gedämpft" not in pf_result

    def test_ohne_aktion_kein_crash(self, state_dir):
        """Trade ohne aktion → kein Crash, portfolio_fit unverändert."""
        pf = {"portfolio_fit_score": 2, "ziel_gewichtung_pct": 7.2}
        trade = _make_trade("KAUFEN")
        del trade["aktion"]
        result = _run_pipeline_with_mocks(trade, pf)
        pf_result = result["portfolio_fit"]
        assert pf_result["ziel_gewichtung_pct"] == 7.2
        assert "ziel_gewichtung_gedämpft" not in pf_result

    def test_portfolio_fit_none_kein_crash(self, state_dir):
        """portfolio_fit None (Agent fehlgeschlagen) → kein Crash."""
        result = _run_pipeline_with_mocks(_make_trade("KAUFEN"), None)
        assert result["portfolio_fit"] is None

    def test_nicht_numerische_gewichtung_unverändert(self, tmp_path, state_dir):
        """ziel_gewichtung_pct als String → keine Dämpfung, kein Crash."""
        _write_calibration_json(tmp_path, hit_rates={"KAUFEN": 0.52})
        pf = {"portfolio_fit_score": 2, "ziel_gewichtung_pct": "hoch"}
        result = _run_pipeline_with_mocks(_make_trade("KAUFEN"), pf)
        pf_result = result["portfolio_fit"]
        assert pf_result["ziel_gewichtung_pct"] == "hoch"
        # Original wird trotzdem gesichert
        assert pf_result["ziel_gewichtung_original"] == "hoch"


# --------------------------------------------------------------------------- #
# Tests: Report-Anzeige (gedämpft-Hinweis)
# --------------------------------------------------------------------------- #


def _full_result(portfolio_fit: dict) -> dict:
    """Baut das vollständige result-dict für generate_report (offline)."""
    return {
        "ticker": "AAPL",
        "no_llm": False,
        "data": {
            "fundamentals": {"name": "Apple", "sector": "Tech"},
            "technicals": {"current_price": 150},
            "sentiment": {},
            "news": [],
        },
        "analysts": {
            "fundamental": {"stimmung": "bullish", "score": 4,
                            "zusammenfassung": "Gut", "_raw": ""},
            "technical": {"stimmung": "bullish", "score": 4,
                          "zusammenfassung": "Gut", "_raw": ""},
            "sentiment": {"stimmung": "neutral", "score": 3,
                          "zusammenfassung": "Ok", "_raw": ""},
        },
        "debate": {
            "bull": {"_raw": "Bull"},
            "bear": {"_raw": "Bear"},
            "bull_confidence": 5,
            "bear_confidence": 4,
        },
        "trade": {
            "aktion": "KAUFEN",
            "rating": "KAUFEN",
            "zielkurs": 340,
            "stop_loss": 285,
            "positionsanteil": 7,
            "begründung": "Test",
            "zeithorizont": "Mittelfristig",
        },
        "risk": {
            "risiko_score": 3,
            "empfehlung": "GENEHMIGT",
            "auflagen": "keine",
        },
        "portfolio_fit": portfolio_fit,
        "final": {
            "entscheidung": "GENEHMIGT",
            "confidence": 4,
            "begründung": "Solide Fundamentals.",
        },
    }


class TestReportZielGewichtungGedämpft:
    """Test die Ziel-Gewichtung-Zeile im Portfolio-Fit-Report-Abschnitt."""

    def test_gedämpft_zeigt_original(self):
        """Bei gedämpft=True: Original in der Zeile angezeigt."""
        result = _full_result({
            "portfolio_fit_score": 3,
            "ziel_gewichtung_pct": 3.6,
            "ziel_gewichtung_original": 7.2,
            "ziel_gewichtung_gedämpft": True,
        })
        report = generate_report(result)
        assert (
            "**Ziel-Gewichtung:** 3.6 % des Portfolios "
            "(nach Kalibrierung gedämpft, original 7.2)" in report
        )

    def test_nicht_gedämpft_ohne_hinweis(self):
        """Ohne gedämpft-Flag: keine Zusatzinfo in der Zeile."""
        result = _full_result({
            "portfolio_fit_score": 3,
            "ziel_gewichtung_pct": 7.2,
        })
        report = generate_report(result)
        assert "**Ziel-Gewichtung:** 7.2 % des Portfolios" in report
        assert "nach Kalibrierung gedämpft" not in report


# --------------------------------------------------------------------------- #
# Tests: Journal-Spalte ziel_gewichtung_original
# --------------------------------------------------------------------------- #


class TestJournalZielGewichtungOriginal:
    """Test die neue Journal-Spalte ziel_gewichtung_original."""

    def test_header_contains_new_column(self):
        """JOURNAL_HEADER enthält ziel_gewichtung_original neben ziel_gewichtung_pct."""
        idx_pct = JOURNAL_HEADER.index("ziel_gewichtung_pct")
        idx_orig = JOURNAL_HEADER.index("ziel_gewichtung_original")
        assert idx_orig == idx_pct + 1  # NEBEN ziel_gewichtung_pct

    def test_original_wird_geschrieben(self, tmp_path):
        """portfolio_fit.ziel_gewichtung_original landet in der CSV."""
        result = {
            "ticker": "AAPL",
            "trade": {"aktion": "KAUFEN", "zielkurs": 340.0},
            "final": {"entscheidung": "GENEHMIGT", "confidence": 4},
            "portfolio_fit": {
                "portfolio_fit_score": 3,
                "ziel_gewichtung_pct": 3.6,
                "ziel_gewichtung_original": 7.2,
                "ziel_gewichtung_gedämpft": True,
            },
        }
        journal_file = str(tmp_path / "journal" / "decisions.csv")
        append_decision(result, journal_file=journal_file)

        import csv as csv_mod

        with open(journal_file, encoding="utf-8") as fh:
            reader = csv_mod.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["ziel_gewichtung_pct"] == "3.6"
        assert rows[0]["ziel_gewichtung_original"] == "7.2"

    def test_ohne_original_leer(self, tmp_path):
        """Fehlendes ziel_gewichtung_original → leere Spalte (kein Crash)."""
        result = {
            "ticker": "MSFT",
            "trade": {"aktion": "HALTEN"},
            "final": {"entscheidung": "ABGELEHNT", "confidence": 3},
            "portfolio_fit": {"portfolio_fit_score": 2, "ziel_gewichtung_pct": 1.5},
        }
        journal_file = str(tmp_path / "journal" / "decisions.csv")
        append_decision(result, journal_file=journal_file)

        import csv as csv_mod

        with open(journal_file, encoding="utf-8") as fh:
            reader = csv_mod.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["ziel_gewichtung_pct"] == "1.5"
        assert rows[0]["ziel_gewichtung_original"] == ""

    def test_header_migration_altes_journal(self, tmp_path):
        """Bestehendes Journal ohne neue Spalte wird migriert (Header ergänzt)."""
        import csv as csv_mod

        journal_dir = tmp_path / "journal"
        journal_dir.mkdir(parents=True, exist_ok=True)
        journal_file = str(journal_dir / "decisions.csv")
        # Alte CSV ohne ziel_gewichtung_original-Spalte schreiben
        old_fields = [f for f in JOURNAL_HEADER if f != "ziel_gewichtung_original"]
        with open(journal_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv_mod.DictWriter(fh, fieldnames=old_fields)
            writer.writeheader()
            writer.writerow({"timestamp": "2026-08-01 10:00:00", "ticker": "OLD.DE"})

        result = {
            "ticker": "AAPL",
            "trade": {"aktion": "KAUFEN"},
            "final": {"entscheidung": "GENEHMIGT", "confidence": 4},
            "portfolio_fit": {
                "ziel_gewichtung_pct": 5.2,
                "ziel_gewichtung_original": 10.0,
            },
        }
        append_decision(result, journal_file=journal_file)

        with open(journal_file, encoding="utf-8") as fh:
            reader = csv_mod.DictReader(fh)
            rows = list(reader)
        assert reader.fieldnames[-1] == "ziel_gewichtung_original"
        assert len(rows) == 2
        # Der neue Eintrag trägt die Werte
        assert rows[-1]["ticker"] == "AAPL"
        assert rows[-1]["ziel_gewichtung_original"] == "10.0"
        # Der alte Eintrag bleibt erhalten (neue Spalte leer)
        assert rows[0]["ticker"] == "OLD.DE"
        assert rows[0]["ziel_gewichtung_original"] == ""
