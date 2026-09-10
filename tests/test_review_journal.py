"""Tests für Phase 1: dedizierter VERKAUFEN-Pfad mit eigenem Journal.

Abgedeckte Bereiche:
(a) append_review_decision (journal/reviews.csv): REVIEW_HEADER, Review-
    Metadaten (verkauf_empfehlung/depot_pct/name), Idempotenz-Guard,
    Robustheit (crasht nie), Header-Migration, getrennte Dateien.
(b) trader_exit: nutzt SYSTEM_TRADER_EXIT (nicht SYSTEM_TRADER), Rückgabe-
    Schema wie trader(), _ensure_ziel_stop-Fallback, Dämpfung analog.
(c) ensemble_trader(exit_mode=True): jeder Run nutzt trader_exit.
(d) run_pipeline(exit_mode=True): Trader-Schritt nutzt trader_exit bzw.
    ensemble_trader(..., exit_mode=True); Default unverändert.
(e) run_review: übergibt exit_mode=True + journal=False an run_pipeline und
    schreibt NACH jedem erfolgreichen Lauf journal/reviews.csv (nicht
    journal/decisions.csv).
(f) CLI --review: stderr-Hinweis auf journal/reviews.csv (nur bei ≥1 Position).

Alle Tests sind OFFLINE-fähig: kein yfinance, kein Netzwerk, kein echtes LLM.
Mock-Muster analog tests/test_ensemble.py (_FakeLLM, thread-sicher,
temperatur-keyed) und tests/test_review.py (g gemockte Pipeline-Schritte).
Konventionen wie in test_journal_hygiene.py: Journal via tmp_path +
monkeypatch.chdir, Zeit via monkeypatch von concilium.journal.datetime
gefroren (deterministisch).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from concilium.agents import (  # noqa: E402
    SYSTEM_TRADER,
    SYSTEM_TRADER_EXIT,
    ensemble_trader,
    trader_exit,
)
from concilium.cli import main  # noqa: E402
from concilium.journal import (  # noqa: E402
    JOURNAL_HEADER,
    REVIEW_HEADER,
    append_decision,
    append_review_decision,
)
from concilium.review import run_review  # noqa: E402

# ---------------------------------------------------------------------------
# Helper — gefrorene Zeit (analog test_journal_hygiene.py)
# ---------------------------------------------------------------------------

class _FrozenDateTime(datetime):
    """datetime-Ersatz mit festem now() — simuliert Resume in derselben Sekunde."""

    @classmethod
    def now(cls, tz=None):  # noqa: ARG003 — Signatur wie datetime.now
        return cls(2026, 9, 5, 12, 0, 0)


class _FrozenDateTimeLater(datetime):
    """Zweiter fester Zeitstempel (5s später) — für "anderer Timestamp"-Tests."""

    @classmethod
    def now(cls, tz=None):  # noqa: ARG003 — Signatur wie datetime.now
        return cls(2026, 9, 5, 12, 0, 5)


def _make_review_result(
    ticker: str,
    *,
    aktion: str = "VERKAUFEN",
    rating: str | None = None,
    zielkurs: float | None = 52.3,
    stop_loss: float | None = 62.7,
    positionsanteil: int | None = 0,
    entscheidung: str = "GENEHMIGT",
    confidence: int = 4,
    ensemble_confidence: float | None = 0.67,
    portfolio_fit_score: float | None = 0.5,
) -> dict:
    """Minimal-result für append_review_decision (Pipeline-Struktur)."""
    trade: dict = {"aktion": aktion}
    if rating is not None:
        trade["rating"] = rating
    trade["zielkurs"] = zielkurs
    trade["stop_loss"] = stop_loss
    trade["positionsanteil"] = positionsanteil
    if ensemble_confidence is not None:
        trade["_ensemble"] = {"ensemble_confidence": ensemble_confidence}
    result: dict = {
        "ticker": ticker,
        "trade": trade,
        "final": {"entscheidung": entscheidung, "confidence": confidence},
    }
    if portfolio_fit_score is not None:
        result["portfolio_fit"] = {"portfolio_fit_score": portfolio_fit_score}
    return result


def _read_rows(path: str) -> list[dict]:
    """Liest alle Datenzeilen einer CSV (ohne Header) als dicts."""
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_header(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return next(csv.reader(fh))


# ---------------------------------------------------------------------------
# Helper — Mock-LLM (analog tests/test_ensemble.py, thread-sicher)
# ---------------------------------------------------------------------------


def _trader_json(
    aktion: str = "KAUFEN",
    zielkurs: float | None = None,
    stop_loss: float | None = None,
    positionsanteil: int = 5,
) -> str:
    return json.dumps(
        {
            "rolle": "Trader",
            "aktion": aktion,
            "zielkurs": zielkurs,
            "stop_loss": stop_loss,
            "positionsanteil": positionsanteil,
            "begründung": "Test-Begründung",
            "zeithorizont": "Mittelfristig",
        }
    )


class _FakeLLM:
    """Thread-sicherer Mock-LLM, der Antworten nach Temperatur dispatcht."""

    _DEFAULT_TEMP_KEYS = [0.3, 0.5, 0.7]

    def __init__(self, responses: list[str], temp_keys: list[float] | None = None):
        keys = temp_keys if temp_keys is not None else self._DEFAULT_TEMP_KEYS
        self._temp_map: dict[float, str] = {}
        for i, resp in enumerate(responses):
            k = round(keys[i % len(keys)], 2)
            self._temp_map[k] = resp
        self.temperatures_seen: list[float] = []
        self._lock = threading.Lock()

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        **kwargs,
    ) -> str | object:
        with self._lock:
            self.temperatures_seen.append(temperature)
        key = round(temperature, 2)
        if key in self._temp_map:
            text = self._temp_map[key]
        else:
            text = list(self._temp_map.values())[0] if self._temp_map else ""
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult

            return StructuredChatResult(text=text, response_format_used=True)
        return text


class _SystemCapturingLLM(_FakeLLM):
    """_FakeLLM, der zusätzlich System-/User-Prompts aufzeichnet (thread-sicher)."""

    def __init__(self, responses: list[str], temp_keys: list[float] | None = None):
        super().__init__(responses, temp_keys=temp_keys)
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        **kwargs,
    ) -> str | object:
        with self._lock:
            self.system_prompts.append(
                str(messages[0].get("content", "")) if messages else ""
            )
            self.user_prompts.append(
                str(messages[1].get("content", "")) if len(messages) > 1 else ""
            )
        return super().chat(messages, temperature=temperature, **kwargs)


_ANALYSTS = {
    "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
    "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Neutral", "_raw": ""},
    "technicals": {"current_price": 57.0},
}

_DEBATE = {
    "bull": {"_raw": "Bull-Argument"},
    "bear": {"_raw": "Bear-Argument"},
}


# ---------------------------------------------------------------------------
# (a) append_review_decision — Journal journal/reviews.csv
# ---------------------------------------------------------------------------


class TestAppendReviewDecision:
    """append_review_decision: Header, Metadaten, Idempotenz, Robustheit."""

    def test_writes_reviews_csv_with_review_header(self, tmp_path, monkeypatch):
        """Default: journal/reviews.csv wird angelegt, Header == REVIEW_HEADER."""
        monkeypatch.chdir(tmp_path)

        append_review_decision(_make_review_result("AAPL"))

        reviews = tmp_path / "journal" / "reviews.csv"
        assert reviews.exists()
        assert _read_header(str(reviews)) == REVIEW_HEADER

    def test_review_header_has_required_review_fields(self):
        """REVIEW_HEADER enthält die Review-spezifischen Pflichtfelder."""
        for field in (
            "timestamp",
            "ticker",
            "action",
            "rating",
            "target",
            "stop",
            "position_pct",
            "final_decision",
            "confidence",
            "ensemble_confidence",
            "portfolio_fit_score",
            "verkauf_empfehlung",
            "depot_pct",
            "name",
            "reflection_status",
            "resolved_at",
            "realised_return_pct",
            "alpha_pct",
            "lesson",
        ):
            assert field in REVIEW_HEADER, f"Fehlendes Feld: {field}"
        # Und die Neukauf-spezifischen Spalten gehören NICHT ins Review-Journal
        assert "einstiegs_level" not in REVIEW_HEADER
        assert "ziel_gewichtung_pct" not in REVIEW_HEADER

    def test_review_metadata_written(self, tmp_path, monkeypatch):
        """verkauf_empfehlung/depot_pct/name + Entscheidungsfelder landen in der Zeile."""
        monkeypatch.chdir(tmp_path)

        append_review_decision(
            _make_review_result("AAPL"),
            verkauf_empfehlung=True,
            depot_pct=5.2,
            name="Apple Inc.",
        )

        rows = _read_rows(str(tmp_path / "journal" / "reviews.csv"))
        assert len(rows) == 1
        row = rows[0]
        assert row["ticker"] == "AAPL"
        assert row["action"] == "VERKAUFEN"
        assert row["target"] == "52.3"
        assert row["stop"] == "62.7"
        assert row["final_decision"] == "GENEHMIGT"
        assert row["confidence"] == "4"
        assert row["ensemble_confidence"] == "0.67"
        assert row["portfolio_fit_score"] == "0.5"
        assert row["verkauf_empfehlung"] == "1"
        assert row["depot_pct"] == "5.2"
        assert row["name"] == "Apple Inc."
        # C6-Analogie: neue Zeilen starten pending (look-ahead-frei)
        assert row["reflection_status"] == "pending"
        assert row["resolved_at"] == ""
        assert row["realised_return_pct"] == ""
        assert row["alpha_pct"] == ""
        assert row["lesson"] == ""

    def test_verkauf_empfehlung_false_and_defaults(self, tmp_path, monkeypatch):
        """verkauf_empfehlung=False → "0"; ohne depot_pct/name → leere Werte."""
        monkeypatch.chdir(tmp_path)

        append_review_decision(
            _make_review_result("BAS.DE", aktion="HALTEN", entscheidung="GENEHMIGT"),
            verkauf_empfehlung=False,
        )

        rows = _read_rows(str(tmp_path / "journal" / "reviews.csv"))
        assert len(rows) == 1
        row = rows[0]
        assert row["verkauf_empfehlung"] == "0"
        assert row["depot_pct"] == ""
        assert row["name"] == ""
        assert row["action"] == "HALTEN"

    def test_idempotenz_same_ticker_same_timestamp(self, tmp_path, monkeypatch):
        """Doppel-Append (ticker+timestamp identisch) → nur 1 Datenzeile."""
        monkeypatch.chdir(tmp_path)
        with patch("concilium.journal.datetime", _FrozenDateTime):
            append_review_decision(_make_review_result("AAPL"))
            append_review_decision(_make_review_result("AAPL"))

        rows = _read_rows(str(tmp_path / "journal" / "reviews.csv"))
        assert len(rows) == 1

    def test_idempotenz_different_ticker_or_timestamp(self, tmp_path, monkeypatch):
        """Anderer Ticker oder anderer Timestamp → Zeile wird normal geschrieben."""
        monkeypatch.chdir(tmp_path)
        with patch("concilium.journal.datetime", _FrozenDateTime):
            append_review_decision(_make_review_result("AAPL"))
            append_review_decision(_make_review_result("BAS.DE"))
        with patch("concilium.journal.datetime", _FrozenDateTimeLater):
            append_review_decision(_make_review_result("AAPL"))

        rows = _read_rows(str(tmp_path / "journal" / "reviews.csv"))
        assert len(rows) == 3
        assert [r["ticker"] for r in rows] == ["AAPL", "BAS.DE", "AAPL"]

    def test_never_crashes_on_garbage_result(self, tmp_path, monkeypatch):
        """Müll-Result (None, Strings statt dicts) → kein Crash, keine Zeile."""
        monkeypatch.chdir(tmp_path)

        # Kein Crash (die Funktion schluckt alles — best effort)
        append_review_decision(None)  # type: ignore[arg-type]
        append_review_decision({"ticker": "AAPL", "trade": "kagut"})  # type: ignore[arg-type]
        append_review_decision({"ticker": None, "trade": {"aktion": 123}})  # type: ignore[arg-type]

        # Ungültige depot_pct-Typen sind ebenfalls kein Problem
        append_review_decision(
            _make_review_result("AAPL"),
            depot_pct=object(),  # type: ignore[arg-type]
            name=None,  # type: ignore[arg-type]
        )

        # Die gültigen Calls davor/danach: Datei existiert mit sauberem Header
        assert (tmp_path / "journal" / "reviews.csv").exists()
        assert _read_header(
            str(tmp_path / "journal" / "reviews.csv")
        ) == REVIEW_HEADER

    def test_journal_dir_creates_missing_folders(self, tmp_path):
        """Explizites journal_dir in nicht existierenden Ordnern → wird angelegt."""
        target_dir = str(tmp_path / "tief" / "verschachtelt")

        append_review_decision(
            _make_review_result("AAPL"), journal_dir=target_dir
        )

        assert os.path.isfile(os.path.join(target_dir, "reviews.csv"))

    def test_explicit_journal_file_creates_parent(self, tmp_path):
        """Explizites journal_file → übergeordneter Ordner wird angelegt."""
        jf = str(tmp_path / "irgendwo" / "meine_reviews.csv")

        append_review_decision(_make_review_result("AAPL"), journal_file=jf)

        assert os.path.isfile(jf)
        assert _read_header(jf) == REVIEW_HEADER

    def test_header_migration_from_legacy_file(self, tmp_path, monkeypatch):
        """Bestehende Legacy-Datei (alter/kürzerer Header) → Migration auf REVIEW_HEADER.

        Alte Zeilen bleiben erhalten (fehlende Spalten leer), neue Zeile
        vollständig angehängt.
        """
        monkeypatch.chdir(tmp_path)
        legacy = tmp_path / "journal" / "reviews.csv"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        with open(legacy, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp", "ticker", "action"])
            writer.writerow(["2026-08-01 10:00:00", "OLD.DE", "HALTEN"])

        append_review_decision(_make_review_result("AAPL"), verkauf_empfehlung=True)

        assert _read_header(str(legacy)) == REVIEW_HEADER
        rows = _read_rows(str(legacy))
        assert len(rows) == 2
        # Legacy-Zeile erhalten, neue Spalten leer aufgefüllt
        assert rows[0]["ticker"] == "OLD.DE"
        assert rows[0]["verkauf_empfehlung"] == ""
        # Neue Zeile vollständig
        assert rows[1]["ticker"] == "AAPL"
        assert rows[1]["verkauf_empfehlung"] == "1"

    def test_reviews_and_decisions_are_separate_files(self, tmp_path, monkeypatch):
        """append_review_decision schreibt reviews.csv, append_decision decisions.csv — getrennt."""
        monkeypatch.chdir(tmp_path)

        append_review_decision(_make_review_result("AAPL"), verkauf_empfehlung=True)
        append_decision(_make_review_result("AAPL"))

        reviews = tmp_path / "journal" / "reviews.csv"
        decisions = tmp_path / "journal" / "decisions.csv"
        assert reviews.exists()
        assert decisions.exists()
        assert _read_header(str(reviews)) == REVIEW_HEADER
        assert _read_header(str(decisions)) == JOURNAL_HEADER
        assert len(_read_rows(str(reviews))) == 1
        assert len(_read_rows(str(decisions))) == 1


# ---------------------------------------------------------------------------
# (b) trader_exit — Exit-Prompt-Pfad
# ---------------------------------------------------------------------------


class TestTraderExit:
    """trader_exit nutzt SYSTEM_TRADER_EXIT, Schema + Nachbearbeitung analog trader()."""

    def test_uses_exit_system_prompt(self):
        """System-Prompt ist exakt SYSTEM_TRADER_EXIT — nicht SYSTEM_TRADER."""
        llm = _SystemCapturingLLM([_trader_json("VERKAUFEN", zielkurs=51.3, stop_loss=62.7)])

        result = trader_exit(_ANALYSTS, _DEBATE, llm)

        assert result["aktion"] == "VERKAUFEN"
        assert llm.system_prompts == [SYSTEM_TRADER_EXIT]
        assert SYSTEM_TRADER_EXIT != SYSTEM_TRADER
        # User-Prompt enthält Analysten + Debatte (analog trader())
        user = llm.user_prompts[0]
        assert "Analysten-Einschätzungen" in user
        assert "Bull-Argumentation" in user
        assert "Bear-Argumentation" in user

    def test_returns_same_schema_as_trader(self):
        """Rückgabe-Schema identisch zu trader() (TRADE_SCHEMA-Felder + _raw)."""
        llm = _FakeLLM([_trader_json("VERKAUFEN", zielkurs=51.3, stop_loss=62.7)])

        result = trader_exit(_ANALYSTS, _DEBATE, llm)

        for key in (
            "rolle",
            "aktion",
            "rating",
            "zielkurs",
            "stop_loss",
            "positionsanteil",
            "begründung",
            "zeithorizont",
            "_raw",
        ):
            assert key in result, f"Fehlendes Feld: {key}"
        assert result["rating"] == "VERKAUFEN"
        assert result["aktion"] == "VERKAUFEN"

    def test_rating_normalization_5_to_3_stufig(self):
        """STARK VERKAUFEN (Rating) → Aktion VERKAUFEN (3-stufig normalisiert)."""
        llm = _FakeLLM([_trader_json("STARK VERKAUFEN")])

        result = trader_exit(_ANALYSTS, _DEBATE, llm)

        assert result["rating"] == "STARK VERKAUFEN"
        assert result["aktion"] == "VERKAUFEN"

    def test_ensure_ziel_stop_fills_missing_values(self):
        """VERKAUFEN ohne Ziel-/Stop → deterministischer Fallback um current_price."""
        llm = _FakeLLM([_trader_json("VERKAUFEN", zielkurs=None, stop_loss=None)])

        result = trader_exit(_ANALYSTS, _DEBATE, llm)

        ziel = result.get("zielkurs")
        stop = result.get("stop_loss")
        assert ziel is not None and float(ziel) < 57.0
        assert stop is not None and float(stop) > 57.0

    def test_plausible_values_not_overwritten(self):
        """Plausible Werte bleiben unangetastet (_ensure_ziel_stop überschreibt nicht)."""
        llm = _FakeLLM([_trader_json("VERKAUFEN", zielkurs=45.0, stop_loss=70.0)])

        result = trader_exit(_ANALYSTS, _DEBATE, llm)

        assert float(result["zielkurs"]) == 45.0
        assert float(result["stop_loss"]) == 70.0

    def test_no_crash_without_current_price(self):
        """Analysten ohne current_price → kein Crash, Werte bleiben wie geliefert."""
        llm = _FakeLLM([_trader_json("VERKAUFEN")])
        analysts_no_price = {
            "fundamental": {"stimmung": "bearish", "score": 2, "_raw": ""},
            "technical": {"stimmung": "bearish", "score": 2, "_raw": ""},
            "sentiment": {"stimmung": "neutral", "score": 3, "_raw": ""},
        }

        result = trader_exit(analysts_no_price, _DEBATE, llm)

        assert result["aktion"] == "VERKAUFEN"

    def test_dampen_stark_active(self, tmp_path, monkeypatch):
        """Überkonfidente Kalibrierung: STARK VERKAUFEN → VERKAUFEN gedämpft."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))
        cal_dir = tmp_path / "state"
        cal_dir.mkdir(parents=True, exist_ok=True)
        (cal_dir / "calibration.json").write_text(
            json.dumps(
                {
                    "erstellt_am": datetime.now().isoformat(),
                    "anzahl_entscheidungen": 10,
                    "hit_rate_gesamt": 0.4,
                    "nach_aktion": {
                        "VERKAUFEN": {
                            "n": 5,
                            "hit_rate": 0.2,
                            "avg_confidence": 0.8,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        llm = _FakeLLM([_trader_json("STARK VERKAUFEN")])

        result = trader_exit(_ANALYSTS, _DEBATE, llm)

        assert result["rating_original"] == "STARK VERKAUFEN"
        assert result["rating"] == "VERKAUFEN"
        assert result["aktion"] == "VERKAUFEN"
        assert result["rating_gedämpft"] is True

    def test_dampen_stark_inactive(self):
        """Ohne Kalibrierung: STARK VERKAUFEN bleibt stehen (nur Aktion normalisiert)."""
        llm = _FakeLLM([_trader_json("STARK VERKAUFEN")])

        result = trader_exit(_ANALYSTS, _DEBATE, llm)

        assert result["rating"] == "STARK VERKAUFEN"
        assert result["aktion"] == "VERKAUFEN"
        assert result["rating_gedämpft"] is False

    def test_contexts_appended_to_user_prompt(self):
        """feedback_context/reflection_context landen im User-Prompt (analog trader())."""
        llm = _SystemCapturingLLM([_trader_json("HALTEN")])

        trader_exit(
            _ANALYSTS,
            _DEBATE,
            llm,
            feedback_context="TRACK-RECORD-BLOCK",
            reflection_context="REFLEXIONS-BLOCK",
        )

        user = llm.user_prompts[0]
        assert "TRACK-RECORD-BLOCK" in user
        assert "REFLEXIONS-BLOCK" in user


# ---------------------------------------------------------------------------
# (c) ensemble_trader(exit_mode=True)
# ---------------------------------------------------------------------------


class TestEnsembleTraderExitMode:
    """exit_mode=True → jeder Ensemble-Run nutzt den Exit-Prompt."""

    def test_exit_mode_all_runs_use_exit_prompt(self):
        llm = _SystemCapturingLLM(
            [
                _trader_json("VERKAUFEN", zielkurs=51.3, stop_loss=62.7),
                _trader_json("VERKAUFEN", zielkurs=52.0, stop_loss=63.0),
                _trader_json("HALTEN"),
            ]
        )

        result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3, exit_mode=True)

        assert result["_ensemble"]["mehrheits_aktion"] == "VERKAUFEN"
        assert result["_ensemble"]["runs"] == 3
        assert llm.system_prompts == [SYSTEM_TRADER_EXIT] * 3

    def test_default_mode_uses_normal_prompt(self):
        """Ohne exit_mode: weiterhin SYSTEM_TRADER (Rückwärtskompatibilität)."""
        llm = _SystemCapturingLLM(
            [
                _trader_json("KAUFEN", zielkurs=65.0, stop_loss=50.0),
                _trader_json("KAUFEN", zielkurs=62.0, stop_loss=52.0),
                _trader_json("HALTEN"),
            ]
        )

        result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        assert result["_ensemble"]["mehrheits_aktion"] == "KAUFEN"
        assert llm.system_prompts == [SYSTEM_TRADER] * 3


# ---------------------------------------------------------------------------
# (d) run_pipeline(exit_mode=True)
# ---------------------------------------------------------------------------


def _pipeline_patches(mock_data=None, mock_analysts=None, mock_trade=None):
    """Gemeinsame Mock-Map für patch.multiple('concilium.pipeline', ...)."""
    if mock_data is None:
        mock_data = {
            "ticker": "AAPL",
            "fundamentals": {"name": "Apple", "sector": "Tech"},
            "technicals": {"current_price": 150},
            "sentiment": {},
            "news": [],
        }
    if mock_analysts is None:
        mock_analysts = {
            "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
            "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
            "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
        }
    if mock_trade is None:
        mock_trade = {
            "rolle": "Trader",
            "aktion": "VERKAUFEN",
            "rating": "VERKAUFEN",
            "zielkurs": 140,
            "stop_loss": 165,
            "positionsanteil": 0,
            "_raw": "",
        }
    return {
        "collect_ticker_data": MagicMock(return_value=mock_data),
        "analyst_team": MagicMock(return_value=mock_analysts),
        "debate": MagicMock(
            return_value={"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}}
        ),
        "trader": MagicMock(return_value=dict(mock_trade)),
        "trader_exit": MagicMock(return_value=dict(mock_trade)),
        "ensemble_trader": MagicMock(return_value=dict(mock_trade)),
        "risk_manager": MagicMock(
            return_value={"risiko_score": 3, "empfehlung": "GENEHMIGT"}
        ),
        "portfolio_fit_agent": MagicMock(return_value=None),
        "trade_revision": MagicMock(return_value=dict(mock_trade)),
        "portfolio_manager": MagicMock(
            return_value={"entscheidung": "GENEHMIGT", "confidence": 3}
        ),
        "build_feedback_context": MagicMock(return_value=""),
        "build_reflection_context": MagicMock(return_value=""),
    }


class TestRunPipelineExitMode:
    """exit_mode=True verdrahtet den Exit-Pfad im Trader-Schritt (Schritt 4)."""

    def test_exit_mode_single_run_uses_trader_exit(self, tmp_path, monkeypatch):
        """exit_mode=True + ensemble=False → trader_exit statt trader."""
        from concilium.pipeline import run_pipeline

        monkeypatch.chdir(tmp_path)
        state = tmp_path / "state"
        state.mkdir(exist_ok=True)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(state))
        patches = _pipeline_patches()

        with patch.multiple("concilium.pipeline", **patches):
            result = run_pipeline("AAPL", llm=MagicMock(), ensemble=False, exit_mode=True)

        patches["trader"].assert_not_called()
        patches["trader_exit"].assert_called_once()
        assert result["trade"]["aktion"] == "VERKAUFEN"

    def test_exit_mode_ensemble_gets_exit_mode_true(self, tmp_path, monkeypatch):
        """exit_mode=True + ensemble=True → ensemble_trader(exit_mode=True)."""
        from concilium.pipeline import run_pipeline

        monkeypatch.chdir(tmp_path)
        state = tmp_path / "state"
        state.mkdir(exist_ok=True)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(state))
        patches = _pipeline_patches()

        with patch.multiple("concilium.pipeline", **patches):
            run_pipeline(
                "AAPL", llm=MagicMock(), ensemble=True, ensemble_runs=3, exit_mode=True
            )

        _, kwargs = patches["ensemble_trader"].call_args
        assert kwargs.get("exit_mode") is True
        patches["trader"].assert_not_called()
        patches["trader_exit"].assert_not_called()

    def test_default_mode_unchanged(self, tmp_path, monkeypatch):
        """Ohne exit_mode: trader bzw. ensemble_trader ohne Exit-Prompt (kompatibel)."""
        from concilium.pipeline import run_pipeline

        monkeypatch.chdir(tmp_path)
        state = tmp_path / "state"
        state.mkdir(exist_ok=True)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(state))
        patches = _pipeline_patches()

        with patch.multiple("concilium.pipeline", **patches):
            run_pipeline("AAPL", llm=MagicMock(), ensemble=False)

        patches["trader"].assert_called_once()
        patches["trader_exit"].assert_not_called()

        patches2 = _pipeline_patches()
        with patch.multiple("concilium.pipeline", **patches2):
            run_pipeline("MSFT", llm=MagicMock(), ensemble=True)
        _, kwargs = patches2["ensemble_trader"].call_args
        assert kwargs.get("exit_mode") is False


# ---------------------------------------------------------------------------
# (e) run_review — Exit-Prompt + Review-Journal
# ---------------------------------------------------------------------------


def _mock_positions() -> list[dict]:
    """2 Aktien + 1 ETF (analog tests/test_review.py)."""
    return [
        {"name": "Apple Inc.", "ticker": "AAPL", "sheet_symbol": "AAPL",
         "type": "Aktie", "region": "USA", "depot_pct": 5.2, "value_eur": 5200.0,
         "_idx": 0},
        {"name": "BASF SE", "ticker": "BAS.DE", "sheet_symbol": "BAS.DE",
         "type": "Aktie", "region": "Europa", "depot_pct": 2.0,
         "value_eur": 2000.0, "_idx": 1},
        {"name": "iShares Core MSCI World UCITS ETF", "ticker": "IS3R.DE",
         "sheet_symbol": "IS3R.DE", "type": "ETF", "region": "Welt",
         "depot_pct": 30.0, "value_eur": 30000.0, "_idx": 2},
    ]


class TestRunReviewExitPath:
    """run_review: exit_mode=True + Review-Journal nach jedem erfolgreichen Lauf."""

    def test_run_review_passes_exit_mode_and_journal_false(self):
        """run_pipeline bekommt exit_mode=True und journal=False."""
        captured: dict = {}

        def mock_run_pipeline(ticker, **kwargs):
            captured.update(kwargs)
            return {
                "ticker": ticker,
                "data": {"ticker": ticker},
                "trade": {"aktion": "HALTEN"},
                "final": {"entscheidung": "GENEHMIGT"},
                "no_llm": False,
            }

        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.review.run_pipeline", side_effect=mock_run_pipeline):
                run_review(None)

        assert captured.get("exit_mode") is True
        assert captured.get("journal") is False

    def test_run_review_writes_reviews_csv(self, tmp_path, monkeypatch):
        """End-to-End (gemockte Agenten): Review-Lauf schreibt journal/reviews.csv."""
        monkeypatch.chdir(tmp_path)
        state = tmp_path / "state"
        state.mkdir(exist_ok=True)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(state))

        llm = MagicMock()
        llm.total_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                           "total_tokens": 0}

        # Per-Ticker-Tracking: collect_ticker_data merkt den aktuellen Ticker,
        # die Trader-Mocks liefern ticker-spezifische Trades (AAPL=VERKAUFEN,
        # BAS.DE=HALTEN).
        current = {"ticker": "AAPL"}

        def fake_collect(ticker, peers=None, as_of=None):
            current["ticker"] = ticker
            return {
                "ticker": ticker,
                "fundamentals": {"name": ticker},
                "technicals": {},
                "sentiment": {},
                "news": [],
            }

        def trade_for() -> dict:
            if current["ticker"] == "AAPL":
                return {"rolle": "Trader", "aktion": "VERKAUFEN",
                        "rating": "VERKAUFEN", "zielkurs": 140, "stop_loss": 165,
                        "positionsanteil": 0, "_raw": ""}
            return {"rolle": "Trader", "aktion": "HALTEN", "rating": "HALTEN",
                    "zielkurs": None, "stop_loss": None, "positionsanteil": 0,
                    "_raw": ""}

        patches = {
            "collect_ticker_data": MagicMock(side_effect=fake_collect),
            "analyst_team": MagicMock(return_value={
                "fundamental": {"stimmung": "bearish", "score": 2,
                                "zusammenfassung": "Schwach", "_raw": ""},
                "technical": {"stimmung": "bearish", "score": 2,
                              "zusammenfassung": "Schwach", "_raw": ""},
                "sentiment": {"stimmung": "neutral", "score": 3,
                              "zusammenfassung": "Ok", "_raw": ""},
            }),
            "debate": MagicMock(
                return_value={"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}}
            ),
            "trader": MagicMock(side_effect=lambda a, d, llm, **kw: trade_for()),
            "ensemble_trader": MagicMock(side_effect=lambda a, d, llm, **kw: trade_for()),
            "risk_manager": MagicMock(
                return_value={"risiko_score": 3, "empfehlung": "GENEHMIGT"}
            ),
            "portfolio_fit_agent": MagicMock(return_value=None),
            "trade_revision": MagicMock(side_effect=lambda t, r, pf, llm, **kw: t),
            "portfolio_manager": MagicMock(
                return_value={"entscheidung": "GENEHMIGT", "confidence": 3}
            ),
            "build_feedback_context": MagicMock(return_value=""),
            "build_reflection_context": MagicMock(return_value=""),
        }

        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch.multiple("concilium.pipeline", **patches):
                review_result = run_review(llm)

        # Reviews-Journal: 2 Zeilen (AAPL=VERKAUFEN → 1, BAS.DE=HALTEN → 0)
        reviews = tmp_path / "journal" / "reviews.csv"
        assert reviews.exists()
        assert _read_header(str(reviews)) == REVIEW_HEADER
        rows = _read_rows(str(reviews))
        assert len(rows) == 2
        by_ticker = {r["ticker"]: r for r in rows}
        aapl = by_ticker["AAPL"]
        assert aapl["action"] == "VERKAUFEN"
        assert aapl["verkauf_empfehlung"] == "1"
        assert aapl["depot_pct"] == "5.2"
        assert aapl["name"] == "Apple Inc."
        assert aapl["final_decision"] == "GENEHMIGT"
        assert aapl["reflection_status"] == "pending"
        bas = by_ticker["BAS.DE"]
        assert bas["action"] == "HALTEN"
        assert bas["verkauf_empfehlung"] == "0"
        assert bas["name"] == "BASF SE"
        # Kein Doppel-Logging: decisions.csv bleibt unberührt
        assert not (tmp_path / "journal" / "decisions.csv").exists()
        # Review-Ergebnis selbst unverändert
        assert review_result["ergebnisse"]["AAPL"]["verkauf_empfehlung"] is True
        assert review_result["ergebnisse"]["BAS.DE"]["verkauf_empfehlung"] is False
        assert review_result["fehler"] == 0

    def test_run_review_journal_crash_does_not_fail_ticker(self, tmp_path, monkeypatch):
        """append_review_decision-Fehler → Warnung, kein Fehler im Review-Ergebnis."""
        monkeypatch.chdir(tmp_path)
        state = tmp_path / "state"
        state.mkdir(exist_ok=True)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(state))

        def mock_run_pipeline(ticker, **kwargs):
            return {
                "ticker": ticker,
                "data": {"ticker": ticker},
                "trade": {"aktion": "VERKAUFEN"},
                "final": {"entscheidung": "GENEHMIGT"},
                "no_llm": False,
            }

        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.review.run_pipeline", side_effect=mock_run_pipeline):
                with patch("concilium.review.append_review_decision",
                           side_effect=RuntimeError("Disk tot")):
                    review_result = run_review(None)

        # Ticker gilt trotzdem als erfolgreich analysiert
        assert "AAPL" in review_result["ergebnisse"]
        assert review_result["fehler"] == 0
        assert not (tmp_path / "journal" / "reviews.csv").exists()


# ---------------------------------------------------------------------------
# (f) CLI --review: stderr-Hinweis auf journal/reviews.csv
# ---------------------------------------------------------------------------


def _run_cli_review_min(monkeypatch, tmp_path, review_result: dict):
    """Führt main(['--review']) mit gemockten Abhängigkeiten aus (wie test_review.py)."""
    from concilium import cli as cli_mod

    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("CONCILIUM_STATE_DIR", str(state_dir))

    with patch.object(
        cli_mod,
        "evaluate_journal",
        side_effect=lambda *a, **k: {
            "anzahl_entscheidungen": 5,
            "hit_rate_gesamt": 0.4,
            "nach_aktion": {},
        },
    ):
        with patch.object(cli_mod, "_write_calibration_json", side_effect=lambda *a, **k: None):
            with patch.object(
                cli_mod,
                "generate_track_record_report",
                side_effect=lambda *a, **k: "# Track Record Report",
            ):
                with patch.object(cli_mod, "LLMClient"):
                    with patch.object(
                        cli_mod, "run_review", side_effect=lambda llm, **kw: review_result
                    ):
                        return main(["--review"])


class TestReviewCliJournalHint:
    """--review: stderr-Hinweis auf das Review-Journal (nur bei ≥1 Position)."""

    def test_hint_shown_with_positions(self, tmp_path, monkeypatch, capsys):
        review_result = {
            "ergebnisse": {
                "AAPL": {
                    "result": {
                        "ticker": "AAPL",
                        "data": {"ticker": "AAPL"},
                        "trade": {"aktion": "VERKAUFEN"},
                        "final": {"entscheidung": "GENEHMIGT"},
                        "no_llm": False,
                    },
                    "report": "# AAPL",
                    "verkauf_empfehlung": True,
                    "depot_pct": 5.2,
                    "name": "Apple",
                },
            },
            "positions_uebersprungen": 0,
            "fehler": 0,
            "gesamt_positionen": 1,
        }

        code = _run_cli_review_min(monkeypatch, tmp_path, review_result)
        assert code == 0

        captured = capsys.readouterr()
        out = captured.out + captured.err  # Hinweis geht auf stderr
        assert "journal/reviews.csv" in out

    def test_no_hint_without_positions(self, tmp_path, monkeypatch, capsys):
        review_result = {
            "ergebnisse": {},
            "positions_uebersprungen": 2,
            "fehler": 0,
            "gesamt_positionen": 2,
        }

        code = _run_cli_review_min(monkeypatch, tmp_path, review_result)
        assert code == 0

        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "journal/reviews.csv" not in out
