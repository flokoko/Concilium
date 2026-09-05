"""Tests für Journal-Hygiene: Idempotenz-Guard + Rotation aufgelöster Einträge.

Analog TradingAgents' TradingMemoryLog (Idempotenz + Rotation):

- Idempotenz-Guard: append_decision schreibt KEINEN zweiten Eintrag für
  dasselbe (ticker, timestamp) — schützt vor Doppel-Logging bei
  Checkpoint-Resume innerhalb derselben Sekunde. Verschiedene Ticker oder
  verschiedene Timestamps werden normal geschrieben. Best effort: Ein
  unlesbares/korruptes Journal blockiert den Append NICHT.
- Rotation (CONCILIUM_JOURNAL_MAX_RESOLVED): Optionaler Cap auf resolved
  Einträge — die ältesten resolved-Zeilen (nach resolved_at, Fallback
  timestamp) werden geprunt. Pending- und Legacy-Zeilen werden NIE geprunt.
  Default 0 = Rotation deaktiviert (Rückwärtskompatibilität).
- config.journal_max_resolved(): Default 0, Env-Parity, lauter Tippfehler.

Konventionen wie in test_deferred_reflection.py: Journal via tmp_path,
Zeit via monkeypatch von concilium.journal.datetime gefroren (deterministisch).
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium import config  # noqa: E402
from concilium.journal import (  # noqa: E402
    JOURNAL_HEADER,
    REFLECTION_STATUS_RESOLVED,
    _prune_resolved,
    append_decision,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_FIXED_TS = "2026-09-05 12:00:00"


class _FrozenDateTime(datetime):
    """datetime-Ersatz mit festem now() — simuliert Resume in derselben Sekunde."""

    @classmethod
    def now(cls, tz=None):  # noqa: ARG003 — Signatur wie datetime.now
        return cls(2026, 9, 5, 12, 0, 0)


def _make_result(ticker: str) -> dict:
    """Minimal-result für append_decision."""
    return {
        "ticker": ticker,
        "trade": {"aktion": "KAUFEN"},
        "final": {"entscheidung": "GENEHMIGT", "confidence": 4},
    }


def _read_rows(journal_file: str) -> list[dict]:
    """Liest alle Daten-Zeilen des Journals."""
    with open(journal_file, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_journal(tmp_path, rows: list[dict]) -> str:
    """Schreibt journal/decisions.csv unter tmp_path und gibt den Pfad zurück."""
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir(exist_ok=True)
    path = journal_dir / "decisions.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in JOURNAL_HEADER})
    return str(path)


def _resolved_row(ticker: str, ts: str, resolved_at: str) -> dict:
    """Ein resolved-Journal-Eintrag."""
    return {
        "timestamp": ts,
        "ticker": ticker,
        "action": "KAUFEN",
        "reflection_status": REFLECTION_STATUS_RESOLVED,
        "resolved_at": resolved_at,
        "realised_return_pct": "+5.00",
        "alpha_pct": "+1.00",
        "lesson": "Test-Lektion",
    }


# ---------------------------------------------------------------------------
# Tests: Idempotenz-Guard
# ---------------------------------------------------------------------------


class TestIdempotenzGuard:
    """append_decision schreibt (ticker, timestamp)-Duplikate NICHT doppelt."""

    def test_same_ticker_same_timestamp_appended_once(self, tmp_path, monkeypatch):
        """Zweiter Append mit demselben (ticker, timestamp) → nur EINE Zeile."""
        monkeypatch.setattr("concilium.journal.datetime", _FrozenDateTime)
        journal_file = str(tmp_path / "journal" / "decisions.csv")

        append_decision(_make_result("AAPL"), journal_file=journal_file)
        append_decision(_make_result("AAPL"), journal_file=journal_file)

        rows = _read_rows(journal_file)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["timestamp"] == _FIXED_TS

    def test_duplicate_logs_warning(self, tmp_path, monkeypatch, caplog):
        """Blockierter Doppel-Append warnt sichtbar (Duplikat verhindert)."""
        monkeypatch.setattr("concilium.journal.datetime", _FrozenDateTime)
        journal_file = str(tmp_path / "journal" / "decisions.csv")
        append_decision(_make_result("AAPL"), journal_file=journal_file)

        with caplog.at_level("WARNING", logger="concilium.journal"):
            append_decision(_make_result("AAPL"), journal_file=journal_file)

        assert any("Duplikat" in rec.message for rec in caplog.records)

    def test_different_timestamp_same_ticker_writes_both(self, tmp_path, monkeypatch):
        """Gleicher Ticker, anderer Timestamp → beide Zeilen (kein Über-Block)."""
        monkeypatch.setattr("concilium.journal.datetime", _FrozenDateTime)
        journal_file = str(tmp_path / "journal" / "decisions.csv")
        append_decision(_make_result("AAPL"), journal_file=journal_file)

        # Zeit verschieben (anderer Tag) → zweiter Eintrag ist legitim
        class _LaterDateTime(_FrozenDateTime):
            @classmethod
            def now(cls, tz=None):  # noqa: ARG003
                return cls(2026, 9, 6, 12, 0, 0)

        monkeypatch.setattr("concilium.journal.datetime", _LaterDateTime)
        append_decision(_make_result("AAPL"), journal_file=journal_file)

        rows = _read_rows(journal_file)
        assert len(rows) == 2
        assert {row["timestamp"] for row in rows} == {
            _FIXED_TS,
            "2026-09-06 12:00:00",
        }

    def test_different_ticker_same_timestamp_writes_both(self, tmp_path, monkeypatch):
        """Gleicher Timestamp, verschiedene Ticker → beide Zeilen."""
        monkeypatch.setattr("concilium.journal.datetime", _FrozenDateTime)
        journal_file = str(tmp_path / "journal" / "decisions.csv")

        append_decision(_make_result("AAPL"), journal_file=journal_file)
        append_decision(_make_result("MSFT"), journal_file=journal_file)

        rows = _read_rows(journal_file)
        assert len(rows) == 2
        assert {row["ticker"] for row in rows} == {"AAPL", "MSFT"}

    def test_guard_case_insensitive_ticker(self, tmp_path, monkeypatch):
        """(ticker, timestamp)-Vergleich ist case-insensitive beim Ticker."""
        monkeypatch.setattr("concilium.journal.datetime", _FrozenDateTime)
        journal_file = str(tmp_path / "journal" / "decisions.csv")

        append_decision(_make_result("aapl"), journal_file=journal_file)
        append_decision(_make_result("AAPL"), journal_file=journal_file)

        rows = _read_rows(journal_file)
        assert len(rows) == 1

    def test_guard_best_effort_on_corrupt_file(self, tmp_path):
        """Korruptes (nicht-CSV-)Journal blockiert den Append NICHT (best effort)."""
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir(exist_ok=True)
        journal_file = str(journal_dir / "decisions.csv")
        with open(journal_file, "w", encoding="utf-8") as fh:
            fh.write("keine csv-zeile\n")

        append_decision(_make_result("AAPL"), journal_file=journal_file)

        rows = _read_rows(journal_file)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Tests: Rotation aufgelöster Einträge
# ---------------------------------------------------------------------------


class TestRotationResolved:
    """CONCILIUM_JOURNAL_MAX_RESOLVED prunt älteste resolved-Zeilen."""

    def test_cap2_with3_resolved_keeps_newest_two_and_pending(self, tmp_path, monkeypatch):
        """Cap=2, 3 resolved → nur die 2 neuesten resolved bleiben, pending bleibt."""
        monkeypatch.setenv("CONCILIUM_JOURNAL_MAX_RESOLVED", "2")
        journal_file = _write_journal(
            tmp_path,
            [
                _resolved_row("OLD.DE", "2026-06-01 10:00:00", "2026-07-01 10:00:00"),
                _resolved_row("MID.DE", "2026-07-01 10:00:00", "2026-08-01 10:00:00"),
                _resolved_row("NEW.DE", "2026-08-01 10:00:00", "2026-09-01 10:00:00"),
            ],
        )

        monkeypatch.setattr("concilium.journal.datetime", _FrozenDateTime)
        append_decision(_make_result("AAPL"), journal_file=journal_file)

        rows = _read_rows(journal_file)
        resolved_tickers = [
            row["ticker"] for row in rows if row["reflection_status"] == "resolved"
        ]
        # Nur die 2 neuesten resolved bleiben
        assert resolved_tickers == ["MID.DE", "NEW.DE"]
        # Pending-Eintrag bleibt
        pending = [row for row in rows if row["reflection_status"] == "pending"]
        assert len(pending) == 1
        assert pending[0]["ticker"] == "AAPL"
        assert pending[0]["timestamp"] == _FIXED_TS

    def test_pending_never_pruned(self, tmp_path, monkeypatch):
        """Viele pending-Zeilen über dem Cap → pendings bleiben vollständig."""
        monkeypatch.setenv("CONCILIUM_JOURNAL_MAX_RESOLVED", "1")
        pending_rows = [
            {
                "timestamp": f"2026-08-0{i} 10:00:00",
                "ticker": f"P{i}.DE",
                "action": "KAUFEN",
                "reflection_status": "pending",
            }
            for i in range(1, 6)
        ]
        resolved = [_resolved_row("R.DE", "2026-07-01 10:00:00", "2026-07-15 10:00:00")]
        journal_file = _write_journal(tmp_path, pending_rows + [resolved[0]])

        append_decision(_make_result("AAPL"), journal_file=journal_file)

        rows = _read_rows(journal_file)
        pendings = [row for row in rows if row["reflection_status"] == "pending"]
        # 5 alte pendings + der neue AAPL-pending = 6
        assert len(pendings) == 6
        assert {row["ticker"] for row in pendings} == {
            "P1.DE",
            "P2.DE",
            "P3.DE",
            "P4.DE",
            "P5.DE",
            "AAPL",
        }

    def test_legacy_never_pruned(self, tmp_path, monkeypatch):
        """Legacy-Zeilen (reflection_status '') werden NIE geprunt."""
        monkeypatch.setenv("CONCILIUM_JOURNAL_MAX_RESOLVED", "1")
        legacy_rows = [
            {
                "timestamp": f"2026-05-0{i} 10:00:00",
                "ticker": f"L{i}.DE",
                "action": "KAUFEN",
                "reflection_status": "",
            }
            for i in range(1, 5)
        ]
        journal_file = _write_journal(tmp_path, legacy_rows)

        append_decision(_make_result("AAPL"), journal_file=journal_file)

        rows = _read_rows(journal_file)
        legacy = [row for row in rows if row["reflection_status"] == ""]
        assert len(legacy) == 4

    def test_cap_zero_no_rotation(self, tmp_path):
        """Cap=0 (Default) → nichts wird geprunt (Rückwärtskompatibilität)."""
        resolved_rows = [
            _resolved_row(
                f"R{i}.DE",
                f"2026-06-0{i} 10:00:00",
                f"2026-07-0{i} 10:00:00",
            )
            for i in range(1, 4)
        ]
        journal_file = _write_journal(tmp_path, resolved_rows)
        before = _read_rows(journal_file)

        append_decision(_make_result("AAPL"), journal_file=journal_file)

        rows = _read_rows(journal_file)
        resolved = [row for row in rows if row["reflection_status"] == "resolved"]
        # Alle 3 resolved bleiben + neuer pending
        assert len(resolved) == 3
        assert len(rows) == len(before) + 1

    def test_cap_exactly_met_no_prune(self, tmp_path, monkeypatch):
        """resolved == Cap → nichts wird geprunt (excess <= 0)."""
        monkeypatch.setenv("CONCILIUM_JOURNAL_MAX_RESOLVED", "2")
        journal_file = _write_journal(
            tmp_path,
            [
                _resolved_row("A.DE", "2026-06-01 10:00:00", "2026-07-01 10:00:00"),
                _resolved_row("B.DE", "2026-07-01 10:00:00", "2026-08-01 10:00:00"),
            ],
        )

        append_decision(_make_result("AAPL"), journal_file=journal_file)

        rows = _read_rows(journal_file)
        resolved = [row for row in rows if row["reflection_status"] == "resolved"]
        assert len(resolved) == 2
        assert {row["ticker"] for row in resolved} == {"A.DE", "B.DE"}

    def test_prune_sorts_by_resolved_at_not_file_position(self, tmp_path, monkeypatch):
        """Ältester resolved_at wird geprunt — auch wenn er NICHT zuerst in der Datei steht."""
        monkeypatch.setenv("CONCILIUM_JOURNAL_MAX_RESOLVED", "2")
        journal_file = _write_journal(
            tmp_path,
            [
                # File-Position 0 = NEUESTE resolved_at → muss bleiben
                _resolved_row("NEWEST.DE", "2026-08-01 10:00:00", "2026-09-01 10:00:00"),
                # File-Position 1 = ÄLTESTE resolved_at → muss geprunt werden
                _resolved_row("OLDEST.DE", "2026-05-01 10:00:00", "2026-06-01 10:00:00"),
                # File-Position 2 = mittlere resolved_at → muss bleiben
                _resolved_row("MIDDLE.DE", "2026-07-01 10:00:00", "2026-08-01 10:00:00"),
            ],
        )

        append_decision(_make_result("AAPL"), journal_file=journal_file)

        rows = _read_rows(journal_file)
        resolved_tickers = [
            row["ticker"] for row in rows if row["reflection_status"] == "resolved"
        ]
        assert resolved_tickers == ["NEWEST.DE", "MIDDLE.DE"]

    def test_prune_resolved_direct_missing_resolved_at_falls_back_to_timestamp(
        self, tmp_path
    ):
        """_prune_resolved direkt: resolved_at leer → Sortierung über timestamp."""
        journal_file = _write_journal(
            tmp_path,
            [
                _resolved_row("A.DE", "2026-06-01 10:00:00", ""),  # kein resolved_at
                _resolved_row("B.DE", "2026-07-01 10:00:00", "2026-08-01 10:00:00"),
                _resolved_row("C.DE", "2026-08-01 10:00:00", "2026-09-01 10:00:00"),
            ],
        )

        _prune_resolved(journal_file, 2)

        rows = _read_rows(journal_file)
        resolved_tickers = [
            row["ticker"] for row in rows if row["reflection_status"] == "resolved"
        ]
        # A.DE (timestamp-Fallback, älteste) wird geprunt
        assert resolved_tickers == ["B.DE", "C.DE"]

    def test_prune_resolved_direct_disabled(self, tmp_path):
        """_prune_resolved mit Cap <= 0 → No-op."""
        journal_file = _write_journal(
            tmp_path,
            [
                _resolved_row("A.DE", "2026-06-01 10:00:00", "2026-07-01 10:00:00"),
                _resolved_row("B.DE", "2026-07-01 10:00:00", "2026-08-01 10:00:00"),
            ],
        )
        before = _read_rows(journal_file)

        _prune_resolved(journal_file, 0)
        _prune_resolved(journal_file, -5)

        assert _read_rows(journal_file) == before

    def test_prune_never_crashes_on_missing_file(self, tmp_path):
        """_prune_resolved auf fehlender Datei → kein Crash (best effort)."""
        missing = str(tmp_path / "journal" / "does_not_exist.csv")
        _prune_resolved(missing, 5)  # darf nicht werfen
        assert not os.path.exists(missing)

    def test_rotation_does_not_interfere_with_pipeline_patch_usage(self, tmp_path, monkeypatch):
        """append_decision bleibt unter patch(...) ersetzbar (Pipeline-Kompatibilität)."""
        monkeypatch.setenv("CONCILIUM_JOURNAL_MAX_RESOLVED", "2")
        with patch("concilium.journal.append_decision"):
            pass  # Smoke: Import + Patch-Mechanik unverändert


# ---------------------------------------------------------------------------
# Tests: config.journal_max_resolved()
# ---------------------------------------------------------------------------


class TestJournalMaxResolvedConfig:
    """CONCILIUM_JOURNAL_MAX_RESOLVED: Default 0, per Env setzbar."""

    def test_default_zero(self, monkeypatch):
        """Ohne Env → 0 (Rotation deaktiviert, Rückwärtskompatibilität)."""
        monkeypatch.delenv("CONCILIUM_JOURNAL_MAX_RESOLVED", raising=False)
        assert config.journal_max_resolved() == 0

    def test_env_cap(self, monkeypatch):
        """CONCILIUM_JOURNAL_MAX_RESOLVED=50 → 50."""
        monkeypatch.setenv("CONCILIUM_JOURNAL_MAX_RESOLVED", "50")
        assert config.journal_max_resolved() == 50

    def test_returns_int_not_str(self, monkeypatch):
        """Rückgabetyp ist int."""
        monkeypatch.setenv("CONCILIUM_JOURNAL_MAX_RESOLVED", "7")
        result = config.journal_max_resolved()
        assert result == 7
        assert isinstance(result, int)

    def test_env_typo_raises_loudly(self, monkeypatch):
        """Tippfehler 'viel' → LAUTE ValueError mit Env-Variablen-Namen."""
        monkeypatch.setenv("CONCILIUM_JOURNAL_MAX_RESOLVED", "viel")
        with pytest.raises(ValueError) as exc_info:
            config.journal_max_resolved()
        msg = str(exc_info.value)
        assert "CONCILIUM_JOURNAL_MAX_RESOLVED" in msg
        assert "viel" in msg

    def test_lazy_loading_env_change_between_calls(self, monkeypatch):
        """Lazy-Loading: Env-Änderung zwischen zwei Aufrufen wirkt sofort."""
        monkeypatch.setenv("CONCILIUM_JOURNAL_MAX_RESOLVED", "10")
        first = config.journal_max_resolved()
        monkeypatch.setenv("CONCILIUM_JOURNAL_MAX_RESOLVED", "0")
        second = config.journal_max_resolved()
        assert first == 10
        assert second == 0

    def test_config_typo_does_not_crash_append(self, tmp_path, monkeypatch):
        """Ungültiger Env-Wert bricht append_decision NICHT (Entry bleibt geschrieben)."""
        monkeypatch.setenv("CONCILIUM_JOURNAL_MAX_RESOLVED", "viel")
        journal_file = str(tmp_path / "journal" / "decisions.csv")

        append_decision(_make_result("AAPL"), journal_file=journal_file)  # crasht nicht

        rows = _read_rows(journal_file)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
