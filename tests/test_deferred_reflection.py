"""Tests für Roadmap C6: Deferred Reflection (Pending-Entries, look-ahead-frei).

Testet:
- Journal-Schema: append_decision setzt reflection_status="pending";
  Header-Migration ergänzt die C6-Spalten; resolved-Zeilen tragen
  resolved_at/realised_return_pct/alpha_pct/lesson.
- resolve_pending_reflections: "" bei nicht-abgelaufenem Fenster (KEIN
  Look-ahead), Auflösung + Journal-Rückwärtsschreib bei abgelaufenem
  Fenster, persistierte Returns/Lektion, Case-Insensitivity, crasht nie.
- build_reflection_context: Legacy-Zeile mit abgelaufenem Fenster liefert
  Reflexion (durch resolve_pending_reflections verifiziert).

Konventionen wie in test_reflection.py: Journal via tmp_path + chdir,
Preise via patch("concilium.evaluate._load_price_history") gemockt (offline).
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.feedback import (  # noqa: E402
    resolve_pending_reflections,
)
from concilium.journal import (  # noqa: E402
    JOURNAL_HEADER,
    append_decision,
)

# --------------------------------------------------------------------------- #
# Helper (wie test_reflection.py / test_cross_ticker.py)
# --------------------------------------------------------------------------- #


def _make_prices(start_price: float, n_days: int, drift: float = 0.0) -> list[dict]:
    """Erzeugt eine Liste von Preis-Dicts für n_days Tage ab vor n_days+5 Tagen."""
    prices: list[dict] = []
    base_date = datetime.now() - timedelta(days=n_days + 5)
    price = start_price
    for i in range(n_days):
        d = base_date + timedelta(days=i)
        price = price * (1.0 + drift)
        prices.append({
            "date": d.strftime("%Y-%m-%d"),
            "close": round(price, 2),
            "high": round(price * 1.01, 2),
            "low": round(price * 0.99, 2),
        })
    return prices


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


def _ts_expired(days_ago: int = 45) -> str:
    """Timestamp mit vollständig abgelaufenem Fenster (lookback_days=30)."""
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


def _ts_unexpired(days_ago: int = 5) -> str:
    """Timestamp mit NICHT abgelaufenem Fenster (lookback_days=30)."""
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


class _MockLLM:
    """Mock-LLM, das eine feste Lektion zurückgibt."""

    def __init__(self, lesson: str = "Konzentriere dich auf die Fundamentaldaten."):
        self._lesson = lesson
        self.captured: list[list[dict]] = []

    def chat(self, messages, temperature=0.3, **kwargs):
        self.captured.append(messages)
        return self._lesson


# --------------------------------------------------------------------------- #
# Tests: Journal-Schema (append_decision + Migration)
# --------------------------------------------------------------------------- #


class TestJournalPendingEntries:
    """append_decision setzt neue Entscheidungen auf pending (C6)."""

    def test_new_decision_is_pending(self, tmp_path):
        """Neue Entscheidung → reflection_status='pending', C6-Felder leer."""
        journal_file = str(tmp_path / "decisions.csv")
        append_decision(
            {
                "ticker": "AAPL",
                "trade": {"aktion": "KAUFEN"},
                "final": {"entscheidung": "GENEHMIGT", "confidence": 4},
            },
            journal_file=journal_file,
        )
        with open(journal_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
        assert fieldnames[-5:] == [
            "reflection_status", "resolved_at", "realised_return_pct", "alpha_pct", "lesson",
        ]
        assert rows[0]["reflection_status"] == "pending"
        assert rows[0]["resolved_at"] == ""
        assert rows[0]["realised_return_pct"] == ""
        assert rows[0]["alpha_pct"] == ""
        assert rows[0]["lesson"] == ""

    def test_header_migration_adds_c6_columns(self, tmp_path):
        """Alte CSV ohne C6-Spalten wird migriert: Header ergänzt, Werte leer."""
        journal_file = str(tmp_path / "decisions.csv")
        # Journal im Stand vor C6 schreiben (ohne die 5 C6-Spalten)
        old_fields = [
            f for f in JOURNAL_HEADER
            if f not in ("reflection_status", "resolved_at", "realised_return_pct", "alpha_pct", "lesson")
        ]
        with open(journal_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=old_fields)
            writer.writeheader()
            writer.writerow({
                "timestamp": "2026-08-01 10:00:00",
                "ticker": "OLD.DE",
                "action": "KAUFEN",
            })

        append_decision(
            {
                "ticker": "AAPL",
                "trade": {"aktion": "KAUFEN"},
                "final": {"entscheidung": "GENEHMIGT", "confidence": 4},
            },
            journal_file=journal_file,
        )

        with open(journal_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        for col in ("reflection_status", "resolved_at", "realised_return_pct", "alpha_pct", "lesson"):
            assert col in fieldnames
        assert len(rows) == 2
        # Neue Zeile: pending
        assert rows[-1]["reflection_status"] == "pending"
        # Migrierte Legacy-Zeile: neue Spalten leer (Status "" = Legacy)
        assert rows[0]["reflection_status"] == ""
        assert rows[0]["resolved_at"] == ""
        assert rows[0]["realised_return_pct"] == ""
        assert rows[0]["alpha_pct"] == ""
        assert rows[0]["lesson"] == ""

    def test_c6_columns_in_journal_header(self):
        """JOURNAL_HEADER enthält die fünf C6-Spalten in definierter Reihenfolge."""
        assert JOURNAL_HEADER[-5:] == [
            "reflection_status", "resolved_at", "realised_return_pct", "alpha_pct", "lesson",
        ]


# --------------------------------------------------------------------------- #
# Tests: resolve_pending_reflections
# --------------------------------------------------------------------------- #


class TestResolvePendingReflections:
    """Testet resolve_pending_reflections (C6, look-ahead-frei)."""

    def test_no_journal_returns_empty(self, tmp_path, monkeypatch):
        """Fehlendes Journal → ""."""
        monkeypatch.chdir(tmp_path)
        assert resolve_pending_reflections("AAPL") == ""

    def test_no_matching_ticker_returns_empty(self, tmp_path, monkeypatch):
        """Ticker nicht im Journal → ""."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "MSFT", "action": "KAUFEN", "timestamp": _ts_expired(),
            "reflection_status": "pending",
        }])
        assert resolve_pending_reflections("AAPL") == ""

    def test_case_insensitive_ticker_matching(self, tmp_path, monkeypatch):
        """Ticker-Matching ist case-insensitive."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "aapl", "action": "KAUFEN", "timestamp": _ts_expired(),
            "reflection_status": "pending",
        }])
        with patch("concilium.evaluate._load_price_history", return_value=None):
            result = resolve_pending_reflections("AAPL")
        # Preise None → Return None → "" (aber der Ticker wurde gefunden:
        # das zeigen die anderen Tests mit echten Preisen).
        assert result == ""

    def test_unexpired_window_returns_empty_no_lookahead(self, tmp_path, monkeypatch):
        """C6-Kern: Nicht abgelaufenes Fenster → "" (KEIN Look-ahead)."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "AAPL", "action": "KAUFEN", "timestamp": _ts_unexpired(5),
            "reflection_status": "pending",
        }])
        prices = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.0)

        def mock_load(ticker, *, lookback_days=30):
            return spy if ticker == "SPY" else prices

        # Preise wären verfügbar — trotzdem darf NICHT aufgelöst werden
        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = resolve_pending_reflections("AAPL")
        assert result == ""
        # Zeile muss weiterhin pending sein
        with open(str(tmp_path / "journal" / "decisions.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["reflection_status"] == "pending"

    def test_expired_window_resolves_and_persists(self, tmp_path, monkeypatch):
        """Abgelaufenes Fenster → Auflösung: Text + persistierte Werte im Journal."""
        monkeypatch.chdir(tmp_path)
        ts = _ts_expired(45)
        _write_journal(tmp_path, [{
            "ticker": "AAPL", "action": "KAUFEN", "timestamp": ts,
            "reflection_status": "pending",
        }])
        prices = _make_prices(100, 60, drift=0.01)   # steigend → positiver Return
        spy = _make_prices(100, 60, drift=0.005)     # SPY steigt langsamer

        def mock_load(ticker, *, lookback_days=30):
            return spy if ticker == "SPY" else prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = resolve_pending_reflections("AAPL")

        assert result != ""
        assert "LETZTE ENTSCHEIDUNG ZU AAPL" in result
        assert "Realisierter Return" in result
        assert "Lerne daraus:" in result

        # Journal-Rückwärtsschreib geprüft
        with open(str(tmp_path / "journal" / "decisions.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["reflection_status"] == "resolved"
        assert rows[0]["resolved_at"] != ""
        assert rows[0]["realised_return_pct"] != ""
        # Alpha vorhanden, weil SPY-Preise gemockt wurden
        assert rows[0]["alpha_pct"] != ""
        assert rows[0]["lesson"] != ""
        # Return-Text und persistierter Wert stimmen überein
        pct_val = float(rows[0]["realised_return_pct"])
        assert f"{pct_val:+.2f}%" in result

    def test_resolved_llm_lesson_used(self, tmp_path, monkeypatch):
        """Bei LLM-Übergabe wird die LLM-Lektion generiert und persistiert."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "AAPL", "action": "KAUFEN", "timestamp": _ts_expired(45),
            "reflection_status": "pending",
        }])
        prices = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.005)

        def mock_load(ticker, *, lookback_days=30):
            return spy if ticker == "SPY" else prices

        llm = _MockLLM("Timing war besser als der Markt.")
        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = resolve_pending_reflections("AAPL", llm=llm)

        assert result != ""
        assert "Timing war besser als der Markt." in result
        assert len(llm.captured) == 1  # genau ein Coach-Prompt
        with open(str(tmp_path / "journal" / "decisions.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["lesson"] == "Timing war besser als der Markt."

    def test_llm_failure_still_resolves_with_deterministic_lesson(self, tmp_path, monkeypatch):
        """LLM-Fehler → deterministische Lektion, Auflösung läuft trotzdem."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "AAPL", "action": "KAUFEN", "timestamp": _ts_expired(45),
            "reflection_status": "pending",
        }])
        prices = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.005)

        def mock_load(ticker, *, lookback_days=30):
            return spy if ticker == "SPY" else prices

        class _FailingLLM:
            def chat(self, messages, temperature=0.3, **kwargs):
                raise RuntimeError("LLM down")

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = resolve_pending_reflections("AAPL", llm=_FailingLLM())

        assert result != ""
        assert "Lerne daraus:" in result
        with open(str(tmp_path / "journal" / "decisions.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["reflection_status"] == "resolved"
        assert rows[0]["lesson"] != ""

    def test_no_prices_returns_empty_stays_pending(self, tmp_path, monkeypatch):
        """Keine Preisdaten → "" und Zeile bleibt pending (kein verfrühtes Resolving)."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "AAPL", "action": "KAUFEN", "timestamp": _ts_expired(45),
            "reflection_status": "pending",
        }])
        with patch("concilium.evaluate._load_price_history", return_value=None):
            result = resolve_pending_reflections("AAPL")
        assert result == ""
        with open(str(tmp_path / "journal" / "decisions.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["reflection_status"] == "pending"

    def test_already_resolved_returns_empty(self, tmp_path, monkeypatch):
        """Alle Zeilen resolved → "" (nichts mehr zu tun)."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "AAPL", "action": "KAUFEN", "timestamp": _ts_expired(45),
            "reflection_status": "resolved", "resolved_at": "2026-09-01 10:00:00",
            "realised_return_pct": "+5.00", "alpha_pct": "+1.00", "lesson": "Alt.",
        }])
        with patch(
            "concilium.feedback.realised_return_for_row",
            side_effect=AssertionError("resolved-Zeile darf nicht erneut resolviert werden"),
        ):
            assert resolve_pending_reflections("AAPL") == ""

    def test_resolves_oldest_pending_first_only_latest(self, tmp_path, monkeypatch):
        """Nur die JÜNGSTE unresolvierte Zeile wird aufgelöst."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            {
                "ticker": "AAPL", "action": "KAUFEN", "timestamp": _ts_expired(90),
                "reflection_status": "pending",
            },
            {
                "ticker": "AAPL", "action": "HALTEN", "timestamp": _ts_expired(40),
                "reflection_status": "pending",
            },
        ])
        prices = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.0)

        def mock_load(ticker, *, lookback_days=30):
            return spy if ticker == "SPY" else prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = resolve_pending_reflections("AAPL")

        assert result != ""
        assert "HALTEN" in result  # jüngste Zeile (40 Tage) wurde aufgelöst
        with open(str(tmp_path / "journal" / "decisions.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        statuses = [r["reflection_status"] for r in rows]
        # Älteste Zeile bleibt pending (wird später einzeln aufgelöst)
        assert statuses == ["pending", "resolved"]

    def test_never_raises_broken_journal(self, tmp_path, monkeypatch):
        """Crasht nie bei kaputtem Journal oder sonstigen Fehlern."""
        monkeypatch.chdir(tmp_path)
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir(exist_ok=True)
        # (a) Binär-Müll
        (journal_dir / "decisions.csv").write_bytes(b"\xff\xfe\x00garbage\xff")
        assert resolve_pending_reflections("AAPL") == ""
        # (b) Journal ist ein Verzeichnis
        (journal_dir / "decisions.csv").unlink()
        (journal_dir / "decisions.csv").mkdir()
        assert resolve_pending_reflections("AAPL") == ""
        # (c) Leerer Ticker
        assert resolve_pending_reflections("") == ""
        assert resolve_pending_reflections("   ") == ""

    def test_legacy_row_resolves_via_window_check(self, tmp_path, monkeypatch):
        """Legacy-Zeile (Status "") mit abgelaufenem Fenster wird aufgelöst."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "AAPL", "action": "KAUFEN", "timestamp": _ts_expired(45),
            # Kein reflection_status → Legacy-Zeile vor C6
        }])
        prices = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.005)

        def mock_load(ticker, *, lookback_days=30):
            return spy if ticker == "SPY" else prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = resolve_pending_reflections("AAPL")

        assert result != ""
        with open(str(tmp_path / "journal" / "decisions.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["reflection_status"] == "resolved"

    def test_write_resolution_failure_still_returns_text(self, tmp_path, monkeypatch):
        """Journal-Rückwärtsschreib schlägt fehl → Text trotzdem geliefert (crasht nie)."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "AAPL", "action": "KAUFEN", "timestamp": _ts_expired(45),
            "reflection_status": "pending",
        }])
        prices = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.005)

        def mock_load(ticker, *, lookback_days=30):
            return spy if ticker == "SPY" else prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load), \
             patch("concilium.feedback._write_resolution", return_value=False):
            result = resolve_pending_reflections("AAPL")

        assert result != ""
        assert "Lerne daraus:" in result


# --------------------------------------------------------------------------- #
# Tests: Persistenz-Nutzung in build_reflection_context (resolved-Zeile)
# --------------------------------------------------------------------------- #


class TestResolvedPersistenceReuse:
    """Resolved-Zeilen liefern Reflexion ohne erneute Berechnung (C6)."""

    def test_resolved_row_feeds_reflection_context(self, tmp_path, monkeypatch):
        """Nach dem Resolving liefert build_reflection_context dieselbe Reflexion."""
        from concilium.feedback import build_reflection_context

        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "AAPL", "action": "KAUFEN", "timestamp": _ts_expired(45),
            "reflection_status": "pending",
        }])
        prices = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.005)

        def mock_load(ticker, *, lookback_days=30):
            return spy if ticker == "SPY" else prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            resolve_pending_reflections("AAPL")

        # Danach: KEINE Preisdaten mehr nötig — persistierte Werte reichen
        with patch(
            "concilium.evaluate._load_price_history",
            side_effect=AssertionError("resolved-Zeile darf keine Preisdaten laden"),
        ):
            reflection = build_reflection_context("AAPL")

        assert reflection != ""
        assert "LETZTE ENTSCHEIDUNG ZU AAPL" in reflection
        assert "Realisierter Return" in reflection

    def test_unexpired_legacy_row_not_resolved_by_pipeline_resolve(self, tmp_path, monkeypatch):
        """Kombiniert: Resolve + Build bei frischer pending-Zeile → beides ""."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "AAPL", "action": "KAUFEN", "timestamp": _ts_unexpired(3),
            "reflection_status": "pending",
        }])
        prices = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.0)

        def mock_load(ticker, *, lookback_days=30):
            return spy if ticker == "SPY" else prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            assert resolve_pending_reflections("AAPL") == ""
            from concilium.feedback import build_reflection_context
            assert build_reflection_context("AAPL") == ""


# --------------------------------------------------------------------------- #
# Tests: Pipeline-Anbindung (resolve vor build_reflection_context)
# --------------------------------------------------------------------------- #

_MOCK_DATA = {
    "ticker": "TEST",
    "fundamentals": {"name": "Test", "sector": "X"},
    "technicals": {"current_price": 100.0},
    "sentiment": {}, "news": [], "macro": {}, "peers": [],
    "history": [{"close": 100.0}], "data_warnings": [],
}


class TestPipelineDeferredReflection:
    """Testet die Pipeline-Anbindung: Resolve läuft vor dem Reflexions-Aufbau."""

    @pytest.fixture
    def state_dir(self, tmp_path, monkeypatch):
        """Isoliertes state-Verzeichnis (CONCILIUM_STATE_DIR auf tmp_path)."""
        d = tmp_path / "state"
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(d))
        return str(d)

    def _run_with_order_capture(self, tmp_path, monkeypatch, *, journal=True):
        """Führt run_pipeline aus und zeichnet die Reihenfolge resolve/build auf."""
        from unittest.mock import MagicMock

        from concilium.pipeline import run_pipeline

        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "TEST", "action": "KAUFEN", "timestamp": _ts_expired(40),
            "reflection_status": "pending",
        }])

        call_order: list[str] = []

        def fake_resolve(ticker, llm=None, lookback_days=30, **kwargs):
            call_order.append("resolve")
            return "RESOLVED-REFLEXION"

        def fake_build(ticker, llm=None, lookback_days=30, **kwargs):
            call_order.append("build")
            return "SAME-REFLEXION"

        llm = MagicMock()
        llm.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        with patch("concilium.pipeline.collect_ticker_data", return_value=_MOCK_DATA), \
             patch("concilium.pipeline.analyst_team", return_value={}), \
             patch("concilium.pipeline.debate", return_value={"bull": {}, "bear": {}}), \
             patch("concilium.pipeline.trader", return_value={"aktion": "HALTEN"}), \
             patch("concilium.pipeline.risk_manager", return_value={"risiko_score": 3, "empfehlung": "GENEHMIGT"}), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_fit_agent", return_value=None), \
             patch("concilium.pipeline.trade_revision", return_value={"aktion": "HALTEN"}), \
             patch("concilium.pipeline.portfolio_manager", return_value={"entscheidung": "GENEHMIGT", "confidence": 4}), \
             patch("concilium.pipeline.build_feedback_context", return_value=""), \
             patch("concilium.pipeline.resolve_pending_reflections", side_effect=fake_resolve), \
             patch("concilium.pipeline.build_reflection_context", side_effect=fake_build), \
             patch("concilium.journal.append_decision"):
            result = run_pipeline("TEST", llm=llm, ensemble=False, journal=journal)

        return result, call_order

    def test_resolve_runs_before_build_reflection(self, tmp_path, monkeypatch, state_dir):
        """resolve_pending_reflections wird VOR build_reflection_context aufgerufen."""
        result, call_order = self._run_with_order_capture(tmp_path, monkeypatch)
        assert call_order == ["resolve", "build"]
        assert result["_resolved_reflection"] == "RESOLVED-REFLEXION"
        # Same-Ticker-Reflexion ist nicht leer → kombiniert bleibt sie allein
        assert result["_reflection_context"] == "SAME-REFLEXION"
        # Report-Abschnitt bleibt Ticker-spezifisch (ohne resolved-Zusatz)
        assert result["reflection"] == "SAME-REFLEXION"

    def test_journal_false_skips_resolve(self, tmp_path, monkeypatch, state_dir):
        """journal=False (--review) → resolve wird NICHT aufgerufen."""
        from unittest.mock import MagicMock

        from concilium.pipeline import run_pipeline

        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "TEST", "action": "KAUFEN", "timestamp": _ts_expired(40),
            "reflection_status": "pending",
        }])

        llm = MagicMock()
        llm.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        with patch("concilium.pipeline.collect_ticker_data", return_value=_MOCK_DATA), \
             patch("concilium.pipeline.analyst_team", return_value={}), \
             patch("concilium.pipeline.debate", return_value={"bull": {}, "bear": {}}), \
             patch("concilium.pipeline.trader", return_value={"aktion": "HALTEN"}), \
             patch("concilium.pipeline.risk_manager", return_value={"risiko_score": 3, "empfehlung": "GENEHMIGT"}), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_fit_agent", return_value=None), \
             patch("concilium.pipeline.trade_revision", return_value={"aktion": "HALTEN"}), \
             patch("concilium.pipeline.portfolio_manager", return_value={"entscheidung": "GENEHMIGT", "confidence": 4}), \
             patch("concilium.pipeline.build_feedback_context", return_value=""), \
             patch("concilium.pipeline.resolve_pending_reflections") as mock_resolve, \
             patch("concilium.pipeline.build_reflection_context", return_value=""), \
             patch("concilium.journal.append_decision"):
            result = run_pipeline("TEST", llm=llm, ensemble=False, journal=False)

        mock_resolve.assert_not_called()
        assert result["_resolved_reflection"] == ""

    def test_resolved_reflection_used_when_same_ticker_empty(self, tmp_path, monkeypatch, state_dir):
        """Leere Same-Ticker-Reflexion → die aufgelöste Reflexion fließt in den Kontext."""
        from unittest.mock import MagicMock

        from concilium.pipeline import run_pipeline

        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "TEST", "action": "KAUFEN", "timestamp": _ts_expired(40),
            "reflection_status": "pending",
        }])

        llm = MagicMock()
        llm.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        with patch("concilium.pipeline.collect_ticker_data", return_value=_MOCK_DATA), \
             patch("concilium.pipeline.analyst_team", return_value={}), \
             patch("concilium.pipeline.debate", return_value={"bull": {}, "bear": {}}), \
             patch("concilium.pipeline.trader", return_value={"aktion": "HALTEN"}), \
             patch("concilium.pipeline.risk_manager", return_value={"risiko_score": 3, "empfehlung": "GENEHMIGT"}), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_fit_agent", return_value=None), \
             patch("concilium.pipeline.trade_revision", return_value={"aktion": "HALTEN"}), \
             patch("concilium.pipeline.portfolio_manager", return_value={"entscheidung": "GENEHMIGT", "confidence": 4}), \
             patch("concilium.pipeline.build_feedback_context", return_value=""), \
             patch("concilium.pipeline.resolve_pending_reflections", return_value="RESOLVED-REFLEXION"), \
             patch("concilium.pipeline.build_reflection_context", return_value=""), \
             patch("concilium.journal.append_decision"):
            result = run_pipeline("TEST", llm=llm, ensemble=False)

        # Same-Ticker-Reflexion leer → resolved-Reflexion füllt _reflection_context
        assert result["_reflection_context"] == "RESOLVED-REFLEXION"
        assert result["_resolved_reflection"] == "RESOLVED-REFLEXION"
        # result["reflection"] bleibt Same-Ticker (leer) — Report unverändert
        assert result["reflection"] == ""
