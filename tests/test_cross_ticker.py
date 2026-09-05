"""Tests für Roadmap C4: Cross-Ticker-Gedächtnis (n_same + n_cross).

Testet:
- build_cross_ticker_context: "" bei leerem Journal / nur gleichem Ticker /
  fehlenden Returns / kaputtem Journal; Inhalte + Sortierung bei Cross-Ticker-
  Zeilen mit Returns; max_lessons-Begrenzung; LLM-Lektion + Fallback.
- build_memory_context: kombiniert Same- + Cross-Ticker-Block.
- Pipeline: Cross-Ticker-Block fließt über reflection_context in die
  Agenten-Prompts; result["reflection"] bleibt Ticker-spezifisch (Report).
- Rückwärtskompatibilität: gepatchte build_reflection_context (wie in den
  Bestands-Pipeline-Tests) verhält sich exakt wie vor C4.

Konventionen wie in test_reflection.py: Journal via tmp_path + monkeypatch.chdir,
Preise via patch("concilium.evaluate._load_price_history") gemockt (offline).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.feedback import (  # noqa: E402
    build_cross_ticker_context,
    build_memory_context,
)
from concilium.journal import JOURNAL_HEADER  # noqa: E402

# --------------------------------------------------------------------------- #
# Helper: synthetische Preisdaten + Journal (wie test_reflection.py)
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
    """Schreibt eine Journal-CSV-Datei (journal/decisions.csv unter tmp_path)."""
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir(exist_ok=True)
    path = journal_dir / "decisions.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
        writer.writeheader()
        for row in rows:
            full_row = {k: row.get(k, "") for k in JOURNAL_HEADER}
            writer.writerow(full_row)
    return str(path)


def _ts(days_ago: int) -> str:
    """Journal-Timestamp im Format wie append_decision.

    C6 look-ahead-frei: Der Wert wird um +35 Tage in die Vergangenheit
    verschoben, damit das Ausgangsfenster (lookback_days=30) bei jedem
    Test vollständig abgelaufen ist — sonst liefern die Legacy-/Pending-
    Zeilen korrekt keine Reflexion mehr (kein Look-ahead). Die relative
    Ordnung (größeres days_ago = älter) bleibt erhalten.
    """
    return (datetime.now() - timedelta(days=days_ago + 35)).strftime("%Y-%m-%d %H:%M:%S")


def _mock_load_factory(price_map: dict[str, list[dict]]):
    """Factory für _load_price_history-Mocks: price_map[ticker] → Preise."""

    def mock_load(ticker, *, lookback_days=30):
        return price_map.get(ticker)

    return mock_load


class _MockLLM:
    """Mock-LLM, das eine feste Lektion zurückgibt."""

    def __init__(self, lesson: str = "Sektorweite Überbewertung wiederholt sich bei Tech-Titeln."):
        self._lesson = lesson
        self.captured: list[list[dict]] = []

    def chat(self, messages, temperature=0.3, **kwargs):
        self.captured.append(messages)
        return self._lesson


class _FailingLLM:
    """Mock-LLM, das immer fehlschlägt."""

    def chat(self, messages, temperature=0.3, **kwargs):
        raise RuntimeError("LLM down")


# --------------------------------------------------------------------------- #
# Tests: build_cross_ticker_context
# --------------------------------------------------------------------------- #


class TestBuildCrossTickerContext:
    """Testet build_cross_ticker_context (C4)."""

    def test_empty_journal_returns_empty(self, tmp_path, monkeypatch):
        """Leeres Journal → leerer String."""
        monkeypatch.chdir(tmp_path)
        result = build_cross_ticker_context("AAPL")
        assert result == ""

    def test_only_same_ticker_returns_empty(self, tmp_path, monkeypatch):
        """Nur Zeilen desselben Tickers → kein Cross-Ticker-Block."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "AAPL",
            "action": "KAUFEN",
            "rating": "KAUFEN",
            "timestamp": _ts(10),
        }])
        result = build_cross_ticker_context("AAPL")
        assert result == ""

    def test_same_ticker_case_insensitive_excluded(self, tmp_path, monkeypatch):
        """Gleicher Ticker in anderer Schreibweise wird ausgeschlossen."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [{
            "ticker": "aapl",  # klein geschrieben
            "action": "KAUFEN",
            "rating": "KAUFEN",
            "timestamp": _ts(10),
        }])
        prices = _make_prices(100, 60, drift=0.01)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return _make_prices(100, 60, drift=0.0)
            return prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_cross_ticker_context("AAPL")
        assert result == ""

    def test_cross_ticker_lines_with_returns(self, tmp_path, monkeypatch):
        """Cross-Ticker-Zeilen mit Returns → Block enthält Ticker, Aktionen, Returns."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            {
                "ticker": "MSFT",
                "action": "KAUFEN",
                "rating": "KAUFEN",
                "timestamp": _ts(20),
            },
            {
                "ticker": "NVDA",
                "action": "VERKAUFEN",
                "rating": "VERKAUFEN",
                "timestamp": _ts(15),
            },
            {
                "ticker": "AAPL",  # eigener Ticker — wird ausgeschlossen
                "action": "HALTEN",
                "rating": "HALTEN",
                "timestamp": _ts(5),
            },
        ])
        msft_prices = _make_prices(200, 60, drift=0.01)   # steigend
        nvda_prices = _make_prices(300, 60, drift=-0.01)  # fallend → VERKAUFEN gewinnt
        spy_prices = _make_prices(450, 60, drift=0.002)
        aapl_prices = _make_prices(100, 60, drift=0.01)

        def mock_load(ticker, *, lookback_days=30):
            return {
                "SPY": spy_prices,
                "MSFT": msft_prices,
                "NVDA": nvda_prices,
                "AAPL": aapl_prices,
            }.get(ticker)

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_cross_ticker_context("AAPL")

        assert result != ""
        assert "LETZTE ENTSCHEIDUNGEN ANDERER TICKER" in result
        assert "Cross-Ticker-Gedächtnis" in result
        assert "MSFT" in result
        assert "NVDA" in result
        assert "KAUFEN" in result
        assert "VERKAUFEN" in result
        assert "Realisierter Return" in result
        assert "Alpha vs SPY" in result
        # Sortierung: NVDA (neuere Entscheidung, vor 15 Tagen) vor MSFT (vor 20)
        assert result.index("NVDA") < result.index("MSFT")
        # Generalisierte Lektion + Übertragungs-Hinweis
        assert "Lektion:" in result
        assert "Lerne aus diesen generalisierten Mustern" in result

    def test_max_lessons_limits_count(self, tmp_path, monkeypatch):
        """max_lessons begrenzt die Anzahl der Lektionen."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            {"ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(30)},
            {"ticker": "NVDA", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(25)},
            {"ticker": "AMD", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(20)},
            {"ticker": "AMZN", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(15)},
        ])
        rising = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.0)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return spy
            return rising

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_cross_ticker_context("AAPL", max_lessons=2)

        # Genau 2 Lektionen (Zeilen beginnen mit "- ")
        lesson_lines = [
            line for line in result.splitlines() if line.startswith("- ")
        ]
        assert len(lesson_lines) == 2
        # Neueste zuerst: AMZN und AMD (nicht MSFT, das am ältesten ist)
        assert "AMZN" in result
        assert "AMD" in result
        assert "MSFT" not in result

    def test_rows_without_return_are_skipped(self, tmp_path, monkeypatch):
        """Zeilen ohne realisierbaren Return werden übersprungen."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            # Keine Kursdaten verfügbar → Return None → übersprungen
            {"ticker": "XYZ", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(25)},
            # Kaputter Timestamp → nicht in cross_rows aufgenommen
            {"ticker": "BROKEN", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": "garbage"},
            # Gültige Cross-Ticker-Zeile mit Kursdaten
            {"ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(10)},
        ])
        msft_prices = _make_prices(200, 60, drift=0.01)
        spy_prices = _make_prices(100, 60, drift=0.0)

        def mock_load(ticker, *, lookback_days=30):
            return {
                "SPY": spy_prices,
                "MSFT": msft_prices,
                # XYZ → None (keine Kursdaten)
            }.get(ticker)

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_cross_ticker_context("AAPL")

        assert result != ""
        assert "MSFT" in result
        assert "XYZ" not in result
        assert "BROKEN" not in result

    def test_no_returns_at_all_returns_empty(self, tmp_path, monkeypatch):
        """Cross-Ticker-Zeilen vorhanden, aber keinerlei Return berechenbar → ""."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            {"ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(20)},
            {"ticker": "NVDA", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(15)},
        ])
        # Keine Kursdaten für irgendwelche Ticker
        with patch("concilium.evaluate._load_price_history", return_value=None):
            result = build_cross_ticker_context("AAPL")
        assert result == ""

    def test_never_raises_broken_journal(self, tmp_path, monkeypatch):
        """Crasht nie bei kaputtem Journal."""
        monkeypatch.chdir(tmp_path)
        # (a) Journal-Verzeichnis mit Binär-Müll
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir(exist_ok=True)
        (journal_dir / "decisions.csv").write_bytes(b"\xff\xfe\x00garbage\xff")
        result = build_cross_ticker_context("AAPL")
        assert result == ""

        # (b) Journal ist ein Verzeichnis statt Datei
        (journal_dir / "decisions.csv").unlink()
        (journal_dir / "decisions.csv").mkdir()
        result = build_cross_ticker_context("AAPL")
        assert result == ""

        # (c) Leerer Ticker-Parameter (Journal bleibt ein Verzeichnis — egal)
        assert build_cross_ticker_context("") == ""
        assert build_cross_ticker_context("   ") == ""

    def test_uses_realised_return_for_row(self, tmp_path, monkeypatch):
        """Verwendet realised_return_for_row (wird gemockt → isoliert vom Netz).

        C6 look-ahead-frei: Der Journal-Timestamp liegt 45 Tage zurück, damit
        das Ausgangsfenster (lookback_days=30) vollständig abgelaufen ist —
        sonst würde die Legacy-Zeile korrekt übersprungen.
        """
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            {"ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN",
             "timestamp": (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d %H:%M:%S")},
        ])

        fake_ts = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d %H:%M:%S")

        def fake_rr(row, lookback_days=30):
            return {
                "ticker": "MSFT",
                "entry_price": 100.0,
                "exit_price": 103.2,
                "raw_return_pct": 3.2,
                "benchmark": "SPY",
                "benchmark_return_pct": 2.1,
                "alpha_pct": 1.1,
                "timestamp": fake_ts,
                "action": "KAUFEN",
            }

        with patch("concilium.feedback.realised_return_for_row", side_effect=fake_rr):
            result = build_cross_ticker_context("AAPL")

        assert result != ""
        assert "MSFT" in result
        assert "+3.20%" in result
        assert "+1.10%" in result

    def test_llm_lesson_used(self, tmp_path, monkeypatch):
        """Bei LLM-Übergabe wird die LLM-Antwort als generalisierte Lektion verwendet."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            {"ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(20)},
        ])
        prices = _make_prices(100, 60, drift=0.01)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return _make_prices(100, 60, drift=0.0)
            return prices

        llm = _MockLLM("Sektorweite Überbewertung wiederholt sich bei Tech-Titeln.")

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_cross_ticker_context("AAPL", llm=llm)

        assert result != ""
        assert "Sektorweite Überbewertung wiederholt sich bei Tech-Titeln." in result
        # LLM wurde genau einmal gerufen (EIN Satz für ALLE Cross-Ticker-Lektionen)
        assert len(llm.captured) == 1
        # Der Prompt enthält die Cross-Ticker-Daten
        prompt_text = llm.captured[0][1]["content"]
        assert "MSFT" in prompt_text
        assert "Cross-Ticker" in prompt_text

    def test_llm_failure_falls_back_to_deterministic(self, tmp_path, monkeypatch):
        """Bei LLM-Fehler → deterministische generalisierte Lektion."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            {"ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(20)},
        ])
        prices = _make_prices(100, 60, drift=0.01)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return _make_prices(100, 60, drift=0.0)
            return prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_cross_ticker_context("AAPL", llm=_FailingLLM())

        assert result != ""
        assert "Lektion:" in result
        assert "LETZTE ENTSCHEIDUNGEN ANDERER TICKER" in result

    def test_deterministic_lesson_positive_negative_neutral(self):
        """Deterministische Cross-Ticker-Lektion reagiert auf Ø-Return."""
        from concilium.feedback import _deterministic_cross_ticker_lesson

        pos = _deterministic_cross_ticker_lesson([
            {"raw_return_pct": 3.0}, {"raw_return_pct": 2.0},
        ])
        assert "positiv" in pos

        neg = _deterministic_cross_ticker_lesson([
            {"raw_return_pct": -3.0}, {"raw_return_pct": -2.0},
        ])
        assert "negativ" in neg

        neu = _deterministic_cross_ticker_lesson([
            {"raw_return_pct": 0.1}, {"raw_return_pct": -0.1},
        ])
        assert "neutral" in neu

        empty = _deterministic_cross_ticker_lesson([])
        assert "Keine aussagekräftigen" in empty


# --------------------------------------------------------------------------- #
# Tests: build_memory_context (Orchestrierer)
# --------------------------------------------------------------------------- #


class TestBuildMemoryContext:
    """Testet build_memory_context (Same-Ticker + Cross-Ticker)."""

    def test_combines_same_and_cross(self, tmp_path, monkeypatch):
        """Kombiniert Same-Ticker-Reflexion und Cross-Ticker-Lektionen."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            {"ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(20)},
            {"ticker": "AAPL", "action": "HALTEN", "rating": "HALTEN", "timestamp": _ts(5)},
        ])
        spy = _make_prices(100, 60, drift=0.0)
        rising = _make_prices(100, 60, drift=0.01)

        def mock_load(ticker, *, lookback_days=30):
            return {"SPY": spy, "MSFT": rising, "AAPL": rising}.get(ticker)

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_memory_context("AAPL")

        assert result != ""
        # Same-Ticker-Teil
        assert "LETZTE ENTSCHEIDUNG ZU AAPL" in result
        # Cross-Ticker-Teil
        assert "LETZTE ENTSCHEIDUNGEN ANDERER TICKER" in result
        assert "MSFT" in result
        # Reihenfolge: Same zuerst, dann Cross
        assert result.index("LETZTE ENTSCHEIDUNG ZU AAPL") < result.index(
            "LETZTE ENTSCHEIDUNGEN ANDERER TICKER"
        )

    def test_only_cross_when_no_same_ticker_rows(self, tmp_path, monkeypatch):
        """Ohne eigene Ticker-Zeilen → nur der Cross-Ticker-Block."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            {"ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(20)},
        ])
        prices = _make_prices(100, 60, drift=0.01)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return _make_prices(100, 60, drift=0.0)
            return prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_memory_context("AAPL")

        assert result != ""
        assert "LETZTE ENTSCHEIDUNG ZU AAPL" not in result
        assert "LETZTE ENTSCHEIDUNGEN ANDERER TICKER" in result

    def test_only_same_when_no_cross_ticker_rows(self, tmp_path, monkeypatch):
        """Nur eigene Ticker-Zeilen → nur die Same-Ticker-Reflexion."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            {"ticker": "AAPL", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(20)},
        ])
        prices = _make_prices(100, 60, drift=0.01)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return _make_prices(100, 60, drift=0.0)
            return prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_memory_context("AAPL")

        assert result != ""
        assert "LETZTE ENTSCHEIDUNG ZU AAPL" in result
        assert "LETZTE ENTSCHEIDUNGEN ANDERER TICKER" not in result

    def test_empty_when_both_empty(self, tmp_path, monkeypatch):
        """Beide Teile leer (leeres Journal) → "".  """
        monkeypatch.chdir(tmp_path)
        result = build_memory_context("AAPL")
        assert result == ""

    def test_max_same_zero_skips_reflection(self, tmp_path, monkeypatch):
        """max_same=0 → nur Cross-Ticker-Block (Reflexion übersprungen)."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            {"ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(20)},
            {"ticker": "AAPL", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(5)},
        ])
        prices = _make_prices(100, 60, drift=0.01)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return _make_prices(100, 60, drift=0.0)
            return prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_memory_context("AAPL", max_same=0)

        assert result != ""
        assert "LETZTE ENTSCHEIDUNG ZU AAPL" not in result
        assert "LETZTE ENTSCHEIDUNGEN ANDERER TICKER" in result

    def test_max_cross_passed_through(self, tmp_path, monkeypatch):
        """max_cross wird an build_cross_ticker_context durchgereicht."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            {"ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(30)},
            {"ticker": "NVDA", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(25)},
            {"ticker": "AMD", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(20)},
        ])
        rising = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.0)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return spy
            return rising

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_memory_context("AAPL", max_cross=2)

        lesson_lines = [
            line for line in result.splitlines()
            if line.startswith("- ") and "LETZTE ENTSCHEIDUNG ZU AAPL" not in line
        ]
        # Same-Ticker-Reflexion hat keine "- "-Zeilen → beide "- "-Zeilen sind Cross
        assert len(lesson_lines) == 2
        assert "AMD" in result and "NVDA" in result and "MSFT" not in result

    def test_never_raises(self, tmp_path, monkeypatch):
        """build_memory_context wirft nie."""
        monkeypatch.chdir(tmp_path)
        assert build_memory_context("AAPL") == ""
        assert build_memory_context("") == ""
        assert build_memory_context("   ") == ""


# --------------------------------------------------------------------------- #
# Tests: Pipeline-Anbindung
# --------------------------------------------------------------------------- #


def _journal_setup(tmp_path, rows: list[dict]) -> None:
    """Schreibt ein Journal unter tmp_path und chdir't dorthin (via monkeypatch)."""
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir(exist_ok=True)
    with open(journal_dir / "decisions.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in JOURNAL_HEADER})


class _PipelineLLM:
    """Minimal-LLM für Pipeline-Tests (rollenspezifische JSON-Antworten)."""

    def __init__(self):
        self.captured: list[list[dict]] = []
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def chat(self, messages, temperature=0.3, **kwargs):
        self.captured.append(messages)
        system = messages[0]["content"]
        if "Fundamental" in system:
            text = json.dumps({"rolle": "F", "stimmung": "bullish", "score": 4, "zusammenfassung": "Gut"})
        elif "technisch" in system:
            text = json.dumps({"rolle": "T", "stimmung": "bullish", "score": 4, "zusammenfassung": "Gut"})
        elif "Sentiment" in system:
            text = json.dumps({"rolle": "S", "stimmung": "neutral", "score": 3, "zusammenfassung": "Ok"})
        elif "Bull" in system:
            text = '{"confidence": 4, "name": "Bull"}\nBull text'
        elif "Bear" in system:
            text = '{"confidence": 3, "name": "Bear"}\nBear text'
        elif "Trader" in system:
            text = json.dumps({"rolle": "Trader", "aktion": "HALTEN", "zielkurs": None,
                               "stop_loss": None, "positionsanteil": 0, "begründung": "Test",
                               "zeithorizont": "Mittelfristig"})
        elif "Risk" in system:
            text = json.dumps({"rolle": "Risk", "risiko_score": 3, "empfehlung": "GENEHMIGT",
                               "auflagen": "keine", "volatilität_bewertung": "moderat",
                               "max_drawdown_schaetzung": "10%", "positionsgröße_empfohlen": "5"})
        elif "Portfolio-Manager" in system:
            text = json.dumps({"rolle": "PM", "entscheidung": "GENEHMIGT", "begründung": "Test", "confidence": 4})
        else:
            text = "{}"
        if kwargs.get("as_structured") and kwargs.get("response_format"):
            from concilium.llm import StructuredChatResult
            return StructuredChatResult(text=text, response_format_used=True)
        return text


_MOCK_DATA = {
    "ticker": "TEST",
    "fundamentals": {"name": "Test", "sector": "X"},
    "technicals": {"current_price": 100.0},
    "sentiment": {}, "news": [], "macro": {}, "peers": [],
    "history": [{"close": 100.0}], "data_warnings": [],
}


class TestPipelineCrossTicker:
    """Testet die Pipeline-Anbindung: Cross-Ticker-Block in den Agenten-Prompts."""

    def test_pipeline_reflection_context_contains_cross_ticker(
        self, tmp_path, monkeypatch
    ):
        """Ungepatcht: _reflection_context = Same + Cross, Both in Trader-Prompt."""
        from concilium.pipeline import run_pipeline

        monkeypatch.chdir(tmp_path)
        _journal_setup(tmp_path, [
            {"ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN",
             "timestamp": (datetime.now() - timedelta(days=55)).strftime("%Y-%m-%d %H:%M:%S")},
            {"ticker": "TEST", "action": "HALTEN", "rating": "HALTEN",
             "timestamp": (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S")},
        ])
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        rising = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.0)

        def mock_load(ticker, *, lookback_days=30):
            return {"SPY": spy, "MSFT": rising, "TEST": rising}.get(ticker)

        llm = _PipelineLLM()

        with patch("concilium.pipeline.collect_ticker_data", return_value=_MOCK_DATA), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_fit_agent", return_value={"portfolio_fit_score": 4}), \
             patch("concilium.evaluate._load_price_history", side_effect=mock_load), \
             patch("concilium.journal.append_decision"):
            result = run_pipeline("TEST", llm=llm, ensemble=False)

        refl = result.get("_reflection_context", "")
        cross = result.get("_cross_ticker_context", "")
        report_refl = result.get("reflection") or ""

        # Kombinierter Kontext: Same + Cross
        assert "LETZTE ENTSCHEIDUNG ZU TEST" in refl
        assert "LETZTE ENTSCHEIDUNGEN ANDERER TICKER" in refl
        assert "MSFT" in refl
        # _cross_ticker_context enthält NUR den Cross-Block
        assert cross.startswith("=== LETZTE ENTSCHEIDUNGEN ANDERER TICKER")
        # result["reflection"] bleibt Ticker-spezifisch (kein Cross-Block)
        assert "LETZTE ENTSCHEIDUNG ZU TEST" in report_refl
        assert "ANDERER TICKER" not in report_refl

        # Cross-Block muss in Trader- und PM-Prompt fließen
        trader_prompts = [
            m for msgs in llm.captured for m in msgs
            if m["role"] == "user" and "Bear-Argumentation" in m["content"]
        ]
        pm_prompts = [
            m for msgs in llm.captured for m in msgs
            if m["role"] == "user" and "Risiko-Bewertung" in m["content"]
        ]
        assert any("LETZTE ENTSCHEIDUNGEN ANDERER TICKER" in p["content"] for p in trader_prompts)
        assert any("LETZTE ENTSCHEIDUNGEN ANDERER TICKER" in p["content"] for p in pm_prompts)

    def test_pipeline_patched_reflection_stays_sole_provider(
        self, tmp_path, monkeypatch
    ):
        """Rückwärtskompatibilität: gepatchte build_reflection_context → wie vor C4.

        Bestehende Pipeline-Tests patchen concilium.pipeline.build_reflection_context;
        dann gilt der Mock als alleiniger Provider und der Cross-Ticker-Teil wird
        NICHT ergänzt (keine zusätzlichen yfinance-Aufrufe, identisches Verhalten).
        """
        from concilium.pipeline import run_pipeline

        monkeypatch.chdir(tmp_path)
        _journal_setup(tmp_path, [
            {"ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN",
             "timestamp": (datetime.now() - timedelta(days=55)).strftime("%Y-%m-%d %H:%M:%S")},
            {"ticker": "TEST", "action": "HALTEN", "rating": "HALTEN",
             "timestamp": (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S")},
        ])
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        llm = _PipelineLLM()

        with patch("concilium.pipeline.collect_ticker_data", return_value=_MOCK_DATA), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_fit_agent", return_value={"portfolio_fit_score": 4}), \
             patch("concilium.pipeline.build_reflection_context", return_value="MOCK-REFLEXION"), \
             patch("concilium.journal.append_decision"):
            result = run_pipeline("TEST", llm=llm, ensemble=False)

        assert result.get("_reflection_context") == "MOCK-REFLEXION"
        assert result.get("reflection") == "MOCK-REFLEXION"
        assert result.get("_cross_ticker_context") == ""

    def test_pipeline_no_journal_no_cross(self, tmp_path, monkeypatch):
        """Ohne Journal → kein Cross-Ticker-Block, kein Reflexions-Block."""
        from concilium.pipeline import run_pipeline

        monkeypatch.chdir(tmp_path)  # kein journal/
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        llm = _PipelineLLM()

        with patch("concilium.pipeline.collect_ticker_data", return_value=_MOCK_DATA), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_fit_agent", return_value={"portfolio_fit_score": 4}), \
             patch("concilium.journal.append_decision"):
            result = run_pipeline("TEST", llm=llm, ensemble=False)

        assert result.get("_reflection_context") == ""
        assert result.get("_cross_ticker_context") == ""
        # reflection bleibt "" (Original-Verhalten: "" oder None, nie Cross-Block)
        assert result.get("reflection") in ("", None)

    def test_pipeline_report_stays_ticker_specific(self, tmp_path, monkeypatch):
        """Der Report-Abschnitt 'Reflexion (Track-Record)' zeigt keine Cross-Ticker-Lektionen."""
        from concilium.pipeline import run_pipeline
        from concilium.report import generate_report

        monkeypatch.chdir(tmp_path)
        _journal_setup(tmp_path, [
            {"ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN",
             "timestamp": (datetime.now() - timedelta(days=55)).strftime("%Y-%m-%d %H:%M:%S")},
            {"ticker": "TEST", "action": "HALTEN", "rating": "HALTEN",
             "timestamp": (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S")},
        ])
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))

        rising = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.0)

        def mock_load(ticker, *, lookback_days=30):
            return {"SPY": spy, "MSFT": rising, "TEST": rising}.get(ticker)

        llm = _PipelineLLM()

        with patch("concilium.pipeline.collect_ticker_data", return_value=_MOCK_DATA), \
             patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]), \
             patch("concilium.pipeline.portfolio_fit_agent", return_value={"portfolio_fit_score": 4}), \
             patch("concilium.evaluate._load_price_history", side_effect=mock_load), \
             patch("concilium.journal.append_decision"):
            result = run_pipeline("TEST", llm=llm, ensemble=False)

        report = generate_report(result)
        assert "Reflexion (Track-Record)" in report
        # Same-Ticker-Reflexion sichtbar, Cross-Ticker-Block NICHT im Report
        assert "LETZTE ENTSCHEIDUNG ZU TEST" in report
        assert "LETZTE ENTSCHEIDUNGEN ANDERER TICKER" not in report


# --------------------------------------------------------------------------- #
# Tests: C6 look-ahead-frei (nur abgelaufene Fenster liefern Lektionen)
# --------------------------------------------------------------------------- #


class TestCrossTickerLookaheadFree:
    """C6: build_cross_ticker_context überspringt nicht-abgelaufene Fenster."""

    def test_unexpired_windows_are_skipped(self, tmp_path, monkeypatch):
        """Cross-Ticker-Zeilen mit laufendem Fenster → keine Lektion (kein Look-ahead)."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            # Abgelaufen (45 Tage) → gültige Lektion
            {"ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(10)},
            # Nicht abgelaufen (3 Tage) → MUSS übersprungen werden
            {"ticker": "NVDA", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(-32)},
        ])
        prices = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.0)

        def mock_load(ticker, *, lookback_days=30):
            return {"SPY": spy, "MSFT": prices, "NVDA": prices}.get(ticker)

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_cross_ticker_context("AAPL")

        assert result != ""
        assert "MSFT" in result
        assert "NVDA" not in result

    def test_resolved_row_uses_persisted_returns(self, tmp_path, monkeypatch):
        """Resolved-Zeile → persistierte Returns werden verwendet (kein RR-Call)."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            {
                "ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN",
                "timestamp": (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d %H:%M:%S"),
                "reflection_status": "resolved",
                "resolved_at": "2026-09-01 10:00:00",
                "realised_return_pct": "+3.20",
                "alpha_pct": "+1.10",
                "lesson": "Persistiert.",
            },
            # Nicht abgelaufene Legacy-Zeile → übersprungen (Look-ahead-Schutz)
            {"ticker": "NVDA", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(-32)},
        ])

        def failing_rr(row, lookback_days=30):
            raise AssertionError("resolved-Zeile darf keinen Return-Lookup machen")

        with patch("concilium.feedback.realised_return_for_row", side_effect=failing_rr):
            result = build_cross_ticker_context("AAPL")

        assert result != ""
        assert "MSFT" in result
        assert "+3.20%" in result
        assert "+1.10%" in result
        assert "NVDA" not in result

    def test_pending_unexpired_row_not_resolved_for_cross(self, tmp_path, monkeypatch):
        """Pending-Zeile mit laufendem Fenster → kein Return-Lookup (crascht nie)."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            # 3 Tage alt → Fenster läuft; NVDA hätte Kursdaten, darf aber
            # NICHT abgefragt werden (kein Look-ahead).
            {"ticker": "NVDA", "action": "KAUFEN", "rating": "KAUFEN", "timestamp": _ts(-32)},
        ])
        prices = _make_prices(100, 60, drift=0.01)
        spy = _make_prices(100, 60, drift=0.0)

        def mock_load(ticker, *, lookback_days=30):
            return {"SPY": spy, "NVDA": prices}.get(ticker)

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_cross_ticker_context("AAPL")

        assert result == ""  # nichts aufnehmbar

    def test_mixed_resolved_and_expired_legacy(self, tmp_path, monkeypatch):
        """Resolved + abgelaufene Legacy-Zeilen kombiniert → beide aufgenommen."""
        monkeypatch.chdir(tmp_path)
        _write_journal(tmp_path, [
            {
                "ticker": "MSFT", "action": "KAUFEN", "rating": "KAUFEN",
                "timestamp": (datetime.now() - timedelta(days=50)).strftime("%Y-%m-%d %H:%M:%S"),
                "reflection_status": "resolved",
                "resolved_at": "2026-09-01 10:00:00",
                "realised_return_pct": "+2.00",
                "alpha_pct": "+0.50",
            },
            # Legacy-Zeile (Status "") mit abgelaufenem Fenster
            {"ticker": "NVDA", "action": "VERKAUFEN", "rating": "VERKAUFEN", "timestamp": _ts(5)},
        ])
        nvda_prices = _make_prices(300, 60, drift=-0.01)  # fallend → VERKAUFEN gewinnt
        spy = _make_prices(100, 60, drift=0.0)

        def mock_load(ticker, *, lookback_days=30):
            return {"SPY": spy, "NVDA": nvda_prices}.get(ticker)

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_cross_ticker_context("AAPL")

        assert result != ""
        assert "MSFT" in result
        assert "+2.00%" in result
        assert "NVDA" in result
        assert "VERKAUFEN" in result
        # Sortierung: NVDA (neuer) vor MSFT (älter)
        assert result.index("NVDA") < result.index("MSFT")
