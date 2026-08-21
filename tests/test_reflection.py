"""Tests für Feature 2: Reflexion + realisierter Return (alpha vs SPY).

Testet:
- realised_return_for_row mit synthetischen Preisdaten (mocked)
- build_reflection_context: "" bei keinem Treffer, nicht-leer bei Treffer
- trader() appendet reflection_context zum Prompt
- portfolio_manager() appendet reflection_context zum Prompt
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.journal import JOURNAL_HEADER  # noqa: E402

# --------------------------------------------------------------------------- #
# Helper: synthetische Preisdaten
# --------------------------------------------------------------------------- #


def _make_prices(start_price: float, n_days: int, drift: float = 0.0) -> list[dict]:
    """Erzeugt eine Liste von Preis-Dicts für n_days Tage ab vor 60 Tagen."""
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
    """Schreibt eine Journal-CSV-Datei und gibt den Pfad zurück."""
    path = str(tmp_path / "decisions.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
        writer.writeheader()
        for row in rows:
            full_row = {k: row.get(k, "") for k in JOURNAL_HEADER}
            writer.writerow(full_row)
    return path


import csv  # noqa: E402 — needed by _write_journal

# --------------------------------------------------------------------------- #
# Tests: realised_return_for_row
# --------------------------------------------------------------------------- #


class TestRealisedReturn:
    """Testet realised_return_for_row mit gemockten Preisdaten."""

    def test_returns_none_on_empty_ticker(self):
        """Leerer Ticker → None."""
        from concilium.evaluate import realised_return_for_row

        row = {"ticker": "", "timestamp": "2026-01-01 10:00:00", "action": "KAUFEN"}
        assert realised_return_for_row(row) is None

    def test_returns_none_on_no_timestamp(self):
        """Kein Timestamp → None."""
        from concilium.evaluate import realised_return_for_row

        row = {"ticker": "AAPL", "timestamp": "", "action": "KAUFEN"}
        assert realised_return_for_row(row) is None

    def test_returns_none_on_no_prices(self):
        """Keine Preisdaten → None."""
        from concilium.evaluate import realised_return_for_row

        row = {"ticker": "AAPL", "timestamp": "2026-01-01 10:00:00", "action": "KAUFEN"}
        with patch("concilium.evaluate._load_price_history", return_value=None):
            assert realised_return_for_row(row) is None

    def test_kaufen_positive_return(self):
        """KAUFEN mit steigenden Kursen → positiver raw_return."""
        from concilium.evaluate import realised_return_for_row

        prices = _make_prices(100, 60, drift=0.01)  # steigend
        spy_prices = _make_prices(100, 60, drift=0.005)  # SPY steigt langsamer

        row = {
            "ticker": "AAPL",
            "timestamp": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),
            "action": "KAUFEN",
        }

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return spy_prices
            return prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = realised_return_for_row(row, lookback_days=30)

        assert result is not None
        assert result["raw_return_pct"] > 0
        assert result["spy_return_pct"] is not None
        assert result["alpha_pct"] is not None
        assert result["alpha_pct"] == result["raw_return_pct"] - result["spy_return_pct"]

    def test_verkaufen_inverted_return(self):
        """VERKAUFEN mit steigenden Kursen → negativer raw_return (invertiert)."""
        from concilium.evaluate import realised_return_for_row

        prices = _make_prices(100, 60, drift=0.01)  # steigend → schlecht für Verkauf

        row = {
            "ticker": "AAPL",
            "timestamp": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),
            "action": "VERKAUFEN",
        }

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return _make_prices(100, 60, drift=0.0)  # flat
            return prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = realised_return_for_row(row, lookback_days=30)

        assert result is not None
        # Bei steigendem Kurs ist price_change positiv → VERKAUFEN invertiert → negativ
        assert result["raw_return_pct"] < 0

    def test_spy_none_when_spy_fails(self):
        """Wenn SPY-Daten nicht ladbar → spy_return_pct=None, alpha_pct=None."""
        from concilium.evaluate import realised_return_for_row

        prices = _make_prices(100, 60, drift=0.01)

        row = {
            "ticker": "AAPL",
            "timestamp": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),
            "action": "KAUFEN",
        }

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return None
            return prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = realised_return_for_row(row, lookback_days=30)

        assert result is not None
        assert result["raw_return_pct"] is not None
        assert result["spy_return_pct"] is None
        assert result["alpha_pct"] is None

    def test_never_raises(self):
        """realised_return_for_row wirft nie eine Exception."""
        from concilium.evaluate import realised_return_for_row

        # Verschiedene ungültige Eingaben
        assert realised_return_for_row({}) is None
        assert realised_return_for_row({"ticker": "AAPL"}) is None
        assert realised_return_for_row(None) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Tests: build_reflection_context
# --------------------------------------------------------------------------- #


class TestBuildReflectionContext:
    """Testet build_reflection_context."""

    def test_no_journal_returns_empty(self, tmp_path, monkeypatch):
        """Kein Journal → leerer String."""
        from concilium.feedback import build_reflection_context

        monkeypatch.chdir(tmp_path)
        result = build_reflection_context("AAPL")
        assert result == ""

    def test_no_matching_ticker_returns_empty(self, tmp_path, monkeypatch):
        """Ticker nicht im Journal → leerer String."""
        from concilium.feedback import build_reflection_context

        monkeypatch.chdir(tmp_path)
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir(exist_ok=True)

        rows = [{
            "ticker": "MSFT",
            "action": "KAUFEN",
            "rating": "KAUFEN",
            "timestamp": "2026-01-01 10:00:00",
        }]
        _write_journal(tmp_path, rows)
        import shutil

        shutil.copy(str(tmp_path / "decisions.csv"), str(journal_dir / "decisions.csv"))

        result = build_reflection_context("AAPL")
        assert result == ""

    def test_matching_ticker_returns_content(self, tmp_path, monkeypatch):
        """Ticker im Journal + gültige Preisdaten → nicht-leerer String."""
        from concilium.feedback import build_reflection_context

        monkeypatch.chdir(tmp_path)
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir(exist_ok=True)

        rows = [{
            "ticker": "AAPL",
            "action": "KAUFEN",
            "rating": "STARK KAUFEN",
            "timestamp": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),
        }]
        _write_journal(tmp_path, rows)
        import shutil

        shutil.copy(str(tmp_path / "decisions.csv"), str(journal_dir / "decisions.csv"))

        prices = _make_prices(100, 60, drift=0.01)
        spy_prices = _make_prices(100, 60, drift=0.005)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return spy_prices
            return prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_reflection_context("AAPL")

        assert result != ""
        assert "LETZTE ENTSCHEIDUNG" in result
        assert "AAPL" in result
        assert "Realisierter Return" in result
        assert "Alpha vs SPY" in result

    def test_deterministic_lesson_positive(self):
        """Deterministische Lektion bei positivem Return."""
        from concilium.feedback import _deterministic_lesson

        lesson = _deterministic_lesson(5.0, "KAUFEN")
        assert "bestätigt" in lesson.lower() or "behalte" in lesson.lower()

    def test_deterministic_lesson_negative(self):
        """Deterministische Lektion bei negativem Return."""
        from concilium.feedback import _deterministic_lesson

        lesson = _deterministic_lesson(-5.0, "KAUFEN")
        assert "gegen dich" in lesson.lower() or "timing" in lesson.lower()

    def test_deterministic_lesson_neutral(self):
        """Deterministische Lektion bei neutralem Return."""
        from concilium.feedback import _deterministic_lesson

        lesson = _deterministic_lesson(0.0, "HALTEN")
        assert "neutral" in lesson.lower() or "justiere" in lesson.lower()

    def test_llm_lesson_used_when_llm_given(self, tmp_path, monkeypatch):
        """Bei LLM-Übergabe wird die LLM-Antwort als Lektion verwendet."""
        from concilium.feedback import build_reflection_context

        monkeypatch.chdir(tmp_path)
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir(exist_ok=True)

        rows = [{
            "ticker": "AAPL",
            "action": "KAUFEN",
            "rating": "STARK KAUFEN",
            "timestamp": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),
        }]
        _write_journal(tmp_path, rows)
        import shutil

        shutil.copy(str(tmp_path / "decisions.csv"), str(journal_dir / "decisions.csv"))

        prices = _make_prices(100, 60, drift=0.01)
        spy_prices = _make_prices(100, 60, drift=0.005)

        class _MockLLM:
            def chat(self, messages, temperature=0.3, **kwargs):
                # Return a lesson
                return "Lerne: Verlasse dich auf fundamentale Daten."

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return spy_prices
            return prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_reflection_context("AAPL", llm=_MockLLM())

        assert result != ""
        assert "Verlasse dich auf fundamentale Daten" in result

    def test_llm_failure_falls_back_to_deterministic(self, tmp_path, monkeypatch):
        """Bei LLM-Fehler → deterministische Lektion."""
        from concilium.feedback import build_reflection_context

        monkeypatch.chdir(tmp_path)
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir(exist_ok=True)

        rows = [{
            "ticker": "AAPL",
            "action": "KAUFEN",
            "rating": "STARK KAUFEN",
            "timestamp": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),
        }]
        _write_journal(tmp_path, rows)
        import shutil

        shutil.copy(str(tmp_path / "decisions.csv"), str(journal_dir / "decisions.csv"))

        prices = _make_prices(100, 60, drift=0.01)
        spy_prices = _make_prices(100, 60, drift=0.005)

        class _FailingLLM:
            def chat(self, messages, temperature=0.3, **kwargs):
                raise RuntimeError("LLM down")

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return spy_prices
            return prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_reflection_context("AAPL", llm=_FailingLLM())

        assert result != ""
        assert "Lerne daraus" in result

    def test_realised_return_none_returns_empty(self, tmp_path, monkeypatch):
        """Wenn realised_return None → leerer String."""
        from concilium.feedback import build_reflection_context

        monkeypatch.chdir(tmp_path)
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir(exist_ok=True)

        rows = [{
            "ticker": "AAPL",
            "action": "KAUFEN",
            "rating": "KAUFEN",
            "timestamp": "2026-01-01 10:00:00",
        }]
        _write_journal(tmp_path, rows)
        import shutil

        shutil.copy(str(tmp_path / "decisions.csv"), str(journal_dir / "decisions.csv"))

        with patch("concilium.evaluate._load_price_history", return_value=None):
            result = build_reflection_context("AAPL")

        assert result == ""

    def test_case_insensitive_ticker(self, tmp_path, monkeypatch):
        """Ticker-Matching ist case-insensitive."""
        from concilium.feedback import build_reflection_context

        monkeypatch.chdir(tmp_path)
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir(exist_ok=True)

        rows = [{
            "ticker": "aapl",
            "action": "KAUFEN",
            "rating": "KAUFEN",
            "timestamp": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),
        }]
        _write_journal(tmp_path, rows)
        import shutil

        shutil.copy(str(tmp_path / "decisions.csv"), str(journal_dir / "decisions.csv"))

        prices = _make_prices(100, 60, drift=0.01)

        def mock_load(ticker, *, lookback_days=30):
            if ticker == "SPY":
                return _make_prices(100, 60, drift=0.005)
            return prices

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = build_reflection_context("AAPL")

        assert result != ""
        assert "AAPL" in result

    def test_never_raises(self, tmp_path, monkeypatch):
        """build_reflection_context wirft nie."""
        from concilium.feedback import build_reflection_context

        monkeypatch.chdir(tmp_path)
        # Kein journal/ Verzeichnis
        assert build_reflection_context("AAPL") == ""
        assert build_reflection_context("") == ""


# --------------------------------------------------------------------------- #
# Tests: trader/portfolio_manager appendet reflection_context
# --------------------------------------------------------------------------- #


class _CapturingLLM:
    """Mock-LLM, der die übergebenen messages speichert und JSON zurückgibt."""

    def __init__(self, response: str = '{"aktion": "HALTEN"}'):
        self._response = response
        self.captured_messages: list[list[dict]] = []

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3, **kwargs) -> str:
        self.captured_messages.append(messages)
        return self._response


class TestReflectionInPrompt:
    """Testet dass reflection_context in den Agenten-Prompts auftaucht."""

    _REFLECTION = (
        "=== DEINE LETZTE ENTSCHEIDUNG ZU AAPL (2026-01-01 10:00:00) ===\n"
        "Aktion: KAUFEN | Realisierter Return: +5.00% | Alpha vs SPY: +2.00%\n"
        "Lerne daraus: Die Marktlage hat deine Einschätzung bestätigt."
    )

    def test_trader_appends_reflection(self):
        """trader() hängt reflection_context an den Prompt an."""
        llm = _CapturingLLM(json.dumps({
            "rolle": "Trader", "aktion": "KAUFEN", "zielkurs": 120,
            "stop_loss": 90, "positionsanteil": 5, "begründung": "Test",
            "zeithorizont": "Mittelfristig",
        }))
        analysts = {
            "fundamental": {"stimmung": "bullish", "score": 4, "_raw": "Gut"},
            "technical": {"stimmung": "bullish", "score": 4, "_raw": "Gut"},
            "sentiment": {"stimmung": "neutral", "score": 3, "_raw": "Neutral"},
            "technicals": {"current_price": 100.0},
        }
        debate = {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}}

        trader(analysts, debate, llm, reflection_context=self._REFLECTION)

        user_content = llm.captured_messages[0][1]["content"]
        assert "LETZTE ENTSCHEIDUNG" in user_content
        assert "Realisierter Return" in user_content

    def test_trader_empty_reflection_no_context(self):
        """trader() mit reflection_context='' → kein Reflexions-Block."""
        llm = _CapturingLLM(json.dumps({
            "rolle": "Trader", "aktion": "HALTEN",
            "zielkurs": None, "stop_loss": None,
            "positionsanteil": 0, "begründung": "Test",
            "zeithorizont": "Mittelfristig",
        }))
        analysts = {
            "fundamental": {"stimmung": "neutral", "score": 3, "_raw": "Ok"},
            "technical": {"stimmung": "neutral", "score": 3, "_raw": "Ok"},
            "sentiment": {"stimmung": "neutral", "score": 3, "_raw": "Neutral"},
            "technicals": {"current_price": 100.0},
        }
        debate = {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}}

        trader(analysts, debate, llm, reflection_context="")

        user_content = llm.captured_messages[0][1]["content"]
        assert "LETZTE ENTSCHEIDUNG" not in user_content

    def test_portfolio_manager_appends_reflection(self):
        """portfolio_manager() hängt reflection_context an den Prompt an."""
        from concilium.agents import portfolio_manager

        llm = _CapturingLLM(json.dumps({
            "entscheidung": "GENEHMIGT", "confidence": 4,
        }))
        trade = {"aktion": "KAUFEN", "rating": "STARK KAUFEN"}
        risk = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}

        portfolio_manager(trade, risk, llm, reflection_context=self._REFLECTION)

        user_content = llm.captured_messages[0][1]["content"]
        assert "LETZTE ENTSCHEIDUNG" in user_content

    def test_portfolio_manager_empty_reflection_no_context(self):
        """portfolio_manager() mit reflection_context='' → kein Reflexions-Block."""
        from concilium.agents import portfolio_manager

        llm = _CapturingLLM(json.dumps({
            "entscheidung": "ABGELEHNT", "confidence": 2,
        }))
        trade = {"aktion": "HALTEN"}
        risk = {"risiko_score": 3, "empfehlung": "ABGELEHNT"}

        portfolio_manager(trade, risk, llm, reflection_context="")

        user_content = llm.captured_messages[0][1]["content"]
        assert "LETZTE ENTSCHEIDUNG" not in user_content

    def test_reflection_after_feedback(self):
        """reflection_context wird NACH feedback_context im Prompt eingefügt."""
        from concilium.agents import portfolio_manager

        llm = _CapturingLLM(json.dumps({
            "entscheidung": "GENEHMIGT", "confidence": 4,
        }))
        trade = {"aktion": "KAUFEN"}
        risk = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}
        feedback = "=== DEIN TRACK-RECORD ===\nGesamt: 10 Entscheidungen"

        portfolio_manager(
            trade, risk, llm,
            feedback_context=feedback,
            reflection_context=self._REFLECTION,
        )

        user_content = llm.captured_messages[0][1]["content"]
        fb_pos = user_content.find("TRACK-RECORD")
        ref_pos = user_content.find("LETZTE ENTSCHEIDUNG")
        assert fb_pos != -1
        assert ref_pos != -1
        assert fb_pos < ref_pos


# --------------------------------------------------------------------------- #
# Tests: report mit Reflexion
# --------------------------------------------------------------------------- #


class TestReportReflection:
    """Testet dass der Report den Reflexions-Block anzeigt."""

    def test_report_shows_reflection(self):
        """Report enthält Reflexion-Abschnitt wenn reflection gesetzt."""
        from concilium.report import generate_report

        result = {
            "ticker": "AAPL",
            "no_llm": False,
            "data": {
                "fundamentals": {"name": "Apple", "sector": "Tech"},
                "technicals": {"current_price": 150},
                "sentiment": {},
            },
            "analysts": {
                "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
                "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
                "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
            },
            "debate": {
                "bull": {"_raw": "Bull"},
                "bear": {"_raw": "Bear"},
            },
            "trade": {
                "aktion": "KAUFEN",
                "rating": "STARK KAUFEN",
                "zielkurs": 180,
                "stop_loss": 130,
                "positionsanteil": 7,
                "begründung": "Test",
                "zeithorizont": "Mittelfristig",
            },
            "risk": {"risiko_score": 3, "empfehlung": "GENEHMIGT"},
            "final": {"entscheidung": "GENEHMIGT", "confidence": 4},
            "reflection": (
                "=== DEINE LETZTE ENTSCHEIDUNG ZU AAPL (2026-01-01) ===\n"
                "Aktion: KAUFEN | Realisierter Return: +5.00% | Alpha vs SPY: +2.00%\n"
                "Lerne daraus: Guter Call."
            ),
        }

        report = generate_report(result)

        assert "Reflexion (Track-Record)" in report
        assert "LETZTE ENTSCHEIDUNG" in report
        assert "Realisierter Return" in report

    def test_report_no_reflection(self):
        """Ohne reflection → kein Reflexion-Abschnitt."""
        from concilium.report import generate_report

        result = {
            "ticker": "AAPL",
            "no_llm": False,
            "data": {
                "fundamentals": {"name": "Apple", "sector": "Tech"},
                "technicals": {"current_price": 150},
                "sentiment": {},
            },
            "analysts": {
                "fundamental": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
                "technical": {"stimmung": "bullish", "score": 4, "zusammenfassung": "Gut", "_raw": ""},
                "sentiment": {"stimmung": "neutral", "score": 3, "zusammenfassung": "Ok", "_raw": ""},
            },
            "debate": {
                "bull": {"_raw": "Bull"},
                "bear": {"_raw": "Bear"},
            },
            "trade": {
                "aktion": "HALTEN",
                "zielkurs": None,
                "stop_loss": None,
                "positionsanteil": 0,
                "begründung": "Neutral",
                "zeithorizont": "Mittelfristig",
            },
            "risk": {"risiko_score": 3, "empfehlung": "GENEHMIGT"},
            "final": {"entscheidung": "ABGELEHNT", "confidence": 2},
            "reflection": None,
        }

        report = generate_report(result)

        assert "Reflexion (Track-Record)" not in report


# --------------------------------------------------------------------------- #
# Helper import
# --------------------------------------------------------------------------- #


# Import trader at module level for the test classes
from concilium.agents import trader  # noqa: E402
