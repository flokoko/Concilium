"""Tests für feedback.py — Kontext-Feedback (Track-Record in Agenten-Prompts).

Alle Tests sind OFFLINE-fähig: kein yfinance, kein Netzwerk.
Der LLMClient wird gemockt, wo Agenten-Funktionen getestet werden.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from unittest.mock import patch

# src zum Pfad hinzufügen
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import portfolio_manager, risk_manager, trader  # noqa: E402
from concilium.feedback import build_feedback_context  # noqa: E402
from concilium.journal import JOURNAL_HEADER  # noqa: E402

# --------------------------------------------------------------------------- #
# Hilfsfunktionen: Journal-CSV erstellen
# --------------------------------------------------------------------------- #


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


def _make_row(
    ticker: str = "AAPL",
    action: str = "KAUFEN",
    final_decision: str = "GENEHMIGT",
    confidence: str = "4",
    ensemble_confidence: str = "",
    portfolio_fit_score: str = "",
    ziel_gewichtung_pct: str = "",
    timestamp: str = "2026-01-01 10:00:00",
) -> dict:
    """Erzeugt eine Journal-Zeile mit Defaults."""
    return {
        "ticker": ticker,
        "action": action,
        "final_decision": final_decision,
        "confidence": confidence,
        "ensemble_confidence": ensemble_confidence,
        "portfolio_fit_score": portfolio_fit_score,
        "ziel_gewichtung_pct": ziel_gewichtung_pct,
        "timestamp": timestamp,
    }


# --------------------------------------------------------------------------- #
# Tests: build_feedback_context — leere/fehlende Datei
# --------------------------------------------------------------------------- #


class TestEmptyAndMissing:
    """Testet leere und fehlende Journal-Dateien."""

    def test_missing_file_returns_empty(self, tmp_path):
        """Fehlende Datei → leerer String, kein Crash."""
        result = build_feedback_context(str(tmp_path / "nichtexistent.csv"))
        assert result == ""

    def test_empty_file_returns_empty(self, tmp_path):
        """Leere CSV (nur Header) → leerer String."""
        path = _write_journal(tmp_path, [])
        result = build_feedback_context(path)
        assert result == ""


# --------------------------------------------------------------------------- #
# Tests: build_feedback_context — zu wenige Entscheidungen
# --------------------------------------------------------------------------- #


class TestTooFewDecisions:
    """Testet dass bei < min_decisions ein leerer String zurückgegeben wird."""

    def test_three_decisions_returns_empty(self, tmp_path):
        """3 Entscheidungen < min_decisions=5 → leerer String."""
        rows = [_make_row(ticker=f"T{i}") for i in range(3)]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)
        assert result == ""

    def test_four_decisions_returns_empty(self, tmp_path):
        """4 Entscheidungen < min_decisions=5 → leerer String."""
        rows = [_make_row(ticker=f"T{i}") for i in range(4)]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)
        assert result == ""

    def test_custom_min_decisions(self, tmp_path):
        """min_decisions=10 → 7 Entscheidungen reichen nicht."""
        rows = [_make_row(ticker=f"T{i}") for i in range(7)]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path, min_decisions=10)
        assert result == ""

    def test_exact_min_decisions_returns_content(self, tmp_path):
        """Genau min_decisions=5 Entscheidungen → Kontext-Block (nicht leer)."""
        rows = [_make_row(ticker=f"T{i}") for i in range(5)]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)
        assert result != ""
        assert "TRACK-RECORD" in result


# --------------------------------------------------------------------------- #
# Tests: build_feedback_context — Statistiken im Kontext-Block
# --------------------------------------------------------------------------- #


class TestFeedbackContent:
    """Testet dass der Kontext-Block die erwarteten Statistiken enthält."""

    def test_contains_total_and_actions(self, tmp_path):
        """Kontext enthält Gesamtanzahl und Aktionen-Verteilung."""
        rows = [
            _make_row(ticker="A", action="KAUFEN"),
            _make_row(ticker="B", action="KAUFEN"),
            _make_row(ticker="C", action="HALTEN"),
            _make_row(ticker="D", action="VERKAUFEN"),
            _make_row(ticker="E", action="KAUFEN"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        assert "Gesamt: 5 Entscheidungen" in result
        assert "KAUFEN: 3" in result
        assert "HALTEN: 1" in result
        assert "VERKAUFEN: 1" in result

    def test_contains_final_decisions(self, tmp_path):
        """Kontext enthält GENEHMIGT/ABGELEHNT-Verteilung."""
        rows = [
            _make_row(ticker="A", final_decision="GENEHMIGT"),
            _make_row(ticker="B", final_decision="GENEHMIGT"),
            _make_row(ticker="C", final_decision="ABGELEHNT"),
            _make_row(ticker="D", final_decision="GENEHMIGT"),
            _make_row(ticker="E", final_decision="ABGELEHNT"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        assert "GENEHMIGT: 3" in result
        assert "ABGELEHNT: 2" in result

    def test_contains_confidence(self, tmp_path):
        """Kontext enthält durchschnittliche Confidence."""
        rows = [
            _make_row(ticker="A", confidence="3"),
            _make_row(ticker="B", confidence="4"),
            _make_row(ticker="C", confidence="5"),
            _make_row(ticker="D", confidence="4"),
            _make_row(ticker="E", confidence="4"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        # Ø = (3+4+5+4+4)/5 = 4.00
        assert "Ø Confidence: 4.00 / 5" in result

    def test_contains_ensemble_confidence(self, tmp_path):
        """Kontext enthält durchschnittliche Ensemble-Confidence."""
        rows = [
            _make_row(ticker="A", ensemble_confidence="0.80"),
            _make_row(ticker="B", ensemble_confidence="0.60"),
            _make_row(ticker="C", ensemble_confidence="1.00"),
            _make_row(ticker="D", ensemble_confidence="0.67"),
            _make_row(ticker="E", ensemble_confidence="0.50"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        # Ø = (0.80+0.60+1.00+0.67+0.50)/5 = 0.714
        assert "Ø Ensemble-Confidence: 0.71" in result

    def test_contains_portfolio_fit(self, tmp_path):
        """Kontext enthält durchschnittlichen Portfolio-Fit-Score."""
        rows = [
            _make_row(ticker="A", portfolio_fit_score="4"),
            _make_row(ticker="B", portfolio_fit_score="5"),
            _make_row(ticker="C", portfolio_fit_score="3"),
            _make_row(ticker="D", portfolio_fit_score="4"),
            _make_row(ticker="E", portfolio_fit_score="4"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        # Ø = (4+5+3+4+4)/5 = 4.00
        assert "Ø Portfolio-Fit-Score: 4.00 / 5" in result

    def test_contains_ziel_gewichtung(self, tmp_path):
        """Kontext enthält durchschnittliche Ziel-Gewichtung."""
        rows = [
            _make_row(ticker="A", ziel_gewichtung_pct="5"),
            _make_row(ticker="B", ziel_gewichtung_pct="10"),
            _make_row(ticker="C", ziel_gewichtung_pct="3"),
            _make_row(ticker="D", ziel_gewichtung_pct="7"),
            _make_row(ticker="E", ziel_gewichtung_pct="5"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        # Ø = (5+10+3+7+5)/5 = 6.0
        assert "Ø Ziel-Gewichtung: 6.0 %" in result

    def test_contains_kaufen_genehmigt_pct(self, tmp_path):
        """Kontext enthält Anteil KAUFEN-Empfehlungen die final genehmigt wurden."""
        rows = [
            _make_row(ticker="A", action="KAUFEN", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="KAUFEN", final_decision="GENEHMIGT"),
            _make_row(ticker="C", action="KAUFEN", final_decision="ABGELEHNT"),
            _make_row(ticker="D", action="HALTEN", final_decision="GENEHMIGT"),
            _make_row(ticker="E", action="KAUFEN", final_decision="GENEHMIGT"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        # 4 KAUFEN, 3 genehmigt → 75.0%
        assert "KAUFEN-Empfehlungen final genehmigt: 75.0 %" in result

    def test_contains_neutral_reflection_prompt(self, tmp_path):
        """Kontext enthält neutrale Selbstreflections-Aufforderung."""
        rows = [_make_row(ticker=f"T{i}") for i in range(5)]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        assert "kalibriere" in result.lower()
        assert "sachlich" in result.lower()
        # KEINE wertenden Anweisungen
        assert "zu optimistisch" not in result.lower()
        assert "zu pessimistisch" not in result.lower()

    def test_missing_optional_fields_show_na(self, tmp_path):
        """Fehlende optionale Felder (portfolio_fit, ziel_gewichtung) → N/A."""
        rows = [_make_row(ticker=f"T{i}") for i in range(5)]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)

        # portfolio_fit_score und ziel_gewichtung_pct sind leer → N/A
        assert "N/A" in result


# --------------------------------------------------------------------------- #
# Tests: build_feedback_context — Robustheit
# --------------------------------------------------------------------------- #


class TestFeedbackRobustness:
    """Testet dass build_feedback_context niemals crasht."""

    def test_corrupt_csv_returns_empty(self, tmp_path):
        """Korrupte CSV-Datei → leerer String, kein Crash."""
        path = str(tmp_path / "corrupt.csv")
        with open(path, "w") as fh:
            fh.write("garbage,no,valid,csv\n{invalid")
        result = build_feedback_context(path)
        assert result == ""

    def test_no_crash_on_none_file(self):
        """None als journal_file (Default-Pfad) crasht nicht."""
        # Default-Pfad journal/decisions.csv — wahrscheinlich nicht vorhanden
        # im Test-Cwd. Sollte leeren String zurückgeben.
        result = build_feedback_context()
        assert isinstance(result, str)


# --------------------------------------------------------------------------- #
# Tests: Rückwärtskompatibilität — Agenten mit leerem feedback_context
# --------------------------------------------------------------------------- #


class _CapturingLLM:
    """Mock-LLM, der die übergebenen messages speichert und JSON zurückgibt."""

    def __init__(self, response: str = '{"aktion": "HALTEN"}'):
        self._response = response
        self.captured_messages: list[list[dict]] = []

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3, **kwargs) -> str:
        self.captured_messages.append(messages)
        return self._response


class TestBackwardCompatibility:
    """Testet dass Agenten mit leerem feedback_context sich wie vorher verhalten."""

    def test_trader_empty_feedback_no_context_in_prompt(self):
        """trader mit feedback_context='' → kein TRACK-RECORD im Prompt."""
        llm = _CapturingLLM('{"aktion": "HALTEN"}')
        analysts = {
            "fundamental": {"stimmung": "bullish", "score": 4, "_raw": "Gut"},
            "technical": {"stimmung": "bullish", "score": 4, "_raw": "Gut"},
            "sentiment": {"stimmung": "neutral", "score": 3, "_raw": "Neutral"},
            "technicals": {"current_price": 100.0},
        }
        debate = {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}}

        trader(analysts, debate, llm, feedback_context="")

        user_content = llm.captured_messages[0][1]["content"]
        assert "TRACK-RECORD" not in user_content

    def test_risk_manager_empty_feedback_no_context_in_prompt(self):
        """risk_manager mit feedback_context='' → kein TRACK-RECORD im Prompt."""
        llm = _CapturingLLM('{"risiko_score": 3, "empfehlung": "GENEHMIGT"}')
        trade = {"aktion": "KAUFEN", "zielkurs": 110}
        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {"current_price": 100.0},
            "sentiment": {},
        }

        risk_manager(trade, data, llm, data_text="dummy data", feedback_context="")

        user_content = llm.captured_messages[0][1]["content"]
        assert "TRACK-RECORD" not in user_content

    def test_portfolio_manager_empty_feedback_no_context_in_prompt(self):
        """portfolio_manager mit feedback_context='' → kein TRACK-RECORD im Prompt."""
        llm = _CapturingLLM('{"entscheidung": "GENEHMIGT", "confidence": 4}')
        trade = {"aktion": "KAUFEN"}
        risk = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}

        portfolio_manager(trade, risk, llm, feedback_context="")

        user_content = llm.captured_messages[0][1]["content"]
        assert "TRACK-RECORD" not in user_content

    def test_trader_no_feedback_param_defaults_empty(self):
        """trader ohne feedback_context-Parameter → wie vorher (kein Kontext)."""
        llm = _CapturingLLM('{"aktion": "HALTEN"}')
        analysts = {
            "fundamental": {"stimmung": "bullish", "score": 4, "_raw": "Gut"},
            "technical": {"stimmung": "bullish", "score": 4, "_raw": "Gut"},
            "sentiment": {"stimmung": "neutral", "score": 3, "_raw": "Neutral"},
            "technicals": {"current_price": 100.0},
        }
        debate = {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}}

        # Ohne feedback_context kwarg — Default ist ""
        trader(analysts, debate, llm)

        user_content = llm.captured_messages[0][1]["content"]
        assert "TRACK-RECORD" not in user_content


# --------------------------------------------------------------------------- #
# Tests: feedback_context erscheint im Prompt
# --------------------------------------------------------------------------- #


class TestFeedbackInPrompt:
    """Testet dass der feedback_context-Block im User-Prompt auftaucht."""

    _FEEDBACK = (
        "=== DEIN TRACK-RECORD (letzte 10 Entscheidungen) ===\n"
        "Gesamt: 10 Entscheidungen (KAUFEN: 6, HALTEN: 2, VERKAUFEN: 2)\n"
        "Berücksichtige diese Historie bei deiner Einschätzung."
    )

    def test_trader_with_feedback_shows_in_prompt(self):
        """trader mit feedback_context → TRACK-RECORD im Prompt."""
        llm = _CapturingLLM('{"aktion": "KAUFEN"}')
        analysts = {
            "fundamental": {"stimmung": "bullish", "score": 4, "_raw": "Gut"},
            "technical": {"stimmung": "bullish", "score": 4, "_raw": "Gut"},
            "sentiment": {"stimmung": "neutral", "score": 3, "_raw": "Neutral"},
            "technicals": {"current_price": 100.0},
        }
        debate = {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}}

        trader(analysts, debate, llm, feedback_context=self._FEEDBACK)

        user_content = llm.captured_messages[0][1]["content"]
        assert "TRACK-RECORD" in user_content
        assert "Gesamt: 10 Entscheidungen" in user_content

    def test_risk_manager_with_feedback_shows_in_prompt(self):
        """risk_manager mit feedback_context → TRACK-RECORD im Prompt."""
        llm = _CapturingLLM('{"risiko_score": 3, "empfehlung": "GENEHMIGT"}')
        trade = {"aktion": "KAUFEN", "zielkurs": 110}
        data = {
            "ticker": "TEST",
            "fundamentals": {},
            "technicals": {"current_price": 100.0},
            "sentiment": {},
        }

        risk_manager(trade, data, llm, data_text="dummy", feedback_context=self._FEEDBACK)

        user_content = llm.captured_messages[0][1]["content"]
        assert "TRACK-RECORD" in user_content

    def test_portfolio_manager_with_feedback_shows_in_prompt(self):
        """portfolio_manager mit feedback_context → TRACK-RECORD im Prompt."""
        llm = _CapturingLLM('{"entscheidung": "GENEHMIGT", "confidence": 4}')
        trade = {"aktion": "KAUFEN"}
        risk = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}

        portfolio_manager(trade, risk, llm, feedback_context=self._FEEDBACK)

        user_content = llm.captured_messages[0][1]["content"]
        assert "TRACK-RECORD" in user_content
        assert "Gesamt: 10 Entscheidungen" in user_content

    def test_portfolio_manager_feedback_after_portfolio_fit(self):
        """feedback_context wird NACH portfolio_fit im Prompt eingefügt."""
        llm = _CapturingLLM('{"entscheidung": "GENEHMIGT", "confidence": 4}')
        trade = {"aktion": "KAUFEN"}
        risk = {"risiko_score": 3, "empfehlung": "GENEHMIGT"}
        pf = {"portfolio_fit_score": 4, "ziel_gewichtung_pct": 5}

        portfolio_manager(trade, risk, llm, portfolio_fit=pf, feedback_context=self._FEEDBACK)

        user_content = llm.captured_messages[0][1]["content"]
        # Portfolio-Fit kommt vor Track-Record
        pf_pos = user_content.find("Portfolio-Fit-Einschätzung")
        tr_pos = user_content.find("TRACK-RECORD")
        assert pf_pos < tr_pos
        assert tr_pos != -1


# --------------------------------------------------------------------------- #
# Tests: Pipeline-Integration — run_pipeline reicht feedback_context durch
# --------------------------------------------------------------------------- #


class TestPipelineIntegration:
    """Testet dass run_pipeline den feedback_context durchreicht."""

    def test_pipeline_with_enough_journal_entries(self, tmp_path, monkeypatch):
        """Bei genug Journal-Einträgen taucht der Kontext im Trader/PM-Prompt auf."""
        # Journal mit 6 Einträgen anlegen (≥ min_decisions=5)
        rows = [_make_row(ticker=f"T{i}", action="KAUFEN", confidence="4") for i in range(6)]
        journal_path = str(tmp_path / "decisions.csv")
        _write_journal(tmp_path, rows)

        # CWD auf tmp_path setzen, damit build_feedback_context() das Journal findet
        monkeypatch.chdir(tmp_path)
        # journal/ Unterverzeichnis anlegen und Datei dorthin kopieren
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir(exist_ok=True)
        import shutil

        shutil.copy(journal_path, journal_dir / "decisions.csv")

        # Pipeline mocken: collect_ticker_data + Agenten durch CapturingLLM ersetzen
        mock_data = {
            "ticker": "TEST",
            "fundamentals": {"name": "Test Corp", "sector": "Tech"},
            "technicals": {"current_price": 100.0, "sma50": 95, "sma200": 90},
            "sentiment": {"positiv": 1, "negativ": 0, "neutral": 0},
            "news": [],
            "macro": {},
            "peers": [],
            "history": [{"close": 100.0}, {"close": 101.0}],
            "data_warnings": [],
        }

        # LLM, der verschiedene Antworten je nach Rollen-Prompt gibt
        class _PipelineLLM:
            def __init__(self):
                self.captured: list[list[dict]] = []

            def chat(self, messages, temperature=0.3, **kwargs):
                self.captured.append(messages)
                system = messages[0]["content"]
                if "Fundamental" in system:
                    return json.dumps({"rolle": "F", "stimmung": "bullish", "score": 4, "zusammenfassung": "Gut"})
                if "technisch" in system:
                    return json.dumps({"rolle": "T", "stimmung": "bullish", "score": 4, "zusammenfassung": "Gut"})
                if "Sentiment" in system:
                    return json.dumps({"rolle": "S", "stimmung": "neutral", "score": 3, "zusammenfassung": "Ok"})
                if "Bull" in system:
                    return '{"confidence": 4, "name": "Bull"}\nBull text'
                if "Bear" in system:
                    return '{"confidence": 3, "name": "Bear"}\nBear text'
                if "Trader" in system:
                    return json.dumps({"rolle": "Trader", "aktion": "KAUFEN", "zielkurs": 110, "stop_loss": 90, "positionsanteil": 5, "begründung": "Test", "zeithorizont": "Mittelfristig"})
                if "Risk" in system:
                    return json.dumps({"rolle": "Risk", "risiko_score": 3, "empfehlung": "GENEHMIGT", "auflagen": "keine", "volatilität_bewertung": "moderat", "max_drawdown_schaetzung": "10%", "positionsgröße_empfohlen": "5"})
                if "Portfolio-Manager" in system:
                    return json.dumps({"rolle": "PM", "entscheidung": "GENEHMIGT", "begründung": "Test", "confidence": 4})
                return '{"error": "unknown role"}'

        llm = _PipelineLLM()

        with patch("concilium.pipeline.collect_ticker_data", return_value=mock_data):
            with patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]):
                with patch("concilium.pipeline.portfolio_fit_agent", return_value={"portfolio_fit_score": 4, "ziel_gewichtung_pct": 5}):
                    from concilium.pipeline import run_pipeline

                    run_pipeline("TEST", llm=llm, ensemble=False)

        # Mindestens der Trader- und PM-Prompt sollten TRACK-RECORD enthalten
        trader_prompts = [
            m for msgs in llm.captured for m in msgs
            if m["role"] == "user" and "Bear-Argumentation" in m["content"]
        ]
        pm_prompts = [
            m for msgs in llm.captured for m in msgs
            if m["role"] == "user" and "Risiko-Bewertung" in m["content"]
        ]
        assert any("TRACK-RECORD" in p["content"] for p in trader_prompts), (
            "Trader-Prompt sollte TRACK-RECORD enthalten"
        )
        assert any("TRACK-RECORD" in p["content"] for p in pm_prompts), (
            "PM-Prompt sollte TRACK-RECORD enthalten"
        )

    def test_pipeline_no_journal_no_feedback(self, tmp_path, monkeypatch):
        """Ohne Journal → kein TRACK-RECORD in irgend einem Prompt."""
        # CWD auf leeres tmp_path (kein journal/ Unterverzeichnis)
        monkeypatch.chdir(tmp_path)

        mock_data = {
            "ticker": "TEST",
            "fundamentals": {"name": "Test", "sector": "X"},
            "technicals": {"current_price": 100.0},
            "sentiment": {"positiv": 0, "negativ": 0, "neutral": 0},
            "news": [],
            "macro": {},
            "peers": [],
            "history": [{"close": 100.0}, {"close": 101.0}],
            "data_warnings": [],
        }

        class _PipelineLLM:
            def __init__(self):
                self.captured: list[list[dict]] = []

            def chat(self, messages, temperature=0.3, **kwargs):
                self.captured.append(messages)
                system = messages[0]["content"]
                if "Fundamental" in system:
                    return json.dumps({"rolle": "F", "stimmung": "bullish", "score": 4, "zusammenfassung": "Gut"})
                if "technisch" in system:
                    return json.dumps({"rolle": "T", "stimmung": "bullish", "score": 4, "zusammenfassung": "Gut"})
                if "Sentiment" in system:
                    return json.dumps({"rolle": "S", "stimmung": "neutral", "score": 3, "zusammenfassung": "Ok"})
                if "Bull" in system:
                    return '{"confidence": 4, "name": "Bull"}\nBull text'
                if "Bear" in system:
                    return '{"confidence": 3, "name": "Bear"}\nBear text'
                if "Trader" in system:
                    return json.dumps({"rolle": "Trader", "aktion": "HALTEN", "zielkurs": None, "stop_loss": None, "positionsanteil": 0, "begründung": "Test", "zeithorizont": "N/A"})
                if "Risk" in system:
                    return json.dumps({"rolle": "Risk", "risiko_score": 3, "empfehlung": "GENEHMIGT", "auflagen": "keine", "volatilität_bewertung": "ok", "max_drawdown_schaetzung": "5%", "positionsgröße_empfohlen": "5"})
                if "Portfolio-Manager" in system:
                    return json.dumps({"rolle": "PM", "entscheidung": "ABGELEHNT", "begründung": "Test", "confidence": 2})
                return '{}'

        llm = _PipelineLLM()

        with patch("concilium.pipeline.collect_ticker_data", return_value=mock_data):
            with patch("concilium.pipeline.fetch_portfolio_positions", return_value=[]):
                with patch("concilium.pipeline.portfolio_fit_agent", return_value=None):
                    from concilium.pipeline import run_pipeline

                    run_pipeline("TEST", llm=llm, ensemble=False)

        # Kein Prompt sollte TRACK-RECORD enthalten
        all_user_contents = [
            m["content"] for msgs in llm.captured for m in msgs if m["role"] == "user"
        ]
        assert not any("TRACK-RECORD" in c for c in all_user_contents), (
            "Ohne Journal sollte kein TRACK-RECORD im Prompt stehen"
        )
