"""Tests für die 5-stufige Rating-Skala (Feature 1).

Testet:
- _rating_to_action Normalisierung
- trader() mit 5-stufigem Rating → normalisierte aktion + rating key
- ensemble_trader: alle_ratings in _ensemble Metadaten
- journal: rating column wird geschrieben
- evaluate: _evaluate_single liefert rating + rating_distance
- evaluate: _aggregate berechnet durchschnitt_rating_distanz
- feedback: build_feedback_context enthält Rating-Verteilung
- report: Rating wird im Trade-Vorschlag gezeigt
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.agents import (  # noqa: E402
    RATING_5,
    _rating_to_action,
    ensemble_trader,
    trader,
)
from concilium.journal import JOURNAL_HEADER  # noqa: E402

# --------------------------------------------------------------------------- #
# Tests: _rating_to_action
# --------------------------------------------------------------------------- #


class TestRatingToAction:
    """Testet die 5-stufig → 3-stufig Normalisierung."""

    def test_stark_kaufen_to_kaufen(self):
        assert _rating_to_action("STARK KAUFEN") == "KAUFEN"

    def test_kaufen_to_kaufen(self):
        assert _rating_to_action("KAUFEN") == "KAUFEN"

    def test_halten_to_halten(self):
        assert _rating_to_action("HALTEN") == "HALTEN"

    def test_verkaufen_to_verkaufen(self):
        assert _rating_to_action("VERKAUFEN") == "VERKAUFEN"

    def test_stark_verkaufen_to_verkaufen(self):
        assert _rating_to_action("STARK VERKAUFEN") == "VERKAUFEN"

    def test_unknown_to_halten(self):
        assert _rating_to_action("UNKNOWN") == "HALTEN"

    def test_empty_to_halten(self):
        assert _rating_to_action("") == "HALTEN"

    def test_none_to_halten(self):
        assert _rating_to_action(None) == "HALTEN"  # type: ignore[arg-type]

    def test_lowercase_works(self):
        assert _rating_to_action("stark kaufen") == "KAUFEN"

    def test_rating_5_constant(self):
        """RATING_5 enthält alle 5 Stufen in der richtigen Reihenfolge."""
        assert RATING_5 == ["STARK KAUFEN", "KAUFEN", "HALTEN", "VERKAUFEN", "STARK VERKAUFEN"]


# --------------------------------------------------------------------------- #
# Tests: trader() mit 5-stufigem Rating
# --------------------------------------------------------------------------- #


class _CapturingLLM:
    """Mock-LLM, der die übergebenen messages speichert und JSON zurückgibt."""

    def __init__(self, response: str = '{"aktion": "HALTEN"}'):
        self._response = response
        self.captured_messages: list[list[dict]] = []

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3, **kwargs) -> str:
        self.captured_messages.append(messages)
        return self._response


class TestTraderRating:
    """Testet dass trader() das 5-stufige Rating normalisiert und in 'rating' speichert."""

    def test_stark_kaufen_normalized(self):
        """trader() mit STARK KAUFEN → aktion=KAUFEN, rating=STARK KAUFEN."""
        llm = _CapturingLLM(json.dumps({
            "rolle": "Trader",
            "aktion": "STARK KAUFEN",
            "zielkurs": 120,
            "stop_loss": 90,
            "positionsanteil": 8,
            "begründung": "Sehr bullish",
            "zeithorizont": "Mittelfristig",
        }))
        analysts = {
            "fundamental": {"stimmung": "bullish", "score": 4, "_raw": "Gut"},
            "technical": {"stimmung": "bullish", "score": 4, "_raw": "Gut"},
            "sentiment": {"stimmung": "neutral", "score": 3, "_raw": "Neutral"},
            "technicals": {"current_price": 100.0},
        }
        debate = {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}}

        result = trader(analysts, debate, llm)

        assert result["aktion"] == "KAUFEN"
        assert result["rating"] == "STARK KAUFEN"

    def test_stark_verkaufen_normalized(self):
        """trader() mit STARK VERKAUFEN → aktion=VERKAUFEN, rating=STARK VERKAUFEN."""
        llm = _CapturingLLM(json.dumps({
            "rolle": "Trader",
            "aktion": "STARK VERKAUFEN",
            "zielkurs": None,
            "stop_loss": None,
            "positionsanteil": 0,
            "begründung": "Sehr bearish",
            "zeithorizont": "Kurzfristig",
        }))
        analysts = {
            "fundamental": {"stimmung": "bearish", "score": 2, "_raw": "Schlecht"},
            "technical": {"stimmung": "bearish", "score": 2, "_raw": "Schlecht"},
            "sentiment": {"stimmung": "negative", "score": 2, "_raw": "Negativ"},
            "technicals": {"current_price": 100.0},
        }
        debate = {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}}

        result = trader(analysts, debate, llm)

        assert result["aktion"] == "VERKAUFEN"
        assert result["rating"] == "STARK VERKAUFEN"

    def test_halten_unchanged(self):
        """trader() mit HALTEN → aktion=HALTEN, rating=HALTEN."""
        llm = _CapturingLLM(json.dumps({
            "rolle": "Trader",
            "aktion": "HALTEN",
            "zielkurs": None,
            "stop_loss": None,
            "positionsanteil": 0,
            "begründung": "Neutral",
            "zeithorizont": "Mittelfristig",
        }))
        analysts = {
            "fundamental": {"stimmung": "neutral", "score": 3, "_raw": "Ok"},
            "technical": {"stimmung": "neutral", "score": 3, "_raw": "Ok"},
            "sentiment": {"stimmung": "neutral", "score": 3, "_raw": "Neutral"},
            "technicals": {"current_price": 100.0},
        }
        debate = {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}}

        result = trader(analysts, debate, llm)

        assert result["aktion"] == "HALTEN"
        assert result["rating"] == "HALTEN"

    def test_unknown_rating_defaults_to_halten(self):
        """trader() mit unbekanntem Rating → aktion=HALTEN."""
        llm = _CapturingLLM(json.dumps({
            "rolle": "Trader",
            "aktion": "UNBEKANNT",
            "zielkurs": None,
            "stop_loss": None,
            "positionsanteil": 0,
            "begründung": "Test",
            "zeithorizont": "Mittelfristig",
        }))
        analysts = {
            "fundamental": {"stimmung": "neutral", "score": 3, "_raw": "Ok"},
            "technical": {"stimmung": "neutral", "score": 3, "_raw": "Ok"},
            "sentiment": {"stimmung": "neutral", "score": 3, "_raw": "Neutral"},
            "technicals": {"current_price": 100.0},
        }
        debate = {"bull": {"_raw": "Bull"}, "bear": {"_raw": "Bear"}}

        result = trader(analysts, debate, llm)

        assert result["aktion"] == "HALTEN"

    def test_rating_in_prompt_5_step_instruction(self):
        """Der SYSTEM_TRADER Prompt enthält die 5-stufige Skala."""
        from concilium.agents import SYSTEM_TRADER

        assert "STARK KAUFEN" in SYSTEM_TRADER
        assert "STARK VERKAUFEN" in SYSTEM_TRADER
        assert "5-stufige" in SYSTEM_TRADER


# --------------------------------------------------------------------------- #
# Tests: ensemble_trader mit 5-stufigem Rating
# --------------------------------------------------------------------------- #


class TestEnsembleRatings:
    """Testet dass ensemble_trader alle_ratings in _ensemble sammelt."""

    def test_alle_ratings_collected(self):
        """ensemble_trader sammelt die 5-stufigen Ratings aller Runs."""
        # Reuse _FakeLLM from test_ensemble.py
        from tests.test_ensemble import _ANALYSTS, _DEBATE, _FakeLLM

        def _trader_json_5(aktion: str) -> str:
            return json.dumps({
                "rolle": "Trader",
                "aktion": aktion,
                "zielkurs": 120,
                "stop_loss": 90,
                "positionsanteil": 5,
                "begründung": "Test",
                "zeithorizont": "Mittelfristig",
            })

        llm = _FakeLLM([
            _trader_json_5("STARK KAUFEN"),  # temp=0.3 → normalized KAUFEN
            _trader_json_5("KAUFEN"),        # temp=0.5 → normalized KAUFEN
            _trader_json_5("HALTEN"),        # temp=0.7 → normalized HALTEN
        ])

        result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=3)

        # Mehrheits-Aktion ist KAUFEN (2 von 3)
        assert result["aktion"] == "KAUFEN"
        assert result["_ensemble"]["mehrheits_aktion"] == "KAUFEN"
        # alle_ratings enthält die rohen 5-stufigen Werte
        assert result["_ensemble"]["alle_ratings"] == ["STARK KAUFEN", "KAUFEN", "HALTEN"]
        # Der gewählte Run behält sein rating
        assert result.get("rating") in ("STARK KAUFEN", "KAUFEN")

    def test_all_runs_fail_has_empty_ratings(self):
        """Bei allen fehlgeschlagenen Runs → leere alle_ratings."""
        from unittest.mock import MagicMock

        import concilium.agents as agents_mod

        with patch.object(agents_mod, "trader", side_effect=RuntimeError("LLM down")):
            result = ensemble_trader({}, {}, MagicMock(), runs=3)

        assert result["aktion"] == "HALTEN"
        assert result["_ensemble"]["alle_ratings"] == []

    def test_single_run_has_ratings(self):
        """Bei nur 1 erfolgreichem Run → alle_ratings mit einem Element."""
        from tests.test_ensemble import _ANALYSTS, _DEBATE, _FakeLLM

        llm = _FakeLLM([
            json.dumps({
                "rolle": "Trader", "aktion": "STARK KAUFEN",
                "zielkurs": 120, "stop_loss": 90, "positionsanteil": 5,
                "begründung": "Test", "zeithorizont": "Mittelfristig",
            }),
        ])

        result = ensemble_trader(_ANALYSTS, _DEBATE, llm, runs=1)
        assert result["aktion"] == "KAUFEN"
        assert result["rating"] == "STARK KAUFEN"
        assert result["_ensemble"]["alle_ratings"] == ["STARK KAUFEN"]


# --------------------------------------------------------------------------- #
# Tests: Journal mit rating column
# --------------------------------------------------------------------------- #


class TestJournalRating:
    """Testet dass das Journal die rating column schreibt."""

    def test_rating_in_header(self):
        """JOURNAL_HEADER enthält 'rating' nach 'action'."""
        assert "rating" in JOURNAL_HEADER
        action_idx = JOURNAL_HEADER.index("action")
        rating_idx = JOURNAL_HEADER.index("rating")
        assert rating_idx == action_idx + 1

    def test_append_decision_writes_rating(self, tmp_path):
        """append_decision schreibt das rating in die CSV."""
        from concilium.journal import append_decision

        journal_file = str(tmp_path / "decisions.csv")
        result = {
            "ticker": "AAPL",
            "trade": {"aktion": "KAUFEN", "rating": "STARK KAUFEN", "zielkurs": 120},
            "final": {"entscheidung": "GENEHMIGT", "confidence": 4},
            "debate": {"bull": {"_raw": ""}, "bear": {"_raw": ""}},
        }
        append_decision(result, journal_file=journal_file)

        with open(journal_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["rating"] == "STARK KAUFEN"
        assert rows[0]["action"] == "KAUFEN"

    def test_append_decision_empty_rating(self, tmp_path):
        """Bei fehlendem rating → leere Spalte."""
        from concilium.journal import append_decision

        journal_file = str(tmp_path / "decisions.csv")
        result = {
            "ticker": "AAPL",
            "trade": {"aktion": "HALTEN"},
            "final": {"entscheidung": "ABGELEHNT", "confidence": 2},
            "debate": {"bull": {"_raw": ""}, "bear": {"_raw": ""}},
        }
        append_decision(result, journal_file=journal_file)

        with open(journal_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["rating"] == ""

    def test_journal_migration_adds_rating(self, tmp_path):
        """Bestehende CSV ohne rating-Spalte wird migriert."""
        # Alte CSV ohne rating column
        old_header = ["timestamp", "ticker", "action", "target", "stop"]
        journal_file = str(tmp_path / "decisions.csv")
        with open(journal_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=old_header)
            writer.writeheader()
            writer.writerow({
                "timestamp": "2026-01-01 10:00:00",
                "ticker": "OLD",
                "action": "KAUFEN",
                "target": "100",
                "stop": "90",
            })

        from concilium.journal import append_decision

        result = {
            "ticker": "NEW",
            "trade": {"aktion": "HALTEN", "rating": "HALTEN"},
            "final": {"entscheidung": "ABGELEHNT", "confidence": 2},
            "debate": {"bull": {"_raw": ""}, "bear": {"_raw": ""}},
        }
        append_decision(result, journal_file=journal_file)

        with open(journal_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # Beide Zeilen vorhanden, alte hat rating=""
        assert len(rows) == 2
        assert rows[0]["rating"] == ""  # migrierte Zeile
        assert rows[1]["rating"] == "HALTEN"  # neue Zeile


# --------------------------------------------------------------------------- #
# Tests: evaluate mit rating
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


class TestEvaluateRating:
    """Testet rating-aware _evaluate_single und _aggregate."""

    def test_evaluate_single_returns_rating(self):
        """_evaluate_single liefert das rating aus der Zeile."""
        from concilium.evaluate import _evaluate_single

        prices = _make_prices(100, 60, drift=0.005)
        row = {
            "ticker": "AAPL",
            "action": "KAUFEN",
            "rating": "STARK KAUFEN",
            "confidence": "4",
            "timestamp": "2026-01-01 10:00:00",
        }
        result = _evaluate_single(row, prices, 90)

        assert result["rating"] == "STARK KAUFEN"

    def test_evaluate_single_returns_rating_distance(self):
        """_evaluate_single liefert rating_distance (int|None)."""
        from concilium.evaluate import _evaluate_single

        prices = _make_prices(100, 60, drift=0.005)
        row = {
            "ticker": "AAPL",
            "action": "KAUFEN",
            "rating": "STARK KAUFEN",
            "confidence": "4",
            "timestamp": "2026-01-01 10:00:00",
        }
        result = _evaluate_single(row, prices, 90)

        assert "rating_distance" in result
        assert result["rating_distance"] is not None
        assert isinstance(result["rating_distance"], int)

    def test_evaluate_single_no_rating(self):
        """Ohne rating in der Zeile → rating='' und rating_distance=None."""
        from concilium.evaluate import _evaluate_single

        prices = _make_prices(100, 60, drift=0.005)
        row = {
            "ticker": "AAPL",
            "action": "KAUFEN",
            "rating": "",
            "confidence": "4",
            "timestamp": "2026-01-01 10:00:00",
        }
        result = _evaluate_single(row, prices, 90)

        assert result["rating"] == ""
        assert result["rating_distance"] is None

    def test_rating_distance_correctness(self):
        """STARK KAUFEN (index 0) vs stark steigende Rendite (>+2% → index 0) → distance=0."""
        from concilium.evaluate import _evaluate_single

        prices = _make_prices(100, 60, drift=0.01)  # ~+1%/Tag → stark steigend
        row = {
            "ticker": "AAPL",
            "action": "KAUFEN",
            "rating": "STARK KAUFEN",
            "confidence": "4",
            "timestamp": "2026-01-01 10:00:00",
        }
        result = _evaluate_single(row, prices, 90)

        # STARK KAUFEN = 0, outcome >+2% = 0 → distance = 0
        assert result["rating_distance"] == 0

    def test_rating_distance_off_by_one(self):
        """KAUFEN (index 1) vs stark steigende Rendite (>+2% → index 0) → distance=1."""
        from concilium.evaluate import _evaluate_single

        prices = _make_prices(100, 60, drift=0.01)
        row = {
            "ticker": "AAPL",
            "action": "KAUFEN",
            "rating": "KAUFEN",
            "confidence": "4",
            "timestamp": "2026-01-01 10:00:00",
        }
        result = _evaluate_single(row, prices, 90)

        # KAUFEN = 1, outcome >+2% = 0 → distance = 1
        assert result["rating_distance"] == 1

    def test_aggregate_durchschnitt_rating_distanz(self):
        """_aggregate berechnet durchschnitt_rating_distanz."""
        from concilium.evaluate import _aggregate

        evaluations = [
            {"hit": True, "rendite_pct": 5.0, "ziel_erreicht": None, "stop_gerissen": None,
             "action": "KAUFEN", "rating": "STARK KAUFEN", "rating_distance": 0,
             "confidence": 4, "portfolio_fit_score": None, "ticker": "A", "timestamp": ""},
            {"hit": False, "rendite_pct": -3.0, "ziel_erreicht": None, "stop_gerissen": None,
             "action": "KAUFEN", "rating": "STARK KAUFEN", "rating_distance": 4,
             "confidence": 3, "portfolio_fit_score": None, "ticker": "B", "timestamp": ""},
        ]
        result = _aggregate(evaluations)

        assert result["durchschnitt_rating_distanz"] is not None
        assert result["durchschnitt_rating_distanz"] == 2.0  # (0 + 4) / 2

    def test_empty_result_has_rating_distanz(self):
        """_empty_result hat durchschnitt_rating_distanz = None."""
        from concilium.evaluate import _empty_result

        result = _empty_result()
        assert "durchschnitt_rating_distanz" in result
        assert result["durchschnitt_rating_distanz"] is None

    def test_rating_index_helper(self):
        """_rating_index mappt korrekt."""
        from concilium.evaluate import _rating_index

        assert _rating_index("STARK KAUFEN") == 0
        assert _rating_index("KAUFEN") == 1
        assert _rating_index("HALTEN") == 2
        assert _rating_index("VERKAUFEN") == 3
        assert _rating_index("STARK VERKAUFEN") == 4
        assert _rating_index("UNKNOWN") is None
        assert _rating_index("") is None

    def test_outcome_rating_index_helper(self):
        """_outcome_rating_index mappt Rendite korrekt."""
        from concilium.evaluate import _outcome_rating_index

        assert _outcome_rating_index(5.0) == 0   # >+2% → STARK KAUFEN
        assert _outcome_rating_index(1.0) == 1   # >0% → KAUFEN
        assert _outcome_rating_index(-1.0) == 3  # <0% → VERKAUFEN
        assert _outcome_rating_index(-5.0) == 4  # <-2% → STARK VERKAUFEN
        assert _outcome_rating_index(0.0) == 2    # ==0 → HALTEN
        assert _outcome_rating_index(None) is None


# --------------------------------------------------------------------------- #
# Tests: feedback mit Rating-Verteilung
# --------------------------------------------------------------------------- #


class TestFeedbackRating:
    """Testet dass build_feedback_context die Rating-Verteilung enthält."""

    def test_rating_verteilung_in_context(self, tmp_path):
        """Bei >=min_decisions → Rating-Verteilung Zeile im Kontext."""
        from concilium.feedback import build_feedback_context

        path = str(tmp_path / "decisions.csv")
        rows = []
        for i in range(5):
            ratings = ["STARK KAUFEN", "KAUFEN", "HALTEN", "VERKAUFEN", "STARK VERKAUFEN"]
            rows.append({
                "ticker": f"T{i}",
                "action": "KAUFEN",
                "rating": ratings[i],
                "final_decision": "GENEHMIGT",
                "confidence": "4",
                "timestamp": "2026-01-01 10:00:00",
            })
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
            writer.writeheader()
            for row in rows:
                full_row = {k: row.get(k, "") for k in JOURNAL_HEADER}
                writer.writerow(full_row)

        result = build_feedback_context(path)

        assert "Rating-Verteilung" in result
        assert "STARK KAUFEN: 1" in result
        assert "KAUFEN: 1" in result
        assert "STARK VERKAUFEN: 1" in result

    def test_rating_verteilung_empty_ratings(self, tmp_path):
        """Bei leeren ratings → Rating-Verteilung mit 0-Werten."""
        from concilium.feedback import build_feedback_context

        path = str(tmp_path / "decisions.csv")
        rows = []
        for i in range(5):
            rows.append({
                "ticker": f"T{i}",
                "action": "KAUFEN",
                "rating": "",
                "final_decision": "GENEHMIGT",
                "confidence": "4",
                "timestamp": "2026-01-01 10:00:00",
            })
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
            writer.writeheader()
            for row in rows:
                full_row = {k: row.get(k, "") for k in JOURNAL_HEADER}
                writer.writerow(full_row)

        result = build_feedback_context(path)

        assert "Rating-Verteilung" in result
        assert "STARK KAUFEN: 0" in result


# --------------------------------------------------------------------------- #
# Tests: report mit Rating
# --------------------------------------------------------------------------- #


class TestReportRating:
    """Testet dass der Report das Rating anzeigt."""

    def test_report_shows_rating(self):
        """Trade-Vorschlag im Report zeigt das Rating."""
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
                "bull": {"_raw": '{"confidence": 4, "name": "Bull"}\nBull text'},
                "bear": {"_raw": '{"confidence": 3, "name": "Bear"}\nBear text'},
            },
            "trade": {
                "aktion": "KAUFEN",
                "rating": "STARK KAUFEN",
                "zielkurs": 180,
                "stop_loss": 130,
                "positionsanteil": 7,
                "begründung": "Test",
                "zeithorizont": "Mittelfristig",
                "_ensemble": {
                    "runs": 3,
                    "mehrheits_aktion": "KAUFEN",
                    "ensemble_confidence": 0.67,
                    "alle_aktionen": ["KAUFEN", "KAUFEN", "HALTEN"],
                    "alle_ratings": ["STARK KAUFEN", "KAUFEN", "HALTEN"],
                },
            },
            "risk": {"risiko_score": 3, "empfehlung": "GENEHMIGT"},
            "final": {"entscheidung": "GENEHMIGT", "confidence": 4},
        }

        report = generate_report(result)

        assert "Rating (5-stufig)" in report
        assert "STARK KAUFEN" in report
        assert "Rating-Verteilung" in report
        assert "STARK KAUFEN, KAUFEN, HALTEN" in report

    def test_report_no_rating(self):
        """Ohne rating → kein Rating-Linie."""
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
        }

        report = generate_report(result)

        assert "Rating (5-stufig)" not in report
