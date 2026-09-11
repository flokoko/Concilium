"""Tests für Stufe 1: Invalidierungs-Kriterien (falsifizierbare Thesen).

Abgedeckte Bereiche:
(a) Schemas: invalidation als optionales property in allen 5 Analysten-
    Schemas (nicht in required, Defaults "", validiert clean).
(b) Journal: append_decision/append_review_decision schreiben die Spalte;
    Legacy-CSV ohne Spalte migriert zu "" ohne Crash.
(c) Pipeline: _aggregate_invalidation deterministisch + nie-crashend;
    result["invalidation"] im Mock-Lauf gefüllt; ohne Analysten "".
(d) Evaluate: LLM-Check (gemockt) pro Zeile; Fallback ohne LLM; Quote/
    Hits/nicht_bewertbar; hit-Berechnung UNVERÄNDERT (STANDALONE-Metrik).
(e) Report: Track-Record-Sektion "Invalidierungs-Trefferquote" + Per-
    Analyse-Sektion "Invalidierungs-Bedingungen".

Alle Tests sind OFFLINE-fähig: kein yfinance, kein Netzwerk, kein echtes LLM.
"""

from __future__ import annotations

import csv
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)


from concilium.agents import (  # noqa: E402
    SYSTEM_FUNDAMENTAL,
    SYSTEM_MACRO_NEWS,
    SYSTEM_SENTIMENT,
    SYSTEM_SOCIAL,
    SYSTEM_TECHNICAL,
)
from concilium.evaluate import (  # noqa: E402
    _evaluate_single,
    check_invalidation_hit,
    evaluate_journal,
)
from concilium.journal import (  # noqa: E402
    JOURNAL_HEADER,
    REVIEW_HEADER,
    append_decision,
    append_review_decision,
)
from concilium.pipeline import (  # noqa: E402
    _aggregate_invalidation,
)
from concilium.report import (  # noqa: E402
    generate_report,
    generate_track_record_report,
)
from concilium.schemas import (  # noqa: E402
    ANALYST_FUNDAMENTAL_SCHEMA,
    ANALYST_MACRO_NEWS_SCHEMA,
    ANALYST_SENTIMENT_SCHEMA,
    ANALYST_SOCIAL_SCHEMA,
    ANALYST_TECHNICAL_SCHEMA,
    defaults_for_schema,
    validate_structured,
)

# --------------------------------------------------------------------------- #
# (a) Schemas
# --------------------------------------------------------------------------- #


ALL_ANALYST_SCHEMAS = [
    ANALYST_FUNDAMENTAL_SCHEMA,
    ANALYST_TECHNICAL_SCHEMA,
    ANALYST_SENTIMENT_SCHEMA,
    ANALYST_SOCIAL_SCHEMA,
    ANALYST_MACRO_NEWS_SCHEMA,
]


class TestAnalystSchemas:
    """invalidation als optionales property in allen Analysten-Schemas."""

    def test_invalidation_property_present_in_all(self):
        for schema in ALL_ANALYST_SCHEMAS:
            inner = schema["json_schema"]["schema"]
            assert inner["properties"]["invalidation"] == {"type": "string"}

    def test_invalidation_not_required(self):
        for schema in ALL_ANALYST_SCHEMAS:
            inner = schema["json_schema"]["schema"]
            assert "invalidation" not in inner["required"]

    def test_additional_properties_still_false(self):
        for schema in ALL_ANALYST_SCHEMAS:
            inner = schema["json_schema"]["schema"]
            assert inner["additionalProperties"] is False

    def test_defaults_validate_clean(self):
        for schema in ALL_ANALYST_SCHEMAS:
            d = defaults_for_schema(schema)
            assert d["invalidation"] == ""
            assert validate_structured(d, schema) == []

    def test_filled_invalidation_validates_clean(self):
        for schema in ALL_ANALYST_SCHEMAS:
            d = defaults_for_schema(schema)
            d["invalidation"] = "KGV über 25; Umsatzwachstum unter 5%"
            assert validate_structured(d, schema) == []

    def test_old_style_response_without_invalidation_validates(self):
        """Rückwärtskompatibilität: Antwort OHNE invalidation bleibt gültig."""
        d = defaults_for_schema(ANALYST_FUNDAMENTAL_SCHEMA)
        d.pop("invalidation")
        assert validate_structured(d, ANALYST_FUNDAMENTAL_SCHEMA) == []

    def test_unknown_field_rejected(self):
        d = defaults_for_schema(ANALYST_FUNDAMENTAL_SCHEMA)
        d["unbekanntes_feld"] = "x"
        assert validate_structured(d, ANALYST_FUNDAMENTAL_SCHEMA) != []


class TestPrompts:
    """Alle 5 SYSTEM-Prompts fordern das invalidation-Feld."""

    def test_all_prompts_mention_invalidation(self):
        for prompt in (
            SYSTEM_FUNDAMENTAL,
            SYSTEM_TECHNICAL,
            SYSTEM_SENTIMENT,
            SYSTEM_SOCIAL,
            SYSTEM_MACRO_NEWS,
        ):
            assert "invalidation" in prompt, "Prompt enthält kein invalidation-Feld"
            assert "WIDERLEGEN" in prompt or "widerlegen" in prompt

    def test_prompts_contain_json_template_field(self):
        for prompt in (
            SYSTEM_FUNDAMENTAL,
            SYSTEM_TECHNICAL,
            SYSTEM_SENTIMENT,
            SYSTEM_SOCIAL,
            SYSTEM_MACRO_NEWS,
        ):
            assert '"invalidation"' in prompt


# --------------------------------------------------------------------------- #
# (b) Journal
# --------------------------------------------------------------------------- #


class TestJournalHeaders:
    def test_journal_header_contains_invalidation(self):
        assert "invalidation" in JOURNAL_HEADER

    def test_review_header_contains_invalidation(self):
        assert "invalidation" in REVIEW_HEADER

    def test_invalidation_before_c6_block(self):
        """invalidation ist Entscheidungszeitpunkt-Daten → vor dem C6-Block."""
        c6 = ["reflection_status", "resolved_at", "realised_return_pct",
              "alpha_pct", "lesson"]
        assert JOURNAL_HEADER[-5:] == c6
        assert JOURNAL_HEADER[-6] == "invalidation"
        assert REVIEW_HEADER[-5:] == c6
        assert REVIEW_HEADER[-6] == "invalidation"

    def test_invalidation_single_occurrence(self):
        assert JOURNAL_HEADER.count("invalidation") == 1
        assert REVIEW_HEADER.count("invalidation") == 1


def _base_result(invalidation: str | None = "Fundamental: KGV über 25") -> dict:
    result: dict = {
        "ticker": "AAPL",
        "trade": {"aktion": "KAUFEN", "zielkurs": 340.0, "stop_loss": 300.0},
        "final": {"entscheidung": "GENEHMIGT", "confidence": 4},
    }
    if invalidation is not None:
        result["invalidation"] = invalidation
    return result


def _read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class TestAppendDecisionInvalidation:
    def test_invalidation_written(self, tmp_path):
        journal_file = str(tmp_path / "journal" / "decisions.csv")
        append_decision(_base_result(), journal_file=journal_file)
        rows = _read_rows(journal_file)
        assert len(rows) == 1
        assert rows[0]["invalidation"] == "Fundamental: KGV über 25"

    def test_missing_invalidation_is_empty_no_crash(self, tmp_path):
        journal_file = str(tmp_path / "journal" / "decisions.csv")
        append_decision(_base_result(invalidation=None), journal_file=journal_file)
        rows = _read_rows(journal_file)
        assert rows[0]["invalidation"] == ""

    def test_none_invalidation_is_empty(self, tmp_path):
        journal_file = str(tmp_path / "journal" / "decisions.csv")
        result = _base_result()
        result["invalidation"] = None
        append_decision(result, journal_file=journal_file)
        rows = _read_rows(journal_file)
        assert rows[0]["invalidation"] == ""


class TestAppendReviewDecisionInvalidation:
    def test_review_invalidation_written(self, tmp_path):
        journal_file = str(tmp_path / "journal" / "reviews.csv")
        append_review_decision(_base_result(), journal_file=journal_file)
        rows = _read_rows(journal_file)
        assert rows[0]["invalidation"] == "Fundamental: KGV über 25"

    def test_review_missing_invalidation_empty(self, tmp_path):
        journal_file = str(tmp_path / "journal" / "reviews.csv")
        append_review_decision(_base_result(invalidation=None), journal_file=journal_file)
        rows = _read_rows(journal_file)
        assert rows[0]["invalidation"] == ""


class TestLegacyJournalMigration:
    def test_legacy_csv_without_column_migrates(self, tmp_path):
        """Alte CSV ohne invalidation-Spalte → Migration füllt "" ohne Crash."""
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir(parents=True, exist_ok=True)
        journal_file = str(journal_dir / "decisions.csv")
        old_fields = [f for f in JOURNAL_HEADER if f != "invalidation"]
        with open(journal_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=old_fields)
            writer.writeheader()
            writer.writerow({"timestamp": "2026-08-01 10:00:00", "ticker": "OLD.DE"})

        append_decision(
            _base_result(invalidation="Fundamental: KGV über 25"),
            journal_file=journal_file,
        )
        rows = _read_rows(journal_file)
        assert len(rows) == 2
        # Legacy-Zeile: leere invalidation-Spalte
        assert rows[0]["invalidation"] == ""
        # Neue Zeile: Wert gesetzt
        assert rows[1]["invalidation"] == "Fundamental: KGV über 25"
        with open(journal_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == JOURNAL_HEADER

    def test_legacy_review_csv_migration(self, tmp_path):
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir(parents=True, exist_ok=True)
        journal_file = str(journal_dir / "reviews.csv")
        old_fields = [f for f in REVIEW_HEADER if f != "invalidation"]
        with open(journal_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=old_fields)
            writer.writeheader()
            writer.writerow({"timestamp": "2026-08-01 10:00:00", "ticker": "OLD.DE"})

        append_review_decision(_base_result(), journal_file=journal_file)
        rows = _read_rows(journal_file)
        assert len(rows) == 2
        assert rows[0]["invalidation"] == ""
        with open(journal_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == REVIEW_HEADER


# --------------------------------------------------------------------------- #
# (c) Pipeline-Aggregation
# --------------------------------------------------------------------------- #


class TestAggregateInvalidation:
    def test_joins_nonempty_prefixed(self):
        analysts = {
            "fundamental": {"invalidation": "KGV über 25"},
            "technical": {"invalidation": "Bruch unter SMA200"},
            "sentiment": {"invalidation": "   "},  # whitespace → übersprungen
            "macro_news": {"invalidation": "VIX über 30"},
            "social": {"invalidation": None},
        }
        result = _aggregate_invalidation(analysts)
        assert result == (
            "fundamental: KGV über 25; technical: Bruch unter SMA200; "
            "macro_news: VIX über 30"
        )

    def test_missing_analysts_empty(self):
        assert _aggregate_invalidation(None) == ""
        assert _aggregate_invalidation({}) == ""
        assert _aggregate_invalidation("junk") == ""

    def test_no_invalidation_fields_empty(self):
        analysts = {
            "fundamental": {"stimmung": "bullish"},
            "technical": {"stimmung": "bearish"},
        }
        assert _aggregate_invalidation(analysts) == ""

    def test_non_string_values_skipped(self):
        analysts = {"fundamental": {"invalidation": 42}, "technical": ["x"]}
        assert _aggregate_invalidation(analysts) == ""

    def test_deterministic_order(self):
        analysts = {
            "social": {"invalidation": "S"},
            "fundamental": {"invalidation": "F"},
            "technical": {"invalidation": "T"},
            "sentiment": {"invalidation": "Se"},
            "macro_news": {"invalidation": "M"},
        }
        result = _aggregate_invalidation(analysts)
        assert result == "fundamental: F; technical: T; sentiment: Se; macro_news: M; social: S"
        # Erneuter Aufruf → identisches Ergebnis
        assert _aggregate_invalidation(analysts) == result


class TestPipelineWiring:
    """run_pipeline schreibt result['invalidation'] aus den Analysten."""

    def _mock_data(self) -> dict:
        return {
            "ticker": "TEST",
            "fundamentals": {"name": "TestCo", "sector": "Tech"},
            "technicals": {"current_price": 100},
            "sentiment": {},
            "news": [],
        }

    def test_result_invalidation_from_analysts(self, tmp_path, monkeypatch):
        d = tmp_path / "state"
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(d))
        monkeypatch.chdir(tmp_path)

        analysts = {
            "fundamental": {"stimmung": "bullish", "score": 4, "invalidation": "KGV über 25"},
            "technical": {"stimmung": "neutral", "score": 3, "invalidation": ""},
        }
        llm = MagicMock()
        llm.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        with patch.multiple(
            "concilium.pipeline",
            collect_ticker_data=MagicMock(return_value=self._mock_data()),
            analyst_team=MagicMock(return_value=analysts),
            debate=MagicMock(return_value={"bull": {"_raw": "B"}, "bear": {"_raw": "B"}}),
            trader=MagicMock(return_value={"aktion": "HALTEN", "rating": "HALTEN", "_raw": ""}),
            ensemble_trader=MagicMock(return_value={"aktion": "HALTEN", "rating": "HALTEN", "_raw": ""}),
            risk_manager=MagicMock(return_value={"risiko_score": 3, "empfehlung": "GENEHMIGT"}),
            fetch_portfolio_positions=MagicMock(return_value=[]),
            portfolio_fit_agent=MagicMock(return_value=None),
            trade_revision=MagicMock(return_value={"aktion": "HALTEN", "rating": "HALTEN", "_raw": ""}),
            portfolio_manager=MagicMock(return_value={"entscheidung": "GENEHMIGT", "confidence": 3}),
            build_feedback_context=MagicMock(return_value=""),
            build_reflection_context=MagicMock(return_value=""),
        ):
            from concilium.pipeline import run_pipeline

            result = run_pipeline("TEST", llm=llm, journal=False)
        assert result["invalidation"] == "fundamental: KGV über 25"

    def test_result_invalidation_empty_without_field(self, tmp_path, monkeypatch):
        d = tmp_path / "state"
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(d))
        monkeypatch.chdir(tmp_path)

        analysts = {"fundamental": {"stimmung": "bullish", "score": 4}}
        llm = MagicMock()
        llm.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        with patch.multiple(
            "concilium.pipeline",
            collect_ticker_data=MagicMock(return_value=self._mock_data()),
            analyst_team=MagicMock(return_value=analysts),
            debate=MagicMock(return_value={"bull": {"_raw": "B"}, "bear": {"_raw": "B"}}),
            trader=MagicMock(return_value={"aktion": "HALTEN", "rating": "HALTEN", "_raw": ""}),
            ensemble_trader=MagicMock(return_value={"aktion": "HALTEN", "rating": "HALTEN", "_raw": ""}),
            risk_manager=MagicMock(return_value={"risiko_score": 3, "empfehlung": "GENEHMIGT"}),
            fetch_portfolio_positions=MagicMock(return_value=[]),
            portfolio_fit_agent=MagicMock(return_value=None),
            trade_revision=MagicMock(return_value={"aktion": "HALTEN", "rating": "HALTEN", "_raw": ""}),
            portfolio_manager=MagicMock(return_value={"entscheidung": "GENEHMIGT", "confidence": 3}),
            build_feedback_context=MagicMock(return_value=""),
            build_reflection_context=MagicMock(return_value=""),
        ):
            from concilium.pipeline import run_pipeline

            result = run_pipeline("TEST", llm=llm, journal=False)
        assert result["invalidation"] == ""


# --------------------------------------------------------------------------- #
# (d) Evaluate — LLM-Check + Quote
# --------------------------------------------------------------------------- #


def _make_prices(start_price: float, n_days: int, drift: float = 0.0) -> list[dict]:
    from datetime import datetime, timedelta

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
    path = str(tmp_path / "decisions.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
        writer.writeheader()
        for row in rows:
            full_row = {k: row.get(k, "") for k in JOURNAL_HEADER}
            writer.writerow(full_row)
    return path


def _make_row(ticker: str = "AAPL", invalidation: str = "", action: str = "KAUFEN") -> dict:
    from datetime import datetime, timedelta

    ts = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "ticker": ticker,
        "action": action,
        "rating": action,
        "confidence": "4",
        "timestamp": ts,
        "invalidation": invalidation,
    }


def _mock_llm(verdict: bool = True, begr: str = "KGV stieg über 25") -> MagicMock:
    llm = MagicMock()
    llm.chat.return_value = (
        '{"invalidiert": ' + ("true" if verdict else "false") + ', "begruendung": "' + begr + '"}'
    )
    return llm


class TestCheckInvalidationHit:
    def test_llm_true(self):
        eval_row = {"ticker": "AAPL", "action": "KAUFEN", "rendite_pct": -12.0}
        result = check_invalidation_hit("KGV über 25", eval_row, _mock_llm(True))
        assert result == (True, "KGV stieg über 25")

    def test_llm_false(self):
        result = check_invalidation_hit("KGV über 25", {"rendite_pct": 5.0}, _mock_llm(False))
        assert result == (False, "KGV stieg über 25")

    def test_fallback_without_llm(self):
        assert check_invalidation_hit("KGV über 25", {}, None) is None

    def test_fallback_empty_invalidation(self):
        assert check_invalidation_hit("", {}, _mock_llm(True)) is None

    def test_fallback_on_llm_error(self):
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("Boom")
        assert check_invalidation_hit("KGV über 25", {}, llm) is None

    def test_fallback_on_unparseable_answer(self):
        llm = MagicMock()
        llm.chat.return_value = "Ich kann kein JSON."
        assert check_invalidation_hit("KGV über 25", {}, llm) is None


class TestEvaluateInvalidationQuote:
    def test_quote_computed_with_mocked_llm(self, tmp_path):
        rows = [
            _make_row(ticker="AAA", invalidation="KGV über 25"),
            _make_row(ticker="BBB", invalidation="Stop unter 90"),
            _make_row(ticker="CCC"),  # ohne invalidation → zählt nicht
        ]
        path = _write_journal(tmp_path, rows)

        verdicts = {"AAA": True, "BBB": False}

        def mock_load(ticker, *, lookback_days=90):
            return _make_prices(100, 60, drift=0.005)

        def chat_side_effect(messages, **kwargs):
            user = messages[1]["content"]
            for t, v in verdicts.items():
                if f"Ticker: {t}" in user:
                    return '{"invalidiert": %s, "begruendung": "Test"}' % (
                        "true" if v else "false"
                    )
            return '{"invalidiert": false, "begruendung": "n/a"}'

        llm = MagicMock()
        llm.chat.side_effect = chat_side_effect

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = evaluate_journal(path, lookback_days=90, llm=llm)

        assert result["invalidation_n"] == 2
        assert result["invalidation_hits"] == 1
        assert result["invalidation_hit_quote"] == 0.5
        assert result["invalidation_nicht_bewertbar"] == 0
        assert len(result["invalidation_details"]) == 2
        by_ticker = {d["ticker"]: d for d in result["invalidation_details"]}
        assert by_ticker["AAA"]["invalidiert"] is True
        assert by_ticker["BBB"]["invalidiert"] is False

    def test_no_llm_quote_stays_none(self, tmp_path):
        """Ohne LLM: Fallback — Quote None, n 0, kein Crash, Hits unverändert."""
        rows = [_make_row(ticker="AAA", invalidation="KGV über 25")]
        path = _write_journal(tmp_path, rows)

        def mock_load(ticker, *, lookback_days=90):
            return _make_prices(100, 60, drift=0.005)

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = evaluate_journal(path, lookback_days=90, llm=None)

        assert result["invalidation_hit_quote"] is None
        assert result["invalidation_n"] == 0
        assert result["invalidation_nicht_bewertbar"] == 0
        # Normale Hit-Berechnung läuft unverändert weiter:
        assert result["anzahl_entscheidungen"] == 1

    def test_llm_error_rows_counted_nicht_bewertbar(self, tmp_path):
        rows = [_make_row(ticker="AAA", invalidation="KGV über 25")]
        path = _write_journal(tmp_path, rows)

        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("LLM down")

        def mock_load(ticker, *, lookback_days=90):
            return _make_prices(100, 60, drift=0.005)

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = evaluate_journal(path, lookback_days=90, llm=llm)

        assert result["invalidation_n"] == 0
        assert result["invalidation_hit_quote"] is None
        assert result["invalidation_nicht_bewertbar"] == 1

    def test_hit_definition_unchanged(self, tmp_path):
        """STANDALONE-Garantie: hit/rendite ändern sich durch den Check nicht."""
        rows = [_make_row(ticker="AAA", invalidation="KGV über 25")]
        path = _write_journal(tmp_path, rows)

        def mock_load(ticker, *, lookback_days=90):
            return _make_prices(100, 60, drift=0.01)  # steigend → KAUFEN-Hit

        llm = _mock_llm(True)  # These verletzt — Hit bleibt trotzdem True
        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result_with = evaluate_journal(path, lookback_days=90, llm=llm)

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result_without = evaluate_journal(path, lookback_days=90, llm=None)

        assert result_with["hit_rate_gesamt"] == result_without["hit_rate_gesamt"]
        assert (
            result_with["nach_aktion"]["KAUFEN"]["hit_rate"]
            == result_without["nach_aktion"]["KAUFEN"]["hit_rate"]
        )

    def test_single_row_hit_fields_present(self, tmp_path):
        """check_invalidation_hit-Ergebnis landet als invalidation_hit in der Zeile."""
        rows = [_make_row(ticker="AAA", invalidation="KGV über 25")]
        _write_journal(tmp_path, rows)
        prices = _make_prices(100, 60, drift=0.005)
        row = rows[0]
        eval_row = _evaluate_single(row, prices, 90)
        eval_row["_row"] = row
        check = check_invalidation_hit(row["invalidation"], eval_row, _mock_llm(True))
        assert check == (True, "KGV stieg über 25")


class TestInvalidationPriceContext:
    def test_context_renders_values(self):
        from concilium.evaluate import _invalidation_price_context

        ctx = _invalidation_price_context({
            "ticker": "AAPL", "action": "KAUFEN", "rendite_pct": -12.5,
            "ziel_erreicht": None, "stop_gerissen": True,
            "timestamp": "2026-09-01 10:00:00",
        })
        assert "Rendite im Bewertungszeitraum: -12.50 %" in ctx
        assert "Stop gerissen: ja" in ctx
        assert "Zielkurs erreicht: n/a" in ctx

    def test_context_never_crashes_on_empty(self):
        from concilium.evaluate import _invalidation_price_context

        ctx = _invalidation_price_context({})
        assert "n/a" in ctx


# --------------------------------------------------------------------------- #
# (e) Report
# --------------------------------------------------------------------------- #


class TestTrackRecordReport:
    def test_invalidation_section_rendered(self):
        eval_result = {
            "anzahl_entscheidungen": 2,
            "invalidation_hit_quote": 0.5,
            "invalidation_n": 2,
            "invalidation_hits": 1,
            "invalidation_nicht_bewertbar": 0,
            "invalidation_details": [
                {"ticker": "AAA", "timestamp": "2026-09-01 10:00:00",
                 "invalidiert": True, "begruendung": "KGV stieg", "invalidation": "KGV über 25"},
            ],
        }
        report = generate_track_record_report(eval_result)
        assert "## Invalidierungs-Trefferquote" in report
        assert "verändert NICHT die" in report
        assert "50.0 %" in report

    def test_no_section_when_no_data(self):
        report = generate_track_record_report({"anzahl_entscheidungen": 0})
        assert "## Invalidierungs-Trefferquote" not in report

    def test_section_with_nicht_bewertbar(self):
        report = generate_track_record_report({
            "anzahl_entscheidungen": 1,
            "invalidation_hit_quote": None,
            "invalidation_n": 0,
            "invalidation_hits": 0,
            "invalidation_nicht_bewertbar": 1,
            "invalidation_details": [],
        })
        assert "## Invalidierungs-Trefferquote" in report
        assert "Nicht bewertbar" in report

    def test_details_truncated_at_20(self):
        details = [
            {"ticker": f"T{i}", "timestamp": "2026-09-01 10:00:00",
             "invalidiert": False, "begruendung": "nein", "invalidation": "x"}
            for i in range(25)
        ]
        report = generate_track_record_report({
            "anzahl_entscheidungen": 25,
            "invalidation_hit_quote": 0.0,
            "invalidation_n": 25,
            "invalidation_hits": 0,
            "invalidation_nicht_bewertbar": 0,
            "invalidation_details": details,
        })
        assert "5 weitere Einträge gekürzt" in report

    def test_report_never_crashes_on_malformed_details(self):
        report = generate_track_record_report({
            "anzahl_entscheidungen": 1,
            "invalidation_hit_quote": None,
            "invalidation_n": 1,
            "invalidation_hits": 0,
            "invalidation_nicht_bewertbar": 0,
            "invalidation_details": [{"invalidiert": "kaputt", "begruendung": None}],
        })
        assert "## Invalidierungs-Trefferquote" in report


class TestPerAnalysisReport:
    def _base_result(self, invalidation: str | None = None) -> dict:
        result = {
            "ticker": "TEST",
            "data": {"fundamentals": {"name": "TestCo", "sector": "Tech"},
                     "technicals": {"current_price": 100}, "sentiment": {}, "news": []},
            "trade": {"aktion": "HALTEN", "rating": "HALTEN", "_raw": ""},
            "risk": {"risiko_score": 3, "empfehlung": "GENEHMIGT"},
            "final": {"entscheidung": "GENEHMIGT", "confidence": 4, "begründung": "Ok"},
            "debate": {"bull": {"_raw": "B"}, "bear": {"_raw": "B"}},
            "analysts": {
                "fundamental": {
                    "stimmung": "bullish", "score": 4,
                    "zusammenfassung": "Gut",
                    "invalidation": "KGV über 25",
                },
            },
            "no_llm": False,
        }
        if invalidation is not None:
            result["analysts"]["technical"] = {
                "stimmung": "bearish", "score": 2, "zusammenfassung": "Schlecht",
                "invalidation": invalidation,
            }
        return result

    def test_invalidation_section_rendered(self):
        report = generate_report(self._base_result(invalidation="RSI über 70"))
        assert "### Invalidierungs-Bedingungen" in report
        assert "**Fundamental:** KGV über 25" in report
        assert "**Technik:** RSI über 70" in report

    def test_no_section_without_invalidation(self):
        result = self._base_result()
        result["analysts"]["fundamental"]["invalidation"] = "   "
        report = generate_report(result)
        assert "### Invalidierungs-Bedingungen" not in report

    def test_legacy_result_without_field_no_crash(self):
        report = generate_report(self._base_result())
        assert "### Invalidierungs-Bedingungen" in report
        assert "KGV über 25" in report
