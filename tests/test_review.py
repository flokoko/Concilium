"""Tests für den Exit-Review-Modus (--review, review.py).

Alle Tests sind OFFLINE-fähig: kein yfinance, kein Netzwerk, kein LLM.
Depot-Positionen werden gemockt (fetch_portfolio_positions), die Pipeline
ebenfalls (run_pipeline).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from concilium.cli import main  # noqa: E402
from concilium.review import (  # noqa: E402
    derive_verkauf_empfehlung,
    run_review,
)


def _reports_dir() -> str:
    """Reports-Verzeichnis (repo-root/reports) — gleiche Logik wie cli.py."""
    cli_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
        "concilium",
        "cli.py",
    )
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(cli_path)), "..", "..", "reports")
    )


# --------------------------------------------------------------------------- #
# Mock-Positionen — analog _parse_positions-Ausgabe (portfolio_fit.py)
# --------------------------------------------------------------------------- #


def _mock_positions() -> list[dict]:
    """2 Aktien + 1 ETF, analog der _parse_positions-Ausgabe aus portfolio_fit.py."""
    return [
        {
            "name": "Apple Inc.",
            "ticker": "AAPL",
            "sheet_symbol": "AAPL",
            "type": "Aktie",
            "region": "USA",
            "depot_pct": 5.2,
            "value_eur": 5200.0,
            "_idx": 0,
        },
        {
            "name": "BASF SE",
            "ticker": "BAS.DE",
            "sheet_symbol": "BAS.DE",
            "type": "Aktie",
            "region": "Europa",
            "depot_pct": 2.0,
            "value_eur": 2000.0,
            "_idx": 1,
        },
        {
            "name": "iShares Core MSCI World UCITS ETF",
            "ticker": "IS3R.DE",
            "sheet_symbol": "IS3R.DE",
            "type": "ETF",
            "region": "Welt",
            "depot_pct": 30.0,
            "value_eur": 30000.0,
            "_idx": 2,
        },
    ]


def _mock_positions_3_aktien() -> list[dict]:
    """3 Aktien (unterschiedliche depot_pct) + 1 ETF — für max_positions-Tests."""
    return [
        {"name": "Apple", "ticker": "AAPL", "sheet_symbol": "AAPL", "type": "Aktie",
         "region": "USA", "depot_pct": 5.2, "value_eur": 5200.0, "_idx": 0},
        {"name": "BASF", "ticker": "BAS.DE", "sheet_symbol": "BAS.DE", "type": "Aktie",
         "region": "Europa", "depot_pct": 2.0, "value_eur": 2000.0, "_idx": 1},
        {"name": "Microsoft", "ticker": "MSFT", "sheet_symbol": "MSFT", "type": "Aktie",
         "region": "USA", "depot_pct": 8.0, "value_eur": 8000.0, "_idx": 2},
        {"name": "iShares World", "ticker": "IS3R.DE", "sheet_symbol": "IS3R.DE",
         "type": "ETF", "region": "Welt", "depot_pct": 30.0, "value_eur": 30000.0,
         "_idx": 3},
    ]


def _make_result(
    ticker: str, aktion: str = "HALTEN", entscheidung: str = "GENEHMIGT"
) -> dict:
    """Baut ein minimales Pipeline-Result-dict (LLM-Pfad)."""
    return {
        "ticker": ticker,
        "data": {
            "ticker": ticker,
            "fundamentals": {},
            "technicals": {},
            "sentiment": {},
            "news": [],
            "history": [],
        },
        "trade": {"aktion": aktion},
        "final": {"entscheidung": entscheidung},
        "no_llm": False,
    }


# --------------------------------------------------------------------------- #
# (a) run_review: nur Aktien werden analysiert
# --------------------------------------------------------------------------- #


class TestRunReviewAktienFilter:
    """run_review analysiert nur Positionen mit type == 'Aktie'."""

    def test_analyses_only_stocks_not_etf(self):
        """2 Aktien + 1 ETF → nur die 2 Aktien via run_pipeline analysiert."""
        analysed: list[str] = []

        def mock_run_pipeline(ticker, **kwargs):
            analysed.append(ticker)
            return _make_result(ticker)

        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.review.run_pipeline", side_effect=mock_run_pipeline):
                result = run_review(None)

        assert sorted(analysed) == ["AAPL", "BAS.DE"]
        assert "AAPL" in result["ergebnisse"]
        assert "BAS.DE" in result["ergebnisse"]
        # Der ETF ist NICHT in den Ergebnissen
        assert "IS3R.DE" not in result["ergebnisse"]
        # 1 Position übersprungen (der ETF)
        assert result["positions_uebersprungen"] == 1
        assert result["fehler"] == 0

    def test_empty_portfolio_returns_empty_results(self):
        """Fehlendes/leeres Depot → leere Ergebnisse, kein Absturz."""
        with patch("concilium.review.fetch_portfolio_positions", side_effect=lambda: []):
            with patch("concilium.review.run_pipeline") as mock_pipe:
                result = run_review(None)

        assert result["ergebnisse"] == {}
        assert result["positions_uebersprungen"] == 0
        assert result["fehler"] == 0
        mock_pipe.assert_not_called()

    def test_all_non_stocks_skipped(self):
        """Nur ETFs/Commodities → alle übersprungen, leere Ergebnisse, keine Pipeline-Calls."""
        positions = [
            {"name": "iShares MSCI EM", "ticker": "IEMA.DE", "sheet_symbol": "IEMA.DE",
             "type": "ETF", "region": "EM", "depot_pct": 12.0, "value_eur": 12000.0},
            {"name": "WisdomTree Gold", "ticker": "WGLD.L", "sheet_symbol": "WGLD.L",
             "type": "Commodity", "region": "Welt", "depot_pct": 3.0,
             "value_eur": 3000.0},
        ]
        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: positions):
            with patch("concilium.review.run_pipeline") as mock_pipe:
                result = run_review(None)

        assert result["ergebnisse"] == {}
        assert result["positions_uebersprungen"] == 2
        assert result["fehler"] == 0
        mock_pipe.assert_not_called()

    def test_pipeline_gets_kwargs_passed(self):
        """run_pipeline bekommt backtest/ensemble/ensemble_runs/resume/debate_rounds/peers durchgereicht."""
        captured: dict = {}

        def mock_run_pipeline(ticker, **kwargs):
            captured.update(kwargs)
            return _make_result(ticker)

        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.review.run_pipeline", side_effect=mock_run_pipeline):
                run_review(
                    None,
                    backtest=True,
                    ensemble=False,
                    ensemble_runs=5,
                    resume=True,
                    debate_rounds=2,
                    peers=["MSFT"],
                )

        assert captured.get("backtest") is True
        assert captured.get("ensemble") is False
        assert captured.get("ensemble_runs") == 5
        assert captured.get("resume") is True
        assert captured.get("debate_rounds") == 2
        assert captured.get("peers") == ["MSFT"]


# --------------------------------------------------------------------------- #
# (b) verkauf_empfehlung-Ableitung
# --------------------------------------------------------------------------- #


class TestDeriveVerkaufEmpfehlung:
    """derive_verkauf_empfehlung: deterministische Ableitung aus dem Result."""

    def test_verkaufen_true(self):
        assert derive_verkauf_empfehlung({"trade": {"aktion": "VERKAUFEN"}}) is True

    def test_stark_verkaufen_true(self):
        assert derive_verkauf_empfehlung(
            {"trade": {"aktion": "STARK VERKAUFEN"}}
        ) is True

    def test_abgelehnt_true(self):
        """ABGELEHNT bei Bestandsposition = 'nicht mehr halten' → Verkaufsempfehlung."""
        assert derive_verkauf_empfehlung({"final": {"entscheidung": "ABGELEHNT"}}) is True

    def test_halten_false(self):
        assert derive_verkauf_empfehlung(
            {"trade": {"aktion": "HALTEN"}, "final": {"entscheidung": "GENEHMIGT"}}
        ) is False

    def test_kaufen_false(self):
        assert derive_verkauf_empfehlung({"trade": {"aktion": "KAUFEN"}}) is False

    def test_stark_kaufen_false(self):
        assert derive_verkauf_empfehlung({"trade": {"aktion": "STARK KAUFEN"}}) is False

    def test_modifiziert_und_halten_false(self):
        assert derive_verkauf_empfehlung(
            {"trade": {"aktion": "HALTEN"}, "final": {"entscheidung": "MODIFIZIERT"}}
        ) is False

    def test_none_result_false(self):
        """None-Ergebnis (z.B. Pipeline-Fehler) → False, kein Absturz."""
        assert derive_verkauf_empfehlung(None) is False

    def test_empty_dict_false(self):
        assert derive_verkauf_empfehlung({}) is False

    def test_abgelehnt_beats_kaufen(self):
        """Selbst bei KAUFEN-Trade zählt ABGELEHNT (final zählt für Bestandsposition)."""
        assert derive_verkauf_empfehlung(
            {"trade": {"aktion": "KAUFEN"}, "final": {"entscheidung": "ABGELEHNT"}}
        ) is True


# --------------------------------------------------------------------------- #
# Ergebnis-Vertrag: {result, report, verkauf_empfehlung} pro Eintrag
# --------------------------------------------------------------------------- #


class TestRunReviewErgebnisVertrag:
    """Jedes analysierte Ergebnis enthält result, report und verkauf_empfehlung."""

    def test_entry_contains_result_report_verkauf(self):
        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.review.run_pipeline",
                       side_effect=lambda t, **kw: _make_result(t)):
                result = run_review(None)

        entry = result["ergebnisse"]["AAPL"]
        assert set(entry) >= {"result", "report", "verkauf_empfehlung"}
        assert isinstance(entry["report"], str) and entry["report"]
        assert isinstance(entry["result"], dict)
        assert entry["result"]["ticker"] == "AAPL"
        assert entry["verkauf_empfehlung"] is False

    def test_verkauf_empfehlung_flows_through(self):
        """VERKAUFEN-Result → verkauf_empfehlung=True im Eintrag, HALTEN → False."""

        def mock_run_pipeline(ticker, **kwargs):
            if ticker == "AAPL":
                return _make_result(ticker, aktion="VERKAUFEN")
            if ticker == "BAS.DE":
                return _make_result(ticker, aktion="HALTEN")
            return _make_result(ticker)

        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.review.run_pipeline", side_effect=mock_run_pipeline):
                result = run_review(None)

        assert result["ergebnisse"]["AAPL"]["verkauf_empfehlung"] is True
        assert result["ergebnisse"]["BAS.DE"]["verkauf_empfehlung"] is False

    def test_verkauf_empfehlung_from_abgelehnt(self):
        """ABGELEHNT-Entscheidung → verkauf_empfehlung=True im Eintrag."""

        def mock_run_pipeline(ticker, **kwargs):
            return _make_result(ticker, aktion="KAUFEN", entscheidung="ABGELEHNT")

        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.review.run_pipeline", side_effect=mock_run_pipeline):
                result = run_review(None)

        assert result["ergebnisse"]["AAPL"]["verkauf_empfehlung"] is True


# --------------------------------------------------------------------------- #
# (e) generate_report review_mode — Review-Hinweiszeile
# --------------------------------------------------------------------------- #


class TestGenerateReportReviewMode:
    """generate_report(review_mode=True) ergänzt die Review-Hinweiszeile."""

    def test_review_mode_adds_hint_line(self):
        from concilium.report import generate_report

        result = _make_result("AAPL")
        report = generate_report(result, review_mode=True)
        assert "REVIEW-MODUS" in report
        assert "Verkaufskandidaten" in report

    def test_default_review_mode_has_no_hint(self):
        """Default review_mode=False → kein Review-Hinweis (Rückwärtskompatibilität)."""
        from concilium.report import generate_report

        result = _make_result("AAPL")
        report = generate_report(result)
        assert "REVIEW-MODUS" not in report

    def test_review_mode_with_no_llm_result(self):
        """Auch ohne LLM-Abschnitte erscheint der Review-Hinweis."""
        from concilium.report import generate_report

        result = {
            "ticker": "AAPL",
            "data": {"ticker": "AAPL", "fundamentals": {}, "technicals": {},
                     "sentiment": {}, "news": [], "history": []},
            "no_llm": True,
        }
        report = generate_report(result, review_mode=True)
        assert "REVIEW-MODUS" in report
        assert "Verkaufskandidaten" in report


# --------------------------------------------------------------------------- #
# (c) Fehlertoleranz
# --------------------------------------------------------------------------- #


class TestRunReviewFehlertoleranz:
    """Ein fehlgeschlagener Ticker crasht den Review nicht."""

    def test_failed_ticker_does_not_crash(self):
        """Ein Ticker wirft Exception → Warnung, Fehlerzählung, Rest läuft weiter."""
        calls: list[str] = []

        def mock_run_pipeline(ticker, **kwargs):
            calls.append(ticker)
            if ticker == "BAS.DE":
                raise ValueError("Ungültiger Ticker")
            return _make_result(ticker)

        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.review.run_pipeline", side_effect=mock_run_pipeline):
                result = run_review(None)

        # Beide Aktien wurden versucht (Reihenfolge: größte zuerst)
        assert calls == ["AAPL", "BAS.DE"]
        # Der erfolgreiche ist da, der fehlgeschlagene nicht
        assert "AAPL" in result["ergebnisse"]
        assert "BAS.DE" not in result["ergebnisse"]
        assert result["fehler"] == 1

    def test_all_failed_still_returns(self):
        """Alle Ticker fehlgeschlagen → kein Crash, fehler=2, leere Ergebnisse."""

        def mock_run_pipeline(ticker, **kwargs):
            raise ValueError(f"Fehler für {ticker}")

        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.review.run_pipeline", side_effect=mock_run_pipeline):
                result = run_review(None)

        assert result["fehler"] == 2
        assert result["ergebnisse"] == {}
        assert result["positions_uebersprungen"] == 1

    def test_fetch_portfolio_crash_does_not_propagate(self):
        """Selbst wenn fetch_portfolio_positions unerwartet crasht → leeres Depot."""

        def mock_fetch():
            raise RuntimeError("Netzwerk-Tot")

        with patch("concilium.review.fetch_portfolio_positions", side_effect=mock_fetch):
            result = run_review(None)

        assert result["ergebnisse"] == {}
        assert result["fehler"] == 0


# --------------------------------------------------------------------------- #
# (d) max_positions — größte Positionen zuerst (nach depot_pct)
# --------------------------------------------------------------------------- #


class TestRunReviewMaxPositions:
    """max_positions begrenzt die Anzahl der analysierten Aktien (größte zuerst)."""

    def test_max_positions_limits_count_largest_first(self):
        """3 Aktien, max_positions=2 → MSFT (8.0) + AAPL (5.2) analysiert, BAS.DE übersprungen."""
        analysed: list[str] = []

        def mock_run_pipeline(ticker, **kwargs):
            analysed.append(ticker)
            return _make_result(ticker)

        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions_3_aktien()):
            with patch("concilium.review.run_pipeline", side_effect=mock_run_pipeline):
                result = run_review(None, max_positions=2)

        assert analysed == ["MSFT", "AAPL"]  # größe zuerst, BAS.DE (2.0) fliegt raus
        assert set(result["ergebnisse"]) == {"MSFT", "AAPL"}
        assert result["positions_uebersprungen"] == 2  # BAS.DE + ETF

    def test_max_positions_none_analyses_all(self):
        """max_positions=None → alle Aktien werden analysiert."""
        analysed: list[str] = []

        def mock_run_pipeline(ticker, **kwargs):
            analysed.append(ticker)
            return _make_result(ticker)

        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions_3_aktien()):
            with patch("concilium.review.run_pipeline", side_effect=mock_run_pipeline):
                result = run_review(None)

        assert len(analysed) == 3
        assert result["positions_uebersprungen"] == 1

    def test_max_positions_larger_than_count(self):
        """max_positions größer als Aktienanzahl → alle Aktien, kein Fehler."""
        analysed: list[str] = []

        def mock_run_pipeline(ticker, **kwargs):
            analysed.append(ticker)
            return _make_result(ticker)

        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions_3_aktien()):
            with patch("concilium.review.run_pipeline", side_effect=mock_run_pipeline):
                result = run_review(None, max_positions=99)

        assert len(analysed) == 3
        assert result["positions_uebersprungen"] == 1

    def test_max_positions_zero(self):
        """max_positions=0 → nichts wird analysiert, alles übersprungen."""
        with patch("concilium.review.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions_3_aktien()):
            with patch("concilium.review.run_pipeline") as mock_pipe:
                result = run_review(None, max_positions=0)

        assert result["ergebnisse"] == {}
        assert result["positions_uebersprungen"] == 4
        mock_pipe.assert_not_called()


# --------------------------------------------------------------------------- #
# (f) CLI: --review Flag — Mutual Exclusion
# --------------------------------------------------------------------------- #


class TestReviewCliMutualExclusion:
    """--review schließt sich mit --ticker/--tickers/--portfolio/--watchlist aus."""

    def test_review_and_ticker_error(self):
        """--review + --ticker → Fehler, Exit 1."""
        from concilium.cli import main

        assert main(["--review", "--ticker", "AAPL"]) == 1

    def test_review_and_tickers_error(self):
        """--review + --tickers → Fehler, Exit 1."""
        from concilium.cli import main

        assert main(["--review", "--tickers", "AAPL,NVDA"]) == 1

    def test_review_and_portfolio_error(self):
        """--review + --portfolio → Fehler, Exit 1."""
        from concilium.cli import main

        assert main(["--review", "--portfolio", "AAPL,NVDA"]) == 1

    def test_review_and_watchlist_error(self):
        """--review + --watchlist → Fehler, Exit 1."""
        from concilium.cli import main

        assert main(["--review", "--watchlist"]) == 1

    def test_review_ticker_order_independent(self):
        """Reihenfolge egal: --ticker + --review → ebenfalls Fehler, Exit 1."""
        from concilium.cli import main

        assert main(["--ticker", "AAPL", "--review"]) == 1


# --------------------------------------------------------------------------- #
# (g) CLI: --review Modus — evaluate vorn, run_review, Summary, Reports
# --------------------------------------------------------------------------- #


def _run_cli_review(monkeypatch, tmp_path, *, hooks: dict, extra_args=None):
    """Hilfsfunktion: führt main(['--review', ...]) mit gemockten Abhängigkeiten aus.

    hooks: dict mit mock_fn für evaluate_journal/_write_calibration_json/
    run_review/generate_track_record_report (Werte: callables oder None).
    Setzt CONCILIUM_STATE_DIR auf ein tmp-Verzeichnis (Schutz der echten
    calibration.json) und gibt (exit_code, call_order) zurück.
    """
    from concilium import cli as cli_mod

    call_order: list[str] = []

    def default_eval(*args, **kwargs):
        call_order.append("evaluate_journal")
        return {"anzahl_entscheidungen": 5, "hit_rate_gesamt": 0.4, "nach_aktion": {}}

    def default_cal(eval_result, **kwargs):
        call_order.append("_write_calibration_json")

    def default_track(eval_result):
        call_order.append("generate_track_record_report")
        return "# Track Record Report"

    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("CONCILIUM_STATE_DIR", str(state_dir))

    def _wrap(name: str, fn):
        """Wickelt einen User-Hook so, dass er seinen Namen in call_order einträgt."""
        if fn is None:
            return None

        def wrapped(*a, **kw):
            call_order.append(name)
            return fn(*a, **kw)

        return wrapped

    patches = [
        patch.object(
            cli_mod,
            "evaluate_journal",
            side_effect=_wrap("evaluate_journal", hooks.get("evaluate_journal"))
            or default_eval,
        ),
        patch.object(
            cli_mod,
            "_write_calibration_json",
            side_effect=_wrap("_write_calibration_json", hooks.get("_write_calibration_json"))
            or default_cal,
        ),
        patch.object(
            cli_mod,
            "generate_track_record_report",
            side_effect=_wrap(
                "generate_track_record_report", hooks.get("generate_track_record_report")
            )
            or default_track,
        ),
        patch.object(cli_mod, "LLMClient"),
    ]
    if hooks.get("run_review"):
        patches.append(
            patch.object(cli_mod, "run_review", side_effect=_wrap("run_review", hooks["run_review"]))
        )
    if hooks.get("generate_report"):
        patches.append(patch.object(cli_mod, "generate_report",
                                    side_effect=hooks["generate_report"]))

    import contextlib

    argv = ["--review"] + (extra_args or [])
    with contextlib.ExitStack() as stack:
        for p in patches:
            p = p.__enter__()
            stack.callback(p.__exit__, None, None, None)
        code = main(argv)

    return code, call_order


class TestReviewCliMode:
    """--review: evaluate vorn, run_review danach, Zusammenfassung + Report-Dateien."""

    def test_review_runs_evaluate_before_run_review(self, tmp_path, monkeypatch):
        """evaluate_journal + calibration VOR run_review (damit Feedback aktuell)."""
        code, call_order = _run_cli_review(
            monkeypatch,
            tmp_path,
            hooks={
                "run_review": lambda llm, **kw: {
                    "ergebnisse": {},
                    "positions_uebersprungen": 0,
                    "fehler": 0,
                },
            },
        )
        assert code == 0
        assert call_order[:2] == ["evaluate_journal", "_write_calibration_json"]
        assert "run_review" in call_order
        assert call_order.index("evaluate_journal") < call_order.index("run_review")

    def test_review_passes_max_positions_flag(self, tmp_path, monkeypatch):
        """--max-positions 5 → run_review(max_positions=5)."""
        captured: dict = {}

        def mock_run_review(llm, **kw):
            captured.update(kw)
            return {"ergebnisse": {}, "positions_uebersprungen": 0, "fehler": 0}

        code, _ = _run_cli_review(
            monkeypatch, tmp_path,
            hooks={"run_review": mock_run_review},
            extra_args=["--max-positions", "5"],
        )
        assert code == 0
        assert captured.get("max_positions") == 5

    def test_review_summary_lists_positions_sorted_by_depot_pct(
        self, tmp_path, monkeypatch, capsys
    ):
        """Zusammenfassung: alle Positionen, sortiert nach depot_pct, mit Status-Symbol."""
        review_result = {
            "ergebnisse": {
                "AAPL": {
                    "result": _make_result("AAPL", aktion="HALTEN"),
                    "report": "# AAPL",
                    "verkauf_empfehlung": False,
                    "depot_pct": 5.2,
                    "name": "Apple",
                },
                "MSFT": {
                    "result": _make_result("MSFT", aktion="VERKAUFEN"),
                    "report": "# MSFT",
                    "verkauf_empfehlung": True,
                    "depot_pct": 8.0,
                    "name": "Microsoft",
                },
                "BAS.DE": {
                    "result": _make_result("BAS.DE", aktion="HALTEN",
                                           entscheidung="ABGELEHNT"),
                    "report": "# BAS",
                    "verkauf_empfehlung": True,
                    "depot_pct": 2.0,
                    "name": "BASF",
                },
            },
            "positions_uebersprungen": 1,
            "fehler": 0,
            "gesamt_positionen": 4,
        }

        code, _ = _run_cli_review(
            monkeypatch, tmp_path, hooks={"run_review": lambda llm, **kw: review_result}
        )
        assert code == 0

        captured = capsys.readouterr()
        out = captured.out + captured.err  # Zusammenfassung geht auf stderr
        # Summary-Zeilen haben das eindeutige Format "TICKER (Name): X% Depot"
        # — Report-Texte könnten die Ticker ebenfalls enthalten.
        assert out.index("MSFT (Microsoft)") < out.index("AAPL (Apple)")
        assert out.index("AAPL (Apple)") < out.index("BAS.DE (BASF)")
        # Status-Symbole: 🔴 für Verkaufsempfehlung, ✅ für behalten
        msft_line = next(
            line for line in out.splitlines() if "MSFT" in line and "🔴" in line
        )
        aapl_line = next(
            line for line in out.splitlines() if "AAPL" in line and "✅" in line
        )
        bas_line = next(
            line for line in out.splitlines() if "BAS.DE" in line and "🔴" in line
        )
        assert msft_line and aapl_line and bas_line

    def test_review_saves_reports_as_review_ticker_timestamp(
        self, tmp_path, monkeypatch
    ):
        """Reports werden als reports/review_{TICKER}_{timestamp}.md gespeichert."""
        reports_dir = _reports_dir()

        review_result = {
            "ergebnisse": {
                "AAPL": {
                    "result": _make_result("AAPL"),
                    "report": "# AAPL Review Report",
                    "verkauf_empfehlung": False,
                    "depot_pct": 5.2,
                    "name": "Apple",
                },
            },
            "positions_uebersprungen": 0,
            "fehler": 0,
            "gesamt_positionen": 3,
        }

        code, _ = _run_cli_review(
            monkeypatch, tmp_path, hooks={"run_review": lambda llm, **kw: review_result}
        )
        assert code == 0

        saved = [f for f in os.listdir(reports_dir) if f.startswith("review_AAPL_")]
        assert saved, f"Kein review_AAPL_*-Report gefunden in {reports_dir}"
        for fname in saved:
            os.remove(os.path.join(reports_dir, fname))

    def test_review_no_positions_exit_0(self, tmp_path, monkeypatch):
        """Leeres Depot → Exit 0 (kein Fehler), Meldung statt Crash."""
        code, _ = _run_cli_review(
            monkeypatch, tmp_path,
            hooks={"run_review": lambda llm, **kw: {
                "ergebnisse": {}, "positions_uebersprungen": 0,
                "fehler": 0, "gesamt_positionen": 0,
            }},
        )
        assert code == 0

    def test_review_run_review_crash_caught(self, tmp_path, monkeypatch):
        """run_review selbst crasht → CLI fängt ab, Exit 1, kein Traceback-Abbruch."""

        def boom(llm, **kw):
            raise RuntimeError("Katastrophe")

        code, _ = _run_cli_review(monkeypatch, tmp_path, hooks={"run_review": boom})
        assert code == 1
