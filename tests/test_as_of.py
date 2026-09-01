"""Tests für Phase 4: Analyse-Datum pinnen via as_of / --date.

Abgedeckte Verhaltensweisen:
  (a) as_of beschränkt die Kurs-Historie (nur Daten <= as_of)
  (b) current_price + technische Indikatoren werden aus der beschränkten
      Historie berechnet (Stand zu as_of, nicht heute)
  (c) Cache-Key berücksichtigt as_of (kein falscher Cache-Treffer)
  (d) ungültiges as_of → ValueError mit deutscher Meldung, kein Crash
  (e) ohne as_of → bisheriges Verhalten (volle Historie, as_of=None)

Alle Tests laufen offline: yfinance + Social-Quellen + Makro werden gemockt.
"""

from __future__ import annotations

import inspect
import os
import sys
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# src zum Pfad hinzufügen
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

from concilium.data import (  # noqa: E402
    _load_cache,
    _save_cache,
    collect_ticker_data,
)

# Fixtures / Helfer -----------------------------------------------------------


_HIST_END = "2026-01-15"  # letzter Kurs-Tag der synthetischen Historie
_AS_OF = "2025-12-15"  # gepinntes Analysedatum (Montag, Werktag)


def _make_hist(n_days: int = 260) -> pd.DataFrame:
    """Synthetische 1y-Kurshistorie (werktäglich, endet an _HIST_END).

    Schließt steigend (100.0 + 0.5 * i), damit jeder Close eindeutig ist und
    der letzte Close vor as_of klar vom heutigen Close unterscheidbar ist.
    """
    idx = pd.bdate_range(end=_HIST_END, periods=n_days)
    closes = [100.0 + 0.5 * i for i in range(n_days)]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000_000] * n_days,
        },
        index=idx,
    )


def _make_mock_ticker(hist: pd.DataFrame) -> MagicMock:
    """yf.Ticker-Mock, der die gegebene Historie liefert (offline)."""
    t = MagicMock()
    t.history.return_value = hist
    t.info = {
        "longName": "Test Corp",
        "shortName": "TEST",
        "sector": "Tech",
        "industry": "Semis",
        "currency": "USD",
    }
    t.news = []
    return t


def _collect(hist: pd.DataFrame, *args: object, **kwargs: object) -> dict:
    """collect_ticker_data mit gemocktem yfinance + Social-/Makro-Quellen.

    Der Tages-Cache bleibt deaktiviert (conftest) und _save_cache wird
    gepatcht — außer in den Cache-Tests, die das explizit anders machen.
    """
    mock_ticker = _make_mock_ticker(hist)
    with ExitStack() as stack:
        stack.enter_context(
            patch("concilium.data.yf.Ticker", return_value=mock_ticker)
        )
        stack.enter_context(patch("concilium.data._fetch_google_news", return_value=[]))
        stack.enter_context(patch("concilium.data._fetch_stocktwits", return_value=[]))
        stack.enter_context(patch("concilium.data._fetch_reddit", return_value=[]))
        stack.enter_context(patch("concilium.data._fetch_macro_data", return_value={}))
        stack.enter_context(patch("concilium.data._save_cache"))
        result = collect_ticker_data(*args, **kwargs)
    return result


# (a) + (b) Historie-Beschränkung, current_price, Indikatoren -----------------


class TestAsOfLimitsHistory:
    """as_of beschränkt die Historie und alle daraus abgeleiteten Werte."""

    def test_history_only_contains_rows_up_to_as_of(self):
        """(a) history enthält nur Zeilen mit Datum <= as_of, lückenlos."""
        hist = _make_hist()
        result = _collect(hist, "TEST", as_of=_AS_OF)

        kept = [
            d.strftime("%Y-%m-%d") for d in hist.index if d.strftime("%Y-%m-%d") <= _AS_OF
        ]
        got = [r["date"] for r in result["history"]]

        assert got, "Historie darf mit as_of nicht leer sein"
        assert got == kept  # exakt die Zeilen <= as_of, in Reihenfolge
        assert max(got) == _AS_OF  # letzter Eintrag ist exakt as_of

    def test_current_price_is_last_close_before_as_of(self):
        """(b) current_price = letzter Close VOR as_of, nicht der heutige."""
        hist = _make_hist()
        result = _collect(hist, "TEST", as_of=_AS_OF)

        last_close_before = float(hist.loc[:_AS_OF, "Close"].iloc[-1])
        full_last_close = float(hist["Close"].iloc[-1])

        assert last_close_before != full_last_close  # Testdaten schneiden ab
        assert result["technicals"]["current_price"] == pytest.approx(last_close_before)

    def test_indicators_computed_from_limited_history(self):
        """(b) SMA50 wird aus der beschränkten Historie berechnet."""
        hist = _make_hist()
        result = _collect(hist, "TEST", as_of=_AS_OF)

        closes_kept = hist.loc[:_AS_OF, "Close"]
        expected_sma50 = float(closes_kept.tail(50).mean())
        assert result["technicals"]["sma50"] == pytest.approx(expected_sma50)

    def test_as_of_in_result_dict(self):
        """as_of wird zur Transparenz ins result-dict aufgenommen."""
        result = _collect(_make_hist(), "TEST", as_of=_AS_OF)
        assert result["as_of"] == _AS_OF


# (c) Cache-Key berücksichtigt as_of ------------------------------------------


class TestAsOfCache:
    """Cache darf gepinnte Läufe nie mit 'heute' (oder anderem as_of) teilen."""

    def test_cache_roundtrip_not_shared_across_as_of(self, tmp_path, monkeypatch):
        """Gleiches as_of → Treffer; anderes as_of / kein as_of → Miss."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", str(tmp_path))
        data = {"ticker": "TEST", "technicals": {"current_price": 100.0}, "history": []}

        _save_cache("TEST", data, today_key="2026-01-01", as_of=_AS_OF)

        # Gleicher as_of → Treffer
        hit = _load_cache("TEST", today_key="2026-01-01", as_of=_AS_OF)
        assert hit is not None

        # Anderes as_of → KEIN Treffer
        assert _load_cache("TEST", today_key="2026-01-01", as_of="2025-12-20") is None

        # Ohne as_of → KEIN Treffer (gepinnter Stand darf nicht als
        # "heute"-Stand durchgehen)
        assert _load_cache("TEST", today_key="2026-01-01", as_of=None) is None

    def test_collect_with_as_of_does_not_reuse_today_cache(self, tmp_path, monkeypatch):
        """End-to-End: Cache-Miss bei as_of-Wechsel, Treffer bei gleichem as_of."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", str(tmp_path))
        hist = _make_hist()
        mock_ticker = _make_mock_ticker(hist)

        def _do(as_of: str | None) -> dict:
            with ExitStack() as stack:
                stack.enter_context(
                    patch("concilium.data.yf.Ticker", return_value=mock_ticker)
                )
                stack.enter_context(
                    patch("concilium.data._fetch_google_news", return_value=[])
                )
                stack.enter_context(
                    patch("concilium.data._fetch_stocktwits", return_value=[])
                )
                stack.enter_context(
                    patch("concilium.data._fetch_reddit", return_value=[])
                )
                stack.enter_context(
                    patch("concilium.data._fetch_macro_data", return_value={})
                )
                return collect_ticker_data("TEST", as_of=as_of)

        # 1. Lauf ohne as_of → Fetch + normaler Tages-Cache-Eintrag
        r1 = _do(None)
        assert r1["as_of"] is None
        n_after_first = mock_ticker.history.call_count
        assert n_after_first == 1

        # 2. Lauf MIT as_of → darf den without-as_of-Cache NICHT treffen
        r2 = _do(_AS_OF)
        assert mock_ticker.history.call_count == n_after_first + 1
        assert r2["as_of"] == _AS_OF

        # 3. Lauf mit gleichem as_of → Cache-Treffer (kein erneuter Fetch)
        r3 = _do(_AS_OF)
        assert mock_ticker.history.call_count == n_after_first + 1
        assert r3["as_of"] == _AS_OF
        assert r3["technicals"]["current_price"] == pytest.approx(
            r2["technicals"]["current_price"]
        )


# (d) Ungültiges as_of → ValueError -------------------------------------------


class TestAsOfValidation:
    """Ungültiges as_of → ValueError mit deutscher Meldung (kein Crash)."""

    def test_unparseable_format_raises(self):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            _collect(_make_hist(), "TEST", as_of="15.12.2025")

    def test_impossible_date_raises(self):
        """Format ok, aber kein reales Datum (Monat 13) → ValueError."""
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            _collect(_make_hist(), "TEST", as_of="2025-13-45")

    def test_future_date_raises(self):
        future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
        with pytest.raises(ValueError, match="Zukunft"):
            _collect(_make_hist(), "TEST", as_of=future)

    def test_date_before_first_close_raises(self):
        """as_of vor dem ersten verfügbaren Kurs → ValueError."""
        with pytest.raises(ValueError, match="vor dem ersten"):
            _collect(_make_hist(), "TEST", as_of="2024-01-01")


# (e) Ohne as_of → bisheriges Verhalten ---------------------------------------


class TestDefaultBehaviorUnchanged:
    """Ohne as_of: volle Historie, heutiger Kurs, as_of=None im result."""

    def test_no_as_of_full_history_and_today_price(self):
        hist = _make_hist()
        result = _collect(hist, "TEST")

        assert result["as_of"] is None
        assert len(result["history"]) == len(hist)
        assert result["technicals"]["current_price"] == pytest.approx(
            float(hist["Close"].iloc[-1])
        )

    def test_signature_backward_compatible(self):
        """collect_ticker_data(ticker, peers=None, as_of=None) — bestehende
        Aufrufe bleiben unverändert gültig."""
        sig = inspect.signature(collect_ticker_data)
        assert list(sig.parameters) == ["ticker", "peers", "as_of"]
        assert sig.parameters["peers"].default is None
        assert sig.parameters["as_of"].default is None


# Durchreichung: Pipeline / Portfolio / CLI / Report ---------------------------


class TestPipelineAsOfPassthrough:
    """run_pipeline reicht as_of an collect_ticker_data durch."""

    def test_run_pipeline_passes_as_of(self):
        from concilium.pipeline import run_pipeline

        with patch("concilium.pipeline.collect_ticker_data") as mock_collect, patch(
            "concilium.pipeline.save_checkpoint"
        ), patch("concilium.pipeline.clear_checkpoint"):
            mock_collect.return_value = {
                "ticker": "TEST",
                "fundamentals": {},
                "technicals": {},
                "history": [],
                "sentiment": {},
                "news": [],
                "macro": {},
                "peers": [],
                "as_of": _AS_OF,
            }
            run_pipeline("TEST", llm=None, as_of=_AS_OF)

        assert mock_collect.call_args.kwargs.get("as_of") == _AS_OF

    def test_run_pipeline_default_as_of_none(self):
        from concilium.pipeline import run_pipeline

        with patch("concilium.pipeline.collect_ticker_data") as mock_collect, patch(
            "concilium.pipeline.save_checkpoint"
        ), patch("concilium.pipeline.clear_checkpoint"):
            mock_collect.return_value = {
                "ticker": "TEST",
                "fundamentals": {},
                "technicals": {},
                "history": [],
                "sentiment": {},
                "news": [],
                "macro": {},
                "peers": [],
                "as_of": None,
            }
            run_pipeline("TEST", llm=None)

        assert mock_collect.call_args.kwargs.get("as_of") is None

    def test_run_portfolio_passes_as_of(self):
        from concilium.pipeline import run_portfolio

        with patch("concilium.pipeline.run_pipeline") as mock_run, patch(
            "concilium.pipeline.fetch_portfolio_positions", return_value=[]
        ), patch(
            "concilium.portfolio_analysis.run_portfolio_analysis", return_value={}
        ):
            mock_run.return_value = {
                "ticker": "AAPL",
                "data": {"fundamentals": {}, "technicals": {}, "history": []},
                "no_llm": True,
            }

            run_portfolio(["AAPL"], llm=None, as_of=_AS_OF)

        assert mock_run.call_args.kwargs.get("as_of") == _AS_OF

    def test_resume_with_different_as_of_restarts(self):
        """Resume-Checkpoint mit anderem as_of darf nicht wiederverwendet
        werden — collect_ticker_data muss mit dem neuen as_of laufen."""
        from concilium.pipeline import run_pipeline

        stale = {
            "ticker": "TEST",
            "data": {"ticker": "TEST", "as_of": None, "fundamentals": {},
                     "technicals": {}, "history": [], "sentiment": {},
                     "news": [], "macro": {}, "peers": []},
            "_completed_steps": ["data"],
            "no_llm": True,
        }
        with patch(
            "concilium.pipeline.load_checkpoint", return_value=stale
        ) as mock_load, patch(
            "concilium.pipeline.collect_ticker_data"
        ) as mock_collect, patch(
            "concilium.pipeline.save_checkpoint"
        ), patch(
            "concilium.pipeline.clear_checkpoint"
        ):
            mock_collect.return_value = {
                "ticker": "TEST",
                "fundamentals": {},
                "technicals": {},
                "history": [],
                "sentiment": {},
                "news": [],
                "macro": {},
                "peers": [],
                "as_of": _AS_OF,
            }
            result = run_pipeline("TEST", llm=None, resume=True, as_of=_AS_OF)

        mock_load.assert_called_once()
        # Checkpoint wurde verworfen → Daten wurden neu gesammelt (mit as_of)
        assert mock_collect.call_args.kwargs.get("as_of") == _AS_OF
        assert result["data"]["as_of"] == _AS_OF


class TestCliDateFlag:
    """CLI: --date wird validiert und an run_pipeline durchgereicht."""

    def test_invalid_format_fails_fast(self, capsys):
        from concilium.cli import main

        with patch("concilium.cli.run_pipeline") as mock_run:
            rc = main(["--ticker", "AAPL", "--no-llm", "--date", "15.12.2025"])

        assert rc == 1
        assert "YYYY-MM-DD" in capsys.readouterr().err
        mock_run.assert_not_called()

    def test_date_with_evaluate_forbidden(self, capsys):
        from concilium.cli import main

        rc = main(["--evaluate", "--date", _AS_OF])

        assert rc == 1
        err = capsys.readouterr().err
        assert "--date" in err and "--evaluate" in err

    def test_single_mode_passes_date_to_pipeline(self):
        """--ticker … --date YYYY-MM-DD → run_pipeline(as_of=…)."""
        from concilium.cli import main

        class _FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):  # noqa: ANN001
                return datetime(2025, 1, 1, 12, 0)

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target = os.path.join(repo_root, "reports", "TEST_20250101_1200.md")
        try:
            with patch("concilium.cli.run_pipeline") as mock_run, patch(
                "concilium.cli.generate_report", return_value="# Report"
            ), patch("concilium.cli.datetime", _FixedDatetime):
                mock_run.return_value = {
                    "ticker": "TEST",
                    "no_llm": True,
                    "data": {"fundamentals": {}, "technicals": {}, "history": []},
                }
                rc = main(["--ticker", "TEST", "--no-llm", "--date", _AS_OF])

            assert rc == 0
            assert mock_run.call_args.kwargs.get("as_of") == _AS_OF
        finally:
            if os.path.isfile(target):
                os.remove(target)


class TestReportAsOfHeader:
    """Report-Header: gepinntes Analysedatum wird ausgewiesen."""

    @staticmethod
    def _result(as_of: str | None) -> dict:
        data: dict = {
            "fundamentals": {
                "name": "Test Corp",
                "sector": "Tech",
                "industry": "Semis",
                "currency": "USD",
                "eur_risiko": False,
            },
            "technicals": {"current_price": 100.0, "macd": {}, "bollinger": {}},
            "sentiment": {},
            "news": [],
            "news_with_dates": [],
            "news_source": "none",
            "macro": {},
            "peers": [],
            "history": [],
            "data_warnings": [],
        }
        if as_of is not None:
            data["as_of"] = as_of
        return {"ticker": "TEST", "no_llm": True, "data": data}

    def test_report_contains_pinned_date_line(self):
        from concilium.report import generate_report

        report = generate_report(self._result(_AS_OF), reports_dir=None)
        assert f"Analysedatum (gepinnt): {_AS_OF}" in report
        assert "Fundamentals/Makro/News aktuell" in report

    def test_report_without_as_of_has_no_line(self):
        from concilium.report import generate_report

        report = generate_report(self._result(None), reports_dir=None)
        assert "Analysedatum (gepinnt)" not in report

    def test_report_missing_as_of_key_no_line(self):
        """Alte results ohne as_of-Key → keine Zeile, kein Crash."""
        from concilium.report import generate_report

        result = self._result(None)
        result["data"].pop("as_of", None)
        report = generate_report(result, reports_dir=None)
        assert "Analysedatum (gepinnt)" not in report
