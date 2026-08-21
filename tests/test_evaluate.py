"""Tests für evaluate.py — Track-Record-Evaluierung.

Alle Tests sind OFFLINE-fähig: yfinance wird gemockt, keine Netzwerkzugriffe.
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

# src zum Pfad hinzufügen
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.evaluate import (  # noqa: E402
    _evaluate_single,
    evaluate_journal,
)
from concilium.report import generate_track_record_report  # noqa: E402

# --------------------------------------------------------------------------- #
# Hilfsfunktionen: Mock-Kursdaten und Journal-CSV erstellen
# --------------------------------------------------------------------------- #


def _make_prices(start_price: float, n_days: int, drift: float = 0.0) -> list[dict]:
    """Erzeugt eine Liste von Preis-Dicts für n_days Tage ab vor 60 Tagen.

    drift: tägliche prozentuale Veränderung (0.01 = +1%/Tag).
    """
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
    from concilium.journal import JOURNAL_HEADER

    path = str(tmp_path / "decisions.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
        writer.writeheader()
        for row in rows:
            full_row = {k: row.get(k, "") for k in JOURNAL_HEADER}
            writer.writerow(full_row)
    return path


def _make_journal_row(
    ticker: str = "AAPL",
    action: str = "KAUFEN",
    target: str = "",
    stop: str = "",
    confidence: str = "4",
    portfolio_fit_score: str = "",
    timestamp: str = "",
) -> dict:
    """Erzeugt eine Journal-Zeile mit Defaults."""
    if not timestamp:
        timestamp = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "ticker": ticker,
        "action": action,
        "target": target,
        "stop": stop,
        "confidence": confidence,
        "portfolio_fit_score": portfolio_fit_score,
        "timestamp": timestamp,
    }


# --------------------------------------------------------------------------- #
# Tests: leere/fehlende Journal-Datei
# --------------------------------------------------------------------------- #


class TestEmptyAndMissing:
    """Testet leere und fehlende Journal-Dateien."""

    def test_missing_file_returns_empty(self, tmp_path):
        """Fehlende Datei → leeres Ergebnis, kein Crash."""
        result = evaluate_journal(str(tmp_path / "nichtexistent.csv"))
        assert result["anzahl_entscheidungen"] == 0
        assert result["hit_rate_gesamt"] is None
        assert result["fehler"] == []

    def test_empty_file_returns_empty(self, tmp_path):
        """Leere CSV (nur Header) → leeres Ergebnis."""
        path = _write_journal(tmp_path, [])
        result = evaluate_journal(path)
        assert result["anzahl_entscheidungen"] == 0
        assert result["hit_rate_gesamt"] is None

    def test_file_with_only_header(self, tmp_path):
        """Datei mit nur Header-Zeile → 0 Entscheidungen."""
        path = str(tmp_path / "only_header.csv")
        from concilium.journal import JOURNAL_HEADER

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
            writer.writeheader()
        result = evaluate_journal(path)
        assert result["anzahl_entscheidungen"] == 0


# --------------------------------------------------------------------------- #
# Tests: KAUFEN-Bewertung
# --------------------------------------------------------------------------- #


class TestKaufenBewertung:
    """Testet die KAUFEN-Bewertung gegen steigende/fallende Kurse."""

    def test_kaufen_rising_price_is_hit(self):
        """KAUFEN mit steigendem Kurs → Hit (rendite > 0)."""
        prices = _make_prices(100, 60, drift=0.005)  # ~+0.5%/Tag → steigend
        row = _make_journal_row(action="KAUFEN", timestamp="2026-01-01 10:00:00")
        result = _evaluate_single(row, prices, 90)
        assert result["hit"] is True
        assert result["rendite_pct"] is not None
        assert result["rendite_pct"] > 0

    def test_kaufen_falling_price_is_miss(self):
        """KAUFEN mit fallendem Kurs → No-Hit (rendite < 0)."""
        prices = _make_prices(100, 60, drift=-0.005)  # fallend
        row = _make_journal_row(action="KAUFEN", timestamp="2026-01-01 10:00:00")
        result = _evaluate_single(row, prices, 90)
        assert result["hit"] is False
        assert result["rendite_pct"] is not None
        assert result["rendite_pct"] < 0


# --------------------------------------------------------------------------- #
# Tests: VERKAUFEN-Bewertung
# --------------------------------------------------------------------------- #


class TestVerkaufenBewertung:
    """Testet die VERKAUFEN-Bewertung."""

    def test_verkaufen_falling_price_is_hit(self):
        """VERKAUFEN mit fallendem Kurs → Hit (rendite invertiert > 0)."""
        prices = _make_prices(100, 60, drift=-0.005)
        row = _make_journal_row(action="VERKAUFEN", timestamp="2026-01-01 10:00:00")
        result = _evaluate_single(row, prices, 90)
        assert result["hit"] is True
        assert result["rendite_pct"] is not None
        assert result["rendite_pct"] > 0

    def test_verkaufen_rising_price_is_miss(self):
        """VERKAUFEN mit steigendem Kurs → No-Hit."""
        prices = _make_prices(100, 60, drift=0.005)
        row = _make_journal_row(action="VERKAUFEN", timestamp="2026-01-01 10:00:00")
        result = _evaluate_single(row, prices, 90)
        assert result["hit"] is False


# --------------------------------------------------------------------------- #
# Tests: HALTEN-Bewertung
# --------------------------------------------------------------------------- #


class TestHaltenBewertung:
    """Testet die HALTEN-Bewertung."""

    def test_halten_stable_is_hit(self):
        """HALTEN mit stabilem Kurs (±5%) → Hit."""
        prices = _make_prices(100, 60, drift=0.0)  # stabil
        row = _make_journal_row(action="HALTEN", timestamp="2026-01-01 10:00:00")
        result = _evaluate_single(row, prices, 90)
        assert result["hit"] is True

    def test_halten_big_drop_is_miss(self):
        """HALTEN mit großem Kurssturz → No-Hit."""
        prices = _make_prices(100, 60, drift=-0.02)  # stark fallend
        row = _make_journal_row(action="HALTEN", timestamp="2026-01-01 10:00:00")
        result = _evaluate_single(row, prices, 90)
        assert result["hit"] is False


# --------------------------------------------------------------------------- #
# Tests: Zielkurs und Stop
# --------------------------------------------------------------------------- #


class TestZielkursAndStop:
    """Testet Zielkurs-Ereichung und Stop-Verletzung."""

    def test_zielkurs_erreicht(self):
        """KAUFEN: Zielkurs wurde erreicht (High ≥ target)."""
        prices = _make_prices(100, 60, drift=0.01)  # steigend → High wird target erreichen
        row = _make_journal_row(
            action="KAUFEN",
            target="105",  # Ziel 105 → wird erreicht
            timestamp="2026-01-01 10:00:00",
        )
        result = _evaluate_single(row, prices, 90)
        assert result["ziel_erreicht"] is True

    def test_zielkurs_nicht_erreicht(self):
        """KAUFEN: Zielkurs nicht erreicht (High < target)."""
        prices = _make_prices(100, 60, drift=0.001)  # kaum steigend
        row = _make_journal_row(
            action="KAUFEN",
            target="200",  # Ziel 200 → unrealistisch
            timestamp="2026-01-01 10:00:00",
        )
        result = _evaluate_single(row, prices, 90)
        assert result["ziel_erreicht"] is False

    def test_stop_gerissen(self):
        """KAUFEN: Stop wurde gerissen (Low ≤ stop)."""
        prices = _make_prices(100, 60, drift=-0.01)  # fallend → Low wird Stop erreichen
        row = _make_journal_row(
            action="KAUFEN",
            stop="95",  # Stop bei 95 → wird gerissen
            timestamp="2026-01-01 10:00:00",
        )
        result = _evaluate_single(row, prices, 90)
        assert result["stop_gerissen"] is True

    def test_stop_nicht_gerissen(self):
        """KAUFEN: Stop nicht gerissen (Low > stop)."""
        prices = _make_prices(100, 60, drift=0.005)  # steigend
        row = _make_journal_row(
            action="KAUFEN",
            stop="50",  # Stop bei 50 → unrealistisch tief
            timestamp="2026-01-01 10:00:00",
        )
        result = _evaluate_single(row, prices, 90)
        assert result["stop_gerissen"] is False

    def test_kein_zielkurs_leer(self):
        """Kein Zielkurs gesetzt → ziel_erreicht = None."""
        prices = _make_prices(100, 60, drift=0.005)
        row = _make_journal_row(action="KAUFEN", target="", timestamp="2026-01-01 10:00:00")
        result = _evaluate_single(row, prices, 90)
        assert result["ziel_erreicht"] is None


# --------------------------------------------------------------------------- #
# Tests: Konfidenz-Band-Aggregation
# --------------------------------------------------------------------------- #


class TestKonfidenzBaende:
    """Testet die Konfidenz-Band-Aggregation."""

    def test_konfidenz_baende_grouped(self, tmp_path):
        """Entscheidungen werden nach Confidence-Bändern gruppiert."""
        rows = [
            _make_journal_row(ticker="AAA", action="KAUFEN", confidence="5"),  # hoch
            _make_journal_row(ticker="BBB", action="KAUFEN", confidence="4"),  # hoch
            _make_journal_row(ticker="CCC", action="KAUFEN", confidence="3"),  # mittel
            _make_journal_row(ticker="DDD", action="KAUFEN", confidence="1"),  # niedrig
        ]
        path = _write_journal(tmp_path, rows)

        # yfinance mocken: steigende Kurse für alle Ticker → alle Hits
        def mock_load(ticker, *, lookback_days=90):
            return _make_prices(100, 60, drift=0.005)

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = evaluate_journal(path, lookback_days=90)

        bands = result["konfidenz_baende"]
        assert len(bands) >= 2  # mindestens hoch und niedrig
        hoch = next(b for b in bands if b["band"] == "hoch")
        assert hoch["n"] == 2
        niedrig = next(b for b in bands if b["band"] == "niedrig")
        assert niedrig["n"] == 1

    def test_konfidenz_band_hit_rate(self, tmp_path):
        """Hit-Rate im hoch-Band sollte > 0 sein wenn steigende Kurse."""
        rows = [
            _make_journal_row(ticker="AAA", action="KAUFEN", confidence="5"),
            _make_journal_row(ticker="BBB", action="KAUFEN", confidence="5"),
        ]
        path = _write_journal(tmp_path, rows)

        with patch(
            "concilium.evaluate._load_price_history",
            side_effect=lambda t, **kw: _make_prices(100, 60, drift=0.01),
        ):
            result = evaluate_journal(path)

        hoch = next(b for b in result["konfidenz_baende"] if b["band"] == "hoch")
        assert hoch["hit_rate"] is not None
        assert hoch["hit_rate"] > 0


# --------------------------------------------------------------------------- #
# Tests: Portfolio-Fit-Zusammenhang
# --------------------------------------------------------------------------- #


class TestPortfolioFit:
    """Testet den Portfolio-Fit-Zusammenhang."""

    def test_portfolio_fit_hoch_present(self, tmp_path):
        """Entscheidungen mit portfolio_fit_score ≥ 4 werden aggregiert."""
        rows = [
            _make_journal_row(ticker="AAA", action="KAUFEN", confidence="4", portfolio_fit_score="5"),
            _make_journal_row(ticker="BBB", action="KAUFEN", confidence="4", portfolio_fit_score="4"),
        ]
        path = _write_journal(tmp_path, rows)

        with patch(
            "concilium.evaluate._load_price_history",
            side_effect=lambda t, **kw: _make_prices(100, 60, drift=0.01),
        ):
            result = evaluate_journal(path)

        pf = result["portfolio_fit_hoch"]
        assert pf is not None
        assert pf["n"] == 2
        assert pf["hit_rate"] is not None

    def test_portfolio_fit_low_not_present(self, tmp_path):
        """Keine Entscheidungen mit portfolio_fit_score ≥ 4 → portfolio_fit_hoch = None."""
        rows = [
            _make_journal_row(ticker="AAA", action="KAUFEN", confidence="4", portfolio_fit_score="2"),
        ]
        path = _write_journal(tmp_path, rows)

        with patch(
            "concilium.evaluate._load_price_history",
            side_effect=lambda t, **kw: _make_prices(100, 60, drift=0.01),
        ):
            result = evaluate_journal(path)

        assert result["portfolio_fit_hoch"] is None


# --------------------------------------------------------------------------- #
# Tests: Fehler-Robustheit
# --------------------------------------------------------------------------- #


class TestFehlerRobustheit:
    """Testet Robustheit bei Kurs-Ladefehlern."""

    def test_ticker_no_data_adds_fehler(self, tmp_path):
        """Ticker ohne Kursdaten → fehler-Eintrag, kein Crash."""
        rows = [
            _make_journal_row(ticker="NODATA", action="KAUFEN"),
        ]
        path = _write_journal(tmp_path, rows)

        with patch("concilium.evaluate._load_price_history", return_value=None):
            result = evaluate_journal(path)

        assert result["anzahl_entscheidungen"] == 0
        assert len(result["fehler"]) == 1
        assert "NODATA" in result["fehler"][0]

    def test_partial_failure_still_evaluates_others(self, tmp_path):
        """Bei Kurs-Ladefehler für einen Ticker werden andere trotzdem ausgewertet."""
        rows = [
            _make_journal_row(ticker="FAIL", action="KAUFEN", confidence="4"),
            _make_journal_row(ticker="OK", action="KAUFEN", confidence="4"),
        ]
        path = _write_journal(tmp_path, rows)

        def mock_load(ticker, *, lookback_days=90):
            if ticker == "FAIL":
                return None
            return _make_prices(100, 60, drift=0.005)

        with patch("concilium.evaluate._load_price_history", side_effect=mock_load):
            result = evaluate_journal(path)

        assert result["anzahl_entscheidungen"] == 1
        assert len(result["fehler"]) == 1
        assert "FAIL" in result["fehler"][0]

    def test_empty_ticker_skipped(self, tmp_path):
        """Zeilen mit leerem Ticker werden übersprungen."""
        rows = [
            {"ticker": "", "action": "KAUFEN", "confidence": "4", "timestamp": "2026-01-01 10:00:00"},
        ]
        path = _write_journal(tmp_path, rows)
        result = evaluate_journal(path)
        assert result["anzahl_entscheidungen"] == 0

    def test_no_llm_no_zusammenfassung(self, tmp_path):
        """Ohne llm-Parameter → zusammenfassung = None."""
        rows = [_make_journal_row(ticker="AAPL", action="KAUFEN")]
        path = _write_journal(tmp_path, rows)

        with patch(
            "concilium.evaluate._load_price_history",
            side_effect=lambda t, **kw: _make_prices(100, 60, drift=0.01),
        ):
            result = evaluate_journal(path)

        assert result["zusammenfassung"] is None


# --------------------------------------------------------------------------- #
# Tests: Report-Generator
# --------------------------------------------------------------------------- #


class TestReportGenerator:
    """Testet generate_track_record_report."""

    def test_report_has_expected_sections(self):
        """Report enthält erwartete Markdown-Abschnitte."""
        eval_result = {
            "anzahl_entscheidungen": 10,
            "nach_aktion": {
                "KAUFEN": {"n": 6, "hit_rate": 0.667, "avg_rendite": 3.5},
                "HALTEN": {"n": 2, "hit_rate": 0.5, "avg_rendite": 0.0},
                "VERKAUFEN": {"n": 2, "hit_rate": 0.5, "avg_rendite": -1.0},
            },
            "hit_rate_gesamt": 0.6,
            "durchschnitt_rendite_gesamt": 2.0,
            "zielkurs_trefferquote": 0.5,
            "stop_verletzungsquote": 0.2,
            "konfidenz_baende": [
                {"band": "hoch", "hit_rate": 0.8, "n": 5},
                {"band": "mittel", "hit_rate": 0.4, "n": 3},
                {"band": "niedrig", "hit_rate": 0.2, "n": 2},
            ],
            "portfolio_fit_hoch": {"hit_rate": 0.75, "n": 4},
            "zusammenfassung": "Das System zeigt eine gute Performance.",
            "fehler": [],
        }
        report = generate_track_record_report(eval_result)

        assert "# Concilium Track-Record-Evaluierung" in report
        assert "## Übersicht" in report
        assert "## Bewertung nach Aktion" in report
        assert "## Konfidenz-Bänder" in report
        assert "## Portfolio-Fit-Zusammenhang" in report
        assert "## LLM-Zusammenfassung" in report
        assert "Disclaimer" in report
        assert "Track-Record-Evaluator" in report

    def test_report_empty_result(self):
        """Leeres Ergebnis → Report mit N/A-Werten, kein Crash."""
        from concilium.evaluate import _empty_result

        report = generate_track_record_report(_empty_result())
        assert "Track-Record-Evaluierung" in report
        assert "N/A" in report
        assert "0" in report  # anzahl_entscheidungen = 0

    def test_report_with_fehler(self):
        """Report mit Fehlern zeigt Fehlerhinweise."""
        eval_result = _empty_result_with_fehler()
        report = generate_track_record_report(eval_result)
        assert "Fehlerhinweise" in report
        assert "Keine Kursdaten" in report

    def test_report_no_konfidenz_baende(self):
        """Report ohne Konfidenz-Bänder → kein Abschnitt."""
        eval_result = {
            "anzahl_entscheidungen": 1,
            "nach_aktion": {
                "KAUFEN": {"n": 1, "hit_rate": None, "avg_rendite": None},
                "HALTEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
                "VERKAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
            },
            "hit_rate_gesamt": None,
            "durchschnitt_rendite_gesamt": None,
            "zielkurs_trefferquote": None,
            "stop_verletzungsquote": None,
            "konfidenz_baende": [],
            "portfolio_fit_hoch": None,
            "zusammenfassung": None,
            "fehler": [],
        }
        report = generate_track_record_report(eval_result)
        assert "Konfidenz-Bänder" not in report


class TestNaNSFormatting:
    """Testet dass NaN-Werte als N/A formatiert werden (nicht 'nan')."""

    def test_fmt_nan_returns_na(self):
        """_fmt(float('nan')) → 'N/A'."""
        from concilium.report import _fmt

        assert _fmt(float("nan")) == "N/A"

    def test_fmt_none_returns_na(self):
        """_fmt(None) → 'N/A' (bestehendes Verhalten)."""
        from concilium.report import _fmt

        assert _fmt(None) == "N/A"

    def test_fmt_valid_value_unchanged(self):
        """_fmt(5.5) → '5.50' (gültige Werte unverändert)."""
        from concilium.report import _fmt

        assert _fmt(5.5) == "5.50"

    def test_fmt_pct_nan_returns_na(self):
        """_fmt_pct(float('nan')) → 'N/A'."""
        from concilium.report import _fmt_pct

        assert _fmt_pct(float("nan")) == "N/A"

    def test_fmt_pct2_nan_returns_na(self):
        """_fmt_pct2(float('nan')) → 'N/A'."""
        from concilium.report import _fmt_pct2

        assert _fmt_pct2(float("nan")) == "N/A"

    def test_fmt_num_nan_returns_na(self):
        """_fmt_num(float('nan')) → 'N/A'."""
        from concilium.report import _fmt_num

        assert _fmt_num(float("nan"), " %") == "N/A"

    def test_report_nan_rendite_no_nan_string(self):
        """Track-Record-Report mit NaN-Ø-Rendite enthält kein 'nan'."""
        eval_result = {
            "anzahl_entscheidungen": 6,
            "nach_aktion": {
                "KAUFEN": {"n": 6, "hit_rate": 0.0, "avg_rendite": float("nan")},
                "HALTEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
                "VERKAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
            },
            "hit_rate_gesamt": 0.0,
            "durchschnitt_rendite_gesamt": float("nan"),
            "zielkurs_trefferquote": None,
            "stop_verletzungsquote": None,
            "konfidenz_baende": [],
            "portfolio_fit_hoch": None,
            "zusammenfassung": None,
            "fehler": [],
        }
        report = generate_track_record_report(eval_result)
        assert "nan" not in report.lower()
        assert "N/A" in report


def _empty_result_with_fehler() -> dict:
    from concilium.evaluate import _empty_result

    r = _empty_result()
    r["fehler"] = ["2026-01-01 AAPL: Keine Kursdaten verfügbar."]
    return r


# --------------------------------------------------------------------------- #
# Tests: CLI-Flag --evaluate
# --------------------------------------------------------------------------- #


class TestCLIEvaluate:
    """Testet die CLI-Flag --evaluate."""

    def test_evaluate_writes_report(self, tmp_path, monkeypatch, capsys):
        """--evaluate mit temporärer Journal-Datei → Report auf stdout + Datei geschrieben."""
        rows = [
            _make_journal_row(ticker="AAPL", action="KAUFEN", confidence="4"),
        ]
        journal_path = _write_journal(tmp_path, rows)

        # yfinance mocken
        with patch(
            "concilium.evaluate._load_price_history",
            side_effect=lambda t, **kw: _make_prices(100, 60, drift=0.01),
        ):
            from concilium.cli import main

            ret = main(["--evaluate", journal_path, "--no-llm"])

        assert ret == 0
        # stdout enthält den Report
        captured = capsys.readouterr()
        assert "Track-Record-Evaluierung" in captured.out
        # stderr enthält den Speicherpfad
        assert "Track-Record-Report gespeichert" in captured.err

    def test_evaluate_no_llm_flag(self, tmp_path):
        """--evaluate mit --no-llm → keine LLM-Zusammenfassung."""
        rows = [
            _make_journal_row(ticker="AAPL", action="KAUFEN", confidence="4"),
        ]
        path = _write_journal(tmp_path, rows)

        with patch(
            "concilium.evaluate._load_price_history",
            side_effect=lambda t, **kw: _make_prices(100, 60, drift=0.01),
        ):
            from concilium.cli import main

            # LLMClient soll nicht aufgerufen werden
            with patch("concilium.cli.LLMClient") as mock_llm:
                mock_llm.return_value = None
                ret = main(["--evaluate", path, "--no-llm"])

        assert ret == 0

    def test_evaluate_default_path(self, tmp_path, monkeypatch):
        """--evaluate ohne Pfad → Default journal/decisions.csv wird verwendet."""
        # Arbeitsverzeichnis auf tmp_path setzen
        monkeypatch.chdir(tmp_path)
        os.makedirs("journal", exist_ok=True)
        from concilium.journal import JOURNAL_HEADER

        with open("journal/decisions.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
            writer.writeheader()
            writer.writerow(_make_journal_row(ticker="AAPL", action="KAUFEN", confidence="4"))

        with patch(
            "concilium.evaluate._load_price_history",
            side_effect=lambda t, **kw: _make_prices(100, 60, drift=0.01),
        ):
            from concilium.cli import main

            ret = main(["--evaluate", "--no-llm"])

        assert ret == 0


# --------------------------------------------------------------------------- #
# Tests: Portfolio-Sheet-Tages-Cache
# --------------------------------------------------------------------------- #


_DUMMY_CSV = (
    'Bestand,Name,Symbol, Kurs, Marktwert, Anteil in %, Region\n'
    '10, Apple, AAPL, "150,00", "1500,00", "4,50", USA\n'
    '5, Microsoft, MSFT, "300,00", "1500,00", "4,50", USA\n'
)


class TestPortfolioCache:
    """Testet den Portfolio-Sheet-Tages-Cache."""

    def test_fetch_calls_network_once_per_day(self, tmp_path, monkeypatch):
        """fetch_portfolio_positions ruft das Sheet nur 1×/Tag ab (Cache)."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", str(tmp_path))
        monkeypatch.chdir(tmp_path)

        from concilium.portfolio_fit import fetch_portfolio_positions

        # Mock urlopen: zählt Aufrufe
        call_count = {"n": 0}

        class _MockResp:
            def __init__(self, data: bytes):
                self._data = data

            def read(self) -> bytes:
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def mock_urlopen(*args, **kwargs):
            call_count["n"] += 1
            return _MockResp(_DUMMY_CSV.encode("utf-8"))

        with patch(
            "concilium.portfolio_fit.urllib.request.urlopen",
            side_effect=mock_urlopen,
        ):
            # Erster Aufruf → Netzwerk
            pos1 = fetch_portfolio_positions()
            assert len(pos1) == 2
            assert call_count["n"] == 1

            # Zweiter Aufruf → Cache-Treffer, kein Netzwerk
            pos2 = fetch_portfolio_positions()
            assert len(pos2) == 2
            assert call_count["n"] == 1  # immer noch 1 → Cache genutzt

    def test_cache_disabled_when_empty_env(self, tmp_path, monkeypatch):
        """CONCILIUM_CACHE_DIR='' → Cache deaktiviert, immer Netzwerk."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", "")

        from concilium.portfolio_fit import fetch_portfolio_positions

        call_count = {"n": 0}

        class _MockResp:
            def __init__(self, data: bytes):
                self._data = data

            def read(self) -> bytes:
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def mock_urlopen(*args, **kwargs):
            call_count["n"] += 1
            return _MockResp(_DUMMY_CSV.encode("utf-8"))

        with patch(
            "concilium.portfolio_fit.urllib.request.urlopen",
            side_effect=mock_urlopen,
        ):
            fetch_portfolio_positions()
            fetch_portfolio_positions()
            assert call_count["n"] == 2  # 2 Aufrufe → Cache deaktiviert

    def test_cache_returns_same_positions(self, tmp_path, monkeypatch):
        """Cache liefert dieselben Positionen wie der erste Abruf."""
        monkeypatch.setenv("CONCILIUM_CACHE_DIR", str(tmp_path))
        monkeypatch.chdir(tmp_path)

        from concilium.portfolio_fit import fetch_portfolio_positions

        class _MockResp:
            def __init__(self, data: bytes):
                self._data = data

            def read(self) -> bytes:
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with patch(
            "concilium.portfolio_fit.urllib.request.urlopen",
            return_value=_MockResp(_DUMMY_CSV.encode("utf-8")),
        ):
            pos1 = fetch_portfolio_positions()
            pos2 = fetch_portfolio_positions()

        assert pos1 == pos2
        assert len(pos1) == 2
