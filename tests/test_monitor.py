"""Tests für den Stop-Monitor (--monitor, monitor.py).

Alle Tests sind OFFLINE-fähig: kein yfinance, kein Netzwerk, kein LLM.
Depot-Positionen werden gemockt (fetch_portfolio_positions), die Kursdaten
ebenfalls (collect_ticker_data) — Mock-Muster analog tests/test_review.py.
"""

from __future__ import annotations

import csv
import os
import sys
from unittest.mock import patch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from concilium.cli import main  # noqa: E402
from concilium.monitor import (  # noqa: E402
    _parse_float,
    load_last_journal_entry,
    run_monitor,
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


# --------------------------------------------------------------------------- #
# Journal-Helper
# --------------------------------------------------------------------------- #


JOURNAL_HEADER = [
    "timestamp",
    "ticker",
    "action",
    "stop",
    "target",
    "final_decision",
    "confidence",
]


def _write_journal(path, rows: list[dict]) -> str:
    """Schreibt eine kleine decisions.csv (nur relevante Spalten)."""
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=JOURNAL_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: "" for k in JOURNAL_HEADER}, **row})
    return str(path)


def _make_journal_rows(
    ticker: str,
    *,
    stop: str = "150",
    target: str = "200",
    action: str = "KAUFEN",
    ts: str = "2026-09-09 10:00:00",
) -> list[dict]:
    return [
        {
            "timestamp": ts,
            "ticker": ticker,
            "action": action,
            "stop": stop,
            "target": target,
            "final_decision": "GENEHMIGT",
            "confidence": "4",
        }
    ]


def _mock_ticker_data(price: float | None) -> dict:
    """Minimal collect_ticker_data-Result mit technicals.current_price."""
    return {
        "ticker": "X",
        "fundamentals": {},
        "technicals": {"current_price": price},
        "sentiment": {},
        "news": [],
        "history": [],
    }


# --------------------------------------------------------------------------- #
# (a) run_monitor: Grundfunktionen
# --------------------------------------------------------------------------- #


class TestRunMonitorBasis:
    """run_monitor: Depot-Filter, Journal-Load, Stop-/Ziel-Check."""

    def test_stop_gerissen_long_erkannt(self, tmp_path):
        """Kurs unter Long-Stop → stop_gerissen=True (🔴)."""
        jf = _write_journal(tmp_path / "journal" / "decisions.csv",
                            _make_journal_rows("AAPL", stop="150", target="200"))

        def mock_collect(ticker, **kwargs):
            return {"technicals": {"current_price": 120.0}}

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.monitor.collect_ticker_data", side_effect=mock_collect):
                result = run_monitor(None, journal_file=jf)

        entry = result["positionen"]["AAPL"]
        assert entry["stop_gerissen"] is True
        assert entry["ziel_erreicht"] is False
        assert entry["stop"] == 150.0
        assert entry["current_price"] == 120.0
        assert "STOP GERISSEN" in entry["hinweis"]
        assert result["fehler"] == 0

    def test_ziel_erreicht_long_erkannt(self, tmp_path):
        """Kurs über Long-Ziel → ziel_erreicht=True (🟢)."""
        jf = _write_journal(tmp_path / "journal" / "decisions.csv",
                            _make_journal_rows("AAPL", stop="150", target="200"))

        def mock_collect(ticker, **kwargs):
            return {"technicals": {"current_price": 210.0}}

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.monitor.collect_ticker_data", side_effect=mock_collect):
                result = run_monitor(None, journal_file=jf)

        entry = result["positionen"]["AAPL"]
        assert entry["ziel_erreicht"] is True
        assert entry["stop_gerissen"] is False
        assert "ZIEL ERREICHT" in entry["hinweis"]

    def test_etf_wird_uebersprungen(self, tmp_path):
        """ETF/Commodity = Buy-and-Hold → nicht geprüft, collect nie aufgerufen."""
        jf = _write_journal(tmp_path / "journal" / "decisions.csv",
                            _make_journal_rows("AAPL"))
        calls: list[str] = []

        def mock_collect(ticker, **kwargs):
            calls.append(ticker)
            return {"technicals": {"current_price": 100.0}}

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.monitor.collect_ticker_data", side_effect=mock_collect):
                result = run_monitor(None, journal_file=jf)

        assert set(result["positionen"]) == {"AAPL", "BAS.DE"}
        assert "IS3R.DE" not in result["positionen"]
        # Nur Positionen MIT Journal-Eintrag lösen einen Kurs-Call aus
        # (BAS.DE hat keinen Eintrag → kein yfinance-Call nötig).
        assert calls == ["AAPL"]
        assert result["gesamt_positionen"] == 3

    def test_kein_journal_eintrag_markiert(self, tmp_path):
        """Position ohne Journal-Eintrag → journal_gefunden=False, kein Crash."""
        jf = _write_journal(tmp_path / "journal" / "decisions.csv",
                            _make_journal_rows("AAPL"))

        def mock_collect(ticker, **kwargs):
            return {"technicals": {"current_price": 100.0}}

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.monitor.collect_ticker_data", side_effect=mock_collect):
                result = run_monitor(None, journal_file=jf)

        entry = result["positionen"]["BAS.DE"]
        assert entry["journal_gefunden"] is False
        assert entry["stop"] is None
        assert entry["stop_gerissen"] is None
        assert "kein journal-eintrag" in entry["hinweis"].lower()

    def test_fehlgeschlagener_ticker_zaehlt_als_fehler(self, tmp_path):
        """collect_ticker_data wirft → fehler+=1, Rest läuft weiter, kein Crash."""
        jf = _write_journal(tmp_path / "journal" / "decisions.csv",
                            _make_journal_rows("AAPL") + _make_journal_rows("BAS.DE"))

        def mock_collect(ticker, **kwargs):
            if ticker == "BAS.DE":
                raise ValueError("Kursdaten fehlgeschlagen")
            return {"technicals": {"current_price": 100.0}}

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.monitor.collect_ticker_data", side_effect=mock_collect):
                result = run_monitor(None, journal_file=jf)

        assert result["fehler"] == 1
        assert "BAS.DE" not in result["positionen"]
        assert "AAPL" in result["positionen"]

    def test_leeres_depot_leeres_ergebnis(self, tmp_path):
        """Fehlendes/leeres Depot → leere Positionen, kein Absturz."""
        jf = _write_journal(tmp_path / "journal" / "decisions.csv",
                            _make_journal_rows("AAPL"))
        with patch("concilium.monitor.fetch_portfolio_positions", side_effect=lambda: []):
            with patch("concilium.monitor.collect_ticker_data") as mock_collect:
                result = run_monitor(None, journal_file=jf)

        assert result["positionen"] == {}
        assert result["fehler"] == 0
        assert result["gesamt_positionen"] == 0
        mock_collect.assert_not_called()

    def test_fetch_portfolio_crash_does_not_propagate(self, tmp_path):
        """Selbst wenn fetch_portfolio_positions unerwartet crasht → leeres Depot."""

        def mock_fetch():
            raise RuntimeError("Netzwerk-Tot")

        with patch("concilium.monitor.fetch_portfolio_positions", side_effect=mock_fetch):
            with patch("concilium.monitor.collect_ticker_data") as mock_collect:
                result = run_monitor(None, journal_file="nicht_vorhanden.csv")

        assert result["positionen"] == {}
        assert result["fehler"] == 0
        mock_collect.assert_not_called()

    def test_verkaufen_short_logik(self, tmp_path):
        """VERKAUFEN = Short: Stop oberhalb, Ziel unterhalb (umgekehrte Vergleiche)."""
        jf = _write_journal(
            tmp_path / "journal" / "decisions.csv",
            _make_journal_rows("AAPL", stop="165", target="140", action="VERKAUFEN"),
        )

        def mock_collect(ticker, **kwargs):
            # Short: Stop 165 (gerissen wenn Kurs >= 165), Ziel 140 (erreicht
            # wenn Kurs <= 140). Kurs 168 → Stop gerissen.
            return {"technicals": {"current_price": 168.0}}

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.monitor.collect_ticker_data", side_effect=mock_collect):
                result = run_monitor(None, journal_file=jf)

        entry = result["positionen"]["AAPL"]
        assert entry["stop_gerissen"] is True
        assert entry["ziel_erreicht"] is False

        # Kurs 135 → Ziel erreicht (unter Short-Ziel 140)
        def mock_collect_unten(ticker, **kwargs):
            return {"technicals": {"current_price": 135.0}}

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.monitor.collect_ticker_data",
                       side_effect=mock_collect_unten):
                result2 = run_monitor(None, journal_file=jf)

        entry2 = result2["positionen"]["AAPL"]
        assert entry2["ziel_erreicht"] is True
        assert entry2["stop_gerissen"] is False

    def test_journal_eintrag_ohne_stop_und_target(self, tmp_path):
        """Journal-Eintrag vorhanden, aber stop/target leer → None, kein Crash."""
        jf = _write_journal(
            tmp_path / "journal" / "decisions.csv",
            _make_journal_rows("AAPL", stop="", target=""),
        )

        def mock_collect(ticker, **kwargs):
            return {"technicals": {"current_price": 100.0}}

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.monitor.collect_ticker_data", side_effect=mock_collect):
                result = run_monitor(None, journal_file=jf)

        entry = result["positionen"]["AAPL"]
        assert entry["journal_gefunden"] is True
        assert entry["stop"] is None
        assert entry["target"] is None
        assert entry["stop_gerissen"] is None
        assert entry["ziel_erreicht"] is None
        assert entry["hinweis"] == "Kein Stop/Ziel im Journal-Eintrag"

    def test_letzte_zeile_gewinnt(self, tmp_path):
        """Mehrere Journal-Zeilen pro Ticker → die LETZTE (jüngste) gewinnt."""
        jf = _write_journal(
            tmp_path / "journal" / "decisions.csv",
            [
                {"timestamp": "2026-09-01 10:00:00", "ticker": "AAPL",
                 "action": "KAUFEN", "stop": "100", "target": "180"},
                {"timestamp": "2026-09-08 10:00:00", "ticker": "AAPL",
                 "action": "HALTEN", "stop": "140", "target": "200"},
            ],
        )

        def mock_collect(ticker, **kwargs):
            # Kurs 135: über altem Stop 100 (OK), unter neuem Stop 140 → gerissen
            return {"technicals": {"current_price": 135.0}}

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.monitor.collect_ticker_data", side_effect=mock_collect):
                result = run_monitor(None, journal_file=jf)

        entry = result["positionen"]["AAPL"]
        # Die letzte (jüngste) Journal-Zeile liefert stop=140
        assert entry["stop"] == 140.0
        assert entry["stop_gerissen"] is True
        assert entry["timestamp"] == "2026-09-08 10:00:00"

    def test_ticker_case_insensitive(self, tmp_path):
        """Journal-Ticker in anderer Schreibweise → wird trotzdem gefunden."""
        jf = _write_journal(
            tmp_path / "journal" / "decisions.csv",
            _make_journal_rows("aapl", stop="150", target="200"),
        )

        def mock_collect(ticker, **kwargs):
            return {"technicals": {"current_price": 160.0}}

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.monitor.collect_ticker_data", side_effect=mock_collect):
                result = run_monitor(None, journal_file=jf)

        entry = result["positionen"]["AAPL"]
        assert entry["journal_gefunden"] is True
        assert entry["stop_gerissen"] is False

    def test_max_positions_begrenzt(self, tmp_path):
        """max_positions=2 → nur die 2 größten Aktien (MSFT, AAPL), BAS.DE raus."""
        jf = _write_journal(
            tmp_path / "journal" / "decisions.csv",
            _make_journal_rows("AAPL") + _make_journal_rows("BAS.DE")
            + _make_journal_rows("MSFT"),
        )

        def mock_collect(ticker, **kwargs):
            return {"technicals": {"current_price": 100.0}}

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions_3_aktien()):
            with patch("concilium.monitor.collect_ticker_data", side_effect=mock_collect):
                result = run_monitor(None, journal_file=jf, max_positions=2)

        assert set(result["positionen"]) == {"MSFT", "AAPL"}
        assert result["gesamt_positionen"] == 4

    def test_fehlender_kurs_zaehlt_als_fehler(self, tmp_path):
        """collect liefert current_price=None → als Fehler gezählt, kein Crash."""
        jf = _write_journal(tmp_path / "journal" / "decisions.csv",
                            _make_journal_rows("AAPL"))

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.monitor.collect_ticker_data",
                       side_effect=lambda t, **kw: {"technicals": {"current_price": None}}):
                result = run_monitor(None, journal_file=jf)

        assert result["fehler"] == 1
        assert "AAPL" not in result["positionen"]
        # BAS.DE hat keinen Journal-Eintrag → regulär markiert, kein Fehler
        assert result["positionen"]["BAS.DE"]["journal_gefunden"] is False

    def test_as_of_wird_durchgereicht(self, tmp_path):
        """as_of wird an collect_ticker_data weitergereicht."""
        jf = _write_journal(tmp_path / "journal" / "decisions.csv",
                            _make_journal_rows("AAPL"))
        captured: dict = {}

        def mock_collect(ticker, **kwargs):
            captured.update(kwargs)
            return {"technicals": {"current_price": 100.0}}

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.monitor.collect_ticker_data", side_effect=mock_collect):
                run_monitor(None, journal_file=jf, as_of="2026-08-01")

        assert captured.get("as_of") == "2026-08-01"


# --------------------------------------------------------------------------- #
# (b) load_last_journal_entry + _parse_float (Unit-Tests)
# --------------------------------------------------------------------------- #


class TestLoadLastJournalEntry:
    """Journal-Reader: letzte passende Zeile, tolerant bei Fehlern."""

    def test_liefert_letzte_passende_zeile(self, tmp_path):
        jf = _write_journal(
            tmp_path / "j.csv",
            [
                {"timestamp": "2026-09-01 10:00:00", "ticker": "AAPL", "stop": "100"},
                {"timestamp": "2026-09-05 10:00:00", "ticker": "MSFT", "stop": "300"},
                {"timestamp": "2026-09-08 10:00:00", "ticker": "AAPL", "stop": "140"},
            ],
        )
        entry = load_last_journal_entry("AAPL", jf)
        assert entry is not None
        assert entry["stop"] == "140"
        assert entry["timestamp"] == "2026-09-08 10:00:00"

    def test_kein_eintrag_gibt_none(self, tmp_path):
        jf = _write_journal(tmp_path / "j.csv", _make_journal_rows("AAPL"))
        assert load_last_journal_entry("NVDA", jf) is None

    def test_fehlende_datei_gibt_none(self, tmp_path):
        assert load_last_journal_entry("AAPL", str(tmp_path / "gibts_nicht.csv")) is None


class TestParseFloat:
    """_parse_float: tolerant, positiv, None bei Müll."""

    def test_zahlen(self):
        assert _parse_float(150) == 150.0
        assert _parse_float(150.5) == 150.5

    def test_strings_mit_komma(self):
        assert _parse_float("165,5") == 165.5
        assert _parse_float(" 165 ") == 165.0

    def test_ungueltig_gibt_none(self):
        assert _parse_float("") is None
        assert _parse_float(None) is None
        assert _parse_float("abc") is None

    def test_null_und_negativ_gibt_none(self):
        assert _parse_float(0) is None
        assert _parse_float(-5) is None
        assert _parse_float("-5") is None


# --------------------------------------------------------------------------- #
# (c) CLI: --monitor Flag — Parsing + Mutual Exclusion
# --------------------------------------------------------------------------- #


class TestMonitorCliMutualExclusion:
    """--monitor schließt sich mit --ticker/--tickers/--portfolio/--watchlist aus."""

    def test_monitor_and_ticker_error(self):
        assert main(["--monitor", "--ticker", "AAPL"]) == 1

    def test_monitor_and_tickers_error(self):
        assert main(["--monitor", "--tickers", "AAPL,NVDA"]) == 1

    def test_monitor_and_portfolio_error(self):
        assert main(["--monitor", "--portfolio", "AAPL,NVDA"]) == 1

    def test_monitor_and_watchlist_error(self):
        assert main(["--monitor", "--watchlist"]) == 1

    def test_monitor_ticker_order_independent(self):
        assert main(["--ticker", "AAPL", "--monitor"]) == 1


# --------------------------------------------------------------------------- #
# (d) CLI: --monitor Modus — evaluate vorn, run_monitor, Zusammenfassung
# --------------------------------------------------------------------------- #


def _run_cli_monitor(monkeypatch, tmp_path, *, hooks: dict, extra_args=None):
    """Hilfsfunktion: führt main(['--monitor', ...]) mit gemockten Abhängigkeiten aus.

    Analog _run_cli_review aus tests/test_review.py: Setzt CONCILIUM_STATE_DIR
    auf ein tmp-Verzeichnis (Schutz der echten calibration.json) und gibt
    (exit_code, call_order) zurück.
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
    ]
    if hooks.get("run_monitor"):
        patches.append(
            patch.object(cli_mod, "run_monitor",
                         side_effect=_wrap("run_monitor", hooks["run_monitor"]))
        )

    import contextlib

    argv = ["--monitor"] + (extra_args or [])
    with contextlib.ExitStack() as stack:
        for p in patches:
            p = p.__enter__()
            stack.callback(p.__exit__, None, None, None)
        code = main(argv)

    return code, call_order


class TestMonitorCliMode:
    """--monitor: evaluate vorn, run_monitor danach, kompakte Zusammenfassung."""

    def test_monitor_runs_evaluate_before_run_monitor(self, tmp_path, monkeypatch):
        """evaluate_journal + calibration VOR run_monitor (analog --review)."""
        code, call_order = _run_cli_monitor(
            monkeypatch,
            tmp_path,
            hooks={
                "run_monitor": lambda llm, **kw: {
                    "positionen": {}, "fehler": 0, "gesamt_positionen": 0,
                },
            },
        )
        assert code == 0
        assert call_order[:2] == ["evaluate_journal", "_write_calibration_json"]
        assert "run_monitor" in call_order
        assert call_order.index("evaluate_journal") < call_order.index("run_monitor")

    def test_monitor_passes_max_positions_flag(self, tmp_path, monkeypatch):
        """--max-positions 5 → run_monitor(max_positions=5)."""
        captured: dict = {}

        def mock_run_monitor(llm, **kw):
            captured.update(kw)
            return {"positionen": {}, "fehler": 0, "gesamt_positionen": 0}

        code, _ = _run_cli_monitor(
            monkeypatch, tmp_path,
            hooks={"run_monitor": mock_run_monitor},
            extra_args=["--max-positions", "5"],
        )
        assert code == 0
        assert captured.get("max_positions") == 5

    def test_monitor_passes_as_of_flag(self, tmp_path, monkeypatch):
        """--date 2026-08-01 → run_monitor(as_of='2026-08-01')."""
        captured: dict = {}

        def mock_run_monitor(llm, **kw):
            captured.update(kw)
            return {"positionen": {}, "fehler": 0, "gesamt_positionen": 0}

        code, _ = _run_cli_monitor(
            monkeypatch, tmp_path,
            hooks={"run_monitor": mock_run_monitor},
            extra_args=["--date", "2026-08-01"],
        )
        assert code == 0
        assert captured.get("as_of") == "2026-08-01"

    def test_monitor_zusammenfassung_symbole(self, tmp_path, monkeypatch, capsys):
        """Zusammenfassung: 🔴 Stop gerissen, 🟢 Ziel erreicht, ⚪ kein Journal."""
        monitor_result = {
            "positionen": {
                "AAPL": {
                    "name": "Apple", "depot_pct": 5.2, "current_price": 120.0,
                    "stop": 150.0, "target": 200.0, "action": "KAUFEN",
                    "timestamp": "2026-09-01 10:00:00", "journal_gefunden": True,
                    "stop_gerissen": True, "ziel_erreicht": False,
                    "hinweis": "STOP GERISSEN",
                },
                "MSFT": {
                    "name": "Microsoft", "depot_pct": 8.0, "current_price": 420.0,
                    "stop": 380.0, "target": 400.0, "action": "KAUFEN",
                    "timestamp": "2026-09-01 10:00:00", "journal_gefunden": True,
                    "stop_gerissen": False, "ziel_erreicht": True,
                    "hinweis": "ZIEL ERREICHT",
                },
                "BAS.DE": {
                    "name": "BASF", "depot_pct": 2.0, "current_price": None,
                    "stop": None, "target": None, "action": "",
                    "timestamp": "", "journal_gefunden": False,
                    "stop_gerissen": None, "ziel_erreicht": None,
                    "hinweis": "Kein Journal-Eintrag (kein Stop bekannt)",
                },
            },
            "fehler": 0,
            "gesamt_positionen": 4,
        }

        code, _ = _run_cli_monitor(
            monkeypatch, tmp_path, hooks={"run_monitor": lambda llm, **kw: monitor_result}
        )
        assert code == 0

        captured = capsys.readouterr()
        out = captured.out  # Monitor-Zusammenfassung geht auf stdout
        aapl_line = next(
            line for line in out.splitlines() if "AAPL" in line and "🔴" in line
        )
        assert "STOP GERISSEN" in aapl_line
        assert "Kurs 120.00 < Stop 150.00" in aapl_line
        assert "(-20.0%)" in aapl_line  # (120-150)/150 = -20%
        msft_line = next(
            line for line in out.splitlines() if "MSFT" in line and "🟢" in line
        )
        assert "ZIEL ERREICHT" in msft_line
        assert "Kurs 420.00 ≥ Ziel 400.00" in msft_line
        bas_line = next(
            line for line in out.splitlines() if "BAS.DE" in line and "⚪" in line
        )
        assert "kein Journal-Eintrag" in bas_line
        assert "Monitor: 3 Positionen geprüft, 1 Stops gerissen, 1 Ziele erreicht, 0 Fehler." in out

    def test_monitor_leeres_depot_exit_0(self, tmp_path, monkeypatch, capsys):
        """Leeres Depot → Exit 0, Meldung statt Crash."""
        code, _ = _run_cli_monitor(
            monkeypatch, tmp_path,
            hooks={"run_monitor": lambda llm, **kw: {
                "positionen": {}, "fehler": 0, "gesamt_positionen": 0,
            }},
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "Keine Aktien-Positionen zu prüfen" in out

    def test_monitor_run_monitor_crash_caught(self, tmp_path, monkeypatch):
        """run_monitor selbst crasht → CLI fängt ab, Exit 1, kein Traceback-Abbruch."""

        def boom(llm, **kw):
            raise RuntimeError("Katastrophe")

        code, _ = _run_cli_monitor(monkeypatch, tmp_path, hooks={"run_monitor": boom})
        assert code == 1

    def test_monitor_plus_evaluate_allowed(self, tmp_path, monkeypatch):
        """--monitor + --evaluate ist erlaubt (evaluate läuft vorn)."""
        jf = _write_journal(tmp_path / "journal" / "decisions.csv",
                            _make_journal_rows("AAPL"))

        code, call_order = _run_cli_monitor(
            monkeypatch, tmp_path,
            hooks={"run_monitor": lambda llm, **kw: {
                "positionen": {}, "fehler": 0, "gesamt_positionen": 0,
            }},
            extra_args=["--evaluate", jf],
        )
        assert code == 0
        assert call_order[:2] == ["evaluate_journal", "_write_calibration_json"]


# --------------------------------------------------------------------------- #
# (e) Rückgabe-Vertrag
# --------------------------------------------------------------------------- #


class TestRunMonitorVertrag:
    """Rückgabe-dict enthält die geforderten Schlüssel."""

    def test_ergebnis_struktur(self, tmp_path):
        jf = _write_journal(tmp_path / "journal" / "decisions.csv",
                            _make_journal_rows("AAPL", stop="150", target="200"))

        def mock_collect(ticker, **kwargs):
            return {"technicals": {"current_price": 160.0}}

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.monitor.collect_ticker_data", side_effect=mock_collect):
                result = run_monitor(None, journal_file=jf)

        assert set(result) == {"positionen", "fehler", "gesamt_positionen"}
        entry = result["positionen"]["AAPL"]
        assert set(entry) >= {
            "name", "depot_pct", "current_price", "stop", "target", "action",
            "timestamp", "journal_gefunden", "stop_gerissen", "ziel_erreicht", "hinweis",
        }
        assert entry["name"] == "Apple Inc."
        assert entry["depot_pct"] == 5.2
        assert entry["action"] == "KAUFEN"
        assert entry["timestamp"] == "2026-09-09 10:00:00"
        assert isinstance(entry["hinweis"], str) and entry["hinweis"]


# --------------------------------------------------------------------------- #
# (f) CLI: --monitor ohne Modus-Argument ist kein required-Args-Fehler mehr
# --------------------------------------------------------------------------- #


class TestMonitorCliFlagParsing:
    """Das Flag allein wird korrekt geparst (kein required-args-Abbruch)."""

    def test_monitor_flag_is_store_true(self):
        """args.monitor wird gesetzt — der Mutex-Fehler beweist das Parsing."""
        assert main(["--monitor", "--ticker", "AAPL"]) == 1

    def test_monitor_alone_does_not_error_without_ticker(self, monkeypatch, tmp_path):
        """--monitor allein → kein parser.error wegen fehlender Ticker-Args."""
        # Ohne Hooks: evaluate läuft real (leeres Journal in tmp → eval ok),
        # run_monitor wird gepatcht, damit kein Netz/kein Sheet nötig ist.
        from concilium import cli as cli_mod

        state_dir = tmp_path / "state"
        state_dir.mkdir(exist_ok=True)
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(state_dir))
        monkeypatch.setenv("CONCILIUM_REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.chdir(tmp_path)

        with patch.object(cli_mod, "evaluate_journal",
                          side_effect=lambda *a, **kw: {
                              "anzahl_entscheidungen": 0, "nach_aktion": {}}):
            with patch.object(cli_mod, "run_monitor",
                              side_effect=lambda llm, **kw: {
                                  "positionen": {}, "fehler": 0,
                                  "gesamt_positionen": 0}):
                code = cli_mod.main(["--monitor"])
        assert code == 0


class TestMonitorKeineSeiteneffekte:
    """Der Monitor schreibt nichts — Journal bleibt unverändert."""

    def test_run_monitor_schreibt_kein_journal(self, tmp_path):
        jf = _write_journal(tmp_path / "journal" / "decisions.csv",
                            _make_journal_rows("AAPL"))
        before = open(jf, encoding="utf-8").read()

        def mock_collect(ticker, **kwargs):
            return {"technicals": {"current_price": 100.0}}

        with patch("concilium.monitor.fetch_portfolio_positions",
                   side_effect=lambda: _mock_positions()):
            with patch("concilium.monitor.collect_ticker_data", side_effect=mock_collect):
                run_monitor(None, journal_file=jf)

        assert open(jf, encoding="utf-8").read() == before
