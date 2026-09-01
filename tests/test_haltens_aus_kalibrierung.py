"""Tests für Phase 1: HALTEN aus der Trade-Kalibrierung herausnehmen.

Fachlicher Hintergrund: HALTEN ist KEIN Trade und KEINE Richtungsprognose
("kein Handlungsbedarf"). Die Konfidenz-Kalibrierung (Brier-Score, Gap,
Ensemble-Gewichtung, Rating-Dämpfung) misst, wie gut die Konfidenz einer
RICHTUNGSPROGNOSE mit dem Ergebnis übereinstimmt — HALTEN verzerrt diese
Metrik und wird daher ausgeschlossen.

Neue Semantik:
- HALTEN bleibt im Journal und wird weiterhin bewertet (hit/rendite),
  geht aber NICHT in hit_rate_gesamt oder die Kalibrierung ein.
- HALTEN wird deskriptiv ausgewiesen: halten_n (Anzahl) und halten_quote
  (Anteil stabiler Verläufe |rendite| <= 2 %).
- KAUFEN/VERKAUFEN werden unverändert als Trades bewertet.

Alle Tests sind OFFLINE-fähig: yfinance wird gemockt, kein Netzwerk.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.evaluate import (  # noqa: E402
    _aggregate,
    _compute_konfidenz_kalibrierung,
    _compute_konfidenz_kalibrierung_segmentiert,
    _compute_reliability_bins,
    _empty_result,
    _evaluate_single,
    evaluate_journal,
)
from concilium.feedback import (  # noqa: E402
    _compute_kalibrierung_proxy,
    _compute_kalibrierung_proxy_per_action,
    _compute_stats,
)
from concilium.journal import JOURNAL_HEADER  # noqa: E402
from concilium.report import generate_track_record_report  # noqa: E402

# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #


def _make_eval(
    confidence: float | None,
    hit: bool | None,
    action: str = "KAUFEN",
    rating: str = "",
) -> dict:
    """Erzeugt ein Einzel-Ergebnis-dict wie aus _evaluate_single."""
    return {
        "hit": hit,
        "rendite_pct": 1.0 if hit else -1.0,
        "ziel_erreicht": None,
        "stop_gerissen": None,
        "action": action,
        "rating": rating,
        "rating_distance": None,
        "confidence": confidence,
        "portfolio_fit_score": None,
        "ticker": "TEST",
        "timestamp": "2026-01-01 10:00:00",
        "ist_trade": action in ("KAUFEN", "VERKAUFEN"),
    }


def _make_prices(start_price: float, n_days: int, drift: float = 0.0) -> list[dict]:
    """Erzeugt eine Liste von Preis-Dicts für n_days Tage."""
    prices: list[dict] = []
    base_date = datetime.now() - timedelta(days=n_days + 5)
    price = start_price
    for i in range(n_days):
        d = base_date + timedelta(days=i)
        price = price * (1.0 + drift)
        prices.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "close": round(price, 2),
                "high": round(price * 1.01, 2),
                "low": round(price * 0.99, 2),
            }
        )
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


def _make_journal_row(
    ticker: str = "AAPL",
    action: str = "KAUFEN",
    confidence: str = "4",
    timestamp: str = "",
    rating: str = "",
) -> dict:
    """Erzeugt eine Journal-Zeile mit Defaults."""
    if not timestamp:
        timestamp = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "ticker": ticker,
        "action": action,
        "confidence": confidence,
        "timestamp": timestamp,
        "rating": rating,
    }


def _make_row(
    ticker: str = "AAPL",
    action: str = "KAUFEN",
    final_decision: str = "GENEHMIGT",
    confidence: str = "4",
    timestamp: str = "2026-01-01 10:00:00",
) -> dict:
    """Erzeugt eine Journal-Zeile (CSV-Felder) für feedback.py."""
    return {
        "ticker": ticker,
        "action": action,
        "final_decision": final_decision,
        "confidence": confidence,
        "timestamp": timestamp,
    }


# --------------------------------------------------------------------------- #
# Tests: ist_trade-Feld in _evaluate_single
# --------------------------------------------------------------------------- #


class TestIstTradeFeld:
    """Testet das neue ist_trade-Feld in _evaluate_single."""

    def test_kaufen_ist_trade(self):
        """KAUFEN-Zeile → ist_trade=True."""
        prices = _make_prices(100, 60, drift=0.005)
        row = _make_journal_row(action="KAUFEN", timestamp="2026-01-01 10:00:00")
        result = _evaluate_single(row, prices, 90)
        assert result["ist_trade"] is True

    def test_verkaufen_ist_trade(self):
        """VERKAUFEN-Zeile → ist_trade=True."""
        prices = _make_prices(100, 60, drift=-0.005)
        row = _make_journal_row(action="VERKAUFEN", timestamp="2026-01-01 10:00:00")
        result = _evaluate_single(row, prices, 90)
        assert result["ist_trade"] is True

    def test_halten_ist_kein_trade(self):
        """HALTEN-Zeile → ist_trade=False (auch bei unbrauchbaren Daten)."""
        row = _make_journal_row(action="HALTEN", timestamp="2026-01-01 10:00:00")
        # Ohne Preise → empty-dict
        result = _evaluate_single(row, [], 90)
        assert result["ist_trade"] is False
        assert result["hit"] is None

    def test_halten_wird_weiterhin_bewertet(self):
        """HALTEN wird weiterhin bewertet (hit/rendite berechnet) — nur kein Trade."""
        prices = _make_prices(100, 60, drift=0.0)  # stabil → hit=True
        row = _make_journal_row(action="HALTEN", confidence="3", timestamp="2026-01-01 10:00:00")
        result = _evaluate_single(row, prices, 90)
        assert result["ist_trade"] is False
        assert result["hit"] is True
        assert result["rendite_pct"] is not None

    def test_empty_result_hat_neue_felder(self):
        """_empty_result enthält halten_n=0 und halten_quote=None."""
        result = _empty_result()
        assert result["halten_n"] == 0
        assert result["halten_quote"] is None


# --------------------------------------------------------------------------- #
# Tests: (a) HALTEN fließt nicht in hit_rate_gesamt ein
# --------------------------------------------------------------------------- #


class TestHaltenNichtInHitRateGesamt:
    """(a) HALTEN-Zeilen fließen nicht in hit_rate_gesamt ein."""

    def test_nur_halten_zu_treffern_hit_rate_gesamt_none(self):
        """Nur HALTEN-Treffer → hit_rate_gesamt=None (keine Trades)."""
        evals = [
            _make_eval(confidence=4.0, hit=True, action="HALTEN"),
            _make_eval(confidence=3.0, hit=True, action="HALTEN"),
            _make_eval(confidence=4.0, hit=True, action="HALTEN"),
        ]
        result = _aggregate(evals)
        assert result["hit_rate_gesamt"] is None
        assert result["halten_n"] == 3
        assert result["halten_quote"] == 1.0

    def test_halten_veraendert_hit_rate_gesamt_nicht(self):
        """Hit-Rate = 2/3 KAUFEN-Treffer; HALTEN-Misses drücken sie NICHT."""
        evals = [
            _make_eval(confidence=4.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=4.0, hit=False, action="KAUFEN"),
            _make_eval(confidence=3.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=4.0, hit=False, action="HALTEN"),
            _make_eval(confidence=4.0, hit=False, action="HALTEN"),
            _make_eval(confidence=4.0, hit=False, action="HALTEN"),
        ]
        result = _aggregate(evals)
        # Nur 3 Trades: 2 hits / 3 rated = 2/3 — die 3 HALTEN-Misses zählen nicht.
        assert math.isclose(result["hit_rate_gesamt"], 2 / 3, rel_tol=1e-9)
        assert result["halten_n"] == 3

    def test_ohne_trades_hit_rate_gesamt_none(self):
        """Ohne echte Trades → hit_rate_gesamt=None, kein Crash."""
        evals = [_make_eval(confidence=4.0, hit=False, action="HALTEN")]
        result = _aggregate(evals)
        assert result["hit_rate_gesamt"] is None
        assert result["halten_n"] == 1
        assert result["halten_quote"] == 0.0

    def test_ist_trade_fallback_bei_feiendem_feld(self):
        """Ohne ist_trade-Feld → Fallback auf action (Robustheit für alte Daten)."""
        evals = [
            {"hit": True, "action": "KAUFEN", "confidence": 4.0,
             "rendite_pct": 1.0, "ziel_erreicht": None, "stop_gerissen": None},
            {"hit": False, "action": "HALTEN", "confidence": 3.0,
             "rendite_pct": -3.0, "ziel_erreicht": None, "stop_gerissen": None},
        ]
        result = _aggregate(evals)
        assert math.isclose(result["hit_rate_gesamt"], 1.0, rel_tol=1e-9)

    def test_evaluate_journal_integration(self, tmp_path):
        """Integrationspfad: HALTEN-Misses drücken hit_rate_gesamt nicht."""
        rows = [
            _make_journal_row(ticker="AAA", action="KAUFEN", confidence="4"),
            _make_journal_row(ticker="BBB", action="KAUFEN", confidence="4"),
            _make_journal_row(ticker="CCC", action="HALTEN", confidence="3"),
            _make_journal_row(ticker="DDD", action="HALTEN", confidence="3"),
        ]
        path = _write_journal(tmp_path, rows)

        with patch(
            "concilium.evaluate._load_price_history",
            side_effect=lambda t, **kw: _make_prices(100, 60, drift=0.01),
        ):
            result = evaluate_journal(path)

        # Steigende Kurse → beide KAUFEN = Hit; HALTEN (4x +8%+) = Miss.
        # Nur Trades zählen: 2/2 = 1.0
        assert math.isclose(result["hit_rate_gesamt"], 1.0, rel_tol=1e-9)
        assert result["halten_n"] == 2
        assert result["halten_quote"] == 0.0  # +8% → nicht stabil


# --------------------------------------------------------------------------- #
# Tests: (b) HALTEN fließt nicht in Brier/Gap/Reliability-Bins ein
# --------------------------------------------------------------------------- #


class TestHaltenNichtInKalibrierung:
    """(b) HALTEN-Zeilen fließen nicht in Brier/Gap/Kalibrierung ein."""

    def test_brier_nur_trades(self):
        """Brier wird nur aus KAUFEN/VERKAUFEN berechnet (nach _aggregate-Filter)."""
        evals = [
            _make_eval(confidence=5.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=5.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=5.0, hit=True, action="KAUFEN"),
            # HALTEN mit p=1.0 aber hit=False würde Brier stark verzerren (0.64-terme):
            _make_eval(confidence=5.0, hit=False, action="HALTEN"),
            _make_eval(confidence=5.0, hit=False, action="HALTEN"),
        ]
        kal = _compute_konfidenz_kalibrierung(evals)
        # Handberechnung NUR KAUFEN: alle p=1.0, hit=1 → Brier 0
        assert math.isclose(kal["brier_score"], 0.0, abs_tol=1e-9)

    def test_gap_nur_trades(self):
        """Gap wird nur aus KAUFEN/VERKAUFEN berechnet."""
        evals = [
            _make_eval(confidence=4.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=4.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=4.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=5.0, hit=False, action="HALTEN"),
            _make_eval(confidence=5.0, hit=False, action="HALTEN"),
        ]
        result = _aggregate(evals)
        kal = result["konfidenz_kalibrierung"]
        # Nur Trades: avg_conf = 0.8, avg_hit = 1.0, gap = -0.2
        assert math.isclose(kal["durchschnittliche_konfidenz"], 0.8, abs_tol=1e-9)
        assert math.isclose(kal["durchschnittliche_tatsaechliche_hit_rate"], 1.0, abs_tol=1e-9)
        assert math.isclose(kal["kalibrierungs_gap"], -0.2, abs_tol=1e-9)
        assert kal["tendenz"] == "unterkonfident"
        assert kal["n"] == 3

    def test_halten_segment_nicht_in_nach_aktion(self):
        """nach_aktion-Segmentierung enthält KEIN HALTEN mehr; nach_rating schon."""
        evals = [
            _make_eval(confidence=5.0, hit=True, action="KAUFEN", rating="KAUFEN"),
            _make_eval(confidence=4.0, hit=False, action="VERKAUFEN", rating="VERKAUFEN"),
            _make_eval(confidence=3.0, hit=False, action="HALTEN", rating="HALTEN"),
            _make_eval(confidence=4.0, hit=True, action="HALTEN", rating="HALTEN"),
        ]
        seg = _compute_konfidenz_kalibrierung_segmentiert(evals)
        assert "HALTEN" not in seg["nach_aktion"]
        assert "KAUFEN" in seg["nach_aktion"]
        assert "VERKAUFEN" in seg["nach_aktion"]
        # nach_rating unverändert — HALTEN-Rating ist legitimes Segment
        assert "HALTEN" in seg["nach_rating"]
        assert seg["nach_rating"]["HALTEN"]["n"] == 2

    def test_reliability_bins_nur_trades(self):
        """Reliability-Bänder zählen keine HALTEN-Zeilen."""
        evals = [
            _make_eval(confidence=5.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=5.0, hit=True, action="VERKAUFEN"),
            _make_eval(confidence=5.0, hit=False, action="HALTEN"),
        ]
        bins = _compute_reliability_bins(evals)
        assert len(bins) == 1
        assert bins[0]["n"] == 2  # nur die 2 Trades


# --------------------------------------------------------------------------- #
# Tests: (c) halten_quote wird korrekt berechnet
# --------------------------------------------------------------------------- #


class TestHaltenQuote:
    """(c) halten_quote = Anteil stabiler HALTEN-Verläufe (hit=True, |rendite|<=2%)."""

    def test_halten_quote_berechnung(self):
        """2 von 4 HALTEN stabil → halten_quote=0.5; KAUFEN fließt nicht ein."""
        evals = [
            _make_eval(confidence=3.0, hit=True, action="HALTEN"),
            _make_eval(confidence=3.0, hit=False, action="HALTEN"),
            _make_eval(confidence=3.0, hit=True, action="HALTEN"),
            _make_eval(confidence=3.0, hit=False, action="HALTEN"),
            _make_eval(confidence=4.0, hit=True, action="KAUFEN"),
        ]
        result = _aggregate(evals)
        assert result["halten_n"] == 4
        assert result["halten_quote"] == 0.5

    def test_halten_quote_alle_hits(self):
        """Alle HALTEN-Treffer → halten_quote=1.0."""
        evals = [
            _make_eval(confidence=3.0, hit=True, action="HALTEN"),
            _make_eval(confidence=3.0, hit=True, action="HALTEN"),
            _make_eval(confidence=3.0, hit=True, action="HALTEN"),
        ]
        result = _aggregate(evals)
        assert result["halten_n"] == 3
        assert result["halten_quote"] == 1.0

    def test_halten_quote_none_ohne_bewertete_halten(self):
        """HALTEN ohne bewertbare hit-Werte → halten_quote=None, kein Crash."""
        evals = [
            _make_eval(confidence=None, hit=None, action="HALTEN"),
        ]
        result = _aggregate(evals)
        assert result["halten_n"] == 1
        assert result["halten_quote"] is None
        assert result["hit_rate_gesamt"] is None

    def test_halten_quote_none_ohne_halten(self):
        """Keine HALTEN-Zeilen → halten_n=0, halten_quote=None."""
        evals = [_make_eval(confidence=4.0, hit=True, action="KAUFEN")]
        result = _aggregate(evals)
        assert result["halten_n"] == 0
        assert result["halten_quote"] is None

    def test_halten_quote_nan_sicher(self):
        """NaN-Rendite bei HALTEN (hit=None) → kein Crash, Quote aus bewertbaren."""
        evals = [
            {"hit": True, "rendite_pct": float("nan"), "action": "HALTEN",
             "confidence": 3.0, "ist_trade": False},
            {"hit": False, "rendite_pct": 5.0, "action": "HALTEN",
             "confidence": 3.0, "ist_trade": False},
        ]
        result = _aggregate(evals)
        assert result["halten_n"] == 2
        # Beide Zeilen haben hit=True/False (bewertbar) → Quote 1/2
        assert result["halten_quote"] == 0.5


# --------------------------------------------------------------------------- #
# Tests: (d) KAUFEN/VERKAUFEN werden weiterhin normal bewertet
# --------------------------------------------------------------------------- #


class TestTradesUnveraendert:
    """(d) KAUFEN/VERKAUFEN werden weiterhin normal bewertet."""

    def test_kaufen_verkaufen_hit_rate_gesamt(self):
        """Gemischte Trades → Gesamt-Hit-Rate korrekt über beide Aktionen."""
        evals = [
            _make_eval(confidence=4.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=4.0, hit=False, action="VERKAUFEN"),
            _make_eval(confidence=4.0, hit=False, action="KAUFEN"),
            _make_eval(confidence=4.0, hit=True, action="VERKAUFEN"),
        ]
        result = _aggregate(evals)
        assert math.isclose(result["hit_rate_gesamt"], 0.5, rel_tol=1e-9)

    def test_brier_perfect_calibration_trades(self):
        """Perfekt kalibrierte Trades (p=1.0, alle Treffer) → Brier ≈ 0."""
        evals = [
            _make_eval(confidence=5.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=5.0, hit=True, action="VERKAUFEN"),
            _make_eval(confidence=5.0, hit=True, action="VERKAUFEN"),
            _make_eval(confidence=5.0, hit=True, action="KAUFEN"),
        ]
        kal = _compute_konfidenz_kalibrierung(evals)
        assert math.isclose(kal["brier_score"], 0.0, abs_tol=1e-9)
        assert kal["n"] == 4

    def test_nach_aktion_statistik_unveraendert(self):
        """nach_aktion enthält weiterhin alle drei Aktionen inkl. HALTEN."""
        evals = [
            _make_eval(confidence=4.0, hit=True, action="KAUFEN"),
            _make_eval(confidence=3.0, hit=False, action="HALTEN"),
            _make_eval(confidence=2.0, hit=True, action="VERKAUFEN"),
        ]
        result = _aggregate(evals)
        for action in ("KAUFEN", "HALTEN", "VERKAUFEN"):
            assert action in result["nach_aktion"]
        # HALTEN-Zeile bewertet: hit_rate 0/1 = 0.0
        assert result["nach_aktion"]["HALTEN"]["hit_rate"] == 0.0
        assert result["nach_aktion"]["HALTEN"]["n"] == 1


# --------------------------------------------------------------------------- #
# Tests: feedback.py — Proxy-Kalibrierung nur Trades
# --------------------------------------------------------------------------- #


class TestFeedbackProxyNurTrades:
    """Proxy-Kalibrierung wertet nur KAUFEN/VERKAUFEN (HALTEN ausgenommen)."""

    def test_proxy_ignoriert_halten(self):
        """HALTEN-Zeilen zählen nicht in avg_confidence/genehmigungs_rate/n."""
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="KAUFEN", confidence="4", final_decision="ABGELEHNT"),
            _make_row(ticker="C", action="HALTEN", confidence="5", final_decision="GENEHMIGT"),
            _make_row(ticker="D", action="HALTEN", confidence="5", final_decision="ABGELEHNT"),
        ]
        result = _compute_kalibrierung_proxy(rows)
        # Nur die 2 KAUFEN-Zeilen: conf=0.8, genehmigt=0.5
        assert result["n"] == 2
        assert math.isclose(result["avg_confidence"], 0.8, abs_tol=1e-9)
        assert math.isclose(result["genehmigungs_rate"], 0.5, abs_tol=1e-9)
        assert math.isclose(result["gap"], 0.3, abs_tol=1e-9)
        assert result["tendenz"] == "überkonfident"

    def test_proxy_nur_halten_leer(self):
        """Nur HALTEN-Zeilen → leeres Proxy-Ergebnis."""
        rows = [
            _make_row(ticker="A", action="HALTEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="HALTEN", confidence="4", final_decision="ABGELEHNT"),
            _make_row(ticker="C", action="HALTEN", confidence="5", final_decision="GENEHMIGT"),
        ]
        result = _compute_kalibrierung_proxy(rows)
        assert result["n"] == 0
        assert result["avg_confidence"] is None
        assert result["genehmigungs_rate"] is None

    def test_proxy_per_action_ignoriert_halten(self):
        """proxy_per_action enthält kein HALTEN mehr."""
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="C", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="D", action="HALTEN", confidence="5", final_decision="GENEHMIGT"),
            _make_row(ticker="E", action="HALTEN", confidence="5", final_decision="GENEHMIGT"),
            _make_row(ticker="F", action="HALTEN", confidence="5", final_decision="GENEHMIGT"),
        ]
        result = _compute_kalibrierung_proxy_per_action(rows)
        assert "HALTEN" not in result
        assert "KAUFEN" in result
        assert result["KAUFEN"]["n"] == 3

    def test_compute_stats_proxy_fallback_nur_trades(self, tmp_path, monkeypatch):
        """_compute_stats mit Proxy-Fallback: HALTEN nicht in der Kalibrierung."""
        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="KAUFEN", confidence="4", final_decision="ABGELEHNT"),
            _make_row(ticker="C", action="KAUFEN", confidence="4", final_decision="ABGELEHNT"),
            _make_row(ticker="D", action="HALTEN", confidence="5", final_decision="GENEHMIGT"),
            _make_row(ticker="E", action="HALTEN", confidence="5", final_decision="GENEHMIGT"),
        ]
        stats = _compute_stats(rows, min_decisions=5)

        # actions-Zählung behält HALTEN (Transparenz)
        assert stats["actions"]["HALTEN"] == 2
        # Kalibrierung (Proxy) zählt nur die 3 KAUFEN-Zeilen
        assert stats["kalibrierung"]["quelle"] == "proxy"
        assert stats["kalibrierung"]["n"] == 3
        # 1/3 genehmigt → gap = 0.8 - 0.333 = 0.467 > 0.15
        assert math.isclose(stats["kalibrierung"]["gap"], 0.8 - 1 / 3, abs_tol=1e-6)
        assert stats["kalibrierung"]["tendenz"] == "überkonfident"
        # Pro Aktion: kein HALTEN
        assert "HALTEN" not in stats["kalibrierung_pro_aktion"]

    def test_feedback_context_kalibrierung_line_ohne_halten(self, tmp_path, monkeypatch):
        """build_feedback_context rendert die Proxy-Zeile ohne HALTEN-Einfluss."""
        from concilium.feedback import build_feedback_context

        monkeypatch.setenv("CONCILIUM_STATE_DIR", str(tmp_path / "state"))
        rows = [
            _make_row(ticker="A", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="B", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="C", action="KAUFEN", confidence="4", final_decision="GENEHMIGT"),
            _make_row(ticker="D", action="HALTEN", confidence="5", final_decision="ABGELEHNT"),
            _make_row(ticker="E", action="HALTEN", confidence="5", final_decision="ABGELEHNT"),
        ]
        path = _write_journal(tmp_path, rows)
        result = build_feedback_context(path)
        # 3 KAUFEN genehmigt → Genehmigungs-Rate 100%, nicht von HALTEN gedrückt
        assert "Genehmigungs-Rate 100%" in result


# --------------------------------------------------------------------------- #
# Tests: feedback.py — echte Kalibrierung (calibration.json) nur Trades
# --------------------------------------------------------------------------- #


class _CalLike:
    """Hilfsfunktion: erzeugt eine Kalibrierungs-JSON-artige Struktur."""

    @staticmethod
    def make(
        hit_rate_gesamt: float = 0.5,
        nach_aktion: dict | None = None,
        anzahl: int = 10,
    ) -> dict:
        from datetime import datetime as _dt

        if nach_aktion is None:
            nach_aktion = {
                "KAUFEN": {"n": 6, "hit_rate": 0.5, "avg_confidence": 0.8},
            }
        return {
            "erstellt_am": _dt.now().isoformat(),
            "anzahl_entscheidungen": anzahl,
            "hit_rate_gesamt": hit_rate_gesamt,
            "nach_aktion": nach_aktion,
        }


class TestFeedbackEchtNurTrades:
    """Echte Kalibrierung: gewichtete Ø-Confidence NUR aus KAUFEN/VERKAUFEN."""

    def test_echt_ignoriert_halten_in_gewichtung(self):
        """HALTEN wird nicht in total_n/conf_sum gezählt."""
        from concilium.feedback import _compute_kalibrierung_echt

        cal = _CalLike.make(
            hit_rate_gesamt=0.5,
            nach_aktion={
                "KAUFEN": {"n": 6, "hit_rate": 0.5, "avg_confidence": 0.8},
                "HALTEN": {"n": 4, "hit_rate": 0.9, "avg_confidence": 1.0},
            },
        )
        result = _compute_kalibrierung_echt(cal)
        # Nur KAUFEN: avg_conf = 0.8 (HALTEN n=4/conf=1.0 ausgenommen)
        assert math.isclose(result["avg_confidence"], 0.8, abs_tol=1e-9)
        assert result["hit_rate"] == 0.5
        assert math.isclose(result["gap"], 0.3, abs_tol=1e-9)
        assert result["tendenz"] == "überkonfident"

    def test_echt_nur_halten_leer(self):
        """Nur HALTEN in nach_aktion → leeres Ergebnis (keine Trades)."""
        from concilium.feedback import _compute_kalibrierung_echt

        cal = _CalLike.make(
            hit_rate_gesamt=0.5,
            nach_aktion={
                "HALTEN": {"n": 4, "hit_rate": 0.9, "avg_confidence": 1.0},
            },
        )
        result = _compute_kalibrierung_echt(cal)
        assert result["avg_confidence"] is None
        assert result["gap"] is None

    def test_echt_per_action_ignoriert_halten(self):
        """_compute_kalibrierung_echt_per_action enthält kein HALTEN."""
        from concilium.feedback import _compute_kalibrierung_echt_per_action

        cal = _CalLike.make(
            nach_aktion={
                "KAUFEN": {"n": 6, "hit_rate": 0.5, "avg_confidence": 0.8},
                "HALTEN": {"n": 4, "hit_rate": 0.9, "avg_confidence": 1.0},
                "VERKAUFEN": {"n": 3, "hit_rate": 0.33, "avg_confidence": 0.6},
            },
        )
        result = _compute_kalibrierung_echt_per_action(cal)
        assert "HALTEN" not in result
        assert "KAUFEN" in result
        assert "VERKAUFEN" in result


# --------------------------------------------------------------------------- #
# Tests: report.py — Beschriftungen (nur Trades / HALTEN separat)
# --------------------------------------------------------------------------- #


class TestReportBeschriftungen:
    """Track-Record-Report kennzeichnet Hit-Rate als nur-Trades und HALTEN separat."""

    def _base_result(self) -> dict:
        result = _empty_result()
        result["anzahl_entscheidungen"] = 6
        result["nach_aktion"] = {
            "KAUFEN": {"n": 3, "hit_rate": 0.667, "avg_rendite": 2.0, "avg_confidence": 0.8},
            "HALTEN": {"n": 2, "hit_rate": 0.5, "avg_rendite": 0.5, "avg_confidence": 0.6},
            "VERKAUFEN": {"n": 1, "hit_rate": 1.0, "avg_rendite": 1.0, "avg_confidence": 0.8},
        }
        result["hit_rate_gesamt"] = 0.75
        result["halten_n"] = 2
        result["halten_quote"] = 0.5
        return result

    def test_hit_rate_label_nur_trades(self):
        """Die Übersicht nennt die Gesamt-Hit-Rate explizit als nur-Trades."""
        report = generate_track_record_report(self._base_result())
        assert "Hit-Rate (nur Trades KAUFEN/VERKAUFEN)" in report
        assert "| Hit-Rate gesamt |" not in report

    def test_halten_separat_ausgewiesen(self):
        """HALTEN wird separat mit n und Stabilitätsquote ausgewiesen."""
        report = generate_track_record_report(self._base_result())
        assert "HALTEN: 2 Entscheidungen, davon 50.0 % stabil (±2%)" in report

    def test_halten_ohne_daten_na(self):
        """Ohne halten-Daten → N/A, kein Crash."""
        result = self._base_result()
        result["halten_n"] = 0
        result["halten_quote"] = None
        report = generate_track_record_report(result)
        assert "HALTEN: 0 Entscheidungen, davon N/A stabil (±2%)" in report

    def test_kalibrierung_sektion_erklaert_ausschluss(self):
        """Kalibrierungs-Sektion erklärt den HALTEN-Ausschluss."""
        result = self._base_result()
        result["konfidenz_kalibrierung"] = {
            "brier_score": 0.2,
            "n": 4,
            "durchschnittliche_konfidenz": 0.8,
            "durchschnittliche_tatsaechliche_hit_rate": 0.75,
            "kalibrierungs_gap": 0.05,
            "tendenz": "gut kalibriert",
        }
        report = generate_track_record_report(result)
        assert "Nur echte Trades (KAUFEN/VERKAUFEN)" in report
        assert "kein Handlungsbedarf" in report
        assert "n (Trades KAUFEN/VERKAUFEN) | 4" in report
        # Neuer Label der tatsächlichen Hit-Rate
        assert "Ø tatsächliche Hit-Rate (Trades)" in report

    def test_nach_aktion_segment_tabelle_nur_trades_label(self):
        """Segment-Tabelle heißt '### Nach Aktion (nur Trades)'."""
        result = self._base_result()
        result["konfidenz_kalibrierung"] = {
            "brier_score": 0.2,
            "n": 4,
            "durchschnittliche_konfidenz": 0.8,
            "durchschnittliche_tatsaechliche_hit_rate": 0.75,
            "kalibrierungs_gap": 0.05,
            "tendenz": "gut kalibriert",
        }
        result["konfidenz_kalibrierung_segmentiert"] = {
            "nach_aktion": {
                "KAUFEN": {
                    "brier_score": 0.2,
                    "n": 3,
                    "durchschnittliche_konfidenz": 0.8,
                    "durchschnittliche_tatsaechliche_hit_rate": 0.667,
                    "kalibrierungs_gap": 0.133,
                    "tendenz": "gut kalibriert",
                },
            },
            "nach_rating": {},
        }
        report = generate_track_record_report(result)
        assert "### Nach Aktion (nur Trades)" in report

    def test_fussnote_halten_kein_trade(self):
        """Fußnote erklärt die HALTEN-Deskriptiv-Kennzahl."""
        report = generate_track_record_report(self._base_result())
        assert "HALTEN ist kein Trade" in report
        assert "fließt NICHT in die Gesamt-Hit-Rate" in report


# --------------------------------------------------------------------------- #
# Tests: Robustheit — leere/gemischte Daten craschen nie
# --------------------------------------------------------------------------- #


class TestRobustheitPhase1:
    """Neue Berechnungen sind robust (None-Guards, kein Crash)."""

    def test_aggregate_leer(self):
        """_aggregate([]) → None/0-Felder, kein Crash."""
        result = _aggregate([])
        assert result["hit_rate_gesamt"] is None
        assert result["halten_n"] == 0
        assert result["halten_quote"] is None
        assert result["anzahl_entscheidungen"] == 0

    def test_aggregate_ohne_action_feld(self):
        """Evals ohne action/ist_trade-Felder → kein Crash (Fallback)."""
        evals = [{"hit": True, "confidence": 4.0}]
        result = _aggregate(evals)
        # Fallback: kein action → kein Trade → hit_rate_gesamt None
        assert result["hit_rate_gesamt"] is None

    def test_report_mit_minimal_feldern(self):
        """Report mit fehlenden halten-Feldern → kein Crash (get-Defaults)."""
        eval_result = {
            "anzahl_entscheidungen": 3,
            "nach_aktion": {
                "KAUFEN": {"n": 3, "hit_rate": 0.667, "avg_rendite": 1.0},
                "HALTEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
                "VERKAUFEN": {"n": 0, "hit_rate": None, "avg_rendite": None},
            },
            "hit_rate_gesamt": 0.667,
            "durchschnitt_rendite_gesamt": 1.0,
            "zielkurs_trefferquote": None,
            "stop_verletzungsquote": None,
            "konfidenz_baende": [],
            "portfolio_fit_hoch": None,
            "zusammenfassung": None,
            "fehler": [],
        }
        report = generate_track_record_report(eval_result)
        assert "HALTEN: 0 Entscheidungen" in report
