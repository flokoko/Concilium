"""Tests für statistische Signifikanz-Metriken (DSR / MBL).

Bailey & López de Prado (2014), "The Deflated Sharpe Ratio" — Implementierung
in concilium/evaluate.py. Alle Tests OFFLINE (reine math-Formeln, kein Netz,
kein LLM, kein scipy-Import im Produktionscode).
"""

from __future__ import annotations

import math
import os
import sys
from unittest.mock import patch

# src zum Pfad hinzufügen
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest  # noqa: E402

from concilium.evaluate import (  # noqa: E402
    DEFAULT_N_TRIALS,
    _aggregate,
    _empty_result,
    _moments_skew_kurt,
    _norm_cdf,
    _norm_ppf,
    _probst,
    compute_dsr,
    compute_mbl,
)
from concilium.report import generate_track_record_report  # noqa: E402

# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #


def _make_trade(
    rendite_pct: float,
    action: str = "KAUFEN",
) -> dict:
    """Erzeugt ein Einzel-Ergebnis-dict wie aus _evaluate_single (Trade)."""
    return {
        "hit": rendite_pct > 0,
        "rendite_pct": rendite_pct,
        "ziel_erreicht": None,
        "stop_gerissen": None,
        "action": action,
        "rating": "",
        "rating_distance": None,
        "confidence": 4.0,
        "portfolio_fit_score": None,
        "ticker": "TEST",
        "timestamp": "2026-01-01 10:00:00",
        "ist_trade": True,
    }


def _trades_with_returns(returns: list[float]) -> list[dict]:
    return [_make_trade(r) for r in returns]


def _normale_renditen(n: int, mittel: float = 1.0, std: float = 1.0) -> list[float]:
    """Deterministische, breit gestreute Renditen (keine Zufalls-API nötig)."""
    basis = [-1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
    return [mittel + std * basis[i % 10] for i in range(n)]


# --------------------------------------------------------------------------- #
# Φ und Φ⁻¹ (norm_cdf / norm_ppf)
# --------------------------------------------------------------------------- #


class TestNormalQuantile:
    def test_norm_cdf_bekannte_werte(self):
        assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
        assert _norm_cdf(1.6448536269514722) == pytest.approx(0.95, abs=1e-9)

    def test_norm_ppf_bekannte_werte(self):
        assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
        assert _norm_ppf(0.95) == pytest.approx(1.6448536269514722, abs=1e-6)
        assert _norm_ppf(0.99) == pytest.approx(2.3263478740408408, abs=1e-5)

    def test_norm_ppf_roundtrip(self):
        for p in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999):
            assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-9)

    def test_norm_ppf_grenzfaelle_kein_crash(self):
        # geklemmt auf extreme Quantile (finite, nicht inf) — kein Crash
        assert _norm_ppf(0.0) is not None
        assert _norm_ppf(1.0) is not None
        assert _norm_ppf(-0.5) is not None  # finite Zahl (extrem negativ)
        assert _norm_ppf(float("nan")) is None
        assert _norm_ppf(float("inf")) is None
        assert _norm_ppf(None) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# PSR (_probst)
# --------------------------------------------------------------------------- #


class TestProbst:
    def test_sr_null_normale_momente_ist_05(self):
        """SR = 0, normale Momente, große n → PSR = Φ(0) = 0.5."""
        assert _probst(0.0, 1000, 0.0, 3.0) == pytest.approx(0.5, abs=1e-12)

    def test_hoher_sharpe_grosses_n_psr_nahe_eins(self):
        # SR = 0.2 per Trade, n = 1000, normal → PSR > 0.999
        assert _probst(0.2, 1000, 0.0, 3.0) > 0.999

    def test_monotonie_in_sharpe_und_n(self):
        psr_low = _probst(0.05, 100, 0.0, 3.0)
        psr_high = _probst(0.15, 100, 0.0, 3.0)
        assert psr_high > psr_low > 0.5
        assert _probst(0.1, 500, 0.0, 3.0) > _probst(0.1, 50, 0.0, 3.0)

    def test_negativer_sharpe_unter_05(self):
        assert _probst(-0.1, 100, 0.0, 3.0) < 0.5

    def test_kein_scipy_notwendig(self):
        """PSR/DSR/MBL funktionieren mit gesperrten scipy/numpy-Imports."""
        with patch.dict("sys.modules", {"scipy": None, "numpy": None}):
            assert _probst(0.1, 100, 0.0, 3.0) is not None
            assert compute_dsr(0.1, 100) is not None
            assert compute_mbl(1.0) is not None

    def test_degenerate_eingaben_kein_crash(self):
        assert _probst(None, 100) is None
        assert _probst(0.1, None) is None
        assert _probst(0.1, 1) is None  # n < 2
        assert _probst(float("nan"), 100) is None
        assert _probst(0.1, 100, skew=None, kurt=None) is not None  # Defaults greifen
        assert _probst(0.1, 100, skew=1e9, kurt=-1e9) is None  # Var-Term ≤ 0


# --------------------------------------------------------------------------- #
# DSR (compute_dsr)
# --------------------------------------------------------------------------- #


class TestComputeDsr:
    def test_paper_beispiel_exhibit_2(self):
        """Bailey/LdP 2014 Exhibit 2: SR=2.5 (ann., 250 d/J), T=1250,
        γ3=−2, γ4=10, V[SR]=0.5 (ann.), N=100 → DSR < 0.95;
        N=46 → DSR ≈ 0.9505 (> 0.95)."""
        sr_p = 2.5 / math.sqrt(250.0)
        v_p = 0.5 / 250.0
        d100 = compute_dsr(sr_p, 1250, skew=-2.0, kurt=10.0, n_trials=100, var_trials=v_p)
        d46 = compute_dsr(sr_p, 1250, skew=-2.0, kurt=10.0, n_trials=46, var_trials=v_p)
        assert d100 is not None and d100 < 0.95
        assert d46 is not None and d46 > 0.95
        assert d46 > d100

    def test_monotonie_in_trials(self):
        """Mehr Trials → höherer Schwellenwert → niedrigerer DSR."""
        vals = [compute_dsr(0.1, 100, 0.0, 3.0, n_trials=n) for n in (2, 10, 100, 1000)]
        assert all(
            vals[i] >= vals[i + 1] for i in range(len(vals) - 1)
        ), vals

    def test_monotonie_in_sharpe(self):
        """Höherer Sharpe → höherer DSR. Schwacher Sharpe wird korrekt
        deflationiert (unter 0.5 bei N=10 Trials, n=100)."""
        lo = compute_dsr(0.05, 100, 0.0, 3.0, n_trials=10)
        hi = compute_dsr(0.3, 100, 0.0, 3.0, n_trials=10)
        assert lo is not None and hi is not None
        assert hi > lo
        assert hi > 0.5 > lo  # Deflationierung senkt schwache Evidenz unter 50 %

    def test_monotonie_in_n(self):
        """Mehr Beobachtungen → höherer DSR (kleinere Sampling-Unsicherheit)."""
        lo = compute_dsr(0.1, 30, 0.0, 3.0, n_trials=10)
        hi = compute_dsr(0.1, 1000, 0.0, 3.0, n_trials=10)
        assert lo is not None and hi is not None and hi > lo

    def test_dsr_kleiner_gleich_psr(self):
        """Deflationierung senkt die Wahrscheinlichkeit (SR0 ≥ 0 für N ≥ 2)."""
        psr = _probst(0.1, 100, 0.0, 3.0, sharpe_benchmark=0.0)
        dsr = compute_dsr(0.1, 100, 0.0, 3.0, n_trials=50)
        assert dsr is not None and psr is not None and dsr < psr

    def test_default_n_trials_30(self):
        assert DEFAULT_N_TRIALS == 30
        d_default = compute_dsr(0.1, 100)
        d_explicit = compute_dsr(0.1, 100, n_trials=30)
        assert d_default == d_explicit

    def test_n_trials_unter_2_wird_geklemmt(self):
        """N=1 (oder 0/negativ) → Φ⁻¹(1−1/N) degeneriert → auf N=2 geklemmt."""
        d = compute_dsr(0.1, 100, 0.0, 3.0, n_trials=1)
        assert d is not None
        assert d == compute_dsr(0.1, 100, 0.0, 3.0, n_trials=2)

    def test_var_trials_wird_verwendet(self):
        """Empirische Trial-Varianz ändert das Ergebnis (größer V → niedriger DSR)."""
        klein = compute_dsr(0.1, 100, 0.0, 3.0, n_trials=10, var_trials=1e-8)
        gross = compute_dsr(0.1, 100, 0.0, 3.0, n_trials=10, var_trials=1.0)
        assert klein is not None and gross is not None
        assert klein > gross

    def test_degenerate_eingaben_kein_crash(self):
        assert compute_dsr(None, 100) is None
        assert compute_dsr(0.1, None) is None
        assert compute_dsr(0.1, 1) is None
        assert compute_dsr(float("nan"), 100) is None
        assert compute_dsr(0.1, 100, skew=1e18) is None  # Var-Term ≤ 0
        assert compute_dsr(0.1, 100, var_trials=0.0) is None
        assert compute_dsr(0.1, 100, var_trials=-1.0) is None
        assert compute_dsr(0.1, 100, n_trials=None) is not None  # Default greift


# --------------------------------------------------------------------------- #
# MBL (compute_mbl)
# --------------------------------------------------------------------------- #


class TestComputeMbl:
    def test_formel_normalfall_handrechnung(self):
        """Normalfall: MBL = 1 + z_p²·(1+0.5·SR_p²)/SR_p², SR_p = SR/√252."""
        m = compute_mbl(2.0)
        z = 1.6448536269514722
        sr_p = 2.0 / math.sqrt(252.0)
        erwartet = 1.0 + z * z * (1.0 + 0.5 * sr_p * sr_p) / (sr_p * sr_p)
        assert m == pytest.approx(erwartet, rel=1e-9)

    def test_hoher_sharpe_kleines_mbl(self):
        assert compute_mbl(3.0) < compute_mbl(2.0) < compute_mbl(1.0)

    def test_niedriger_sharpe_sehr_grosses_mbl(self):
        assert compute_mbl(0.2) > 2000.0
        assert compute_mbl(0.1) > compute_mbl(0.2) > compute_mbl(1.0)

    def test_annualisierung_skaliert_korrekt(self):
        """Gleiches Ergebnis, ob 252 annualisierte Tages- oder 52 Wochen-Sharpe."""
        m252 = compute_mbl(2.0, annualization=252.0)
        # 52 Wochen-Sharpe desselben Prozesses: SR_w = 2.0·√(52/252)
        m52 = compute_mbl(2.0 * math.sqrt(52.0 / 252.0), annualization=52.0)
        assert m252 == pytest.approx(m52, rel=1e-12)  # identisch, da SR_p identisch

    def test_pval_aenderung(self):
        """Strengeres Niveau (p=0.01) braucht mehr Trades als p=0.05."""
        assert compute_mbl(1.0, pval=0.01) > compute_mbl(1.0, pval=0.05)

    def test_nicht_positive_sharpe_gibt_none(self):
        assert compute_mbl(0.0) is None
        assert compute_mbl(-1.0) is None
        assert compute_mbl(None) is None

    def test_degenerate_eingaben_kein_crash(self):
        assert compute_mbl(1.0, pval=0.0) is None  # pval außerhalb (0,1)
        assert compute_mbl(1.0, pval=1.0) is None
        assert compute_mbl(1.0, annualization=0.0) is None
        assert compute_mbl(1.0, annualization=None) is not None  # Default 252
        assert compute_mbl(float("nan")) is None


# --------------------------------------------------------------------------- #
# Momente (skew/kurt)
# --------------------------------------------------------------------------- #


class TestMomente:
    def test_schiefe_symmetrisch_null(self):
        sk, _ = _moments_skew_kurt([-2.0, -1.0, 1.0, 2.0])
        assert sk == pytest.approx(0.0, abs=1e-12)

    def test_linksschief_negativ(self):
        # 3 × +1, eine −3: m3 = (−27+1+1+1)/4 < 0 → skew < 0 (linksschief)
        sk, _ = _moments_skew_kurt([-3.0, 1.0, 1.0, 1.0])
        assert sk is not None and sk < 0.0

    def test_rechtsschief_positiv(self):
        sk, _ = _moments_skew_kurt([-1.0, -1.0, -1.0, 3.0])
        assert sk is not None and sk > 0.0

    def test_kurtosis_normalverteilung_naiv(self):
        # (±1, ±2): Varianz-Varianz-Verhältnis → flach
        sk, ku = _moments_skew_kurt([-2.0, -1.0, 1.0, 2.0])
        assert sk == pytest.approx(0.0, abs=1e-12)
        assert ku is not None and ku < 3.0  # flacher als normal

    def test_zu_wenige_punkte_none(self):
        assert _moments_skew_kurt([]) == (None, None)
        assert _moments_skew_kurt([1.0]) == (None, None)
        assert _moments_skew_kurt([1.0, 1.0]) == (None, None)
        # Varianz 0 → None
        assert _moments_skew_kurt([2.0, 2.0, 2.0]) == (None, None)

    def test_garbage_kein_crash(self):
        vals = [None, 1.0, "x"]  # type: ignore[list-item]
        # 'x' kann nicht konvertiert werden → Exception → (None, None)
        assert _moments_skew_kurt(vals) == (None, None)


# --------------------------------------------------------------------------- #
# Integration: _aggregate
# --------------------------------------------------------------------------- #


class TestAggregationSignifikanz:
    def test_dsr_mbl_vorhanden_bei_genug_trades(self):
        """12 Trades mit Streuung → dsr/psr/mbl gesetzt, psr ≥ dsr."""
        evals = _trades_with_returns(
            [1.5, -0.5, 2.0, -1.0, 0.8, -0.3, 1.2, 0.2, -0.8, 1.0, 0.5, -0.2]
        )
        result = _aggregate(evals)
        assert result["dsr"] is not None
        assert result["dsr_ps"] is not None
        assert result["mbl"] is not None
        assert 0.0 < result["dsr"] < 1.0
        assert result["dsr_ps"] >= result["dsr"]
        assert result["siginifikanz_hinweis"] == ""

    def test_zu_wenige_trades_none_mit_hinweis(self):
        """< 10 Trade-Renditen → None + deutscher Hinweis."""
        evals = _trades_with_returns([1.0, -1.0, 0.5, -0.5, 0.3])
        result = _aggregate(evals)
        assert result["dsr"] is None
        assert result["dsr_ps"] is None
        assert result["mbl"] is None
        assert "⚠️ zu wenige Trades für Signifikanz" in result["siginifikanz_hinweis"]

    def test_result_hat_alle_neuen_keys(self):
        """Auch im Leerfall müssen alle Keys vorhanden sein (Backward-Compat)."""
        result = _aggregate([])
        for key in ("dsr", "dsr_ps", "mbl", "siginifikanz_hinweis", "signifikanz_details"):
            assert key in result
        assert result["dsr"] is None
        assert result["mbl"] is None

    def test_n_trials_aus_anzahl_entscheidungen(self):
        evals = _trades_with_returns([0.5 + (i % 7) * 0.2 - 0.6 for i in range(15)])
        result = _aggregate(evals)
        assert result["signifikanz_details"]["n_trials"] == 15
        assert result["signifikanz_details"]["n_trade_renditen"] == 15
        assert result["signifikanz_details"]["sharpe_trades_annualisiert"] is not None

    def test_halten_fliesst_nicht_ein(self):
        """HALTEN-Renditen fließen nicht in die Signifikanz (kein Trade)."""
        evals = _trades_with_returns([1.0, -1.0, 0.5, -0.5, 0.3, -0.3, 0.7, -0.7, 0.1, -0.1])
        halten = _make_trade(5.0, action="HALTEN")
        halten["ist_trade"] = False
        evals.append(halten)
        result = _aggregate(evals)
        assert result["signifikanz_details"]["n_trade_renditen"] == 10
        # 10 Renditen → Metriken berechnet
        assert result["dsr"] is not None

    def test_halten_macht_aus_zu_wenig_genug(self):
        """9 Trades + HALTEN → immer noch zu wenig (HALTEN zählt nicht)."""
        evals = _trades_with_returns([1.0, -1.0, 0.5, -0.5, 0.3, -0.3, 0.7, -0.7, 0.1])
        halten = _make_trade(5.0, action="HALTEN")
        halten["ist_trade"] = False
        evals.append(halten)
        result = _aggregate(evals)
        assert result["dsr"] is None
        assert "⚠️ zu wenige Trades für Signifikanz" in result["siginifikanz_hinweis"]

    def test_bestehende_kennzahlen_unveraendert(self):
        """Strikt additiv: hit_rate_gesamt & Brier identisch zum bisherigen Wert."""
        evals = _trades_with_returns(
            [1.5, -0.5, 2.0, -1.0, 0.8, -0.3, 1.2, 0.2, -0.8, 1.0, 0.5, -0.2]
        )
        result = _aggregate(evals)
        # 7 positive Renditen der 12 → hit_rate_gesamt muss 7/12 sein
        n_hits = len([e for e in evals if e["hit"]])
        assert result["hit_rate_gesamt"] == pytest.approx(n_hits / len(evals))
        kal = result["konfidenz_kalibrierung"]
        assert kal["n"] == 12  # alle Trades haben confidence + hit

    def test_sharpe_konvention_wie_backtest(self):
        """Ø/std(ddof=1) · √252 — derselbe Wert muss in signifikanz_details stehen."""
        renditen = [3.0, -1.0, 2.0, -2.0, 1.0, 0.5, 1.5, -0.5, 0.8, -0.8, 1.2, -1.2]
        evals = _trades_with_returns(renditen)
        result = _aggregate(evals)
        n = len(renditen)
        mittel = sum(renditen) / n
        varianz = sum((r - mittel) ** 2 for r in renditen) / (n - 1)
        erwartet = mittel / math.sqrt(varianz) * math.sqrt(252.0)
        assert result["signifikanz_details"]["sharpe_trades_annualisiert"] == pytest.approx(
            erwartet, rel=1e-9
        )

    def test_keine_streuung_hinweis_ohne_crash(self):
        """Alle Renditen identisch → kein Sharpe → Hinweis, dsr/mbl None."""
        evals = _trades_with_returns([2.0] * 12)
        result = _aggregate(evals)
        assert result["dsr"] is None
        assert result["mbl"] is None
        assert "nicht berechenbar" in result["siginifikanz_hinweis"]

    def test_nan_renditen_werden_gefiltert(self):
        evals = _trades_with_returns([1.0, -1.0, 0.5, -0.5, 0.3, -0.3, 0.7, -0.7, 0.1, float("nan")])
        # 9 gültige Renditen (NaN gefiltert) → zu wenige
        result = _aggregate(evals)
        assert result["signifikanz_details"]["n_trade_renditen"] == 9
        assert result["dsr"] is None

    def test_garbage_eingaben_never_crash(self):
        """Garbage-Zeilen (rendite_pct=None/inf/nan, fehlende Keys) → kein Crash."""
        garbage = [
            {"action": "KAUFEN", "rendite_pct": None, "hit": None},
            {"action": "KAUFEN", "rendite_pct": float("inf"), "hit": False},
            {"action": "VERKAUFEN", "rendite_pct": float("nan"), "hit": True},
            {},
        ]
        result = _aggregate(garbage)  # darf nicht werfen
        assert result["dsr"] is None
        assert result["mbl"] is None
        assert "⚠️" in result["siginifikanz_hinweis"]
        assert result["signifikanz_details"]["n_trade_renditen"] == 0
        # Nicht-numerische rendite_pct (Strings) werfen in den VORHANDENEN
        # Renditen-Comprehensions von _aggregate (Verhalten vor dieser
        # Änderung, strikt additive Metrik → nicht angetastet). Die
        # Signifikanz-Berechnung selbst fängt alle Fehler ab (try/except).


class TestEmptyResultSignifikanz:
    def test_keys_initialisiert(self):
        er = _empty_result()
        assert er["dsr"] is None
        assert er["dsr_ps"] is None
        assert er["mbl"] is None
        assert er["siginifikanz_hinweis"] == ""
        assert er["signifikanz_details"] == {
            "n_trials": 0,
            "sharpe_trades_annualisiert": None,
            "n_trade_renditen": 0,
        }

    def test_strikte_additivitaet_alle_alten_keys(self):
        """Alle bisherigen Keys bleiben unverändert vorhanden."""
        er = _empty_result()
        alte_keys = {
            "anzahl_entscheidungen", "nach_aktion", "hit_rate_gesamt", "halten_n",
            "halten_quote", "durchschnitt_rendite_gesamt", "durchschnitt_rating_distanz",
            "zielkurs_trefferquote", "stop_verletzungsquote", "konfidenz_baende",
            "portfolio_fit_hoch", "zusammenfassung", "fehler", "konfidenz_kalibrierung",
            "konfidenz_kalibrierung_segmentiert", "reliability_bins", "uebersprungen",
            "invalidation_hit_quote", "invalidation_n", "invalidation_hits",
            "invalidation_nicht_bewertbar", "invalidation_details",
        }
        for k in alte_keys:
            assert k in er, f"fehlender alter Key: {k}"


# --------------------------------------------------------------------------- #
# Report-Sektion
# --------------------------------------------------------------------------- #


class TestReportSignifikanz:
    def test_sektion_mit_werten(self):
        evals = _trades_with_returns(
            [1.5, -0.5, 2.0, -1.0, 0.8, -0.3, 1.2, 0.2, -0.8, 1.0, 0.5, -0.2]
        )
        result = _aggregate(evals)
        report = generate_track_record_report(result)
        assert "## Statistische Signifikanz (DSR / MBL)" in report
        assert "DSR (P[Strategie-Schärfe > 0]" in report
        assert "MBL (Mindest-Trades" in report
        assert "Wahrscheinlichkeit, dass die tatsächliche Strategie-Schärfe > 0 ist" in report

    def test_sektion_leerfall_hinweis(self):
        """DSR/MBL None → deutscher 'zu wenige Trades'-Hinweis im Report."""
        result = _empty_result()
        result["anzahl_entscheidungen"] = 3
        result["siginifikanz_hinweis"] = "⚠️ zu wenige Trades für Signifikanz."
        report = generate_track_record_report(result)
        assert "## Statistische Signifikanz (DSR / MBL)" in report
        assert "⚠️ zu wenige Trades für Signifikanz." in report

    def test_sektion_vor_portfolio_fit(self):
        """Sektion steht NACH der Konfidenz-Kalibrierung und VOR Portfolio-Fit."""
        evals = _trades_with_returns(
            [1.5, -0.5, 2.0, -1.0, 0.8, -0.3, 1.2, 0.2, -0.8, 1.0, 0.5, -0.2]
        )
        # Portfolio-Fit-Sektion rendiert nur mit fit-Score ≥ 4
        for e in evals[:4]:
            e["portfolio_fit_score"] = 5.0
        result = _aggregate(evals)
        report = generate_track_record_report(result)
        idx_sig = report.find("## Statistische Signifikanz (DSR / MBL)")
        idx_pf = report.find("## Portfolio-Fit-Zusammenhang")
        assert idx_sig != -1 and idx_pf != -1 and idx_sig < idx_pf

    def test_report_empty_result_hat_sektion_ohne_crash(self):
        report = generate_track_record_report(_empty_result())
        assert "## Statistische Signifikanz (DSR / MBL)" in report
        assert "zu wenige Trades für Signifikanz" in report
