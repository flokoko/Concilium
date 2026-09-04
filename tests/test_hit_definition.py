"""Tests für die ehrlichere Hit-Definition (Option 3).

Neu: Die Endrendite ist die primäre Hit-Bedingung für KAUFEN/VERKAUFEN.
"Ziel erreicht" allein reicht nicht mehr, wenn der Trade am Ende negativ ist.
Ein Trade ist nur ein Hit, wenn er am Ende profitabel ist UND der Stop nicht
gerissen wurde. Die HALTEN-Logik (|rendite| <= 2%) bleibt unverändert.

Alle Tests sind offline (kein Netzwerk).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from concilium.evaluate import _evaluate_single  # noqa: E402


def _prices_with_path(days: list[tuple[str, float, float, float]]) -> list[dict]:
    """Baut Kursdaten aus (date, close, high, low)-Tupeln.

    Erster Tag ist Entry-Referenz, letzter Tag Exit-Referenz.
    """
    return [
        {"date": d, "close": c, "high": h, "low": lo}
        for d, c, h, lo in days
    ]


def _row(action: str, target: str = "", stop: str = "") -> dict:
    """Journal-Zeile mit Timestamp vor 30 Tagen."""
    ts = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "ticker": "TEST",
        "action": action,
        "target": target,
        "stop": stop,
        "confidence": "4",
        "timestamp": ts,
    }


# Kursdatum-Basis: Entscheidungen vor 30 Tagen; Daten reichen von vor 35 Tagen
# bis heute. Dates dynamisch relativ zu heute bauen, damit die Tests nicht altern.
_D = datetime.now()


def _d(days_ago: int) -> str:
    return (_D - timedelta(days=days_ago)).strftime("%Y-%m-%d")


class TestEhrlicheHitDefinitionKaufen:
    """KAUFEN: Endrendite ist die primäre Hit-Bedingung."""

    def test_kaufens_ziel_erreicht_aber_endrendite_negativ_ist_miss(self):
        """KAUFEN: Ziel erreicht, aber Trade am Ende negativ → hit=False."""
        # Entry 100 (vor 35 Tagen), Ziel 105 wird vor 20 Tagen erreicht (High 106),
        # danach fällt der Kurs — Exit heute bei 95 (rendite -5%).
        prices = _prices_with_path([
            (_d(35), 100.0, 101.0, 99.0),
            (_d(20), 102.0, 106.0, 101.0),  # High 106 ≥ Ziel 105 → Ziel erreicht
            (_d(0), 95.0, 96.0, 94.0),      # Exit: -5%
        ])
        row = _row("KAUFEN", target="105")
        result = _evaluate_single(row, prices, 90)
        assert result["ziel_erreicht"] is True
        assert result["rendite_pct"] < 0
        assert result["hit"] is False

    def test_kaufens_rendite_positiv_ohne_stop_ist_hit(self):
        """KAUFEN: rendite > 0, Stop nicht gerissen → hit=True."""
        prices = _prices_with_path([
            (_d(35), 100.0, 101.0, 99.0),
            (_d(0), 108.0, 109.0, 107.0),
        ])
        row = _row("KAUFEN")
        result = _evaluate_single(row, prices, 90)
        assert result["hit"] is True

    def test_kaufens_rendite_positiv_ohne_ziel_ist_hit(self):
        """KAUFEN: rendite > 0 ohne Zielangabe → hit=True (bisheriges Verhalten)."""
        prices = _prices_with_path([
            (_d(35), 100.0, 101.0, 99.0),
            (_d(20), 103.0, 104.0, 102.0),
            (_d(0), 102.0, 103.0, 101.0),
        ])
        row = _row("KAUFEN")
        result = _evaluate_single(row, prices, 90)
        assert result["hit"] is True

    def test_kaufens_stop_gerissen_ist_miss_auch_bei_positiver_rendite(self):
        """KAUFEN: Stop gerissen → hit=False, auch wenn Endrendite > 0."""
        prices = _prices_with_path([
            (_d(35), 100.0, 101.0, 94.0),   # Low 94 ≤ Stop 95 → Stop gerissen
            (_d(0), 102.0, 103.0, 101.0),   # Exit: +2%
        ])
        row = _row("KAUFEN", stop="95")
        result = _evaluate_single(row, prices, 90)
        assert result["stop_gerissen"] is True
        assert result["rendite_pct"] > 0
        assert result["hit"] is False

    def test_kaufens_rendite_null_ist_miss(self):
        """KAUFEN: rendite exakt 0 → hit=False (rendite_pct > 0 gefordert)."""
        prices = _prices_with_path([
            (_d(35), 100.0, 101.0, 99.0),
            (_d(0), 100.0, 101.0, 99.0),
        ])
        row = _row("KAUFEN")
        result = _evaluate_single(row, prices, 90)
        assert result["rendite_pct"] == 0
        assert result["hit"] is False

    def test_kaufens_ziel_erreicht_und_rendite_positiv_ist_hit(self):
        """KAUFEN: Ziel erreicht UND Endrendite positiv (kein Stop) → hit=True."""
        prices = _prices_with_path([
            (_d(35), 100.0, 101.0, 99.0),
            (_d(20), 106.0, 107.0, 105.0),  # High ≥ 105 → Ziel erreicht
            (_d(0), 106.0, 107.0, 105.0),   # Exit: +6%
        ])
        row = _row("KAUFEN", target="105")
        result = _evaluate_single(row, prices, 90)
        assert result["ziel_erreicht"] is True
        assert result["rendite_pct"] > 0
        assert result["hit"] is True


class TestEhrlicheHitDefinitionVerkaufen:
    """VERKAUFEN: gleiche ehrliche Logik (rendite bereits invertiert)."""

    def test_verkaufens_ziel_erreicht_aber_endrendite_negativ_ist_miss(self):
        """VERKAUFEN: Ziel erreicht, aber invertierte Endrendite negativ → miss."""
        # Entry 100, VERKAUFEN-Ziel 95 wird erreicht (Low 94), danach steigt der
        # Kurs — Exit heute bei 104 (invertierte rendite -4%).
        prices = _prices_with_path([
            (_d(35), 100.0, 101.0, 99.0),
            (_d(20), 96.0, 97.0, 94.0),    # Low 94 ≤ Ziel 95 → Ziel erreicht
            (_d(0), 104.0, 105.0, 103.0),  # Exit: invertiert -4%
        ])
        row = _row("VERKAUFEN", target="95")
        result = _evaluate_single(row, prices, 90)
        assert result["ziel_erreicht"] is True
        assert result["rendite_pct"] < 0
        assert result["hit"] is False

    def test_verkaufens_rendite_positiv_ist_hit(self):
        """VERKAUFEN: invertierte rendite > 0, kein Stop → hit=True."""
        prices = _prices_with_path([
            (_d(35), 100.0, 101.0, 99.0),
            (_d(0), 95.0, 96.0, 94.0),  # Exit: Kurs gefallen → invertiert +5%
        ])
        row = _row("VERKAUFEN")
        result = _evaluate_single(row, prices, 90)
        assert result["rendite_pct"] > 0
        assert result["hit"] is True


class TestHaltenUnveraendert:
    """HALTEN-Logik bleibt exakt wie sie ist."""

    def test_halten_stabil_ist_hit(self):
        """HALTEN: |rendite| <= 2% → hit=True (unverändert)."""
        prices = _prices_with_path([
            (_d(35), 100.0, 101.0, 99.0),
            (_d(0), 101.5, 102.5, 100.5),  # +1.5%
        ])
        row = _row("HALTEN")
        result = _evaluate_single(row, prices, 90)
        assert result["hit"] is True

    def test_halten_stark_bewegt_ist_miss(self):
        """HALTEN: |rendite| > 2% → hit=False (unverändert)."""
        prices = _prices_with_path([
            (_d(35), 100.0, 101.0, 99.0),
            (_d(0), 104.0, 105.0, 103.0),  # +4%
        ])
        row = _row("HALTEN")
        result = _evaluate_single(row, prices, 90)
        assert result["hit"] is False

    def test_halten_grenze_2pct_ist_hit(self):
        """HALTEN: exakt 2% → hit=True (Grenzfall unverändert)."""
        prices = _prices_with_path([
            (_d(35), 100.0, 101.0, 99.0),
            (_d(0), 102.0, 103.0, 101.0),  # exakt +2%
        ])
        row = _row("HALTEN")
        result = _evaluate_single(row, prices, 90)
        assert result["hit"] is True

    def test_halten_negativ_innerhalb_2pct_ist_hit(self):
        """HALTEN: -1.5% → hit=True (unverändert, auch negative Rendite)."""
        prices = _prices_with_path([
            (_d(35), 100.0, 101.0, 99.0),
            (_d(0), 98.5, 99.5, 97.5),  # -1.5%
        ])
        row = _row("HALTEN")
        result = _evaluate_single(row, prices, 90)
        assert result["hit"] is True
