"""Tests für charts.py — Chart-Generierung (matplotlib optional).

Diese Tests sind robust gegen matplotlib-Vorhandensein:
- Wenn matplotlib nicht installiert ist: Tests für None-Rückgabe.
- Wenn matplotlib installiert ist: Tests für PNG-Erzeugung.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from concilium.charts import generate_chart, is_chart_available  # noqa: E402

# --- Test-Daten ---

_MOCK_DATA = {
    "ticker": "TEST",
    "fundamentals": {"name": "Test Inc."},
    "technicals": {
        "current_price": 150.0,
        "sma50": 145.0,
        "sma200": 140.0,
        "rsi14": 55.0,
        "bollinger": {"upper": 160.0, "middle": 150.0, "lower": 140.0, "position": 0.5},
        "current_volume": 1_000_000,
        "avg_volume_30d": 900_000,
    },
    "history": [
        {"date": f"2025-01-{day:02d}", "open": 140.0, "high": 145.0, "low": 138.0,
         "close": 140.0 + day, "volume": 1_000_000}
        for day in range(1, 21)
    ],
    "sentiment": {},
    "news": [],
}


class TestChartAvailability:
    """Tests für is_chart_available()."""

    def test_returns_bool(self):
        """is_chart_available gibt einen bool zurück."""
        assert isinstance(is_chart_available(), bool)


class TestGenerateChartNoMatplotlib:
    """Tests für generate_chart bei fehlendem matplotlib."""

    def test_returns_none_when_matplotlib_unavailable(self):
        """Wenn matplotlib nicht importierbar ist, gibt generate_chart None zurück."""
        if is_chart_available():
            pytest.skip("matplotlib ist installiert — Test für Nicht-Verfügbarkeit übersprungen.")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_chart(_MOCK_DATA, tmpdir)
        assert result is None

    def test_report_section_absent_without_matplotlib(self):
        """Der Report enthält keinen Chart-Abschnitt, wenn matplotlib fehlt."""
        if is_chart_available():
            pytest.skip("matplotlib ist installiert — Test für Nicht-Verfügbarkeit übersprungen.")

        from concilium.report import generate_report

        result = {"data": _MOCK_DATA, "ticker": "TEST", "no_llm": True}
        with tempfile.TemporaryDirectory() as tmpdir:
            report = generate_report(result, reports_dir=tmpdir)
        assert "Chart" not in report
        assert "![Chart]" not in report


class TestGenerateChartWithMatplotlib:
    """Tests für generate_chart bei vorhandenem matplotlib."""

    def test_generates_png_when_available(self):
        """Wenn matplotlib verfügbar, wird ein PNG erzeugt und ein relativer Pfad zurückgegeben."""
        if not is_chart_available():
            pytest.skip("matplotlib nicht installiert — PNG-Test übersprungen.")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_chart(_MOCK_DATA, tmpdir)
            assert result is not None
            assert result.startswith("charts/")
            assert result.endswith(".png")
            # Datei existiert
            full_path = os.path.join(tmpdir, result)
            assert os.path.isfile(full_path)
            assert os.path.getsize(full_path) > 0

    def test_report_contains_chart_when_available(self):
        """Der Report enthält einen Chart-Abschnitt, wenn matplotlib verfügbar ist."""
        if not is_chart_available():
            pytest.skip("matplotlib nicht installiert — Test übersprungen.")

        from concilium.report import generate_report

        result = {"data": _MOCK_DATA, "ticker": "TEST", "no_llm": True}
        with tempfile.TemporaryDirectory() as tmpdir:
            report = generate_report(result, reports_dir=tmpdir)
        assert "![Chart]" in report
        assert "charts/" in report

    def test_no_chart_when_reports_dir_none(self):
        """Wenn reports_dir=None, wird kein Chart-Abschnitt im Report erzeugt."""
        if not is_chart_available():
            pytest.skip("matplotlib nicht installiert — Test übersprungen.")

        from concilium.report import generate_report

        result = {"data": _MOCK_DATA, "ticker": "TEST", "no_llm": True}
        report = generate_report(result, reports_dir=None)
        assert "![Chart]" not in report


class TestGenerateChartEdgeCases:
    """Edge Cases für generate_chart."""

    def test_empty_history_returns_none(self):
        """Leere Historie → None, kein Crash."""
        data = {**_MOCK_DATA, "history": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_chart(data, tmpdir)
        assert result is None

    def test_single_history_entry_returns_none(self):
        """Nur ein Historieneintrag → None (braucht mindestens 2)."""
        data = {**_MOCK_DATA, "history": [_MOCK_DATA["history"][0]]}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_chart(data, tmpdir)
        assert result is None

    def test_none_history_returns_none(self):
        """history=None → None, kein Crash."""
        data = {**_MOCK_DATA, "history": None}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_chart(data, tmpdir)
        assert result is None

    def test_no_crash_on_bad_data(self):
        """Bei kaputten Daten → None, kein Crash."""
        data = {"ticker": "BAD", "history": "not-a-list", "technicals": {}}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_chart(data, tmpdir)
        assert result is None
