"""Pytest-Konfiguration — isoliert Tests vom Tages-Cache.

Setzt CONCILIUM_CACHE_DIR="" (Cache deaktiviert) für alle Tests,
damit Offline-Tests (die yfinance mocken und deterministische Werte
erwarten) nicht durch persistierte Cache-Dateien brechen.

Individuelle Tests, die den Cache explizit testen, setzen CONCILIUM_CACHE_DIR
in ihrem eigenen monkeypatch neu.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_market_cache():
    """Deaktiviert den Tages-Cache für alle Tests (best effort).

    Setzt CONCILIUM_CACHE_DIR="" → Cache deaktiviert.
    Tests, die den Cache testen, überschreiben dies via monkeypatch.

    Scope=session, damit class-scoped fixtures (z.B. aapl_data in test_data.py)
    den Cache bereits beim ersten Aufruf deaktiviert haben.
    """
    old = os.environ.get("CONCILIUM_CACHE_DIR")
    os.environ["CONCILIUM_CACHE_DIR"] = ""
    yield
    # Restore
    if old is not None:
        os.environ["CONCILIUM_CACHE_DIR"] = old
    else:
        os.environ.pop("CONCILIUM_CACHE_DIR", None)
