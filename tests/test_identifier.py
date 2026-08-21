"""Tests für die Identifier-Auflösung (ISIN / WKN / Ticker-Erkennung).

Reine Unit-Tests für die Klassifizierung (kein Netzwerk).
Mock-basierte Tests für resolve_identifier mit Ticker (kein Netzwerk)
und für ISIN/WKN-Auflösung (mit Mock der HTTP-Aufrufe).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# src zum Pfad hinzufügen
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from concilium.data import (  # noqa: E402
    _detect_identifier_type,
    _wkn_to_isin,
    resolve_identifier,
)

# ---------------------------------------------------------------------------
# _detect_identifier_type — reine Unit-Tests (kein Netzwerk)
# ---------------------------------------------------------------------------


class TestDetectIdentifierType:
    """Tests für die Bezeichner-Klassifizierung."""

    @pytest.mark.parametrize(
        "identifier",
        [
            "DE000BASF111",
            "US0378331005",
            "IE00B4L5Y983",
            "DE0007164600",
            "LU1234567890",
        ],
    )
    def test_isin_detected(self, identifier):
        assert _detect_identifier_type(identifier) == "ISIN"

    @pytest.mark.parametrize(
        "identifier",
        [
            "716460",
            "BASF11",
            "A1EWWW",
            "123456",
            "ABC123",
        ],
    )
    def test_wkn_detected(self, identifier):
        assert _detect_identifier_type(identifier) == "WKN"

    @pytest.mark.parametrize(
        "identifier",
        [
            "AAPL",
            "MSFT",
            "NVDA",
            "RWE.DE",
            "SHEL.L",
            "ENEL.MI",
            "BRK.B",
        ],
    )
    def test_ticker_detected(self, identifier):
        assert _detect_identifier_type(identifier) == "TICKER"

    def test_lowercase_isin_treated_as_ticker(self):
        """Kleinbuchstaben werden nicht als ISIN erkannt (regex ist uppercase-only)."""
        # Die Funktion erwartet uppercase; lowercase ISINs sind Ticker
        assert _detect_identifier_type("de000basf111") == "TICKER"

    def test_too_short_is_ticker(self):
        assert _detect_identifier_type("ABC") == "TICKER"

    def test_too_long_is_ticker(self):
        """13 Zeichen sind weder ISIN (12) noch WKN (6) → Ticker."""
        assert _detect_identifier_type("ABCDEFGHIJKLM") == "TICKER"

    def test_empty_string_is_ticker(self):
        assert _detect_identifier_type("") == "TICKER"

    def test_5_chars_is_ticker(self):
        """5 Zeichen sind keine WKN (braucht genau 6)."""
        assert _detect_identifier_type("ABCDE") == "TICKER"

    def test_7_chars_is_ticker(self):
        """7 Zeichen sind keine WKN (braucht genau 6)."""
        assert _detect_identifier_type("ABCDEFG") == "TICKER"


# ---------------------------------------------------------------------------
# resolve_identifier — Ticker (kein Netzwerk nötig)
# ---------------------------------------------------------------------------


class TestResolveTicker:
    """resolve_identifier mit reinem Ticker — kein Netzwerkaufruf."""

    def test_plain_ticker_returns_as_is(self):
        ticker, meta = resolve_identifier("AAPL")
        assert ticker == "AAPL"
        assert meta == {"input_type": "TICKER", "isin": None, "wkn": None}

    def test_ticker_with_dot_returns_as_is(self):
        ticker, meta = resolve_identifier("RWE.DE")
        assert ticker == "RWE.DE"
        assert meta["input_type"] == "TICKER"
        assert meta["isin"] is None
        assert meta["wkn"] is None

    def test_lowercase_ticker_uppercased(self):
        ticker, meta = resolve_identifier("aapl")
        assert ticker == "AAPL"
        assert meta["input_type"] == "TICKER"

    def test_ticker_no_network_call(self):
        """Bei Ticker-Eingabe darf kein requests.get aufgerufen werden."""
        with patch("concilium.data.requests.get") as mock_get:
            ticker, meta = resolve_identifier("MSFT")
            mock_get.assert_not_called()
        assert ticker == "MSFT"

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="leer"):
            resolve_identifier("")


# ---------------------------------------------------------------------------
# resolve_identifier — ISIN (mit Mock)
# ---------------------------------------------------------------------------


class TestResolveISIN:
    """resolve_identifier mit ISIN — Yahoo Search API wird gemockt."""

    def test_isin_resolves_to_ticker(self):
        """DE000BASF111 → BASF.DE via Yahoo Search API (gemockt)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "quotes": [{"symbol": "BAS.DE"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("concilium.data.requests.get", return_value=mock_response):
            ticker, meta = resolve_identifier("DE000BASF111")

        assert ticker == "BAS.DE"
        assert meta["input_type"] == "ISIN"
        assert meta["isin"] == "DE000BASF111"
        assert meta["wkn"] is None

    def test_isin_us_aapl(self):
        """US0378331005 → AAPL via Yahoo Search API (gemockt)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "quotes": [{"symbol": "AAPL"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("concilium.data.requests.get", return_value=mock_response):
            ticker, meta = resolve_identifier("US0378331005")

        assert ticker == "AAPL"
        assert meta["input_type"] == "ISIN"
        assert meta["isin"] == "US0378331005"

    def test_isin_empty_quotes_raises(self):
        """Yahoo liefert keine quotes → ValueError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"quotes": []}
        mock_response.raise_for_status = MagicMock()

        with patch("concilium.data.requests.get", return_value=mock_response):
            with pytest.raises(ValueError, match="keine Treffer"):
                resolve_identifier("DE000BASF111")

    def test_isin_network_error_raises(self):
        """Netzwerkfehler → ValueError mit deutscher Meldung."""
        with patch("concilium.data.requests.get", side_effect=ConnectionError("DNS failed")):
            with pytest.raises(ValueError, match="nicht erreichbar"):
                resolve_identifier("DE000BASF111")


# ---------------------------------------------------------------------------
# resolve_identifier — WKN (mit Mock)
# ---------------------------------------------------------------------------


class TestResolveWKN:
    """resolve_identifier mit WKN — wallstreet-online + Yahoo werden gemockt."""

    def test_wkn_resolves_to_ticker(self):
        """716460 → DE0007164600 → SAP.DE (beide Aufrufe gemockt)."""
        # 1. Aufruf: wallstreet-online (HTML mit ISIN)
        wso_response = MagicMock()
        wso_response.text = '<html>...DE0007164600...</html>'
        wso_response.raise_for_status = MagicMock()

        # 2. Aufruf: Yahoo Search API (JSON mit Ticker)
        yahoo_response = MagicMock()
        yahoo_response.json.return_value = {
            "quotes": [{"symbol": "SAP.DE"}]
        }
        yahoo_response.raise_for_status = MagicMock()

        with patch(
            "concilium.data.requests.get",
            side_effect=[wso_response, yahoo_response],
        ):
            ticker, meta = resolve_identifier("716460")

        assert ticker == "SAP.DE"
        assert meta["input_type"] == "WKN"
        assert meta["isin"] == "DE0007164600"
        assert meta["wkn"] == "716460"

    def test_wkn_alphanumeric_resolves(self):
        """BASF11 → DE000BASF111 → BASF.DE (beide Aufrufe gemockt)."""
        wso_response = MagicMock()
        wso_response.text = '<html>...DE000BASF111...</html>'
        wso_response.raise_for_status = MagicMock()

        yahoo_response = MagicMock()
        yahoo_response.json.return_value = {
            "quotes": [{"symbol": "BAS.DE"}]
        }
        yahoo_response.raise_for_status = MagicMock()

        with patch(
            "concilium.data.requests.get",
            side_effect=[wso_response, yahoo_response],
        ):
            ticker, meta = resolve_identifier("BASF11")

        assert ticker == "BAS.DE"
        assert meta["input_type"] == "WKN"
        assert meta["isin"] == "DE000BASF111"
        assert meta["wkn"] == "BASF11"

    def test_wkn_no_isin_in_html_raises(self):
        """Beide Quellen liefern keine ISIN → ValueError."""
        wso_response = MagicMock()
        wso_response.text = "<html>keine ISIN hier</html>"
        wso_response.raise_for_status = MagicMock()

        with patch("concilium.data.requests.get", return_value=wso_response):
            with pytest.raises(ValueError, match="keine ISIN"):
                resolve_identifier("716460")

    def test_wkn_network_error_raises(self):
        """Beide Quellen nicht erreichbar → ValueError."""
        with patch("concilium.data.requests.get", side_effect=ConnectionError("DNS failed")):
            with pytest.raises(ValueError, match="nicht erreichbar"):
                resolve_identifier("716460")


# ---------------------------------------------------------------------------
# _wkn_to_isin — Fallback-Reihenfolge (mit Mock)
# ---------------------------------------------------------------------------


class TestWknToIsinFallback:
    """Tests für die Fallback-Reihenfolge in _wkn_to_isin."""

    def test_wallstreet_primary_wins_onvista_not_called(self):
        """wallstreet-online liefert ISIN → onvista wird NICHT aufgerufen."""
        wso_response = MagicMock()
        wso_response.text = '<html>...DE0007164600...</html>'
        wso_response.raise_for_status = MagicMock()

        with patch("concilium.data.requests.get", return_value=wso_response) as mock_get:
            isin = _wkn_to_isin("716460")

        assert isin == "DE0007164600"
        # Genau ein Aufruf (nur wallstreet-online)
        assert mock_get.call_count == 1
        called_url = mock_get.call_args[0][0]
        assert "wallstreet-online.de" in called_url

    def test_fallback_to_onvista_when_wallstreet_fails(self):
        """wallstreet-online schlägt fehl → onvista wird versucht und liefert ISIN."""
        # 1. Aufruf: wallstreet-online → ConnectionError
        # 2. Aufruf: onvista → HTML mit ISIN
        onvista_response = MagicMock()
        onvista_response.text = '<html>...DE0007164600...</html>'
        onvista_response.raise_for_status = MagicMock()

        with patch(
            "concilium.data.requests.get",
            side_effect=[ConnectionError("rate limited"), onvista_response],
        ) as mock_get:
            isin = _wkn_to_isin("716460")

        assert isin == "DE0007164600"
        # Zwei Aufrufe: wallstreet-online, dann onvista
        assert mock_get.call_count == 2
        first_url = mock_get.call_args_list[0][0][0]
        second_url = mock_get.call_args_list[1][0][0]
        assert "wallstreet-online.de" in first_url
        assert "onvista.de" in second_url

    def test_fallback_to_onvista_when_wallstreet_no_isin(self):
        """wallstreet-online HTML ohne ISIN → onvista liefert ISIN."""
        wso_response = MagicMock()
        wso_response.text = "<html>keine ISIN hier</html>"
        wso_response.raise_for_status = MagicMock()

        onvista_response = MagicMock()
        onvista_response.text = '<html>...DE000BASF111...</html>'
        onvista_response.raise_for_status = MagicMock()

        with patch(
            "concilium.data.requests.get",
            side_effect=[wso_response, onvista_response],
        ) as mock_get:
            isin = _wkn_to_isin("BASF11")

        assert isin == "DE000BASF111"
        assert mock_get.call_count == 2

    def test_both_sources_fail_raises_value_error(self):
        """Beide Quellen schlagen fehl → ValueError erwähnt beide Quellen."""
        with patch(
            "concilium.data.requests.get",
            side_effect=[ConnectionError("DNS failed"), ConnectionError("DNS failed")],
        ):
            with pytest.raises(ValueError, match="weder wallstreet-online noch onvista"):
                _wkn_to_isin("716460")

    def test_both_sources_no_isin_raises_value_error(self):
        """Beide Quellen liefern HTML ohne ISIN → ValueError."""
        empty_response = MagicMock()
        empty_response.text = "<html>keine ISIN hier</html>"
        empty_response.raise_for_status = MagicMock()

        with patch("concilium.data.requests.get", return_value=empty_response):
            with pytest.raises(ValueError, match="keine ISIN"):
                _wkn_to_isin("716460")

    def test_onvista_uses_full_user_agent(self):
        """onvista-Fallback verwendet den vollen Chrome-User-Agent."""
        wso_response = MagicMock()
        wso_response.text = "<html>keine ISIN</html>"
        wso_response.raise_for_status = MagicMock()

        onvista_response = MagicMock()
        onvista_response.text = '<html>...DE0007164600...</html>'
        onvista_response.raise_for_status = MagicMock()

        with patch(
            "concilium.data.requests.get",
            side_effect=[wso_response, onvista_response],
        ) as mock_get:
            _wkn_to_isin("716460")

        # onvista ist der 2. Aufruf
        onvista_headers = mock_get.call_args_list[1][1]["headers"]
        assert onvista_headers["User-Agent"] == (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        )
