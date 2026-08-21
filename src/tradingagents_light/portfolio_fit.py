"""Portfolio-Fit-Modul — bewertet eine Aktie als Baustein im realen Depot.

Lädt Florians Depot aus einer Google-Sheet-Tabelle (CSV-Export) und bewertet
aus zwei Risikoperspektiven (Konzentrationsrisiko, Sektor-Overlap) sowie einer
empfohlenen Ziel-Gewichtung.
"""

from __future__ import annotations

import csv
import io
import logging
import urllib.request
from typing import Any

from .agents import _build_data_text, _call_agent
from .llm import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

SHEET_ID = "1-cTQ95Ftrw9nNnYHWAgw4EGoztNVnSw2"
_PORTFOLIO_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
_USER_AGENT = "Mozilla/5.0 (TradingAgents-Light/1.0)"

# Sheet-Symbol → Yahoo-Ticker Korrektur-Tabelle
TICKER_FIX: dict[str, str] = {
    "IS3R.TG": "IS3R.DE",
    "GZUR.DE": "GZURD.XD",
    "0FHM.TG": "COCO.L",
}

# ---------------------------------------------------------------------------
# System-Prompt für den Portfolio-Fit-Analysten
# ---------------------------------------------------------------------------

SYSTEM_PORTFOLIO_FIT = """\
Du bist ein Portfolio-Fit-Analyst. Du bewertest, ob eine analysierte Aktie ein \
sinnvoller Baustein im bestehenden Depot ist — aus ZWEI Risikoperspektiven.

Du erhältst:
1. Die analysierte Aktie mit Fundamentals (Sektor/Industry, Beta, Marktkap, Volatilität).
2. Das reale Depot mit Positionen und deren Anteil in % am Gesamtportfolio.

Bewerte explizit aus zwei Perspektiven:

(1) Konzentrationsrisiko:
    Liegt die neue Position in der Nähe schon stark gewichteter Top-Positionen?
    5%-Regel: Eine Einzelposition sollte im Idealfall nicht deutlich über ~5% liegen.
    Wenn der Ticker ODER derselbe Sektor bereits stark im Depot vertreten ist,
    erhöht das das Konzentrationsrisiko.

(2) Sektor-/Branchen-Overlap:
    Überlappt der Sektor/Industry der analysierten Aktie mit bestehenden Positionen?
    Hoher Overlap = weniger Diversifikation.

Gib zusätzlich:
- portfolio_fit_score (1-5, 5 = sehr guter Portfolio-Baustein, 1 = ungeeignet/Redundanz)
- ziel_gewichtung_pct (empfohlene Ziel-Gewichtung in % des Gesamtportfolios, \
plausible Zahl, z. B. 0 bis ~10)
- begründung (2-4 Sätze auf Deutsch)

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{
  "rolle": "Portfolio-Fit-Analyst",
  "konzentrationsrisiko_bewertung": "...",
  "sektor_overlap_bewertung": "...",
  "portfolio_fit_score": 1-5,
  "ziel_gewichtung_pct": <Zahl>,
  "begründung": "..."
}

Falls keine Portfolio-Daten verfügbar sind (portfolio_daten_verfuegbar = false), \
bewerte nur anhand der Sektor-/Branchen-Überlegung und setze das Feld \
"portfolio_daten_verfuegbar" auf false (sonst true).
"""

# ---------------------------------------------------------------------------
# Helper: Zahlen parsen (deutsches Locale, Komma-Dezimaltrenner)
# ---------------------------------------------------------------------------


def _safe_float(val: str | None) -> float | None:
    """Parst einen deutschen Zahlen-String (Komma als Dezimaltrenner).

    "34768,88607"  → 34768.88607
    "4,5"          → 4.5
    "1.234,56"     → 1234.56   (Punkt = Tausendertrenner)
    "1.234.567,89" → 1234567.89
    "" / None      → None
    """
    if val is None:
        return None
    s = val.strip()
    if not s:
        return None
    has_dot = "." in s
    has_comma = "," in s
    if has_dot and has_comma:
        # Punkt = Tausendertrenner, Komma = Dezimaltrenner
        s = s.replace(".", "").replace(",", ".")
    elif has_comma:
        # Nur Komma → Dezimaltrenner
        s = s.replace(",", ".")
    # else: nur Punkt oder keins → float() handhabt es
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# CSV-Parser (testbar ohne Netzwerk)
# ---------------------------------------------------------------------------


def _parse_positions(csv_text: str) -> list[dict[str, Any]]:
    """Parst den CSV-Text des Google-Sheets in eine Liste von Positions-Dicts.

    Erwartete Spalten (deutsches Locale): Bestand, Name, Symbol, Kurs,
    Marktwert, Anteil in %, Region.

    Zeilen mit leerem "Bestand" (Gruppierungs-/Summenzeilen) werden übersprungen.
    TICKER_FIX wird auf das Sheet-Symbol angewendet.

    Returns:
        Liste von dicts: {name, ticker, sheet_symbol, type, region,
        depot_pct (float %), value_eur (float | None)}.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    positions: list[dict[str, Any]] = []

    for row in reader:
        # Normalisiere Spaltennamen (Leerzeichen trimmen)
        row = {k.strip() if k else k: v for k, v in row.items()}

        bestand = (row.get("Bestand") or "").strip()
        if not bestand:
            continue  # Gruppierungs-/Summenzeile überspringen

        name = (row.get("Name") or "").strip()
        sheet_symbol = (row.get("Symbol") or "").strip()
        region = (row.get("Region") or "").strip()

        # Ticker-Korrektur anwenden
        ticker = TICKER_FIX.get(sheet_symbol, sheet_symbol)

        # Anteil in % direkt als Prozent-Float
        depot_pct = _safe_float(row.get("Anteil in %"))
        if depot_pct is None:
            depot_pct = 0.0

        # Marktwert (kann None sein)
        value_eur = _safe_float(row.get("Marktwert"))

        # Typ ableiten: Region-basiert als Heuristic
        if region in ("Deutschland", "Europa"):
            typ = "Aktie"
        elif region in ("USA", "Nordamerika"):
            typ = "Aktie"
        elif region in ("ETF", "Fonds"):
            typ = "ETF"
        else:
            typ = "Aktie"

        positions.append({
            "name": name,
            "ticker": ticker,
            "sheet_symbol": sheet_symbol,
            "type": typ,
            "region": region,
            "depot_pct": depot_pct,
            "value_eur": value_eur,
        })

    return positions


# ---------------------------------------------------------------------------
# Portfolio-Loader (Netzwerk, niemals crashen)
# ---------------------------------------------------------------------------


def fetch_portfolio_positions() -> list[dict[str, Any]]:
    """Lädt Florians Depot aus dem Google-Sheet (CSV-Export).

    Verwendet nur Python-Stdlib (urllib.request, csv, io) — keine neuen Dependencies.

    Returns:
        Liste von Positions-Dicts (siehe _parse_positions).
        Bei jedem Fehler (Netzwerk, CSV-Parse, keine Positionen): leere Liste [].
        NIEMALS crashen.
    """
    try:
        req = urllib.request.Request(  # noqa: S310 — HTTPS zu Google Sheets
            _PORTFOLIO_URL,
            headers={"User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            csv_text = resp.read().decode("utf-8")

        positions = _parse_positions(csv_text)

        if not positions:
            logger.warning("Portfolio-Sheet: keine Positionen gefunden.")
            return []

        logger.info("Portfolio geladen: %d Positionen.", len(positions))
        return positions

    except Exception as exc:  # noqa: BLE001 — niemals crashen
        logger.warning("Portfolio konnte nicht geladen werden: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Portfolio-Fit-Agent
# ---------------------------------------------------------------------------


def _build_portfolio_text(positions: list[dict[str, Any]]) -> str:
    """Baut einen kompakten deutschen Portfolio-Text für den LLM-Prompt.

    Zeigt nur die ~10 größten Positionen (nach depot_pct sortiert) plus
    die Gesamtanzahl der Depot-Positionen.
    """
    if not positions:
        return "Keine Portfolio-Daten verfügbar."

    total = len(positions)
    # Nach depot_pct absteigend sortieren, Top 10
    sorted_pos = sorted(positions, key=lambda p: p.get("depot_pct", 0), reverse=True)
    top = sorted_pos[:10]

    lines = [f"Depot: {total} Positionen insgesamt.", "", "Größte Positionen (Top 10):"]
    for p in top:
        name = p.get("name", "N/A")
        ticker = p.get("ticker", "N/A")
        typ = p.get("type", "N/A")
        region = p.get("region", "N/A")
        pct = p.get("depot_pct", 0)
        lines.append(f"  - {name} ({ticker}, {typ}, {region}): {pct:.1f}%")

    return "\n".join(lines)


def portfolio_fit_agent(
    data: dict[str, Any],
    llm: LLMClient,
    positions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ruft den Portfolio-Fit-Analysten auf.

    Bewertet die analysierte Aktie als Baustein im realen Depot aus zwei
    Risikoperspektiven (Konzentrationsrisiko, Sektor-Overlap) und gibt eine
    empfohlene Ziel-Gewichtung in % des Gesamtportfolios aus.

    Args:
        data: Das Daten-dict aus collect_ticker_data (mit fundamentals etc.).
        llm: LLMClient für den Agenten-Call.
        positions: Liste von Positions-Dicts aus fetch_portfolio_positions().
            Kann leer sein — dann nur Sektor-Bewertung.

    Returns:
        dict mit den Feldern aus SYSTEM_PORTFOLIO_FIT (plus _raw).
    """
    data_text = _build_data_text(data)
    portfolio_text = _build_portfolio_text(positions)

    portfolio_daten_verfuegbar = len(positions) > 0

    user_text = (
        f"Analysierte Aktie:\n{data_text}\n\n"
        f"Portfolio-Daten verfügbar: {'ja' if portfolio_daten_verfuegbar else 'nein'}\n\n"
        f"Real Depot:\n{portfolio_text}"
    )

    result = _call_agent(llm, SYSTEM_PORTFOLIO_FIT, user_text)
    result["portfolio_daten_verfuegbar"] = portfolio_daten_verfuegbar
    return result
