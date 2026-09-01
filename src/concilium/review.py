"""Exit-Review-Modul — scannt das reale Depot (Google-Sheet) auf Verkaufskandidaten.

Der Review-Modus (``--review``) ist das Gegenstück zur Neukauf-Analyse: Statt
neue Kaufkandidaten zu prüfen, wird jede Aktien-Position im bestehenden Depot
auf die VERKAUFEN-Frage hin analysiert ("Sollte ich diese Position verkaufen?").

Ablauf:
  1. Depot laden via fetch_portfolio_positions() (gleiche Quelle wie
     portfolio_fit.py — Google-Sheet mit Tages-Cache).
  2. Filter: nur type == "Aktie" (ETFs/Commodities sind Buy-and-Hold).
  3. Optional max_positions: nur die größten Positionen (nach depot_pct).
  4. Pro Aktie: normale Pipeline (run_pipeline), aber der Report wird im
     Review-Kontext gerendert (generate_report(..., review_mode=True)).
  5. verkauf_empfehlung wird deterministisch abgeleitet (siehe
     derive_verkauf_empfehlung).

Crasht nie: Ein fehlgeschlagener Ticker wird gezählt (Warnung), aber wirft
keinen Fehler — analog zum Batch-Modus. Bei fehlendem Depot wird nichts
analysiert (leere Ergebnis-Liste statt Absturz).
"""

from __future__ import annotations

import logging
from typing import Any

from .llm import LLMClient
from .pipeline import run_pipeline
from .portfolio_fit import fetch_portfolio_positions
from .report import generate_report

logger = logging.getLogger(__name__)

# Trade-Aktionen, die für eine Bestandsposition eine Verkaufsempfehlung bedeuten.
_VERKAUF_AKTIONEN = {"VERKAUFEN", "STARK VERKAUFEN"}

# Finale Entscheidung, die für eine Bestandsposition "nicht mehr halten" bedeutet.
_ABGELEHNT = "ABGELEHNT"


def derive_verkauf_empfehlung(result: dict[str, Any] | None) -> bool:
    """Leitet deterministisch ab, ob eine Bestandsposition verkauft werden sollte.

    True, wenn:
    - ``result["trade"]["aktion"]`` in ("VERKAUFEN", "STARK VERKAUFEN") liegt, ODER
    - ``result["final"]["entscheidung"]`` == "ABGELEHNT" ist.
      ABGELEHNT bedeutet bei einer Bestandsposition "nicht mehr halten" und
      zählt daher als Verkaufsempfehlung — auch dann, wenn der Trader selbst
      (z. B. revidiert) KAUFEN vorschlug.

    False sonst (HALTEN, KAUFEN, GENEHMIGT, MODIFIZIERT …).

    Robust gegen fehlende Werte (None, leere dicts, falsche Typen) — crasht nie.
    """
    if not isinstance(result, dict):
        return False
    try:
        trade = result.get("trade")
        if isinstance(trade, dict):
            aktion = trade.get("aktion")
            if isinstance(aktion, str) and aktion.strip().upper() in _VERKAUF_AKTIONEN:
                return True
        final = result.get("final")
        if isinstance(final, dict):
            entscheidung = final.get("entscheidung")
            if (
                isinstance(entscheidung, str)
                and entscheidung.strip().upper() == _ABGELEHNT
            ):
                return True
    except Exception as exc:  # noqa: BLE001 — Ableitung crasht nie
        logger.debug("verkauf_empfehlung-Ableitung fehlgeschlagen: %s", exc)
    return False


def _depot_pct(pos: dict[str, Any]) -> float:
    """Liest depot_pct als Float (0.0 bei fehlendem/ungültigem Wert)."""
    try:
        val = pos.get("depot_pct")
        if val is None:
            return 0.0
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def run_review(
    llm: LLMClient | None,
    *,
    backtest: bool = False,
    ensemble: bool = True,
    ensemble_runs: int = 3,
    resume: bool = False,
    debate_rounds: int = 1,
    peers: list[str] | None = None,
    max_positions: int | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Exit-Review: analysiert alle Aktien des realen Depots auf Verkaufskandidaten.

    Lädt das Depot via fetch_portfolio_positions() (Google-Sheet, Tages-Cache),
    filtert auf type == "Aktie" (keine ETFs/Commodities — die sind
    Buy-and-Hold) und führt pro Aktie die normale Pipeline aus. Der Report
    wird im Review-Kontext gerendert (review_mode=True).

    Args:
        llm: LLMClient oder None (--no-llm → nur Datensnapshots).
        backtest: Backtest-Signalproxy ausführen (an run_pipeline durchgereicht).
        ensemble: Ensemble-Trader aktiv (an run_pipeline durchgereicht).
        ensemble_runs: Anzahl Ensemble-Runs (an run_pipeline durchgereicht).
        resume: Checkpoint fortsetzen (an run_pipeline durchgereicht).
        debate_rounds: Bull/Bear-Debatten-Runden (an run_pipeline durchgereicht).
        peers: Optionale Peer-Ticker (an run_pipeline durchgereicht).
        max_positions: Wenn gesetzt, werden nur die N größten Positionen
            (nach depot_pct absteigend) analysiert; der Rest zählt als
            übersprungen. None = alle Aktien analysieren.
        as_of: Optionales gepinntes Analysedatum (YYYY-MM-DD) — wird an
            run_pipeline durchgereicht (Kurs-Historie bis zu diesem Datum).

    Returns:
        dict mit:
          - "ergebnisse": {ticker: {result, report, verkauf_empfehlung,
            depot_pct, name}} für jede analysierte Aktie
          - "positions_uebersprungen": Anzahl nicht analysierter Positionen
            (ETFs/Commodities sowie von max_positions ausgeschlossene Aktien)
          - "fehler": Anzahl fehlgeschlagener Ticker (crasht den Review nicht)
          - "gesamt_positionen": Anzahl aller Depot-Positionen (inkl. ETFs)
    """
    ergebnisse: dict[str, dict[str, Any]] = {}
    fehler = 0
    positions_uebersprungen = 0

    # --- 1. Depot laden (fetch_portfolio_positions crasht nie — trotzdem
    #     defensiv absichern, damit der Review NIEMALS am Depot-Load scheitert).
    positions: list[dict[str, Any]] = []
    try:
        positions = fetch_portfolio_positions() or []
        if not isinstance(positions, list):
            positions = []
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Depot konnte nicht geladen werden: %s", exc)
        positions = []

    # --- 2. Filter: nur Aktien analysieren (ETFs/Commodities = Buy-and-Hold).
    aktien = [
        p
        for p in positions
        if isinstance(p, dict) and p.get("type") == "Aktie" and p.get("ticker")
    ]
    positions_uebersprungen = len(positions) - len(aktien)

    # --- 3. max_positions: größte Positionen zuerst (nach depot_pct) ---
    aktien.sort(key=_depot_pct, reverse=True)
    if max_positions is not None and max_positions >= 0:
        uebersprungene_aktien = aktien[max_positions:]
        aktien = aktien[:max_positions]
        positions_uebersprungen += len(uebersprungene_aktien)

    if not aktien:
        logger.info(
            "Review: keine analysierbaren Aktien-Positionen (%d Positionen gesamt, "
            "%d übersprungen).",
            len(positions),
            positions_uebersprungen,
        )
        return {
            "ergebnisse": ergebnisse,
            "positions_uebersprungen": positions_uebersprungen,
            "fehler": fehler,
            "gesamt_positionen": len(positions),
        }

    logger.info(
        "Review: analysiere %d Aktien-Positionen (%d Positionen übersprungen).",
        len(aktien),
        positions_uebersprungen,
    )

    # --- 4. Pipeline pro Aktie (fehlgeschlagener Ticker crasht den Review nicht) ---
    for pos in aktien:
        ticker = str(pos.get("ticker", "")).strip()
        if not ticker:
            positions_uebersprungen += 1
            continue

        try:
            logger.info("Review: Analyse '%s' (%.1f%% Depot)", ticker, _depot_pct(pos))
            result = run_pipeline(
                ticker,
                llm=llm,
                backtest=backtest,
                ensemble=ensemble,
                ensemble_runs=ensemble_runs,
                resume=resume,
                debate_rounds=debate_rounds,
                peers=peers,
                as_of=as_of,
            )
            if not isinstance(result, dict):
                raise ValueError(f"Unerwartetes Pipeline-Ergebnis für '{ticker}'")

            report = generate_report(result, review_mode=True)
            ergebnisse[ticker] = {
                "result": result,
                "report": report,
                "verkauf_empfehlung": derive_verkauf_empfehlung(result),
                "depot_pct": _depot_pct(pos),
                "name": pos.get("name", ticker),
            }
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — nie crashen (analog Batch-Modus)
            logger.warning("Review für '%s' fehlgeschlagen: %s", ticker, exc)
            fehler += 1

    if fehler > 0:
        logger.warning(
            "Review: %d von %d Ticker-Positionen fehlgeschlagen.", fehler, len(aktien)
        )

    return {
        "ergebnisse": ergebnisse,
        "positions_uebersprungen": positions_uebersprungen,
        "fehler": fehler,
        "gesamt_positionen": len(positions),
    }
