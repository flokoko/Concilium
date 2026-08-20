"""Pipeline-Modul — Orchestrierung der Agenten-Kette."""

from __future__ import annotations

import logging
from typing import Any

from .agents import analyst_team, debate, portfolio_manager, risk_manager, trader
from .data import collect_ticker_data
from .llm import LLMClient

logger = logging.getLogger(__name__)


def run_pipeline(
    ticker: str,
    llm: LLMClient | None = None,
    backtest: bool = False,
    peers: list[str] | None = None,
) -> dict[str, Any]:
    """Führt die komplette Trading-Analysis-Pipeline aus.

    Schritte:
      1. Datensammlung (yfinance)
      2. Analysten-Team (3 LLM-Calls) — nur wenn llm gegeben
      3. Bull/Bear-Debatte (2 LLM-Calls) — nur wenn llm gegeben
      4. Trade-Vorschlag (1 LLM-Call) — nur wenn llm gegeben
      5. Risk-Manager (1 LLM-Call) — nur wenn llm gegeben
      6. Portfolio-Manager finale Entscheidung (1 LLM-Call) — nur wenn llm gegeben
      7. Optional: Backtest-Signalproxy

    Args:
        ticker: Ticker-Symbol.
        llm: LLMClient oder None für --no-llm Modus.
        backtest: Ob Backtest-Signalproxy ausgeführt werden soll.
        peers: Optionale Liste von Peer-Ticker-Symbolen für den Vergleich.

    Returns:
        dict mit allen Zwischenergebnissen.
    """
    result: dict[str, Any] = {}

    # --- 1. Daten sammeln ---
    logger.info("Schritt 1: Sammle Marktdaten für %s", ticker)
    data = collect_ticker_data(ticker, peers=peers)
    result["data"] = data
    result["ticker"] = data["ticker"]

    # --- Optional: Backtest ---
    if backtest:
        logger.info("Schritt 1b: Führe Backtest-Signalproxy aus")
        from .backtest import run_backtest

        result["backtest"] = run_backtest(data)

    # --- Wenn kein LLM: nur Datensnapshot ---
    if llm is None:
        logger.info("Kein LLM-Client — nur Datensnapshot, Agenten übersprungen.")
        result["no_llm"] = True
        return result

    result["no_llm"] = False

    # --- 2. Analysten-Team ---
    logger.info("Schritt 2: Analysten-Team wird aufgerufen")
    analysts = analyst_team(data, llm)
    result["analysts"] = analysts

    # --- 3. Debatte ---
    logger.info("Schritt 3: Bull/Bear-Debatte")
    debate_result = debate(analysts, llm)
    result["debate"] = debate_result

    # --- 4. Trader ---
    logger.info("Schritt 4: Trader erstellt Trade-Vorschlag")
    trade = trader(analysts, debate_result, llm)
    result["trade"] = trade

    # --- 5. Risk-Manager ---
    logger.info("Schritt 5: Risk-Manager bewertet Risiko")
    risk = risk_manager(trade, data, llm)
    result["risk"] = risk

    # --- 6. Portfolio-Manager ---
    logger.info("Schritt 6: Portfolio-Manager trifft finale Entscheidung")
    final = portfolio_manager(trade, risk, llm)
    result["final"] = final

    # --- Feature 4: Entscheidungs-Journal ---
    # Nur im LLM-Modus (llm nicht None) und wenn final existiert
    try:
        from .journal import append_decision

        append_decision(result)
        result["_journal_written"] = True
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Entscheidung konnte nicht ins Journal geschrieben werden: %s", exc)
        result["_journal_written"] = False

    return result
