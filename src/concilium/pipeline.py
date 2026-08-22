"""Pipeline-Modul — Orchestrierung der Agenten-Kette."""

from __future__ import annotations

import logging
from typing import Any

from .agents import (
    _build_data_text,
    analyst_team,
    debate,
    ensemble_trader,
    portfolio_manager,
    risk_manager,
    trade_revision,
    trader,
)
from .checkpoint import clear_checkpoint, load_checkpoint, save_checkpoint
from .data import collect_ticker_data
from .feedback import build_feedback_context, build_reflection_context
from .llm import LLMClient
from .portfolio_fit import fetch_portfolio_positions, portfolio_fit_agent

logger = logging.getLogger(__name__)

# Reihenfolge der Agenten-Schritte (für _completed_steps-Buchhaltung).
# "data" = Schritt 1 (Daten + Kontext), "analysts" = 2, "debate" = 3, etc.
_STEP_ORDER = [
    "data",
    "analysts",
    "debate",
    "trade",
    "risk",
    "portfolio_fit",
    "trade_revision",
    "final",
]


def _mark_completed(result: dict[str, Any], step: str) -> None:
    """Trägt step in result['_completed_steps'] ein (idempotent, ordnungs-erhaltend)."""
    completed = result.setdefault("_completed_steps", [])
    if step not in completed:
        completed.append(step)


def _is_completed(result: dict[str, Any], step: str) -> bool:
    """Gibt True zurück, wenn step in _completed_steps enthalten ist."""
    return step in result.get("_completed_steps", [])


def _save_step(result: dict[str, Any], ticker: str, step: str) -> None:
    """Markiert step als abgeschlossen und schreibt einen Checkpoint."""
    _mark_completed(result, step)
    try:
        save_checkpoint(result, ticker)
        logger.info("Checkpoint gespeichert (Schritt %s)", step)
    except Exception as exc:  # noqa: BLE001 — Checkpoint-Fehler dürfen nie crashen
        logger.warning("Checkpoint-Fehler bei Schritt %s: %s", step, exc)


def run_pipeline(
    ticker: str,
    llm: LLMClient | None = None,
    backtest: bool = False,
    peers: list[str] | None = None,
    ensemble: bool = True,
    ensemble_runs: int = 3,
    resume: bool = False,
) -> dict[str, Any]:
    """Führt die komplette Trading-Analysis-Pipeline aus.

    Schritte:
      1. Datensammlung (yfinance)
      2. Analysten-Team (3 LLM-Calls) — nur wenn llm gegeben
      3. Bull/Bear-Debatte (2 LLM-Calls) — nur wenn llm gegeben
      4. Trade-Vorschlag (1 LLM-Call oder Ensemble) — nur wenn llm gegeben
      5. Risk-Manager (1 LLM-Call) — nur wenn llm gegeben
      5b. Portfolio-Fit-Analyst (1 LLM-Call) — nur wenn llm gegeben
      6. Portfolio-Manager finale Entscheidung (1 LLM-Call) — nur wenn llm gegeben
      7. Optional: Backtest-Signalproxy

    Args:
        ticker: Ticker-Symbol.
        llm: LLMClient oder None für --no-llm Modus.
        backtest: Ob Backtest-Signalproxy ausgeführt werden soll.
        peers: Optionale Liste von Peer-Ticker-Symbolen für den Vergleich.
        ensemble: Ob der Trader als Ensemble (Mehrere Runs) ausgeführt wird.
        ensemble_runs: Anzahl der Ensemble-Runs (nur relevant wenn ensemble=True).
        resume: Wenn True, wird ein vorhandener Checkpoint geladen und nur die
            fehlenden Schritte ab der letzten abgeschlossenen Stelle ausgeführt.
            Default False — unverändertes Verhalten (von vorn).

    Returns:
        dict mit allen Zwischenergebnissen.
    """
    result: dict[str, Any] = {}

    # --- Resume: Checkpoint laden, falls vorhanden und gewünscht ---
    if resume:
        cp = load_checkpoint(ticker)
        if cp is not None:
            result = cp
            completed = result.get("_completed_steps", [])
            logger.info(
                "Resume aktiv — Checkpoint geladen, abgeschlossen: %s",
                ", ".join(completed) if completed else "(keine)",
            )
        else:
            logger.info("Resume aktiv, aber kein Checkpoint gefunden — starte von vorn.")
    else:
        # Auch ohne resume: eventuell vorhandenen Checkpoint ignorieren (nicht löschen).
        pass

    # --- 1. Daten sammeln ---
    if not _is_completed(result, "data"):
        logger.info("Schritt 1: Sammle Marktdaten für %s", ticker)
        data = collect_ticker_data(ticker, peers=peers)
        result["data"] = data
        result["ticker"] = data["ticker"]

        # --- 1b. data_text einmal berechnen (für alle Agenten-Prompts) ---
        data_text = _build_data_text(data) if llm is not None else None
        result["_data_text"] = data_text

        # --- 1c. Feedback-Kontext einmal berechnen (Track-Record-Historie) ---
        feedback_context = build_feedback_context() if llm is not None else ""
        result["_feedback_context"] = feedback_context

        # --- 1d. Reflexions-Kontext (realisierter Return der letzten Entscheidung) ---
        reflection_context = ""
        if llm is not None:
            reflection_context = build_reflection_context(ticker=ticker, llm=llm)
        result["reflection"] = reflection_context or None
        result["_reflection_context"] = reflection_context

        _save_step(result, ticker, "data")
    else:
        # Daten aus Checkpoint übernehmen
        data = result["data"]
        data_text = result.get("_data_text")
        feedback_context = result.get("_feedback_context", "")
        reflection_context = result.get("_reflection_context", "")

    # --- Optional: Backtest ---
    if backtest and "backtest" not in result:
        logger.info("Schritt 1b: Führe Backtest-Signalproxy aus")
        from .backtest import run_backtest

        result["backtest"] = run_backtest(data)

    # --- Wenn kein LLM: nur Datensnapshot ---
    if llm is None:
        logger.info("Kein LLM-Client — nur Datensnapshot, Agenten übersprungen.")
        result["no_llm"] = True
        # Im No-LLM-Modus räumen wir den Checkpoint ebenfalls auf (alles fertig).
        clear_checkpoint(ticker)
        return result

    result["no_llm"] = False

    # --- 2. Analysten-Team ---
    if not _is_completed(result, "analysts"):
        logger.info("Schritt 2: Analysten-Team wird aufgerufen")
        analysts = analyst_team(data, llm, data_text=data_text)
        result["analysts"] = analysts
        _save_step(result, ticker, "analysts")
    else:
        analysts = result["analysts"]

    # --- 3. Debatte ---
    if not _is_completed(result, "debate"):
        logger.info("Schritt 3: Bull/Bear-Debatte")
        debate_result = debate(analysts, llm)
        result["debate"] = debate_result
        _save_step(result, ticker, "debate")
    else:
        debate_result = result["debate"]

    # --- 4. Trader (oder Ensemble-Trader) ---
    if not _is_completed(result, "trade"):
        if ensemble:
            logger.info(
                "Schritt 4: Ensemble-Trader (%d Runs) erstellt Trade-Vorschlag",
                ensemble_runs,
            )
            trade = ensemble_trader(
                analysts,
                debate_result,
                llm,
                runs=ensemble_runs,
                feedback_context=feedback_context,
                reflection_context=reflection_context,
            )
        else:
            logger.info("Schritt 4: Trader erstellt Trade-Vorschlag (Single-Run)")
            trade = trader(
                analysts,
                debate_result,
                llm,
                feedback_context=feedback_context,
                reflection_context=reflection_context,
            )
        result["trade"] = trade
        _save_step(result, ticker, "trade")
    else:
        trade = result["trade"]

    # --- 5. Risk-Manager ---
    if not _is_completed(result, "risk"):
        logger.info("Schritt 5: Risk-Manager bewertet Risiko")
        risk = risk_manager(
            trade, data, llm, data_text=data_text, feedback_context=feedback_context
        )
        result["risk"] = risk
        _save_step(result, ticker, "risk")
    else:
        risk = result["risk"]

    # --- 5b. Portfolio-Fit (zwischen Risk-Manager und Portfolio-Manager) ---
    if not _is_completed(result, "portfolio_fit"):
        result["portfolio_fit"] = None
        try:
            logger.info("Schritt 5b: Portfolio-Fit-Analyst bewertet Depot-Fit")
            positions = fetch_portfolio_positions()
            portfolio_fit = portfolio_fit_agent(data, llm, positions, data_text=data_text)
            result["portfolio_fit"] = portfolio_fit
        except Exception as exc:  # noqa: BLE001 — nie crashen
            logger.warning("Portfolio-Fit fehlgeschlagen: %s", exc)
            result["portfolio_fit"] = None
        _save_step(result, ticker, "portfolio_fit")

    # --- 5c. Trade-Revision (2nd Pass) ---
    if not _is_completed(result, "trade_revision"):
        result["trade_original"] = None
        result["trade_revised"] = False
        try:
            logger.info("Schritt 5c: Trade-Revision (2nd Pass)")
            original_trade = trade
            revised = trade_revision(
                original_trade,
                risk,
                result.get("portfolio_fit"),
                llm,
                feedback_context=feedback_context,
                reflection_context=reflection_context,
            )
            result["trade_original"] = original_trade
            result["trade"] = revised
            result["trade_revised"] = True
            trade = revised
        except Exception as exc:  # noqa: BLE001 — nie crashen
            logger.warning("Trade-Revision fehlgeschlagen: %s", exc)
        _save_step(result, ticker, "trade_revision")

    # --- 6. Portfolio-Manager ---
    if not _is_completed(result, "final"):
        logger.info("Schritt 6: Portfolio-Manager trifft finale Entscheidung")
        final = portfolio_manager(
            trade,
            risk,
            llm,
            portfolio_fit=result.get("portfolio_fit"),
            feedback_context=feedback_context,
            reflection_context=reflection_context,
        )
        result["final"] = final
        _save_step(result, ticker, "final")

    # --- Feature 4: Entscheidungs-Journal ---
    # Nur im LLM-Modus (llm nicht None) und wenn final existiert
    try:
        from .journal import append_decision

        append_decision(result)
        result["_journal_written"] = True
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Entscheidung konnte nicht ins Journal geschrieben werden: %s", exc)
        result["_journal_written"] = False

    # --- Erfolgreicher Lauf: Checkpoint aufräumen ---
    clear_checkpoint(ticker)

    return result
