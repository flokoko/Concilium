"""Pipeline-Modul — Orchestrierung der Agenten-Kette."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

# Referenz auf das feedback-Modul (für die Rückwärtskompatibilitäts-Prüfung in
# Schritt 1d: build_reflection_context.__module__ == _feedback_module.__name__
# heißt "ungepatcht" → Cross-Ticker-Teil (C4) wird ergänzt).
import concilium.feedback as _feedback_module

from . import config
from .agents import (
    _apply_technik_signal,
    _build_data_text,
    _cap_position_by_volatility,
    _extract_current_price,
    analyst_team,
    debate,
    ensemble_trader,
    portfolio_manager,
    risk_manager,
    trade_revision,
    trader,
    trader_exit,
)
from .checkpoint import clear_checkpoint, load_checkpoint, save_checkpoint
from .data import collect_ticker_data
from .feedback import (
    build_cross_ticker_context,
    build_feedback_context,
    build_reflection_context,
    resolve_pending_reflections,
)
from .llm import LLMClient

# Import-Hinweis: _dampen_ziel_gewichtung wird bewusst direkt importiert —
# die Dämpfung ist ein expliziter, getesteter Pipeline-Schritt (siehe unten).
from .portfolio_fit import (  # noqa: F401 — _dampen_ziel_gewichtung: siehe Schritt 5b
    _dampen_ziel_gewichtung,
    fetch_portfolio_positions,
    portfolio_fit_agent,
)

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


# ---------------------------------------------------------------------------
# Final-Guard (Punkt 4): harte Obergrenze für die Ziel-Gewichtung
# ---------------------------------------------------------------------------


def _apply_final_position_guard(
    portfolio_fit: dict[str, Any] | None,
    trade: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Kappt die Ziel-Gewichtung am harten Maximum (in-place) — Final-Guard (P4).

    Letzter deterministischer Sizing-Schritt NACH dem Portfolio-Manager:
    Kein LLM-Output (auch kein PM-MODIFIZIERT-Re-Weighting) darf
    ``portfolio_fit["ziel_gewichtung_pct"]`` über ``config.max_position_pct()``
    (CONCILIUM_MAX_POSITION_PCT, Default 15.0 %) heben.

    Verhalten:
    - Clamp nur bei endlichem, numerischem ``ziel_gewichtung_pct > 0``;
      der Cap senkt NUR (nie anheben) — Werte <= Max bleiben unverändert.
    - Metadaten ``portfolio_fit["_final_guard"]`` = {"max_pct", "original",
      "gekappt"} (gekappt=False, wenn kein Cap griff). provenance-Feld
      ``ziel_gewichtung_original`` (Dämpfung) wird NICHT angetastet.
    - Konsistenz-Hinweis: Bei KAUFEN/STARK KAUFEN wird
      ``trade["_final_guard_consistency"]`` (bool) gesetzt — finale
      Ziel-Gewichtung <= hartes Maximum. Reine Metadaten (per-trade
      ``positionsanteil`` und Portfolio-Ziel-Gewichtung bleiben
      unterschiedliche Konzepte — kein Zwangs-Match).
    - Resume-Idempotenz: Der Cap senkt nur — ein bereits gekappter Wert
      wird gegen denselben Max-Wert nicht verschoben; ein bereits
      gespeichertes ``original`` bleibt erhalten (kein Provenance-Verlust).
      Ein NACHträgliches Anheben (z. B. PM-Re-Weight im Portfolio-Modus)
      wird beim erneuten Apply erneut gekappt.
    - Crasht nie (try/except; bei Fehlern portfolio_fit unverändert).
    """
    try:
        if not isinstance(portfolio_fit, dict):
            return portfolio_fit

        max_pct = config.max_position_pct()

        ziel_raw = portfolio_fit.get("ziel_gewichtung_pct")
        ziel_num: float | None = None
        if isinstance(ziel_raw, int | float) and not isinstance(ziel_raw, bool):
            ziel_f = float(ziel_raw)
            if math.isfinite(ziel_f):  # NaN/±Inf → kein Clamp möglich
                ziel_num = ziel_f

        # Bereits vorhandene Guard-Metadaten (Resume): original + gekappt
        # bewahren, damit keine Provenance verloren geht.
        existing = portfolio_fit.get("_final_guard")
        existing = existing if isinstance(existing, dict) else {}

        original = existing.get("original")
        if ziel_num is None:
            # Nicht-numerisches Ziel: kein Clamp möglich (nur Metadaten).
            if original is None:
                original = ziel_raw if ziel_raw is not None else existing.get("original")
            portfolio_fit["_final_guard"] = {
                "max_pct": max_pct,
                "original": original,
                "gekappt": False,
            }
            return portfolio_fit

        ziel_f = ziel_num
        if original is None:
            original = ziel_f

        gekappt = bool(ziel_f > max_pct) or bool(existing.get("gekappt"))
        if ziel_f > max_pct:
            portfolio_fit["ziel_gewichtung_pct"] = max_pct

        portfolio_fit["_final_guard"] = {
            "max_pct": max_pct,
            "original": original,
            "gekappt": gekappt,
        }

        # Konsistenz-Hinweis (nur Metadaten, kein Zwangs-Match mit
        # positionsanteil): finale Ziel-Gewichtung <= hartes Maximum?
        if isinstance(trade, dict):
            aktion = str(trade.get("aktion", "")).strip().upper()
            if aktion in ("KAUFEN", "STARK KAUFEN"):
                final_ziel = portfolio_fit.get("ziel_gewichtung_pct")
                final_ziel_num: float | None = None
                if isinstance(final_ziel, int | float) and not isinstance(
                    final_ziel, bool
                ):
                    fz = float(final_ziel)
                    if math.isfinite(fz):
                        final_ziel_num = fz
                trade["_final_guard_consistency"] = bool(
                    final_ziel_num is not None and final_ziel_num <= max_pct
                )
    except Exception:  # noqa: BLE001 — Guard darf nie crashen
        return portfolio_fit
    return portfolio_fit


# Analysten-Keys, deren invalidation-Feld aggregiert wird (Reihenfolge
# deterministisch wie im Report).
_INVALIDATION_ROLES: list[str] = [
    "fundamental",
    "technical",
    "sentiment",
    "macro_news",
    "social",
]


def _aggregate_invalidation(analysts: Any) -> str:
    """Aggregiert die Analysten-Invalidierungen zu einem kompakten String.

    Stufe 1: Jeder Analyst liefert optional ein ``invalidation``-Feld
    (freitextliche, überprüfbare Bedingungen). Für das Journal werden alle
    nicht-leeren Beiträge pro Rolle geprefixt und mit "; " verbunden
    (Format: "fundamental: ...; technical: ..."). Deterministisch und
    nie-crashend: fehlende/leere/nicht-String-Werte werden übersprungen,
    bei gar keinem Beitrag kommt "" zurück (Legacy-Verhalten).

    Args:
        analysts: Das analysts-dict aus analyst_team (oder beliebiger
            Ersatzwert — z. B. MagicMock-Rückgabe in Tests; alles Nicht-dict
            ergibt "").

    Returns:
        Kompakter String oder "" (nie None, nie ein Crash).
    """
    if not isinstance(analysts, dict):
        return ""
    parts: list[str] = []
    for key in _INVALIDATION_ROLES:
        a = analysts.get(key)
        if not isinstance(a, dict):
            continue
        raw = a.get("invalidation")
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if not text:
            continue
        parts.append(f"{key}: {text}")
    return "; ".join(parts)


# Key unter dem der Konfigurations-Fingerprint im result-dict (und damit im
# Checkpoint) persistiert wird. Underscore-Präfix = interner Bookkeeping-Key,
# wird vom Journal/Report nicht ausgewertet (analog _completed_steps etc.).
_FINGERPRINT_KEY = "_pipeline_fingerprint"


def _pipeline_fingerprint(
    ensemble: bool,
    ensemble_runs: int,
    peers: list[str] | None,
    debate_rounds: int,
    backtest: bool,
    deep_think_model: str | None = None,
    quick_think_model: str | None = None,
) -> str:
    """Erzeugt einen deterministischen Fingerprint der Pipeline-Konfiguration.

    Der Fingerprint deckt alle run_pipeline-Parameter ab, die die Zwischen-
    ergebnisse (Analysten, Debatte, Trade, Risk, Portfolio-Fit) beeinflussen
    können — mit Ausnahme von ``as_of`` (wird separat über den bestehenden
    as_of-Check geprüft). Bei Resume stellen Checkpoint-Fingerprint und
    aktueller Fingerprint sicher, dass nur bei identischer Konfiguration
    fortgeschrieben wird.

    Die Deep-/Quick-Think-Split-Modelle sind Teil des Fingerprints, weil sie
    die LLM-Antworten der Agenten direkt beeinflussen — ein Resume mit
    anderem Split würde inkonsistente Zwischenergebnisse mischen. Bewusst
    nur aufgenommen, WENN gesetzt (None/'' → gleiche Konfiguration wie vor
    dem Split): Vor dem Split erstellte Checkpoints (ohne Modell-Keys im
    Fingerprint-JSON) bleiben damit kompatibel, solange kein Split aktiv ist.

    Deterministisch: identische Eingaben → identischer String. Peers werden
    sortiert normalisiert (None und [] sind äquivalent), damit die Listen-
    reihenfolge keinen falschen Konflikt erzeugt.
    """
    config = {
        "ensemble": bool(ensemble),
        "ensemble_runs": int(ensemble_runs),
        "peers": sorted(peers) if peers else [],
        "debate_rounds": int(debate_rounds),
        "backtest": bool(backtest),
    }
    # Split-Modelle nur bei aktivem Split aufnehmen (Rückwärtskompatibilität
    # mit Altdaten-Checkpoints, deren Fingerprint diese Keys nicht enthält).
    if deep_think_model:
        config["deep_think_model"] = str(deep_think_model)
    if quick_think_model:
        config["quick_think_model"] = str(quick_think_model)
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


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
    portfolio_context: dict[str, Any] | None = None,
    skip_final: bool = False,
    debate_rounds: int = 1,
    as_of: str | None = None,
    journal: bool = True,
    exit_mode: bool = False,
    deep_think_model: str | None = None,
    quick_think_model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Führt die komplette Trading-Analysis-Pipeline aus.

    Schritte:
      1. Datensammlung (yfinance)
      2. Analysten-Team (4 LLM-Calls: Fundamental, Technik, Sentiment, Makro/News) — nur wenn llm gegeben
      3. Bull/Bear-Debatte (2 LLM-Calls) — nur wenn llm gegeben
      4. Trade-Vorschlag (1 LLM-Call oder Ensemble) — nur wenn llm gegeben
      5. Risk-Manager (1 LLM-Call) — nur wenn llm gegeben
      5b. Portfolio-Fit-Analyst (1 LLM-Call) — nur wenn llm gegeben
      5c. Trade-Revision (2nd Pass)
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
        portfolio_context: Optionaler Gesamt-Portfolio-Kontext (Korrelation,
            Overlap, Konzentration über alle analysierten Titel). Wenn gesetzt,
            wird er dem Portfolio-Manager als zusätzlicher Kontext übergeben.
        skip_final: Wenn True, werden der Portfolio-Manager-Schritt (Schritt 6)
            UND der Journal-Schritt (append_decision) übersprungen. Stattdessen
            wird ``result["_final_pending"] = True`` gesetzt und
            ``result["final"]`` bleibt None. Die Vor-Schritte laufen normal.
            Dies wird vom Portfolio-Modus (``run_portfolio``) verwendet, um den
            PM erst nach Berechnung des Portfolio-Kontexts einmalig aufzurufen.
            Default False — unverändertes Verhalten.
        as_of: Optionales gepinntes Analysedatum (YYYY-MM-DD) — wird an
            collect_ticker_data durchgereicht (Kurs-Historie bis zu diesem
            Datum; Fundamentals/Makro/News bleiben aktuell). Default None =
            bisheriges Verhalten.
        journal: Wenn True (Default — bisheriges Verhalten), wird die finale
            Entscheidung via append_decision ins Entscheidungs-Journal
            (journal/decisions.csv) geschrieben. Wenn False, wird NUR der
            append_decision-Aufruf unterdrückt — Portfolio-Manager-Schritt,
            Usage-Recording und Checkpoint-Cleanup laufen normal weiter.
            Der Exit-Review-Modus (``--review``) nutzt journal=False, damit
            die Depot-Review-Läufe (Verkauf-Fragestellung) die Kalibrierung/
            den Track-Record der Neukauf-Analysen nicht verunreinigen.
        deep_think_model: Optionales stärkeres Modell für komplexe
            Reasoning-Agenten (Risiko-Debatte, Trade-Revision, Portfolio-
            Manager). None liest LLM_DEEP_THINK_MODEL aus der Env (leer =
            kein Split, primäres Modell).
        quick_think_model: Optionales schnelles Modell für schnelle Agenten
            (Analysten, Bull/Bear-Debatte, Trader). None liest
            LLM_QUICK_THINK_MODEL aus der Env (leer = kein Split, primäres
            Modell).
        reasoning_effort: Optionale Reasoning-Tiefe ('low'/'medium'/'high')
            für ALLE Agenten-Calls. None liest LLM_REASONING_EFFORT aus der
            Env (leer = deaktiviert, kein reasoning_effort im Payload —
            bisheriges Verhalten). Bewusst NICHT Teil des Konfigurations-
            Fingerprints: Env-Konfiguration, kein Pipeline-Parameter wie der
            Deep-/Quick-Think-Split.
        exit_mode: Phase 1 (dedizierter VERKAUFEN-Pfad): Wenn True, nutzt der
            Trader-Schritt (Schritt 4) ``trader_exit`` bzw.
            ``ensemble_trader(..., exit_mode=True)`` — der SYSTEM_TRADER_EXIT-
            Prompt stellt die VERKAUFEN-Frage ('Sollte ich diese Position
            verkaufen?') explizit statt sie implizit aus einer Neukauf-Analyse
            abzuleiten. Gedacht für den Exit-Review (``--review``), der die
            Bestands-Positionen des Depots prüft. Default False = bisheriges
            Verhalten (Neukauf-Analyse). Bewusst NICHT Teil des Konfigurations-
            Fingerprints: Bestehende Checkpoints (Neukauf- und Review-Läufe
            vor Phase 1) bleiben kompatibel.

    Resume-Kompatibilität:
        Ein Checkpoint wird nur wiederverwendet, wenn (a) das gepinnte
        Analysedatum (as_of) übereinstimmt UND (b) der beim Checkpoint-
        Schreiben persistierte Konfigurations-Fingerprint (_pipeline_fingerprint)
        dem des aktuellen Aufrufs entspricht. Der Fingerprint deckt ensemble,
        ensemble_runs, peers (sortiert normalisiert), debate_rounds und
        backtest ab. Checkpoints ohne Fingerprint (Altdaten) gelten als
        konfigurations-inkompatibel und werden ignoriert — die Pipeline
        startet dann von vorn.

    Returns:
        dict mit allen Zwischenergebnissen.
    """
    # --- Deep-Think/Quick-Think Modell-Split --------------------------------
    # None → aus der Env lesen (leer = kein Split → None = primäres Modell).
    # Explizit gesetzte Werte haben Vorrang (param > env), "" wird wie
    # "nicht gesetzt" behandelt.
    if deep_think_model is None:
        deep_think_model = config.llm_deep_think_model() or None
    if quick_think_model is None:
        quick_think_model = config.llm_quick_think_model() or None
    # --- Reasoning-Effort (Reasoning-Tiefe, analog zum Modell-Split) ---------
    # None → aus der Env lesen (leer/ungesetzt = deaktiviert → None). Explizit
    # gesetzte Werte haben Vorrang (param > env). Bewusst KEIN Fingerprint-
    # Bestandteil (Umgebungs-Konfiguration, kein Pipeline-Parameter) —
    # der Fingerprint bleibt unverändert.
    if reasoning_effort is None:
        reasoning_effort = config.llm_reasoning_effort() or None

    result: dict[str, Any] = {}

    # --- Konfigurations-Fingerprint (Roadmap C5) -------------------------------
    # Einmalig VOR allen Schritten berechnen und in result persistieren: Er
    # landet damit über _save_step → save_checkpoint automatisch in jedem
    # Checkpoint. as_of gehört bewusst NICHT hinein — es wird separat über
    # den bestehenden as_of-Check unten geprüft.
    fingerprint = _pipeline_fingerprint(
        ensemble=ensemble,
        ensemble_runs=ensemble_runs,
        peers=peers,
        debate_rounds=debate_rounds,
        backtest=backtest,
        deep_think_model=deep_think_model,
        quick_think_model=quick_think_model,
    )

    # --- Resume: Checkpoint laden, falls vorhanden und gewünscht ---
    # Bei gepinntem Analysedatum (as_of) darf ein Checkpoint nur wiederverwendet
    # werden, wenn er mit demselben as_of erzeugt wurde — sonst sind Historie,
    # Indikatoren und Kurs inkonsistent zum gepinnten Datum.
    # Zusätzlich (C5): Der Checkpoint darf nur bei identischer Pipeline-
    # Konfiguration fortgeschrieben werden — Zwischenergebnisse aus einer
    # anderen Konfiguration (z. B. anderes ensemble_runs, andere peers) wären
    # inkonsistent zum aktuellen Aufruf. Checkpoints ohne Fingerprint
    # (Altdaten) gelten ebenfalls als inkompatibel (starte von vorn).
    if resume:
        cp = load_checkpoint(ticker)
        if cp is not None and (cp.get("data") or {}).get("as_of") != as_of:
            logger.info(
                "Resume-Checkpoint ignoriert (anderes Analysedatum: "
                "Checkpoint as_of=%s, angefordert as_of=%s) — starte von vorn.",
                (cp.get("data") or {}).get("as_of"),
                as_of,
            )
            cp = None
        if cp is not None and cp.get(_FINGERPRINT_KEY) != fingerprint:
            logger.info(
                "Resume-Checkpoint ignoriert (Konfiguration geändert: "
                "Checkpoint-Fingerprint=%s, angefordert=%s) — starte von vorn.",
                cp.get(_FINGERPRINT_KEY) or "(keiner — Altdaten)",
                fingerprint,
            )
            cp = None
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

    # Fingerprint in result persistieren — wird über _save_step mit jedem
    # Checkpoint geschrieben (C5). Nach dem Resume-Block gesetzt, damit der
    # geladene Checkpoint (falls akzeptiert) seinen gespeicherten Fingerprint
    # nicht verliert bzw. der frische Lauf ihn gleich korrekt trägt.
    result[_FINGERPRINT_KEY] = fingerprint

    # --- 1. Daten sammeln ---
    if not _is_completed(result, "data"):
        logger.info("Schritt 1: Sammle Marktdaten für %s", ticker)
        data = collect_ticker_data(ticker, peers=peers, as_of=as_of)
        result["data"] = data
        result["ticker"] = data["ticker"]

        # --- 1a. Token-Usage-Zähler pro Analyse zurücksetzen ---
        # total_usage akkumuliert über alle LLM-Calls EINER Analyse. Bei Batch-
        # Läufen (--tickers/--watchlist) wird run_pipeline pro Ticker aufgerufen;
        # ohne Reset würde der Zähler über den ganzen Batch kumulieren statt pro
        # Ticker. Nur im LLM-Modus relevant.
        if llm is not None:
            llm.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # --- 1b. data_text einmal berechnen (für alle Agenten-Prompts) ---
        data_text = _build_data_text(data) if llm is not None else None
        result["_data_text"] = data_text

        # --- 1c. Feedback-Kontext einmal berechnen (Track-Record-Historie) ---
        feedback_context = build_feedback_context() if llm is not None else ""
        result["_feedback_context"] = feedback_context

        # --- 1d. Reflexions-Kontext (realisierter Return der letzten Entscheidung) ---
        # Roadmap C4 (Cross-Ticker-Gedächtnis): Die Agenten-Prompts erhalten
        # zusätzlich zur Same-Ticker-Reflexion die jüngsten Entscheidungen
        # ANDERER Ticker mit realisiertem Return (build_cross_ticker_context).
        #
        # Aufteilung bewusst so:
        #   - result["reflection"]  → NUR Ticker-spezifische Reflexion (Report-
        #     Abschnitt "Reflexion (Track-Record)" bleibt kompakt und zeigt
        #     keine Cross-Ticker-Lektionen).
        #   - result["_reflection_context"] → kombiniert (Same + Cross-Ticker),
        #     wird an trader/ensemble_trader/risk_manager/portfolio_manager
        #     durchgereicht (dort als reflection_context an den Prompt angehängt).
        #   - result["_cross_ticker_context"] → nur der Cross-Ticker-Block
        #     (Debug-/Test-Zugriff, vom Journal/Report nicht ausgewertet —
        #     Underscore-Präfix wie _feedback_context/_data_text).
        #
        # Rückwärtskompatibilität mit bestehenden Tests: Pipeline-Tests patchen
        # concilium.pipeline.build_reflection_context. Ist die Funktion im
        # Pipeline-Namespace durch etwas anderes ersetzt (MagicMock oder eigene
        # Funktion), gilt sie als SOLE-Provider des Reflexions-Kontexts und der
        # Cross-Ticker-Teil wird NICHT ergänzt — Verhalten exakt wie vor C4.
        # resolve_pending_reflections (C6) läuft analog zu append_decision immer
        # im Normal-Modus (journal=True) — in gemockten Pipeline-Tests liefert
        # es "" (kein passender/auflösbarer Journal-Eintrag) und stört nicht;
        # isolierende Tests patchen concilium.pipeline.resolve_pending_reflections.
        reflection_context = ""
        cross_ticker_context = ""
        same_ticker_context = ""
        resolved_reflection = ""
        if llm is not None and journal:
            # C6: Zuerst einen evtl. auflösbaren Pending-Eintrag resolven
            # (Ausgangsfenster vollständig abgelaufen) — das persistiert
            # Return + Lektion im Journal und liefert den Reflexions-Text.
            # Nur im Normal-Modus (journal=True); --review schreibt kein
            # Journal und soll auch keine Journaleinträge auflösen/ändern.
            resolved_reflection = resolve_pending_reflections(ticker=ticker, llm=llm)
            same_ticker_context = build_reflection_context(ticker=ticker, llm=llm)
            reflection_context = same_ticker_context
            if getattr(
                build_reflection_context, "__module__", None
            ) == _feedback_module.__name__:
                # Ungepatcht → Cross-Ticker-Lektionen (C4) ergänzen.
                cross_ticker_context = build_cross_ticker_context(
                    ticker=ticker, llm=llm
                )
                if cross_ticker_context:
                    reflection_context = (
                        f"{same_ticker_context}\n\n{cross_ticker_context}"
                        if same_ticker_context
                        else cross_ticker_context
                    )
            # C6: Die frisch aufgelöste Reflexion fließt in den kombinierten
            # Kontext ein, falls der Same-Ticker-/Cross-Ticker-Teil leer ist
            # (typisch beim ersten Lauf nach Ablauf des Fensters: pending →
            # resolved, aber build_reflection_context liefert erst beim
            # NÄCHSTEN Lauf denselben Ticker wieder eine Reflexion).
            if resolved_reflection and not reflection_context:
                reflection_context = resolved_reflection
        result["reflection"] = same_ticker_context if llm is not None else None
        result["_reflection_context"] = reflection_context
        result["_cross_ticker_context"] = cross_ticker_context
        result["_resolved_reflection"] = resolved_reflection

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
        analysts = analyst_team(
            data, llm, model=quick_think_model, reasoning_effort=reasoning_effort,
        )  # data_text=None → rollenspezifische Filter greifen
        result["analysts"] = analysts
        _save_step(result, ticker, "analysts")
    else:
        analysts = result["analysts"]

    # --- 2a. Invalidierungen aggregieren (Stufe 1) --------------------------
    # Die pro-Analyst-Invalidierungs-Bedingungen werden deterministisch zu
    # einem kompakten String verbunden und im Result persistiert — append_
    # decision (und append_review_decision) schreiben ihn in die CSV-Spalte
    # 'invalidation'. Nie-crashend: fehlt 'analysts' (z. B. gemockter Lauf),
    # bleibt result["invalidation"] = "".
    result["invalidation"] = _aggregate_invalidation(analysts)

    # --- 3. Debatte ---
    if not _is_completed(result, "debate"):
        logger.info("Schritt 3: Bull/Bear-Debatte")
        debate_result = debate(
            analysts, llm, rounds=debate_rounds, model=quick_think_model,
            reasoning_effort=reasoning_effort,
        )
        result["debate"] = debate_result
        _save_step(result, ticker, "debate")
    else:
        debate_result = result["debate"]

    # --- 4. Trader (oder Ensemble-Trader) ---
    # Phase 1 (exit_mode): Der Exit-Review nutzt trader_exit /
    # ensemble_trader(exit_mode=True) — der SYSTEM_TRADER_EXIT-Prompt stellt
    # die VERKAUFEN-Frage explizit ('Sollte ich diese Position verkaufen?')
    # statt sie implizit aus einer Neukauf-Analyse abzuleiten.
    if not _is_completed(result, "trade"):
        if exit_mode:
            logger.info(
                "Schritt 4: Exit-Trader prüft BESTEHENDE Position (Exit-Frage)"
            )
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
                model=quick_think_model,
                reasoning_effort=reasoning_effort,
                exit_mode=exit_mode,
            )
        elif exit_mode:
            logger.info("Schritt 4: Exit-Trader erstellt Trade-Vorschlag (Single-Run)")
            trade = trader_exit(
                analysts,
                debate_result,
                llm,
                feedback_context=feedback_context,
                reflection_context=reflection_context,
                model=quick_think_model,
                reasoning_effort=reasoning_effort,
            )
        else:
            logger.info("Schritt 4: Trader erstellt Trade-Vorschlag (Single-Run)")
            trade = trader(
                analysts,
                debate_result,
                llm,
                feedback_context=feedback_context,
                reflection_context=reflection_context,
                model=quick_think_model,
                reasoning_effort=reasoning_effort,
            )
        result["trade"] = trade
        _save_step(result, ticker, "trade")
    else:
        trade = result["trade"]

    # --- 5. Risk-Manager ---
    if not _is_completed(result, "risk"):
        logger.info("Schritt 5: Risk-Manager bewertet Risiko")
        risk = risk_manager(
            trade, data, llm, data_text=data_text, feedback_context=feedback_context,
            model=deep_think_model, reasoning_effort=reasoning_effort,
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
            portfolio_fit = portfolio_fit_agent(
                data, llm, positions, data_text=data_text,
                reasoning_effort=reasoning_effort,
            )
            result["portfolio_fit"] = portfolio_fit
        except Exception as exc:  # noqa: BLE001 — nie crashen
            logger.warning("Portfolio-Fit fehlgeschlagen: %s", exc)
            result["portfolio_fit"] = None
        _save_step(result, ticker, "portfolio_fit")

    # --- 5c. Trade-Revision (2nd Pass) --- #
    if not _is_completed(result, "trade_revision"):
        result["trade_original"] = None
        result["trade_revised"] = False
        try:
            logger.info("Schritt 5c: Trade-Revision (2nd Pass)")
            original_trade = trade
            # current_price aus Analysten-Daten extrahieren (für Ziel-/Stop-Fallback)
            rev_current_price = _extract_current_price(analysts)
            revised = trade_revision(
                original_trade,
                risk,
                result.get("portfolio_fit"),
                llm,
                feedback_context=feedback_context,
                reflection_context=reflection_context,
                current_price=rev_current_price,
                model=deep_think_model,
                reasoning_effort=reasoning_effort,
            )
            result["trade_original"] = original_trade
            result["trade"] = revised
            result["trade_revised"] = True
            trade = revised
        except Exception as exc:  # noqa: BLE001 — nie crashen
            logger.warning("Trade-Revision fehlgeschlagen: %s", exc)
        _save_step(result, ticker, "trade_revision")

    # --- 5c'. Technik-Signal NACH der Trade-Revision erneut anwenden ---
    # Die Trade-Revision (LLM) darf das Technik-Signal (Kurs unter SMA200)
    # nicht umgehen: Liefert sie wieder KAUFEN, obwohl das Signal aktiv ist,
    # wird die Positionsgröße deterministisch mit dem Faktor skaliert (KAUFEN
    # bleibt KAUFEN — kein binäres Blocken mehr; bei der RSI-Ausnahme wird der
    # Stop nachjustiert). Der Signal-Check ist idempotent — eine bereits
    # gespeicherte Basis-Positionsgröße wird wiederverwendet, anstatt
    # kumulativ zu skalieren; HALTEN-Trades werden nie angetastet.
    try:
        # Basis-Idempotenz: trade_revision baut ein frisches dict ohne die
        # Metadaten des Original-Trades (nur Schema-Keys). Die beim ersten
        # Apply gespeicherte Basis-Positionsgröße wird daher VOR dem erneuten
        # Signal-Apply aus dem Original-Trade übernommen — sonst skaliert das
        # Signal zweimal (quadratisch).
        if isinstance(trade, dict) and trade.get("_technik_signal_basis") is None:
            _orig = result.get("trade_original")
            if isinstance(_orig, dict):
                _b = _orig.get("_technik_signal_basis")
                if _b is not None:
                    trade["_technik_signal_basis"] = _b
        _apply_technik_signal(trade, analysts)
        # trade kann in-place mutiert worden sein (dict-Referenz) — sicherheitshalber
        # zurückschreiben, falls die Revision ein neues dict eingesetzt hat.
        result["trade"] = trade
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Technik-Signal (nach Revision) fehlgeschlagen: %s", exc)

    # --- 5c''. Volatility-Targeting als harte Obergrenze ---
    # NACH dem Technik-Signal (5c'): Das Signal skaliert die Positionsgröße
    # zuerst (Faktor), dann wird der rechnerische Wert aus dem Risikomodell
    # (positionsgröße_rechnerisch_pct) als harte Obergrenze angewendet —
    # Volatility-Targeting als harte Obergrenze — der LLM darf nicht mehr
    # Position empfehlen, als das rechnerische Risikomodell erlaubt.
    # Der Vol-Cap senkt nur, hebt aber nie an: Hat das Technik-Signal die
    # Position bereits unter den Cap gesenkt, greift er nicht. Ohne
    # rechnerische Positionsgröße (keine Historie/Volatilität) greift kein
    # Cap — Verhalten wie bisher (rückwärtskompatibel). Nur KAUFEN/
    # STARK KAUFEN wird gekappt; HALTEN/VERKAUFEN bleiben unangetastet.
    try:
        _cap_position_by_volatility(trade, risk)
        result["trade"] = trade
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Vol-Cap (nach Technik-Signal) fehlgeschlagen: %s", exc)

    # --- 5b'. Kalibrierungs-gestützte Dämpfung der Ziel-Gewichtung ---
    # Wird NACH Schritt 5c (Trade-Revision) ausgeführt: Die Dämpfung basiert
    # auf der FINALEN (revidierten) Trade-Aktion — die Trade-Revision kann die
    # Aktion ändern (z. B. KAUFEN→HALTEN), und die Ziel-Gewichtung muss zur
    # Aktion konsistent sein, die Journal + PM schließlich sehen.
    # Der Track-Record zeigt, dass das System überkonfident ist (Konfidenz 4-5,
    # aber ~29-34% Hit-Rate). Die LLM-empfohlene Ziel-Gewichtung wird daher
    # deterministisch mit der historischen Trefferquote der Aktion skaliert.
    # Original wird in ziel_gewichtung_original erhalten; die Dämpfung darf
    # NIEMALS crashen (bei Exception läuft die Pipeline unverändert weiter).
    # Resume-Idempotenz: Bei Resume mit bereits abgeschlossenem trade_revision
    # ist `trade` der revidierte Trade aus dem Checkpoint — der Block läuft
    # dann trotzdem (er steht bewusst NICHT unter dem trade_revision-Guard),
    # darf aber einen bereits gedämpften Wert nicht erneut dämpfen (sonst
    # würde z. B. 10.0 → 5.2 → 2.7 doppelt skaliert).
    pf_for_dampen = result.get("portfolio_fit")
    if isinstance(pf_for_dampen, dict):
        bereits_gedämpft = bool(pf_for_dampen.get("ziel_gewichtung_gedämpft"))
        try:
            aktion = (trade or {}).get("aktion")
            if aktion and not bereits_gedämpft:
                pf_for_dampen["ziel_gewichtung_original"] = pf_for_dampen.get(
                    "ziel_gewichtung_pct"
                )
                ziel = pf_for_dampen.get("ziel_gewichtung_pct")
                if isinstance(ziel, int | float) and not isinstance(ziel, bool):
                    gedämpft = _dampen_ziel_gewichtung(float(ziel), str(aktion))
                    if gedämpft is not None:
                        pf_for_dampen["ziel_gewichtung_pct"] = gedämpft
                        pf_for_dampen["ziel_gewichtung_gedämpft"] = True
                        logger.info(
                            "Ziel-Gewichtung kalibrierungs-gedämpft (%s): "
                            "%.1f → %.1f",
                            aktion,
                            float(ziel),
                            gedämpft,
                        )
        except Exception as exc:  # noqa: BLE001 — nie crashen
            logger.warning(
                "Dämpfung der Ziel-Gewichtung fehlgeschlagen: %s", exc
            )

    # --- 6. Portfolio-Manager ---
    if skip_final:
        # Im Portfolio-Modus wird der PM zurückgehalten bis der Portfolio-Kontext
        # berechnet ist. Nur ein Marker wird gesetzt; final bleibt None.
        logger.info("Schritt 6 übersprungen (skip_final=True) — PM pending")
        result["final"] = None
        result["_final_pending"] = True
        # "final" wird NICHT in _completed_steps eingetragen.
    elif not _is_completed(result, "final"):
        logger.info("Schritt 6: Portfolio-Manager trifft finale Entscheidung")
        final = portfolio_manager(
            trade,
            risk,
            llm,
            portfolio_fit=result.get("portfolio_fit"),
            feedback_context=feedback_context,
            reflection_context=reflection_context,
            portfolio_context=portfolio_context,
            model=deep_think_model,
            reasoning_effort=reasoning_effort,
        )
        result["final"] = final
        _save_step(result, ticker, "final")

    # --- Final-Guard (Punkt 4): harte Obergrenze für die Ziel-Gewichtung ---
    # Bewusst NACH dem Portfolio-Manager (Schritt 6): Der PM kann in
    # MODIFIZIERT-Auflagen die Ziel-Gewichtung anpassen — der Guard ist die
    # LETZTE deterministische Instanz und kappt jeden LLM-Wert (auch ein
    # PM-Re-Weighting) am harten Maximum (CONCILIUM_MAX_POSITION_PCT,
    # Default 15.0 %). Auch im skip_final-Modus (Portfolio-Phase 1) läuft
    # er gegen den vorhandenen portfolio_fit — bei erneutem Apply (Phase 2
    # / Resume) senkt der Cap nur und schreibt keine Provenance um, also
    # idempotent. Crasht nie (bei Fehlern unverändert weiter).
    try:
        pf_guard = result.get("portfolio_fit")
        if isinstance(pf_guard, dict):
            _apply_final_position_guard(pf_guard, result.get("trade"))
            result["portfolio_fit"] = pf_guard
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Final-Guard fehlgeschlagen: %s", exc)

    # --- Feature 4: Entscheidungs-Journal ---
    # Nur im LLM-Modus (llm nicht None), wenn final existiert, NICHT
    # im skip_final-Modus (dort wird das Journal später von run_portfolio
    # mit dem Portfolio-Kontext-final geschrieben) und nur wenn journal=True.
    # journal=False (z. B. Exit-Review) unterdrückt ausschließlich das
    # append_decision — PM, Usage-Recording und Checkpoint-Cleanup laufen
    # unverändert weiter.
    if not skip_final and journal:
        try:
            from .journal import append_decision

            append_decision(result)
            result["_journal_written"] = True
        except Exception as exc:  # noqa: BLE001 — nie crashen
            logger.warning("Entscheidung konnte nicht ins Journal geschrieben werden: %s", exc)
            result["_journal_written"] = False

    # --- Feature 4: Token-Usage-Logging ---
    # Nur im LLM-Modus: kumulativen Token-Verbrauch der gesamten Analyse
    # in usage/usage.csv protokollieren. Crasht nie und beeinflusst die
    # Pipeline nicht.
    if llm is not None:
        try:
            from .usage import record_usage

            record_usage(ticker, llm.total_usage)
        except Exception as exc:  # noqa: BLE001 — nie crashen
            logger.warning("Usage-Recording fehlgeschlagen: %s", exc)

    # --- Erfolgreicher Lauf: Checkpoint aufräumen ---
    # Im skip_final-Modus wird der Checkpoint NICHT aufgeräumt, da der
    # PM-Schritt noch aussteht (run_portfolio übernimmt die Endabwicklung).
    if not skip_final:
        clear_checkpoint(ticker)

    return result


# ---------------------------------------------------------------------------
# Portfolio-Modus: mehrere Ticker als Ganzheit analysieren
# ---------------------------------------------------------------------------


def run_portfolio(
    tickers: list[str],
    llm: LLMClient | None = None,
    backtest: bool = False,
    ensemble: bool = True,
    ensemble_runs: int = 3,
    resume: bool = False,
    peers: list[str] | None = None,
    debate_rounds: int = 1,
    as_of: str | None = None,
    deep_think_model: str | None = None,
    quick_think_model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Portfolio-Modus: analysiert mehrere Ticker als Depot-Ganzheit.

    Führt für jeden Ticker die Einzel-Pipeline aus (allerdings OHNE den
    finalen Portfolio-Manager-Schritt), berechnet dann die Portfolio-Analyse
    (Korrelation, Overlap, Konzentration) über alle History-Daten + Bestand,
    und ruft den PM erst dann — EINMAL pro Ticker — mit dem Gesamt-Exposure-
    Kontext auf.

    Genauer Ablauf:
      1. Phase 1: Für jeden Ticker run_pipeline mit ``skip_final=True`` —
         die Vor-Schritte (data, analysts, debate, trade, risk, portfolio_fit,
         trade_revision) laufen, aber der PM wird zurückgehalten.
      2. Portfolio-Analyse über alle Ergebnisse berechnen (Korrelation,
         Overlap, Konzentration).
      3. Phase 2: Für jeden Ticker wird der PM EINMAL aufgerufen, diesmal
         MIT portfolio_context (Gesamt-Exposure). Erst danach wird das
         Journal geschrieben — konsistent mit der angezeigten Entscheidung.

    Wenn llm=None (--no-llm), werden nur Datensnapshots gesammelt und die
    Portfolio-Analyse deterministisch berechnet (kein PM).

    Args:
        tickers: Liste der zu analysierenden Ticker-Symbole.
        llm: LLMClient oder None für --no-llm Modus.
        backtest: Ob Backtest-Signalproxy ausgeführt werden soll.
        ensemble: Ob der Trader als Ensemble ausgeführt wird.
        ensemble_runs: Anzahl der Ensemble-Runs.
        resume: Resume-Modus für Einzel-Pipelines.
        as_of: Optionales gepinntes Analysedatum (YYYY-MM-DD) — wird an jede
            Einzel-Pipeline (run_pipeline) durchgereicht.
        deep_think_model: Optionales Deep-Think-Modell — wird an jede
            Einzel-Pipeline UND den Phase-2-PM-Call durchgereicht. None liest
            LLM_DEEP_THINK_MODEL aus der Env (leer = kein Split).
        quick_think_model: Optionales Quick-Think-Modell — wird an jede
            Einzel-Pipeline durchgereicht. None liest LLM_QUICK_THINK_MODEL
            aus der Env (leer = kein Split).
        reasoning_effort: Optionale Reasoning-Tiefe ('low'/'medium'/'high')
            für alle Agenten-Calls — wird an jede Einzel-Pipeline UND den
            Phase-2-PM-Call durchgereicht. None liest LLM_REASONING_EFFORT
            aus der Env (leer = deaktiviert, bisheriges Verhalten).

    Returns:
        dict mit:
          - results: {ticker: pipeline_result} (alle Ticker)
          - portfolio_analysis: Ergebnis von run_portfolio_analysis()
          - tickers: Liste der analysierten Ticker
    """
    from .portfolio_analysis import run_portfolio_analysis

    # --- Deep-Think/Quick-Think Split auflösen (param > env, leer = None) ---
    # run_portfolio ruft run_pipeline rekursiv auf — dort würde None erneut
    # aus der Env gelesen; hier lösen wir trotzdem EINMAL zentral, damit der
    # Phase-2-PM-Call (direkter portfolio_manager-Aufruf!) dasselbe Deep-Think-
    # Modell bekommt wie die Einzel-Pipelines.
    if deep_think_model is None:
        deep_think_model = config.llm_deep_think_model() or None
    if quick_think_model is None:
        quick_think_model = config.llm_quick_think_model() or None
    # Reasoning-Effort ebenfalls EINMAL zentral auflösen (param > env), damit
    # der Phase-2-PM-Call denselben Effort bekommt wie die Einzel-Pipelines.
    if reasoning_effort is None:
        reasoning_effort = config.llm_reasoning_effort() or None

    # --- Phase 1: Einzel-Pipelines für jeden Ticker (ohne PM) ---
    # skip_final=True hält den PM+Journal zurück, bis der Portfolio-Kontext
    # berechnet ist. So läuft der PM nur EINMAL (mit Kontext) pro Ticker.
    results: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        logger.info("Portfolio-Modus Phase 1: Analyse %s", ticker)
        try:
            result = run_pipeline(
                ticker,
                llm=llm,
                backtest=backtest,
                peers=peers,
                ensemble=ensemble,
                ensemble_runs=ensemble_runs,
                resume=resume,
                portfolio_context=None,
                skip_final=llm is not None,
                debate_rounds=debate_rounds,
                as_of=as_of,
                deep_think_model=deep_think_model,
                quick_think_model=quick_think_model,
                reasoning_effort=reasoning_effort,
            )
            results[ticker] = result
        except Exception as exc:  # noqa: BLE001 — nie crashen
            logger.warning("Ticker '%s' fehlgeschlagen im Portfolio-Modus: %s", ticker, exc)
            results[ticker] = {
                "ticker": ticker,
                "error": str(exc),
                "data": {},
                "no_llm": True,
            }

    # --- Portfolio-Analyse berechnen (deterministisch) ---
    positions: list[dict[str, Any]] = []
    try:
        positions = fetch_portfolio_positions()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Portfolio-Positionen konnten nicht geladen werden: %s", exc)

    portfolio_analysis = run_portfolio_analysis(results, positions)

    # --- Phase 2: PM mit Portfolio-Kontext (nur im LLM-Modus) ---
    # Der PM wird jetzt EINMAL pro Ticker aufgerufen — mit Portfolio-Kontext.
    # Erst DANACH wird das Journal geschrieben (konsistent mit angezeigtem final).
    if llm is not None:
        from .journal import append_decision

        for ticker in tickers:
            result = results.get(ticker, {})
            if result.get("error"):
                continue
            # PM nur aufrufen, wenn Vor-Schritte erfolgreich waren
            trade = result.get("trade")
            risk = result.get("risk")
            if not trade or not risk:
                continue

            logger.info("Portfolio-Modus Phase 2: PM mit Kontext für %s", ticker)
            try:
                feedback_context = result.get("_feedback_context", "")
                reflection_context = result.get("_reflection_context", "")

                final = portfolio_manager(
                    trade,
                    risk,
                    llm,
                    portfolio_fit=result.get("portfolio_fit"),
                    feedback_context=feedback_context,
                    reflection_context=reflection_context,
                    portfolio_context=portfolio_analysis,
                    model=deep_think_model,
                    reasoning_effort=reasoning_effort,
                )
                result["final"] = final
                result["portfolio_context"] = portfolio_analysis
                result["_final_pending"] = False

                # Final-Guard NACH dem PM auch im Portfolio-Modus: Ein
                # PM-MODIFIZIERT-Re-Weighting (hochgesetzte Ziel-Gewichtung)
                # wird erneut am harten Maximum gekappt. Der Guard ist
                # re-idempotent gegen nachträgliches Anheben (Docstring oben),
                # senkt nur, crasht nie.
                try:
                    _apply_final_position_guard(
                        result.get("portfolio_fit"), result.get("trade")
                    )
                except Exception as exc:  # noqa: BLE001 — nie crashen
                    logger.warning(
                        "Final-Guard im Portfolio-Modus für '%s' fehlgeschlagen: %s",
                        ticker,
                        exc,
                    )

                # Journal EINMAL schreiben — mit dem final MIT Portfolio-Kontext
                try:
                    append_decision(result)
                    result["_journal_written"] = True
                except Exception as exc:  # noqa: BLE001 — nie crashen
                    logger.warning(
                        "Journal für '%s' konnte nicht geschrieben werden: %s",
                        ticker,
                        exc,
                    )
                    result["_journal_written"] = False

                # Checkpoint aufräumen — PM ist jetzt abgeschlossen
                clear_checkpoint(ticker)
            except Exception as exc:  # noqa: BLE001 — nie crashen
                logger.warning(
                    "PM-Lauf für '%s' fehlgeschlagen: %s", ticker, exc
                )

    return {
        "results": results,
        "portfolio_analysis": portfolio_analysis,
        "tickers": tickers,
    }
