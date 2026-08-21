"""CLI-Modul — Concilium Befehlszeilen-Schnittstelle."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

from .evaluate import evaluate_journal
from .llm import LLMClient
from .pipeline import run_pipeline
from .report import generate_report, generate_track_record_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Concilium — Multi-Agenten-Fonds-Entscheidungssystem"
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Ticker-Symbol, ISIN oder WKN (z. B. AAPL, DE000BASF111, 716460)",
    )
    parser.add_argument(
        "--evaluate",
        nargs="?",
        const="journal/decisions.csv",
        default=None,
        metavar="JOURNAL_DATEI",
        help="Track-Record-Evaluierung ausführen (optionaler Pfad zur Journal-CSV). "
        "Standard: journal/decisions.csv. Führt NICHT die Pipeline aus.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=90,
        help="Lookback-Tage für Track-Record-Evaluierung (Default: 90).",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Führe Backtest-Signalproxy aus (SMA50/200 + RSI)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Nur Datensnapshot + Report ohne LLM-Agenten (kein API-Aufruf)",
    )
    parser.add_argument(
        "--peers",
        type=str,
        default=None,
        help="Kommagetrennte Peer-Ticker für Vergleich (z. B. RWE.DE,SHEL.L)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ausführliche Logging-Ausgabe",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Ensemble-Trader deaktivieren (nur Single-Run). Standard: Ensemble aktiv (3 Runs).",
    )
    parser.add_argument(
        "--ensemble-runs",
        type=int,
        default=3,
        help="Anzahl der Ensemble-Runs für den Trader (Default: 3). "
        "Wird ignoriert, wenn --no-ensemble gesetzt ist.",
    )
    args = parser.parse_args(argv)

    # --evaluate ist eigenständig: Pipeline wird nicht ausgeführt
    if args.evaluate is not None:
        level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        llm = None if args.no_llm else LLMClient()
        try:
            eval_result = evaluate_journal(
                args.evaluate,
                lookback_days=args.lookback,
                llm=llm,
            )
            report = generate_track_record_report(eval_result)
            print(report)

            # Report als Datei speichern
            reports_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports"
            )
            os.makedirs(reports_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y%m%d")
            filepath = os.path.join(reports_dir, f"track_record_{date_str}.md")
            with open(filepath, "w", encoding="utf-8") as fh:
                fh.write(report)
            print(f"\n---\nTrack-Record-Report gespeichert: {filepath}", file=sys.stderr)
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"FEHLER bei Track-Record-Evaluierung: {exc}", file=sys.stderr)
            logging.exception("Track-Record-Fehler")
            return 1

    # --- Pipeline-Modus: --ticker required ---
    if not args.ticker:
        parser.error("--ticker ist erforderlich, wenn --evaluate nicht gesetzt ist.")

    # Logging konfigurieren
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # LLM-Client erstellen (oder None für --no-llm)
    llm = None if args.no_llm else LLMClient()

    # Peers-Liste parsen (kommagetrennt)
    peers_list: list[str] | None = None
    if args.peers:
        peers_list = [p.strip() for p in args.peers.split(",") if p.strip()]

    try:
        # Pipeline ausführen
        result = run_pipeline(
            args.ticker,
            llm=llm,
            backtest=args.backtest,
            peers=peers_list,
            ensemble=not args.no_ensemble,
            ensemble_runs=args.ensemble_runs,
        )

        # Report generieren
        report = generate_report(result)

        # Auf stdout ausgeben
        print(report)

        # Als Datei speichern
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        ticker = result.get("ticker", args.ticker.upper())
        filename = f"{ticker}_{timestamp}.md"
        filepath = os.path.join(reports_dir, filename)
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(report)

        print(f"\n---\nReport gespeichert: {filepath}", file=sys.stderr)
        return 0

    except ValueError as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"UNERWARTETER FEHLER: {exc}", file=sys.stderr)
        logging.exception("Unerwarteter Fehler")
        return 1


if __name__ == "__main__":
    sys.exit(main())
