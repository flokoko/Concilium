"""CLI-Modul — Concilium Befehlszeilen-Schnittstelle."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

from .llm import LLMClient
from .pipeline import run_pipeline
from .report import generate_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Concilium — Multi-Agenten-Fonds-Entscheidungssystem"
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker-Symbol (z. B. AAPL, MSFT, NVDA)",
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
