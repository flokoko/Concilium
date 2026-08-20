"""CLI-Modul — TradingAgents-Light Befehlszeilen-Schnittstelle."""

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
        description="TradingAgents-Light — Trading-Entscheidungs-Pipeline"
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
        "--verbose",
        action="store_true",
        help="Ausführliche Logging-Ausgabe",
    )
    args = parser.parse_args(argv)

    # Logging konfigurieren
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # LLM-Client erstellen (oder None für --no-llm)
    llm = None if args.no_llm else LLMClient()

    try:
        # Pipeline ausführen
        result = run_pipeline(args.ticker, llm=llm, backtest=args.backtest)

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
