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
        "--tickers",
        default=None,
        metavar="TICKER1,TICKER2,…",
        help="Kommagetrennte Ticker-Liste für Batch-Modus (z. B. AAPL,NVDA,MSFT). "
        "Führt mehrere Analysen hintereinander aus. "
        "Schließt sich mit --ticker gegenseitig aus.",
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
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="Setzt einen abgebrochenen Lauf fort (Checkpoint unter state/ wird geladen, "
        "nur fehlende Schritte werden neu ausgeführt).",
    )
    resume_group.add_argument(
        "--no-resume",
        action="store_true",
        help="Explizit kein Resume (Default-Verhalten). Schließt sich mit --resume aus.",
    )
    args = parser.parse_args(argv)

    # --- Frühe Validierung: Kombinations-Verbote ---
    # --evaluate + --ticker/--tickers → Fehler (vor jeglicher Ausführung)
    if args.evaluate is not None and (args.ticker or args.tickers):
        print(
            "FEHLER: --evaluate kann nicht mit --ticker oder --tickers kombiniert werden.",
            file=sys.stderr,
        )
        return 1

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

    # --- Pipeline-Modus: --ticker oder --tickers required ---
    # Mutual exclusion: --ticker und --tickers
    if args.ticker and args.tickers:
        parser.error("--ticker und --tickers schließen sich gegenseitig aus.")

    if not args.ticker and not args.tickers:
        parser.error("--ticker oder --tickers ist erforderlich, wenn --evaluate nicht gesetzt ist.")

    # Logging konfigurieren
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # LLM-Client erstellen (oder None für --no-llm)
    llm = None if args.no_llm else LLMClient()

    # Peers-Liste parsen (kommagetrennt)
    peers_list: list[str] | None = None
    if args.peers:
        peers_list = [p.strip() for p in args.peers.split(",") if p.strip()]

    # Reports-Verzeichnis
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # --- Batch-Modus (--tickers) ---
    if args.tickers:
        ticker_list = [t.strip() for t in args.tickers.split(",") if t.strip()]
        if not ticker_list:
            parser.error("--tickers darf nicht leer sein.")

        failures = 0
        successes = 0

        for i, ticker in enumerate(ticker_list):
            if i > 0:
                print("\n" + "=" * 70 + "\n", file=sys.stderr)

            try:
                result = run_pipeline(
                    ticker,
                    llm=llm,
                    backtest=args.backtest,
                    peers=peers_list,
                    ensemble=not args.no_ensemble,
                    ensemble_runs=args.ensemble_runs,
                    resume=args.resume,
                )
                report = generate_report(result, reports_dir=reports_dir)
                print(report)

                # Report-Datei speichern
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                resolved_ticker = result.get("ticker", ticker.upper())
                filename = f"{resolved_ticker}_{timestamp}.md"
                filepath = os.path.join(reports_dir, filename)
                with open(filepath, "w", encoding="utf-8") as fh:
                    fh.write(report)
                print(f"\n---\nReport gespeichert: {filepath}", file=sys.stderr)
                successes += 1

            except KeyboardInterrupt:
                print(
                    f"\nABGEBROCHEN (Ticker '{ticker}') — "
                    "Checkpoint bleibt unter state/ erhalten.",
                    file=sys.stderr,
                )
                return 130
            except ValueError as exc:
                print(f"FEHLER bei Ticker '{ticker}': {exc}", file=sys.stderr)
                logging.warning("Ticker '%s' fehlgeschlagen: %s", ticker, exc)
                failures += 1
            except Exception as exc:  # noqa: BLE001
                print(f"UNERWARTETER FEHLER bei Ticker '{ticker}': {exc}", file=sys.stderr)
                logging.exception("Unerwarteter Fehler bei Ticker '%s'", ticker)
                failures += 1

        if failures > 0:
            print(
                f"\nWARNUNG: {failures} von {len(ticker_list)} Tickern fehlgeschlagen.",
                file=sys.stderr,
            )

        return 0 if successes > 0 else 1

    # --- Einzelmodus (--ticker) ---
    try:
        # Pipeline ausführen
        result = run_pipeline(
            args.ticker,
            llm=llm,
            backtest=args.backtest,
            peers=peers_list,
            ensemble=not args.no_ensemble,
            ensemble_runs=args.ensemble_runs,
            resume=args.resume,
        )

        # Report generieren
        report = generate_report(result, reports_dir=reports_dir)

        # Auf stdout ausgeben
        print(report)

        # Als Datei speichern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        ticker = result.get("ticker", args.ticker.upper())
        filename = f"{ticker}_{timestamp}.md"
        filepath = os.path.join(reports_dir, filename)
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(report)

        print(f"\n---\nReport gespeichert: {filepath}", file=sys.stderr)
        return 0

    except KeyboardInterrupt:
        print(
            "\nABGEBROCHEN — Checkpoint bleibt unter state/ erhalten.",
            file=sys.stderr,
        )
        return 130
    except ValueError as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"UNERWARTETER FEHLER: {exc}", file=sys.stderr)
        logging.exception("Unerwarteter Fehler")
        return 1


if __name__ == "__main__":
    sys.exit(main())
