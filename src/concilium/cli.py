"""CLI-Modul — Concilium Befehlszeilen-Schnittstelle."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

from .evaluate import evaluate_journal
from .llm import LLMClient
from .pipeline import run_pipeline, run_portfolio
from .report import generate_report, generate_track_record_report


def _print_portfolio_summary(pa: dict, file=None) -> None:
    """Gibt eine kompakte Portfolio-Zusammenfassung auf stderr aus."""
    import sys

    f = file or sys.stderr

    tickers = pa.get("analysed_tickers", [])
    correlations = pa.get("correlations", {})
    target_weights = pa.get("target_weights", {})
    concentration_warnings = pa.get("concentration_warnings", [])
    overlap = pa.get("overlap")

    print(f"Analysierte Ticker: {', '.join(tickers)}", file=f)

    if target_weights:
        weight_strs = []
        for t in tickers:
            w = target_weights.get(t)
            if w is not None:
                try:
                    weight_strs.append(f"{t}: {float(w):.1f}%")
                except (TypeError, ValueError):
                    weight_strs.append(f"{t}: n/a")
            else:
                weight_strs.append(f"{t}: n/a")
        print(f"Ziel-Gewichtungen: {', '.join(weight_strs)}", file=f)

    if correlations and len(tickers) >= 2:
        print("\nKorrelationen (|r| > 0.7 hervorgehoben):", file=f)
        for i, t_a in enumerate(tickers):
            for t_b in tickers[i + 1:]:
                r = correlations.get(t_a, {}).get(t_b)
                if r is not None:
                    try:
                        r_float = float(r)
                        marker = " ⚠️" if abs(r_float) > 0.7 else ""
                        print(f"  {t_a} – {t_b}: r={r_float:.2f}{marker}", file=f)
                    except (TypeError, ValueError):
                        print(f"  {t_a} – {t_b}: n/a", file=f)
                else:
                    print(f"  {t_a} – {t_b}: n/a (zu wenige Daten)", file=f)

    if overlap and isinstance(overlap, dict):
        total = overlap.get("total_overlap_pct", 0.0)
        if total > 0:
            print(f"\nGesamt-Overlap mit Depot: {total:.1f}%", file=f)
        overlap_warnings = overlap.get("warnings", [])
        for w in overlap_warnings:
            print(f"  ⚠️ {w}", file=f)

    if concentration_warnings:
        print("\nKonzentrationswarnungen:", file=f)
        for w in concentration_warnings:
            print(f"  - {w}", file=f)


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
        "--portfolio",
        default=None,
        metavar="TICKER1,TICKER2,…",
        help="Kommagetrennte Ticker-Liste für Portfolio-Modus (z. B. RWE.DE,SHEL.L,NEE). "
        "Analysiert mehrere Ticker als Depot-Ganzheit mit Korrelation, Overlap "
        "und Konzentrationsanalyse. Schließt sich mit --ticker/--tickers aus.",
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
    # --evaluate + --ticker/--tickers/--portfolio → Fehler (vor jeglicher Ausführung)
    if args.evaluate is not None and (args.ticker or args.tickers or args.portfolio):
        print(
            "FEHLER: --evaluate kann nicht mit --ticker, --tickers oder --portfolio "
            "kombiniert werden.",
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

    # --- Pipeline-Modus: --ticker, --tickers oder --portfolio required ---
    # Mutual exclusion: --ticker, --tickers, --portfolio
    mode_count = sum(1 for x in (args.ticker, args.tickers, args.portfolio) if x)
    if mode_count > 1:
        parser.error(
            "--ticker, --tickers und --portfolio schließen sich gegenseitig aus."
        )

    if not args.ticker and not args.tickers and not args.portfolio:
        parser.error(
            "--ticker, --tickers oder --portfolio ist erforderlich, "
            "wenn --evaluate nicht gesetzt ist."
        )

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

    # --- Portfolio-Modus (--portfolio) ---
    if args.portfolio:
        ticker_list = [t.strip() for t in args.portfolio.split(",") if t.strip()]
        if not ticker_list:
            parser.error("--portfolio darf nicht leer sein.")

        try:
            portfolio_result = run_portfolio(
                ticker_list,
                llm=llm,
                backtest=args.backtest,
                ensemble=not args.no_ensemble,
                ensemble_runs=args.ensemble_runs,
                resume=args.resume,
            )

            # Pro Ticker einen Report generieren (mit portfolio_analysis)
            pa = portfolio_result.get("portfolio_analysis", {})
            results = portfolio_result.get("results", {})

            successes = 0
            for i, ticker in enumerate(ticker_list):
                if i > 0:
                    print("\n" + "=" * 70 + "\n", file=sys.stderr)

                result = results.get(ticker, {})
                if result.get("error"):
                    print(
                        f"FEHLER bei Ticker '{ticker}': {result['error']}",
                        file=sys.stderr,
                    )
                    continue

                # portfolio_analysis in den Result injizieren für Report-Sektion
                result["portfolio_analysis"] = pa
                report = generate_report(result, reports_dir=reports_dir)
                print(report)

                # Report-Datei speichern
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                resolved_ticker = result.get("ticker", ticker.upper())
                filename = f"portfolio_{resolved_ticker}_{timestamp}.md"
                filepath = os.path.join(reports_dir, filename)
                with open(filepath, "w", encoding="utf-8") as fh:
                    fh.write(report)
                print(f"\n---\nReport gespeichert: {filepath}", file=sys.stderr)
                successes += 1

            # Zusammenfassungs-Report für das gesamte Portfolio
            if len(ticker_list) >= 2:
                print("\n" + "=" * 70 + "\n", file=sys.stderr)
                print("## Portfolio-Zusammenfassung", file=sys.stderr)
                print(file=sys.stderr)

                # Portfolio-Analyse bereits berechnet — nur anzeigen
                _print_portfolio_summary(pa, file=sys.stderr)
                print(file=sys.stderr)

            return 0 if successes > 0 else 1

        except KeyboardInterrupt:
            print(
                "\nABGEBROCHEN (Portfolio-Modus) — Checkpoints bleiben unter state/ erhalten.",
                file=sys.stderr,
            )
            return 130
        except ValueError as exc:
            print(f"FEHLER: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"UNERWARTETER FEHLER im Portfolio-Modus: {exc}", file=sys.stderr)
            logging.exception("Unerwarteter Fehler im Portfolio-Modus")
            return 1

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
