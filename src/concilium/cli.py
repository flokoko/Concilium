"""CLI-Modul — Concilium Befehlszeilen-Schnittstelle."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from .data import _parse_as_of
from .evaluate import evaluate_journal
from .llm import LLMClient
from .pipeline import run_pipeline, run_portfolio
from .report import generate_report, generate_track_record_report
from .review import run_review
from .usage import summarize_usage

# Platform-Guard: fcntl ist Linux/Unix-only; auf anderen Plattformen None.
try:
    import fcntl
except ImportError:  # pragma: no cover — Windows hat kein fcntl
    fcntl = None


def _default_watchlist_path() -> str:
    """Liefert den Standardpfad für watchlist.txt (Repo-Root)."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "watchlist.txt"
    )


def _read_watchlist(path: str | None = None) -> list[str]:
    """Liest die Watchlist-Datei und gibt eine Liste der Ticker zurück.

    - Ein Ticker pro Zeile, '#'-Kommentare und Leerzeilen werden ignoriert.
    - Whitespace wird getrimmt.
    - Crasht nie: leere Liste bei fehlender Datei oder Lesefehler.
    - Pfad: expliziter Parameter > CONCILIUM_WATCHLIST-Env > Standardpfad.
    """
    if path is None:
        path = os.environ.get("CONCILIUM_WATCHLIST") or _default_watchlist_path()
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        logging.debug("Watchlist-Datei nicht gefunden: %s", path)
        return []
    except OSError as exc:  # noqa: BLE001 — nie crashen
        logging.warning("Watchlist-Datei konnte nicht gelesen werden: %s", exc)
        return []

    tickers: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tickers.append(stripped)
    return tickers


def _state_dir(state_dir: str | None = None) -> str:
    """Löst das State-Verzeichnis auf (gleicher Mechanismus wie checkpoint.py).

    Priorität: expliziter Parameter > CONCILIUM_STATE_DIR-Env > 'state'.
    """
    if state_dir is not None:
        return state_dir
    env = os.environ.get("CONCILIUM_STATE_DIR")
    if env:
        return env
    return "state"


def _write_calibration_json(eval_result: dict, *, state_dir: str | None = None) -> None:
    """Schreibt eine netzfreie Kalibrierungs-JSON aus den evaluierten Kennzahlen.

    Inhalt: nur aggregierte Werte (hit_rate_gesamt, nach_aktion mit hit_rate
    und avg_confidence) — KEINE Rohdaten, KEINE Kurse.

    Phase 1: hit_rate_gesamt aus evaluate.py enthaelt bereits nur echte Trades
    (KAUFEN/VERKAUFEN); HALTEN verbleibt in nach_aktion zur Transparenz, wird
    aber von den Konsumenten (feedback.py, agents.py) aus der Kalibrierung
    ausgenommen. HALTEN bleibt hier drin, damit alte Konsumenten (und der
    HALTEN-Rating-Fallback) nicht brechen.

    Atomar (tmpfile + os.replace) mit fcntl-Lock (best effort).
    Crasht nie — bei Fehler wird nur gewarnt.
    """
    try:
        base_dir = _state_dir(state_dir)
        os.makedirs(base_dir, exist_ok=True)
        cal_path = os.path.join(base_dir, "calibration.json")

        nach_aktion_out: dict[str, dict] = {}
        for action in ("KAUFEN", "HALTEN", "VERKAUFEN"):
            adata = eval_result.get("nach_aktion", {}).get(action, {})
            if adata.get("n", 0) == 0:
                continue
            nach_aktion_out[action] = {
                "n": adata["n"],
                "hit_rate": adata.get("hit_rate"),
                "avg_confidence": adata.get("avg_confidence"),
            }

        payload = {
            "erstellt_am": datetime.now().isoformat(),
            "anzahl_entscheidungen": eval_result.get("anzahl_entscheidungen", 0),
            "hit_rate_gesamt": eval_result.get("hit_rate_gesamt"),
            "nach_aktion": nach_aktion_out,
        }

        tmp_path = cal_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            if fcntl is not None:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                except Exception:  # noqa: BLE001 — best effort
                    pass
            try:
                json.dump(payload, fh, ensure_ascii=False)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except Exception:  # noqa: BLE001 — best effort
                        pass
        os.replace(tmp_path, cal_path)
        logging.debug("Kalibrierungs-JSON gespeichert: %s", cal_path)
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logging.warning("Kalibrierungs-JSON konnte nicht gespeichert werden: %s", exc)


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
        "--watchlist",
        action="store_true",
        help="Watchlist-Analyse: liest watchlist.txt (Env CONCILIUM_WATCHLIST = Pfad), "
        "führt ZUERST --evaluate + calibration.json aus, dann Batch-Analyse aller Ticker. "
        "Kann mit --evaluate, --no-llm, --no-ensemble, --ensemble-runs, --peers, "
        "--lookback, --date kombiniert werden. Schließt sich mit "
        "--ticker/--tickers/--portfolio aus.",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Exit-Review: scannt das reale Depot (Google-Sheet) auf Verkaufskandidaten. "
        "Führt ZUERST --evaluate + calibration.json aus, dann eine verkürzte, "
        "verkaufsfokussierte Analyse jeder Aktien-Position (keine ETFs/Commodities). "
        "Kann mit --evaluate, --no-llm, --no-ensemble, --ensemble-runs, --peers, "
        "--lookback, --max-positions, --date kombiniert werden. "
        "Schließt sich mit --ticker/--tickers/--portfolio/--watchlist aus.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="[nur --review] Begrenzt die Anzahl der analysierten Depot-Positionen "
        "(größte zuerst, nach depot_pct). Default: alle Aktien-Positionen.",
    )
    parser.add_argument(
        "--usage",
        action="store_true",
        help="Token-Usage-Report: aggregiert den LLM-Token-Verbrauch aus usage/usage.csv "
        "und gibt eine Zusammenfassung aus. Führt NICHT die Pipeline aus. "
        "Schließt sich mit --ticker/--tickers/--portfolio aus.",
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
    parser.add_argument(
        "--debate-rounds",
        type=int,
        default=1,
        help="Anzahl der Bull/Bear-Debatten-Runden (Default: 1). "
        "Höhere Werte = tiefere Debatte, mehr LLM-Calls.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Analysedatum pinnen (nur Kurs-Historie bis zu diesem Datum). "
        "Fundamentals/Makro/News bleiben aktuell.",
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
    # --watchlist schließt sich mit --ticker/--tickers/--portfolio aus, KANN mit --evaluate kombiniert werden
    if args.evaluate is not None and (args.ticker or args.tickers or args.portfolio):
        print(
            "FEHLER: --evaluate kann nicht mit --ticker, --tickers oder --portfolio "
            "kombiniert werden.",
            file=sys.stderr,
        )
        return 1

    # --date ist mit allen Analyse-Modi kombinierbar, aber NICHT mit --evaluate
    # (evaluate hat seinen eigenen Lookback über die Journal-Historie).
    if args.date and args.evaluate is not None:
        print(
            "FEHLER: --date kann nicht mit --evaluate kombiniert werden "
            "(--evaluate hat seinen eigenen Lookback).",
            file=sys.stderr,
        )
        return 1

    # --date früh validieren (Format YYYY-MM-DD, kein Zukunftsdatum) —
    # Fail-fast vor jeglicher Ausführung. Die Details (vor dem ersten Kurs,
    # zu wenig Historie) prüft collect_ticker_data mit echten Kursdaten.
    if args.date:
        try:
            _parse_as_of(args.date)
        except ValueError as exc:
            print(f"FEHLER: {exc}", file=sys.stderr)
            return 1

    if args.watchlist and (args.ticker or args.tickers or args.portfolio):
        print(
            "FEHLER: --watchlist kann nicht mit --ticker, --tickers oder --portfolio "
            "kombiniert werden.",
            file=sys.stderr,
        )
        return 1

    if args.review and (args.ticker or args.tickers or args.portfolio or args.watchlist):
        print(
            "FEHLER: --review kann nicht mit --ticker, --tickers, --portfolio "
            "oder --watchlist kombiniert werden.",
            file=sys.stderr,
        )
        return 1

    if args.usage and (args.ticker or args.tickers or args.portfolio):
        print(
            "FEHLER: --usage kann nicht mit --ticker, --tickers oder --portfolio "
            "kombiniert werden.",
            file=sys.stderr,
        )
        return 1

    # --usage ist eigenständig: Pipeline wird nicht ausgeführt
    if args.usage:
        usage_data = summarize_usage()
        if usage_data.get("anzahl_calls", 0) == 0:
            print("Noch keine Usage-Daten erfasst.")
            return 0

        print("=== Token-Usage-Report ===")
        print()
        print(f"Anzahl LLM-Calls:  {usage_data['anzahl_calls']}")
        print(f"Summe Prompt-Tokens:      {usage_data['summe_prompt_tokens']:,}")
        print(f"Summe Completion-Tokens:  {usage_data['summe_completion_tokens']:,}")
        print(f"Summe Total-Tokens:       {usage_data['summe_total_tokens']:,}")
        print(f"Eindeutige Ticker:        {usage_data['anzahl_ticker']}")
        print()

        ticker_tokens = usage_data.get("ticker_tokens", {})
        if ticker_tokens:
            print("Token-Verbrauch pro Ticker:")
            print(f"  {'Ticker':<12} {'Total-Tokens':>14}")
            print(f"  {'-' * 12} {'-' * 14}")
            for ticker in sorted(ticker_tokens):
                print(f"  {ticker:<12} {ticker_tokens[ticker]:>14,}")

        return 0

    # --evaluate ist eigenständig: Pipeline wird nicht ausgeführt
    # (außer bei --watchlist, dort läuft evaluate vorn mit — siehe unten)
    if args.evaluate is not None and not args.watchlist and not args.review:
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

            # Kalibrierungs-JSON schreiben (netzfreie Aggregate für feedback.py)
            _write_calibration_json(eval_result)

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

    # --- Watchlist-Modus (--watchlist) ---
    # Führt ZUERST evaluate_journal + _write_calibration_json aus, dann Batch-Analyse
    # aller Ticker aus watchlist.txt (analog --tickers).
    if args.watchlist:
        ticker_list = _read_watchlist()
        if not ticker_list:
            print(
                "FEHLER: Watchlist ist leer oder watchlist.txt nicht gefunden.",
                file=sys.stderr,
            )
            return 1

        level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        llm = None if args.no_llm else LLMClient()

        # Peers-Liste parsen (kommagetrennt)
        peers_list: list[str] | None = None
        if args.peers:
            peers_list = [p.strip() for p in args.peers.split(",") if p.strip()]

        reports_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports"
        )
        os.makedirs(reports_dir, exist_ok=True)

        # Schritt 1: evaluate_journal + calibration.json (damit Feedback aktuell ist)
        eval_journal_path = args.evaluate if args.evaluate is not None else "journal/decisions.csv"
        try:
            print("--- Watchlist: Track-Record-Evaluierung ---", file=sys.stderr)
            eval_result = evaluate_journal(
                eval_journal_path,
                lookback_days=args.lookback,
                llm=llm,
            )
            _write_calibration_json(eval_result)
            report = generate_track_record_report(eval_result)
            print(report)

            # Track-Record-Report speichern
            date_str = datetime.now().strftime("%Y%m%d")
            track_filepath = os.path.join(reports_dir, f"track_record_{date_str}.md")
            with open(track_filepath, "w", encoding="utf-8") as fh:
                fh.write(report)
            print(
                f"\n---\nTrack-Record-Report gespeichert: {track_filepath}",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"FEHLER bei Track-Record-Evaluierung (Watchlist): {exc}",
                file=sys.stderr,
            )
            logging.exception("Track-Record-Fehler (Watchlist)")
            return 1

        # Schritt 2: Batch-Analyse aller Watchlist-Ticker (analog --tickers)
        print(
            f"\n--- Watchlist: Analyse von {len(ticker_list)} Tickern ---",
            file=sys.stderr,
        )
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
                    debate_rounds=args.debate_rounds,
                    as_of=args.date,
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
                print(
                    f"UNERWARTETER FEHLER bei Ticker '{ticker}': {exc}",
                    file=sys.stderr,
                )
                logging.exception("Unerwarteter Fehler bei Ticker '%s'", ticker)
                failures += 1

        if failures > 0:
            print(
                f"\nWARNUNG: {failures} von {len(ticker_list)} Tickern fehlgeschlagen.",
                file=sys.stderr,
            )

        return 0 if successes > 0 else 1

    # --- Exit-Review-Modus (--review) ---
    # Führt ZUERST evaluate_journal + _write_calibration_json aus (damit
    # calibration.json aktuell ist — wichtig für Feedback/Dämpfung), dann
    # run_review über alle Aktien-Positionen des realen Depots.
    if args.review:
        level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        llm = None if args.no_llm else LLMClient()

        # Peers-Liste parsen (kommagetrennt)
        peers_list: list[str] | None = None
        if args.peers:
            peers_list = [p.strip() for p in args.peers.split(",") if p.strip()]

        reports_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports"
        )
        os.makedirs(reports_dir, exist_ok=True)

        # Schritt 1: evaluate_journal + calibration.json (Feedback/Dämpfung aktuell)
        eval_journal_path = (
            args.evaluate if args.evaluate is not None else "journal/decisions.csv"
        )
        try:
            print("--- Review: Track-Record-Evaluierung ---", file=sys.stderr)
            eval_result = evaluate_journal(
                eval_journal_path,
                lookback_days=args.lookback,
                llm=llm,
            )
            _write_calibration_json(eval_result)
            track_report = generate_track_record_report(eval_result)
            print(track_report)

            # Track-Record-Report speichern
            date_str = datetime.now().strftime("%Y%m%d")
            track_filepath = os.path.join(reports_dir, f"track_record_{date_str}.md")
            with open(track_filepath, "w", encoding="utf-8") as fh:
                fh.write(track_report)
            print(
                f"\n---\nTrack-Record-Report gespeichert: {track_filepath}",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"FEHLER bei Track-Record-Evaluierung (Review): {exc}",
                file=sys.stderr,
            )
            logging.exception("Track-Record-Fehler (Review)")
            return 1

        # Schritt 2: Depot-Review (Verkaufskandidaten)
        print(
            "\n--- Review: Depot-Scan auf Verkaufskandidaten ---",
            file=sys.stderr,
        )
        try:
            review_result = run_review(
                llm,
                backtest=args.backtest,
                ensemble=not args.no_ensemble,
                ensemble_runs=args.ensemble_runs,
                resume=args.resume,
                debate_rounds=args.debate_rounds,
                peers=peers_list,
                max_positions=args.max_positions,
                as_of=args.date,
            )
        except KeyboardInterrupt:
            print(
                "\nABGEBROCHEN (Review) — Checkpoints bleiben unter state/ erhalten.",
                file=sys.stderr,
            )
            return 130
        except Exception as exc:  # noqa: BLE001 — Review crasht die CLI nicht hart
            print(f"UNERWARTETER FEHLER im Review-Modus: {exc}", file=sys.stderr)
            logging.exception("Unerwarteter Fehler im Review-Modus")
            return 1

        ergebnisse = review_result.get("ergebnisse", {})
        skipped = review_result.get("positions_uebersprungen", 0)
        failures = review_result.get("fehler", 0)

        # Reports pro analysierter Position ausgeben + speichern
        successes = 0
        for i, (ticker, entry) in enumerate(ergebnisse.items()):
            if i > 0:
                print("\n" + "=" * 70 + "\n", file=sys.stderr)

            try:
                result = entry.get("result") or {}
                report_text = entry.get("report") or generate_report(
                    result, review_mode=True
                )
                print(report_text)

                # Report-Datei speichern: review_{TICKER}_{timestamp}.md
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                resolved_ticker = result.get("ticker", ticker.upper())
                filename = f"review_{resolved_ticker}_{timestamp}.md"
                filepath = os.path.join(reports_dir, filename)
                with open(filepath, "w", encoding="utf-8") as fh:
                    fh.write(report_text)
                print(f"\n---\nReport gespeichert: {filepath}", file=sys.stderr)
                successes += 1
            except Exception as exc:  # noqa: BLE001 — Report-Fehler zählen, nicht crashen
                print(f"FEHLER bei Report für '{ticker}': {exc}", file=sys.stderr)
                logging.warning("Report für '%s' fehlgeschlagen: %s", ticker, exc)
                failures += 1

        # --- Review-Zusammenfassung (sortiert nach depot_pct) ---
        print("\n" + "=" * 70 + "\n", file=sys.stderr)
        print("## Review-Zusammenfassung", file=sys.stderr)
        print(file=sys.stderr)

        if not ergebnisse:
            print(
                "Keine Aktien-Positionen analysiert "
                f"({skipped} Position(en) übersprungen).",
                file=sys.stderr,
            )
        else:
            sorted_entries = sorted(
                ergebnisse.items(),
                key=lambda kv: kv[1].get("depot_pct") or 0.0,
                reverse=True,
            )
            for ticker, entry in sorted_entries:
                pct = entry.get("depot_pct") or 0.0
                if entry.get("verkauf_empfehlung"):
                    marker = "🔴 VERKAUFEN"
                else:
                    marker = "✅ behalten"
                name = entry.get("name") or ticker
                print(
                    f"  {marker}  {ticker} ({name}): {pct:.1f}% Depot",
                    file=sys.stderr,
                )

            empfohlen = sum(
                1 for e in ergebnisse.values() if e.get("verkauf_empfehlung")
            )
            print(file=sys.stderr)
            print(
                f"{len(ergebnisse)} Positionen analysiert, "
                f"{empfohlen} Verkaufsempfehlung(en), "
                f"{skipped} Position(en) übersprungen"
                + (f", {failures} Fehler" if failures > 0 else "")
                + ".",
                file=sys.stderr,
            )

        if failures > 0:
            print(
                f"\nWARNUNG: {failures} Ticker-Position(en) fehlgeschlagen.",
                file=sys.stderr,
            )

        return 0 if (successes > 0 or (not ergebnisse and failures == 0)) else 1

    # --- Pipeline-Modus: --ticker, --tickers oder --portfolio required ---
    # Mutual exclusion: --ticker, --tickers, --portfolio
    mode_count = sum(1 for x in (args.ticker, args.tickers, args.portfolio) if x)
    if mode_count > 1:
        parser.error(
            "--ticker, --tickers und --portfolio schließen sich gegenseitig aus."
        )

    if not args.ticker and not args.tickers and not args.portfolio:
        parser.error(
            "--ticker, --tickers, --portfolio, --usage oder --watchlist ist erforderlich, "
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
                peers=peers_list,
                debate_rounds=args.debate_rounds,
                as_of=args.date,
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
                    debate_rounds=args.debate_rounds,
                    as_of=args.date,
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
            debate_rounds=args.debate_rounds,
            as_of=args.date,
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
