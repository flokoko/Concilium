"""Kontext-Feedback — injiziert Track-Record-Historie in Agenten-Prompts.

Liest journal/decisions.csv und berechnet einfache Statistiken aus den im
Journal gespeicherten Feldern (OHNE Netzwerk / yfinance — nur CSV-Felder).
Die Statistiken werden als deutscher Kontext-Block formatiert und können
in die Trader-/Risk-Manager-/Portfolio-Manager-Prompts eingefügt werden,
damit die Agenten ihre Kalibrierung an der eigenen Historie ausrichten.

Robust: crasht niemals — bei jedem Fehler wird ein leerer String zurückgegeben.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from typing import Any

from .evaluate import realised_return_for_row
from .llm import LLMClient

logger = logging.getLogger(__name__)


def _safe_float(val: Any) -> float | None:
    """Konvertiert einen Wert sicher zu float oder None."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float | None]) -> float | None:
    """Durchschnitt einer Liste von optionalen floats — None bei keinen gültigen Werten."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _read_journal_rows(journal_file: str) -> list[dict[str, str]]:
    """Liest die Journal-CSV und gibt die Zeilen als Liste von dicts zurück.

    Bei Fehlern wird eine leere Liste zurückgegeben (crasht nie).
    """
    try:
        if not os.path.isfile(journal_file):
            return []
        with open(journal_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return list(reader)
    except Exception as exc:  # noqa: BLE001 — crasht nie
        logger.warning("Journal konnte für Feedback nicht gelesen werden: %s", exc)
        return []


def _compute_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Berechnet Track-Record-Statistiken aus Journal-Zeilen.

    Nutzt NUR die im Journal gespeicherten Felder — kein yfinance.
    """
    n_total = len(rows)

    # Aktionen zählen (3-stufig)
    actions = {"KAUFEN": 0, "HALTEN": 0, "VERKAUFEN": 0}
    for row in rows:
        action = (row.get("action") or "").strip().upper()
        if action in actions:
            actions[action] += 1

    # Ratings zählen (5-stufig)
    ratings = {r: 0 for r in ("STARK KAUFEN", "KAUFEN", "HALTEN", "VERKAUFEN", "STARK VERKAUFEN")}
    for row in rows:
        rating = (row.get("rating") or "").strip().upper()
        if rating in ratings:
            ratings[rating] += 1

    # Finale Entscheidungen (GENEHMIGT / ABGELEHNT)
    genehmigt = 0
    abgelehnt = 0
    for row in rows:
        final_dec = (row.get("final_decision") or "").strip().upper()
        if "GENEHMIGT" in final_dec:
            genehmigt += 1
        elif "ABGELEHNT" in final_dec:
            abgelehnt += 1

    # Durchschnittliche confidence / ensemble_confidence
    avg_confidence = _avg([_safe_float(row.get("confidence")) for row in rows])
    avg_ensemble_confidence = _avg([_safe_float(row.get("ensemble_confidence")) for row in rows])

    # Durchschnittlicher portfolio_fit_score (falls vorhanden)
    avg_portfolio_fit = _avg([_safe_float(row.get("portfolio_fit_score")) for row in rows])

    # Durchschnittliche ziel_gewichtung_pct (falls vorhanden)
    avg_ziel_gewichtung = _avg([_safe_float(row.get("ziel_gewichtung_pct")) for row in rows])

    # Anteil KAUFEN-Empfehlungen, die final GENEHMIGT wurden
    kaufen_rows = [row for row in rows if (row.get("action") or "").strip().upper() == "KAUFEN"]
    kaufen_genehmigt = sum(
        1 for row in kaufen_rows
        if "GENEHMIGT" in (row.get("final_decision") or "").strip().upper()
    )
    kauf_genehmigt_pct = (kaufen_genehmigt / len(kaufen_rows) * 100) if kaufen_rows else None

    return {
        "n_total": n_total,
        "actions": actions,
        "ratings": ratings,
        "genehmigt": genehmigt,
        "abgelehnt": abgelehnt,
        "avg_confidence": avg_confidence,
        "avg_ensemble_confidence": avg_ensemble_confidence,
        "avg_portfolio_fit": avg_portfolio_fit,
        "avg_ziel_gewichtung": avg_ziel_gewichtung,
        "kauf_genehmigt_pct": kauf_genehmigt_pct,
    }


def _format_pct(val: float | None) -> str:
    """Formatiert einen Prozentwert (0-100) mit einer Nachkommastelle."""
    if val is None:
        return "N/A"
    return f"{val:.1f}"


def _format_score(val: float | None) -> str:
    """Formatiert einen Score-Wert (z.B. 1-5) mit zwei Nachkommastellen."""
    if val is None:
        return "N/A"
    return f"{val:.2f}"


def build_feedback_context(
    journal_file: str | None = None,
    *,
    min_decisions: int = 5,
) -> str:
    """Baut einen Kontext-Block mit Track-Record-Statistiken für Agenten-Prompts.

    Liest das Entscheidungs-Journal (CSV), berechnet einfache Statistiken
    (Anzahl, Aktionen, Confidence, Portfolio-Fit etc.) und formatiert sie
    als deutschen Text-Block, der in LLM-Prompts eingefügt werden kann.

    Args:
        journal_file: Pfad zur Journal-CSV. Default: journal/decisions.csv.
        min_decisions: Mindestanzahl Entscheidungen für sinnvolles Feedback.
            Bei weniger Entscheidungen wird ein leerer String zurückgegeben,
            damit die Agenten nicht auf Rauschen reagieren (Default: 5).

    Returns:
        Deutscher Kontext-Block (String) oder leerer String bei zu wenigen
        Entscheidungen, fehlender Datei oder Fehlern. Crasht niemals.
    """
    try:
        if journal_file is None:
            journal_file = os.path.join("journal", "decisions.csv")

        rows = _read_journal_rows(journal_file)
        if len(rows) < min_decisions:
            return ""

        stats = _compute_stats(rows)

        n = stats["n_total"]
        a = stats["actions"]
        avg_conf = _format_score(stats["avg_confidence"])
        avg_ens = _format_score(stats["avg_ensemble_confidence"])
        avg_pf = _format_score(stats["avg_portfolio_fit"])
        avg_zg = _format_pct(stats["avg_ziel_gewichtung"])
        kauf_pct = _format_pct(stats["kauf_genehmigt_pct"])

        lines = [
            f"=== DEIN TRACK-RECORD (letzte {n} Entscheidungen) ===",
            f"Gesamt: {n} Entscheidungen (KAUFEN: {a['KAUFEN']}, HALTEN: {a['HALTEN']}, VERKAUFEN: {a['VERKAUFEN']})",
            f"Rating-Verteilung: STARK KAUFEN: {stats['ratings']['STARK KAUFEN']}, KAUFEN: {stats['ratings']['KAUFEN']}, HALTEN: {stats['ratings']['HALTEN']}, VERKAUFEN: {stats['ratings']['VERKAUFEN']}, STARK VERKAUFEN: {stats['ratings']['STARK VERKAUFEN']}",
            f"Finale Entscheidungen: GENEHMIGT: {stats['genehmigt']}, ABGELEHNT: {stats['abgelehnt']}",
            f"Ø Confidence: {avg_conf} / 5 | Ø Ensemble-Confidence: {avg_ens}",
            f"Ø Portfolio-Fit-Score: {avg_pf} / 5 | Ø Ziel-Gewichtung: {avg_zg} %",
            f"KAUFEN-Empfehlungen final genehmigt: {kauf_pct} %",
            "",
            "Berücksichtige diese Historie bei deiner Einschätzung und kalibriere "
            "deine Empfehlungen entsprechend. Bleib sachlich und faktenbasiert.",
        ]

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 — crasht nie
        logger.warning("Feedback-Kontext konnte nicht erstellt werden: %s", exc)
        return ""


# --------------------------------------------------------------------------- #
# Reflexions-Kontext — realisierter Return der letzten Entscheidung für einen Ticker
# --------------------------------------------------------------------------- #


def _deterministic_lesson(raw_return_pct: float | None, action: str) -> str:
    """Erzeugt eine deterministische Lektion abhängig vom Vorzeichen der Rendite."""
    if raw_return_pct is None:
        return "Keine aussagekräftige Rendite verfügbar — prüfe deine Annahmen sorgfältig."
    if raw_return_pct > 0.5:
        return "Die Marktlage hat deine Einschätzung bestätigt; behalte deine Methodik bei."
    if raw_return_pct < -0.5:
        return "Die Marktlage lief gegen dich; überprüfe Timing und Ziel-/Stop-Setzung."
    return "Die Marktlage blieb weitgehend neutral — justiere deine Erwartungen nicht über."


def build_reflection_context(
    ticker: str,
    llm: LLMClient | None = None,
    lookback_days: int = 30,
) -> str:
    """Baut einen Reflexions-Kontext-Block für den letzten Entscheidungs-Eintrag eines Tickers.

    Liest das Entscheidungs-Journal, findet die jüngste Zeile für den Ticker,
    berechnet den realisierten Return (inkl. Alpha vs SPY) via evaluate.realised_return
    und erzeugt einen deutschen Reflexions-Absatz.

    Wenn ein LLMClient übergeben wird, wird die "Lektion" vom LLM generiert
    (einziger Satz). Bei Fehlern oder ohne LLM wird eine deterministische
    Lektion verwendet.

    Args:
        ticker: Ticker-Symbol.
        llm: Optionaler LLMClient für die LLM-generierte Lektion.
        lookback_days: Zeitfenster für die Rendite-Berechnung (Default 30).

    Returns:
        Deutscher Reflexions-String oder "" wenn kein Eintrag/Fehler.
        Crasht niemals.
    """
    try:
        journal_file = os.path.join("journal", "decisions.csv")
        rows = _read_journal_rows(journal_file)
        if not rows:
            return ""

        # Jüngste Zeile für diesen Ticker finden (case-insensitive)
        target = (ticker or "").strip().lower()
        matching_rows: list[dict[str, str]] = []
        for row in rows:
            row_ticker = (row.get("ticker") or "").strip().lower()
            if row_ticker == target:
                ts = row.get("timestamp", "")
                # Nur Zeilen mit parsebarem Timestamp berücksichtigen
                if ts and ts.strip():
                    matching_rows.append(row)

        if not matching_rows:
            return ""

        # Jüngste Zeile = letzte im Journal (Annahme: chronologisch sortiert)
        row = matching_rows[-1]

        rr = realised_return_for_row(row, lookback_days=lookback_days)
        if rr is None:
            return ""

        raw_return_pct = rr.get("raw_return_pct")
        alpha_pct = rr.get("alpha_pct")
        action = rr.get("action", "")
        ts = rr.get("timestamp", "")

        # Lektion generieren
        lesson = _deterministic_lesson(raw_return_pct, action)
        if llm is not None:
            try:
                alpha_str = (
                    f"{alpha_pct:+.2f}%" if alpha_pct is not None else "nicht verfügbar"
                )
                prompt = (
                    f"Du bist ein Trading-Coach. Formuliere EIN deutschen Satz als Lektion "
                    f"aus einer vergangenen Handelsentscheidung.\n\n"
                    f"Ticker: {ticker}\n"
                    f"Aktion: {action}\n"
                    f"Realisierter Return: {raw_return_pct:+.2f}%\n"
                    f"Alpha vs SPY: {alpha_str}\n\n"
                    f"Antworte mit genau einem deutschen Satz (maximal 30 Wörter)."
                )
                messages = [
                    {"role": "system", "content": "Du bist ein präziser Trading-Coach."},
                    {"role": "user", "content": prompt},
                ]
                llm_lesson = llm.chat(messages, temperature=0.3)
                if llm_lesson and llm_lesson.strip():
                    lesson = llm_lesson.strip()
            except Exception as llm_exc:  # noqa: BLE001 — Fallback
                logger.debug("LLM-Lektion fehlgeschlagen, verwende deterministische Lektion: %s", llm_exc)

        alpha_str = f"{alpha_pct:+.2f}%" if alpha_pct is not None and math.isfinite(alpha_pct) else "-"
        raw_str = f"{raw_return_pct:+.2f}%" if raw_return_pct is not None and math.isfinite(raw_return_pct) else "N/A"
        text = (
            f"=== DEINE LETZTE ENTSCHEIDUNG ZU {ticker.upper()} ({ts}) ===\n"
            f"Aktion: {action} | Realisierter Return: {raw_str} "
            f"| Alpha vs SPY: {alpha_str}\n"
            f"Lerne daraus: {lesson}"
        )
        return text
    except Exception as exc:  # noqa: BLE001 — crasht nie
        logger.warning("Reflexions-Kontext konnte nicht erstellt werden: %s", exc)
        return ""
