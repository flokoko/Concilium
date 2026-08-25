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
import json
import logging
import math
import os
from datetime import datetime, timedelta
from typing import Any

from .evaluate import realised_return_for_row
from .llm import LLMClient

logger = logging.getLogger(__name__)

# Maximales Alter der Kalibrierungs-JSON in Tagen (danach Fallback auf Proxy).
_CALIBRATION_MAX_AGE_DAYS = 7


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


def _load_calibration_json(state_dir: str | None = None) -> dict[str, Any] | None:
    """Liest state/calibration.json und gibt das dict zurück.

    Gibt None zurück bei fehlender Datei, kaputtem JSON oder wenn die Datei
    älter als _CALIBRATION_MAX_AGE_DAYS ist. Crasht nie.
    """
    try:
        cal_path = os.path.join(_state_dir(state_dir), "calibration.json")
        if not os.path.isfile(cal_path):
            return None
        with open(cal_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None

        # Alters-Check: erstellt_am muss nicht zu alt sein
        erstellt_am = data.get("erstellt_am")
        if isinstance(erstellt_am, str) and erstellt_am.strip():
            try:
                erstellt_dt = datetime.fromisoformat(erstellt_am)
                age = datetime.now() - erstellt_dt
                if age > timedelta(days=_CALIBRATION_MAX_AGE_DAYS):
                    logger.debug(
                        "Kalibrierungs-JSON älter als %d Tage — ignoriere",
                        _CALIBRATION_MAX_AGE_DAYS,
                    )
                    return None
            except (ValueError, TypeError):
                # Wenn nicht parsebar → ignoriere (behalte None)
                return None
        else:
            # Kein erstellt_am → ungültig
            return None

        return data
    except Exception as exc:  # noqa: BLE001 — crasht nie
        logger.debug("Kalibrierungs-JSON konnte nicht geladen werden: %s", exc)
        return None


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


def _compute_kalibrierung_proxy(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Berechnet eine netzfreie Kalibrierungs-Näherung aus Journal-CSV-Feldern.

    Da feedback.py kein yfinance laden darf, wird die final_decision als
    Proxy für hit verwendet: GENEHMIGT → 1 (Erfolg), ABGELEHNT → 0 (kein Erfolg).
    Ø Confidence wird auf 0-1 normalisiert (conf/5).

    Gap = Ø_Konfidenz - Genehmigungs-Rate (positiv = überkonfident).

    Returns:
        dict mit avg_confidence, genehmigungs_rate, gap, tendenz.
        Alle None bei zu wenigen / fehlenden Daten. Crasht nie.
    """
    empty: dict[str, Any] = {
        "avg_confidence": None,
        "genehmigungs_rate": None,
        "gap": None,
        "tendenz": None,
        "n": 0,
    }

    # Nur Zeilen mit confidence und final_decision verwenden
    valid: list[dict[str, str]] = []
    for row in rows:
        conf = _safe_float(row.get("confidence"))
        final = (row.get("final_decision") or "").strip().upper()
        if conf is None or not math.isfinite(conf) or conf <= 0:
            continue
        if "GENEHMIGT" not in final and "ABGELEHNT" not in final:
            continue
        valid.append(row)

    if not valid:
        return empty

    n = len(valid)
    conf_sum = 0.0
    genehmigt_sum = 0
    for row in valid:
        conf = _safe_float(row.get("confidence"))
        conf_sum += conf / 5.0
        final = (row.get("final_decision") or "").strip().upper()
        if "GENEHMIGT" in final:
            genehmigt_sum += 1

    avg_conf = conf_sum / n
    genehmigungs_rate = genehmigt_sum / n
    gap = avg_conf - genehmigungs_rate

    if gap > 0.15:
        tendenz = "überkonfident"
    elif gap < -0.15:
        tendenz = "unterkonfident"
    else:
        tendenz = "gut kalibriert"

    return {
        "avg_confidence": avg_conf,
        "genehmigungs_rate": genehmigungs_rate,
        "gap": gap,
        "tendenz": tendenz,
        "n": n,
    }


def _compute_kalibrierung_proxy_per_action(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Berechnet eine netzfreie Kalibrierungs-Näherung pro Aktion (3-stufig).

    Wie ``_compute_kalibrierung_proxy``, aber aufgespalten nach KAUFEN / HALTEN /
    VERKAUFEN.  Nutzt ebenfalls die final_decision als Hit-Proxy.

    Returns:
        dict {aktion: {avg_confidence, genehmigungs_rate, gap, tendenz, n}}.
        Nur Aktionen mit ≥3 gültigen Zeilen. Leeres dict bei zu wenigen Daten.
    """
    # Zeilen nach Aktion gruppieren (nur 3-stufig)
    per_action: dict[str, list[dict[str, str]]] = {
        "KAUFEN": [],
        "HALTEN": [],
        "VERKAUFEN": [],
    }
    for row in rows:
        action = (row.get("action") or "").strip().upper()
        if action in per_action:
            per_action[action].append(row)

    result: dict[str, dict[str, Any]] = {}
    for action, action_rows in per_action.items():
        # Gleiche Filter-Logik wie _compute_kalibrierung_proxy
        valid: list[dict[str, str]] = []
        for row in action_rows:
            conf = _safe_float(row.get("confidence"))
            final = (row.get("final_decision") or "").strip().upper()
            if conf is None or not math.isfinite(conf) or conf <= 0:
                continue
            if "GENEHMIGT" not in final and "ABGELEHNT" not in final:
                continue
            valid.append(row)

        if len(valid) < 3:
            continue  # Rauschen vermeiden

        n = len(valid)
        conf_sum = 0.0
        genehmigt_sum = 0
        for row in valid:
            conf = _safe_float(row.get("confidence"))
            conf_sum += conf / 5.0
            final = (row.get("final_decision") or "").strip().upper()
            if "GENEHMIGT" in final:
                genehmigt_sum += 1

        avg_conf = conf_sum / n
        genehmigungs_rate = genehmigt_sum / n
        gap = avg_conf - genehmigungs_rate

        if gap > 0.15:
            tendenz = "überkonfident"
        elif gap < -0.15:
            tendenz = "unterkonfident"
        else:
            tendenz = "gut kalibriert"

        result[action] = {
            "avg_confidence": avg_conf,
            "genehmigungs_rate": genehmigungs_rate,
            "gap": gap,
            "tendenz": tendenz,
            "n": n,
        }

    return result


# --------------------------------------------------------------------------- #
# Echte Hit-Rate aus Kalibrierungs-JSON
# --------------------------------------------------------------------------- #


def _compute_kalibrierung_echt(cal: dict[str, Any]) -> dict[str, Any]:
    """Berechnet die Kalibrierung aus der echten Hit-Rate (aus calibration.json).

    avg_confidence = Ø über alle Aktionen, gewichtet nach n.
    hit_rate = echte hit_rate_gesamt aus der JSON.
    gap = avg_confidence - hit_rate.
    tendenz = Schwellen ±0.15 wie bestehend.

    Returns:
        dict mit avg_confidence, hit_rate, gap, tendenz.
        Alle None bei fehlenden/ungültigen Daten. Crasht nie.
    """
    empty: dict[str, Any] = {
        "avg_confidence": None,
        "hit_rate": None,
        "gap": None,
        "tendenz": None,
    }

    hit_rate = cal.get("hit_rate_gesamt")
    if hit_rate is None or not isinstance(hit_rate, (int, float)):
        return empty

    # Gewichtete Ø-Confidence über alle Aktionen
    total_n = 0
    conf_sum = 0.0
    for action, adata in (cal.get("nach_aktion") or {}).items():
        n = adata.get("n", 0)
        avg_conf = adata.get("avg_confidence")
        if n > 0 and avg_conf is not None and isinstance(avg_conf, (int, float)):
            total_n += n
            conf_sum += avg_conf * n

    if total_n == 0:
        return empty

    avg_confidence = conf_sum / total_n
    gap = avg_confidence - hit_rate

    if gap > 0.15:
        tendenz = "überkonfident"
    elif gap < -0.15:
        tendenz = "unterkonfident"
    else:
        tendenz = "gut kalibriert"

    return {
        "avg_confidence": avg_confidence,
        "hit_rate": hit_rate,
        "gap": gap,
        "tendenz": tendenz,
    }


def _compute_kalibrierung_echt_per_action(
    cal: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Berechnet die echte Kalibrierung pro Aktion aus calibration.json.

    Pro Aktion: {avg_confidence, hit_rate (echt), gap, tendenz, n}.
    Nur Aktionen mit n >= 3.

    Returns:
        dict {aktion: {avg_confidence, hit_rate, gap, tendenz, n}}.
        Leeres dict bei fehlenden/ungültigen Daten. Crasht nie.
    """
    result: dict[str, dict[str, Any]] = {}
    for action in ("KAUFEN", "HALTEN", "VERKAUFEN"):
        adata = (cal.get("nach_aktion") or {}).get(action)
        if not adata or not isinstance(adata, dict):
            continue
        n = adata.get("n", 0)
        if n < 3:
            continue

        avg_conf = adata.get("avg_confidence")
        hit_rate = adata.get("hit_rate")
        if avg_conf is None or hit_rate is None:
            continue
        if not isinstance(avg_conf, (int, float)) or not isinstance(hit_rate, (int, float)):
            continue

        gap = avg_conf - hit_rate

        if gap > 0.15:
            tendenz = "überkonfident"
        elif gap < -0.15:
            tendenz = "unterkonfident"
        else:
            tendenz = "gut kalibriert"

        result[action] = {
            "avg_confidence": float(avg_conf),
            "hit_rate": float(hit_rate),
            "gap": gap,
            "tendenz": tendenz,
            "n": n,
        }

    return result


def _compute_stats(rows: list[dict[str, str]], *, min_decisions: int = 5) -> dict[str, Any]:
    """Berechnet Track-Record-Statistiken aus Journal-Zeilen.

    Nutzt NUR die im Journal gespeicherten Felder — kein yfinance.
    Wenn eine gültige calibration.json existiert, wird die echte Hit-Rate
    verwendet (statt des Genehmigungs-Rate-Proxys).
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

    # --- Kalibrierung (echte Hit-Rate aus JSON, Fallback auf Proxy) --- #
    cal_json = _load_calibration_json()
    if cal_json is not None and (cal_json.get("anzahl_entscheidungen") or 0) >= min_decisions:
        kalibrierung = _compute_kalibrierung_echt(cal_json)
        kalibrierung["quelle"] = "echte_hit_rate"
        kalibrierung_pro_aktion = _compute_kalibrierung_echt_per_action(cal_json)
    else:
        # Fallback: Proxy (netzfrei, aus Journal-CSV-Feldern)
        kalibrierung = _compute_kalibrierung_proxy(rows)
        kalibrierung["quelle"] = "proxy"
        kalibrierung_pro_aktion = _compute_kalibrierung_proxy_per_action(rows)

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
        "kalibrierung": kalibrierung,
        "kalibrierung_pro_aktion": kalibrierung_pro_aktion,
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

        stats = _compute_stats(rows, min_decisions=min_decisions)

        n = stats["n_total"]
        a = stats["actions"]
        avg_conf = _format_score(stats["avg_confidence"])
        avg_ens = _format_score(stats["avg_ensemble_confidence"])
        avg_pf = _format_score(stats["avg_portfolio_fit"])
        avg_zg = _format_pct(stats["avg_ziel_gewichtung"])
        kauf_pct = _format_pct(stats["kauf_genehmigt_pct"])

        # Kalibrierungs-Zeile (echte Hit-Rate aus JSON oder Proxy-Fallback)
        kal = stats.get("kalibrierung", {})
        kal_quelle = kal.get("quelle", "proxy")
        kal_gap = kal.get("gap")
        kal_tendenz = kal.get("tendenz")
        kal_avg_conf = kal.get("avg_confidence")
        kal_hit_rate = kal.get("hit_rate")
        kal_genehm_rate = kal.get("genehmigungs_rate")
        if kal_gap is not None and math.isfinite(kal_gap):
            avg_conf_display = (
                f"{kal_avg_conf * 5:.1f}/5" if kal_avg_conf is not None else "N/A"
            )
            if kal_quelle == "echte_hit_rate" and kal_hit_rate is not None:
                rate_display = f"{kal_hit_rate * 100:.0f}%"
                kalibrierung_line = (
                    f"Konfidenz-Kalibrierung: Ø Confidence {avg_conf_display} vs. "
                    f"echte Hit-Rate {rate_display}. Tendenz: {kal_tendenz}."
                )
            else:
                # Proxy-Fallback: Genehmigungs-Rate
                genehm_display = (
                    f"{kal_genehm_rate * 100:.0f}%" if kal_genehm_rate is not None else "N/A"
                )
                kalibrierung_line = (
                    f"Konfidenz-Kalibrierung (Proxy): Ø Confidence {avg_conf_display} vs. "
                    f"Genehmigungs-Rate {genehm_display}. Tendenz: {kal_tendenz}."
                )
        else:
            kalibrierung_line = (
                "Konfidenz-Kalibrierung: noch zu wenige Daten für eine Aussage."
            )

        lines = [
            f"=== DEIN TRACK-RECORD (letzte {n} Entscheidungen) ===",
            f"Gesamt: {n} Entscheidungen (KAUFEN: {a['KAUFEN']}, HALTEN: {a['HALTEN']}, VERKAUFEN: {a['VERKAUFEN']})",
            f"Rating-Verteilung: STARK KAUFEN: {stats['ratings']['STARK KAUFEN']}, KAUFEN: {stats['ratings']['KAUFEN']}, HALTEN: {stats['ratings']['HALTEN']}, VERKAUFEN: {stats['ratings']['VERKAUFEN']}, STARK VERKAUFEN: {stats['ratings']['STARK VERKAUFEN']}",
            f"Finale Entscheidungen: GENEHMIGT: {stats['genehmigt']}, ABGELEHNT: {stats['abgelehnt']}",
            f"Ø Confidence: {avg_conf} / 5 | Ø Ensemble-Confidence: {avg_ens}",
            f"Ø Portfolio-Fit-Score: {avg_pf} / 5 | Ø Ziel-Gewichtung: {avg_zg} %",
            f"KAUFEN-Empfehlungen final genehmigt: {kauf_pct} %",
            kalibrierung_line,
        ]

        # --- Kalibrierung pro Aktion (nur wenn Daten vorhanden) --- #
        kal_pro_aktion = stats.get("kalibrierung_pro_aktion", {})
        if kal_pro_aktion:
            lines.append("")
            lines.append("Kalibrierung pro Aktion:")
            for action_name in ("KAUFEN", "HALTEN", "VERKAUFEN"):
                entry = kal_pro_aktion.get(action_name)
                if entry is None:
                    continue
                ak_avg_conf = entry.get("avg_confidence")
                ak_gap = entry.get("gap")
                ak_tendenz = entry.get("tendenz")
                gap_sign = f"{ak_gap:+.2f}" if ak_gap is not None else "N/A"
                conf_display = (
                    f"{ak_avg_conf:.2f}" if ak_avg_conf is not None else "N/A"
                )
                if kal_quelle == "echte_hit_rate":
                    ak_hit = entry.get("hit_rate")
                    hit_display = f"{ak_hit:.2f}" if ak_hit is not None else "N/A"
                    lines.append(
                        f"- {action_name}: Ø Confidence {conf_display}, "
                        f"Hit-Rate {hit_display}, "
                        f"Gap {gap_sign} ({ak_tendenz})"
                    )
                else:
                    # Proxy-Fallback: Genehmigungs-Rate
                    ak_genehm = entry.get("genehmigungs_rate")
                    genehm_display = (
                        f"{ak_genehm:.2f}" if ak_genehm is not None else "N/A"
                    )
                    lines.append(
                        f"- {action_name}: Ø Confidence {conf_display}, "
                        f"Genehmigungs-Rate {genehm_display}, "
                        f"Gap {gap_sign} ({ak_tendenz})"
                    )

        lines.extend([
            "",
            "Berücksichtige diese Historie bei deiner Einschätzung und kalibriere "
            "deine Empfehlungen entsprechend. Bleib sachlich und faktenbasiert.",
            "Passe deine Konfidenz an die historische Trefferquote deiner Aktion an "
            "— bei überkonfidenter Historie sei vorsichtiger mit hohen Konfidenzwerten.",
        ])

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
