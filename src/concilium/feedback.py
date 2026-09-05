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

from . import config
from .evaluate import _parse_timestamp, realised_return_for_row
from .journal import (  # noqa: F401 — JOURNAL_HEADER: re-export für _write_resolution-Fallback
    JOURNAL_HEADER,
    REFLECTION_STATUS_PENDING,
    REFLECTION_STATUS_RESOLVED,
    _acquire_lock,
    _release_lock,
)
from .llm import LLMClient

logger = logging.getLogger(__name__)

# Maximales Alter der Kalibrierungs-JSON in Tagen (danach Fallback auf Proxy).
_CALIBRATION_MAX_AGE_DAYS = 7


def _state_dir(state_dir: str | None = None) -> str:
    """Löst das State-Verzeichnis auf (gleicher Mechanismus wie checkpoint.py).

    Priorität: expliziter Parameter > CONCILIUM_STATE_DIR-Env > 'state'.
    """
    return config.state_dir(state_dir)


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

    Phase 1: Nur echte Trades (KAUFEN/VERKAUFEN) werden gewertet —
    HALTEN ist kein Trade und bleibt außen vor (konsistent mit der
    echten Hit-Rate aus evaluate.py).

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

    # Nur Zeilen mit confidence, final_decision und echter Trade-Aktion
    valid: list[dict[str, str]] = []
    for row in rows:
        action = (row.get("action") or "").strip().upper()
        if action not in ("KAUFEN", "VERKAUFEN"):
            continue  # HALTEN (und alles andere) ist kein Trade
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
    """Berechnet eine netzfreie Kalibrierungs-Näherung pro Aktion.

    Wie ``_compute_kalibrierung_proxy``, aber aufgespalten nach KAUFEN und
    VERKAUFEN (Phase 1: HALTEN ist kein Trade → kein Proxy-Segment).
    Nutzt ebenfalls die final_decision als Hit-Proxy.

    Returns:
        dict {aktion: {avg_confidence, genehmigungs_rate, gap, tendenz, n}}.
        Nur Aktionen mit ≥3 gültigen Zeilen. Leeres dict bei zu wenigen Daten.
    """
    # Zeilen nach Aktion gruppieren (nur echte Trades; HALTEN wird ausgeschlossen)
    per_action: dict[str, list[dict[str, str]]] = {
        "KAUFEN": [],
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

    avg_confidence = Ø über echte Trades (KAUFEN/VERKAUFEN), gewichtet nach n.
    HALTEN wird ignoriert (kein Trade → nicht in total_n/conf_sum).
    hit_rate = echte hit_rate_gesamt aus der JSON (bereits nur Trades).
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

    # Gewichtete Ø-Confidence über echte Trades (KAUFEN/VERKAUFEN);
    # HALTEN fließt nicht ein (Phase 1: HALTEN ist kein Trade).
    total_n = 0
    conf_sum = 0.0
    for action, adata in (cal.get("nach_aktion") or {}).items():
        if action not in ("KAUFEN", "VERKAUFEN"):
            continue  # HALTEN aus der Kalibrierung ausnehmen
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
    Nur echte Trades (KAUFEN/VERKAUFEN) mit n >= 3; HALTEN wird ignoriert
    (Phase 1: HALTEN ist kein Trade und bleibt außen vor).

    Returns:
        dict {aktion: {avg_confidence, hit_rate, gap, tendenz, n}}.
        Leeres dict bei fehlenden/ungültigen Daten. Crasht nie.
    """
    result: dict[str, dict[str, Any]] = {}
    for action in ("KAUFEN", "VERKAUFEN"):
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

    Phase 1: Die ``actions``-Zählung behält HALTEN (Transparenz im Report),
    aber die Kalibrierungs-Berechnung (echt wie Proxy) wertet nur echte
    Trades (KAUFEN/VERKAUFEN) — HALTEN ist kein Trade.
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
            f"Ø Confidence: {avg_conf} / 5 | Ø Ensemble-Confidence: {avg_ens}",
            f"KAUFEN-Empfehlungen final genehmigt: {kauf_pct} %",
            kalibrierung_line,
        ]

        # --- Kalibrierung pro Aktion (nur wenn Daten vorhanden) --- #
        # Phase 1: nur echte Trades (KAUFEN/VERKAUFEN) — HALTEN ist kein Trade.
        kal_pro_aktion = stats.get("kalibrierung_pro_aktion", {})
        if kal_pro_aktion:
            lines.append("")
            lines.append("Kalibrierung pro Aktion:")
            for action_name in ("KAUFEN", "VERKAUFEN"):
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


def _window_elapsed(decision_date: Any, lookback_days: int, *, today: Any = None) -> bool:
    """Prüft, ob das Ausgangsfenster einer Entscheidung vollständig abgelaufen ist.

    Look-ahead-frei (Roadmap C6): Eine Entscheidung liefert erst dann eine
    Reflexion/Lektion, wenn ``decision_date + lookback_days <= today`` gilt —
    also der komplette Ausgangszeitraum real vergangen ist und die Kurse für
    das volle Fenster existieren können.

    Args:
        decision_date: Journal-Timestamp (String) oder bereits geparstes
            datetime-Objekt. Nicht parsebare Werte → False (nie look-ahead).
        lookback_days: Zeitfenster in Tagen.
        today: Optionaler Stichtag (für Tests); Default ``datetime.now()``.

    Returns:
        True wenn das Fenster vollständig abgelaufen ist, sonst False.
        Crasht nie.
    """
    try:
        if isinstance(decision_date, datetime):
            parsed = decision_date
        else:
            parsed = _parse_timestamp(str(decision_date or ""))
        if parsed is None:
            return False
        days = max(0, int(lookback_days))
        deadline = parsed + timedelta(days=days)
        now = today if today is not None else datetime.now()
        return deadline <= now
    except Exception as exc:  # noqa: BLE001 — crasht nie
        logger.debug("Fenster-Check fehlgeschlagen: %s", exc)
        return False


def _status(row: dict[str, Any]) -> str:
    """Liest den reflection_status einer Journal-Zeile (normalisiert)."""
    status = (row.get("reflection_status") or "").strip().lower()
    if status in ("pending", "resolved"):
        return status
    return ""  # Legacy-Zeile (vor C6) oder unbekannter Wert


def _resolved_returns_from_row(row: dict[str, Any]) -> tuple[float | None, float | None]:
    """Extrahiert die persistierten Returns (realised_return_pct, alpha_pct).

    Gibt ``(None, None)`` zurück, wenn keine gültigen Werte in der Zeile
    stehen (z. B. Legacy-Zeile oder unvollständige Auflösung). Crasht nie.
    """
    raw = _safe_float(row.get("realised_return_pct"))
    alpha = _safe_float(row.get("alpha_pct"))
    return raw, alpha


def _lesson_from_returns(
    ticker: str,
    raw_return_pct: float | None,
    alpha_pct: float | None,
    action: str,
    ts: str,
    llm: LLMClient | None = None,
) -> str:
    """Erzeugt die Lektion (LLM oder deterministisch) für persistierte Returns.

    Analog zum Lektionsteil von ``build_reflection_context``: Bei übergebenem
    LLM wird ein Ein-Satz-Coach-Prompt gestellt, bei Fehlern/ohne LLM greift
    ``_deterministic_lesson``. Crasht nie.
    """
    lesson = _deterministic_lesson(raw_return_pct, action)
    if llm is None:
        return lesson
    try:
        alpha_str = (
            f"{alpha_pct:+.2f}%" if alpha_pct is not None and math.isfinite(alpha_pct) else "nicht verfügbar"
        )
        raw_str = (
            f"{raw_return_pct:+.2f}%"
            if raw_return_pct is not None and math.isfinite(raw_return_pct)
            else "nicht verfügbar"
        )
        prompt = (
            f"Du bist ein Trading-Coach. Formuliere EIN deutschen Satz als Lektion "
            f"aus einer vergangenen Handelsentscheidung.\n\n"
            f"Ticker: {ticker}\n"
            f"Aktion: {action}\n"
            f"Realisierter Return: {raw_str}\n"
            f"Alpha vs SPY: {alpha_str}\n\n"
            f"Antworte mit genau einem deutschen Satz (maximal 30 Wörter)."
        )
        messages = [
            {"role": "system", "content": "Du bist ein präziser Trading-Coach."},
            {"role": "user", "content": prompt},
        ]
        llm_lesson = llm.chat(messages, temperature=0.3)
        if llm_lesson and llm_lesson.strip():
            return llm_lesson.strip()
    except Exception as llm_exc:  # noqa: BLE001 — Fallback
        logger.debug(
            "LLM-Lektion fehlgeschlagen, verwende deterministische Lektion: %s", llm_exc
        )
    return lesson


def _format_reflection_text(
    ticker: str,
    raw_return_pct: float | None,
    alpha_pct: float | None,
    action: str,
    ts: str,
    lesson: str,
) -> str:
    """Formatiert den Reflexions-Block (gleiche Form wie bisher)."""
    alpha_str = (
        f"{alpha_pct:+.2f}%"
        if alpha_pct is not None and math.isfinite(alpha_pct)
        else "-"
    )
    raw_str = (
        f"{raw_return_pct:+.2f}%"
        if raw_return_pct is not None and math.isfinite(raw_return_pct)
        else "N/A"
    )
    return (
        f"=== DEINE LETZTE ENTSCHEIDUNG ZU {ticker.upper()} ({ts}) ===\n"
        f"Aktion: {action} | Realisierter Return: {raw_str} "
        f"| Alpha vs SPY: {alpha_str}\n"
        f"Lerne daraus: {lesson}"
    )


def _latest_pending_row(rows: list[dict[str, str]], ticker: str) -> dict[str, str] | None:
    """Findet die jüngste Zeile für den Ticker, die noch NICHT resolved ist.

    Case-insensitive Ticker-Matching; Zeilen mit ``reflection_status`` in
    ("", "pending") zählen als unaufgelöst. Rückgabe: die letzte passende
    Zeile (Annahme: chronologisch sortiert) oder None.
    """
    target = (ticker or "").strip().lower()
    for row in reversed(rows):
        row_ticker = (row.get("ticker") or "").strip().lower()
        if row_ticker != target:
            continue
        if _status(row) in ("", REFLECTION_STATUS_PENDING):
            return row
    return None


def resolve_pending_reflections(
    ticker: str,
    llm: LLMClient | None = None,
    lookback_days: int = 30,
    *,
    journal_file: str | None = None,
    _today: datetime | None = None,
) -> str:
    """Löst den jüngsten Pending-Eintrag eines Tickers auf, sobald das Ausgangs-
    fenster vollständig abgelaufen ist (Roadmap C6, look-ahead-frei).

    Ablauf:
      1. Journal lesen; jüngste Zeile des Tickers mit Status "" oder "pending"
         finden (case-insensitive). Keine → "".
      2. Prüfen, ob ``decision_date + lookback_days <= today`` gilt. Wenn das
         Fenster noch läuft → "" (KEIN Look-ahead: der Ausgang existiert noch
         nicht, also gibt es auch keine Reflexion).
      3. Realisierten Return (inkl. Alpha vs SPY) via
         ``evaluate.realised_return_for_row`` berechnen. None → "".
      4. Lektion generieren (LLM-Ein-Satz oder deterministischer Fallback).
      5. Auflösung atomar + lock-sicher ins Journal zurückschreiben:
         reflection_status="resolved", resolved_at=jetzt, realised_return_pct,
         alpha_pct. Bei Schreibfehlern wird trotzdem der Text zurückgegeben
         (nächster Lauf würde erneut resolven — idempotent, crasht nie).

    Args:
        ticker: Ticker-Symbol.
        llm: Optionaler LLMClient für die LLM-generierte Lektion.
        lookback_days: Zeitfenster für die Rendite-Berechnung (Default 30).
        journal_file: Optionaler Pfad zur Journal-CSV (für Tests). Default:
            journal/decisions.csv (relativ zum Arbeitsverzeichnis).
        _today: Optionaler Stichtag (nur für Tests, um Zeitabhängigkeit zu
            eliminieren). Default None = datetime.now().

    Returns:
        Der aufgelöste Reflexions-Text im Format von build_reflection_context
        oder "" wenn nichts auflösbar ist. Crasht niemals.
    """
    try:
        if journal_file is None:
            journal_file = os.path.join("journal", "decisions.csv")
        rows = _read_journal_rows(journal_file)
        if not rows:
            return ""

        row = _latest_pending_row(rows, ticker)
        if row is None:
            return ""

        ts = (row.get("timestamp") or "").strip()
        decision_date = _parse_timestamp(ts)
        if decision_date is None:
            return ""

        # Look-ahead-Schutz: Das Ausgangsfenster muss vollständig abgelaufen
        # sein, bevor der Return berechnet und die Lektion generiert wird.
        if not _window_elapsed(decision_date, lookback_days, today=_today):
            return ""

        # Realisierter Return (yfinance-basiert; im Offline-Test gemockt)
        rr = realised_return_for_row(row, lookback_days=lookback_days)
        if rr is None:
            return ""

        raw_return_pct = rr.get("raw_return_pct")
        alpha_pct = rr.get("alpha_pct")
        if raw_return_pct is None or not isinstance(raw_return_pct, (int, float)) \
                or not math.isfinite(raw_return_pct):
            return ""

        action = str(rr.get("action") or "")
        resolved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Lektion generieren (LLM oder deterministischer Fallback) — VOR dem
        # Zurückschreiben, damit sie mit persistiert wird.
        lesson = _lesson_from_returns(ticker, raw_return_pct, alpha_pct, action, ts, llm)

        # Auflösung ins Journal zurückschreiben (atomar, lock-sicher,
        # crasht nie — Erfolg ist keine Voraussetzung für den Rückgabewert).
        _write_resolution(
            journal_file,
            row,
            resolved_at=resolved_at,
            realised_return_pct=float(raw_return_pct),
            alpha_pct=(float(alpha_pct) if isinstance(alpha_pct, (int, float)) and math.isfinite(alpha_pct) else None),
            lesson=lesson,
        )

        return _format_reflection_text(
            ticker, raw_return_pct, alpha_pct, action, ts, lesson
        )
    except Exception as exc:  # noqa: BLE001 — crasht nie
        logger.warning("Pending-Reflexion konnte nicht aufgelöst werden: %s", exc)
        return ""


def _write_resolution(
    journal_file: str,
    target_row: dict[str, Any],
    *,
    resolved_at: str,
    realised_return_pct: float,
    alpha_pct: float | None,
    lesson: str = "",
) -> bool:
    """Schreibt die Pending-Auflösung atomar + lock-sicher ins Journal.

    Liest alle Zeilen, ersetzt die Zielfeile (Identifikation über ticker +
    timestamp), setzt reflection_status/resolved_at/realised_return_pct/
    alpha_pct und schreibt die Datei neu (tmpfile + os.replace, fcntl-Lock
    analog journal.py). Gibt True bei Erfolg zurück, False bei Fehlern
    (crasht nie — die Zeile bleibt in dem Fall unresolve't und würde beim
    nächsten Lauf erneut resolviert — idempotent).
    """
    tmp_path = ""
    try:
        with open(journal_file, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or JOURNAL_HEADER)
            rows = list(reader)

        # C6-Spalten im Header ergänzen (falls die Datei noch vor C6 angelegt
        # wurde — die append_decision-Migration deckt nur den append-Pfad ab).
        c6_cols = (
            "reflection_status",
            "resolved_at",
            "realised_return_pct",
            "alpha_pct",
            "lesson",
        )
        for col in c6_cols:
            if col not in fieldnames:
                fieldnames.append(col)

        target_ticker = (target_row.get("ticker") or "").strip().lower()
        target_ts = (target_row.get("timestamp") or "").strip()
        replaced = False
        out_rows: list[dict[str, str]] = []
        for row in rows:
            row_ticker = (row.get("ticker") or "").strip().lower()
            row_ts = (row.get("timestamp") or "").strip()
            if (
                not replaced
                and row_ticker == target_ticker
                and row_ts == target_ts
                and _status(row) in ("", REFLECTION_STATUS_PENDING)
            ):
                row = dict(row)
                row["reflection_status"] = REFLECTION_STATUS_RESOLVED
                row["resolved_at"] = resolved_at
                row["realised_return_pct"] = f"{realised_return_pct:+.2f}"
                row["alpha_pct"] = f"{alpha_pct:+.2f}" if alpha_pct is not None else ""
                row["lesson"] = str(lesson or "") if lesson else ""
                replaced = True
            out_rows.append(row)

        if not replaced:
            return False

        # Atomar schreiben: erst in tmpfile, dann os.replace. Auf der
        # ZIELDATEI wird ein exklusiver fcntl-Lock gehalten (analog
        # _rewrite_journal_with_header in journal.py), damit parallele
        # append_decision-Schreiber nicht kollidieren.
        tmp_path = f"{journal_file}.tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in out_rows:
                for field in fieldnames:
                    if field not in row:
                        row[field] = ""
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        try:
            with open(journal_file, encoding="utf-8") as lock_fh:
                _acquire_lock(lock_fh)
                try:
                    os.replace(tmp_path, journal_file)
                finally:
                    _release_lock(lock_fh)
        except OSError:
            # Lock auf der Zieldatei nicht möglich → replace ohne Lock (best effort)
            os.replace(tmp_path, journal_file)
        return True
    except Exception as exc:  # noqa: BLE001 — crasht nie
        logger.warning("Auflösung konnte nicht ins Journal geschrieben werden: %s", exc)
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:  # noqa: BLE001
            pass
        return False


def build_reflection_context(
    ticker: str,
    llm: LLMClient | None = None,
    lookback_days: int = 30,
) -> str:
    """Baut einen Reflexions-Kontext-Block für den letzten Entscheidungs-Eintrag eines Tickers.

    Look-ahead-frei (Roadmap C6): Reflexionen entstehen NUR aus
    Entscheidungen, deren Ausgangsfenster vollständig abgelaufen ist
    (``decision_date + lookback_days <= today``). Solange das Fenster läuft,
    gibt es KEINE Reflexion — der Ausgang existiert zum Zeitpunkt der
    Reflexions-Generierung sonst nur in der Zukunft (Look-ahead-Bias).

    Es gibt zwei Pfade:
      - ``reflection_status="resolved"``: Der Return wurde beim Resolving
        (resolve_pending_reflections) bereits berechnet und im Journal
        persistiert (realised_return_pct/alpha_pct) → NUR diese gespeicherten
        Werte werden verwendet, KEIN neuer realised_return_for_row-Aufruf.
      - Legacy-Zeile ("" bzw. "pending"): Erst den vollständigen Ablauf des
        Fensters prüfen; nur dann via evaluate.realised_return_for_row
        berechnen (Legacy-Pfad für Zeilen vor C6 bzw. noch nicht resolvierte
        Einträge).

    Wenn ein LLMClient übergeben wird, wird die "Lektion" vom LLM generiert
    (einziger Satz). Bei Fehlern oder ohne LLM wird eine deterministische
    Lektion verwendet.

    Args:
        ticker: Ticker-Symbol.
        llm: Optionaler LLMClient für die LLM-generierte Lektion.
        lookback_days: Zeitfenster für die Rendite-Berechnung (Default 30).

    Returns:
        Deutscher Reflexions-String oder "" wenn kein Eintrag/Fehler/Fenster
        nicht abgelaufen. Crasht niemals.
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
        ts = (row.get("timestamp") or "").strip()
        action = (row.get("action") or "").strip().upper()
        status = _status(row)

        if status == REFLECTION_STATUS_RESOLVED:
            # Resolved: gespeicherte Returns verwenden — kein Netz-Zugriff,
            # kein Look-ahead (die Auflösung erfolgte erst nach Ablauf).
            raw_return_pct, alpha_pct = _resolved_returns_from_row(row)
            # Persistierte Lektion wiederverwenden (nicht bei jedem Lauf neu
            # berechnen — Kern des C6-Persistenz-Gedankens).
            lesson = (row.get("lesson") or "").strip()
            if raw_return_pct is None or not math.isfinite(raw_return_pct):
                # Unvollständige Auflösung (Datenanomalie) → Legacy-Pfad,
                # aber weiterhin nur bei abgelaufenem Fenster.
                if not _window_elapsed(ts, lookback_days):
                    return ""
                rr = realised_return_for_row(row, lookback_days=lookback_days)
                if rr is None:
                    return ""
                raw_return_pct = rr.get("raw_return_pct")
                alpha_pct = rr.get("alpha_pct")
                if raw_return_pct is None:
                    return ""
            if not lesson:
                lesson = _lesson_from_returns(
                    ticker, raw_return_pct, alpha_pct, action, ts, llm
                )
            return _format_reflection_text(
                ticker, raw_return_pct, alpha_pct, action, ts, lesson
            )

        # Legacy- oder Pending-Zeile: KEIN Look-ahead — Reflexion nur,
        # wenn das Ausgangsfenster vollständig abgelaufen ist.
        if not _window_elapsed(ts, lookback_days):
            return ""
        rr = realised_return_for_row(row, lookback_days=lookback_days)
        if rr is None:
            return ""
        raw_return_pct = rr.get("raw_return_pct")
        alpha_pct = rr.get("alpha_pct")
        if raw_return_pct is None:
            return ""

        # Lektion generieren (LLM oder deterministischer Fallback)
        lesson = _lesson_from_returns(
            ticker, raw_return_pct, alpha_pct, action, ts, llm
        )

        return _format_reflection_text(
            ticker, raw_return_pct, alpha_pct, action, ts, lesson
        )
    except Exception as exc:  # noqa: BLE001 — crasht nie
        logger.warning("Reflexions-Kontext konnte nicht erstellt werden: %s", exc)
        return ""


# --------------------------------------------------------------------------- #
# Cross-Ticker-Gedächtnis — generalisierte Lektionen aus anderen Tickern (C4)
# --------------------------------------------------------------------------- #


def _deterministic_cross_ticker_lesson(lessons: list[dict[str, Any]]) -> str:
    """Erzeugt eine deterministische generalisierte Lektion aus Cross-Ticker-Lektionen.

    Analog ``_deterministic_lesson``, aber aggregiert über mehrere Lektionen:
    Das Vorzeichen des DURCHSCHNITTLICHEN realisierten Returns bestimmt die
    Tendenz. Erwartet Zeilen mit garantiert gültigem (nicht-None, finite)
    ``raw_return_pct`` (so wird die Liste von build_cross_ticker_context gefüllt).
    """
    valid = [
        _safe_float(entry.get("raw_return_pct"))
        for entry in lessons
        if isinstance(entry, dict)
    ]
    valid = [v for v in valid if v is not None and math.isfinite(v)]
    if not valid:
        return (
            "Keine aussagekräftigen Cross-Ticker-Renditen verfügbar — "
            "prüfe deine Annahmen sorgfältig."
        )
    avg = sum(valid) / len(valid)
    if avg > 0.5:
        return (
            "Die letzten Entscheidungen anderer Ticker liefen insgesamt positiv — "
            "behalte deine Methodik bei, prüfe aber, ob sich diese Marktlage "
            "auf den aktuellen Ticker überträgt."
        )
    if avg < -0.5:
        return (
            "Die letzten Entscheidungen anderer Ticker liefen insgesamt negativ — "
            "prüfe, ob sich Sektor- oder Marktrisiken auch auf den aktuellen "
            "Ticker übertragen, und sei vorsichtiger mit Timing und Zielsetzung."
        )
    return (
        "Die letzten Entscheidungen anderer Ticker blieben insgesamt neutral — "
        "übertrage deren Muster nur bei klarer Übereinstimmung mit dem "
        "aktuellen Ticker."
    )


def build_cross_ticker_context(
    ticker: str,
    llm: LLMClient | None = None,
    lookback_days: int = 30,
    max_lessons: int = 3,
) -> str:
    """Baut einen Kontext-Block mit den letzten Entscheidungen ANDERER Ticker (C4).

    Liest das Entscheidungs-Journal, findet die jüngsten Entscheidungen
    ANDERER Ticker (case-insensitive, neueste zuerst), berechnet für jede
    den realisierten Return via evaluate.realised_return_for_row und nimmt
    die ersten ``max_lessons`` Zeilen mit verfügbarem Return.

    Wenn ein LLMClient übergeben wird, wird EIN deutscher Satz als
    generalisierte Lektion über alle Cross-Ticker-Lektionen generiert
    (analog build_reflection_context). Bei Fehlern oder ohne LLM wird eine
    deterministische Lektion verwendet.

    Args:
        ticker: Ticker-Symbol (dessen EIGENE Entscheidungen werden ausgeschlossen).
        llm: Optionaler LLMClient für die generalisierte Lektion.
        lookback_days: Zeitfenster für die Rendite-Berechnung (Default 30).
        max_lessons: Maximale Anzahl Cross-Ticker-Lektionen (Default 3) —
            hält den Block kompakt, damit die Prompts nicht aufblähen.

    Returns:
        Deutscher Kontext-String oder "" bei leerem/fehlerhaftem Journal,
        keinen Cross-Ticker-Zeilen oder keinen berechenbaren Returns.
        Crasht niemals.
    """
    try:
        if not (ticker or "").strip():
            return ""

        journal_file = os.path.join("journal", "decisions.csv")
        rows = _read_journal_rows(journal_file)
        if not rows:
            return ""

        target = (ticker or "").strip().lower()

        # Cross-Ticker-Zeilen: anderer Ticker (case-insensitive) mit parsebarem
        # Timestamp. Zeilen ohne Timestamp sind für die Chronologie unbrauchbar.
        cross_rows: list[dict[str, str]] = []
        for row in rows:
            row_ticker = (row.get("ticker") or "").strip().lower()
            if not row_ticker or row_ticker == target:
                continue
            ts = (row.get("timestamp") or "").strip()
            if ts:
                cross_rows.append(row)

        if not cross_rows:
            return ""

        # Neueste zuerst: Journal-Timestamps sind ISO-ähnlich
        # ('YYYY-MM-DD HH:MM:SS') → lexikalische Sortierung = chronologisch.
        cross_rows.sort(key=lambda r: (r.get("timestamp") or "").strip(), reverse=True)

        # Erste max_lessons Zeilen MIT berechenbarem Return sammeln
        # (Zeilen ohne realisierbaren Return werden übersprungen).
        #
        # Look-ahead-frei (Roadmap C6): Nur Entscheidungen mit VOLLSTÄNDIG
        # abgelaufenem Ausgangsfenster (decision_date + lookback_days <= today)
        # liefern Cross-Ticker-Lektionen. Resolved-Zeilen nutzen die
        # persistierten Returns (kein neuer realised_return_for_row-Aufruf);
        # Legacy-/Pending-Zeilen werden nur bei abgelaufenem Fenster über
        # realised_return_for_row aufgelöst. Nicht abgelaufene Fenster werden
        # übersprungen — auch dann, wenn sie "neuere" Timestamps haben.
        lessons: list[dict[str, Any]] = []
        budget = max(0, max_lessons)
        for row in cross_rows:
            if len(lessons) >= budget:
                break
            status = _status(row)
            if status == REFLECTION_STATUS_RESOLVED:
                raw, alpha = _resolved_returns_from_row(row)
                if raw is None or not math.isfinite(raw):
                    # Unvollständige Auflösung → Legacy-Pfad mit Fenster-Check
                    if not _window_elapsed(row.get("timestamp", ""), lookback_days):
                        continue
                    rr = realised_return_for_row(row, lookback_days=lookback_days)
                    if rr is None:
                        continue
                    raw = rr.get("raw_return_pct")
                    alpha = rr.get("alpha_pct")
                    if raw is None or not math.isfinite(raw):
                        continue
                    lessons.append(rr)
                    continue
                lessons.append({
                    "ticker": row.get("ticker", ""),
                    "raw_return_pct": raw,
                    "spy_return_pct": None,
                    "alpha_pct": alpha,
                    "timestamp": (row.get("timestamp") or "").strip(),
                    "action": (row.get("action") or "").strip().upper(),
                })
                continue

            # Legacy-/Pending-Zeile: Look-ahead-Schutz — Fenster muss
            # vollständig abgelaufen sein, bevor der Return berechnet wird.
            if not _window_elapsed(row.get("timestamp", ""), lookback_days):
                continue
            rr = realised_return_for_row(row, lookback_days=lookback_days)
            if rr is None:
                continue
            raw = rr.get("raw_return_pct")
            if raw is None or not isinstance(raw, (int, float)) or not math.isfinite(raw):
                continue
            lessons.append(rr)

        if not lessons:
            return ""

        # Generalisierte Lektion: LLM (ein Satz über alle Lektionen) oder
        # deterministischer Fallback.
        lesson = _deterministic_cross_ticker_lesson(lessons)
        if llm is not None:
            try:
                lehrzeilen = []
                for rr in lessons:
                    alpha = rr.get("alpha_pct")
                    alpha_str = (
                        f"{alpha:+.2f}%"
                        if alpha is not None and math.isfinite(alpha)
                        else "nicht verfügbar"
                    )
                    lehrzeilen.append(
                        f"- {rr.get('ticker', '?')} ({rr.get('timestamp', '')}): "
                        f"Aktion {rr.get('action', '')}, realisierter Return "
                        f"{rr.get('raw_return_pct'):+.2f}%, Alpha vs SPY {alpha_str}"
                    )
                prompt = (
                    "Du bist ein Trading-Coach. Formuliere EINEN deutschen Satz als "
                    "generalisierte Lektion aus den letzten Entscheidungen ANDERER "
                    "Ticker (Cross-Ticker-Gedächtnis). Erkenne übergreifende Muster "
                    "(z. B. Sektor-Trends) statt Ticker-Details.\n\n"
                    + "\n".join(lehrzeilen)
                    + "\n\nAntworte mit genau einem deutschen Satz (maximal 30 Wörter)."
                )
                messages = [
                    {"role": "system", "content": "Du bist ein präziser Trading-Coach."},
                    {"role": "user", "content": prompt},
                ]
                llm_lesson = llm.chat(messages, temperature=0.3)
                if llm_lesson and llm_lesson.strip():
                    lesson = llm_lesson.strip()
            except Exception as llm_exc:  # noqa: BLE001 — Fallback
                logger.debug(
                    "LLM-Cross-Ticker-Lektion fehlgeschlagen, verwende "
                    "deterministische Lektion: %s",
                    llm_exc,
                )

        lines = [
            "=== LETZTE ENTSCHEIDUNGEN ANDERER TICKER (Cross-Ticker-Gedächtnis) ==="
        ]
        for rr in lessons:
            raw = rr.get("raw_return_pct")
            raw_str = (
                f"{raw:+.2f}%"
                if isinstance(raw, (int, float)) and math.isfinite(raw)
                else "N/A"
            )
            alpha = rr.get("alpha_pct")
            alpha_str = (
                f"{alpha:+.2f}%"
                if alpha is not None and math.isfinite(alpha)
                else "-"
            )
            lines.append(
                f"- {str(rr.get('ticker') or '').upper()} ({rr.get('timestamp', '')}): "
                f"Aktion {rr.get('action', '')} | Realisierter Return {raw_str} "
                f"| Alpha vs SPY {alpha_str}"
            )
        lines.append(f"Lektion: {lesson}")
        lines.append(
            "Lerne aus diesen generalisierten Mustern — übertrage sie aber nur, "
            "wenn sie auf den aktuellen Ticker passen."
        )
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 — crasht nie
        logger.warning("Cross-Ticker-Kontext konnte nicht erstellt werden: %s", exc)
        return ""


def build_memory_context(
    ticker: str,
    llm: LLMClient | None = None,
    lookback_days: int = 30,
    max_same: int = 1,
    max_cross: int = 3,
) -> str:
    """Orchestriert Same-Ticker-Reflexion + Cross-Ticker-Lektionen (C4).

    Kombiniert die bestehende Ticker-spezifische Reflexion
    (``build_reflection_context`` — die letzte Entscheidung DESSELBEN
    Tickers) mit generalisierten Cross-Ticker-Lektionen
    (``build_cross_ticker_context`` — die jüngsten Entscheidungen ANDERER
    Ticker mit realisiertem Return) zu EINEM Kontext-Block für die
    Trader-/Ensemble-Trader-/Risk-/PM-Prompts.

    Args:
        ticker: Ticker-Symbol.
        llm: Optionaler LLMClient für die LLM-generierten Lektionen.
        lookback_days: Zeitfenster für die Rendite-Berechnung (Default 30).
        max_same: Wenn <= 0, wird der Same-Ticker-Teil übersprungen
            (Default 1 = aktiv; die Reflexion ist ein einzelner Block).
        max_cross: Maximale Anzahl Cross-Ticker-Lektionen (Default 3).

    Returns:
        Kombinierter deutscher Kontext-String (Teile durch Leerzeile
        getrennt) oder "" wenn beide Teile leer sind. Crasht niemals.
    """
    try:
        parts: list[str] = []

        if max_same > 0:
            same = build_reflection_context(
                ticker, llm=llm, lookback_days=lookback_days
            )
            if same:
                parts.append(same)

        cross = build_cross_ticker_context(
            ticker,
            llm=llm,
            lookback_days=lookback_days,
            max_lessons=max_cross,
        )
        if cross:
            parts.append(cross)

        return "\n\n".join(parts)
    except Exception as exc:  # noqa: BLE001 — crasht nie
        logger.warning("Memory-Kontext konnte nicht erstellt werden: %s", exc)
        return ""
