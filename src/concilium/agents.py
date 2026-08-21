"""Agenten-Modul — spezialisierte LLM-Rollen-Aufrufe für die Trading-Pipeline.

Jede Rolle ist ein strukturierter LLM-Call mit deutschen Prompts.
Agenten liefern Stimmung (bullish/neutral/bearish) + Score (1-5).
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from typing import Any

from .factors import compute_multi_factor_score
from .llm import LLMClient

logger = logging.getLogger(__name__)

# Maximale Anzahl paralleler Threads für unabhängige LLM-Calls
_MAX_PARALLEL = 3

# ---------------------------------------------------------------------------
# Prompt-Templates (alle auf Deutsch)
# ---------------------------------------------------------------------------

SYSTEM_FUNDAMENTAL = """\
Du bist ein erfahrener Fundamental-Analyst. Du analysierst Aktien basierend auf
unternehmensbezogenen Kennzahlen: Marktkapitalisierung, KGV, EPS, Umsatz, Wachstumsraten,
Gewinnmargen, PEG, Dividendenrendite und 52-Wochen-Hoch/Tief.

Bewerte die fundamentals und gib deine Einschätzung ab.
Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{
  "rolle": "Fundamental-Analyst",
  "stimmung": "bullish" | "neutral" | "bearish",
  "score": 1-5,
  "zusammenfassung": "2-4 Sätze Zusammenfassung auf Deutsch",
  "kennzahlen_bewertung": "Kurze Bewertung der wichtigsten Kennzahlen"
}
"""

SYSTEM_TECHNICAL = """\
Du bist ein erfahrener technischer Analyst. Du analysierst Charts und Indikatoren:
SMA50, SMA200, RSI(14), MACD, Bollinger-Bänder, Volumen.

Gib an, ob der Trend aufwärts, seitwärts oder abwärts gerichtet ist und ob Überkauft-/\
Überverkauft-Signale vorliegen.
Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{
  "rolle": "Technik-Analyst",
  "stimmung": "bullish" | "neutral" | "bearish",
  "score": 1-5,
  "zusammenfassung": "2-4 Sätze Zusammenfassung auf Deutsch",
  "trend": "aufwärts" | "seitwärts" | "abwärts",
  "signale": "Wichtigste technische Signale"
}
"""

SYSTEM_SENTIMENT = """\
Du bist ein Sentiment-Analyst. Du bewertest Nachrichten-Headlines zu einer Aktie.
Du erhältst eine Liste von Headlines und eine einfache Positiv/Negativ/Neutral-Zählung.

Bewerte das Markt-Sentiment und ob es kauf- oder verkaufsfördernd ist.
Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{
  "rolle": "Sentiment-Analyst",
  "stimmung": "bullish" | "neutral" | "bearish",
  "score": 1-5,
  "zusammenfassung": "2-4 Sätze Zusammenfassung auf Deutsch",
  "dominant": "positiv" | "negativ" | "neutral"
}
"""

SYSTEM_BULL = """\
Du bist ein Bull-Stratege (Bull-Befürworter). Du argumentierst für eine Long-Position \
in der Aktie. Nutze die Analysten-Einschätzungen, um die stärksten Argumente für \
einen Kauf zu sammeln.

Antworte auf Deutsch in 3-6 Sätzen. Formuliere überzeugend, aber sachlich. \
Gib am Anfang ein JSON-Block mit confidence (1-5) und einem Kurznamen an:
{"confidence": 1-5, "name": "Bull-Argumentation"}
Danach folgt dein Fließtext.
"""

SYSTEM_BEAR = """\
Du bist ein Bear-Stratege (Bear-Kritiker). Du argumentierst gegen eine Long-Position \
in der Aktie. Nutze die Analysten-Einschätzungen, um die größten Risiken und Gegenargumente \
zu sammeln.

Antworte auf Deutsch in 3-6 Sätzen. Formuliere überzeugend, aber sachlich. \
Gib am Anfang ein JSON-Block mit confidence (1-5) und einem Kurznamen an:
{"confidence": 1-5, "name": "Bear-Argumentation"}
Danach folgt dein Fließtext.
"""

SYSTEM_TRADER = """\
Du bist ein professioneller Trader. Basierend auf den Analysten-Einschätzungen und \
der Bull/Bear-Debatte erstellst du einen konkreten Trade-Vorschlag.

Nutze die volle 5-stufige Skala. 'STARK KAUFEN'/'STARK VERKAUFEN' nur bei hoher \
Überzeugung (sehr klare Fundamental- und/oder technische Signale). Bei Unsicherheit \
nimm 'KAUFEN'/'VERKAUFEN' bzw. 'HALTEN'.

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{
  "rolle": "Trader",
  "aktion": "STARK KAUFEN" | "KAUFEN" | "HALTEN" | "VERKAUFEN" | "STARK VERKAUFEN",
  "zielkurs": "Zielkurs als Zahl oder null",
  "stop_loss": "Stop-Loss als Zahl oder null",
  "positionsanteil": "Empfohlener Positionsanteil in % (z.B. 5)",
  "begründung": "2-4 Sätze Begründung auf Deutsch",
  "zeithorizont": "Kurzfristig" | "Mittelfristig" | "Langfristig"
}
"""

# 5-stufige Rating-Skala (von bullisch zu bearisch)
RATING_5 = ["STARK KAUFEN", "KAUFEN", "HALTEN", "VERKAUFEN", "STARK VERKAUFEN"]


def _rating_to_action(rating: str) -> str:
    """Mapt eine 5-stufige Bewertung auf die 3-stufige Aktion (Rückwärtskompatibilität).

    STARK KAUFEN/KAUFEN -> KAUFEN; HALTEN -> HALTEN;
    VERKAUFEN/STARK VERKAUFEN -> VERKAUFEN.
    Unbekannt/leer -> HALTEN.
    """
    r = (rating or "").strip().upper()
    if r in ("STARK KAUFEN", "KAUFEN"):
        return "KAUFEN"
    if r == "HALTEN":
        return "HALTEN"
    if r in ("VERKAUFEN", "STARK VERKAUFEN"):
        return "VERKAUFEN"
    return "HALTEN"

SYSTEM_RISK = """\
Du bist ein Risk-Manager. Du bewertest das Risiko eines vorgeschlagenen Trades \
basierend auf Volatilität (Beta), historischem Drawdown, Marktbedingungen und \
Positionsgrösse. Du kannst den Trade ablehnen oder modifizieren.

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{
  "rolle": "Risk-Manager",
  "risiko_score": 1-5 (1=niedrig, 5=sehr hoch),
  "volatilität_bewertung": "Kurze Bewertung",
  "max_drawdown_schaetzung": "Geschätzter Max-Drawdown in %",
  "positionsgröße_empfohlen": "Empfohlene Positionsgrösse in %",
  "auflagen": "Auflagen oder Bedingungen, oder 'keine'",
  "empfehlung": "GENEHMIGT" | "MODIFIZIERT" | "ABGELEHNT"
}
"""

SYSTEM_TRADE_REVISION = """\
Du bist der Trader in der zweiten Runde. Dein ursprünglicher Trade wurde vom \
Risk-Manager und Portfolio-Fit-Analysten bewertet. Passe deinen Trade an: Du \
darfst Aktion, Zielkurs, Stop-Loss und Positionsanteil ändern, wenn die \
Risiko-/Portfolio-Einwände begründet sind. Bleib konsistent mit deinen \
Kernargumenten.

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{
  "rolle": "Trader",
  "aktion": "STARK KAUFEN" | "KAUFEN" | "HALTEN" | "VERKAUFEN" | "STARK VERKAUFEN",
  "zielkurs": "Zielkurs als Zahl oder null",
  "stop_loss": "Stop-Loss als Zahl oder null",
  "positionsanteil": "Empfohlener Positionsanteil in % (z.B. 5)",
  "begründung": "2-4 Sätze Begründung auf Deutsch",
  "zeithorizont": "Kurzfristig" | "Mittelfristig" | "Langfristig"
}
"""

SYSTEM_PM = """\
Du bist der Portfolio-Manager. Du triffst die finale Entscheidung über den Trade, \
basierend auf dem Trade-Vorschlag und der Risiko-Bewertung. Du kannst den Trade \
genehmigen, mit Auflagen modifizieren oder ablehnen.

- GENEHMIGT: Trade wie vorgeschlagen genehmigen.
- MODIFIZIERT: Trade grundsätzlich genehmigen, aber mit klaren Auflagen/Bedingungen \
(z.B. kleinere Position, Zeitfenster).
- ABGELEHNT: Trade ablehnen.

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{
  "rolle": "Portfolio-Manager",
  "entscheidung": "GENEHMIGT" | "MODIFIZIERT" | "ABGELEHNT",
  "begründung": "2-4 Sätze Begründung auf Deutsch",
  "confidence": 1-5
}
"""

# ---------------------------------------------------------------------------
# JSON-Parsing-Helper (tolerant)
# ---------------------------------------------------------------------------


def parse_json(text: str) -> dict[str, Any]:
    """Versucht, JSON aus einem LLM-Text zu extrahieren (tolerant).

    Versucht zuerst den ganzen Text, dann den ersten JSON-Block, dann Code-Blöcke.
    Gibt bei Misserfolg ein dict mit rohem Text zurück.
    """
    if not text:
        return {"_raw": ""}

    # 1. Direkter Versuch
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Erster {...}-Block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Code-Block ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    return {"_raw": text}


# ---------------------------------------------------------------------------
# Daten-Vorbereitung für Prompts
# ---------------------------------------------------------------------------


def _fmt_num(val: Any, suffix: str = "") -> str:
    """Formatiert eine Zahl lesbar oder gibt 'N/A' zurück."""
    if val is None:
        return "N/A"
    try:
        fval = float(val)
        if abs(fval) >= 1e12:
            return f"{fval / 1e12:.2f} Bio{suffix}"
        if abs(fval) >= 1e9:
            return f"{fval / 1e9:.2f} Mrd{suffix}"
        if abs(fval) >= 1e6:
            return f"{fval / 1e6:.2f} Mio{suffix}"
        if abs(fval) >= 1e3:
            return f"{fval / 1e3:.2f} K{suffix}"
        return f"{fval:.2f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def _build_data_text(data: dict[str, Any], role: str = "alle") -> str:
    """Erstellt einen kompakten deutschen Daten-Text für LLM-Prompts.

    Args:
        data: Daten-dict aus collect_ticker_data.
        role: Rollenspezifische Filterung der Daten-Sektionen.
            - ``"alle"`` (Default): alle Sektionen (rückwärtskompatibel).
            - ``"fundamental"``: Aktien-Identität, Datenqualitäts-Warnungen,
              FUNDAMENTALS, Analysten-Erwartungen, Makro/Zinsen, Peer-Vergleich.
              Keine TECHNIK- oder SENTIMENT-Sektion.
            - ``"technik"``: Aktien-Identität, TECHNIK-Sektion, aktueller Kurs,
              Makro-Zinstrend (kurz). Keine FUNDAMENTALS- oder SENTIMENT-Sektion.
            - ``"sentiment"``: Aktien-Identität, SENTIMENT-Sektion, Headlines.
              Keine FUNDAMENTALS- oder TECHNIK-Sektion.
    """
    f = data.get("fundamentals", {})
    t = data.get("technicals", {})
    s = data.get("sentiment", {})
    news = data.get("news", [])
    macro = data.get("macro", {})
    peers = data.get("peers", [])

    # Aktien-Identität — immer enthalten
    lines = [
        f"Aktie: {data.get('ticker', '?')} ({f.get('name', 'N/A')})",
        f"Sektor: {f.get('sector', 'N/A')} / {f.get('industry', 'N/A')}",
    ]

    # Datenqualitäts-Warnungen — für fundamental und alle
    if role in ("alle", "fundamental"):
        data_warnings = data.get("data_warnings", [])
        if data_warnings:
            lines.append("")
            lines.append("=== DATENQUALITÄTS-WARNUNGEN ===")
            lines.append(
                "  Die folgenden Kennzahlen sind möglicherweise unzuverlässig "
                "(ADR-Fehler, Datenfehler). Werte weiterhin anzeigen, aber kritisch bewerten:"
            )
            for w in data_warnings:
                lines.append(f"  - {w}")

    # FUNDAMENTALS — für fundamental und alle
    if role in ("alle", "fundamental"):
        lines.extend([
            "",
            "=== FUNDAMENTALS ===",
            f"  Marktkapitalisierung: {_fmt_num(f.get('market_cap'), ' ')}",
            f"  KGV (trailing): {_fmt_num(f.get('pe_ratio'))}",
            f"  EPS: {_fmt_num(f.get('eps'))}",
            f"  Umsatz: {_fmt_num(f.get('revenue'), ' ')}",
            f"  Umsatzwachstum: {_fmt_num(f.get('revenue_growth'))} (als Anteil)",
            f"  Gewinnmarge: {_fmt_num(f.get('profit_margin'))}",
            f"  PEG: {_fmt_num(f.get('peg_ratio'))}",
            f"  Dividendenrendite: {_fmt_num(f.get('dividend_yield'))}",
            f"  Beta: {_fmt_num(f.get('beta'))}",
            f"  52W Hoch: {_fmt_num(f.get('fifty_two_week_high'))}",
            f"  52W Tief: {_fmt_num(f.get('fifty_two_week_low'))}",
            # Feature 1: Analysten-Erwartungen
            f"  Analysten-Konsens: {f.get('recommendation_key', 'N/A')}",
            f"  Analysten-Mean (Skala 1=strong buy … 5=sell): {_fmt_num(f.get('recommendation_mean'))}",
            f"  Anzahl Analysten: {f.get('analyst_count', 'N/A')}",
            f"  Zielkurs Ø: {_fmt_num(f.get('analyst_target_mean'))}",
            f"  Zielkurs hoch: {_fmt_num(f.get('analyst_target_high'))}",
            f"  Zielkurs tief: {_fmt_num(f.get('analyst_target_low'))}",
            f"  Upside (geschätzt): {_fmt_num(f.get('analyst_upside_pct'))} %",
        ])
        # Quantitativer Multi-Faktor-Score-Anker (deterministischer Referenzwert)
        mf = compute_multi_factor_score(f)
        if mf.get("overall_score") is not None:
            lines.append(
                "  Quant-Score (deterministisch, Referenz): "
                f"Value {_fmt_num(mf.get('value_score'))}/5, "
                f"Momentum {_fmt_num(mf.get('momentum_score'))}/5, "
                f"Qualität {_fmt_num(mf.get('quality_score'))}/5, "
                f"Gesamt {_fmt_num(mf.get('overall_score'))}/5 — {mf.get('kurzeinschaetzung', '')}"
            )
            lines.append(
                "  → Anker nur zur Einordnung: stimme NICHT blind zu, "
                "hinterfrage ihn kritisch anhand der Kennzahlen."
            )

    # TECHNIK — für technik und alle
    if role in ("alle", "technik"):
        lines.extend([
            "",
            "=== TECHNIK ===",
            f"  Aktueller Kurs: {_fmt_num(t.get('current_price'))}",
            f"  SMA50: {_fmt_num(t.get('sma50'))}",
            f"  SMA200: {_fmt_num(t.get('sma200'))}",
            f"  RSI(14): {_fmt_num(t.get('rsi14'))}",
            f"  MACD: {_fmt_num(t.get('macd', {}).get('macd'))} / Signal: {_fmt_num(t.get('macd', {}).get('signal'))}",
            f"  Bollinger: Unter {_fmt_num(t.get('bollinger', {}).get('lower'))} / Mitte {_fmt_num(t.get('bollinger', {}).get('middle'))} / Ober {_fmt_num(t.get('bollinger', {}).get('upper'))}",
            f"  Bollinger-Position: {_fmt_num(t.get('bollinger', {}).get('position'))} (0=unteres Band, 1=oberes Band)",
            f"  Volumen: {_fmt_num(t.get('current_volume'), ' ')}",
            f"  Ø Volumen 30T: {_fmt_num(t.get('avg_volume_30d'), ' ')}",
        ])

    # Makro/Zins-Daten — für alle (vollständig), fundamental (vollständig),
    # technik (nur Zinstrend-Kurzform), sentiment (keine)
    if macro:
        if role in ("alle", "fundamental"):
            lines.append("")
            lines.append("=== MAKRO / ZINSEN ===")
            lines.append(f"  10y US Treasury Yield: {_fmt_num(macro.get('us_10y_yield'))} %")
            lines.append(f"  10y Yield vor 1 Monat: {_fmt_num(macro.get('us_10y_yield_1m_ago'))} %")
            lines.append(f"  10y Zinstrend: {macro.get('us_10y_trend', 'N/A')}")
            source = macro.get("sp500_source", "")
            source_label = f" ({source})" if source and source != "none" else ""
            lines.append(f"  S&P 500 KGV{source_label}: {_fmt_num(macro.get('sp500_pe'))}")
            lines.append(f"  S&P 500 Marktkap: {_fmt_num(macro.get('sp500_market_cap'), ' ')}")
            lines.append(
                "  Hinweis: Hohe/steigende Zinsen belasten kapitalintensive "
                "und erneuerbare Sektoren."
            )
        elif role == "technik":
            lines.append("")
            lines.append("=== MAKRO / ZINSEN (Kurz) ===")
            lines.append(f"  10y US Treasury Yield: {_fmt_num(macro.get('us_10y_yield'))} %")
            lines.append(f"  10y Zinstrend: {macro.get('us_10y_trend', 'N/A')}")

    # Peer-Vergleich — für fundamental und alle
    if role in ("alle", "fundamental"):
        if peers:
            lines.append("")
            lines.append("=== PEER-VERGLEICH ===")
            lines.append(f"  Eigener KGV: {_fmt_num(f.get('pe_ratio'))}")
            for p in peers:
                lines.append(
                    f"  {p.get('ticker', '?')}: KGV {_fmt_num(p.get('pe_ratio'))}, "
                    f"Marktkap {_fmt_num(p.get('market_cap'), ' ')} ({p.get('name', 'N/A')})"
                )
            if macro:
                lines.append(f"  S&P 500 KGV (Benchmark): {_fmt_num(macro.get('sp500_pe'))}")

    # SENTIMENT — für sentiment und alle
    if role in ("alle", "sentiment"):
        lines.append("")
        lines.append("=== SENTIMENT ===")
        lines.append(f"  Positive Headlines: {s.get('positiv', 0)}")
        lines.append(f"  Negative Headlines: {s.get('negativ', 0)}")
        lines.append(f"  Neutrale Headlines: {s.get('neutral', 0)}")

        # Zeitgewichtete / dominante Stimmung hinzufügen, falls verfügbar
        is_weighted = s.get("weighted", False)
        if is_weighted:
            lines.append("  Zeitgewichtung: ja (Halbwertszeit 7 Tage)")
            lines.append(f"  Dominante Stimmung: {s.get('dominant', 'N/A')}")
        else:
            lines.append("  Zeitgewichtung: nein (ungewichtete Zählung)")
            if s.get("dominant"):
                lines.append(f"  Dominante Stimmung: {s.get('dominant', 'N/A')}")
        lines.append(f"  Sample-Größe: {s.get('sample_size', len(news))}")

        if news:
            lines.append("  Headlines (neueste):")
            for h in news[:10]:
                lines.append(f"    - {h}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agenten-Funktionen
# ---------------------------------------------------------------------------


def _analyst_consistency_warning(stimmung: Any, score: Any) -> str:
    """Prüft Stimmung/Score-Konsistenz und gibt ggf. eine deutsche Warnung zurück.

    Inkonsistenzen (z.B. bearish + Score 4) deuten auf Modell-Halluzination.

    Regeln:
        - bullish und score <= 1 → Warnung
        - bearish und score >= 4 → Warnung
        - neutral und (score <= 1 oder score >= 5) → Warnung
        - sonst "" (keine Warnung)

    Args:
        stimmung: Stimmung als String ("bullish"/"neutral"/"bearish").
        score: Score als Zahl (1-5).

    Returns:
        Deutschen Warn-String bei Inkonsistenz, sonst leeren String "".
    """
    try:
        score_int = int(score)
    except (TypeError, ValueError):
        return ""

    stim = str(stimmung).strip().lower() if stimmung else ""
    if stim == "bullish" and score_int <= 1:
        return (
            f"Konsistenz-Warnung: Stimmung='bullish' mit Score={score_int} "
            "ist inkonsistent (bullish erwartet Score ≥ 2). Mögliche Halluzination."
        )
    if stim == "bearish" and score_int >= 4:
        return (
            f"Konsistenz-Warnung: Stimmung='bearish' mit Score={score_int} "
            "ist inkonsistent (bearish erwartet Score ≤ 3). Mögliche Halluzination."
        )
    if stim == "neutral" and (score_int <= 1 or score_int >= 5):
        return (
            f"Konsistenz-Warnung: Stimmung='neutral' mit Score={score_int} "
            "ist inkonsistent (neutral erwartet Score 2-4). Mögliche Halluzination."
        )
    return ""


def _call_agent(llm: LLMClient, system_prompt: str, user_text: str, temperature: float = 0.3) -> dict[str, Any]:
    """Führt einen einzelnen Agenten-Call aus und parst das Ergebnis."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    raw = llm.chat(messages, temperature=temperature)
    parsed = parse_json(raw)
    parsed["_raw"] = raw
    return parsed


def analyst_team(
    data: dict[str, Any],
    llm: LLMClient,
    data_text: str | None = None,
) -> dict[str, Any]:
    """Ruft 3 Analysten-Rollen auf (Fundamental, Technical, Sentiment).

    Returns dict mit 'fundamental', 'technical', 'sentiment' und 'technicals' Schlüsseln.
    Die 3 Analysten-Calls werden PARALLEL über ThreadPoolExecutor ausgeführt.
    Bei einem Teilfehler wird eine Warnung geloggt und für den betroffenen key
    ein Fehlereintrag geliefert — die Pipeline crasht nicht.

    Jeder Analyst erhält einen rollenspezifischen Daten-Text via _build_data_text
    (weniger Rauschen). Wenn data_text extern gesetzt ist, wird dieser unverändert
    für alle Analysten verwendet (nicht übersteuert).

    Nach jedem Analysten-Ergebnis wird _analyst_consistency_warning geprüft;
    bei Inkonsistenz wird ein Feld "konsistenz_warnung" angehängt.

    Args:
        data: Daten-dict aus collect_ticker_data.
        llm: LLMClient für die Agenten-Calls.
        data_text: Optional vorberechneter Daten-Text (vermeidet mehrfache
            _build_data_text-Berechnung). Wenn None, wird pro Analyst ein
            rollenspezifischer Text gebaut.
    """
    # Rollen-Mapping: analyst_team key → _build_data_text role
    role_map = {
        "fundamental": "fundamental",
        "technical": "technik",
        "sentiment": "sentiment",
    }

    # (key, system_prompt) — Reihenfolge bleibt fundamental/technical/sentiment
    analyst_specs = [
        ("fundamental", SYSTEM_FUNDAMENTAL),
        ("technical", SYSTEM_TECHNICAL),
        ("sentiment", SYSTEM_SENTIMENT),
    ]

    results: dict[str, Any] = {}

    def _run_one(key: str, system_prompt: str) -> tuple[str, dict[str, Any]]:
        if data_text is not None:
            text = data_text
        else:
            text = _build_data_text(data, role=role_map[key])
        return key, _call_agent(llm, system_prompt, text)

    max_workers = min(len(analyst_specs), _MAX_PARALLEL)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, key, prompt): key for key, prompt in analyst_specs}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                _, result = future.result()
                results[key] = result
            except Exception as exc:  # noqa: BLE001 — nie crashen
                logger.warning("Analyst '%s' fehlgeschlagen: %s", key, exc)
                results[key] = {"_raw": "", "fehler": str(exc)}

    # Sicherstellen, dass alle 3 Keys vorhanden sind (defensiv)
    for key, _ in analyst_specs:
        results.setdefault(key, {"_raw": "", "fehler": "nicht ausgeführt"})

    # Konsistenz-Wächter: nach jedem Analysten-Ergebnis prüfen
    for key in role_map:
        a = results.get(key)
        if not isinstance(a, dict):
            continue
        warning = _analyst_consistency_warning(a.get("stimmung"), a.get("score"))
        if warning:
            a["konsistenz_warnung"] = warning

    # technicals durchreichen, damit _extract_current_price sauber arbeiten kann
    results["technicals"] = data.get("technicals", {})

    return results


def _analyst_summary_text(analysts: dict[str, Any]) -> str:
    """Kompakte Zusammenfassung aller Analysten für Debatte/Trader."""
    parts = []
    for role_key, label in [("fundamental", "Fundamental"), ("technical", "Technik"), ("sentiment", "Sentiment")]:
        a = analysts.get(role_key, {})
        parts.append(
            f"{label}-Analyst: Stimmung={a.get('stimmung', 'N/A')}, "
            f"Score={a.get('score', 'N/A')}, "
            f"Zusammenfassung={a.get('zusammenfassung', a.get('_raw', 'N/A'))[:300]}"
        )
    return "\n".join(parts)


def debate(analysts: dict[str, Any], llm: LLMClient) -> dict[str, Any]:
    """Führt Bull/Bear-Debatte durch (2 LLM-Calls).

    Returns dict mit 'bull' und 'bear' Schlüsseln.
    """
    summary = _analyst_summary_text(analysts)

    bull = _call_agent(llm, SYSTEM_BULL, f"Analysten-Einschätzungen:\n{summary}", temperature=0.5)
    bear = _call_agent(llm, SYSTEM_BEAR, f"Analysten-Einschätzungen:\n{summary}", temperature=0.5)

    return {"bull": bull, "bear": bear}


def trader(
    analysts: dict[str, Any],
    debate_result: dict[str, Any],
    llm: LLMClient,
    temperature: float = 0.3,
    feedback_context: str = "",
    reflection_context: str = "",
) -> dict[str, Any]:
    """Erstellt Trade-Vorschlag aus Analysten + Debatte.

    Args:
        analysts: Analysten-Ergebnisse.
        debate_result: Bull/Bear-Debatte-Ergebnis.
        llm: LLMClient.
        temperature: Sampling-Temperatur für den LLM-Call (Default 0.3).
            Wird von ``ensemble_trader`` pro Run variiert, um eine
            Mehrheitsabstimmung über unterschiedlich sampling-erzeugte Trades
            durchzuführen.
        feedback_context: Optionaler Track-Record-Kontext-Block (leer = kein
            Feedback). Wird am Ende des User-Prompts angehängt, damit der
            Trader seine Kalibrierung an der Historie ausrichten kann.
        reflection_context: Optionaler Reflexions-Block (leer = keine
            Reflexion). Wird nach feedback_context am Ende des User-Prompts
            angehängt.
    """
    summary = _analyst_summary_text(analysts)
    bull_text = debate_result.get("bull", {}).get("_raw", "")
    bear_text = debate_result.get("bear", {}).get("_raw", "")

    user_text = (
        f"Analysten-Einschätzungen:\n{summary}\n\n"
        f"Bull-Argumentation:\n{bull_text}\n\n"
        f"Bear-Argumentation:\n{bear_text}"
    )
    if feedback_context:
        user_text += f"\n\n{feedback_context}"
    if reflection_context:
        user_text += f"\n\n{reflection_context}"
    result = _call_agent(llm, SYSTEM_TRADER, user_text, temperature=temperature)
    # 5-stufige Rating normalisieren: rohes Rating in 'rating', 3-stufige Aktion in 'aktion'
    raw_rating = str(result.get("aktion", "")).strip().upper()
    result["rating"] = raw_rating
    result["aktion"] = _rating_to_action(raw_rating)
    return result


# ---------------------------------------------------------------------------
# Ensemble-Trader — mehrere Runs mit Mehrheitsabstimmung + Plausibilitäts-Check
# ---------------------------------------------------------------------------

# Standard-Temperatur-Spread für Ensemble-Runs (leicht variierend)
_DEFAULT_TEMPERATURES: list[float] = [0.3, 0.5, 0.7]


def _extract_current_price(analysts: dict[str, Any]) -> float | None:
    """Extrahiert den aktuellen Kurs aus den Analysten-Daten.

    Primärer Weg: direkt aus ``analysts["technicals"]["current_price"]`` (wird
    von ``analyst_team`` zuverlässig aus dem data-dict durchgereicht).
    Fallback: Suche in den Analysten-Subdicts nach einem ``current_price``-Feld.
    Kein Regex-Parsing aus rohem LLM-Fließtext mehr (unzuverlässig).
    """
    # 1. Direkt aus technicals (primärer Weg — von analyst_team durchgereicht)
    technicals = analysts.get("technicals")
    if isinstance(technicals, dict):
        price = technicals.get("current_price")
        if price is not None:
            try:
                return float(price)
            except (TypeError, ValueError):
                pass

    # 2. Fallback: in Analysten-Subdicts nach current_price suchen
    for key in ("fundamental", "technical", "sentiment"):
        a = analysts.get(key, {})
        if not isinstance(a, dict):
            continue
        price = a.get("current_price")
        if price is not None:
            try:
                return float(price)
            except (TypeError, ValueError):
                pass

    return None


def _is_plausible_kauf(trade: dict[str, Any], current_price: float | None) -> bool:
    """Prüft, ob ein KAUFEN-Trade plausible Ziel-/Stop-Werte hat.

    Für KAUFEN: zielkurs > current_price und stop_loss < current_price (falls angegeben).
    """
    if current_price is None:
        return True  # Ohne current_price können wir nicht prüfen
    try:
        ziel = trade.get("zielkurs")
        if ziel is not None:
            ziel_f = float(ziel)
            if ziel_f <= current_price:
                return False
    except (TypeError, ValueError):
        pass
    try:
        stop = trade.get("stop_loss")
        if stop is not None:
            stop_f = float(stop)
            if stop_f >= current_price:
                return False
    except (TypeError, ValueError):
        pass
    return True


def _fix_implausible_trade(trade: dict[str, Any], current_price: float | None) -> dict[str, Any]:
    """Korrigiert unplausible Ziel-/Stop-Werte bei KAUFEN.

    - zielkurs <= current_price → None (nicht vertrauenswürdig)
    - stop_loss >= current_price → None
    Bei anderen Aktionen wird nichts geändert.
    """
    if current_price is None:
        return trade
    if str(trade.get("aktion", "")).upper() != "KAUFEN":
        return trade

    fixed = dict(trade)
    try:
        ziel = fixed.get("zielkurs")
        if ziel is not None and float(ziel) <= current_price:
            fixed["zielkurs"] = None
    except (TypeError, ValueError):
        pass
    try:
        stop = fixed.get("stop_loss")
        if stop is not None and float(stop) >= current_price:
            fixed["stop_loss"] = None
    except (TypeError, ValueError):
        pass
    return fixed


def ensemble_trader(
    analysts: dict[str, Any],
    debate_result: dict[str, Any],
    llm: LLMClient,
    runs: int = 3,
    temperature_range: list[float] | None = None,
    feedback_context: str = "",
    reflection_context: str = "",
) -> dict[str, Any]:
    """Führt den Trader mehrfach aus (Ensemble) und aggregiert per Mehrheitsentscheid.

    Args:
        analysts: Analysten-Ergebnisse (wie bei trader()).
        debate_result: Bull/Bear-Debatte-Ergebnis.
        llm: LLMClient.
        runs: Anzahl der Ensemble-Runs (Default 3).
        temperature_range: Temperaturen pro Run (Default [0.3, 0.5, 0.7]).
            Bei weniger/mehr Runs wird zyklisch verwendet bzw. abgeschnitten.
        feedback_context: Optionaler Track-Record-Kontext-Block (leer = kein
            Feedback). Wird an jeden trader()-Aufruf durchgereicht.
        reflection_context: Optionaler Reflexions-Block (leer = keine
            Reflexion). Wird an jeden trader()-Aufruf durchgereicht.

    Returns:
        dict mit dem gewählten Trade plus _ensemble-Metadaten:
          _ensemble: {runs, mehrheits_aktion, ensemble_confidence, alle_aktionen, alle_ratings}
    """
    if temperature_range is None:
        temperature_range = _DEFAULT_TEMPERATURES

    current_price = _extract_current_price(analysts)

    # Temperatur-Spread an Anzahl der Runs anpassen
    temps = []
    for i in range(runs):
        temps.append(temperature_range[i % len(temperature_range)])

    # Mehrere Trader-Runs PARALLEL ausführen
    all_runs: list[dict[str, Any]] = []

    def _run_trader(temp: float) -> dict[str, Any]:
        return trader(
            analysts, debate_result, llm,
            temperature=temp,
            feedback_context=feedback_context,
            reflection_context=reflection_context,
        )

    max_workers = min(len(temps), _MAX_PARALLEL)
    # Reihenfolge der Ergebnisse muss der Temp-Reihenfolge entsprechen für
    # deterministische Aggregation (Mehrheitsabstimmung, basis_run-Auswahl).
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_temp = {pool.submit(_run_trader, temp): temp for temp in temps}
        # Ergebnisse in der gleichen Reihenfolge wie temps sammeln
        temp_to_result: dict[float, dict[str, Any] | None] = {}
        for future in concurrent.futures.as_completed(future_to_temp):
            temp = future_to_temp[future]
            try:
                temp_to_result[temp] = future.result()
            except Exception as exc:  # noqa: BLE001 — nie crashen
                logger.warning("Ensemble-Run fehlgeschlagen (temp=%.1f): %s", temp, exc)
                temp_to_result[temp] = None

    for temp in temps:
        run = temp_to_result.get(temp)
        if run is not None:
            all_runs.append(run)

    # Robust: wenn gar kein Run erfolgreich war
    if not all_runs:
        logger.error("Ensemble: Alle %d Runs fehlgeschlagen — gebe leeres dict zurück.", runs)
        return {
            "rolle": "Trader",
            "aktion": "HALTEN",
            "rating": "HALTEN",
            "zielkurs": None,
            "stop_loss": None,
            "positionsanteil": 0,
            "begründung": "Ensemble: Alle Runs fehlgeschlagen.",
            "zeithorizont": "N/A",
            "_raw": "",
            "_ensemble": {
                "runs": runs,
                "mehrheits_aktion": "HALTEN",
                "ensemble_confidence": 0.0,
                "alle_aktionen": [],
                "alle_ratings": [],
            },
        }

    # Single-Fallback: nur 1 erfolgreicher Run → direkt übernehmen
    if len(all_runs) == 1:
        result = dict(all_runs[0])
        aktion = str(result.get("aktion", "HALTEN")).upper()
        rating = str(result.get("rating", aktion)).upper()
        result["_ensemble"] = {
            "runs": 1,
            "mehrheits_aktion": aktion,
            "ensemble_confidence": 1.0,
            "alle_aktionen": [aktion],
            "alle_ratings": [rating],
        }
        return result

    # Mehrheitsabstimmung über aktion (3-stufig normalisiert)
    aktionen = [str(r.get("aktion", "HALTEN")).upper() for r in all_runs]
    aktion_counts: dict[str, int] = {}
    for a in aktionen:
        aktion_counts[a] = aktion_counts.get(a, 0) + 1

    mehrheits_aktion = max(aktion_counts, key=lambda k: aktion_counts[k])
    confidence = aktion_counts[mehrheits_aktion] / len(all_runs)

    # 5-stufige Ratings sammeln (Fallback auf aktion wenn rating fehlt)
    alle_ratings = [
        str(r.get("rating", r.get("aktion", "HALTEN"))).strip().upper()
        for r in all_runs
    ]

    # Den ersten Run mit der Mehrheits-Aktion als Basis wählen
    basis_run = None
    for r in all_runs:
        if str(r.get("aktion", "")).upper() == mehrheits_aktion:
            basis_run = r
            break

    # Fallback (sollte nie passieren, aber sicherheitshalber)
    if basis_run is None:
        basis_run = all_runs[0]

    result = dict(basis_run)

    # Plausibilitäts-Check für den gewählten Trade
    if mehrheits_aktion == "KAUFEN" and not _is_plausible_kauf(result, current_price):
        # Versuche, einen plausiblen Zielkurs aus anderen Runs zu übernehmen
        for r in all_runs:
            if str(r.get("aktion", "")).upper() != "KAUFEN":
                continue
            if _is_plausible_kauf(r, current_price):
                # Plausible Werte übernehmen
                try:
                    ziel = r.get("zielkurs")
                    if ziel is not None and result.get("zielkurs") is None:
                        result["zielkurs"] = ziel
                except (TypeError, ValueError):
                    pass
                try:
                    stop = r.get("stop_loss")
                    if stop is not None and result.get("stop_loss") is None:
                        result["stop_loss"] = stop
                except (TypeError, ValueError):
                    pass
                break
        # Falls immer noch unplausibel: Werte auf None setzen
        result = _fix_implausible_trade(result, current_price)

    result["_ensemble"] = {
        "runs": len(all_runs),
        "mehrheits_aktion": mehrheits_aktion,
        "ensemble_confidence": round(confidence, 2),
        "alle_aktionen": aktionen,
        "alle_ratings": alle_ratings,
    }

    return result


def compute_position_size(
    volatility: float | None,
    risk_budget_pct: float = 2.0,
    max_position_pct: float = 10.0,
) -> float | None:
    """Berechnet eine rechnerische Positionsgröße via Volatility-Targeting.

    Formel: positions_pct = min(risk_budget_pct / volatility, max_position_pct)

    Args:
        volatility: Annualisierte Volatilität als Dezimalbruch (z.B. 0.30 = 30%).
            Bei None, <= 0 oder nicht float-konvertierbar → None.
        risk_budget_pct: Risiko-Budget in % (Default 2.0).
        max_position_pct: Maximale Positionsgröße in % (Default 10.0).

    Returns:
        Empfohlene Positionsgröße in % (float) oder None bei ungültiger Volatilität.

    Beispiel:
        risk_budget 2%, annualisierte Vol 30% → 0.02/0.30 = 6.67% → 6.67%
    """
    if volatility is None:
        return None
    try:
        vol = float(volatility)
    except (TypeError, ValueError):
        return None
    if vol <= 0:
        return None
    positions_pct = risk_budget_pct / vol
    return round(min(positions_pct, max_position_pct), 2)


def _compute_annualized_volatility(data: dict[str, Any]) -> float | None:
    """Berechnet die annualisierte Volatilität aus Tagesrenditen der Historie.

    Verwendet data["history"] (Liste von dicts mit "close"-Werten).
    Formel: std der Tagesrenditen * sqrt(252).

    Returns:
        Annualisierte Volatilität als Dezimalbruch (z.B. 0.30) oder None
        bei fehlender/zu kurzer Historie.
    """
    history = data.get("history", [])
    if not history or len(history) < 2:
        return None
    try:
        closes = [float(h["close"]) for h in history if h.get("close") is not None]
    except (TypeError, ValueError, KeyError):
        return None
    if len(closes) < 2:
        return None
    # Tagesrenditen berechnen
    returns: list[float] = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0:
            continue
        returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
    if len(returns) < 2:
        return None
    # Std der Tagesrenditen
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std = variance**0.5
    if std <= 0:
        return None
    import math

    annualized = std * math.sqrt(252)
    return round(annualized, 6)


def risk_manager(
    trade: dict[str, Any],
    data: dict[str, Any],
    llm: LLMClient,
    data_text: str | None = None,
    feedback_context: str = "",
) -> dict[str, Any]:
    """Bewertet Risiko des Trades.

    Der LLM-basierte Risk-Manager-Aufruf bleibt unverändert. Zusätzlich wird
    eine rechnerische Positionsgröße via Volatility-Targeting ergänzt
    (positionsgröße_rechnerisch_pct, volatilität_annualisiert_pct).

    Args:
        trade: Trade-Vorschlag vom Trader/Ensemble.
        data: Daten-dict aus collect_ticker_data.
        llm: LLMClient.
        data_text: Optional vorberechneter Daten-Text (vermeidet mehrfache
            _build_data_text-Berechnung). Wenn None, wird er intern berechnet.
        feedback_context: Optionaler Track-Record-Kontext-Block (leer = kein
            Feedback). Wird am Ende des User-Prompts angehängt.
    """
    trade_text = json.dumps(trade, ensure_ascii=False, indent=2, default=str)
    if data_text is None:
        data_text = _build_data_text(data)

    # --- Rechnerische Positionsgröße VOR dem LLM-Call berechnen ---
    # Einmal berechnen, im Prompt UND im Rückgabedict verwenden (keine Doppelberechnung).
    annualized_vol = _compute_annualized_volatility(data)
    vol_pct = round(annualized_vol * 100, 2) if annualized_vol is not None else None
    pos_rechnerisch = compute_position_size(annualized_vol)

    vol_str = f"{vol_pct} %" if vol_pct is not None else "N/A"
    pos_str = f"{pos_rechnerisch} %" if pos_rechnerisch is not None else "N/A"

    risk_block = (
        "\n\n=== RECHNERISCHES RISIKO-MODELL (deterministisch) ===\n"
        f"Annualisierte Volatilität: {vol_str}\n"
        f"Rechnerische Positionsgröße (Volatility-Targeting, "
        f"Risiko-Budget 2%, Cap 10%): {pos_str}\n"
        "Anweisung: Dein \"positionsgröße_empfohlen\" sollte in der Nähe der "
        "rechnerischen Positionsgröße liegen, es sei denn, du begründest eine "
        "Abweichung (z.B. in den auflagen)."
    )

    user_text = f"Trade-Vorschlag:\n{trade_text}\n\nMarktdaten:\n{data_text}{risk_block}"
    if feedback_context:
        user_text += f"\n\n{feedback_context}"
    risk = _call_agent(llm, SYSTEM_RISK, user_text)

    # Rechnerische Werte ins Rückgabedict übernehmen (einmal berechnet, nicht doppelt)
    risk["volatilität_annualisiert_pct"] = vol_pct
    risk["positionsgröße_rechnerisch_pct"] = pos_rechnerisch

    return risk


def portfolio_manager(
    trade: dict[str, Any],
    risk: dict[str, Any],
    llm: LLMClient,
    portfolio_fit: dict[str, Any] | None = None,
    feedback_context: str = "",
    reflection_context: str = "",
) -> dict[str, Any]:
    """Trifft finale Entscheidung.

    Args:
        trade: Trade-Vorschlag vom Trader/Ensemble.
        risk: Risiko-Bewertung vom Risk-Manager.
        llm: LLMClient.
        portfolio_fit: Optional Portfolio-Fit-Ergebnis (Ziel-Gewichtung etc.).
            Wird dem PM als zusätzlicher Kontext übergeben.
        feedback_context: Optionaler Track-Record-Kontext-Block (leer = kein
            Feedback). Wird am Ende des User-Prompts angehängt, damit der PM
            seine Kalibrierung an der Historie ausrichten kann.
        reflection_context: Optionaler Reflexions-Block (leer = keine
            Reflexion). Wird nach feedback_context am Ende des User-Prompts
            angehängt.
    """
    trade_text = json.dumps(trade, ensure_ascii=False, indent=2, default=str)
    risk_text = json.dumps(risk, ensure_ascii=False, indent=2, default=str)

    user_text = f"Trade-Vorschlag:\n{trade_text}\n\nRisiko-Bewertung:\n{risk_text}"

    if portfolio_fit is not None:
        pf_text = json.dumps(portfolio_fit, ensure_ascii=False, indent=2, default=str)
        user_text += f"\n\nPortfolio-Fit-Einschätzung:\n{pf_text}"

    if feedback_context:
        user_text += f"\n\n{feedback_context}"

    if reflection_context:
        user_text += f"\n\n{reflection_context}"

    return _call_agent(llm, SYSTEM_PM, user_text)


def trade_revision(
    trade: dict[str, Any],
    risk: dict[str, Any],
    portfolio_fit: dict[str, Any] | None,
    llm: LLMClient,
    feedback_context: str = "",
    reflection_context: str = "",
) -> dict[str, Any]:
    """Trade-Revision (2nd Pass) — der Trader überarbeitet seinen Trade.

    Nachdem Risk-Manager und Portfolio-Fit-Analyst den Trade bewertet haben,
    bekommt der Trader eine zweite Runde, um seinen Trade anzupassen.

    Args:
        trade: Ursprünglicher Trade-Vorschlag vom Trader/Ensemble.
        risk: Risiko-Bewertung vom Risk-Manager.
        portfolio_fit: Portfolio-Fit-Ergebnis (oder None, wenn nicht verfügbar).
        llm: LLMClient.
        feedback_context: Optionaler Track-Record-Kontext-Block.
        reflection_context: Optionaler Reflexions-Block.

    Returns:
        dict mit dem revidierten Trade (gleiche Felder wie trader(),
        plus rating = rohes 5-stufig, aktion = 3-stufig normalisiert).
    """
    trade_text = json.dumps(trade, ensure_ascii=False, indent=2, default=str)
    risk_text = json.dumps(risk, ensure_ascii=False, indent=2, default=str)

    user_text = f"Ursprünglicher Trade-Vorschlag:\n{trade_text}\n\nRisiko-Bewertung:\n{risk_text}"

    if portfolio_fit is not None:
        pf_text = json.dumps(portfolio_fit, ensure_ascii=False, indent=2, default=str)
        user_text += f"\n\nPortfolio-Fit-Einschätzung:\n{pf_text}"
    else:
        user_text += "\n\nPortfolio-Fit-Einschätzung: Nicht verfügbar."

    if feedback_context:
        user_text += f"\n\n{feedback_context}"
    if reflection_context:
        user_text += f"\n\n{reflection_context}"

    result = _call_agent(llm, SYSTEM_TRADE_REVISION, user_text, temperature=0.3)
    # 5-stufige Rating normalisieren (wie bei trader())
    raw_rating = str(result.get("aktion", "")).strip().upper()
    result["rating"] = raw_rating
    result["aktion"] = _rating_to_action(raw_rating)
    return result
