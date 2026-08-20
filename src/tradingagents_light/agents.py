"""Agenten-Modul — spezialisierte LLM-Rollen-Aufrufe für die Trading-Pipeline.

Jede Rolle ist ein strukturierter LLM-Call mit deutschen Prompts.
Agenten liefern Stimmung (bullish/neutral/bearish) + Score (1-5).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .llm import LLMClient

logger = logging.getLogger(__name__)

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

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{
  "rolle": "Trader",
  "aktion": "KAUFEN" | "HALTEN" | "VERKAUFEN",
  "zielkurs": "Zielkurs als Zahl oder null",
  "stop_loss": "Stop-Loss als Zahl oder null",
  "positionsanteil": "Empfohlener Positionsanteil in % (z.B. 5)",
  "begründung": "2-4 Sätze Begründung auf Deutsch",
  "zeithorizont": "Kurzfristig" | "Mittelfristig" | "Langfristig"
}
"""

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

SYSTEM_PM = """\
Du bist der Portfolio-Manager. Du triffst die finale Entscheidung über den Trade, \
basierend auf dem Trade-Vorschlag und der Risiko-Bewertung. Du kannst den Trade \
genehmigen oder ablehnen.

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{
  "rolle": "Portfolio-Manager",
  "entscheidung": "GENEHMIGT" | "ABGELEHNT",
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
            return f"{fval / 1e12:.2f}B{suffix}"
        if abs(fval) >= 1e9:
            return f"{fval / 1e9:.2f} Mrd{suffix}"
        if abs(fval) >= 1e6:
            return f"{fval / 1e6:.2f} Mio{suffix}"
        if abs(fval) >= 1e3:
            return f"{fval / 1e3:.2f} K{suffix}"
        return f"{fval:.2f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def _build_data_text(data: dict[str, Any]) -> str:
    """Erstellt einen kompakten deutschen Daten-Text für LLM-Prompts."""
    f = data.get("fundamentals", {})
    t = data.get("technicals", {})
    s = data.get("sentiment", {})
    news = data.get("news", [])
    macro = data.get("macro", {})
    peers = data.get("peers", [])

    lines = [
        f"Aktie: {data.get('ticker', '?')} ({f.get('name', 'N/A')})",
        f"Sektor: {f.get('sector', 'N/A')} / {f.get('industry', 'N/A')}",
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
    ]

    # Feature 2: Makro/Zins-Daten
    if macro:
        lines.append("")
        lines.append("=== MAKRO / ZINSEN ===")
        lines.append(f"  10y US Treasury Yield: {_fmt_num(macro.get('us_10y_yield'))} %")
        lines.append(f"  10y Yield vor 1 Monat: {_fmt_num(macro.get('us_10y_yield_1m_ago'))} %")
        lines.append(f"  10y Zinstrend: {macro.get('us_10y_trend', 'N/A')}")
        lines.append(f"  S&P 500 KGV: {_fmt_num(macro.get('sp500_pe'))}")
        lines.append(f"  S&P 500 Marktkap: {_fmt_num(macro.get('sp500_market_cap'), ' ')}")
        lines.append(
            "  Hinweis: Hohe/steigende Zinsen belasten kapitalintensive "
            "und erneuerbare Sektoren."
        )

    # Feature 3: Peer-Vergleich
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


def analyst_team(data: dict[str, Any], llm: LLMClient) -> dict[str, Any]:
    """Ruft 3 Analysten-Rollen auf (Fundamental, Technical, Sentiment).

    Returns dict mit 'fundamental', 'technical', 'sentiment' Schlüsseln.
    """
    data_text = _build_data_text(data)

    fundamental = _call_agent(llm, SYSTEM_FUNDAMENTAL, data_text)
    technical = _call_agent(llm, SYSTEM_TECHNICAL, data_text)
    sentiment = _call_agent(llm, SYSTEM_SENTIMENT, data_text)

    return {
        "fundamental": fundamental,
        "technical": technical,
        "sentiment": sentiment,
    }


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


def trader(analysts: dict[str, Any], debate_result: dict[str, Any], llm: LLMClient) -> dict[str, Any]:
    """Erstellt Trade-Vorschlag aus Analysten + Debatte."""
    summary = _analyst_summary_text(analysts)
    bull_text = debate_result.get("bull", {}).get("_raw", "")
    bear_text = debate_result.get("bear", {}).get("_raw", "")

    user_text = (
        f"Analysten-Einschätzungen:\n{summary}\n\n"
        f"Bull-Argumentation:\n{bull_text}\n\n"
        f"Bear-Argumentation:\n{bear_text}"
    )
    return _call_agent(llm, SYSTEM_TRADER, user_text)


def risk_manager(trade: dict[str, Any], data: dict[str, Any], llm: LLMClient) -> dict[str, Any]:
    """Bewertet Risiko des Trades."""
    trade_text = json.dumps(trade, ensure_ascii=False, indent=2, default=str)
    data_text = _build_data_text(data)

    user_text = f"Trade-Vorschlag:\n{trade_text}\n\nMarktdaten:\n{data_text}"
    return _call_agent(llm, SYSTEM_RISK, user_text)


def portfolio_manager(trade: dict[str, Any], risk: dict[str, Any], llm: LLMClient) -> dict[str, Any]:
    """Trifft finale Entscheidung."""
    trade_text = json.dumps(trade, ensure_ascii=False, indent=2, default=str)
    risk_text = json.dumps(risk, ensure_ascii=False, indent=2, default=str)

    user_text = f"Trade-Vorschlag:\n{trade_text}\n\nRisiko-Bewertung:\n{risk_text}"
    return _call_agent(llm, SYSTEM_PM, user_text)
