"""Agenten-Modul — spezialisierte LLM-Rollen-Aufrufe für die Trading-Pipeline.

Jede Rolle ist ein strukturierter LLM-Call mit deutschen Prompts.
Agenten liefern Stimmung (bullish/neutral/bearish) + Score (1-5).
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import os
import re
from datetime import datetime, timedelta
from typing import Any

from . import config
from .factors import compute_multi_factor_score
from .llm import LLMClient, StructuredChatResult
from .schemas import (
    ANALYST_FUNDAMENTAL_SCHEMA,
    ANALYST_MACRO_NEWS_SCHEMA,
    ANALYST_SENTIMENT_SCHEMA,
    ANALYST_SOCIAL_SCHEMA,
    ANALYST_TECHNICAL_SCHEMA,
    DEBATE_SCHEMA,
    FINAL_SCHEMA,
    RISK_SCHEMA,
    TRADE_SCHEMA,
    defaults_for_schema,
)

logger = logging.getLogger(__name__)

# Maximale Anzahl paralleler Threads für unabhängige LLM-Calls
_MAX_PARALLEL = 5

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

Wichtige Ergänzung — Relatives Momentum (cross-sectional): Bewerte das Momentum \
des Tickers RELATIV zum S&P 500 (relatives_momentum_6m, in Prozentpunkten). \
Positives relatives Momentum = relative Stärke (Ticker schlägt den Markt), \
negatives = relative Schwäche (Ticker verliert gegen den Markt). Dieses Signal \
ist empirisch robuster als ein einzelner SMA-Check und ergänzt die \
SMA/RSI/MACD-Analyse — es ersetzt sie nicht.

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

SYSTEM_SOCIAL = """\
Du bist ein Social-Media-Analyst. Du bewertest die Stimmung der RETAIL-COMMUNITY \
aus StockTwits- und Reddit-Posts zu einer Aktie (NICHT Nachrichten-Headlines).

Dein Fokus:
- Crowd-Stimmung: Wie ist die Gesamtstimmung der Retail-Anleger (bullish/bearish/\
neutral)? Beachte die Anzahl und das Verhältnis der Posts.
- Konträr-Indikator: Extreme Retail-Euphorie (jeder ist bullish, Hype-Wörter wie \
"to the moon", "rocket") kann ein KONTRÄRES Warnsignal sein (euphorische Massen \
sitzen bereits im Trade). Extreme Retail-Panik kann ein Boden-Signal sein. \
Einseitige Euphorie → nüchtern-konträres Sentiment, nicht blind nachlaufen.
- Meme-/Retail-Dynamik: Erwähne wenn Posts meme-getrieben sind, Hype-Phasen \
erkennbar sind oder die Diskussion wenig Substanz enthält.

Wenn keine oder sehr wenige Posts vorliegen, sage das explizit und liefere eine \
vorsichtig-neutrale Einschätzung (keine Stimmung aus dünnen Daten ableiten).

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{
  "rolle": "Social-Media-Analyst",
  "stimmung": "bullish" | "neutral" | "bearish",
  "score": 1-5,
  "zusammenfassung": "2-4 Sätze Zusammenfassung auf Deutsch",
  "dominant": "positiv" | "negativ" | "neutral",
  "community_stimmung": "retail-bullish" | "retail-bearish" | "retail-neutral"
}
"""

SYSTEM_MACRO_NEWS = """\
Du bist ein Makro/News-Analyst. Du bewertest das MAKRO-UMFELD und die \
NEWS-HEADLINES für eine Aktie.

Makro-Kennzahlen: 10y US Treasury Yield (und Zinstrend), VIX, Ölpreis (WTI), \
EURUSD, S&P 500 Trend und S&P 500 KGV. News: Headlines des Tickers samt \
positiv/negativ/neutral-Zählung und dominanter Stimmung.

Dein Fokus:
- Wie wirkt das Makro-Umfeld (Zinsniveau und Zinstrend, Risiko-Regime laut VIX, \
Ölpreis, EURUSD, Gesamtmarkt-Trend, Bewertung des Gesamtmarkts) auf DIESEN \
Ticker und seinen Sektor? Berücksichtige den Sektor der Aktie bei der \
Einordnung (z.B. Zinssensitivität, Rohstoff- und Währungsexposure).
- Welche Headlines sind MATERIAL (kursrelevant für den Ticker) und welche \
sind Rauschen? Gewichte sie entsprechend.
- Setze Makro- und News-Eindrücke zu einer Gesamt-Einschätzung zusammen.

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{
  "rolle": "Makro/News-Analyst",
  "stimmung": "bullish" | "neutral" | "bearish",
  "score": 1-5,
  "zusammenfassung": "2-4 Sätze Zusammenfassung auf Deutsch",
  "makro_einschaetzung": "Kurze Bewertung des Makro-Umfelds für diesen Ticker",
  "relevante_headlines": "Die materialsten Headlines mit kurzer Bewertung"
}
"""

SYSTEM_BULL = """\
Du bist der Bull. Fokussiere auf die konkreten STÄRKEN und Bull-Fälle aus den \
Analysten-Daten: Wachstum, Margen, technisches Momentum, positives Sentiment, \
günstige relative Bewertung.

Rollenverständnis (Hedgefonds-Praxis): Die FUNDAMENTALE Analyse (Wachstum, \
Margen, Bewertung, Sektor) ist der PRIMÄRE Treiber für die RICHTUNG (ob \
gekauft wird). Die TECHNIK (SMA, RSI, MACD) ist nur ein SEKUNDÄRER \
Timing-Filter: Sie sagt, WANN ein Einstieg günstig ist — nicht OB die These \
stimmt. Ordne technische Signale daher als TIMING-BESTÄTIGUNG ein, nicht als \
eigenständigen Kaufgrund.

Schwerpunkte:
- Wachstum: Umsatzwachstum, Gewinnmargen, EPS-Trend — wo wächst das Unternehmen?
- Marktanteil & Wettbewerbsvorteil: Sektor-Position, Differenzierung.
- Technisches Momentum: Aufwärtstrend (SMA50 > SMA200), RSI im gesunden Bereich, \
MACD positiv.
- Positives Sentiment: Dominante Stimmung, kaufsfördernde Headlines.
- Bewertung relativ zu Wachstum: PEG-Ratio, KGV im Vergleich zu Peers und S&P 500 \
— ist die Aktie relativ zu ihrem Wachstum günstig?

Ignoriere die Risiken bewusst — der Bear-Stratege kümmert sich darum. \
Dein Job ist es, die stärksten Argumente FÜR einen Kauf herauszuarbeiten.

Antworte auf Deutsch in 3-6 Sätzen. Formuliere überzeugend, aber sachlich. \
Gib am Anfang einen JSON-Block mit confidence (1-5) und einem Kurznamen an:
{"confidence": 1-5, "name": "Bull-Argumentation"}
Danach folgt dein Fließtext.
"""

SYSTEM_BEAR = """\
Du bist der Bear. Fokussiere auf die konkreten RISIKEN und Gegenargumente: \
überhöhte Bewertung, KGV zu teuer, Zinsbelastung, Konzentration, Margin-Rückgang, \
technische Schwäche.

Schwerpunkte:
- Bewertungsrisiken: KGV vs. Peers und S&P 500 — ist die Aktie zu teuer? \
PEG unplausibel?
- Konzentrationsrisiko: Abhängigkeit von einzelnen Produkten/Märkten/Kunden.
- Sektor-Overlap: Redundanz mit Peers, zyklische Sektorgefahren.
- Makroökonomische Risiken: Steigende Zinsen belasten kapitalintensive Sektoren \
und hohe Bewertungen; Zinstrend berücksichtigen.
- Technische Gegenanzeichen: Überkauft (RSI > 70), Abwärtstrend, MACD negativ, \
Bollinger-Band-Bruch nach unten.
- Margin-Erosion: Rückläufige Gewinnmargen, sinkendes Umsatzwachstum — \
wo schwächt sich das Geschäftsmodell?

Rollenverständnis (Hedgefonds-Praxis): Die FUNDAMENTALE Analyse (Bewertung, \
Margen, Wachstum, Sektor) ist der PRIMÄRE Treiber für die RICHTUNG (ob \
verkauft wird). Die TECHNIK (SMA, RSI, MACD) ist nur ein SEKUNDÄRER \
Timing-Filter: Sie sagt, WANN ein Ausstieg günstig ist — nicht OB \
die Gegen-These stimmt. Ordne technische Schwächen daher als \
TIMING-WARNUNG ein, nicht als eigenständigen Verkaufsgrund.

Ignoriere die Stärken bewusst — der Bull-Stratege kümmert sich darum. \
Dein Job ist es, die stärksten Gegenargumente GEGEN einen Kauf herauszuarbeiten.

Antworte auf Deutsch in 3-6 Sätzen. Formuliere überzeugend, aber sachlich. \
Gib am Anfang einen JSON-Block mit confidence (1-5) und einem Kurznamen an:
{"confidence": 1-5, "name": "Bear-Argumentation"}
Danach folgt dein Fließtext.
"""

SYSTEM_TRADER = """\
Du bist ein professioneller Trader. Basierend auf den Analysten-Einschätzungen \
und der Bull/Bear-Debatte erstellst du einen konkreten Trade-Vorschlag.

Rollenverständnis (Hedgefonds-Praxis) — RICHTUNG und TIMING sind getrennt:
- RICHTUNG (Aktion KAUFEN/VERKAUFEN/HALTEN): leite sie PRIMÄR aus der \
Fundamental-Analyse und der Bull/Bear-Debatte ab.
- TIMING (Einstiegszeitpunkt): leite es aus der Technik (SMA, RSI, MACD) ab.

Limit-Order-Disziplin: Gib das Feld 'einstiegs_level' an — den konkreten \
Limit-Order-Preis (Zahl oder null), zu dem der Einstieg idealerweise \
ausgeführt wird:
- Bei KAUFEN/STARK KAUFEN: ein günstigerer Einstiegspunkt an einem \
Support-Level (z.B. SMA50, Bollinger-Unterband, Rücksetzer-Level). Ist \
der aktuelle Kurs bereits attraktiv, darf einstiegs_level dem aktuellen \
Kurs entsprechen. Nutze die technischen Daten für das konkrete Level.
- Bei HALTEN/VERKAUFEN/STARK VERKAUFEN: einstiegs_level = null.

Die Technik darf die RICHTUNG nicht kippen — sie verfeinert nur das TIMING:
- Fundamental-These sagt KAUFEN, aber die Technik ist bearish (z.B. Kurs \
unter SMA200, RSI überkauft): empfiehl TROTZDEM KAUFEN (die These zählt) \
und nenne in der Begründung einen besseren Einstiegspunkt (z.B. "Rücksetzer \
an SMA50 abwarten").
- Fundamental-These sagt VERKAUFEN, aber die Technik ist bullish: empfiehl \
TROTZDEM VERKAUFEN und weise in der Begründung auf ein günstigeres \
Verkaufsfenster hin (z.B. "Verkauf in Tranchen bei Rücksetzern \
verteilen").

Nutze die volle 5-stufige Skala. 'STARK KAUFEN'/'STARK VERKAUFEN' nur bei \
hoher Überzeugung (sehr klare FUNDAMENTALE Signale — nicht bloß technische; \
technische Signale betreffen nur das TIMING). Bei Unsicherheit nimm \
'KAUFEN'/'VERKAUFEN' bzw. 'HALTEN'.

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{
  "rolle": "Trader",
  "aktion": "STARK KAUFEN" | "KAUFEN" | "HALTEN" | "VERKAUFEN" | "STARK VERKAUFEN",
  "zielkurs": "Zielkurs als Zahl oder null",
  "stop_loss": "Stop-Loss als Zahl oder null",
  "einstiegs_level": "Limit-Order-Preis für den Einstieg als Zahl oder null (bei KAUFEN: Support-Level wie SMA50/Bollinger-Unterband/Rücksetzer oder aktueller Kurs; sonst null)",
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
Berücksichtige bei nicht-EUR-Währung zusätzliches Währungsrisiko \
(Wechselkursbewegung) im Risiko-Score.

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

# Hinweis: SYSTEM_RISK wird vom Single-Pass-Risk-Manager nicht mehr verwendet
# (ersetzt durch die 3-Perspektiven-Risiko-Debatte, siehe risk_debate), bleibt
# aber aus Rückwärtskompatibilität erhalten.
SYSTEM_RISK_AGGRESSIVE = """\
Du bist der Aggressive Risk-Analyst. Du championst hohe Renditechancen, betonst \
Wachstum und Wettbewerbsvorteile und akzeptierst höheres Risiko, wenn die Upside \
es rechtfertigt.

Schwerpunkte:
- Wachstumschancen: Wo ist das Renditepotenzial größer als das Risiko?
- Wettbewerbsvorteil: Warum trägt das Geschäftsmodell das höhere Risiko?
- Chance-Risiko-Verhältnis: Upside vs. Downside in Zahlen.
- Übermäßige Vorsicht ist auch ein Risiko: verpasste Gewinne, Verharren im Cash.

Bewerte den Trade-Vorschlag positiv, wenn die Upside das Risiko rechtfertigt. \
Antworte auf Deutsch in 3-6 Sätzen als Fließtext (kein JSON).
"""

SYSTEM_RISK_NEUTRAL = """\
Du bist der Neutrale Risk-Analyst. Du wägst Chancen und Risiken ausgewogen ab \
und bewertest den Trade sachlich.

Schwerpunkte:
- Volatilität: annualisierte Volatilität und implizite Schwankungsbreite.
- Drawdown: realistisches maximales Verlustszenario für die Position.
- Positionsgröße: Passung zum rechnerischen Volatility-Targeting \
(Risiko-Budget 2%, Cap 10%).
- Marktbedingungen: Trend, Bewertung, Makro-Umfeld, anstehende Katalysatoren.

Nenne konkrete Auflagen, wenn der Trade nur mit Bedingungen tragbar ist. \
Antworte auf Deutsch in 3-6 Sätzen als Fließtext (kein JSON).
"""

SYSTEM_RISK_CONSERVATIVE = """\
Du bist der Konservative Risk-Analyst. Du priorisierst Kapitalerhalt und bist \
skeptisch gegenüber hohen Positionsgrößen.

Schwerpunkte:
- Drawdown-Risiko: Wie tief kann die Position im schlechtesten Fall fallen?
- Währungsrisiko: zusätzliche Schwankung bei nicht-EUR-Währung (Wechselkurs).
- Konzentrationsrisiko: Klumpenrisiko mit bestehenden Positionen und Sektoren.
- Position und Stops: kleine Positionen, enge Stop-Loss-Margen, Nachschieben \
statt Alles-oder-Nichts.

Verlange enge Stops und kleine Positionen, wenn die Datenlage unklar ist. \
Antworte auf Deutsch in 3-6 Sätzen als Fließtext (kein JSON).
"""

SYSTEM_RISK_SYNTHESIS = """\
Du bist der Risk-Manager und fasst die Risiko-Debatte zusammen. Dir liegen die \
Argumente von drei Perspektiven (aggressiv, neutral, konservativ) über 2 Runden \
vor. Triff eine ausgewogene finale Risiko-Bewertung.

Gewichtung: Bei hohem Risiko (hohe Volatilität, großer geschätzter Drawdown, \
Währungsrisiko, Konzentrationsrisiko) sollen die konservative und die neutrale \
Sicht mehr Gewicht haben. Bei klarem Datengerüst und niedrigem Risiko darf die \
aggressive Sicht die Empfehlung anheben. Berücksichtige die rechnerischen Werte \
(Volatilität, Volatility-Targeting-Positionsgröße) — dein \
"positionsgröße_empfohlen" sollte in der Nähe der rechnerischen Positionsgröße \
liegen, sofern die Debatte keine begründete Abweichung ergibt.

Berücksichtige bei nicht-EUR-Währung zusätzliches Währungsrisiko \
(Wechselkursbewegung) im Risiko-Score.

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{
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
  "einstiegs_level": "Limit-Order-Preis für den Einstieg als Zahl oder null — aus dem Original-Trade übernehmen oder anpassen (bei KAUFEN: Support-Level oder aktueller Kurs; bei HALTEN/VERKAUFEN: null)",
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


def _fmt_pct(val: Any) -> str:
    """Formatiert einen Anteil (0.35 = 35%) als Prozent-String.

    Für None/NaN → 'N/A'. Multipliziert mit 100 und hängt ' %' an.
    """
    if val is None:
        return "N/A"
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if fval != fval:  # NaN
        return "N/A"
    return f"{fval * 100:.1f} %"


def _build_data_text(data: dict[str, Any], role: str = "alle") -> str:
    """Erstellt einen kompakten deutschen Daten-Text für LLM-Prompts.

    Args:
        data: Daten-dict aus collect_ticker_data.
        role: Rollenspezifische Filterung der Daten-Sektionen.
            - ``"alle"`` (Default): alle Sektionen (rückwärtskompatibel).
            - ``"fundamental"``: Aktien-Identität, Datenqualitäts-Warnungen,
              FUNDAMENTALS, Analysten-Erwartungen, Insider-Transaktionen,
              Makro/Zinsen, Peer-Vergleich.
              Keine TECHNIK- oder SENTIMENT-Sektion.
            - ``"technik"``: Aktien-Identität, TECHNIK-Sektion, aktueller Kurs,
              Makro-Zinstrend (kurz). Keine FUNDAMENTALS- oder SENTIMENT-Sektion.
            - ``"sentiment"``: Aktien-Identität, SENTIMENT-Sektion, Headlines.
              Keine FUNDAMENTALS- oder TECHNIK-Sektion.
            - ``"macro_news"``: Aktien-Identität, MAKRO-Sektion (vollständig),
              Global-Makro-News, Prediction Markets, SENTIMENT-Sektion und
              Headlines. Keine FUNDAMENTALS- oder TECHNIK-Sektion, kein
              Währungsrisiko-Block.
            - ``"social"``: Aktien-Identität und SOCIAL-MEDIA-Sektion
              (StockTwits-/Reddit-Posts). Keine Headlines, FUNDAMENTALS-,
              TECHNIK- oder MAKRO-Sektion.

    Der Prolog (Aktien-Identität + INSTRUMENT-KONTEXT) ist rollenunabhängig
    immer enthalten. Im TECHNIK-Block sind die Werte als verbindlicher
    Markt-Snapshot (Ground-Truth) gekennzeichnet.
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

    # Instrument-Identity (Roadmap C3) — immer enthalten (alle Rollen).
    # Zeigt die deterministisch aufgelösten Firmen-/Instrument-Fakten aus
    # yfinance (Land, Börse, Währung, Typ), damit kein Agent das Unternehmen
    # aus dem Chart "erfindet". Felder sind optional — fehlen sie, wird der
    # Block weggelassen (kein Crash, kein N/A-Rauschen).
    _boerse = f.get("full_exchange_name") or f.get("exchange")
    _identity_fields: list[str] = [
        f.get("instrument_type"),
        _boerse,
        f.get("country"),
        f.get("currency"),
        f.get("market"),
    ]
    if any(_identity_fields):
        lines.append("")
        lines.append("=== INSTRUMENT-KONTEXT ===")
        ident_parts: list[str] = []
        if f.get("instrument_type"):
            ident_parts.append(f"Typ: {f['instrument_type']}")
        if _boerse:
            ident_parts.append(f"Börse: {_boerse}")
        if f.get("country"):
            ident_parts.append(f"Land: {f['country']}")
        if ident_parts:
            lines.append("  " + " | ".join(ident_parts))
        # Währung/Markt in zweiter Zeile (falls vorhanden)
        waehrung_markt: list[str] = []
        if f.get("currency"):
            waehrung_markt.append(f"Währung: {f['currency']}")
        if f.get("market"):
            waehrung_markt.append(f"Markt: {f['market']}")
        if waehrung_markt:
            lines.append("  " + " | ".join(waehrung_markt))

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
            f"  Umsatzwachstum: {_fmt_pct(f.get('revenue_growth'))} (pro Jahr)",
            f"  Gewinnmarge: {_fmt_pct(f.get('profit_margin'))}",
            f"  PEG: {_fmt_num(f.get('peg_ratio'))}",
            f"  Dividendenrendite: {_fmt_pct(f.get('dividend_yield'))}",
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
            # Erweiterte Fundamentalkennzahlen
            f"  Free Cash Flow: {_fmt_num(f.get('free_cash_flow'), ' ')}",
            f"  Operating Cash Flow: {_fmt_num(f.get('operating_cashflow'), ' ')}",
            f"  Nettoverschuldung: {_fmt_num(f.get('net_debt'), ' ')}",
            f"  FCF-Marge: {_fmt_num(f.get('fcf_margin'))} %",
            f"  Net-Debt/EBITDA: {_fmt_num(f.get('net_debt_to_ebitda'))}",
            f"  EBITDA: {_fmt_num(f.get('ebitda'), ' ')}",
            f"  Gesamtverschuldung: {_fmt_num(f.get('total_debt'), ' ')}",
            f"  Liquidität (Cash): {_fmt_num(f.get('total_cash'), ' ')}",
            f"  Current Ratio: {_fmt_num(f.get('current_ratio'))}",
            f"  ROE: {_fmt_num(f.get('return_on_equity'))}",
            f"  Bruttomarge: {_fmt_num(f.get('gross_margin'))}",
            f"  operatives Marge: {_fmt_num(f.get('operating_margin'))}",
            f"  Price-to-Book: {_fmt_num(f.get('price_to_book'))}",
            f"  Buchwert: {_fmt_num(f.get('book_value'))}",
            f"  Forward EPS: {_fmt_num(f.get('forward_eps'))}",
            f"  Forward KGV: {_fmt_num(f.get('forward_pe'))}",
        ])
        # PEG-Konsistenz-Warnung nur anzeigen wenn gesetzt
        peg_warnung = f.get("peg_konsistenz_warnung")
        if peg_warnung:
            lines.append(f"  ⚠ PEG-Konsistenz: {peg_warnung}")
        # Quantitativer Multi-Faktor-Score-Anker (deterministischer Referenzwert)
        # current_price aus technicals durchreichen, damit der Momentum-Score
        # die 52W-Nähe-Komponente berechnen kann (factors.py liet f.get("current_price")).
        mf = compute_multi_factor_score({**f, "current_price": t.get("current_price")})
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

    # Währungsrisiko-Block — für alle und risk (risk_manager bekommt "alle")
    if f.get("eur_risiko") and role in ("alle", "risk", "fundamental"):
        currency = f.get("currency", "USD")
        eurusd_val = f.get("eurusd")
        eurusd_str = _fmt_num(eurusd_val) if eurusd_val is not None else "N/A"
        lines.append("")
        lines.append("=== WÄHRUNGSRISIKO ===")
        lines.append(
            f"  Dieser Ticker notiert in {currency} (nicht EUR). "
            "Für einen EUR-basierten Anleger besteht Wechselkursrisiko."
        )
        lines.append(f"  EURUSD: {eurusd_str}. Berücksichtige das Währungsrisiko in deiner Risikobewertung.")

    # TECHNIK — für technik und alle
    if role in ("alle", "technik"):
        lines.extend([
            "",
            "=== TECHNIK ===",
            # Roadmap C2: Verbindlicher Markt-Snapshot als Ground-Truth-Anker —
            # der Technik-Analyst soll exakt diese Zahlen übernehmen statt
            # Kurs-/Indikator-Werte zu halluzinieren.
            "  (VERBINDLICHE QUELLE: Diese Werte sind der verifizierte Markt-Snapshot.",
            "  Nutze exakt diese Zahlen für Kurs-, SMA-, RSI- und MACD-Angaben — erfinde keine abweichenden Werte.)",
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
        # Relatives Momentum (cross-sectional, Phase 2) — nur anzeigen, wenn
        # verfügbar (best effort: fehlende Benchmark → kein N/A-Rauschen).
        if t.get("relatives_momentum_6m") is not None or t.get("momentum_6m") is not None:
            lines.extend([
                f"  Momentum 6M: {_fmt_num(t.get('momentum_6m'))} %",
                f"  S&P 500 Momentum 6M: {_fmt_num(t.get('sp500_momentum_6m'))} %",
                f"  Relatives Momentum 6M: {_fmt_num(t.get('relatives_momentum_6m'))} Prozentpunkte",
                "  (Relatives Momentum = 6M-Rendite Ticker minus 6M-Rendite S&P 500.",
                "   Positiv = Outperformance vs. Markt, negativ = Underperformance.)",
            ])

    # Makro/Zins-Daten — für alle (vollständig), fundamental (vollständig),
    # macro_news (vollständig), technik (nur Zinstrend-Kurzform), sentiment (keine)
    if macro:
        if role in ("alle", "fundamental", "macro_news"):
            lines.append("")
            lines.append("=== MAKRO / ZINSEN ===")
            lines.append(f"  10y US Treasury Yield: {_fmt_num(macro.get('us_10y_yield'))} %")
            lines.append(f"  10y Yield vor 1 Monat: {_fmt_num(macro.get('us_10y_yield_1m_ago'))} %")
            lines.append(f"  10y Zinstrend: {macro.get('us_10y_trend', 'N/A')}")
            source = macro.get("sp500_source", "")
            source_label = f" ({source})" if source and source != "none" else ""
            lines.append(f"  S&P 500 KGV{source_label}: {_fmt_num(macro.get('sp500_pe'))}")
            lines.append(f"  S&P 500 Marktkap: {_fmt_num(macro.get('sp500_market_cap'), ' ')}")
            # Erweiterte Makro-Kennzahlen
            macro_extra: list[str] = []
            if macro.get("eurusd") is not None:
                macro_extra.append(f"EURUSD: {_fmt_num(macro.get('eurusd'))}")
            if macro.get("vix") is not None:
                macro_extra.append(f"VIX: {_fmt_num(macro.get('vix'))}")
            if macro.get("oel_preis") is not None:
                macro_extra.append(f"Öl (WTI): {_fmt_num(macro.get('oel_preis'))}")
            if macro.get("sp500_trend") is not None:
                macro_extra.append(f"S&P500-Trend: {macro.get('sp500_trend')}")
            if macro_extra:
                lines.append(f"  {' | '.join(macro_extra)}")
                lines.append(
                    "  Hinweis: Erhöhter VIX (>20) = Risiko-Off-Regime, "
                    "Ölpreis relevant für Energie/Kapitalkosten."
                )
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

    # Insider-Transaktionen — für fundamental und alle (Phase A)
    insider_tx = data.get("insider_transactions", [])
    if role in ("alle", "fundamental") and insider_tx:
        lines.append("")
        lines.append("=== INSIDER-TRANSAKTIONEN ===")
        lines.append("  (Neueste Transaktionen — Käufe/Verkäufe von Insidern, best-effort)")
        for tx in insider_tx[:8]:
            parts: list[str] = []
            if tx.get("date"):
                parts.append(str(tx["date"])[:10])
            if tx.get("insider"):
                parts.append(str(tx["insider"]))
            if tx.get("transaction"):
                parts.append(str(tx["transaction"]))
            if tx.get("shares") is not None:
                parts.append(f"{_fmt_num(tx['shares'], ' ')} Aktien")
            if tx.get("price") is not None:
                parts.append(f"Kurs {_fmt_num(tx['price'])}")
            if tx.get("value") is not None:
                parts.append(f"Wert {_fmt_num(tx['value'], ' ')}")
            if parts:
                lines.append(f"  - {' | '.join(parts)}")

    # Global-Makro-News — für macro_news und alle (Phase A)
    global_macro = data.get("global_macro_news", [])
    if role in ("alle", "macro_news") and global_macro:
        lines.append("")
        lines.append("=== GLOBAL-MAKRO-NEWS ===")
        lines.append("  (Globale Konjunktur-/Zins-/Geopolitik-Headlines, nicht ticker-spezifisch)")
        for item in global_macro[:10]:
            title = item.get("title") if isinstance(item, dict) else None
            if title:
                lines.append(f"    - {title}")

    # Prediction Markets — für macro_news und alle (Phase A)
    pred_markets = data.get("prediction_markets", [])
    if role in ("alle", "macro_news") and pred_markets:
        lines.append("")
        lines.append("=== PREDICTION MARKETS ===")
        lines.append("  (Polymarket-Wahrscheinlichkeiten, best-effort — falls Daten verfügbar)")
        for m in pred_markets[:5]:
            m_title = m.get("title")
            if not m_title:
                continue
            prob = m.get("probability")
            prob_str = f"{prob * 100:.0f} %" if prob is not None else "N/A"
            m_category = m.get("category")
            cat_str = f" [{m_category}]" if m_category else ""
            lines.append(f"    - {m_title}{cat_str}: {prob_str}")

    # SENTIMENT — für sentiment, macro_news und alle
    if role in ("alle", "sentiment", "macro_news"):
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

    # SOCIAL MEDIA (StockTwits/Reddit) — für social und alle
    # Rückwärtskompatibel: Die Posts fließen weiterhin über news_with_dates in
    # die SENTIMENT-Zählung ein; dieser Block reicht sie zusätzlich separat
    # durch (ohne Nachrichten-Headlines), damit der Social-Media-Analyst nur
    # die Retail-Community-Stimmung bewertet.
    social_items = data.get("stocktwits_items", []) + data.get("reddit_items", [])
    if role in ("alle", "social") and social_items:
        lines.append("")
        lines.append("=== SOCIAL MEDIA (StockTwits/Reddit) ===")
        lines.append(f"  Anzahl StockTwits-Posts: {len(data.get('stocktwits_items', []))}")
        lines.append(f"  Anzahl Reddit-Posts: {len(data.get('reddit_items', []))}")
        lines.append("  Posts (neueste, gekürzt):")
        for item in social_items[:10]:
            title = item.get("title") if isinstance(item, dict) else None
            if not title:
                continue
            source = item.get("source", "?") if isinstance(item, dict) else "?"
            lines.append(f"    - [{source}] {str(title)[:300]}")
    elif role == "social":
        # Keine Social-Daten → explizit melden (best effort, kein Crash):
        # Der Social-Media-Analyst soll transparent neutral bleiben, statt
        # eine Stimmung aus fehlenden Daten zu erfinden.
        lines.append("")
        lines.append("=== SOCIAL MEDIA (StockTwits/Reddit) ===")
        lines.append("  Keine StockTwits- oder Reddit-Posts verfügbar.")

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


def _call_agent(
    llm: LLMClient,
    system_prompt: str,
    user_text: str,
    temperature: float = 0.3,
    response_format: dict[str, Any] | None = None,
    structured: bool = False,
    max_tokens: int = 4000,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Führt einen einzelnen Agenten-Call aus und parst das Ergebnis.

    Bei ``structured=True`` und gesetztem ``response_format`` wird
    ``llm.chat(...)`` mit ``as_structured=True`` aufgerufen. Wenn der
    Provider response_format unterstützt (``response_format_used=True``),
    wird der Text direkt als JSON geparsed (json.loads). Andernfalls
    (Fallback-Pfad) wird ``parse_json`` verwendet.

    Bei ``structured=False`` (Default) wird kein response_format gesendet und
    das Ergebnis wird wie bisher via ``parse_json`` extrahiert.

    In beiden Fällen wird ``_raw`` auf den rohen Text gesetzt, damit
    Downstream-Consumer (z.B. _clean_debate_text, _parse_debate_confidence)
    darauf zugreifen können.

    ``max_tokens`` (Default 4000) caps die Output-Länge pro LLM-Call, um
    Token-Verschwendung durch ausufernde Antworten zu verhindern. 4000 ist
    großzügig für alle Agenten-Rollen (Debatte ~300-400, Analysten ~150,
    Trader/Risk/PM ~200) und nötig, weil die Reasoning-Modelle (glm-5.x)
    einen Teil der Tokens fürs Reasoning verbrauchen — bei 1000 blieb der
    eigentliche Content leer (finish_reason=length).

    ``model``: Optionales Modell-Override (Deep-Think/Quick-Think-Split),
    wird an ``llm.chat(model=...)`` durchgereicht. None (Default) = primäres
    Modell des Clients (bisheriges Verhalten).

    ``reasoning_effort``: Optionale Reasoning-Tiefe ('low'/'medium'/'high'),
    wird an ``llm.chat(reasoning_effort=...)`` durchgereicht. None/''
    (Default) = kein reasoning_effort im Payload (bisheriges Verhalten).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    if structured and response_format is not None:
        result_obj = llm.chat(
            messages,
            temperature=temperature,
            response_format=response_format,
            as_structured=True,
            max_tokens=max_tokens,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        if isinstance(result_obj, StructuredChatResult):
            raw = result_obj.text
            if result_obj.response_format_used:
                # Strukturierter Pfad: direktes json.loads
                try:
                    parsed = json.loads(raw) if raw else {}
                except (json.JSONDecodeError, TypeError):
                    parsed = parse_json(raw)
            else:
                # Fallback-Pfad: parse_json auf Fließtext
                parsed = parse_json(raw)
        else:
            # Sollte nicht passieren, aber sicherheitshalber
            raw = str(result_obj)
            parsed = parse_json(raw)
    else:
        raw = llm.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        parsed = parse_json(raw)

    if not isinstance(parsed, dict):
        parsed = {"_raw": raw}

    # Struktur-Garantie: fehlende Schema-Felder mit sicheren Defaults auffüllen.
    # Nur im strukturierten Pfad (response_format gesetzt) — stellt sicher,
    # dass das zurückgegebene dict IMMER alle Schema-Keys enthält.
    # setdefault überschreibt keine vorhandenen Werte (Modell-Antwort hat Vorrang).
    if structured and response_format is not None:
        defaults = defaults_for_schema(response_format)
        for key, default in defaults.items():
            parsed.setdefault(key, default)

    parsed["_raw"] = raw
    return parsed


def analyst_team(
    data: dict[str, Any],
    llm: LLMClient,
    data_text: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Ruft 5 Analysten-Rollen auf (Fundamental, Technical, Sentiment,
    Macro/News, Social-Media).

    Returns dict mit 'fundamental', 'technical', 'sentiment', 'macro_news',
    'social' und 'technicals' Schlüsseln.
    Die 5 Analysten-Calls werden PARALLEL über ThreadPoolExecutor ausgeführt.
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
        model: Optionales Modell-Override (Quick-Think-Split), wird an alle
            Analysten-Calls durchgereicht. None = primäres Modell.
        reasoning_effort: Optionale Reasoning-Tiefe ('low'/'medium'/'high'),
            wird an alle Analysten-Calls durchgereicht. None/'' (Default) =
            kein reasoning_effort im Payload (bisheriges Verhalten).
    """
    # Rollen-Mapping: analyst_team key → _build_data_text role
    role_map = {
        "fundamental": "fundamental",
        "technical": "technik",
        "sentiment": "sentiment",
        "macro_news": "macro_news",
        "social": "social",
    }

    # (key, system_prompt, response_format) — strukturierte Schemas pro Rolle
    analyst_specs = [
        ("fundamental", SYSTEM_FUNDAMENTAL, ANALYST_FUNDAMENTAL_SCHEMA),
        ("technical", SYSTEM_TECHNICAL, ANALYST_TECHNICAL_SCHEMA),
        ("sentiment", SYSTEM_SENTIMENT, ANALYST_SENTIMENT_SCHEMA),
        ("macro_news", SYSTEM_MACRO_NEWS, ANALYST_MACRO_NEWS_SCHEMA),
        ("social", SYSTEM_SOCIAL, ANALYST_SOCIAL_SCHEMA),
    ]

    results: dict[str, Any] = {}

    def _run_one(key: str, system_prompt: str, resp_format: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if data_text is not None:
            text = data_text
        else:
            text = _build_data_text(data, role=role_map[key])
        return key, _call_agent(
            llm,
            system_prompt,
            text,
            response_format=resp_format,
            structured=True,
            model=model,
            reasoning_effort=reasoning_effort,
        )

    max_workers = min(len(analyst_specs), _MAX_PARALLEL)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, key, prompt, fmt): key for key, prompt, fmt in analyst_specs}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                _, result = future.result()
                results[key] = result
            except Exception as exc:  # noqa: BLE001 — nie crashen
                logger.warning("Analyst '%s' fehlgeschlagen: %s", key, exc)
                results[key] = {"_raw": "", "fehler": str(exc)}

    # Sicherstellen, dass alle 5 Keys vorhanden sind (defensiv)
    for key, _, _ in analyst_specs:
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


def _parse_debate_confidence(agent: dict[str, Any]) -> int | None:
    """Extrahiert die confidence (1-5) aus einem Bull/Bear-Agent-Dict.

    Versucht zuerst, den JSON-Block aus ``agent["_raw"]`` via parse_json zu
    extrahieren. Wenn dort eine ``confidence`` gefunden wird, wird sie als
    int zurückgegeben. Als Fallback wird ein direktes Feld ``confidence``
    im Agent-Dict geprüft.

    Gibt None bei Fehler oder fehlender confidence zurück.
    """
    if not isinstance(agent, dict):
        return None

    # 1. Versuch: direktes Feld confidence im Agent-Dict
    direct = agent.get("confidence")
    if direct is not None:
        try:
            return int(direct)
        except (TypeError, ValueError):
            pass

    # 2. Versuch: JSON-Block aus _raw parsen
    raw = agent.get("_raw", "")
    if raw:
        parsed = parse_json(raw)
        conf = parsed.get("confidence")
        if conf is not None:
            try:
                return int(conf)
            except (TypeError, ValueError):
                pass

    return None


def _debate_skew_text(bull_conf: int | None, bear_conf: int | None) -> str:
    """Baut einen deutschen Kontext-Block über die Debatten-Konfidenz für den Trader.

    - Beide Konfidenzen vorhanden: "Debatten-Konfidenz: Bull X/5 vs Bear Y/5
      (Nettoneigung: Z)." mit Z = round((bull_conf - bear_conf) / 2, 1).
      Bei Z > 0.5: "die Bull-Seite hat die Oberhand."
      Bei Z < -0.5: "die Bear-Seite hat die Oberhand."
      Sonst: "ausgewogene Debatte."
    - Nur eine Seite verfügbar: Zeigt die verfügbare Konfidenz.
    - Keine Konfidenz → leerer String "".

    Sachlich, kein Alarmismus.
    """
    if bull_conf is None and bear_conf is None:
        return ""

    if bull_conf is not None and bear_conf is not None:
        skew = round((bull_conf - bear_conf) / 2, 1)
        if skew > 0.5:
            tendenz = "die Bull-Seite hat die Oberhand."
        elif skew < -0.5:
            tendenz = "die Bear-Seite hat die Oberhand."
        else:
            tendenz = "ausgewogene Debatte."
        return (
            f"Debatten-Konfidenz: Bull {bull_conf}/5 vs Bear {bear_conf}/5 "
            f"(Nettoneigung: {skew:+.1f}) — {tendenz}"
        )

    # Nur eine Seite verfügbar
    if bull_conf is not None:
        return f"Debatten-Konfidenz: Bull {bull_conf}/5 (Bear-Seite nicht verfügbar)."
    return f"Debatten-Konfidenz: Bear {bear_conf}/5 (Bull-Seite nicht verfügbar)."


def _analyst_summary_text(analysts: dict[str, Any]) -> str:
    """Kompakte Zusammenfassung aller Analysten für Debatte/Trader."""
    parts = []
    for role_key, label in [
        ("fundamental", "Fundamental-Analyst"),
        ("technical", "Technik-Analyst (Timing-Filter, nicht Richtung)"),
        ("sentiment", "Sentiment-Analyst"),
        ("macro_news", "Makro/News-Analyst"),
        ("social", "Social-Media-Analyst"),
    ]:
        a = analysts.get(role_key, {})
        parts.append(
            f"{label}: Stimmung={a.get('stimmung', 'N/A')}, "
            f"Score={a.get('score', 'N/A')}, "
            f"Zusammenfassung={a.get('zusammenfassung', a.get('_raw', 'N/A'))[:300]}"
        )
    return "\n".join(parts)


def debate(
    analysts: dict[str, Any],
    llm: LLMClient,
    rounds: int = 1,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Führt Bull/Bear-Debatte durch (2 LLM-Calls pro Runde).

    Args:
        analysts: Analysten-Ergebnisse.
        llm: LLMClient.
        rounds: Anzahl der Bull/Bear-Runden (Default 1 = bisheriges
            Verhalten, rückwärtskompatibel). Bei rounds > 1 bekommt jede
            Seite ab Runde 2 die Argumentation der Gegenseite aus der
            vorherigen Runde als Kontext, mit der Anweisung, konkret darauf
            einzugehen.
        model: Optionales Modell-Override (Quick-Think-Split), wird an beide
            Bull/Bear-Calls durchgereicht. None = primäres Modell.
        reasoning_effort: Optionale Reasoning-Tiefe ('low'/'medium'/'high'),
            wird an beide Bull/Bear-Calls durchgereicht. None/'' (Default) =
            kein reasoning_effort im Payload (bisheriges Verhalten).

    Returns dict mit 'bull', 'bear', 'bull_confidence', 'bear_confidence'
    und 'rounds' Schlüsseln. Die finalen bull/bear-Dicts stammen aus der
    letzten Runde. Die Konfidenzen werden via _parse_debate_confidence aus
    den jeweiligen Agent-Ergebnissen extrahiert (None bei Fehlschlag).

    Verwendet DEBATE_SCHEMA für strukturierte LLM-Outputs. Bei Fallback
    (Provider unterstützt response_format nicht) wird die Konfidenz aus dem
    Fließtext via _parse_debate_confidence extrahiert.
    """
    summary = _analyst_summary_text(analysts)
    bull_text = ""
    bear_text = ""
    bull: dict[str, Any] = {}
    bear: dict[str, Any] = {}

    for r in range(rounds):
        bull_user = f"Analysten-Einschätzungen:\n{summary}"
        if r > 0 and bear_text:
            bull_user += (
                "\n\n=== Argumentation der Gegenseite (letzte Runde) ===\n"
                f"{bear_text}\n\n"
                "Gehe konkret auf diese Argumente ein: stimme zu, widersprich "
                "mit Daten, oder ergänze. Wiederhole nicht deine vorherigen "
                "Punkte, sondern vertiefe/verteidige sie."
            )
        bull = _call_agent(
            llm, SYSTEM_BULL,
            bull_user,
            temperature=0.5,
            response_format=DEBATE_SCHEMA,
            structured=True,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        bull_text = _get_debate_argument(bull)

        bear_user = f"Analysten-Einschätzungen:\n{summary}"
        if r > 0 and bull_text:
            bear_user += (
                "\n\n=== Argumentation der Gegenseite (letzte Runde) ===\n"
                f"{bull_text}\n\n"
                "Gehe konkret auf diese Argumente ein: stimme zu, widersprich "
                "mit Daten, oder ergänze. Wiederhole nicht deine vorherigen "
                "Punkte, sondern vertiefe/verteidige sie."
            )
        bear = _call_agent(
            llm, SYSTEM_BEAR,
            bear_user,
            temperature=0.5,
            response_format=DEBATE_SCHEMA,
            structured=True,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        bear_text = _get_debate_argument(bear)

    bull_conf = _parse_debate_confidence(bull)
    bear_conf = _parse_debate_confidence(bear)

    return {
        "bull": bull,
        "bear": bear,
        "bull_confidence": bull_conf,
        "bear_confidence": bear_conf,
        "rounds": rounds,
    }


def _get_debate_argument(agent: dict[str, Any]) -> str:
    """Extrahiert den Debatten-Fließtext aus einem Bull/Bear-Agent-Dict.

    Liest zuerst das ``argumente``-Feld (strukturierter Pfad), dann
    ``_raw`` (Fallback-Pfad). Gibt einen leeren String zurück, wenn keines
    vorhanden ist.
    """
    if not isinstance(agent, dict):
        return ""
    val = agent.get("argumente")
    if val and str(val).strip():
        return str(val)
    return str(agent.get("_raw", ""))


def _safe_float_or_none(val: Any) -> float | None:
    """Konvertiert einen Wert tolerant zu float (None/NaN/ungültig → None)."""
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN-Check (NaN != NaN)


# ---------------------------------------------------------------------------
# Technik-Signal (SMA200) — fallendes Messer skaliert Positionen graduell
# ---------------------------------------------------------------------------

def _technik_signal(analysts: dict[str, Any]) -> dict[str, Any]:
    """Prüft das graduelle Technik-Signal (SMA200) — skaliert Positionen statt zu blocken.

    Returns dict mit:
      - faktor: float 0.0–1.0 (Positions-Skalierungsfaktor)
      - grund: str (deutscher Grund, leer wenn kein Abschlag)
      - ausnahme: bool (True wenn RSI-Ausnahme greift)

    Konservativ: Wenn current_price oder sma200 fehlen/None/NaN sind, wird
    NICHT abgeschlagen (kein Skalieren, wenn Daten fehlen).
    Kurs >= SMA200 → faktor 1.0 (kein Abschlag).
    Kurs < SMA200:
      - RSI-Ausnahme (RSI < 30 bei intaktem SMA50-Umfeld): faktor 0.5,
        ausnahme=True (kleine Position mit strengem Stop).
      - Sonst: gradueller Faktor nach Abstand unter SMA200 —
        max(0.3, 1.0 - abstand_pct / 10.0) (bei 7%+ unter SMA200 → Untergrenze 0.3).
    """
    no_discount = {"faktor": 1.0, "grund": "", "ausnahme": False}
    t = analysts.get("technicals") if isinstance(analysts, dict) else None
    if not isinstance(t, dict):
        return no_discount

    price = _safe_float_or_none(t.get("current_price"))
    sma200 = _safe_float_or_none(t.get("sma200"))
    if price is None or sma200 is None or sma200 <= 0:
        return no_discount

    if price >= sma200:
        return no_discount  # kein Abschlag — Kurs auf oder über SMA200

    # Kurs unter SMA200 → RSI-Ausnahme (fester Faktor 0.5) oder gradueller Faktor
    rsi = _safe_float_or_none(t.get("rsi14", t.get("rsi")))
    sma50 = _safe_float_or_none(t.get("sma50"))
    if rsi is not None and rsi < 30 and sma50 is not None and price > sma50:
        return {
            "faktor": 0.5,
            "grund": (
                "RSI < 30 bei intaktem SMA50-Umfeld — kleine Position "
                "erlaubt (Technik-Ausnahme)."
            ),
            "ausnahme": True,
        }

    abstand_pct = (sma200 - price) / sma200 * 100.0
    faktor = max(0.3, 1.0 - abstand_pct / 10.0)
    grund = (
        f"Kurs {abstand_pct:.1f}% unter SMA200 — Positionsgröße um Faktor "
        f"{faktor:.2f} reduziert (fallendes Messer)."
    )
    return {"faktor": faktor, "grund": grund, "ausnahme": False}


def _technik_signal_basis(trade: dict[str, Any]) -> float:
    """Basis-Positionsgröße für die Signal-Skalierung (idempotenter Re-Apply).

    Liest die beim ersten Apply gespeicherte Original-Größe
    (``_technik_signal_basis``) und fällt sonst auf ``positionsanteil``
    (Default 3.0) zurück. Ohne gespeicherte Basis bleibt eine erneute
    Anwendung des Signals wirkungslos-idempotent statt kumulativ zu skalieren.
    """
    stored = _safe_float_or_none(trade.get("_technik_signal_basis"))
    if stored is not None:
        return stored
    current = _safe_float_or_none(trade.get("positionsanteil"))
    if current is not None:
        return current
    return 3.0


def _apply_technik_signal(trade: dict[str, Any], analysts: dict[str, Any]) -> dict[str, Any]:
    """Wendet das graduelle Technik-Signal auf ein Trade-dict an (in-place, gibt trade zurück).

    - Nur für KAUFEN/STARK KAUFEN (sonst unverändert) — KAUFEN bleibt KAUFEN.
    - faktor >= 1.0 → unverändert, kein _technik_signal-Key.
    - faktor < 1.0 → Positionsgröße mit dem Faktor skaliert (Basis: beim ersten
      Apply gespeicherte Original-Größe, sonst aktueller Wert, Default 3.0;
      Untergrenze 0.5; idempotent). Ziel-/Stop-Werte bleiben unangetastet.
    - RSI-Ausnahme (ausnahme=True) → zusätzlich Stop deterministisch auf 5%
      unter Kurs gesetzt (falls fehlend oder zu locker).
    - Metadaten trade["_technik_signal"] gesetzt (faktor/grund/ausnahme).
    - Crasht nie (try/except um die Skalierung; bei Fehler Trade unverändert).
    """
    action = str(trade.get("aktion", "")).strip().upper()
    if action not in ("KAUFEN", "STARK KAUFEN"):
        return trade

    try:
        signal = _technik_signal(analysts)
        faktor = float(signal.get("faktor", 1.0))
        if faktor >= 1.0:
            return trade

        # Unpassender positionsanteil-Typ (nicht None, nicht konvertierbar) →
        # Abbruch, Trade bleibt unverändert (keine Datenverfälschung).
        raw_pa = trade.get("positionsanteil")
        if raw_pa is not None and _safe_float_or_none(raw_pa) is None:
            return trade

        original = _technik_signal_basis(trade)
        neuer_pa = max(0.5, round(original * faktor, 2))
        metadaten = {
            "faktor": faktor,
            "grund": signal.get("grund", ""),
            "ausnahme": bool(signal.get("ausnahme", False)),
        }

        # RSI-Ausnahme: Stop deterministisch auf 5% unter Kurs (falls fehlend
        # oder zu locker) — nur anwenden, wenn der Preis verfügbar ist.
        neuer_stop = None
        if signal.get("ausnahme"):
            price = _safe_float_or_none(
                (analysts.get("technicals") or {}).get("current_price")
            )
            if price is not None:
                try:
                    stop = trade.get("stop_loss")
                    if stop is None or float(stop) > price * 0.95:
                        neuer_stop = round(price * 0.95, 2)
                except (TypeError, ValueError):
                    neuer_stop = round(price * 0.95, 2)

        # Mutationen erst am Ende (alle Werte vorberechnet) — der Trade bleibt
        # bei Fehlern im Vorfeld unverändert.
        trade["positionsanteil"] = neuer_pa
        if neuer_stop is not None:
            trade["stop_loss"] = neuer_stop
        trade["_technik_signal"] = metadaten
        trade["_technik_signal_basis"] = original
    except Exception:  # noqa: BLE001 — Trade-Änderung darf nie crashen
        return trade
    return trade


def _cap_position_by_volatility(
    trade: dict[str, Any],
    risk: dict[str, Any],
) -> dict[str, Any]:
    """Kappt die Positionsgröße am rechnerischen Volatility-Targeting (in-place).

    Risk-Parity-Praxis: Die rechnerische Positionsgröße aus dem Risikomodell
    (``positionsgröße_rechnerisch_pct``) ist eine harte Obergrenze — der LLM
    darf nicht mehr Position empfehlen, als das Volatility-Targeting erlaubt.

    - Nur für KAUFEN/STARK KAUFEN (sonst unverändert — HALTEN/VERKAUFEN
      wird nie angetastet).
    - ``positionsgröße_rechnerisch_pct`` fehlt/None/ungültig → Trade
      unverändert (kein Cap möglich, rückwärtskompatibel).
    - ``positionsanteil`` größer als der rechnerische Wert → auf den
      rechnerischen Wert gekappt (2 Dezimalstellen). Ist die Position
      bereits kleiner (z. B. durch das Technik-Signal), greift kein Cap —
      der Vol-Cap senkt nur, hebt aber nie an.
    - Metadaten ``trade["_vol_cap"]`` = {"rechnerisch", "original",
      "gekappt"} werden gesetzt (gekappt=False, wenn kein Cap griff).
    - Crasht nie (try/except; bei Fehler Trade unverändert).
    """
    action = str(trade.get("aktion", "")).strip().upper()
    if action not in ("KAUFEN", "STARK KAUFEN"):
        return trade

    try:
        rechnerisch = _safe_float_or_none(
            (risk or {}).get("positionsgröße_rechnerisch_pct")
        )
        if rechnerisch is None:
            return trade

        original = _safe_float_or_none(trade.get("positionsanteil"))
        gekappt = original is not None and original > rechnerisch

        # Mutationen erst am Ende (alle Werte vorberechnet) — der Trade bleibt
        # bei Fehlern im Vorfeld unverändert.
        metadaten = {
            "rechnerisch": round(rechnerisch, 2),
            "original": original,
            "gekappt": bool(gekappt),
        }
        if gekappt:
            trade["positionsanteil"] = round(rechnerisch, 2)
        trade["_vol_cap"] = metadaten
    except Exception:  # noqa: BLE001 — Trade-Änderung darf nie crashen
        return trade
    return trade


def trader(
    analysts: dict[str, Any],
    debate_result: dict[str, Any],
    llm: LLMClient,
    temperature: float = 0.3,
    feedback_context: str = "",
    reflection_context: str = "",
    model: str | None = None,
    reasoning_effort: str | None = None,
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
        model: Optionales Modell-Override (Quick-Think-Split). None =
            primäres Modell.
        reasoning_effort: Optionale Reasoning-Tiefe ('low'/'medium'/'high'),
            wird an den Trader-Call durchgereicht. None/'' (Default) = kein
            reasoning_effort im Payload (bisheriges Verhalten).
    """
    summary = _analyst_summary_text(analysts)
    # debate_result bull/bear kann "argumente" (strukturierter Pfad) oder
    # "_raw" (Fallback-Pfad) enthalten — beide unterstützen.
    bull_text = _get_debate_argument(debate_result.get("bull", {}))
    bear_text = _get_debate_argument(debate_result.get("bear", {}))

    # Debatten-Konfidenz extrahieren und Kontext-Block bauen
    bull_conf = debate_result.get("bull_confidence")
    if bull_conf is None:
        bull_conf = _parse_debate_confidence(debate_result.get("bull", {}))
    bear_conf = debate_result.get("bear_confidence")
    if bear_conf is None:
        bear_conf = _parse_debate_confidence(debate_result.get("bear", {}))
    skew_text = _debate_skew_text(bull_conf, bear_conf)

    user_text = (
        f"Analysten-Einschätzungen:\n{summary}\n\n"
        f"Bull-Argumentation:\n{bull_text}\n\n"
        f"Bear-Argumentation:\n{bear_text}"
    )
    if skew_text:
        user_text += f"\n\n{skew_text}"
    if feedback_context:
        user_text += f"\n\n{feedback_context}"
    if reflection_context:
        user_text += f"\n\n{reflection_context}"
    result = _call_agent(
        llm, SYSTEM_TRADER, user_text,
        temperature=temperature,
        response_format=TRADE_SCHEMA,
        structured=True,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    # 5-stufige Rating normalisieren: rohes Rating in 'rating', 3-stufige Aktion in 'aktion'
    raw_rating = str(result.get("aktion", "")).strip().upper()
    result["rating"] = raw_rating
    result["aktion"] = _rating_to_action(raw_rating)
    # Entscheidungs-Disziplin: STARK KAUFEN/STARK VERKAUFEN dämpfen wenn überkonfident
    _dampen_stark_rating(result, raw_rating)
    # Technik-Signal (graduell): KAUFEN unter SMA200 bleibt KAUFEN, aber die
    # Positionsgröße wird per Faktor (0.3–1.0) reduziert; RSI-Ausnahme: Faktor
    # 0.5 + strenger Stop. Bewusst am Ende — skaliert jede vorherige Logik.
    _apply_technik_signal(result, analysts)
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


def _is_plausible_value(
    trade: dict[str, Any], field: str, current_price: float, *, expected_above: bool,
) -> bool:
    """Prüft, ob ein Ziel-/Stop-Wert plausibel bezüglich current_price ist.

    expected_above=True  → Wert muss > current_price (z.B. zielkurs bei KAUFEN).
    expected_above=False → Wert muss < current_price (z.B. stop_loss bei KAUFEN).
    """
    val = trade.get(field)
    if val is None:
        return False
    try:
        val_f = float(val)
    except (TypeError, ValueError):
        return False
    if expected_above:
        return val_f > current_price
    return val_f < current_price


def _ensure_ziel_stop(
    result: dict[str, Any], action: str, current_price: float,
) -> dict[str, Any]:
    """Setzt deterministisch plausible zielkurs/stop_loss, falls fehlend/unplausibel.

    KAUFEN / STARK KAUFEN:
      zielkurs = current_price * 1.10, stop_loss = current_price * 0.90
    VERKAUFEN / STARK VERKAUFEN:
      zielkurs = current_price * 0.90, stop_loss = current_price * 1.10
    HALTEN: keine Werte erzwungen (bleiben None wenn nicht vom Trader geliefert).

    Bereits plausible Werte werden NICHT überschrieben.
    """
    action_upper = action.strip().upper()

    if action_upper in ("KAUFEN", "STARK KAUFEN"):
        if not _is_plausible_value(result, "zielkurs", current_price, expected_above=True):
            result["zielkurs"] = round(current_price * 1.10, 2)
        if not _is_plausible_value(result, "stop_loss", current_price, expected_above=False):
            result["stop_loss"] = round(current_price * 0.90, 2)

    elif action_upper in ("VERKAUFEN", "STARK VERKAUFEN"):
        if not _is_plausible_value(result, "zielkurs", current_price, expected_above=False):
            result["zielkurs"] = round(current_price * 0.90, 2)
        if not _is_plausible_value(result, "stop_loss", current_price, expected_above=True):
            result["stop_loss"] = round(current_price * 1.10, 2)

    # HALTEN: keine Erzwingung — Trader-Werte bleiben, None bleibt None
    return result


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


# --------------------------------------------------------------------------- #
# Kalibrierungs-gewichtete Ensemble-Abstimmung
# --------------------------------------------------------------------------- #

# Maximales Alter der Kalibrierungs-JSON in Tagen (danach keine Gewichtung).
_ENSEMBLE_CALIBRATION_MAX_AGE_DAYS = 7

# Aktionen, für die Hit-Raten erwartet werden.
_ENSEMBLE_ACTIONS = ("KAUFEN", "HALTEN", "VERKAUFEN")


def _ensemble_state_dir() -> str:
    """Löst das State-Verzeichnis auf (gleicher Mechanismus wie feedback.py).

    Priorität: CONCILIUM_STATE_DIR-Env > 'state'.
    """
    return config.state_dir()


def _load_ensemble_weights() -> dict[str, float] | None:
    """Liest Hit-Raten pro Aktion aus state/calibration.json.

    Liest die gleiche JSON, die ``--evaluate`` schreibt (über cli.py), aber
    OHNE Import von feedback.py (um Zirkularität zu vermeiden).

    Returns:
        dict {aktion: hit_rate} für KAUFEN/HALTEN/VERKAUFEN, oder None bei:
        - fehlender Datei
        - ungültigem JSON
        - Datei älter als _ENSEMBLE_CALIBRATION_MAX_AGE_DAYS
        - fehlendem/ungültigem erstellt_am
        Crasht nie.
    """
    try:
        cal_path = os.path.join(_ensemble_state_dir(), "calibration.json")
        if not os.path.isfile(cal_path):
            return None
        with open(cal_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None

        # Alters-Check: erstellt_am muss vorhanden und nicht zu alt sein
        erstellt_am = data.get("erstellt_am")
        if not isinstance(erstellt_am, str) or not erstellt_am.strip():
            return None
        try:
            erstellt_dt = datetime.fromisoformat(erstellt_am)
        except (ValueError, TypeError):
            return None
        age = datetime.now() - erstellt_dt
        if age > timedelta(days=_ENSEMBLE_CALIBRATION_MAX_AGE_DAYS):
            logger.debug(
                "Kalibrierungs-JSON älter als %d Tage — keine Ensemble-Gewichtung",
                _ENSEMBLE_CALIBRATION_MAX_AGE_DAYS,
            )
            return None

        # Hit-Raten pro Aktion extrahieren
        nach_aktion = data.get("nach_aktion")
        if not isinstance(nach_aktion, dict):
            return None

        weights: dict[str, float] = {}
        for action in _ENSEMBLE_ACTIONS:
            adata = nach_aktion.get(action)
            if not isinstance(adata, dict):
                continue
            hit_rate = adata.get("hit_rate")
            if hit_rate is None or not isinstance(hit_rate, (int, float)):
                continue
            weights[action] = float(hit_rate)

        # Mindestens eine Aktion muss eine Hit-Rate haben
        if not weights:
            return None

        return weights
    except Exception as exc:  # noqa: BLE001 — crasht nie
        logger.debug("Ensemble-Gewichte konnten nicht geladen werden: %s", exc)
        return None


def _smooth_weight(hit_rate: float) -> float:
    """Glättet die Hit-Rate zu einem Gewicht im Bereich 0.5 bis 1.0.

    Formel: 0.5 + 0.5 * hit_rate
    - hit_rate 0.0 → 0.5 (Mindestgewicht, wird nie komplett ignoriert)
    - hit_rate 0.5 → 0.75
    - hit_rate 1.0 → 1.0
    """
    return 0.5 + 0.5 * max(0.0, min(1.0, hit_rate))


# --------------------------------------------------------------------------- #
# Entscheidungs-Disziplin — aggressive Ratings dämpfen bei überkonfidenter Historie
# --------------------------------------------------------------------------- #

_DAMPEN_MIN_DECISIONS = 5
_DAMPEN_GAP_THRESHOLD = 0.15
# Phase 6: Gemeinsame Schwellen-Referenz mit der Prompt-Hit-Rate-Direktive
# (feedback.py, 0.20/0.35/0.50): Bei echter Hit-Rate < 0.35 ist 'STARK …'
# verboten — die deterministische Dämpfung erzwingt das (hat Vorrang vor dem
# Prompt, das nur die zusätzliche, verhaltensleitende Ebene ist).
_DAMPEN_HITRATE_MAX = 0.35


def _should_dampen_stark(action: str | None = None) -> bool:
    """Prüft, ob aggressive Ratings (STARK KAUFEN/STARK VERKAUFEN) gedämpft werden sollen.

    Liest ``state/calibration.json`` (netzfrei, gleicher Mechanismus wie
    ``_load_ensemble_weights``: ``CONCILIUM_STATE_DIR``-Übersteuerung, <7 Tage
    aktuell, ``anzahl_entscheidungen`` >= 5).

    Gibt ``True`` zurück, wenn die Kalibrierungs-Tendenz überkonfident ist:
    - Gesamt-Gap (Ø-Confidence - hit_rate_gesamt) > 0.15, ODER
    - Gap der betroffenen Aktion (avg_confidence - hit_rate) > 0.15
      (nur geprüft, wenn ``action`` angegeben, z.B. "KAUFEN" oder "VERKAUFEN"),
      ODER
    - Echte Hit-Rate der betroffenen Aktion < 0.35 (Phase 6: gemeinsame
      Schwellen mit der Prompt-Hit-Rate-Direktive in feedback.py —
      0.20/0.35/0.50; bei unzuverlässiger Historie ist 'STARK …' verboten).

    Gibt ``False`` zurück bei fehlender/zu alter/ungültiger JSON oder
    anzahl_entscheidungen < 5. Crasht nie.
    """
    try:
        cal_path = os.path.join(_ensemble_state_dir(), "calibration.json")
        if not os.path.isfile(cal_path):
            return False
        with open(cal_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return False

        # Alters-Check
        erstellt_am = data.get("erstellt_am")
        if not isinstance(erstellt_am, str) or not erstellt_am.strip():
            return False
        try:
            erstellt_dt = datetime.fromisoformat(erstellt_am)
        except (ValueError, TypeError):
            return False
        age = datetime.now() - erstellt_dt
        if age > timedelta(days=_ENSEMBLE_CALIBRATION_MAX_AGE_DAYS):
            return False

        # Mindest-Anzahl Entscheidungen
        anzahl = data.get("anzahl_entscheidungen")
        if not isinstance(anzahl, (int, float)) or anzahl < _DAMPEN_MIN_DECISIONS:
            return False

        nach_aktion = data.get("nach_aktion")
        if not isinstance(nach_aktion, dict):
            return False

        # Gesamt-Gap: gewichtete Ø-Confidence - hit_rate_gesamt
        hit_rate_gesamt = data.get("hit_rate_gesamt")
        total_n = 0
        conf_sum = 0.0
        for a, adata in nach_aktion.items():
            if not isinstance(adata, dict):
                continue
            n = adata.get("n", 0)
            avg_conf = adata.get("avg_confidence")
            if isinstance(n, (int, float)) and n > 0 and isinstance(avg_conf, (int, float)):
                total_n += n
                conf_sum += avg_conf * n
        if total_n > 0 and isinstance(hit_rate_gesamt, (int, float)):
            avg_confidence = conf_sum / total_n
            gap_gesamt = avg_confidence - hit_rate_gesamt
            if gap_gesamt > _DAMPEN_GAP_THRESHOLD:
                return True

        # Per-Action Gap der betroffenen Aktion
        if action is not None:
            adata = nach_aktion.get(action)
            if isinstance(adata, dict):
                avg_conf = adata.get("avg_confidence")
                hit_rate = adata.get("hit_rate")
                if isinstance(avg_conf, (int, float)) and isinstance(hit_rate, (int, float)):
                    gap = avg_conf - hit_rate
                    if gap > _DAMPEN_GAP_THRESHOLD:
                        return True
                # Phase 6: Konsistenz mit der Prompt-Hit-Rate-Direktive —
                # bei echter Hit-Rate < 0.35 ist 'STARK …' verboten
                # (Schwellen 0.20/0.35/0.50 als gemeinsame Referenz).
                if isinstance(hit_rate, (int, float)) and 0 <= hit_rate < _DAMPEN_HITRATE_MAX:
                    return True

        return False
    except Exception as exc:  # noqa: BLE001 — crasht nie
        logger.debug("Dämpfungs-Check fehlgeschlagen: %s", exc)
        return False


def _dampen_stark_rating(result: dict[str, Any], raw_rating: str) -> None:
    """Dämpft STARK KAUFEN/STARK VERKAUFEN im result-dict in-place.

    Wenn ``_should_dampen_stark`` True liefert:
    - STARK KAUFEN → Rating KAUFEN, Aktion KAUFEN
    - STARK VERKAUFEN → Rating VERKAUFEN, Aktion VERKAUFEN
    Setzt ``result["rating_gedämpft"]`` und ``result["rating_original"]``.
    """
    result["rating_gedämpft"] = False
    if raw_rating not in ("STARK KAUFEN", "STARK VERKAUFEN"):
        return
    action_to_check = "KAUFEN" if raw_rating == "STARK KAUFEN" else "VERKAUFEN"
    if _should_dampen_stark(action_to_check):
        damped = "KAUFEN" if raw_rating == "STARK KAUFEN" else "VERKAUFEN"
        result["rating_original"] = raw_rating
        result["rating"] = damped
        result["aktion"] = _rating_to_action(damped)
        result["rating_gedämpft"] = True


def ensemble_trader(
    analysts: dict[str, Any],
    debate_result: dict[str, Any],
    llm: LLMClient,
    runs: int = 3,
    temperature_range: list[float] | None = None,
    feedback_context: str = "",
    reflection_context: str = "",
    model: str | None = None,
    reasoning_effort: str | None = None,
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
        model: Optionales Modell-Override (Quick-Think-Split), wird an jeden
            trader()-Run durchgereicht. None = primäres Modell.
        reasoning_effort: Optionale Reasoning-Tiefe ('low'/'medium'/'high'),
            wird an jeden trader()-Run durchgereicht. None/'' (Default) = kein
            reasoning_effort im Payload (bisheriges Verhalten).

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
            model=model,
            reasoning_effort=reasoning_effort,
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
            "einstiegs_level": None,
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

    # Kalibrierungs-Gewichte laden (netzfrei, deterministisch)
    cal_weights = _load_ensemble_weights()
    gewichtet = cal_weights is not None

    # Verwendete Gewichte pro Aktion (für Metadaten)
    aktion_gewichte: dict[str, float] = {}
    if cal_weights:
        for action in _ENSEMBLE_ACTIONS:
            hr = cal_weights.get(action)
            if hr is not None:
                aktion_gewichte[action] = round(_smooth_weight(hr), 2)

    # Gewichtete Abstimmung
    aktion_gewicht_sum: dict[str, float] = {}
    for a in aktionen:
        if cal_weights:
            hr = cal_weights.get(a)
            gewicht = _smooth_weight(hr) if hr is not None else 1.0
        else:
            gewicht = 1.0
        aktion_gewicht_sum[a] = aktion_gewicht_sum.get(a, 0.0) + gewicht

    mehrheits_aktion = max(aktion_gewicht_sum, key=lambda k: aktion_gewicht_sum[k])
    total_gewicht = sum(aktion_gewicht_sum.values())
    confidence = aktion_gewicht_sum[mehrheits_aktion] / total_gewicht if total_gewicht > 0 else 0.0

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

    # Technik-Signal (Safety-Net, Ensemble-Ebene): Normalerweise wendet trader()
    # das Signal bereits pro Run an, sodass die Position der Runs bereits skaliert
    # ist. Liefert die Mehrheit dennoch KAUFEN (Runs umgehen trader(), gemockte
    # Trades o. ä.), skaliert dieses Safety-Net die Position des finalen
    # KAUFEN-Trades graduell — KAUFEN wird nie mehr auf HALTEN zurückgesetzt.
    if mehrheits_aktion == "KAUFEN":
        _apply_technik_signal(result, analysts)

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
        "gewichtet": gewichtet,
        "aktion_gewichte": aktion_gewichte,
    }

    # Entscheidungs-Disziplin: finales Rating dämpfen wenn überkonfident
    # (pro-Run-Dämpfung ist bereits via trader() passiert; hier wird das
    # finale Rating nach Mehrheitsabstimmung zusätzlich gedämpft)
    _final_dampen_ensemble(result)

    return result


def _final_dampen_ensemble(result: dict[str, Any]) -> None:
    """Dämpft das finale Ensemble-Rating wenn es STARK KAUFEN/STARK VERKAUFEN ist.

    Wird NACH der Mehrheitsabstimmung auf das finale result angewendet.
    Nutzt die gleiche ``_should_dampen_stark``-Logik wie ``_dampen_stark_rating``.
    """
    final_rating = str(result.get("rating", "")).strip().upper()
    if final_rating not in ("STARK KAUFEN", "STARK VERKAUFEN"):
        # rating_gedämpft sicherstellen, falls noch nicht gesetzt
        if "rating_gedämpft" not in result:
            result["rating_gedämpft"] = False
        return
    action_to_check = "KAUFEN" if final_rating == "STARK KAUFEN" else "VERKAUFEN"
    if _should_dampen_stark(action_to_check):
        damped = "KAUFEN" if final_rating == "STARK KAUFEN" else "VERKAUFEN"
        result["rating_original"] = final_rating
        result["rating"] = damped
        result["aktion"] = _rating_to_action(damped)
        result["rating_gedämpft"] = True
    else:
        if "rating_gedämpft" not in result:
            result["rating_gedämpft"] = False


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
    annualized = std * math.sqrt(252)
    return round(annualized, 6)


def _normalize_pct_string(val: Any) -> float | None:
    """Extrahiert einen float aus einem Wert, der number oder String sein kann.

    Akzeptiert Zahlen direkt und Strings wie "5 %", "5%", "5,5", "5.5".
    Leerzeichen und %-Zeichen werden entfernt, Komma→Punkt konvertiert.
    Gibt None zurück bei None oder nicht parsebarem Wert.
    """
    if val is None:
        return None
    if isinstance(val, int | float):
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return None
        return fval if fval == fval else None  # NaN-Check
    s = str(val).strip()
    if not s:
        return None
    # %-Zeichen und Leerzeichen entfernen, Komma→Punkt
    s = s.replace("%", "").strip()
    s = s.replace(",", ".")
    try:
        fval = float(s)
    except (TypeError, ValueError):
        return None
    return fval if fval == fval else None  # NaN-Check


def _risk_perspective_call(
    llm: LLMClient,
    system_prompt: str,
    user_text: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Führt einen Perspektiven-Call der Risiko-Debatte aus (best-effort).

    Analog zu Bull/Bear: DEBATE_SCHEMA für strukturierte Outputs, das
    Fließtext-Argument via _get_debate_argument extrahieren. Bei Fehlschlag
    wird eine Warnung geloggt und ein leerer String zurückgegeben — die
    Debatte crasht nie.
    """
    try:
        result = _call_agent(
            llm,
            system_prompt,
            user_text,
            temperature=0.5,
            response_format=DEBATE_SCHEMA,
            structured=True,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        return _get_debate_argument(result)
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning("Risk-Debatte: Perspektiven-Call fehlgeschlagen: %s", exc)
        return ""


def _run_risk_perspectives_parallel(
    llm: LLMClient,
    jobs: list[tuple[str, str, str]],
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, str]:
    """Führt (Name, System-Prompt, User-Text)-Jobs parallel aus (best-effort).

    Gleiche Technik wie analyst_team: ThreadPoolExecutor mit _MAX_PARALLEL
    Workern; bei einem Teilfehler wird eine Warnung geloggt und für den
    betroffenen Key ein leerer String geliefert — die Pipeline crasht nicht.
    """
    results: dict[str, str] = {}

    def _run_one(name: str, prompt: str, text: str) -> tuple[str, str]:
        return name, _risk_perspective_call(
            llm, prompt, text, model=model, reasoning_effort=reasoning_effort,
        )

    max_workers = min(len(jobs), _MAX_PARALLEL)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one, name, prompt, text): name
            for name, prompt, text in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                _, arg = future.result()
            except Exception as exc:  # noqa: BLE001 — nie crashen
                logger.warning(
                    "Risk-Debatte: Perspektive '%s' fehlgeschlagen: %s", name, exc
                )
                arg = ""
            results[name] = arg
    return results


def risk_debate(
    trade: dict[str, Any],
    data: dict[str, Any],
    llm: LLMClient,
    data_text: str | None = None,
    feedback_context: str = "",
    rounds: int | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """3-Perspektiven-Risiko-Debatte (aggressiv/neutral/konservativ, 2 Runden).

    Ersetzt den Single-Pass-Risk-Manager-Call (Phase B):

    1. Rechnerische Werte (annualisierte Volatilität, Volatility-Targeting-
       Positionsgröße) werden wie bisher deterministisch berechnet und als
       risk_block an alle Prompts angehängt.
    2. Runde 1: Alle 3 Perspektiven bekommen Trade + Marktdaten + risk_block
       (parallel, wie analyst_team) und liefern je ein Fließtext-Argument.
    3. Runde 2 (nur bei rounds >= 2): Jede Perspektive bekommt zusätzlich
       die Argumente der anderen beiden aus Runde 1 und antwortet konkret
       darauf (analog zur Bull/Bear-Debatte).
    4. Synthese: Ein finaler LLM-Call (SYSTEM_RISK_SYNTHESIS) bekommt alle
       Argumente der gelaufenen Runden + Trade + Marktdaten + risk_block
       und liefert das finale risk-dict via RISK_SCHEMA.

    Alle Calls sind best-effort: Fällt eine Perspektive aus, fährt die
    Debatte mit den anderen fort; schlägt die Synthese fehl, greift der
    defaults_for_schema(RISK_SCHEMA)-Fallback — das risk-dict kommt immer.

    Args:
        trade: Trade-Vorschlag vom Trader/Ensemble.
        data: Daten-dict aus collect_ticker_data.
        llm: LLMClient.
        data_text: Optional vorberechneter Daten-Text (vermeidet mehrfache
            _build_data_text-Berechnung). Wenn None, wird er intern berechnet.
        feedback_context: Optionaler Track-Record-Kontext-Block (leer = kein
            Feedback). Wird an Perspektiven- UND Synthese-Prompt angehängt.
        rounds: Anzahl der Debatten-Runden (1 = nur Runde 1, spart 3 LLM-Calls;
            2 = Runde 1 + Runde 2). None liest CONCILIUM_RISK_DEBATE_ROUNDS
            (Default 2). Werte < 2 überspringen Runde 2.
        model: Optionales Modell-Override (Deep-Think-Split), wird an ALLE
            Calls durchgereicht (3 Perspektiven × Runden + Synthese). None =
            primäres Modell.
        reasoning_effort: Optionale Reasoning-Tiefe ('low'/'medium'/'high'),
            wird an ALLE Calls durchgereicht. None/'' (Default) = kein
            reasoning_effort im Payload (bisheriges Verhalten).

    Returns:
        risk-dict mit denselben Feldern wie der bisherige Single-Pass-Call
        (risiko_score, volatilität_bewertung, max_drawdown_schaetzung,
        positionsgröße_empfohlen, auflagen, empfehlung) plus den
        rechnerischen Feldern (volatilität_annualisiert_pct,
        positionsgröße_rechnerisch_pct). Zusätzlich werden die
        Debatten-Argumente der GELAUFENEN Runden unter dem Key
        "_risk_debate" mitgeliefert (für den Report) — bei 1 Runde nur
        "runde1", kein leerer "runde2"-Eintrag.
    """
    # Runden-Zahl auflösen: expliziter Parameter > Config (Default 2).
    if rounds is None:
        rounds = config.risk_debate_rounds()

    trade_text = json.dumps(trade, ensure_ascii=False, indent=2, default=str)
    if data_text is None:
        data_text = _build_data_text(data)

    # --- Rechnerische Positionsgröße VOR den LLM-Calls berechnen ---
    # Einmal berechnen, im Prompt UND im Rückgabedict verwenden (keine
    # Doppelberechnung). Identisch zum bisherigen Single-Pass-Verhalten.
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

    base_user = f"Trade-Vorschlag:\n{trade_text}\n\nMarktdaten:\n{data_text}{risk_block}"
    if feedback_context:
        base_user += f"\n\n{feedback_context}"

    perspektiven = [
        ("Aggressiv", SYSTEM_RISK_AGGRESSIVE),
        ("Neutral", SYSTEM_RISK_NEUTRAL),
        ("Konservativ", SYSTEM_RISK_CONSERVATIVE),
    ]

    # --- Runde 1: alle drei Perspektiven parallel ---
    runde1 = _run_risk_perspectives_parallel(
        llm,
        [(name, prompt, base_user) for name, prompt in perspektiven],
        model=model,
        reasoning_effort=reasoning_effort,
    )

    # --- Runde 2 (nur wenn >= 2 Runden): Reaktion auf die anderen beiden ---
    runde2: dict[str, str] = {}
    if rounds > 1:
        jobs_runde2: list[tuple[str, str, str]] = []
        for name, prompt in perspektiven:
            andere_texte = "\n\n".join(
                f"--- {anderer_name} (Runde 1) ---\n"
                f"{runde1.get(anderer_name) or '(kein Argument geliefert)'}"
                for anderer_name, _ in perspektiven
                if anderer_name != name
            )
            user2 = (
                f"{base_user}\n\n"
                "=== Argumentation der anderen Risiko-Perspektiven (Runde 1) ===\n"
                f"{andere_texte}\n\n"
                "Gehe konkret auf diese Argumente ein: stimme zu, widersprich "
                "mit Daten, oder ergänze. Wiederhole nicht deine vorherigen "
                "Punkte, sondern vertiefe/verteidige sie."
            )
            jobs_runde2.append((name, prompt, user2))
        runde2 = _run_risk_perspectives_parallel(
            llm, jobs_runde2, model=model, reasoning_effort=reasoning_effort,
        )

    # --- Synthese: finaler LLM-Call mit den Argumenten der gelaufenen Runden ---
    gelaufene_runden: list[tuple[int, dict[str, str]]] = [(1, runde1)]
    if rounds > 1:
        gelaufene_runden.append((2, runde2))

    synth_parts: list[str] = []
    for runde_nr, runde_args in gelaufene_runden:
        for name, _ in perspektiven:
            text = runde_args.get(name) or "(kein Argument geliefert)"
            synth_parts.append(f"=== {name} — Runde {runde_nr} ===\n{text}")
    synthese_args = "\n\n".join(synth_parts)

    anzahl_runden = len(gelaufene_runden)
    runden_wort = "Runde" if anzahl_runden == 1 else "Runden"
    synth_user = (
        f"Trade-Vorschlag:\n{trade_text}\n\n"
        f"Marktdaten:\n{data_text}{risk_block}\n\n"
        f"=== DEBATTEN-ARGUMENTE (3 Perspektiven × {anzahl_runden} "
        f"{runden_wort}) ===\n{synthese_args}"
    )
    if feedback_context:
        synth_user += f"\n\n{feedback_context}"

    schema_defaults = defaults_for_schema(RISK_SCHEMA)
    try:
        risk = _call_agent(
            llm,
            SYSTEM_RISK_SYNTHESIS,
            synth_user,
            response_format=RISK_SCHEMA,
            structured=True,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        # Defensiv: sicherstellen, dass ALLE Schema-Keys vorhanden sind
        # (setdefault überschreibt vorhandene Werte nicht).
        for key, default in schema_defaults.items():
            risk.setdefault(key, default)
    except Exception as exc:  # noqa: BLE001 — nie crashen
        logger.warning(
            "Risk-Debatte: Synthese fehlgeschlagen (%s) — Fallback-Defaults.", exc
        )
        risk = dict(schema_defaults)

    # Rechnerische Werte ins Rückgabedict übernehmen (einmal berechnet, nicht doppelt)
    risk["volatilität_annualisiert_pct"] = vol_pct
    risk["positionsgröße_rechnerisch_pct"] = pos_rechnerisch

    # Defensive Normalisierung: max_drawdown_schaetzung und positionsgröße_empfohlen
    # können vom LLM als String ("5 %", "5,5") geliefert werden → float machen.
    risk["max_drawdown_schaetzung"] = _normalize_pct_string(
        risk.get("max_drawdown_schaetzung")
    )
    risk["positionsgröße_empfohlen"] = _normalize_pct_string(
        risk.get("positionsgröße_empfohlen")
    )

    # Debatten-Argumente für den Report (optionale Anzeige, nie crashen) —
    # nur die tatsächlich gelaufenen Runden (bei 1 Runde kein "runde2"-Key).
    risk["_risk_debate"] = {"runde1": dict(runde1)}
    if rounds > 1:
        risk["_risk_debate"]["runde2"] = dict(runde2)

    return risk


def risk_manager(
    trade: dict[str, Any],
    data: dict[str, Any],
    llm: LLMClient,
    data_text: str | None = None,
    feedback_context: str = "",
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Bewertet Risiko des Trades via 3-Perspektiven-Risiko-Debatte (Phase B).

    Dünne Hülle um risk_debate: Statt eines Single-Pass-Calls debattieren
    drei Perspektiven (aggressiv/neutral/konservativ) über die konfigurierte
    Anzahl Runden (CONCILIUM_RISK_DEBATE_ROUNDS, Default 2); eine Synthese
    liefert das finale risk-dict (siehe risk_debate).

    Signatur und Rückgabetyp sind identisch zur bisherigen Implementierung —
    pipeline.py, report.py, trade_revision, portfolio_manager und die
    Kalibrierung laufen unverändert weiter.

    Zusätzlich wird eine rechnerische Positionsgröße via Volatility-Targeting
    ergänzt (positionsgröße_rechnerisch_pct, volatilität_annualisiert_pct).

    Args:
        trade: Trade-Vorschlag vom Trader/Ensemble.
        data: Daten-dict aus collect_ticker_data.
        llm: LLMClient.
        data_text: Optional vorberechneter Daten-Text (vermeidet mehrfache
            _build_data_text-Berechnung). Wenn None, wird er intern berechnet.
        feedback_context: Optionaler Track-Record-Kontext-Block (leer = kein
            Feedback). Wird am Ende der User-Prompts angehängt.
        model: Optionales Modell-Override (Deep-Think-Split), wird an
            risk_debate durchgereicht. None = primäres Modell.
        reasoning_effort: Optionale Reasoning-Tiefe ('low'/'medium'/'high'),
            wird an risk_debate durchgereicht. None/'' (Default) = kein
            reasoning_effort im Payload (bisheriges Verhalten).
    """
    # Config hier lesen (pipeline.py bleibt unverändert) und explizit
    # durchreichen; risk_debate selbst hätte bei rounds=None denselben
    # Fallback — beides abzudecken ist robust gegen beide Aufrufpfade.
    rounds = config.risk_debate_rounds()
    return risk_debate(
        trade,
        data,
        llm,
        data_text=data_text,
        feedback_context=feedback_context,
        rounds=rounds,
        model=model,
        reasoning_effort=reasoning_effort,
    )


def portfolio_manager(
    trade: dict[str, Any],
    risk: dict[str, Any],
    llm: LLMClient,
    portfolio_fit: dict[str, Any] | None = None,
    feedback_context: str = "",
    reflection_context: str = "",
    portfolio_context: dict[str, Any] | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
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
        portfolio_context: Optionaler Gesamt-Portfolio-Kontext (Korrelation,
            Overlap, Konzentration über alle analysierten Titel). Wenn gesetzt,
            wird er als „Gesamt-Exposure“-Block in den User-Prompt injiziert.
        model: Optionales Modell-Override (Deep-Think-Split). None =
            primäres Modell.
        reasoning_effort: Optionale Reasoning-Tiefe ('low'/'medium'/'high'),
            wird an den PM-Call durchgereicht. None/'' (Default) = kein
            reasoning_effort im Payload (bisheriges Verhalten).
    """
    trade_text = json.dumps(trade, ensure_ascii=False, indent=2, default=str)
    risk_text = json.dumps(risk, ensure_ascii=False, indent=2, default=str)

    user_text = f"Trade-Vorschlag:\n{trade_text}\n\nRisiko-Bewertung:\n{risk_text}"

    if portfolio_fit is not None:
        pf_text = json.dumps(portfolio_fit, ensure_ascii=False, indent=2, default=str)
        user_text += f"\n\nPortfolio-Fit-Einschätzung:\n{pf_text}"

    if portfolio_context is not None:
        from .portfolio_analysis import portfolio_context_to_text

        pc_text = portfolio_context_to_text(portfolio_context)
        user_text += f"\n\nGesamt-Exposure (Portfolio-Kontext aller analysierten Titel):\n{pc_text}"

    if feedback_context:
        user_text += f"\n\n{feedback_context}"

    if reflection_context:
        user_text += f"\n\n{reflection_context}"

    return _call_agent(
        llm, SYSTEM_PM, user_text,
        response_format=FINAL_SCHEMA,
        structured=True,
        model=model,
        reasoning_effort=reasoning_effort,
    )


def trade_revision(
    trade: dict[str, Any],
    risk: dict[str, Any],
    portfolio_fit: dict[str, Any] | None,
    llm: LLMClient,
    feedback_context: str = "",
    reflection_context: str = "",
    current_price: float | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Trade-Revision (2nd Pass) — der Trader überarbeitet seinen Trade.

    Nachdem Risk-Manager und Portfolio-Fit-Analyst den Trade bewertet haben,
    bekommt der Trader eine zweite Runde, um seinen Trade anzupassen.

    Nach dem LLM-Call werden zielkurs/stop_loss auf plausible Werte erzwungen,
    falls sie fehlen oder unplausibel sind (deterministischer Fallback basierend
    auf current_price). HALTEN-Trades behalten None-Werte.

    Args:
        trade: Ursprünglicher Trade-Vorschlag vom Trader/Ensemble.
        risk: Risiko-Bewertung vom Risk-Manager.
        portfolio_fit: Portfolio-Fit-Ergebnis (oder None, wenn nicht verfügbar).
        llm: LLMClient.
        feedback_context: Optionaler Track-Record-Kontext-Block.
        reflection_context: Optionaler Reflexions-Block.
        current_price: Aktueller Kurs — für deterministischen Ziel-/Stop-Fallback.
            Bei None wird der Fallback übersprungen (kein Crash).
        model: Optionales Modell-Override (Deep-Think-Split). None =
            primäres Modell.
        reasoning_effort: Optionale Reasoning-Tiefe ('low'/'medium'/'high'),
            wird an den Revision-Call durchgereicht. None/'' (Default) = kein
            reasoning_effort im Payload (bisheriges Verhalten).

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

    result = _call_agent(
        llm, SYSTEM_TRADE_REVISION, user_text,
        temperature=0.3,
        response_format=TRADE_SCHEMA,
        structured=True,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    # 5-stufige Rating normalisieren (wie bei trader())
    raw_rating = str(result.get("aktion", "")).strip().upper()
    result["rating"] = raw_rating
    result["aktion"] = _rating_to_action(raw_rating)

    # Ziel-/Stop-Erzwingung: deterministischer Fallback bei fehlenden/unplausiblen Werten
    if current_price is not None:
        try:
            _ensure_ziel_stop(result, result["aktion"], float(current_price))
        except (TypeError, ValueError):
            pass  # current_price nicht konvertierbar — kein Fallback, kein Crash

    # Entscheidungs-Disziplin: STARK-Ratings dämpfen wenn Kalibrierung überkonfident.
    # Der revidierte Trade ersetzt den Original-Trade (ins Journal + PM), daher muss
    # die gleiche Dämpfung wie bei trader()/ensemble_trader() angewendet werden.
    _dampen_stark_rating(result, result["rating"])

    return result
