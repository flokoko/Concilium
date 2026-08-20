"""Report-Modul — generiert deutschen Markdown-Report aus Pipeline-Ergebnissen."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _fmt(val: Any, suffix: str = "") -> str:
    """Formatiert einen Wert lesbar."""
    if val is None:
        return "N/A"
    try:
        fval = float(val)
        if abs(fval) >= 1e12:
            return f"{fval / 1e12:.2f} Mrd{suffix}"
        if abs(fval) >= 1e9:
            return f"{fval / 1e9:.2f} Mrd{suffix}"
        if abs(fval) >= 1e6:
            return f"{fval / 1e6:.2f} Mio{suffix}"
        if abs(fval) >= 1e3:
            return f"{fval / 1e3:.2f} K{suffix}"
        return f"{fval:.2f}{suffix}"
    except (TypeError, ValueError):
        return str(val) if val else "N/A"


def _fmt_pct(val: Any) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val) * 100:.1f} %"
    except (TypeError, ValueError):
        return "N/A"


def _clean_debate_text(agent: dict[str, Any]) -> str:
    """Entfernt das JSON-Preamble und Markdown-Codeblock-Wrapper aus Debatten-Texten.

    Die Bull/Bear-Agenten geben einen JSON-Block (confidence/name) gefolgt vom
    Fließtext zurück. Der Report soll nur den lesbaren Fließtext zeigen.
    """
    import re as _re

    raw = str(agent.get("_raw", ""))
    if not raw:
        return "N/A"

    # JSON-Block am Anfang entfernen: {"confidence": X, "name": "..."} oder ```json {...}```
    text = _re.sub(r"```json\s*\{.*?\}\s*```", "", raw, flags=_re.DOTALL).strip()
    text = _re.sub(r"^\{.*?\}", "", text, flags=_re.DOTALL).strip()
    text = text.strip("`").strip()
    return text if text else raw


def generate_report(result: dict[str, Any]) -> str:
    """Generiert einen deutschen Markdown-Report aus den Pipeline-Ergebnissen.

    Funktioniert mit und ohne LLM-Abschnitten (no_llm-Modus).
    """
    data = result.get("data", {})
    f = data.get("fundamentals", {})
    t = data.get("technicals", {})
    s = data.get("sentiment", {})
    news = data.get("news", [])
    ticker = result.get("ticker", data.get("ticker", "?"))
    now = datetime.now().strftime("%d.%m.%Y %H:%M")

    no_llm = result.get("no_llm", False)

    lines: list[str] = []
    lines.append(f"# TradingAgents-Light Analyse: {ticker}")
    lines.append("")
    lines.append(f"**Erstellt am:** {now}")
    lines.append(f"**Unternehmen:** {f.get('name', ticker)}")
    lines.append(f"**Sektor:** {f.get('sector', 'N/A')} / {f.get('industry', 'N/A')}")
    lines.append(f"**Währung:** {f.get('currency', 'USD')}")
    lines.append("")

    # --- Disclaimer ---
    lines.append("> ⚠️ **Disclaimer:** Dies ist keine Anlageberatung. Die Analysen basieren auf \
LLM-Textgenerierung und Heuristiken und dienen nur Demonstrationszwecken.")
    lines.append("")

    # === Übersicht ===
    lines.append("## 1. Übersicht")
    lines.append("")
    lines.append("| Kennzahl | Wert |")
    lines.append("|---|---|")
    lines.append(f"| Aktueller Kurs | {_fmt(t.get('current_price'))} |")
    lines.append(f"| Marktkapitalisierung | {_fmt(f.get('market_cap'))} |")
    lines.append(f"| KGV (trailing) | {_fmt(f.get('pe_ratio'))} |")
    lines.append(f"| EPS | {_fmt(f.get('eps'))} |")
    lines.append(f"| PEG | {_fmt(f.get('peg_ratio'))} |")
    lines.append(f"| Umsatz | {_fmt(f.get('revenue'))} |")
    lines.append(f"| Umsatzwachstum | {_fmt_pct(f.get('revenue_growth'))} |")
    lines.append(f"| Gewinnmarge | {_fmt_pct(f.get('profit_margin'))} |")
    lines.append(f"| Dividendenrendite | {_fmt_pct(f.get('dividend_yield'))} |")
    lines.append(f"| Beta | {_fmt(f.get('beta'))} |")
    lines.append(f"| 52W Hoch | {_fmt(f.get('fifty_two_week_high'))} |")
    lines.append(f"| 52W Tief | {_fmt(f.get('fifty_two_week_low'))} |")
    lines.append("")

    # === Technische Indikatoren ===
    lines.append("## 2. Technische Indikatoren")
    lines.append("")
    lines.append("| Indikator | Wert |")
    lines.append("|---|---|")
    lines.append(f"| SMA50 | {_fmt(t.get('sma50'))} |")
    lines.append(f"| SMA200 | {_fmt(t.get('sma200'))} |")
    lines.append(f"| RSI(14) | {_fmt(t.get('rsi14'))} |")
    macd = t.get("macd", {})
    lines.append(f"| MACD | {_fmt(macd.get('macd'))} |")
    lines.append(f"| MACD Signal | {_fmt(macd.get('signal'))} |")
    lines.append(f"| MACD Histogramm | {_fmt(macd.get('histogram'))} |")
    boll = t.get("bollinger", {})
    lines.append(f"| Bollinger Ober | {_fmt(boll.get('upper'))} |")
    lines.append(f"| Bollinger Mitte | {_fmt(boll.get('middle'))} |")
    lines.append(f"| Bollinger Unter | {_fmt(boll.get('lower'))} |")
    lines.append(f"| Bollinger-Position | {_fmt(boll.get('position'))} (0=unter, 1=ober) |")
    lines.append(f"| Volumen | {_fmt(t.get('current_volume'))} |")
    lines.append(f"| Ø Volumen 30T | {_fmt(t.get('avg_volume_30d'))} |")
    lines.append("")

    # === Sentiment ===
    lines.append("## 3. Sentiment-Heuristik")
    lines.append("")

    is_weighted = s.get("weighted", False)
    methode_hinweis = "zeitgewichtet" if is_weighted else "ungewichtet"

    if is_weighted:
        # Zeitgewichtete Werte (Gleitkommazahlen)
        lines.append(f"_Methode: {methode_hinweis} (Halbwertszeit 7 Tage, exponentieller Zerfall)_")
        lines.append("")
        lines.append("| Kategorie | Gewichtung |")
        lines.append("|---|---|")
        lines.append(f"| Positiv | {s.get('positiv', 0):.2f} |")
        lines.append(f"| Negativ | {s.get('negativ', 0):.2f} |")
        lines.append(f"| Neutral | {s.get('neutral', 0):.2f} |")
        lines.append("")
        lines.append(f"**Dominante Stimmung:** {s.get('dominant', 'N/A')}")
        lines.append(f"**Anzahl Headlines (Sample):** {s.get('sample_size', 0)}")
    else:
        # Ungewichtete Werte (ganze Zahlen)
        lines.append(f"_Methode: {methode_hinweis}_")
        lines.append("")
        lines.append("| Kategorie | Anzahl |")
        lines.append("|---|---|")
        lines.append(f"| Positiv | {s.get('positiv', 0)} |")
        lines.append(f"| Negativ | {s.get('negativ', 0)} |")
        lines.append(f"| Neutral | {s.get('neutral', 0)} |")
        lines.append("")
        lines.append(f"**Dominante Stimmung:** {s.get('dominant', 'N/A')}")
        lines.append(f"**Anzahl Headlines (Sample):** {s.get('sample_size', 0)}")

    lines.append("")

    if news:
        lines.append("### Aktuelle Headlines")
        lines.append("")
        for h in news[:15]:
            lines.append(f"- {h}")
        lines.append("")

    # === Backtest (falls vorhanden) ===
    if result.get("backtest"):
        bt = result["backtest"]
        lines.append("## 4. Backtest-Signalproxy")
        lines.append("")
        if bt.get("strategie_rendite") is not None:
            lines.append("| Metrik | Wert |")
            lines.append("|---|---|")
            lines.append(f"| Strategie-Rendite | {bt['strategie_rendite']} % |")
            lines.append(f"| Buy & Hold Rendite | {bt['buy_hold_rendite']} % |")
            lines.append(f"| Outperformance | {bt['outperformance']} % |")
            lines.append(f"| Anzahl Signale | {bt.get('anzahl_signale', 0)} |")
            lines.append(f"| Zeitraum | {bt.get('startdatum', '?')} – {bt.get('enddatum', '?')} |")
            lines.append("")
            if bt.get("signale"):
                lines.append("### Letzte Signalwechsel")
                lines.append("")
                for sig in bt["signale"]:
                    lines.append(f"- {sig['date']}: **{sig['aktion']}** @ {sig['close']}")
                lines.append("")
        else:
            lines.append(f"_{bt.get('hinweis', 'Backtest nicht möglich.')}_")
            lines.append("")

    # --- Agenten-Abschnitte (nur mit LLM) ---
    if not no_llm and result.get("analysts"):
        # Agenten-Abschnitte beginnen nach Daten/Backtest-Abschnitten.
        # Basis: ohne Backtest = 4 (Übersicht, Technik, Sentiment, dann Analysten)
        #        mit Backtest = 5 (…, Backtest, dann Analysten)
        section_num = 5 if result.get("backtest") else 4

        # Analysten-Tabelle
        lines.append(f"## {section_num}. Analysten-Team")
        lines.append("")
        lines.append("| Rolle | Stimmung | Score | Zusammenfassung |")
        lines.append("|---|---|---|---|")
        analysts = result["analysts"]
        for key, label in [("fundamental", "Fundamental"), ("technical", "Technik"), ("sentiment", "Sentiment")]:
            a = analysts.get(key, {})
            lines.append(
                f"| {label} | {a.get('stimmung', 'N/A')} | {a.get('score', 'N/A')} | "
                f"{str(a.get('zusammenfassung', a.get('_raw', 'N/A')))[:200]} |"
            )
        lines.append("")

        # Debatte
        section_num += 1
        debate = result.get("debate", {})
        lines.append(f"## {section_num}. Bull/Bear-Debatte")
        lines.append("")
        bull = debate.get("bull", {})
        bear = debate.get("bear", {})
        lines.append("### Bull-Argumentation")
        lines.append("")
        lines.append(_clean_debate_text(bull))
        lines.append("")
        lines.append("### Bear-Argumentation")
        lines.append("")
        lines.append(_clean_debate_text(bear))
        lines.append("")

        # Trade
        section_num += 1
        trade = result.get("trade", {})
        lines.append(f"## {section_num}. Trade-Vorschlag")
        lines.append("")
        lines.append(f"**Aktion:** {trade.get('aktion', 'N/A')}")
        lines.append(f"**Zielkurs:** {_fmt(trade.get('zielkurs'))}")
        lines.append(f"**Stop-Loss:** {_fmt(trade.get('stop_loss'))}")
        lines.append(f"**Positionsanteil:** {trade.get('positionsanteil', 'N/A')} %")
        lines.append(f"**Zeithorizont:** {trade.get('zeithorizont', 'N/A')}")
        lines.append(f"**Begründung:** {trade.get('begründung', 'N/A')}")
        lines.append("")

        # Risk
        section_num += 1
        risk = result.get("risk", {})
        lines.append(f"## {section_num}. Risiko-Bewertung")
        lines.append("")
        lines.append(f"**Risiko-Score:** {risk.get('risiko_score', 'N/A')} / 5")
        lines.append(f"**Volatilität:** {risk.get('volatilität_bewertung', 'N/A')}")
        lines.append(f"**Max. Drawdown (Schätzung):** {risk.get('max_drawdown_schaetzung', 'N/A')}")
        lines.append(f"**Empf. Positionsgröße:** {risk.get('positionsgröße_empfohlen', 'N/A')}")
        lines.append(f"**Auflagen:** {risk.get('auflagen', 'N/A')}")
        lines.append(f"**Empfehlung:** {risk.get('empfehlung', 'N/A')}")
        lines.append("")

        # Final
        section_num += 1
        final = result.get("final", {})
        lines.append(f"## {section_num}. Finale Entscheidung (Portfolio-Manager)")
        lines.append("")
        entscheidung = final.get("entscheidung", "N/A")
        emoji = "✅" if "GENEHMIGT" in str(entscheidung).upper() else "❌"
        lines.append(f"### {emoji} {entscheidung}")
        lines.append("")
        lines.append(f"**Begründung:** {final.get('begründung', 'N/A')}")
        lines.append(f"**Confidence:** {final.get('confidence', 'N/A')} / 5")
        lines.append("")
    elif no_llm:
        lines.append("## Agenten-Abschnitte")
        lines.append("")
        lines.append("_--no-llm Modus: Agenten-Abschnitte wurden übersprungen. \
Obiger Report zeigt nur den Datensnapshot._")
        lines.append("")

    # --- Footer ---
    lines.append("---")
    lines.append("*Erstellt von TradingAgents-Light · Keine Anlageberatung*")

    return "\n".join(lines)
