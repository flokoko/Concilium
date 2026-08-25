"""Report-Modul — generiert deutschen Markdown-Report aus Pipeline-Ergebnissen."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any


def _fmt(val: Any, suffix: str = "") -> str:
    """Formatiert einen Wert lesbar."""
    if val is None:
        return "N/A"
    try:
        fval = float(val)
        if math.isnan(fval):
            return "N/A"
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
        return str(val) if val else "N/A"


def _fmt_pct(val: Any) -> str:
    if val is None:
        return "N/A"
    try:
        fval = float(val)
        if math.isnan(fval):
            return "N/A"
        return f"{fval * 100:.1f} %"
    except (TypeError, ValueError):
        return "N/A"


def _clean_debate_text(agent: dict[str, Any]) -> str:
    """Entfernt das JSON-Preamble und Markdown-Codeblock-Wrapper aus Debatten-Texten.

    Die Bull/Bear-Agenten geben einen JSON-Block (confidence/name) gefolgt vom
    Fließtext zurück. Der Report soll nur den lesbaren Fließtext zeigen.

    Im strukturierten Pfad enthält das dict ein ``argumente``-Feld mit dem
    Fließtext direkt — dieses wird bevorzugt. Im Fallback-Pfad wird ``_raw``
    gesäubert (JSON-Preamble entfernt).

    Wenn ``_raw`` leer ist oder nach Entfernung des JSON-Blocks nichts übrig
    bleibt, wird ein klarer Platzhalter zurückgegeben statt nacktem "N/A".
    """
    import re as _re

    # Strukturierter Pfad: argumente-Feld direkt verwenden
    argumente = agent.get("argumente")
    if argumente and str(argumente).strip():
        return str(argumente).strip()

    raw = str(agent.get("_raw", ""))
    if not raw or not raw.strip():
        return "⚠️ Argumentation nicht verfügbar (Analysten-Ausfall)."

    # JSON-Block am Anfang entfernen: {"confidence": X, "name": "..."} oder ```json {...}```
    text = _re.sub(r"```json\s*\{.*?\}\s*```", "", raw, flags=_re.DOTALL).strip()
    text = _re.sub(r"^\{.*?\}", "", text, flags=_re.DOTALL).strip()
    text = text.strip("`").strip()
    if not text:
        return "⚠️ Argumentation nicht verfügbar (Analysten-Ausfall)."
    return text


def _management_summary(result: dict[str, Any], no_llm: bool) -> list[str]:
    """Erstellt eine kompakte Management-Summary aus dem result-dict.

    Deterministisch, kein LLM-Call. Robust gegen fehlende Werte (N/A).
    Liefert die Zeilen für die ``## Management-Summary``-Sektion.
    """
    lines: list[str] = []
    lines.append("## Management-Summary")
    lines.append("")

    # --- 1. Gesamturteil (TL;DR) ---
    if no_llm:
        lines.append("**Urteil:** Datensnapshot (kein LLM)")
    else:
        final = result.get("final") or {}
        trade = result.get("trade") or {}
        entscheidung = final.get("entscheidung", "N/A")
        entscheidung_upper = str(entscheidung).upper()
        if "GENEHMIGT" in entscheidung_upper:
            emoji = "✅"
        elif "MODIFIZIERT" in entscheidung_upper:
            emoji = "⚡"
        else:
            emoji = "❌"

        trade_aktion = trade.get("aktion", "N/A")
        trade_parts = [f"Trade: {trade_aktion}"]
        trade_rating = trade.get("rating")
        if trade_rating:
            trade_parts.append(f"Rating: {trade_rating}")
        zielkurs = trade.get("zielkurs")
        if zielkurs is not None:
            trade_parts.append(f"Zielkurs {_fmt(zielkurs)}")
        stop_loss = trade.get("stop_loss")
        if stop_loss is not None:
            trade_parts.append(f"Stop {_fmt(stop_loss)}")

        lines.append(f"**Urteil:** {emoji} {entscheidung} — {', '.join(trade_parts)}")

        # Entscheidungs-Disziplin: Hinweis bei gedämpftem Rating
        if trade.get("rating_gedämpft"):
            original = trade.get("rating_original", "STARK KAUFEN/STARK VERKAUFEN")
            lines.append(
                f"_⚠️ Rating gedämpft ({original} → {trade.get('rating', 'N/A')}) "
                f"wegen überkonfidenter Historie_"
            )

    # --- 2. Score-Zeile ---
    score_parts: list[str] = []

    risk = result.get("risk") or {}
    risiko_score = risk.get("risiko_score")
    if risiko_score is not None:
        score_parts.append(f"Risiko {risiko_score}/5")

    portfolio_fit = result.get("portfolio_fit")
    if isinstance(portfolio_fit, dict):
        pf_score = portfolio_fit.get("portfolio_fit_score")
        if pf_score is not None:
            score_parts.append(f"Portfolio-Fit {pf_score}/5")

    debate = result.get("debate") or {}
    bull_conf = debate.get("bull_confidence")
    bear_conf = debate.get("bear_confidence")
    if bull_conf is not None or bear_conf is not None:
        deb_parts: list[str] = []
        if bull_conf is not None:
            deb_parts.append(f"Bull {bull_conf}")
        if bear_conf is not None:
            deb_parts.append(f"Bear {bear_conf}")
        score_parts.append(f"Debatte {' vs '.join(deb_parts)}")

    trade = result.get("trade") or {}
    ensemble_info = trade.get("_ensemble")
    if ensemble_info and isinstance(ensemble_info, dict):
        ens_conf = ensemble_info.get("ensemble_confidence", 0)
        if isinstance(ens_conf, int | float):
            ens_conf_pct = int(ens_conf * 100)
        else:
            ens_conf_pct = 0
        score_parts.append(f"Ensemble-Konfidenz {ens_conf_pct}%")

    if score_parts:
        lines.append(f"**Scores:** {' · '.join(score_parts)}")

    # --- 3. Kernrisiken ---
    risks: list[str] = []

    # Risk-Auflagen
    auflagen = risk.get("auflagen")
    if auflagen and str(auflagen).strip().lower() != "keine":
        risks.append(f"- {str(auflagen)[:120]}")

    # Konzentrations-/Overlap-Risiko
    if isinstance(portfolio_fit, dict):
        konz = portfolio_fit.get("konzentrationsrisiko_bewertung")
        if konz and str(konz).strip() and str(konz).strip().lower() != "n/a":
            risks.append(f"- {str(konz)[:120]}")
        else:
            overlap = portfolio_fit.get("sektor_overlap_bewertung")
            if overlap and str(overlap).strip() and str(overlap).strip().lower() != "n/a":
                risks.append(f"- {str(overlap)[:120]}")

    # Analysten-Konsistenz-Warnungen
    analysts = result.get("analysts") or {}
    if isinstance(analysts, dict):
        for key in ("fundamental", "technical", "sentiment"):
            a = analysts.get(key)
            if isinstance(a, dict):
                warn = a.get("konsistenz_warnung")
                if warn and str(warn).strip():
                    risks.append(f"- {str(warn)[:120]}")

    # Debatten-Nettoneigung bearisch
    if bull_conf is not None and bear_conf is not None:
        try:
            bc = int(bull_conf)
            be = int(bear_conf)
            if be - bc >= 2:
                risks.append("- ⚠️ Debatte tendiert bearisch")
        except (TypeError, ValueError):
            pass

    # Max 4 Risiken, sonst Platzhalter
    if not risks:
        lines.append("- Keine auffälligen Risiken erkannt.")
    else:
        for r in risks[:4]:
            lines.append(r)

    # --- 4. Kurz-Begründung ---
    if not no_llm:
        final = result.get("final") or {}
        begründung = final.get("begründung")
        if begründung and str(begründung).strip():
            lines.append(f"**Kurz-Begründung:** {str(begründung)[:200]}")

    lines.append("")
    return lines


def _fmt_corr(val: Any) -> str:
    """Formatiert einen Korrelationskoeffizienten."""
    if val is None:
        return "n/a"
    try:
        fval = float(val)
        if math.isnan(fval):
            return "n/a"
        return f"{fval:.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _portfolio_blick_section(pa: dict[str, Any]) -> list[str]:
    """Erstellt die '## Portfolio-Blick'-Sektion aus portfolio_analysis.

    Zeigt:
    - Ziel-Gewichtungen der analysierten Titel
    - Korrelations-Matrix (Paare mit |r| > 0.7 hervorgehoben)
    - Konzentrations-/Overlap-Warnungen

    Robust gegen fehlende Werte (n/a statt Zahlen).
    """
    lines: list[str] = []
    lines.append("## Portfolio-Blick")
    lines.append("")

    tickers = pa.get("analysed_tickers", [])
    correlations = pa.get("correlations", {})
    target_weights = pa.get("target_weights", {})
    overlap = pa.get("overlap")
    concentration_warnings = pa.get("concentration_warnings", [])

    # --- Ziel-Gewichtungen ---
    if target_weights:
        lines.append("### Ziel-Gewichtungen")
        lines.append("")
        lines.append("| Ticker | Ziel-Gewichtung % |")
        lines.append("|---|---|")
        for ticker in tickers:
            w = target_weights.get(ticker)
            if w is not None:
                try:
                    lines.append(f"| {ticker} | {float(w):.1f} % |")
                except (TypeError, ValueError):
                    lines.append(f"| {ticker} | n/a |")
            else:
                lines.append(f"| {ticker} | n/a |")
        lines.append("")
    else:
        lines.append("_Keine Ziel-Gewichtungen verfügbar._")
        lines.append("")

    # --- Korrelations-Matrix ---
    if correlations and len(tickers) >= 2:
        lines.append("### Korrelations-Matrix (Tagesrenditen, Pearson)")
        lines.append("")

        # Header-Zeile
        header = "| Ticker | " + " | ".join(tickers) + " |"
        sep = "|---|" + "|".join(["---" for _ in tickers]) + "|"
        lines.append(header)
        lines.append(sep)

        for t_a in tickers:
            row_vals: list[str] = []
            row_data = correlations.get(t_a, {})
            for t_b in tickers:
                r = row_data.get(t_b)
                formatted = _fmt_corr(r)
                # Hervorheben bei |r| > 0.7 (außer Diagonale)
                if r is not None and t_a != t_b:
                    try:
                        r_float = float(r)
                        if abs(r_float) > 0.7:
                            formatted = f"**{formatted}** ⚠️"
                    except (TypeError, ValueError):
                        pass
                row_vals.append(formatted)
            lines.append(f"| {t_a} | " + " | ".join(row_vals) + " |")

        lines.append("")
        lines.append(
            "_Fett markierte Werte (|r| > 0.7) deuten auf hohe Korrelation "
            "und geringe Diversifikation hin._"
        )
        lines.append("")
    elif len(tickers) >= 2:
        lines.append("### Korrelations-Matrix")
        lines.append("")
        lines.append("_Nicht genügend überlappende Daten für Korrelations-Berechnung._")
        lines.append("")

    # --- Overlap-Warnungen ---
    if overlap and isinstance(overlap, dict):
        direct_overlaps = overlap.get("direct_overlaps", [])
        total_overlap_pct = overlap.get("total_overlap_pct", 0.0)
        overlap_warnings = overlap.get("warnings", [])

        if direct_overlaps or total_overlap_pct > 0:
            lines.append("### Overlap mit bestehendem Depot")
            lines.append("")
            if direct_overlaps:
                lines.append("| Ticker | Depot-Position | Depot-% |")
                lines.append("|---|---|---|")
                for ov in direct_overlaps:
                    lines.append(
                        f"| {ov.get('ticker', '?')} | "
                        f"{ov.get('position_name', '?')} | "
                        f"{ov.get('depot_pct', 0):.1f} % |"
                    )
                lines.append("")
            lines.append(f"**Gesamt-Overlap:** {total_overlap_pct:.1f} % des Depots")
            lines.append("")

            for w in overlap_warnings:
                lines.append(f"- ⚠️ {w}")
            if overlap_warnings:
                lines.append("")

    # --- Konzentrationswarnungen ---
    if concentration_warnings:
        lines.append("### Konzentrationswarnungen")
        lines.append("")
        for w in concentration_warnings:
            marker = "⚠️ " if w.startswith("⚠️") else ""
            text = w[2:] if w.startswith("⚠️") else w
            lines.append(f"- {marker}{text}")
        lines.append("")
    else:
        lines.append("### Konzentrationswarnungen")
        lines.append("")
        lines.append("_Keine auffälligen Konzentrationsrisiken erkannt._")
        lines.append("")

    return lines


def _source_label(source: str) -> str:
    """Mappt einen internen Source-Bezeichner auf ein lesbares Label für den Report.

    Bekannte Werte: yfinance → "yfinance", google/google_news → "Google",
    stocktwits/StockTwits → "StockTwits", reddit → "Reddit".
    Unbekannte Werte werden capitalized zurückgegeben.
    """
    s = source.lower().strip()
    if s in ("yfinance", "yahoo"):
        return "yfinance"
    if s in ("google", "google_news", "googlenews"):
        return "Google"
    if s in ("stocktwits",):
        return "StockTwits"
    if s in ("reddit",):
        return "Reddit"
    # Best-effort: ersten Buchstaben groß
    return source.capitalize() if source else "unknown"


def generate_report(
    result: dict[str, Any],
    reports_dir: str | None = None,
) -> str:
    """Generiert einen deutschen Markdown-Report aus den Pipeline-Ergebnissen.

    Funktioniert mit und ohne LLM-Abschnitten (no_llm-Modus).

    Args:
        result: Das Ergebnis-dict aus run_pipeline.
        reports_dir: Verzeichnis für Report-Dateien (z. B. 'reports/').
            Falls gegeben und matplotlib verfügbar, wird ein Chart erzeugt
            und als relatives Bild eingebettet. Falls None, wird kein Chart
            erzeugt.
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
    lines.append(f"# Concilium Analyse: {ticker}")
    lines.append("")
    lines.append(f"**Erstellt am:** {now}")
    lines.append(f"**Unternehmen:** {f.get('name', ticker)}")
    lines.append(f"**Sektor:** {f.get('sector', 'N/A')} / {f.get('industry', 'N/A')}")
    lines.append(f"**Währung:** {f.get('currency', 'USD')}")
    # ISIN/WKN anzeigen, falls verfügbar (bei Auflösung über ISIN/WKN)
    isin = data.get("isin")
    wkn = data.get("wkn")
    if isin or wkn:
        parts: list[str] = []
        if isin:
            parts.append(f"ISIN: {isin}")
        if wkn:
            parts.append(f"WKN: {wkn}")
        lines.append(f"**Kennung:** {' · '.join(parts)}")
    lines.append("")

    # --- Management-Summary ---
    lines.extend(_management_summary(result, no_llm))

    # --- Disclaimer ---
    lines.append("> ⚠️ **Disclaimer:** Dies ist keine Anlageberatung. Die Analysen basieren auf \
LLM-Textgenerierung und Heuristiken und dienen nur Demonstrationszwecken.")
    lines.append("")

    # --- Datenqualitäts-Hinweise (nur bei Warnungen) ---
    data_warnings = data.get("data_warnings", [])
    if data_warnings:
        lines.append("## ⚠️ Datenqualitäts-Hinweise")
        lines.append("")
        for w in data_warnings:
            lines.append(f"- {w}")
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

    # === Analysten-Erwartungen (Feature 1) ===
    has_analyst = any(
        k in f and f.get(k) is not None
        for k in ("analyst_target_mean", "analyst_target_high", "analyst_target_low",
                  "recommendation_key", "analyst_count", "recommendation_mean",
                  "analyst_upside_pct")
    )
    if has_analyst:
        lines.append("## 1b. Analysten-Erwartungen")
        lines.append("")
        lines.append("| Kennzahl | Wert |")
        lines.append("|---|---|")
        lines.append(f"| Konsens-Empfehlung | {f.get('recommendation_key', 'N/A')} |")
        lines.append(f"| Recommendation Mean (1=stark kaufen … 5=verkaufen) | {_fmt(f.get('recommendation_mean'))} |")
        lines.append(f"| Anzahl Analysten | {_fmt(f.get('analyst_count'))} |")
        lines.append(f"| Zielkurs Ø (12M) | {_fmt(f.get('analyst_target_mean'))} |")
        lines.append(f"| Zielkurs Hoch | {_fmt(f.get('analyst_target_high'))} |")
        lines.append(f"| Zielkurs Tief | {_fmt(f.get('analyst_target_low'))} |")
        if f.get("analyst_upside_pct") is not None:
            lines.append(f"| Upside zum Zielkurs | {_fmt(f.get('analyst_upside_pct'))} % |")
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

    # === Chart (optional, nur wenn matplotlib verfügbar und reports_dir gegeben) ===
    if reports_dir is not None:
        from .charts import generate_chart

        chart_path = generate_chart(data, reports_dir)
        if chart_path is not None:
            lines.append("## 2b. Chart")
            lines.append("")
            lines.append(f"![Chart]({chart_path})")
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

    # Quellen anzeigen (Phase 3)
    sentiment_sources = s.get("sources")
    if sentiment_sources:
        source_labels = [_source_label(src) for src in sentiment_sources]
        lines.append(f"**Quellen:** {', '.join(source_labels)}")

    lines.append("")

    if news:
        lines.append("### Aktuelle Headlines")
        lines.append("")
        # news_with_dates enthält optionale source-Felder (Phase 3)
        news_wd = data.get("news_with_dates", [])
        # Mapping title → source für schnelle Lookup (erste Fundstelle)
        source_map: dict[str, str] = {}
        if news_wd:
            for item in news_wd:
                if isinstance(item, dict):
                    title = item.get("title", "")
                    src = item.get("source")
                    if title and src and title not in source_map:
                        source_map[title] = src
        for h in news[:15]:
            src = source_map.get(h)
            if src:
                # Quelle als Tag anzeigen: [StockTwits]/[Reddit]/[yfinance]/[Google]
                src_label = _source_label(src)
                lines.append(f"- [{src_label}] {h}")
            else:
                lines.append(f"- {h}")
        lines.append("")

    # === Makro / Zinsen (Feature 2) ===
    macro = data.get("macro", {})
    if macro and any(v is not None for v in macro.values()):
        lines.append("## Makro / Zinsen")
        lines.append("")
        lines.append("| Kennzahl | Wert |")
        lines.append("|---|---|")
        lines.append(f"| 10y US Treasury Yield | {_fmt(macro.get('us_10y_yield'))} % |")
        lines.append(f"| 10y Yield vor 1 Monat | {_fmt(macro.get('us_10y_yield_1m_ago'))} % |")
        lines.append(f"| 10y Zinstrend | {macro.get('us_10y_trend', 'N/A')} |")
        sp500_source = macro.get("sp500_source", "none")
        source_label = f" ({sp500_source})" if sp500_source and sp500_source != "none" else ""
        lines.append(f"| S&P 500 KGV{source_label} | {_fmt(macro.get('sp500_pe'))} |")
        lines.append(f"| S&P 500 Marktkap | {_fmt(macro.get('sp500_market_cap'))} |")
        lines.append("")
        lines.append(
            "> Hinweis: Hohe/steigende Zinsen belasten kapitalintensive "
            "und erneuerbare Sektoren."
        )
        lines.append("")

    # === Peer-Vergleich (Feature 3) ===
    peers = data.get("peers", [])
    if peers:
        lines.append("## Peer-Vergleich")
        lines.append("")
        lines.append("| Ticker | KGV | Marktkap | Name |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| **{ticker}** | **{_fmt(f.get('pe_ratio'))}** | "
            f"**{_fmt(f.get('market_cap'))}** | **{f.get('name', 'N/A')}** |"
        )
        for p in peers:
            lines.append(
                f"| {p.get('ticker', '?')} | {_fmt(p.get('pe_ratio'))} | "
                f"{_fmt(p.get('market_cap'))} | {p.get('name', 'N/A')} |"
            )
        if macro.get("sp500_pe") is not None:
            sp500_src = macro.get("sp500_source", "none")
            src_label = f" ({sp500_src})" if sp500_src and sp500_src != "none" else ""
            lines.append(
                f"| S&P 500 (Benchmark){src_label} | {_fmt(macro.get('sp500_pe'))} | "
                f"{_fmt(macro.get('sp500_market_cap'))} | S&P 500 |"
            )
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
            lines.append(f"| Sharpe Ratio (annualisiert) | {_fmt(bt.get('sharpe_ratio'))} |")
            lines.append(f"| Max. Drawdown | {_fmt(bt.get('max_drawdown_pct'))} % |")
            lines.append(f"| Win-Rate | {_fmt(bt.get('win_rate_pct'))} % |")
            lines.append(f"| Anzahl Trades | {bt.get('anzahl_trades', 0)} |")
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

    # --- Portfolio-Blick (nur wenn portfolio_analysis vorhanden, in jedem Modus) ---
    portfolio_analysis = result.get("portfolio_analysis")
    if isinstance(portfolio_analysis, dict):
        lines.extend(_portfolio_blick_section(portfolio_analysis))

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

        # Debatten-Konfidenz anzeigen (falls vorhanden)
        bull_conf = debate.get("bull_confidence")
        bear_conf = debate.get("bear_confidence")
        if bull_conf is not None or bear_conf is not None:
            conf_parts: list[str] = []
            if bull_conf is not None:
                conf_parts.append(f"Bull {bull_conf}/5")
            if bear_conf is not None:
                conf_parts.append(f"Bear {bear_conf}/5")
            lines.append(f"_Debatten-Konfidenz: {' vs '.join(conf_parts)}_")
            lines.append("")

        # Reflexion (Track-Record) — vor Trade-Vorschlag
        reflection = result.get("reflection")
        if reflection:
            section_num += 1
            lines.append(f"## {section_num}. Reflexion (Track-Record)")
            lines.append("")
            lines.append(reflection)
            lines.append("")

        # Trade
        section_num += 1
        trade = result.get("trade", {})
        lines.append(f"## {section_num}. Trade-Vorschlag")
        lines.append("")
        # Trade-Revision Hinweis (Feature A)
        if result.get("trade_revised"):
            lines.append("> ⚠️ Trade wurde nach Risk-/Portfolio-Fit-Einwand revidiert.")
            lines.append("")
            original = result.get("trade_original")
            if isinstance(original, dict):
                lines.append(
                    f"_Original-Trade: {original.get('aktion', 'N/A')}"
                    f" → revidiert: {trade.get('aktion', 'N/A')}_"
                )
                lines.append("")
        lines.append(f"**Aktion:** {trade.get('aktion', 'N/A')}")
        if trade.get("rating"):
            lines.append(f"**Rating (5-stufig):** {trade['rating']}")
        # Entscheidungs-Disziplin: Hinweis bei gedämpftem Rating
        if trade.get("rating_gedämpft"):
            original = trade.get("rating_original", "STARK KAUFEN/STARK VERKAUFEN")
            lines.append(
                f"_⚠️ Rating gedämpft ({original} → {trade['rating']}) "
                f"wegen überkonfidenter Historie_"
            )
        lines.append(f"**Zielkurs:** {_fmt(trade.get('zielkurs'))}")
        lines.append(f"**Stop-Loss:** {_fmt(trade.get('stop_loss'))}")
        lines.append(f"**Positionsanteil:** {trade.get('positionsanteil', 'N/A')} %")
        lines.append(f"**Zeithorizont:** {trade.get('zeithorizont', 'N/A')}")
        lines.append(f"**Begründung:** {trade.get('begründung', 'N/A')}")
        lines.append("")

        # Ensemble-Info anzeigen (falls vorhanden)
        ensemble_info = trade.get("_ensemble")
        if ensemble_info:
            ens_runs = ensemble_info.get("runs", "?")
            ens_aktion = ensemble_info.get("mehrheits_aktion", "N/A")
            ens_conf = ensemble_info.get("ensemble_confidence", 0)
            ens_conf_pct = int(ens_conf * 100) if isinstance(ens_conf, float | int) else 0
            ens_line = (
                f"_Ensemble: {ens_runs} Runs, Aktion {ens_aktion} "
                f"(Konfidenz {ens_conf_pct}%)_"
            )
            alle_ratings = ensemble_info.get("alle_ratings")
            if alle_ratings:
                ens_line += f", Rating-Verteilung: {', '.join(alle_ratings)}"
            lines.append(ens_line)

            # Kalibrierungs-Gewichtungs-Hinweis (nur wenn aktiv)
            if ensemble_info.get("gewichtet"):
                gewichte = ensemble_info.get("aktion_gewichte") or {}
                gewicht_teile = []
                for action in ("KAUFEN", "HALTEN", "VERKAUFEN"):
                    g = gewichte.get(action)
                    if g is not None:
                        gewicht_teile.append(f"{action} {g:.2f}")
                if gewicht_teile:
                    lines.append(
                        f"_Ensemble kalibrierungs-gewichtet ({', '.join(gewicht_teile)})_"
                    )
            lines.append("")

        # Risk
        section_num += 1
        risk = result.get("risk", {})
        lines.append(f"## {section_num}. Risiko-Bewertung")
        lines.append("")
        lines.append(f"**Risiko-Score:** {risk.get('risiko_score', 'N/A')} / 5")
        lines.append(f"**Volatilität:** {risk.get('volatilität_bewertung', 'N/A')}")
        lines.append(f"**Max. Drawdown (Schätzung):** {risk.get('max_drawdown_schaetzung', 'N/A')}")
        lines.append(f"**Empf. Positionsgröße (LLM):** {risk.get('positionsgröße_empfohlen', 'N/A')}")
        lines.append(
            f"**Positionsgröße (rechnerisch, Volatility-Targeting):** "
            f"{risk.get('positionsgröße_rechnerisch_pct', 'N/A')} %"
        )
        if risk.get("volatilität_annualisiert_pct") is not None:
            lines.append(
                f"**Volatilität (annualisiert):** {risk.get('volatilität_annualisiert_pct', 'N/A')} %"
            )
        lines.append(f"**Auflagen:** {risk.get('auflagen', 'N/A')}")
        lines.append(f"**Empfehlung:** {risk.get('empfehlung', 'N/A')}")
        lines.append("")

        # Portfolio-Fit (nur wenn ein dict vorhanden ist)
        portfolio_fit = result.get("portfolio_fit")
        if isinstance(portfolio_fit, dict):
            section_num += 1
            lines.append(f"## {section_num}. Portfolio-Fit")
            lines.append("")
            pf_score = portfolio_fit.get("portfolio_fit_score", "N/A")
            lines.append(f"**Portfolio-Fit-Score:** {pf_score} / 5")
            lines.append(f"**Ziel-Gewichtung:** {portfolio_fit.get('ziel_gewichtung_pct', 'N/A')} % des Portfolios")
            lines.append(f"**Konzentrationsrisiko:** {portfolio_fit.get('konzentrationsrisiko_bewertung', 'N/A')}")
            lines.append(f"**Sektor-/Branchen-Overlap:** {portfolio_fit.get('sektor_overlap_bewertung', 'N/A')}")
            lines.append(f"**Begründung:** {portfolio_fit.get('begründung', 'N/A')}")
            if portfolio_fit.get("portfolio_daten_verfuegbar") is False:
                lines.append("")
                lines.append("> ⚠️ Portfolio-Daten nicht verfügbar — nur Sektor-Bewertung.")
            lines.append("")

        # Final
        section_num += 1
        final = result.get("final", {})
        lines.append(f"## {section_num}. Finale Entscheidung (Portfolio-Manager)")
        lines.append("")
        entscheidung = final.get("entscheidung", "N/A")
        entscheidung_upper = str(entscheidung).upper()
        if "GENEHMIGT" in entscheidung_upper:
            emoji = "✅"
        elif "MODIFIZIERT" in entscheidung_upper:
            emoji = "⚡"
        else:
            emoji = "❌"
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
    lines.append("*Erstellt von Concilium · Keine Anlageberatung*")

    # Feature 4: Journal-Hinweis (nur im LLM-Modus, wenn Eintrag geschrieben)
    if not no_llm and result.get("_journal_written"):
        lines.append("")
        lines.append("Entscheidung im Journal gespeichert: journal/decisions.csv")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Track-Record-Report
# --------------------------------------------------------------------------- #


def _fmt_pct2(val: Any) -> str:
    """Formatiert einen Anteil (0-1) als Prozentangabe."""
    if val is None:
        return "N/A"
    try:
        fval = float(val)
        if math.isnan(fval):
            return "N/A"
        return f"{fval * 100:.1f} %"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_num(val: Any, suffix: str = "") -> str:
    """Formatiert eine Zahl (z.B. Rendite) mit 2 Nachkommastellen."""
    if val is None:
        return "N/A"
    try:
        fval = float(val)
        if math.isnan(fval):
            return "N/A"
        return f"{fval:.2f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def generate_track_record_report(eval_result: dict[str, Any]) -> str:
    """Generiert einen deutschen Markdown-Track-Record-Report.

    Erzeugt einen strukturierten Report mit Übersicht, Aktions-Tabellen,
    Konfidenz-Bändern, Portfolio-Fit-Zusammenhang und LLM-Zusammenfassung.

    Args:
        eval_result: Das dict aus evaluate_journal().

    Returns:
        Markdown-String mit Tabellen und Disclaimer-Footer.
    """
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    lines: list[str] = []

    lines.append("# Concilium Track-Record-Evaluierung")
    lines.append("")
    lines.append(f"**Erstellt am:** {now}")
    lines.append(f"**Bewertungszeitraum:** Letzte {eval_result.get('lookback_days', 90)} Tage (falls zutreffend)")
    lines.append("")

    # Disclaimer
    lines.append(
        '> ⚠️ **Disclaimer:** Dies ist keine Anlageberatung. Die Evaluierung basiert '
        "auf historischen Kursdaten und Heuristiken und dienen nur Demonstrationszwecken."
    )
    lines.append("")

    # Warnung bei übersprungenen Zeilen
    uebersprungen = eval_result.get("uebersprungen", 0)
    n = eval_result.get("anzahl_entscheidungen", 0)
    if uebersprungen and uebersprungen > 0:
        total = uebersprungen + n
        lines.append(
            f"> ⚠️ **Hinweis:** {uebersprungen} von {total} Journal-Entscheidungen "
            "konnten wegen fehlender Kursdaten nicht bewertet werden und wurden "
            "übersprungen. Die Kennzahlen basieren auf einer Teilmenge."
        )
        lines.append("")

    # --- Übersicht ---
    lines.append("## Übersicht")
    lines.append("")
    lines.append("| Kennzahl | Wert |")
    lines.append("|---|---|")
    lines.append(f"| Anzahl Entscheidungen | {n} |")
    lines.append(f"| Hit-Rate gesamt | {_fmt_pct2(eval_result.get('hit_rate_gesamt'))} |")
    lines.append(
        f"| Durchschnittliche Rendite | {_fmt_num(eval_result.get('durchschnitt_rendite_gesamt'), ' %')} |"
    )
    lines.append(
        f"| Zielkurs-Trefferquote | {_fmt_pct2(eval_result.get('zielkurs_trefferquote'))} |"
    )
    lines.append(
        f"| Stop-Verletzungsquote | {_fmt_pct2(eval_result.get('stop_verletzungsquote'))} |"
    )
    lines.append("")

    # --- Tabelle nach Aktion ---
    lines.append("## Bewertung nach Aktion")
    lines.append("")
    lines.append("| Aktion | n | Hit-Rate | Ø Rendite |")
    lines.append("|---|---|---|---|")
    nach_aktion = eval_result.get("nach_aktion", {})
    for action in ("KAUFEN", "HALTEN", "VERKAUFEN"):
        a = nach_aktion.get(action, {})
        lines.append(
            f"| {action} | {a.get('n', 0)} | "
            f"{_fmt_pct2(a.get('hit_rate'))} | "
            f"{_fmt_num(a.get('avg_rendite'), ' %')} |"
        )
    lines.append("")

    # --- Konfidenz-Bänder ---
    konfidenz_baende = eval_result.get("konfidenz_baende", [])
    if konfidenz_baende:
        lines.append("## Konfidenz-Bänder (Trefferquote nach Confidence)")
        lines.append("")
        lines.append("| Band | n | Hit-Rate |")
        lines.append("|---|---|---|")
        for b in konfidenz_baende:
            lines.append(
                f"| {b.get('band', '?')} | {b.get('n', 0)} | "
                f"{_fmt_pct2(b.get('hit_rate'))} |"
            )
        lines.append("")
        lines.append(
            "_Bänder: hoch (≥4), mittel (3), niedrig (≤2). "
            "Steigt die Hit-Rate mit höherer Confidence?_"
        )
        lines.append("")

    # --- Konfidenz-Kalibrierung (Brier-Score, Gap, Reliability-Bänder) ---
    kal = eval_result.get("konfidenz_kalibrierung") or {}
    reliability_bins = eval_result.get("reliability_bins") or []
    if kal.get("brier_score") is not None or reliability_bins:
        lines.append("## Konfidenz-Kalibrierung")
        lines.append("")
        lines.append("| Kennzahl | Wert |")
        lines.append("|---|---|")
        brier = kal.get("brier_score")
        lines.append(f"| Brier-Score | {_fmt_num(brier)} |")
        lines.append(
            f"| Ø Konfidenz (normalisiert) | {_fmt_num(kal.get('durchschnittliche_konfidenz'))} |"
        )
        lines.append(
            f"| Ø tatsächliche Hit-Rate | {_fmt_num(kal.get('durchschnittliche_tatsaechliche_hit_rate'))} |"
        )
        gap = kal.get("kalibrierungs_gap")
        lines.append(f"| Kalibrierungs-Gap | {_fmt_num(gap)} |")
        lines.append(f"| Tendenz | {kal.get('tendenz', 'N/A')} |")
        lines.append(f"| n (bewertete Zeilen) | {kal.get('n', 0)} |")
        lines.append("")

        # Reliability-Bänder-Tabelle
        if reliability_bins:
            lines.append("### Reliability-Bänder")
            lines.append("")
            lines.append("| Konfidenz-Bin | n | Ø Konfidenz | Hit-Rate |")
            lines.append("|---|---|---|---|")
            for rb in reliability_bins:
                lines.append(
                    f"| {rb.get('bin', '?')} | {rb.get('n', 0)} | "
                    f"{_fmt_num(rb.get('mittlere_konfidenz'))} | "
                    f"{_fmt_pct2(rb.get('hit_rate'))} |"
                )
            lines.append("")
            lines.append(
                "_Ideale Kalibrierung: Hit-Rate ≈ Konfidenz pro Bin._"
            )
            lines.append("")

        # Segmentierte Brier-Scores (pro Aktion / pro Rating)
        seg = eval_result.get("konfidenz_kalibrierung_segmentiert") or {}
        nach_aktion_seg = seg.get("nach_aktion", {})
        nach_rating_seg = seg.get("nach_rating", {})

        if nach_aktion_seg:
            lines.append("### Nach Aktion")
            lines.append("")
            lines.append("| Aktion | n | Brier | Ø Konfidenz | Hit-Rate | Tendenz |")
            lines.append("|---|---|---|---|---|---|")
            for action in ("KAUFEN", "HALTEN", "VERKAUFEN"):
                a = nach_aktion_seg.get(action)
                if a is None:
                    continue
                lines.append(
                    f"| {action} | {a.get('n', 0)} | "
                    f"{_fmt_num(a.get('brier_score'))} | "
                    f"{_fmt_num(a.get('durchschnittliche_konfidenz'))} | "
                    f"{_fmt_pct2(a.get('durchschnittliche_tatsaechliche_hit_rate'))} | "
                    f"{a.get('tendenz', 'N/A')} |"
                )
            lines.append("")

        if nach_rating_seg:
            lines.append("### Nach Rating")
            lines.append("")
            lines.append("| Rating | n | Brier | Ø Konfidenz | Hit-Rate | Tendenz |")
            lines.append("|---|---|---|---|---|---|")
            for rating in (
                "STARK KAUFEN", "KAUFEN", "HALTEN", "VERKAUFEN", "STARK VERKAUFEN",
            ):
                r = nach_rating_seg.get(rating)
                if r is None:
                    continue
                lines.append(
                    f"| {rating} | {r.get('n', 0)} | "
                    f"{_fmt_num(r.get('brier_score'))} | "
                    f"{_fmt_num(r.get('durchschnittliche_konfidenz'))} | "
                    f"{_fmt_pct2(r.get('durchschnittliche_tatsaechliche_hit_rate'))} | "
                    f"{r.get('tendenz', 'N/A')} |"
                )
            lines.append("")

    # --- Portfolio-Fit ---
    pf = eval_result.get("portfolio_fit_hoch")
    if pf is not None:
        lines.append("## Portfolio-Fit-Zusammenhang")
        lines.append("")
        lines.append("| Kennzahl | Wert |")
        lines.append("|---|---|")
        lines.append(f"| n (Portfolio-Fit ≥ 4) | {pf.get('n', 0)} |")
        lines.append(f"| Hit-Rate (hohes Portfolio-Fit) | {_fmt_pct2(pf.get('hit_rate'))} |")
        lines.append("")

    # --- LLM-Zusammenfassung ---
    zusammenfassung = eval_result.get("zusammenfassung")
    if zusammenfassung:
        lines.append("## LLM-Zusammenfassung")
        lines.append("")
        lines.append(zusammenfassung)
        lines.append("")

    # --- Fehler ---
    fehler = eval_result.get("fehler", [])
    if fehler:
        lines.append("## Fehlerhinweise")
        lines.append("")
        for f in fehler:
            lines.append(f"- {f}")
        lines.append("")

    # --- Footer ---
    lines.append("---")
    lines.append("*Erstellt von Concilium Track-Record-Evaluator · Keine Anlageberatung*")

    return "\n".join(lines)
