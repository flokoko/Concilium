# INVEST.md — Rollen-Schemata der Concilium-Analysten

INVEST.md-Dateien sind strukturierte Dokumentation des Entscheidungssystems,
das jede Analyst-Rolle von Concilium intern kodiert. Sie zerlegen jede Rolle
in dieselbe Kette: Philosophie → Signale → Filter → Invalidierungs-Regeln →
Sizing/Risiko → Monitoring → Playbook. Sie beschreiben exakt das, was die
SYSTEM_*-Prompts in `src/concilium/agents.py` vorgeben, und die Daten-Sektionen,
die jede Rolle via `_build_data_text()` erhält — als lesbare Referenz mit
maschinenlesbarem YAML-Frontmatter.

Diese Schemata sind die Referenz für die zukünftigen Pitch- und
Devil's-Advocate-Agenten des **Deliberium**-Moduls (Ideations-Schicht, die
die Concilium-Watchlist speist). Deliberium braucht dieselben Rollen-Schemata,
um Thesen zu formulieren und anzugreifen; Concilium bewertet und entscheidet.

## Die 5 Rollen-Dateien

| Datei | Rolle | Kern |
|---|---|---|
| `fundamental.invest.md` | Fundamental-Analyst | Richtungstreiber: Bewertung (KGV, PEG), Wachstum, Margen, Bilanz, Peers, Insider, Analysten-Konsens |
| `technical.invest.md` | Technik-Analyst | Sekundärer Timing-Filter: SMA50/SMA200, RSI(14), MACD, Bollinger, Volumen, relatives Momentum 6M vs. S&P 500 |
| `sentiment.invest.md` | Sentiment-Analyst | Kauf-/Verkaufsförderlichkeit der News-Headlines (Zeitgewichtung, Halbwertszeit 7 Tage, Sample-Größe) |
| `social.invest.md` | Social-Media-Analyst | Retail-Crowd aus StockTwits/Reddit — konträres Reading von Euphorie und Panik, Meme-Dynamik |
| `macro_news.invest.md` | Makro/News-Analyst | Makro-Regime (10y-Zins, VIX, Öl, EURUSD, S&P 500) + materiale Headlines je Sektor, Prediction Markets |

## Gemeinsame Struktur

Jede Datei hat YAML-Frontmatter (`name`, `role`, `universe`, `marketRegime`,
`signals`, `filters`, `invalidation`, `sizing`, `risk`, `monitoring`,
`playbook`) plus die Sektionen `## Philosophie`, `## Analyse-Prozess`,
`## Signale & Filter`, `## Invalidierungs-Regeln`, `## Sizing & Risiko`,
`## Monitoring`, `## Guardrails`.

## Quellen

- SYSTEM_*-Prompts und `_build_data_text()`: `src/concilium/agents.py`
- Stufe 1 (verpflichtendes `invalidation`-Feld im Analysten-JSON):
  falsifizierbare Bedingungen (Kennzahl-Schwelle, Kursniveau oder Ereignis)
  statt Allgemeinplätze wie „wenn der Markt fällt".