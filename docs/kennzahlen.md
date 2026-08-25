# Concilium — Kennzahlen & Messwerte

Dieses Dokument definiert alle Kennzahlen und Messwerte, die Concilium erzeugt.
Es ist die Referenz, um die Reports und den Track-Record-Evaluator zu verstehen.
Stand: 2026-08-25.

---

## 1. Die zentrale Kennzahl: Brier-Score (Konfidenz-Kalibrierung)

Der Brier-Score misst, **wie gut die angegebene Konfidenz zur tatsächlichen
Trefferquote passt** — nicht "wie oft richtig", sondern "wie ehrlich war die
Selbst-Einschätzung".

**Formel** (pro Entscheidung):

```
Brier_i = (p − hit)²
```

- `p` = normalisierte Konfidenz (Confidence 1–5 → 0.2 bis 1.0, also ÷5)
- `hit` = 1 wenn richtig, 0 wenn falsch
- Gesamt-Brier = Durchschnitt über alle Entscheidungen

**Anschaulich:** Entscheidet Concilium mit Konfidenz 5 (p=1.0) und liegt richtig →
`(1−1)² = 0` (perfekt). Liegt es falsch → `(1−0)² = 1.0` (maximal falsch
eingeschätzt). Bei Konfidenz 3 (p=0.6) und richtig → `(0.6−1)² = 0.16`.

**Interpretation:**

| Wert | Bedeutung |
|---|---|
| 0 | Perfekt kalibriert (Konfidenz = Trefferquote) |
| 0.25 | Zufällig (kein Zusammenhang) |
| 0.5 | Systematisch überkonfident (hohe Konfidenz, wenige Treffer) |
| 1.0 | Maximal falsch eingeschätzt |

---

## 2. Track-Record-Evaluator (`--evaluate`)

Wertet `journal/decisions.csv` gegen die tatsächliche Kursentwicklung aus.

| Kennzahl | Definition | Was sie sagt |
|---|---|---|
| **Hit-Rate gesamt** | % der Entscheidungen mit `hit=True` | Grundqualität des Systems |
| **Hit-Definition** | siehe §3 | Was als "richtig" zählt |
| **Ø Rendite** | Ø Kurswechsel über die Bewertungsperiode (Default 90 Tage) | Wie viel Wert geschaffen wurde |
| **Zielkurs-Trefferquote** | % der Entscheidungen, deren Zielkurs erreicht wurde | Wird die Prognose getroffen? |
| **Stop-Verletzungsquote** | % der Entscheidungen, bei denen der Stop ausgelöst wurde | Risikodisziplin |
| **Konfidenz-Bänder** | Hit-Rate gruppiert nach Confidence (hoch ≥4 / mittel 3 / niedrig ≤2) | Steigt die Qualität mit der Konfidenz? |
| **Brier-Score** | `avg((conf/5 − hit)²)` | Kalibrierung der Selbst-Einschätzung |
| **Kalibrierungs-Gap** | Ø-Konfidenz − Ø-Hit-Rate | Positiv = überkonfident |
| **Tendenz** | Gap >+0.15 über / <−0.15 unter / sonst gut kalibriert | Klassifikation der Kalibrierung |
| **Reliability-Bänder** | Hit-Rate pro feinem Konfidenz-Intervall ([0.6-0.8), [0.8-1.0]) | Detaillierte Kalibrierung |
| **Segmentierte Brier-Scores** | Brier/Gap/Tendenz pro Aktion & Rating | Wo genau liegt die Fehlkalibrierung? |
| **Ø Rating-Distanz** | Stufen zwischen Rating und tatsächlichem Outcome | Wie weit lag die Einschätzung daneben |
| **Portfolio-Fit-Zusammenhang** | Hit-Rate bei Portfolio-Fit ≥4 | Korreliert guter Depot-Fit mit Erfolg? |
| **übersprungen** | Zeilen, die nicht bewertet werden konnten | Datenqualität des Evaluators |

---

## 3. Hit-Definition (Stand 2026-08-25, verfeinert)

Die binäre Hit-Kennzahl fließt in Brier-Score, Konfidenz-Bänder,
Portfolio-Fit-Zusammenhang und das Kontext-Feedback. Definition in
`evaluate.py` → `_evaluate_single`.

**KAUFEN / VERKAUFEN** (fachlich priorisiert):

| # | Bedingung | Hit? |
|---|---|---|
| 1 | Stop gerissen | ❌ **Miss** (Risikoregeln verletzt) |
| 2 | sonst: Ziel erreicht | ✅ **Hit** (Prognose erfüllt) |
| 3 | sonst | Rendite-Richtung (`rendite > 0`; VERKAUFEN invertiert) |

**HALTEN:**

| # | Bedingung | Hit? |
|---|---|---|
| 1 | Stop gerissen | ❌ **Miss** |
| 2 | sonst | `\|Rendite\| ≤ 2%` (strikt) |

**Wichtige Nuance:** Der Stop hat **Vorrang** vor dem Ziel. Wenn der Kurs erst den
Stop gerissen hat und danach wieder zum Ziel gestiegen ist, zählt es als **Miss** —
weil die Positionsführung fehlgeschlagen ist (man wäre ausgestoppt worden).

---

## 4. Einzel-Ticker-Analyse (Report)

| Kennzahl | Definition |
|---|---|
| **Risiko-Score** (1-5) | Vom Risk-Manager; niedriger = besser |
| **Portfolio-Fit-Score** (1-5) | Wie gut passt die Aktie ins echte Depot (Konzentration, Sektor-Overlap) |
| **Debatte Bull vs Bear** | Confidence-Werte der beiden Argumentations-Seiten |
| **Ensemble-Konfidenz** | Ø Konfidenz über die 3 Trader-Runs (Temperatur 0.3/0.5/0.7) |
| **Trade-Urteil** | KAUFEN/HALTEN/VERKAUFEN + 5-stufiges Rating + Zielkurs + Stop + Positionsanteil |
| **Multi-Faktor-Score** | Deterministisch: Value/Momentum/Qualität je 1-5 (Referenz-Anker) |
| **Rechnerische Positionsgröße** | `min(0.02/vol, 0.10)` — Volatility-Targeting |

---

## 5. Backtest (`--backtest`)

| Kennzahl | Definition |
|---|---|
| **Sharpe-Ratio** | Annualisierte Rendite / Volatilität — Rendite pro Risikoeinheit |
| **Max-Drawdown** | Größter Kursverlust von Hoch zu Tief |
| **Win-Rate** | % profitabler Trades |
| **Anzahl Trades** | Anzahl der Signal-Auslöser (SMA50/200 + RSI) |

---

## 6. Portfolio-Ebene (`--portfolio`)

| Kennzahl | Definition |
|---|---|
| **Korrelations-Matrix** | Pearson-Korrelation der Tagesrenditen (Paare \|r\|>0.7 = rot) |
| **Sektor-/Overlap** | Überschneidung geplanter Positionen mit Depot-Bestand |
| **Konzentrationswarnung** | Wenn eine Position die 5%-Regel verletzt |
| **Ziel-Gewichtung** | Empfohlener Depot-Anteil (%) der analysierten Aktie |

---

## Aktuelle Diagnose (Live, 2026-08-25, 32 Entscheidungen)

- **Hit-Rate gesamt:** 34.4 %
- **Brier-Score KAUFEN:** 0.41 · HALTEN: 0.55 · VERKAUFEN: 1.00
- **Tendenz:** überkonfident in allen Segmenten (Gap > 0.15)
- **Kernbefund:** Concilium ist mäßig treffsicher (~34 %) und deutlich
  überkonfident. Die Fehlkalibrierung ist **systemisch** — sie tritt in allen
  Aktionen und Ratings gleichmäßig auf, nicht bei einzelnen Agenten.
