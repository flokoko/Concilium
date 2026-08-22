# Concilium — Roadmap: „Anschluss an TradingAgents & besser"

**Stand:** 2026-08-22 · **Status:** ✅ alle 5 Phasen umgesetzt
**Ziel:** Concilium strukturell an das Vorbild (TauricResearch/TradingAgents) annähern und
zugleich die vorhandenen Alleinstellungsmerkmale ausbauen, sodass es dort, wo es zählt,
besser ist als das Original.

> Strategie-Prinzip: **Nicht blind nachbauen.** TradingAgents ist ein produktionsreifes,
> provider-reiches Framework (LangGraph, Docker, viele LLM-Anbieter, Broker-Execution).
> Diese Dinge sind für ein Einzelnutzer-CLI großteils unnötig. Wir übernehmen nur die
> *harten strukturellen Stärken*, die Concilium wirklich fehlen, und investieren den Rest
> in die Differenzierung, wo Concilium das Original bereits übertrifft.

---

## Analyse: Wo wir stehen

**Geerbte DNA (gemeinsam mit Original):** Analysten-Team → Bull/Bear-Debatte → Trader →
Risk → PM. Lern-Mechanismen Decision-Log + realisierter Return + Reflexion. Disclaimer.

**Concilium-Stärken (bereits besser als Original):**
- Portfolio-Fit auf das **echte Depot** (Google-Sheet) + **Portfolio-Ebene** (`--portfolio`)
- Deterministische Anker: Multi-Faktor-Score, Volatility-Targeting, Datenqualitäts-Validierung
- Ensemble-Trader (Temperatur-Abstimmung + Plausibilitäts-Check)
- Track-Record-Evaluierung (`--evaluate`) mit **Konfidenz-Kalibrierung**, Trade-Revision (2nd Pass), deutsche Reports

---

## Phase-Übersicht (Umsetzungs-Status)

| Phase | Inhalt | Hebel | Aufwand | Status |
|---|---|---|---|---|
| **P0** | Strukturierte LLM-Outputs | Fundament; behebt fragilste Bugs | mittel-groß | ✅ umgesetzt |
| **P2** | Portfolio-Ebene (`--portfolio`) | Alleinstellungsmerkmal, direkt für Flo nutzbar | groß | ✅ umgesetzt |
| **P1** | Checkpoint-Resume / Crash-Resilienz | Batch-/Crash-Resilienz | mittel | ✅ umgesetzt |
| **P4** | Konfidenz-Kalibrierung | Lernen messbar machen | mittel | ✅ umgesetzt |
| **P3** | Sozial-Sentiment (StockTwits + Reddit) | Sentiment-Dichte erhöhen | mittel | ✅ umgesetzt |

*Umsetzung erfolgte phasenweise via coding-subagent (GLM-5.2), jeweils mit
ruff + pytest + Push + CI + Live-Test. Teststand: 830 Tests.*

---

## PHASE 0 — Fundament: Strukturierte LLM-Outputs  *(größter technischer Hebel)* ✅

**Warum zuerst:** Alle Parser-Fixes sind Symptom-Kurat. `_extract_current_price` via Regex,
`_parse_debate_confidence` aus JSON-Preamble im Fließtext — jeder Formulierungsbruch des
Modells erzeugte `+nan%` oder `fehler`-Einträge (bekannter `+nan%`-Bug). Strukturierte
Outputs eliminieren diese Klasse von Fehlern dauerhaft und sind die Voraussetzung für
stabile Automatisierung (Portfolio-Ebene, Batch, Scheduler).

**Umgesetzt:**
- `LLMClient.chat()` unterstützt OpenAI-kompatibles `response_format` / `json_schema`
  (neuer optionaler Parameter, Default ohne Schema = unverändertes Verhalten).
- `trader` / `risk_manager` / `portfolio_manager` / `debate` / `analyst_team` liefern
  garantiert getyptes JSON (Schema-Definitionen in `schemas.py`); kein Regex-Parsing mehr
  in diesen Pfaden.
- Fehlende Schema-Felder werden mit sicheren Defaults aufgefüllt (`defaults_for_schema`)
  — der Report bricht nie durch fehlende Keys oder `+nan%`.
- **Fallback**: Wenn der Provider kein `response_format` unterstützt (z. B. lokales Ollama),
  fällt der Client automatisch auf das bisherige Parsing zurück (rückwärtskompatibel).
  Der echte GLM-5.2 akzeptiert das `response_format` (live verifiziert, `response_format_used=True`).

---

## PHASE 1 — Crash-Resilienz / Checkpoint-Resume  *(Original-Feature, schlank nachgebaut)* ✅

**Warum:** Das Original setzt via LangGraph-Checkpoint abgebrochene Läufe an der Stelle
fort. Concilium startete bei jedem 429/Crash neu. Für deterministische Läufe (lang, viele
LLM-Calls) teuer und frustrierend. Ein eigenes, schlankes State-Persistenz-Modul reicht.

**Umgesetzt:**
- `checkpoint.py` (neu): save/load/clear von Pipeline-Zwischenzuständen (atomar + fcntl-Lock,
  `CONCILIUM_STATE_DIR`-Übersteuerung, tolerante JSON-Serialisierung).
- `pipeline.py`: Zwischenstände nach **jedem** Agent-Schritt schreiben (Buchführung via
  `_completed_steps`).
- `cli.py`: `--resume` / `--no-resume`, sauberes `SIGINT`-Handling (Exit-Code 130, Checkpoint bleibt).
- Erfolgreicher Lauf räumt seinen Checkpoint auf. `--evaluate` bleibt unberührt.
- **Verifikation**: deterministischer Unit-Test — Fake-LLM schlägt nach Schritt N fehl,
  zweiter Lauf mit `--resume` führt nur N..Ende aus (frühere Agenten-Calls nicht wiederholt).

---

## PHASE 2 — Portfolio-Ebene (unser größter Vorteil)  *(Differenzierung)* ✅

**Warum:** TradingAgents analysiert Aktien isoliert. Concilium kennt bereits Florians
echtes Depot. Entscheidungen **zusammen** zu betrachten — über mehrere Ticker die
Korrelation, den Sektor-/Exposure-Overlap und die Gesamtkonzentration — hat das Original
strukturell nicht und ist für einen Fondsmanager direkt nutzbar.

**Umgesetzt:**
- `portfolio_analysis.py` (neu): deterministische Korrelations-Matrix (Pearson auf
  Tagesrenditen), Overlap-Erkennung gegen den Sheet-Bestand, Konzentrationswarnungen.
- `--portfolio TICKER1,TICKER2,...`-Modus: analysiert mehrere Ticker als Depot-Ganzheit.
- PM-Entscheidung berücksichtigt den **Gesamt-Exposure-Kontext** (geplante Positionen +
  Bestand + Korrelationen). Der PM läuft genau **einmal** pro Ticker mit Kontext
  (`skip_final`-Mechanik); das Journal wird konsistent mit dem finalen Ergebnis geschrieben.
- Report-Sektion **„Portfolio-Blick"**: Ziel-Gewichtungen, Korrelations-Matrix (Paare mit
  |r| > 0.7 hervorgehoben), Konzentrations-/Overlap-Warnungen.
- Bestehende Modi (`--ticker` / `--tickers` / `--evaluate`) unverändert.

---

## Phase 3 — Sozial-Sentiment erweitern  *(Original-Feature)* ✅

**Warum:** TradingAgents nutzt StockTwits + Reddit. Concilium nutzte nur
yfinance/Google-News — der Sentiment-Analyst war dünn (yfinance liefert oft 0 Headlines).

**Umgesetzt:**
- `data.py`: `_fetch_stocktwits()` + `_fetch_reddit()` — öffentliche Endpoints, **ohne
  API-Key**, rate-limit-respektvoll (je ein Call, Timeout), nie crashen.
- Sentiment-Aggregation zählt alle Quellen (yfinance → Google-News → StockTwits → Reddit);
  gewichtete Stimmungs-Zählung über alle Quellen.
- Report zeigt **Quelle je Headline** (`[yfinance]` / `[Google]` / `[StockTwits]` / `[Reddit]`).
- Fallback-Kaskade bleibt (keine Quelle = bisheriges Verhalten, kein Crash).
- **Hinweis (Stand 2026-08-22):** In der Live-Container-Umgebung liefern StockTwits und
  Reddit HTTP 403 (geblockte IP). Die Fallback-Kaskade greift sauber; die
  Code-Integration (inkl. Quellen-Tagging) ist über Unit-Tests mit gemockten Antworten bewiesen.
  Die echten Calls greifen in einem nicht-blockierten Netzwerk (z. B. Florians Rechner).

---

## Phase 4 — Lernen harten (Konfidenz-Kalibrierung)  *(Differenzierung)* ✅

**Warum:** Beide Systeme haben Reflexion/Kontext-Feedback. Aber nur Concilium hat bereits
`--evaluate`. Die Lern-Schleife lässt sich **quantifizieren**: über viele Läufe prüfen,
ob hohe Konfidenz wirklich öfter richtig ist (Konfidenz-Kalibrierung) und ob Portfolio-Fit
mit Erfolg korreliert. Das Original misst das nicht.

**Umgesetzt:**
- `evaluate.py`: neue Kennzahl **Konfidenz-Kalibrierung** — **Brier-Score** (binär,
  `(confidence/5 − hit)²`), **Kalibrierungs-Gap** (Ø-Konfidenz vs. Hit-Rate) +
  **Tendenz-Klassifikation** (über-/unterkonfident / gut kalibriert), **Reliability-Bänder**
  (Konfidenz-Intervalle mit n / Ø-Konfidenz / Hit-Rate).
- `feedback.py`: Kalibrierungs-Tendenz als Kontext-Block in Agenten-Prompts (netzfrei,
  GENEHMIGT/ABGELEHNT als Erfolgs-Proxi).
- Report-Erweiterung in `--evaluate`: neue Sektion **„## Konfidenz-Kalibrierung"**.
- **Live-Beweis**: Das System zeigte eine massiv überkonfidente Kalibrierung (Brier 0.62,
  Gap 0.73, Hit-Rate 10 %) — genau die Diagnose, die das Feature liefern soll.

---

## Abseits / bewusst NICHT übernommen (bewusst)

- **LangGraph / Framework-Lock-in:** Nicht nötig — unser Checkpoint-Modul (Phase 1)
  liefert denselben Nutzen ohne Abhängigkeit.
- **Viele LLM-Provider (OpenAI, Azure, Bedrock, …):** Concilium nutzt genau ein
  OpenAI-kompatibles Backend (GLM-5.2 via Ollama-Cloud). Kein Mehrwert.
- **Broker-Execution / Enterprise:** Research/Demo-Pfad, keine echte Orderausführung —
  Absichtlich.
- **Interaktive TUI / Docker:** Concilium ist ein schlankes CLI für einen Einzelnutzer.

---

## Nächste Schritte (optional)

Alle priorisierten Phasen sind umgesetzt. Potenzielle nächste Schritte (nicht priorisiert):

- Sozial-Sentiment real aus einer nicht-blockierten Umgebung verifizieren (StockTwits/Reddit).
- Weitere Portfolio-Analysen (z. B. Sektor-Overlap-Matrix als eigene Report-Tabelle).
- Ausbau der Konfidenz-Kalibrierung (z. B. pro-Aktion- oder pro-Rating-segmente Brier-Scores).
