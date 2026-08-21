# Concilium — Roadmap: „Anschluss an TradingAgents & besser"

**Stand:** 2026-08-21
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
- Portfolio-Fit auf das **echte Depot** (Google-Sheet)
- Deterministische Anker: Multi-Faktor-Score, Volatility-Targeting, Datenqualitäts-Validierung
- Ensemble-Trader (Temperatur-Abstimmung + Plausibilitäts-Check)
- Track-Record-Evaluierung (`--evaluate`), Trade-Revision (2nd Pass), deutsche Reports

**Concilium-Schwächen vs. Original (technischer Schulden, nicht Feature-Lücke):**
1. Fließtext-/Regex-Parsing statt strukturierter LLM-Outputs (fragil, NaN/`fehler`-Bugs)
2. Kein Checkpoint-Resume bei Crash/429 (startet immer von vorn)
3. Dünnes Sentiment (nur yfinance-News + Google-News, kein Sozial-Sentiment)
4. Einzelticker-Fokus, keine Portfolio-Aggregation über mehrere Entscheidungen

---

## PHASE 0 — Fundament: Strukturierte LLM-Outputs  *(größter technischer Hebel)*

**Warum zuerst:** Alle Parser-Fixes sind Symptom-Kurat. `_extract_current_price` via Regex,
`_parse_debate_confidence` aus JSON-Preamble im Fließtext — jeder Formulierungsbruch des
Modells erzeugt `+nan%` oder `fehler`-Einträge (bekannter `+nan%`-Bug). Strukturierte
Outputs eliminieren diese Klasse von Fehlern dauerhaft und sind die Voraussetzung für
stabile Automatisierung (Portfolio-Ebene, Batch, Scheduler).

**Ziele:**
- LLMClient unterstützt OpenAI-kompatibles `response_format` / `json_schema` (ein
  neuer Parameter, Default ohne Schema = unverändertes Verhalten).
- `trader` / `risk_manager` / `portfolio_manager` (und später `debate`) liefern garantiert
  getyptes JSON — keine Regex-Nachbearbeitung mehr in diesen Pfaden.
- Fallback bleibt: Wenn der Provider kein `response_format` unterstützt (Ollama local),
  automatisch auf heutiges Parsing zurückfallen (rückwärtskompatibel).

**Schritte:**
1. `llm.py`: `chat(..., response_format=...)` Parameter + Fehlerbehandlung/Retry bei JSON-Fehlschlag.
2. Prompts: jedes strukturierte Agent-Ergebnis auf ein `json_schema` umstellen.
3. `agents.py`: `_parse_*` durch direkte dict-Rückgabe ersetzen; Konsistenz-Wächter bleibt.
4. Fallback-Logik + ausführliches Test-Handling (Schema-validierte Mocks in Tests).
5. Verdrahtung in `pipeline.py`; `current_price`/`confidence` direkt aus dem strukturierten Feld.

**Verifikation:**
- pytest: neue Schema-Validierungs-Tests, alte Parser-Tests auf Fallback-Pfad umgestellt.
- Live-Test mit echtem GLM-5.2 für 2-3 Ticker (Key aus `.env`); Report ohne `+nan%`/`fehler`.
- ruff + CI grün; Push.

**Aufwand:** mittel-groß · **Risiko:** niedrig-mittel · **Wert:** sehr hoch (Fundament).

---

## PHASE 1 — Crash-Resilienz / Checkpoint-Resume  *(Original-Feature, schlank nachgebaut)*

**Warum:** Das Original setzt via LangGraph-Checkpoint abgebrochene Läufe an der Stelle
fort. Concilium startet bei jedem 429/Crash neu. Für deterministische Läufe (lang, viele
LLM-Calls) teuer und frustrierend. Wir brauchen nicht LangGraph — ein eigenes, schlankes
State-Persistenz-Modul reicht.

**Ziele:**
- Pipeline-Zwischenstände (analyst_team → debate → trader → risk → portfolio_fit → PM)
  werden pro Ticker in eine JSON/SQLite unter `state/` (`.gitignore`) geschrieben.
- CLI-Flag `--resume`: Bei vorhandenem Zwischenstand werden nur die fehlenden Schritte
  neu ausgeführt. Ohne Flag bleibt Verhalten unverändert.
- Abgebrochene/fehlerhafte Runs hinterlassen einen eindeutig markierten Zustand.

**Anforderungen:**
1. `checkpoint.py` (neu): save/load/resume von Pipeline-Zwischenzuständen (fcntl-Lock analog journal).
2. `pipeline.py`: Zwischenstände nach jedem Agent-Schritt schreiben.
3. `cli.py`: `--resume`/`--no-resume`, Verhalten bei Interrupt (`SIGINT`) sauber erfassen.
4. Ablauf ab Version: erfolgreicher Lauf räumt eigenen Checkpoint auf.

**Verifikation:** Unit-Test mit Fake-LLM, der nach Schritt N fehlschlägt → zweiter Lauf
führt nur N..Ende aus. `--evaluate` bleibt unberührt.

**Aufwand:** mittel · **Risiko:** mittel · **Wert:** hoch (nutzbar für Batch)

---

## PHASE 2 — Portfolio-Ebene (unser größter Vorteil)  *(Differenzierung)*

**Warum:** TradingAgents analysiert Aktien isoliert. Concilium kennt bereits Florians
echtes Depot. Der nächste Schritt: Entscheidungen **zusammen** betrachten — über mehrere
Ticker die Korrelation, den Sektor-/Exposure-Overlap und die Gesamtkonzentration. Das hat
das Original strukturell nicht und ist für einen Fondsmanager direkt nutzbar.

**Ziele:**
- Neuer `--portfolio`-Modus: mehrere Tickern werden nicht mehr isoliert, sondern als
  Depot als Ganzheit analysiert.
- PM-Entscheidung berücksichtigt **Gesamt-Exposure**: geplante Positionen + Bestand +
  Korrelationen zwischen den analysierten Titeln.
- Neue Report-Sektion „Portfolio-Blick": Ziel-Gewichtungen, Konzentrationswarnung über
  alle analysierten Titel hinweg, Sektor-Overlap-Matrix.

**Anforderung:**
1. `portfolio_analysis.py` (neu): Korrelations-/Overlap-Berechnung zwischen analysierten
   Titeln + Bestand (bereits vorhandene Sheet-Loader wiederverwenden).
2. `pipeline.py`: Portfolio-Kontext an risk_manager + portfolio_fit + PM durchreichen.
3. `report.py`: „Portfolio-Blick"-Sektion + Warnungen.
4. `cli.py`: `--portfolio TICKER1,TICKER2,...` (vs. bestehendes `--tickers` nur sequenziell).

**Verifikation:** Live-Test mit realistischem Korpus (z.B. RWE.DE + SHEL.L + NEE —
erneuerbarer-Sektor-Beispiel), Prüfen der Konzentrationswarnungen. `--evaluate` unberührt.

**Aufwand:** groß · **Risiko:** mittel · **Wert:** sehr hoch (nur Concilium hat das)

---

## PHASE 3 — Sozial-Sentiment erweitern  *(Original-Feature)*

**Warum:** TradingAgents nutzt StockTwits + Reddit. Concilium nur yfinance/Google-News.
Der Sentiment-Analyst ist aktuell dünn. Zwei Zusatz-Quellen (öffentliche APIs, kostenlos)
machen das Sentiment realistischer.

**Ziel:**
- Sentiment sammelt Headlines aus bis zu 3 Quellen: yfinance / Google-News-RSS /
  StockTwits (öffentlich ohne Key) und optional Reddit.
- `news_source` erweitert sich; gewichtete Stimmungs-Zählung über alle Quellen.
- Fallback-Kaskade bleibt (keine Quelle = `none`).

**Anforderung:**
1. `data.py`: `_fetch_stocktwits()` + `_fetch_reddit()` (Rate-Limit-respektvoll, Zeitstempel).
2. Sentiment-Aggregation in `_count_sentiment_weighted` über Quellen.
3. Report zeigt Quelle je Headline.

**Verifikation:** Live-Test mit NVDA/RWE.DE, prüfen dass Headlines >0 kommen (yfinance
liefert oft 0). Testdaten offline mit Fixtures.

**Aufwand:** mittel · **Risiko:** mittel (Rate-Limits) · **Wert:** mittel

---

## PHASE 4 — Lernen harten (Konfidenz-Kalibrierung)  *(Differenzierung)*

**Warum:** Beide Systeme haben Reflexion/Kontext-Feedback. Aber nur Concilium hat bereits
`--evaluate`. Wir können die Lern-Schleife **quantifizieren**: über viele Läufe prüfen,
ob hohe Konfidenz wirklich öfter richtig ist (Konfidenz-Kalibrierung) und ob Portfolio-Fit
mit Erfolg korreliert. Das Original misst das nicht.

**Anforderung:**
1. `evaluate.py`: neue Kennzahl **Konfidenz-Kalibrierungs-Score** (bspw. Brier-Score /
   Reliability-Bänder) über das bestehende Journal.
2. `feedback.py`: Kalibrierungs-Statistik als Kontext-Block injizieren, damit Agenten
   gezielt über-/unterkonfident korrigieren.
3. Report-Erweiterung in `--evaluate`.

**Aufwand:** mittel · **Risiko:** niedrig · **Wert:** hoch (messbares Lernen)

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

## Priorisierte Roadmap (Empfehlung)

| Priorität | Phase | Hebel | Aufwand |
|---|---|---|---|
| **1** | P0 Strukturierte Outputs | Fundament; behebt fragilste Bugs | mittel-groß |
| **2** | P2 Portfolio-Ebene | Alleinstellungsmerkmal, direkt für Flo nutzbar | groß |
| **3** | P1 Checkpoint-Resume | Batch-/Crash-Resilienz | mittel |
| **4** | P4 Konfidenz-Kalibrierung | Lernen messbar machen | mittel |
| **5** | P3 Sozial-Sentiment | Sentiment-Dichte erhöhen | mittel |

**Nächste Schritte:** Nach Flo's Freigabe → Phase 0 im Detail zerlegen (Stories,
Testliste), via coding-subagent (GLM-5.2) umsetzen, ruff+pytest+push+CI, Live-Test.
