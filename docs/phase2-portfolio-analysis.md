# Concilium — Phase 2: Portfolio-Ebene (Korrelation / Overlap / Gesamt-Exposure)

> **Handoff-Dokument für eine frische, kontextfreie Session.**
> Lies diese Datei vollständig und folge ihr. Zielbild aus `docs/ROADMAP.md` (Phase 2).

---

## Kontext (für die Session, die das umsetzt)

Concilium ist ein Multi-Agenten-Fonds-Entscheidungssystem (Python-CLI, Repo
`/opt/data/Concilium`, GitHub `flokoko/Concilium`). Die Pipeline
(`src/concilium/pipeline.py::run_pipeline`) analysiert einen einzelnen Ticker
sequenziell über LLM-Agenten. `--tickers` (Batch) analysiert mehrere Ticker
**isoliert nacheinander** — ohne gemeinsamen Kontext.

**Das Problem dieser Phase:** TradingAgents analysiert Aktien isoliert.
Concilium kennt bereits Florians echtes Depot (Google-Sheet). Der nächste Schritt:
Entscheidungen **zusammen** betrachten — über mehrere Ticker die Korrelation, den
Sektor-/Exposure-Overlap und die Gesamtkonzentration. Das hat das Original
strukturell nicht und ist für einen Fondsmanager direkt nutzbar — das
**Alleinstellungsmerkmal** von Concilium.

**Ziel:** Ein neuer `--portfolio`-Modus: mehrere Ticker werden nicht isoliert,
sondern als Depot als Ganzheit analysiert. Die PM-Entscheidung berücksichtigt
**Gesamt-Exposure**: geplante Positionen + Bestand + Korrelationen zwischen den
analysierten Titeln. Neue Report-Sektion „Portfolio-Blick": Ziel-Gewichtungen,
Konzentrationswarnung über alle analysierten Titel hinweg, Sektor-Overlap-Matrix.

**Konventionen (unbedingt einhalten):**
- Lauf: `/opt/data/depot-venv/bin/python` (NICHT System-Python).
- LLM-Key aus `OLLAMA_API_KEY` in `/opt/data/.env`.
- Ruff-Check: `/opt/data/depot-venv/bin/python -m ruff check src/ tests/ main.py`
- Tests: `/opt/data/depot-venv/bin/python -m pytest tests/ -q` (695 Tests, offline).
- Verifikationspflicht: ruff grün + pytest grün + `git push` + CI grün + Live-Test
  mit echtem LLM (Key aus .env) für einen realistischen Korpus.
- `main.py` hat `sys.path.insert`-Workaround → `# noqa: E402, I001` auf Import belassen.
- `state/` bleibt in `.gitignore`.

---

## Derzeitige Architektur (relevant)

**`src/concilium/portfolio_fit.py`** — bewertet EINE Aktie als Baustein im realen Depot:
- `fetch_portfolio_positions()` → Liste von dicts `{name, ticker, sheet_symbol, type, region, depot_pct, value_eur}` aus Florians Google-Sheet (mit Tages-Cache).
- `_parse_positions(csv_text)` — testbar ohne Netzwerk.
- `_build_portfolio_text(positions)` / `_build_portfolio_summary(positions)` — Typen-/Regionen-Allokation + Top-10.
- `portfolio_fit_agent(data, llm, positions, data_text=None)` — ruft LLM für Portfolio-Fit-Score auf.

**`src/concilium/pipeline.py` — `run_pipeline(ticker, llm, backtest, peers, ensemble, ensemble_runs, resume=False)`**
- Schritt 1: `data = collect_ticker_data(...)`; Schritt 1b `data_text`; 1c feedback; 1d reflection; Checkpoint (Phase 1) pro Ticker.
- Schritt 5b: `portfolio_fit` via `portfolio_fit_agent(...)` mit `positions = fetch_portfolio_positions()`.
- Schritt 6: `portfolio_manager(trade, risk, llm, portfolio_fit=...)`.

**`src/concilium/report.py`** — Renderer:
- `_management_summary(result, no_llm)`, `generate_report(result, reports_dir=None)`.
- Sektionen nummeriert (Übersicht, Technik, Sentiment, [Backtest], Analysten, Debatte, …, Trade, Risk, Portfolio-Fit, PM).
- Portfolio-Fit-Sektion liest `result["portfolio_fit"]` (dict) mit `portfolio_fit_score`, `ziel_gewichtung_pct`, `konzentrationsrisiko_bewertung`, `sektor_overlap_bewertung`.

**`src/concilium/data.py`** — `collect_ticker_data(ticker, peers)`:
- `result["history"]` = `history_records` — Liste von dicts mit `time`/`close`-Werten (täglich). Der exakte Schema von history_records: Prüfe `data.py` Zeile ~1229 ff. und `_compute_annualized_volatility` in `agents.py` (liest `h["close"]`). Für Korrelations-Berechnung brauchst du die `close`-Reihen der analysierten Ticker über einen gemeinsamen Zeitraum.
- `data["technicals"]["current_price"]` — aktueller Kurs.

**`src/concilium/cli.py`** — argparse mit `--ticker`, `--tickers` (Batch, sequenziell), `--evaluate`, `--backtest`, `--no-llm`, `--peers`, `--verbose`, `--no-ensemble`, `--ensemble-runs`, `--resume`/`--no-resume`.

---

## Umsetzungsauftrag

### Schritt 1 — `src/concilium/portfolio_analysis.py` (neu)

Ein deterministisches (kein LLM) Berechnungsmodul für Portfolio-Aggregation über mehrere analysierte Titel + Bestand:

- **Korrelations-Matrix**: Berechne Pearson-Korrelation der Tagesrenditen zwischen den analysierten Ticker-Paaren aus ihrer gemeinsamen Historie. 
  - Eingabe: dict `{ticker: data_dict}` (die collect_ticker_data-Ergebnisse) oder direkt die History-Reihen.
  - Methode: Tagesrenditen `pct_change` aus `close`-Reihen; nur gemeinsame Datenpunkte (inner join auf Datum) verwenden; fehlende Daten tolerant (min_samples, z.B. ≥ 30 überlappende Tage).
  - Rückgabe: Matrix `{tickerA: {tickerB: korrelations_koeffizient}}` (float zwischen -1 und 1), und die zugrunde liegende Sample-Größe.
  - Wenn zu wenig Daten für ein Paar → None/leer (im Report als "n/a").
- **Konzentrations-Maß**: Berechne die **kumulierte Ziel-Gewichtung** der analysierten Titel (aus deren `portfolio_fit.ziel_gewichtung_pct` bzw. als geplante Position) relativ zum Bestand. Warnung, wenn eine Einzelposition > ~5% oder eine Sektor/Region-Gruppe konzentriert ist.
- **Overlap-Berechnung**: Gegen den Bestand (fetch_portfolio_positions) prüfen: überlappen die analysierten Ticker mit bestehenden Positionen (nach Name/Ticker/Region/Sektor)? Aggregiere den Depot-Anteil des überlappten Bestands.
- **API** (mindestens):
  - `compute_correlations(history_map: dict[str, list[dict]]) -> dict[str, dict[str, float]]`
  - `portfolio_overlap(analysed_tickers: list[str], positions: list[dict], analysed_names: dict) -> dict` (Overlap-Warnungen)
  - `portfolio_concentration(positions, weights) -> list[str]` (Konzentrationswarnungen, deutsche Sätze)
- **Deterministisch & robust**: nie crashen, tolerante Eingaben, `math.isfinite()`-Guards.

### Schritt 2 — `pipeline.py`: Portfolio-Kontext durchreichen

- Neuer Modus: `run_pipeline` soll optional einen **Portfolio-Kontext** für mehrere Ticker akzeptieren. Design-Entscheidung: Phase 2 führt einen **Portfolio-Modus** ein, der mehrere Ticker als Ganzheit analysiert. Das ist KEIN einfaches sequenzielles Batch (das ist `--tickers`). Implementiere dies am saubersten so:
  - Neue CLI-Flags (Schritt 4) sammeln eine Ticker-Liste und führen für jeden Ticker die Einzel-Analyse (analyst_team, debate, trader, risk) aus — ABER die **Portfolio-Manager-Entscheidung** und die Report-"Portfolio-Blick"-Sektion bekommen die **Gesamt-Portfolio-Analysis** (Korrelation, Overlap, Konzentration) als Kontext.
  - In `pipeline.py`: füge der PM eine neue optionale `portfolio_context`-Argument hinzu (aggregierte Portfolio-Analysis über alle analysierten Ticker + Bestand). `portfolio_manager(...)` bekommt diesen Kontext im User-Prompt (statt nur portfolio_fit). Die PM-Entscheidung kann so auf Gesamt-Exposure reagieren.
  - Alternativ und einfacher (empfohlen): Ein neues Top-Level-Modul `portfolio_run.py` (oder eine Funktion in pipeline.py), das mehrere run_pipeline-Ergebnisse einsammelt, `portfolio_analysis` berechnet, und dann pro Ticker eine PM-Zweitröhre mit dem Portfolio-Kontext aufruft — oder die Reports aggregiert.
  
  Du hast Gestaltungsspielraum, aber die Ziele sind klar:
  1. Die analysierten Ticker werden **gemeinsam** als Depot betrachtet (nicht isoliert).
  2. PM/Report bekommt Gesamt-Exposure-Informationen.
  3. Rückwärtskompatibel: bestehender `--tickers`/`--ticker`-Modus unverändert.

  Empfohlener Ansatz (robust, minimal-invasiv):
  - Füge `run_pipeline` einen optionalen Param `portfolio_context: dict | None = None` hinzu. Wenn gesetzt, wird er in die PM-User-Prompt injiziert (als JSON-Text „Gesamt-Exposure/Overlap/Konzentration des analysierten Set").
  - Füge ein neues Top-Level-Modul/Funktion `run_portfolio(tickers, llm, ...) -> dict` hinzu, das:
    1. Für jeden Ticker `run_pipeline` ausführt (mit `skip_final=True`-Option ODER führt die Vor-Schritte manuell aus und sammelt die Teil-Ergebnisse; DANN die portfolio_analysis über alle analysierten history-Daten + Bestand berechnet;
    2. Für jeden Ticker einen **Portfolio-Manager** mit dem Gesamt-Kontext (portfolio_context) aufruft (Zweit-Pass oder als PM), und das finale Ergebnis zurückgibt.
  - Du darfst es so umsetzen, dass die PM nur einmal läuft und die aggregate Kontext bekommt, ODER dass es pro Ticker einen PM-Lauf mit Kontext gibt. Wähle den saubersten. Achte auf Kostenkontrolle (nicht doppelt so viele LLM-Calls wie nötig).
- Achte auf Rückwärtskompatibilität: `run_pipeline` Signature unverändert bis auf neue optionale Params; bestehende Tests grün.

## Schritt 3 — `report.py`: „Portfolio-Blick"-Sektion

- Neue Report-Sektion „## Portfolio-Blick" (nach der Portfolio-Fit-Sektion, vor PM oder nach PM):
  - **Ziel-Gewichtungen** der analysierten Titel (aus portfolio_fit.ziel_gewichtung_pct, sofern vorhanden).
  - **Korrelations-Matrix** zwischen den analysierten Titeln (aus portfolio_analysis; kompakt, z.B. Tabelle). Hervorheben von Paaren mit |r| > 0.7 (hohe Korrelation = wenig Diversifikation).
  - **Konzentrationswarnung** über alle analysierten Titel hinweg (Einzelposition > 5%, Sektor/Region-Overlap, kumulierte Gewichtung).
  - **Sektor-Overlap-Matrix** oder zumindest Sektor-Overlap-Warnungen.
  - Robust gegen fehlende Daten (N/A).
- Diese Sektion erscheint NUR wenn Portfolio-Daten vorhanden (portfolio_analysis im result). Bestehende Einzel-Reporte (--ticker, --tickers) ohne portfolio_context bekommen KEINE neue Sektion (Rückwärtskompatibilität).

## Schritt 4 — `cli.py`: `--portfolio`-Modus

- Neues Flag `--portfolio TICKER1,TICKER2,...` (kommagetrennt), das den neuen Portfolio-Modus aktiviert. Es schließt sich mit `--ticker` und `--tickers` gegenseitig aus (argparse mutual exclusion group oder parser.error).
- `--portfolio` führt die Portfolio-Analyse aus (siehe Schritt 2) und ruft `generate_report` mit der Portfolio-Blick-Sektion.
- `--evaluate` bleibt unberührt.
- Verhalten: Wenn `--portfolio` gesetzt → neuer Modus; sonst bisheriges Verhalten.

## Schritt 5 — Tests

- **`tests/test_portfolio_analysis.py`** (neu): deterministische Korrelations-Berechnung mit konstruierten history-Reihen (perfekt korrelierte Reihen → r=1.0, anti-korreliert → r=-1.0, unabhängig → ≈0), Overlap-Erkennung, Konzentrationswarnung. Testbar ohne Netz (Fixtures/konstruierte Daten).
- **Tests für CLI**: `--portfolio`-Flag parsing; Mutual-Exclusion mit `--ticker`/`--tickers`.
- **Tests für report**: Portfolio-Blick-Sektion erscheint nur bei portfolio_analysis im result; robust gegen fehlende Werte.
- **Bestehende Tests grün halten** (695+). Nutze MagicMock für LLM in Tests.

## Verifikations-Checkliste (REIHENFOLGE ERZIELEN)
1. `/opt/data/depot-venv/bin/python -m ruff check src/ tests/ main.py` → 0 Fehler.
2. `/opt/data/depot-venv/bin/python -m pytest tests/ -q` → alle grün (695 + neue).
3. `git add -A && git commit` — aussagekräftige Message.
4. `git push origin main`.
5. GitHub Actions CI abwarten (API). CI grün.
6. **Live-Test** mit echtem LLM (z.B. RWE.DE + SHEL.L + NEE — erneuerbarer-Sektor-Beispiel):
   ```
   cd /opt/data/Concilium
   export LLM_API_KEY=$(grep -oP '^OLLAMA_API_KEY=\K.*' /opt/data/.env | tr -d '"' | tr -d "'" | head -1)
   export LLM_BASE_URL="https://ollama.com/v1"
   export LLM_MODEL="glm-5.2:cloud"
   /opt/data/depot-venv/bin/python main.py --portfolio RWE.DE,SHEL.L,NEE
   ```
   Report prüfen: „Portfolio-Blick"-Sektion mit Korrelations-Matrix, Ziel-Gewichtungen, Konzentrations-/Overlap-Warnungen; keine `+nan%`/`fehler`.
7. Abschluss-Meldung an Flo: was umgesetzt wurde, wie Portfolio-Blick funktioniert, was aggregiert wird, Verifikations-Ergebnis.

---

## Fallstricke / Bewertung
- **Rückwärtskompatibilität**: `run_pipeline` neue optionale Params; `--ticker`/`--tickers`/`--evaluate` unverändert. Bestehende 695 Tests grün.
- **Korrelations-Daten**: history kann Lücken haben → gemeinsames Fenster, min Sample (z.B. ≥30), `math.isfinite()`. Nie durch NaN crashen.
- **Sheet-Loader wiederverwenden**: `fetch_portfolio_positions()` + `_build_portfolio_text()` sind die Quelle für Bestand (nicht neu implementieren).
- **Kein doppelter LLM-Aufwand**: Portfolio-Modus soll nicht z.B. 2x PM pro Ticker auslösen, wenn ein einzelner mit Kontext reicht. Balance zwischen "Gesamt-Portfolio-Entscheidung" und Laufzeit.
- **Report nicht brechen**: fehlende portfolio_analysis → kein Portfolio-Blick Sektion, kein Fehler.
- **Test-Isolation**: portfolio_analysis deterministisch ohne Netz; tests nutzen konstruierte history.

---

## Done-Definition
- `portfolio_analysis.py` (neu): Korrelations-Matrix, Overlap, Konzentrationswarnung — deterministisch, robust, keine NaN.
- Neuer `--portfolio TICKER,...`-Modus in cli.py + `run_pipeline`/Top-Level-Funktion, der analysierte Titel als Depot als Ganzheit betrachtet und dem PM den Gesamt-Exposure-Kontext gibt.
- Report-Sektion „Portfolio-Blick" mit Korrelationsmatrix, Ziel-Gewichtungen, Konzentrations-/Overlap-Warnungen.
- Bestehende Modi (`--ticker`/`--tickers`/`--evaluate`) unverändert.
- 695+ Tests grün, ruff grün, CI grün, Live-Test mit erneuerbarer-Sektor-Korpus ohne `+nan%`/`fehler`, Portfolio-Blick korrekt.

Wenn du das umgesetzt hast: gib die Zusammenfassung an Flo zurück mit den Verifikations-Ergebnissen.
