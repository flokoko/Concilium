# Concilium

Concilium — Multi-Agenten-Fonds-Entscheidungssystem (yfinance + LLM-Agenten, OpenAI-kompatibel), deutsche Reports.

## Was ist das?

Concilium ist ein eigenständiges CLI-Python-Paket, das eine fondsorientierte
Handelsentscheidung für einen Ticker emuliert. Es ist ein Multi-Agenten-System
(Hedgefonds-Imitat): spezialisierte LLM-Rollen-Agenten arbeiten nacheinander —
von der Datenanalyse über Bull/Bear-Debatte bis zur finalen Genehmigung durch
den Portfolio-Manager. Alle Marktdaten werden via yfinance (kostenlos, kein
API-Key) bezogen. Die Agenten kommunizieren über eine OpenAI-kompatible
`/chat/completions` Schnittstelle.

**Dokumentation der Kennzahlen:** [`docs/kennzahlen.md`](docs/kennzahlen.md) —
Definitionen aller Messwerte (Brier-Score, Hit-Rate, Kalibrierungs-Gap, …).

## Installation

```bash
git clone https://github.com/flokoko/Concilium.git
cd Concilium
pip install -e ".[dev]"
```

Alternativ: einfach das Depot-venv verwenden:

```bash
/opt/data/depot-venv/bin/python main.py --ticker AAPL
```

## Nutzung

```bash
# Vollständige Analyse (mit LLM)
python main.py --ticker NVDA

# Nur Datensnapshot (ohne LLM, kein API-Aufruf)
python main.py --ticker AAPL --no-llm

# Mit Backtest-Signalproxy (SMA50/200-Crossover + RSI-Filter)
python main.py --ticker MSFT --backtest

# Mit Peer-Vergleich (KGV/Marktkap-Tabelle im Report)
python main.py --ticker TSM --peers NVDA,AMD

# Batch-Modus: mehrere Ticker nacheinander (ein Fehler crasht den Batch nicht)
python main.py --tickers NVDA,MSFT,RWE.DE

# Portfolio-Modus: mehrere Ticker als Depot-Ganzheit
# (Korrelations-Matrix, Overlap, Konzentrationswarnung, "Portfolio-Blick"-Sektion)
python main.py --portfolio RWE.DE,SHEL.L,NEE

# Track-Record-Evaluierung (Journal vs. tatsächliche Kursentwicklung,
# inkl. Konfidenz-Kalibrierung / Brier-Score / Reliability-Bänder)
python main.py --evaluate

# Watchlist analysieren (pflegt state/calibration.json automatisch vor)
python main.py --watchlist

# Tiefere Bull/Bear-Debatte über mehrere Runden (Ping-Pong, je 2 LLM-Calls pro Runde)
python main.py --ticker AAPL --debate-rounds 2

# Token-Usage-Report: aggregiert den LLM-Token-Verbrauch aus usage/usage.csv
python main.py --usage

# Ensemble-Trader deaktivieren / Anzahl Runs steuern
python main.py --ticker AAPL --no-ensemble
python main.py --ticker AAPL --ensemble-runs 5

# Crash-Resume: abgebrochenen Lauf an der letzten Stelle fortsetzen
python main.py --ticker AAPL --resume
```

Der Report wird auf stdout ausgegeben und zusätzlich als Datei unter
`reports/{ticker}_{YYYYMMDD}_{HHMM}.md` gespeichert (im Portfolio-Modus als
`reports/portfolio_{ticker}_{YYYYMMDD}_{HHMM}.md`, plus eine Portfolio-Zusammenfassung
auf stderr). Jeder Report beginnt mit einer **Management-Summary** (Gesamturteil,
Scores, Kernrisiken, Kurz-Begründung) gefolgt von den Detail-Abschnitten.

## Makro-Daten

Concilium bezieht kontextuelle Makro-Kennzahlen via yfinance (best effort, kein
zusätzlicher API-Key): **10Y US-Treasury-Yield** (aktueller Zins + Wert vor einem
Monat → Zinstrend), **EUR/USD**, **VIX**, **S&P-500-Trend** (1-Monats-Richtung)
sowie **Ölpreis (WTI)**. Diese Werte fließen in den Risiko-Off-Regime-Hinweis, den
Analysten-/Risk-Kontext und den Report ein und geben den Agenten einen
Markt-Breit-Kontext.

## Währungsrisiko (EUR-basiert)

Concilium ist auf einen EUR-basierten Fondsmanager ausgerichtet. USD-Ticker
(z. B. AAPL, MSFT) tragen ein Wechselkursrisiko: der Report zeigt einen
Währungsrisiko-Hinweis und den aktuellen EURUSD-Kurs, und der Portfolio-Fit
berücksichtigt das Währungsrisiko explizit in der empfohlenen
Ziel-Gewichtung, wenn der Ticker nicht in EUR notiert. Ein deterministischer
Währungsrisiko-Score (1–5, 5 = wenig Risiko) wird dem LLM als Anker
mitgegeben.

## Agenten-Architektur

Die Pipeline simuliert ein Team von Agenten, die nacheinander arbeiten — inspiriert
von der Rollenverteilung in einem Investmentfonds / Hedgefonds:

1. **Analysten-Team** — drei Rollen, die **parallel** laufen und jeweils einen
   **rollenspezifischen Datenkontext** erhalten (nur die für ihre Rolle
   relevanten Kennzahlen, kein Rauschen):
   - **Fundamental-Analyst**: Fundamentals (MarketCap, KGV, EPS, Revenue, Margen, Wachstum) + Analysten-Erwartungen + **quantitativer Multi-Faktor-Score** (deterministischer Value/Momentum/Qualität-Anker, den der LLM kritisch einordnet)
   - **Technik-Analyst**: Technische Indikatoren (SMA50/200, RSI14, MACD, Bollinger)
   - **Sentiment-Analyst**: News-Headlines (yfinance, Fallback auf Google-News-RSS, **ergänzt durch StockTwits + Reddit**), Positiv/Negativ/Neutral-Zählung (zeitgewichtet wenn Zeitstempel verfügbar), **mit Quellen-Kennzeichnung je Headline**

   Jeder Analyst liefert eine Stimmung (`bullish`/`neutral`/`bearish`) und einen
   Score (1-5). Ein **Konsistenz-Wächter** erkennt Stimmungs-/Score-Widersprüche
   (z. B. bearish + Score 4) und markiert sie als Warnung.

2. **Bull/Bear-Debatte** — zwei Rollen mit **differenzierten Schwerpunkten**:
   - **Bull**: fokussiert auf Stärken (Wachstum, Margen, Momentum, PEG, Sentiment)
   - **Bear**: fokussiert auf Risiken (Bewertung, Konzentration, Zinslast, Makro, technische Gegenanzeichen)

   Beide liefern eine **Konfidenz (1-5)**, die an den Trader durchgereicht wird
   (Nettoneigung Bull vs. Bear). Über `--debate-rounds N` (Default 1) läuft die
   Debatte als **Multi-Runden-Ping-Pong**: Ab Runde 2 bekommt jede Seite die
   Argumentation der Gegenseite aus der Vorrunde mit der Anweisung, konkret
   darauf einzugehen (zustimmen/widersprechen/ergänzen) statt zu wiederholen.
   Das vertieft die Analyse, kostet aber 2 zusätzliche LLM-Calls pro Runde.

3. **Trader** — schlägt eine konkrete Order vor, mit **5-stufiger Rating-Skala**:
   `STARK KAUFEN` / `KAUFEN` / `HALTEN` / `VERKAUFEN` / `STARK VERKAUFEN`
   (plus Zielkurs, Stop-Loss, Positionsanteil). Standardmäßig läuft der Trader
   als **Ensemble** (3 Runs mit variierender Temperatur, Mehrheitsabstimmung,
   Plausibilitäts-Check für Ziel-/Stop-Werte). Das 5-stufige Rating wird im
   Report und im Entscheidungs-Journal gespeichert; intern wird daraus für
   Ensemble-Abstimmung und Track-Record eine 3-stufige Aktion abgeleitet.

4. **Risk-Manager** — bewertet Volatilität, Drawdown-Risiko und Positionsgröße.
   Ein **rechnerisches Volatility-Targeting-Modell** (Risiko-Budget 2 %, Cap 10 %)
   wird dem LLM als deterministischer Anker in den Prompt gegeben, damit die
   LLM-Positionsgröße konsistent bleibt.

5. **Portfolio-Fit-Analyst** — bewertet die Aktie als Baustein im realen Depot
   (lädt Florians Depot aus einer Google-Sheet-Tabelle):
   - Konzentrationsrisiko (Ist die Aktie bereits stark gewichtet?)
   - Sektor-/Branchen-Overlap (Sektor bereits überrepräsentiert?) — inkl. **Sektor-/Regions-Summary** über alle Depot-Positionen
   - Empfohlene Ziel-Gewichtung

6. **Trade-Revision (2nd Pass)** — nach Risk- und Portfolio-Fit-Bewertung bekommt
   der Trader eine zweite Runde, um seinen Trade an die Einwände anzupassen
   (Aktion, Zielkurs, Stop-Loss, Positionsgröße). Der revidierte Trade geht an
   den Portfolio-Manager; der Original-Trade bleibt im Report sichtbar.

7. **Portfolio-Manager** — trifft die finale Entscheidung mit drei Optionen:
   `GENEHMIGT` / `MODIFIZIERT` (genehmigen mit Auflagen) / `ABGELEHNT`.
   Im Portfolio-Modus berücksichtigt er zusätzlich den **Gesamt-Exposure-Kontext**
   (Korrelationen und Overlap über alle analysierten Titel).

## Strukturierte LLM-Outputs

Seit Phase 0 (Fundament) liefern alle strukturierten Agent-Ergebnisse (trader,
risk, portfolio-manager, debate, analysten) **getyptes JSON** über OpenAI-kompatibles
`response_format` / `json_schema` (`schemas.py`). Kein fragiles Regex-Parsing mehr in
diesen Pfaden. Wenn der Provider kein `response_format` unterstützt (z. B. lokales
Ollama), fällt der Client automatisch auf das bisherige Text-Parsing zurück
(rückwärtskompatibel). Fehlende Schema-Felder werden mit sicheren Defaults
aufgefüllt, sodass der Report nie durch fehlende Keys oder `+nan%` bricht.

## Portfolio-Ebene (`--portfolio`)

Concilium kann mehrere Ticker **als Depot-Ganzheit** analysieren (nicht isoliert):

- **Korrelations-Matrix** der Tagesrenditen zwischen den analysierten Titeln.
- **Overlap** mit dem realen Depot (Sheet) + **Konzentrationswarnungen**
  (Einzelposition > ~5 %, Sektor-/Regions-Kumulation).
- Der Portfolio-Manager bekommt den **Gesamt-Exposure-Kontext** im Prompt.
- Report-Sektion **„Portfolio-Blick"** mit Ziel-Gewichtungen, Korrelations-Matrix
  (Paare mit |r| > 0.7 hervorgehoben) und Warnungen.

## Crash-Resume (Checkpoint)

Seit Phase 1 speichert Concilium nach **jedem Agent-Schritt** einen Checkpoint unter
`state/` (`.gitignore`). Bei Crash, Timeout oder 429 setzt `--resume` den Lauf an der
letzten abgeschlossenen Stelle fort — die bereits berechneten Agent-Schritte werden
nicht erneut ausgeführt. Erfolgreiche Läufe räumen ihren Checkpoint auf. Bei
`SIGINT` wird sauber beendet (Exit-Code 130) und der Checkpoint bleibt erhalten.

## Lernen aus dem Track-Record

Concilium ist explizit lernend über mehrere Mechanismen:

- **Kontext-Feedback**: Vor jeder Analyse liest Concilium das Entscheidungs-Journal
  (`journal/decisions.csv`) und injiziert neutrale Track-Record-Statistiken
  (Aktions-Verteilung, Ø Confidence, Portfolio-Fit, KAUFEN-Genehmigungsquote, **Kalibrierungs-Tendenz**) in die Trader-/Risk-/PM-Prompts, damit die Agenten ihre
  Kalibrierung an der eigenen Historie ausrichten.
- **Reflexion**: Vor jeder Analyse desselben Tickers holt Concilium den
  **realisierten Return** der letzten Entscheidung zu diesem Ticker (roh **und
  Alpha vs. SPY**), generiert eine kurze deutsche **Reflexion** und injiziert sie
  in den Trader- und Portfolio-Manager-Prompt.
- **Konfidenz-Kalibrierung** (seit Phase 4): `--evaluate` misst die Kalibrierung
  über den **Brier-Score**, den **Kalibrierungs-Gap** (Ø-Konfidenz vs. tatsächliche
  Hit-Rate) und **Reliability-Bänder** — und klassifiziert die Tendenz als über-/
  unterkonfident oder gut kalibriert. Diese Information fließt als Feedback in die
  Agenten zurück, damit sie gezielt gegen Fehlkalibrierung korrigieren.
- **Ensemble-Kalibrierung**: Das Trader-Ensemble ist **kalibrierungs-gewichtet**
  — Aktionen mit historisch schlechter Trefferquote (echte Hit-Rate) bekommen ein
  niedrigeres Stimmgewicht. Aggressive Ratings (STARK KAUFEN/STARK VERKAUFEN)
  werden automatisch zu KAUFEN/VERKAUFEN **gedämpft**, wenn die Historie
  überkonfident ist. `--evaluate` schreibt die Diagnose in
  `state/calibration.json`, die bei Folge-Läufen als Kontext dient.
- **Kalibrierungs-gestützte Dämpfung der Ziel-Gewichtung**: Die vom
  Portfolio-Fit-Analysten empfohlene Ziel-Gewichtung wird deterministisch an die
  **echte historische Trefferquote** der Aktion skaliert
  (`faktor = clamp(hit_rate, 0.3, 1.0)`), statt an die (oft überkonfidente)
  LLM-Konfidenz. Beispiel: KAUFEN mit 52 % Hit-Rate → Gewichtung ×0.52; HALTEN
  mit 14 % → ×0.3 (Untergrenze). Der Originalwert bleibt als
  `ziel_gewichtung_original` im Journal und Report sichtbar („nach Kalibrierung
  gedämpft, original X"). Damit wird die Positionsgröße an die reale
  Trefferquote gekoppelt — der fehlende Hebel gegen Überkonfidenz.

Die 5-stufige Skala macht zudem die `--evaluate`-Track-Record-Auswertung
granularer: zusätzlich zur Hit-Rate wird die **durchschnittliche Rating-Distanz**
(Anzahl Stufen, um die die Einschätzung neben dem tatsächlichen Verlauf lag)
ausgewiesen.

## LLM-Umgebungsvariablen

Die Agenten verwenden eine OpenAI-kompatible Schnittstelle, konfiguriert über Umgebungsvariablen:

| Variable | Default | Beschreibung |
|---|---|---|
| `LLM_BASE_URL` | `https://ollama.com/v1` | Base URL der OpenAI-kompatiblen API |
| `LLM_API_KEY` | aus `OLLAMA_API_KEY` | API-Key für Authentifizierung |
| `LLM_MODEL` | `glm-5.2` | Modellname (ohne `:cloud`-Suffix — das Suffix verursacht HTTP 400) |
| `LLM_FALLBACK_MODEL` | – | Fallback-Modell nach erschöpften Retries bei 429/5xx |
| `CONCILIUM_CACHE_DIR` | `<repo>/cache` | Tages-Cache für Marktdaten; leer = deaktiviert |
| `CONCILIUM_STATE_DIR` | `<repo>/state` | Checkpoint-Verzeichnis für `--resume`; leer = deaktiviert |

Beispiel:

```bash
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o"
python main.py --ticker AAPL
```

## Token-Usage-Logging

Concilium erfasst den **LLM-Token-Verbrauch** pro Analyse und protokolliert ihn in
`usage/usage.csv` (`.gitignore`). Der `LLMClient` akkumuliert das `usage`-Feld aus
jeder API-Antwort über alle Agenten-Calls einer Analyse; am Ende schreibt
`run_pipeline` den kumulierten Verbrauch (Prompt-/Completion-/Total-Tokens) pro
Ticker in die CSV.

Den aggregierten Verbrauch rufst du mit `--usage` ab:

```bash
python main.py --usage
```

Ausgabe z. B.:

```
=== Token-Usage-Report ===
Anzahl LLM-Calls:  12
Summe Prompt-Tokens:      21.400
Summe Completion-Tokens:  2.300
Summe Total-Tokens:      23.700
Eindeutige Ticker:        4

Token-Verbrauch pro Ticker:
  Ticker       Total-Tokens
  ------------ -------------
  AAPL                 6.100
  NVDA                 5.900
  ...
```

Das Logging ist **best effort** (crasht nie) und beeinflusst die Pipeline nicht.
Die CSV füllt sich ab dem ersten echten LLM-Lauf; `--usage` zeigt vorher
„Noch keine Usage-Daten erfasst."

## Hinweis zu externen Sentiment-Quellen

StockTwits und Reddit (Sozial-Sentiment, Phase 3) werden über öffentliche Endpoints
ohne API-Key bezogen. Beide können in Netzwerkumgebungen blockiert sein (HTTP 403,
z. B. Container-IPs); dann greift automatisch die Fallback-Kaskade (yfinance →
Google-News), sodass der Report nie leer bricht.

## Disclaimer

Dieses Projekt ist eine akademische Demo / Proof-of-Concept. Es ist **keine Anlageberatung**.
Die erzeugten Berichte basieren auf LLM-Textgenerierung und simplen Heuristiken. Sie sollten
nicht als Grundlage für tatsächliche Investmententscheidungen verwendet werden. Trading birgt
Risiken bis hin zum Totalverlust.
