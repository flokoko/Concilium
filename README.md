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

# Ensemble-Trader deaktivieren / Anzahl Runs steuern
python main.py --ticker AAPL --no-ensemble
python main.py --ticker AAPL --ensemble-runs 5

# Track-Record-Evaluierung (Journal vs. tatsächliche Kursentwicklung)
python main.py --evaluate
```

Der Report wird auf stdout ausgegeben und zusätzlich als Datei unter
`reports/{ticker}_{YYYYMMDD}_{HHMM}.md` gespeichert. Jeder Report beginnt mit
einer **Management-Summary** (Gesamturteil, Scores, Kernrisiken, Kurz-Begründung)
gefolgt von den Detail-Abschnitten.

## Agenten-Architektur

Die Pipeline simuliert ein Team von Agenten, die nacheinander arbeiten —
inspiriert von der Rollenverteilung in einem Investmentfonds / Hedgefonds:

1. **Analysten-Team** — drei Rollen, die **parallel** laufen und jeweils einen
   **rollenspezifischen Datenkontext** erhalten (nur die für ihre Rolle
   relevanten Kennzahlen, kein Rauschen):
   - **Fundamental-Analyst**: Fundamentals (MarketCap, KGV, EPS, Revenue, Margen, Wachstum) + Analysten-Erwartungen + **quantitativer Multi-Faktor-Score** (deterministischer Value/Momentum/Qualität-Anker, den der LLM kritisch einordnet)
   - **Technik-Analyst**: Technische Indikatoren (SMA50/200, RSI14, MACD, Bollinger)
   - **Sentiment-Analyst**: News-Headlines (yfinance, Fallback auf Google-News-RSS), Positiv/Negativ/Neutral-Zählung (zeitgewichtet wenn Zeitstempel verfügbar)

   Jeder Analyst liefert eine Stimmung (`bullish`/`neutral`/`bearish`) und einen
   Score (1-5). Ein **Konsistenz-Wächter** erkennt Stimmungs-/Score-Widersprüche
   (z. B. bearish + Score 4) und markiert sie als Warnung.

2. **Bull/Bear-Debatte** — zwei Rollen mit **differenzierten Schwerpunkten**:
   - **Bull**: fokussiert auf Stärken (Wachstum, Margen, Momentum, PEG, Sentiment)
   - **Bear**: fokussiert auf Risiken (Bewertung, Konzentration, Zinslast, Makro, technische Gegenanzeichen)

   Beide liefern eine **Konfidenz (1-5)**, die an den Trader durchgereicht wird
   (Nettoneigung Bull vs. Bear).

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

## Lernen aus dem Track-Record

Concilium ist explizit lernend über zwei Mechanismen:

- **Kontext-Feedback**: Vor jeder Analyse liest Concilium das Entscheidungs-Journal
  (`journal/decisions.csv`) und injiziert neutrale Track-Record-Statistiken
  (Aktions-Verteilung, Ø Confidence, Portfolio-Fit, KAUFEN-Genehmigungsquote) in
  die Trader-/Risk-/PM-Prompts, damit die Agenten ihre Kalibrierung an der
  eigenen Historie ausrichten.
- **Reflexion**: Vor jeder Analyse desselben Tickers holt Concilium den
  **realisierten Return** der letzten Entscheidung zu diesem Ticker (roh **und
  Alpha vs. SPY**), generiert eine kurze deutsche **Reflexion** und injiziert sie
  in den Trader- und Portfolio-Manager-Prompt.

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
| `LLM_MODEL` | `glm-5.2:cloud` | Modellname |
| `LLM_FALLBACK_MODEL` | – | Fallback-Modell nach erschöpften Retries bei 429/5xx |
| `CONCILIUM_CACHE_DIR` | `<repo>/cache` | Tages-Cache für Marktdaten; leer = deaktiviert |

Beispiel:

```bash
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o"
python main.py --ticker AAPL
```

## Disclaimer

Dieses Projekt ist eine akademische Demo / Proof-of-Concept. Es ist **keine Anlageberatung**.
Die erzeugten Berichte basieren auf LLM-Textgenerierung und simplen Heuristiken. Sie sollten
nicht als Grundlage für tatsächliche Investmententscheidungen verwendet werden. Trading birgt
Risiken bis hin zum Totalverlust.
