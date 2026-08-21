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

# Mit Backtest-Signalproxy
python main.py --ticker MSFT --backtest
```

Der Report wird auf stdout ausgegeben und zusätzlich als Datei unter
`reports/{ticker}_{YYYYMMDD}_{HHMM}.md` gespeichert.

## Agenten-Architektur

Die Pipeline simuliert ein Team von Agenten, die nacheinander arbeiten —
inspiriert von der Rollenverteilung in einem Investmentfonds / Hedgefonds:

1. **Analysten-Team** — drei Rollen, die jeweils das gleiche Datenpaket unterschiedlich auswerten:
   - **Fundamental-Analyst**: Fundamentals (MarketCap, KGV, EPS, Revenue, Margen, Wachstum)
   - **Technik-Analyst**: Technische Indikatoren (SMA50/200, RSI14, MACD, Bollinger)
   - **Sentiment-Analyst**: News-Headlines aus ticker.news, Positiv/Negativ/Neutral-Zählung

   Jeder Analyst liefert eine Stimmung (`bullish`/`neutral`/`bearish`) und einen Score (1-5).

2. **Bull/Bear-Debatte** — zwei Rollen diskutieren die gesammelten Analysten-Einschätzungen:
   - **Bull**: Argumente für eine Long-Position
   - **Bear**: Gegenargumente und Risiken

3. **Trader** — schlägt eine konkrete Order vor (Kaufen/Halten/Verkaufen, Zielkurs, Stop-Loss)

4. **Risk-Manager** — bewertet Volatilität, Drawdown-Risiko und Positionsgröße

5. **Portfolio-Fit-Analyst** — bewertet die Aktie als Baustein im realen Depot:
   - Konzentrationsrisiko (Ist die Aktie bereits stark gewichtet?)
   - Sektor-/Branchen-Overlap (Sektor bereits überrepräsentiert?)
   - Empfohlene Ziel-Gewichtung

6. **Portfolio-Manager** — trifft die finale Genehmigungs-Entscheidung

## LLM-Umgebungsvariablen

Die Agenten verwenden eine OpenAI-kompatible Schnittstelle, konfiguriert über Umgebungsvariablen:

| Variable | Default | Beschreibung |
|---|---|---|
| `LLM_BASE_URL` | `https://ollama.com/v1` | Base URL der OpenAI-kompatiblen API |
| `LLM_API_KEY` | aus `OLLAMA_API_KEY` | API-Key für Authentifizierung |
| `LLM_MODEL` | `glm-5.2:cloud` | Modellname |

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