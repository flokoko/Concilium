# Concilium — Phase 3: Sozial-Sentiment erweitern (StockTwits + Reddit)

> **Handoff-Dokument für eine frische, kontextfreie Session.**
> Lies diese Datei vollständig und folge ihr. Zielbild aus `docs/ROADMAP.md` (Phase 3).

---

## Kontext (für die Session, die das umsetzt)

Concilium ist ein Multi-Agenten-Fonds-Entscheidungssystem (Python-CLI, Repo
`/opt/data/Concilium`, GitHub `flokoko/Concilium`). Der Sentiment-Analyst sammelt
aktuell Headlines aus yfinance (Primär) + Google-News-RSS (Fallback). Der
Sentiment-Abschnitt ist dünn — gerade bei kleineren Titeln liefert yfinance oft
0 Headlines.

**Ziel:** Sentiment sammelt Headlines aus bis zu 3 Quellen: yfinance /
Google-News-RSS / **StockTwits** (öffentlich ohne Key) und optional **Reddit**.
`news_source` erweitert sich; gewichtete Stimmungs-Zählung über alle Quellen.
Fallback-Kaskade bleibt (keine Quelle = `none`). Report zeigt Quelle je Headline.

**WICHTIG — keine neuen API-Keys nötig:**
- **StockTwits**: `https://api.stocktwits.com/api/2/streams/symbol/{SYMBOL}.json` —
  öffentlich, ohne Key. Liefert `messages[]` mit `body`, `created_at`, `entities`.
- **Reddit**: öffentliche JSON-Endpunkte (`https://www.reddit.com/r/{sub}/search.json?q={ticker}`)
  mit **eigenem `User-Agent`** (Reddit blockt Default-UAs und ist rate-limitiert).
  KEIN OAuth/Credentials. Kann geblockt/429 sein → sauberer Fallback.

**Konventionen (unbedingt einhalten):**
- Lauf: `/opt/data/depot-venv/bin/python` (NICHT System-Python).
- LLM-Key aus `OLLAMA_API_KEY` in `/opt/data/.env`.
- Ruff-Check: `/opt/data/depot-venv/bin/python -m ruff check src/ tests/ main.py`
- Tests: `/opt/data/depot-venv/bin/python -m pytest tests/ -q` (798 Tests, offline).
- Verifikationspflicht: ruff grün + pytest grün + `git push` + CI grün + Live-Test
  mit echtem Netz (kein LLM nötig für den Sentiment-Fetch — prüfe direkt, dass
  Headlines > 0 kommen).
- `main.py` hat `sys.path.insert`-Workaround → `# noqa: E402, I001` auf Import belassen.

---

## Derzeitige Architektur (relevant)

**`src/concilium/data.py`** — Datensammlung:
- `_fetch_google_news(ticker, company_name, limit)` → Google News RSS (Fallback), liste von dicts.
- `_classify_headline(headline)` → "positiv"/"negativ"/"neutral" (Keyword + Negations-Handling).
- `_count_sentiment(headlines)` und `_count_sentiment_weighted(headlines)` → Stimmungs-Zählung.
- `collect_ticker_data(...)` baut `result["news"]` (list[str]) und `result["news_with_dates"]`
  (list[dict] mit ISO-datetime) und `result["sentiment"]` (dict mit positiv/negativ/neutral,
  ggf. `weighted`, `dominant`, `sample_size`).
- Es gibt bereits Cache-Helper `_get_cache_dir()` / `_get_today_key()` (Tages-Cache),
  und einen User-Agent-Konstante `_USER_AGENT`.

**Report:** `report.py` zeigt die Sentiment-Sektion. Der Sentiment-Agent
(`agents.py::SYSTEM_SENTIMENT`) nutzt die Zählung.

---

## Umsetzungsauftrag

### Schritt 1 — `data.py`: `_fetch_stocktwits()` + `_fetch_reddit()`

Ergänze zwei neue Fetch-Funktionen im Stil von `_fetch_google_news`:

**`_fetch_stocktwits(ticker: str, limit: int = 10) -> list[dict]`**
- URL: `https://api.stocktwits.com/api/2/streams/symbol/{TICKER}.json`
- parse `data["messages"]`, extrahiere je Message:
  - `text` (der Body)
  - `created_at` (ISO-datetime, als `date`-Feld)
  - `source` (z.B. "web", "StockTwits")
- `_USER_AGENT` setzen. Timeout (z.B. 10s). **NIE crashen** — bei Fehler `[]`.
- Rate-Limit: nur EIN Call, `limit` beschränkt die Anzahl zurückgegebener Posts.
- Rückgabe: Liste von dicts `{text, date, source}` (gleiche Struktur wie news_with_dates).
- Ticker normalisieren: für Yahoo-Ticker mit Punkt (z.B. `RWE.DE`) → für StockTwits
  das Symbol evtl. ohne Suffix? StockTwits nutzt Cashtags wie `$RWE`. Verwende den
  rohen Ticker nach best effort; wenn der Call fehlschlägt → `[]`.

**`_fetch_reddit(ticker: str, limit: int = 5) -> list[dict]`**
- URL: `https://www.reddit.com/search.json?q={ticker}&sort=top&t=week&limit={limit}`
  (Such-Call, KEIN Auth). ODER via `r/stocks`/`r/wallstreetbets` search. Nutze den
  global search-Endpoint — robust.
- **Pflicht: eigener `User-Agent`** (z.B. `Concilium/1.0 (python-requests)`). Sonst
  blockt Reddit (403).
- parse `data["children"]`, je Post:
  - `title` + `selftext` (kombiniert als `text`)
  - `created_utc` (epoch → ISO `date`)
  - `source` = `"reddit"`
- **Rate-limit-respektvoll**: nur EIN Call, Timeout, bei 403/429 → `[]`.
- **NIE crashen.**

### Schritt 2 — Sentiment-Aggregation über Quellen

In `collect_ticker_data` (bzw. in der Funktion, die `news`/`news_with_dates`/
`sentiment` baut):
- Sammle Headlines aus: yfinance (Primär) + Google-News (Fallback, wie heute) +
  **StockTwits** + **Reddit**.
- Die Reihenfolge/Quellen-Priorität: yfinance zuerst, dann Google-News, dann
  StockTwits, dann Reddit. Wenn yfinance > 0 liefert, trotzdem die Zusatzquellen
  ergänzen (Dichte erhöhen), NICHT ersetzen.
- Für jedes News-Dict `{text, date, source}` das `source`-Feld setzen. Wenn du
  `result["news_with_dates"]` als Liste von dicts mit `date` führst, ergänze
  `source` je Headline.
- `result["news"]` (list[str]) weiterhin für Rückwärtskompatibilität befüllen —
  aber `news_with_dates` um `source` erweitern.
- `news_source`-Meta: Führe ein Feld ein, das die tatsächlich gelieferten Quellen
  angibt (z.B. `sentiment["sources"]` = ["yfinance","google","stocktwits","reddit"])
  und die Fallback-Kaskade dokumentiert. Bestehendes Verhalten (keine Quelle =
  neutral/leer) bleibt.
- `_count_sentiment_weighted` / `_count_sentiment`: erweitern, dass auch
  StockTwits/Reddit-Texte mitgezählt werden (der `_classify_headline`-Mechanismus
  ist quellenunabhängig — wende ihn auf die zusätzlichen Texte an).

### 3 — Report: Quelle je Headline zeigen

In `report.py` Sentiment-Sektion:
- Zeige je Headline die Quelle (z.B. `[StockTwits]`, `[Reddit]`, `[yfinance]`,
  `[Google]`). Wenn keine Quellen-Infos vorhanden (alte Daten), kein Quell-Suffix
  (N/A — robust).

### 4 — Tests

- **`tests/test_social_sentiment.py`** (neu):
  - `_fetch_stocktwits` und `_fetch_reddit` mit **gemocktem requests** (offline) —
    konstruierte JSON-Antworten → korrekt geparst, Fehler → `[]`.
  - Aggregation: `collect_ticker_data`-Sentiment zählt StockTwits/Reddit-Text mit;
    `news_with_dates` enthält `source`.
  - Fallback: bei StockTwits/Reddit-Fehler (mock wirft) → kein Crash, yfinance
    allein bleibt.
  - Report: Quelle je Headline erscheint.
- Bestehende 798 Tests grün halten. Nutze MagicMock/patch für requests.

---

## Verifikations-Checkliste (REIHENFOLGE ERZIELEN)
1. `/opt/data/depot-venv/bin/python -m ruff check src/ tests/ main.py` → 0 Fehler.
2. `/opt/data/depot-venv/bin/python -m pytest tests/ -q` → alle grün (798 + neue).
3. `git add -A && git commit` — aussagekräftige Message.
4. `git push origin main`.
5. GitHub Actions CI abwarten (API). CI grün.
6. **Live-Test** (kein LLM nötig — Sentiment-Fetch direkt prüfen):
   ```
   cd /opt/data/Concilium
   /opt/data/depot-venv/bin/python -c "from concilium.data import _fetch_stocktwits,_fetch_reddit; print('ST', len(_fetch_stocktwits('NVDA'))); print('RD', len(_fetch_reddit('NVDA')))"
   ```
   UND ein voller Pipeline-Lauf (ohne LLM reicht für Sentiment, aber mach einen
   echten mit LLM wenn möglich):
   ```
   export LLM_API_KEY=$(grep -oP '^OLLAMA_API_KEY=\K.*' /opt/data/.env | tr -d '"' | tr -d "'" | head -1)
   export LLM_BASE_URL="https://ollama.com/v1"
   export LLM_MODEL="glm-5.2:cloud"
   /opt/data/depot-venv/bin/python main.py --ticker NVDA
   ```
   Report prüfen: Sentiment-Sektion hat Headlines aus StockTwits/Reddit (Quelle je
   Headline), kein `+nan%`/`fehler`. WICHTIG: Wenn Reddit in der Live-Umgebung
   geblockt wird (403/429), ist das OK — die Fallback-Kaskade sorgt, dass
   StockTwits/Google/yfinance trotzdem kommen. Nicht als Fehler werten.
7. Abschluss-Meldung an Flo: welche Quellen aktiv, welche im Fallback, Verifikation.

---

## Fallstricke / Bewertung
- **Keine neuen Deps**: nur requests/urllib (vorhanden).
- **Rate-Limits**: je Quelle genau EIN Call, Timeout, nie crashen, bei Fehler `[]`.
- **Fallback-Kaskade**: yfinance Primär; wenn alle Zusatzquellen fehlschlagen →
  bisheriges Verhalten (yfinance/Google) unverändert.
- **Rückwärtskompatibilität**: `result["news"]` (list[str]) bleibt; `news_with_dates`
  erweitert um `source` (optional, robust bei fehlendem Feld).
- **Source-Kennzeichnung**: je Headline.
- **CI**: ruff + pytest offline grün.

---

## Done-Definition
- `_fetch_stocktwits()` + `_fetch_reddit()` (öffentlich, ohne Key, rate-limit-respektvoll, nie crashen).
- Sentiment-Aggregation zählt alle Quellen; `news_with_dates` + `news_source`.
- Report zeigt Quelle je Headline.
- Fallback bleibt (kein Crash wenn Quellen fehlen).
- 798+ Tests grün, ruff grün, CI grün, Live-Test: StockTwits/Reddit-Headlines >0
  (wenn Netz es erlaubt), kein `+nan%`/`fehler`.

Wenn du das umgesetzt hast: gib die Zusammenfassung an Flo zurück mit den Verifikations-Ergebnissen.
