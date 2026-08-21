# Concilium — Phase 0: Strukturierte LLM-Outputs

> **Handoff-Dokument für eine frische, kontextfreie Session.**
> Lies diese Datei vollständig und folge ihr. Zielbild aus `docs/ROADMAP.md` (Phase 0).

---

## Kontext (für die Session, die das umsetzt)

Concilium ist ein Multi-Agenten-Fonds-Entscheidungssystem (Python-CLI, Repo
`/opt/data/Concilium`, GitHub `flokoko/Concilium`). Es emuliert eine
Handelsentscheidung über spezialisierte LLM-Rollen-Agenten, die nacheinander
arbeiten. Die Agenten rufen ein OpenAI-kompatibles `/chat/completions`-Backend
(Standard: GLM-5.2 via Ollama-Cloud).

**Das Problem dieser Phase:** Mehrere Agenten-Ausgaben (Bull/Bear-Debatte, Trader,
Risk-Manager, Portfolio-Manager) werden heute aus **freiem LLM-Fließtext per
Regex/`parse_json`** extrahiert. Jede Formulierung, die vom erwarteten Format
abweicht, erzeugt `None`, `+nan%` oder `fehler`-Einträge im Report. Bekannte
Symptome: `_parse_debate_confidence`, `_extract_current_price`, `+nan%`-Bug in
realisierten Returns.

**Ziel:** strukturierte, getypte LLM-Outputs (OpenAI-kompatibles `response_format`
mit `json_schema`) für die strukturierten Agent-Ergebnisse, mit automatischem
Fallback auf das bisherige Parsing, falls der Provider kein `response_format`
unterstützt (z.B. lokales Ollama).

**Konventionen (unbedingt einhalten):**
- Lauf: `/opt/data/depot-venv/bin/python` (NICHT System-Python).
- LLM-Key aus `OLLAMA_API_KEY` in `/opt/data/.env`.
- Ruff-Check: `/opt/data/depot-venv/bin/python -m ruff check src/ tests/ main.py`
- Tests: `/opt/data/depot-venv/bin/python -m pytest tests/ -q` (326 Tests, offline).
- Verifikationspflicht: ruff grün + pytest grün + `git push` + CI grün + Live-Test
  mit echtem LLM (Key aus .env) für 2-3 Ticker.
- Änderungen an `src/` kommen NICHT in die .gitignore. `main.py` hat
  `sys.path.insert`-Workaround → `# noqa: E402, I001` auf dem Import belassen.

---

## Derzeitige Architektur (relevant)

**`src/concilium/llm.py`** — `LLMClient.chat(messages, temperature, max_tokens) -> str`
- Baut Payload `{model, messages, temperature, max_tokens?}`.
- Retry bei 429/5xx mit Backoff-Jitter (`_send_with_retries`), Fallback-Modell.
- Liefert **Text** zurück (`choices[0].message.content`).

**`src/concilium/agents.py`** — die Agenten-Funktionen, die strukturierte Ergebnisse
liefern und aktuell Fließtext parsen:
- `_parse_debate_confidence(agent)` — Regex/parse_json aus `_raw` (Zeile ~582).
- `_extract_current_price(analysts)` — current price aus technicals (Zeile ~753).
- `trader()`, `risk_manager()`, `portfolio_manager()`, `debate()`,
  `analyst_team()` — rufen `llm.chat()` und parsen danach.

**`src/concilium/pipeline.py`** — `run_pipeline(...)`: sequenzielle Verdrahtung der
Agenten, übergibt `data_text`, `feedback_context`, `reflection_context`.

---

## Umsetzungsauftrag

### Schritt 1 — `llm.py`: `response_format`-Support

Erweitere `LLMClient.chat()` um einen neuen optionalen Parameter
`response_format: dict | None = None`. Beim Setzen wird er in den Payload als
`"response_format": response_format` aufgenommen (OpenAI-kompatible API).

WICHTIG: Wenn `response_format` gesetzt ist und die API-Fehler meldet (400 wegen
nicht unterstütztem `response_format`, o.ä.), soll der Caller **automatisch** auf
einen text-basierten Fallback ausweichen können. Implementiere das am saubersten so:

- Füge in `chat()` eine Rückgabe-Variante hinzu: statt `str` ein
  `ChatResult`-NamedTuple `{text, structured: bool, response_format_used: bool}`.
  ABER: Das bricht alle bestehenden Aufrufer. → Rückwärtskompatibel halten:
  - Default-Verhalten unverändert: `chat()` gibt weiterhin `str` zurück.
  - Neuer optionaler Parameter `as_structured=False`. Ist `as_structured=True`
    UND `response_format` gesetzt, gibt `chat()` ein NamedTuple
    `StructuredChat(text, response_format_used)` zurück. Ist `as_structured=False`
    (Default) → `str`.
  - Bei `json_schema`-Modus, wo die API das Feld nicht unterstützt (4xx/400),
    soll im `_send_with_retries`-Pfad erkannt werden und der Aufruf **ohne**
    `response_format` wiederholt werden (einmalig), `response_format_used=False`
    setzen. Kein Endlos-Loop.

Nutze die vorhandene Fehlerklassen (`_RetryableHTTPError`, `RuntimeError`) passend.

---

### Schritt 2: JSON-Schema-Definitionen

Lege die Schemas für die strukturierten Agent-Ergebnisse an — entweder als
Konstanten in `agents.py` oder in einer neuen `src/concilium/schemas.py`. Diese
Agent-Ergebnisse sollen strukturiert werden (Reihenfolge nach Impact):

1. **Trader-Ergebnis** (`trade`):
   - `rating`: "STARK KAUFEN"|"KAUFEN"|"HALTEN"|"VERKAUFEN"|"STARK VERKAUFEN"
   - `action`: "KAUFEN"|"HALTEN"|"VERKAUFEN" (abgeleitet)
   - `target_price` (float|null), `stop_loss` (float|null), `position_pct` (float|null),
   - `begründung` (str), `confidence` (int 1-5), `zeit_horizont` (str|null)
2. **Risk-Manager-Ergebnis** (`risk`):
   - `risiko_score` (int 1-5), `drawdown_risiko` (str), `positionsgröße_pct` (float|null),
   - `auflagen` (list[str]), `begründung` (str), `confidence` (int 1-5)
3. **Portfolio-Manager-Ergebnis** (`final`):
   - `entscheidung` ("GENEHMIGT"|"MODIFIZIERT"|"ABGELEHNT"),
   - `begründung` (str), `auflagen` (list[str]), `confidence` (int 1-5)
4. **Bull/Bear-Debatte** (`bull`/`bear`):
   - `argumente` (str), `confidence` (int 1-5), `stimmung` ("bullish"|"bearish")
5. **Analyst-Ergebnis** (`fundamentals`/`technicals`/`sentiment`):
   - `stimmung` ("bullish"|"neutral"|"bearish"), `score` (int 1-5),
   - `konsistenz_warnung` (str|null), Felder nach Bedarf.

JSON-Schemas mit `additionalProperties: false`, `required`-Felder, `enum`s.

---

### Schritt 3: Agenten auf strukturierte Ausgabe umstellen

Für `trader`, `risk_manager`, `portfolio_manager`, `debate`, `analyst_team`:

- Baue den `response_format`-Dict gemäß `json_schema` (OpenAI-Syntax:
  `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}`,
  bzw. so wie es dein Backend versteht — GLM-5.2 via Ollama-Cloud unterstützt
  `response_format` laut Vorbild; prüfe beim Live-Test).
- Rufe `llm.chat(..., as_structured=True, response_format=...)` auf.
- Wenn `response_format_used=True`: `json.loads(text)` direkt als Ergebnis-dict
  verwenden (ohne Regex-Nachbearbeitung). Scheineleid / Type-Härtung anwenden
  (defaults setzen bei fehlenden Feldern).
- Wenn `response_format_used=False` (Fallback): bisheriges Parsing nutzen.
- Struktur-Garantie: Die Agenten-Funktionen sollen danach **immer** ein dict mit
  den Schlüsseln des Schemas zurückgeben — gefüllt oder mit sicheren Defaults
  (nie `None`/fehlende Keys, die den Report brechen).

---

### Schritt 4: Konsistenz-Wächter & Downstream

- `_parse_debate_confidence` / `_extract_current_price`: Diese dürfen im
  strukturierten Pfad ersetzt werden durch direkten Zugriff auf das
  Agent-dict (`trade["confidence"]`, `analysts["technicals"]["current_price"]`).
  Im Fallback-Pfad bleiben die Parser als Fallback erhalten.
- `journal.py` `_parse_confidence_from_debate`: an strukturiertes Verhalten anpassen.
- `portfolio_fit.py`: unberührt (kein LLM-JSON-Parsing dort — Sheet-basiert).

---

### Schritt 5: Tests

- **Neue Schema-Validierungs-Tests**: für jeden strukturierten Agent, dass das
  Ergebnis dem Schema entspricht (Pydantic via `jsonschema` lib oder einfach
  `jsonschema.validate`). `jsonschema` ggf. als dev-Extra in `pyproject.toml`.
- **Fallback-Pfad-Tests**: LLM-Antwort ohne nutzbares `response_format` (Simulieren,
  dass `_send_with_retries` bei 400 feuert → zweiter Versuch ohne `response_format`)
  → Verhalten identisch zu heute.
- **Mock-LLM**: Tests in `tests/test_ensemble.py` nutzen temperatur-keyed `_FakeLLM`.
  Erweitern, dass er `response_format`-unterstützende Fake-Antworten liefert.

---

### Schritt 6: Report & CLI

- `report.py`-Renderer sollen mit den strukturierten dicts arbeiten (die jetzt
  dieselben Schlüssel haben wie vorher — dadurch minimale Änderung).
- Der Report soll nirgends mehr `+nan%` / `fehler` aus strukturierten Ergebnissen
  zeigen. Die `math.isfinite()`-Guards aus dem Pitfall bleiben bestehen.

---

## Verifikations-Checkliste (REIHENFOLGE ERZIELEN)

1. `/opt/data/depot-venv/bin/python -m ruff check src/ tests/ main.py` → 0 Fehler.
2. `/opt/data/depot-venv/bin/python -m pytest tests/ -q` → alle grün (326+).
3. `git add -A && git commit` — aussagekräftige Message.
4. `git push origin main`.
5. GitHub Actions CI abwarten (API: `gh run list` — Status abfragen; Fine-Grained
   PAT liest Runs/Logs). CI muss grün sein.
6. **Live-Test** mit echtem LLM:
   ```
   cd /opt/data/Concilium
   export LLM_API_KEY=$(grep -oP '^OLLAMA_API_KEY=\K.*' /opt/data/.env | tr -d '"' | tr -d "'" | head -1)
   export LLM_BASE_URL="https://ollama.com/v1"
   export LLM_MODEL="glm-5.2:cloud"
   /opt/data/depot-venv/bin/python main.py --ticker AAPL
   /opt/data/depot-venv/bin/python main.py --ticker RWE.DE --peers SHEL.L
   ```
   Report prüfen: keine `+nan%`, kein `fehler`-Eintrag bei Trader/Risk/PM/Debatte,
   `current_price` korrekt, Ensemble liefert Rating-Verteilung.
7. In der Abschluss-Meldung an Flo: was umgesetzt wurde, welche Agenten
   strukturiert sind, was im Fallback läuft, Live-Test-Ergebnis.

---

## Fallstricke / Bewertung

- **Retry-Verhalten**: `_send_with_retries` feuert bei 429/5xx. Bei 400 (invalid
  `response_format`) DARF NICHT endlos retried werden — einmalig ohne
  `response_format` wiederholen, dann fertig. Kein Endlos-Loop.
- **Thread-Sicherheit**: `analyst_team`/`ensemble_trader` laufen parallel
  (`ThreadPoolExecutor`, max_workers=3). `LLMClient.chat()` ist thread-sicher
  (nur lokale Variablen). Achte darauf, dass neue Parameter/Klassen keine
  shared mutable state bekommen.
- **Rückwärtskompatibilität**: Default `chat()` gibt weiterhin `str` zurück.
  Alle bestehenden Tests müssen ohne Änderung der Aufrufer funktionieren.
- **Ensemble**: Ensemble-Abstimmung nutzt `_rating_to_action()` auf 3-stufig —
  bleib unverändert; nur das Roh-`rating` kommt jetzt garantiert sauber.

---

## Done-Definition

- `response_format`-Support in `LLMClient` mit Fallback.
- Trader / Risk / PM / Debatte / Analysten liefern strukturierte dicts gemäß Schema.
- Kein regex-Parsing mehr im strukturierten Pfad dieser Agenten.
- 326+ Tests grün, ruff grün, CI grün, Live-Test ohne `+nan%`/`fehler`.
- Push & Merge abgeschlossen.

Wenn du (die Session) das umgesetzt hast: gib die Zusammenfassung an Flo zurück mit
den Verifikations-Ergebnissen.
