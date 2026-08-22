# Concilium — Phase 1: Crash-Resilienz / Checkpoint-Resume

> **Handoff-Dokument für eine frische, kontextfreie Session.**
> Lies diese Datei vollständig und folge ihr. Zielbild aus `docs/ROADMAP.md` (Phase 1).

---

## Kontext (für die Session, die das umsetzt)

Concilium ist ein Multi-Agenten-Fonds-Entscheidungssystem (Python-CLI, Repo
`/opt/data/Concilium`, GitHub `flokoko/Concilium`). Die Pipeline
(`src/concilium/pipeline.py::run_pipeline`) führt mehrere LLM-Agenten
nacheinander aus (siehe Schritte unten). Jeder Agent-Schritt ist ein LLM-Call
über ein OpenAI-kompatibles `/chat/completions`-Backend (GLM-5.2 via Ollama-Cloud).

**Das Problem dieser Phase:** Die Pipeline ist lang (bis zu 8+ LLM-Calls pro
Ticker inkl. Ensemble-Trader mit 3 Runs). Bei einem Crash, Timeout oder 429
startet sie **immer von vorn** — die bereits bezahlten und berechneten
Agenten-Schritte sind verloren. Für deterministische, lange Läufe ist das teuer
und frustrierend. Das Original (TradingAgents) setzt via LangGraph-Checkpoint
abgebrochene Läufe an der Stelle fort. Concilium braucht kein LangGraph — ein
eigenes, schlankes State-Persistenz-Modul reicht.

**Ziel:** Pipeline-Zwischenstände werden pro Ticker in ein JSON unter `state/`
geschrieben. Ein neues CLI-Flag `--resume` setzt einen abgebrochenen Lauf an der
letzten abgeschlossenen Stelle fort — nur die fehlenden Schritte werden neu
ausgeführt. Ohne Flag bleibt das Verhalten unverändert. Abgebrochene/fehlerhafte
Runs hinterlassen einen eindeutig markierten Zustand. Ein erfolgreicher Lauf
räumt seinen eigenen Checkpoint auf.

**Konventionen (unbedingt einhalten):**
- Lauf: `/opt/data/depot-venv/bin/python` (NICHT System-Python).
- LLM-Key aus `OLLAMA_API_KEY` in `/opt/data/.env`.
- Ruff-Check: `/opt/data/depot-venv/bin/python -m ruff check src/ tests/ main.py`
- Tests: `/opt/data/depot-venv/bin/python -m pytest tests/ -q` (663 Tests, offline).
- Verifikationspflicht: ruff grün + pytest grün + `git push` + CI grün + Live-Test
  mit echtem LLM (Key aus .env) für 1-2 Ticker, inkl. Resume-Verhalten.
- `main.py` hat `sys.path.insert`-Workaround → `# noqa: E402, I001` auf Import belassen.
- `state/` kommt in `.gitignore` (nicht versioniert).

---

## Derzeitige Architektur (relevant)

**`src/concilium/pipeline.py` — `run_pipeline(ticker, llm, backtest, peers, ensemble, ensemble_runs) -> dict`**
Sequenzielle Verdrahtung der Agenten, schreibt nach jedem Schritt ins `result`-dict:

1. `data = collect_ticker_data(ticker, peers=peers)`; `result["data"]`, `result["ticker"]`
   1b. `data_text = _build_data_text(data)` (nur LLM-Modus)
   1c. `feedback_context = build_feedback_context()` (nur LLM-Modus)
   1d. `reflection_context = build_reflection_context(...)` (nur LLM-Modus)
   (Optional) `result["backtest"] = run_backtest(data)`
   Wenn `llm is None`: `result["no_llm"] = True`, sofort return.
2. `analysts = analyst_team(data, llm, data_text=data_text)`; `result["analysts"]`
3. `debate_result = debate(analysts, llm)`; `result["debate"]`
4. `trade` via `ensemble_trader(...)` oder `trader(...)`; `result["trade"]`
5. `risk = risk_manager(...)`; `result["risk"]`
   5b. `portfolio_fit` via `portfolio_fit_agent(...)`; `result["portfolio_fit"]`
   5c. `trade_revision(...)`; `result["trade_original"]`, `result["trade"]` (revidiert), `result["trade_revised"]`
6. `final = portfolio_manager(...)`; `result["final"]`
   (Journal) `append_decision(result)`; `result["_journal_written"]`

**`src/concilium/cli.py` — `main(argv)`**
argparse mit `--ticker`, `--tickers` (Batch), `--evaluate`, `--backtest`,
`--no-llm`, `--peers`, `--verbose`, `--no-ensemble`, `--ensemble-runs`.
Batch-Schleife ruft `run_pipeline` + `generate_report` pro Ticker.
`--evaluate` ist eigenständig und berührt die Pipeline NICHT.

**`src/concilium/journal.py` — Referenz für fcntl-Locking**
- `_acquire_lock(fh)` / `_release_lock(fh)`: `fcntl.flock(LOCK_EX/LOCK_UN)`, best-effort,
  crasht nie (fcntl optional auf Nicht-Linux).
- Pattern: `with open(file, ...) as fh: _acquire_lock(fh); try: ... finally: _release_lock(fh)`.

---

## Umsetzungsauftrag

### Schritt 1 — `state/` in `.gitignore`

`.gitignore` hat bereits `reports/`, `journal/`, `cache/`. Füge `state/` hinzu.
(Checkpoint-Dateien sollen nicht versioniert werden.)

### Schritt 2 — `src/concilium/checkpoint.py` (neu)

Ein schlankes, robustes State-Persistenz-Modul analog zum Journal-Lock-Pattern:

- **Pfad-Auflösung**: Checkpoints landen unter `state/` relativ zum Arbeitsverzeichnis.
  Env-Var `CONCILIUM_STATE_DIR` (optional) übersteuert das Basisverzeichnis — wichtig
  für Tests, damit Tests NICHT in das echte `state/` schreiben.
- **Datei-Namens-Schema**: `state/<ticker>_checkpoint.json` (Ticker normalisiert:
  `.` → `_`, z.B. `RWE.DE` → `RWE_DE`), oder besser mit einem Hash/Zeitstempel wenn
  mehrere Runs pro Ticker möglich sind. Empfehlung: Der Checkpoint wird pro Ticker
  (nicht pro Run) geführt, aber von einem erfolgreichen Lauf aufgeräumt. Du darfst
  ein feld `run_started_at`/`version` in der Datei führen, um veraltete Checkpoints
  zu erkennen.
- **API** (mindestens):
  - `save_checkpoint(result: dict, ticker: str, *, state_dir=None)` — schreibt den
    aktuellen Pipeline-Zwischenstand als JSON atomar + unter fcntl-Lock (analog journal).
  - `load_checkpoint(ticker: str, *, state_dir=None) -> dict | None` — liest den
    Checkpoint; gibt `None` zurück, wenn keiner existiert oder das JSON kaputt ist.
  - `clear_checkpoint(ticker: str, *, state_dir=None)` — entfernt den Checkpoint.
- **Was in den Checkpoint**: die **Teilergebnisse** `result` (alle schon vorhandenen
  Keys außer den großen, ggf. kostbaren Daten). Du musst entscheiden, ob `result["data"]`
  (yfinance-Datensnapshot, groß) mitgespeichert wird. Empfehlung: Speichere die
  Agenten-Zwischenstände (analysts, debate, trade, risk, portfolio_fit, trade_original,
  trade_revised, final) und die Kostenfelder (data_text, feedback_context, reflection)
  — aber NICHT unbedingt den rohen `data`-Snapshot (der kann bei Resume neu geholt
  werden und ist nicht das Problem). WICHTIG: Das Handoff soll deterministisch sein —
  wenn beim Resume die Daten neu geholt werden müssen, dann achte darauf, dass
  data_text/feedback/reflection konsistent neu berechnet werden (siehe Pipeline-Fix).
  Einfachster und sicherster Ansatz: **Speichere den vollständigen `result` (inkl.
  `data`) als JSON**. Falls `data` nicht JSON-serialisierbar ist (numpy-Typen etc.),
  verwende einen tolerant JSON-Serializer (`default=str`). Das ist am robustesten und
  vermeidet Inkonsistenzen beim Resume.
- **Atomar schreiben**: in eine Temp-Datei schreiben, dann `os.replace` (atomic).
  Unter fcntl-Lock, best-effort, nie crashen (try/except + logging.warning).
- **Kaputte/teilgeschriebene Datei**: beim `load_checkpoint` → `json.JSONDecodeError`
  abfangen → `None` + warn-Log. (Der Lauf startet dann von vorn.)

### Schritt 3 — `pipeline.py`: Zwischenstände nach jedem Agent-Schritt schreiben

- Am Ende von `run_pipeline`, NACH jedem Agent-Schritt (nach Schritt 1 bis nach
  Schritt 7), wird der aktuelle `result`-Stand via `checkpoint.save_checkpoint(...)`
  geschrieben. Einfachster Ort: am Ende der Funktion (nach `result["_journal_written"]`).
  ABER das würde nur am Ende schreiben — der Sinn ist, dass auch bei Crash
  mittendrin der Stand von VORHER da ist. Also: speichere nach JEDEM Schritt
  (2,3,4,5,5b,5c,6). Du darfst nach jedem Schritt einen `logger.info("Checkpoint
  gespeichert (Schritt X)")`-Log machen.
- **Resume-Verhalten**: `run_pipeline` bekommt einen neuen Parameter
  `resume: bool = False`. Wenn `resume=True` und ein Checkpoint für den Ticker
  existiert → lade den Checkpoint als `result`, bestimme die **letzte
  abgeschlossene Schrittnummer** (du musst markieren, welcher Schritt zuletzt
  fertig war — z.B. ein spezieller Key `_checkpoint_step` in `result`, den du
  beim Speichern setzt), und führe nur die **fehlenden** Schritte ab diesem Punkt
  neu aus. Alle bereits im Checkpoint vorhandenen Teilergebnisse werden NICHT neu
  berechnet.
- Ohne `resume=True` → Verhalten unverändert (von vorn), egal ob Checkpoint existiert.
- Nach einem **erfolgreichen** Lauf (alle Schritte inkl. journal) → `clear_checkpoint`
  aufrufen, damit der eigene Checkpoint aufgeräumt wird.
- **`_checkpoint_step`-Buchhaltung**: Der Checkpoint soll wissen, welche Schritte
  schon fertig sind. Einfachstes robustes Modell: ein Set/Liste `result["_completed_steps"]`
  (z.B. `["data","analysts","debate","trade","risk","portfolio_fit","trade_revision","final"]`).
  Beim Resume wird nur der erste noch fehlende Schritt neu ausgeführt; alles davor
  wird aus dem Checkpoint übernommen.
- **Thread-Sicherheit**: pipeline ist im LLM-Modus ggf. aus Batch-Schleife sequentiell
  pro Ticker — kein paralleler Pipeline-Lauf desselben Tickers. Du musst keine
  Lock-Verfeinerung über den Datei-Lock hinaus machen.

### Schritt 4 — `cli.py`: `--resume`/`--no-resume` + sauberes Interrupt-Verhalten

- Neues Flag `--resume` (action="store_true") und `--no-resume`
  (action="store_true", dient als explizites Ausschalten, Default ist NICHT-resume).
  Beide schließen sich gegenseitig aus (parser.error wenn beide gesetzt).
- Einzelmodus: `run_pipeline(..., resume=args.resume)`.
- Batch-Modus: `run_pipeline(..., resume=args.resume)` pro Ticker. (Optional: bei
  Resume das Datei-Zeitstempel im Report-Filename anpassen, aber nicht zwingend.)
- **Sauberes Interrupt-Verhalten**: Um `SIGINT`/`KeyboardInterrupt` herum so
  verhalten, dass der bis dahin geschriebene Checkpoint erhalten bleibt und der
  Prozess mit einem klaren Exit-Code 130 beendet wird (Standard für SIGINT).
  Der Checkpoint wird bereits während der Pipeline (per Schritt-Savepoint)
  geschrieben — also muss der Interrupt den Checkpoint NICHT extra schreiben,
  sondern nur sauber durchreichen und den Exit-Code setzen. Fange in cli.py
  um `run_pipeline` herum `KeyboardInterrupt` ab: logge "Abgebrochen —
  Checkpoint bleibt unter state/... erhalten", return 130. WICHTIG: nicht
  `traceback.print_exc()` bei KeyboardInterrupt — sauber beenden.
- `--evaluate` bleibt UNBERÜHRT (kein resume, kein checkpoint).
- Der `try/except Exception` in cli.py fängt derzeit alle Exception ab — achte
  darauf, dass `KeyboardInterrupt` (BaseException, nicht Exception) NICHT von
  `except Exception` geschluckt wird, sonst kommt der saubere Interrupt-Exit nicht
  durch. Du musst also KeyboardInterrupt VOR dem `except Exception`-Block abfangen
  (eigenes `except KeyboardInterrupt`).

### Schritt 5 — Tests

- **Test-Datei** `tests/test_checkpoint.py`:
  - `save`/`load`-Runde (Atomarität, fcntl-locked, JSON-Zirkular, normalisierter
    Dateiname für `RWE.DE`).
  - `load` bei kaputter Datei → `None`.
  - `clear` entfernt Datei.
  - Env `CONCILIUM_STATE_DIR` (tmp_path) nutzt Isolation.
- **Test-Datei** `tests/test_pipeline_resume.py` (Kern-Verifikation):
  - Fake-LLM, der nach Schritt N (z.B. nach debate, vor trader) einen
    `RuntimeError`/`HTTP 429` wirft → erster `run_pipeline(..., resume=False)`
    schlägt fehl, hinterlässt Checkpoint bis Schritt N.
  - Zweiter `run_pipeline(..., resume=True)` → führt NUR Schritte N..Ende aus,
    die schon vorhandenen Teilergebnisse (analysts, debate) werden nicht neu
    berechnet (priffe via Fake-LLM-Zähler, dass die früheren Agenten NICHT erneut
    aufgerufen werden).
  - Erfolgreicher Lauf räumt den Checkpoint auf (Datei existiert nicht mehr).
  - `resume=False` ignoriert bestehenden Checkpoint (von vorn, alle Schritte neu).
  - Simulation: Um den "Crash nach Schritt N" zu simulieren, kannst du
    `patch.object(pipeline, 'debate', side_effect=RuntimeError(...))` bzw. die
    jeweilige Agent-Funktion patchen; oder einen Fake-LLM, der beim n-ten Call wirft.
    Nutze das Muster, das am saubersten ist.
- Verifiziere, dass alle bestehenden Tests (die `run_pipeline` mit MagicMock-LLM
  aufrufen, z.B. test_trade_revision, test_batch) weiterhin grün sind — der neue
  `resume=False` Default darf kein bestehendes Verhalten ändern.

### Schritt 6 — Report & CLI-Integration

- `generate_report` arbeitet mit dem fertigen `result` — unverändert (kein Resume
  im Renderer nötig).
- Nach einem Resume soll der Report normal generiert werden können.

---

## Verifikations-Checkliste (REIHENFOLGE ERZIELEN)

1. `/opt/data/depot-venv/bin/python -m ruff check src/ tests/ main.py` → 0 Fehler.
2. `/opt/data/depot-venv/bin/python -m pytest tests/ -q` → alle grün (bestehende +
   neue ~20-30).
3. `git add -A && git commit` — aussagekräftige Message.
4. `git push origin main`.
5. GitHub Actions CI abwarten (`gh run list` oder API; Fine-Grained PAT liest
   Runs/Logs). CI muss grün sein.
6. **Live-Test** mit echtem LLM:
   ```
   cd /opt/data/Concilium
   export LLM_API_KEY=$(grep -oP '^OLLAMA_API_KEY=\K.*' /opt/data/.env | tr -d '"' | tr -d "'" | head -1)
   export LLM_BASE_URL="https://ollama.com/v1"
   export LLM_MODEL="glm-5.2:cloud"
   /opt/data/depot-venv/bin/python main.py --ticker AAPL --resume
   ```
   Report prüfen: keine `+nan%`, kein `fehler`-Eintrag. Live-Resume-Test: einen
   Lauf per SIGINT abbrechen (z.B. `kill -INT` während des LLM-Calls) und mit
   `--resume` erneut starten → er setzt fort statt von vorn. (Wenn das Live-Abbruch
   in der Praxis schwer automatisierbar ist, genügt der deterministische Unit-Test
   als Beweis + ein normaler sauberer Live-Lauf mit `--resume`.)
7. Abschluss-Meldung an Flo: was umgesetzt wurde, wie Resume funktioniert, was im
   Fallback läuft, Verifikations-Ergebnis.

---

## Fallstricke / Bewertung

- **Rückwärtskompatibilität**: `run_pipeline` bekommt neuen optionalen Param
  `resume=False`. Alle bestehenden Aufrufer (cli.py, tests) ohne resume →
  unverändertes Verhalten. `chat()` / Agenten-Signaturen ändern sich NICHT.
- **`except Exception` schluckt kein `KeyboardInterrupt`**: `KeyboardInterrupt` ist
  eine BaseException, nicht Exception. Der `except Exception`-Block in cli.py darf
  ihn nicht fangen, sonst kein Exit-Code 130. Explizit `except KeyboardInterrupt`
  VOR den `except Exception`-Blöcken.
- **Test-Isolation**: Tests MÜSSEN `CONCILIUM_STATE_DIR` auf tmpdir setzen, damit
  sie nicht in echtes `state/` schreiben und Tests deterministisch bleiben.
- **JSON-Serialisierbarkeit**: `data` (yfinance) kann numpy-Typen enthalten →
  `json.dumps(..., default=str)`. Beim Load wieder tolerant.
- **Atomare Schreibsemantik**: in tmpfile → `os.replace` → danach clear.
  Nie halbgeschriebene Datei hinterlassen (für den Crash-Case selbst relevant).
- **Batch + Resume**: pro Ticker separat. Ein Ticker-Fehler crasht den Batch
  nicht (bestehendes Verhalten). Resume-Dateien pro Ticker unabhängig.
- **`state/` nicht versionieren** (Schritt 1).

---

## Done-Definition

- `checkpoint.py` (neu) mit save/load/clear, atomar + fcntl-Lock, `CONCILIUM_STATE_DIR`
  Übersteuerung, tolerante JSON-Serialisierung.
- `run_pipeline` schreibt nach jedem Agent-Schritt einen Checkpoint und unterstützt
  `resume=True`, das nur die fehlenden Schritte ab der letzten abgeschlossenen
  Stelle neu ausführt.
- Erfolgreicher Lauf räumt den Checkpoint auf.
- `cli.py` hat `--resume`/`--no-resume` und fängt `KeyboardInterrupt` sauber
  (Exit-Code 130, Checkpoint bleibt).
- `--evaluate` bleibt unberührt.
- `state/` in .gitignore.
- 665+ Tests grün, ruff grün, CI grün, Live-Test ohne `+nan%`/`fehler`, Resume
  verifiziert (Unit-Test deterministisch + sauberer Live-Lauf).

Wenn du das umgesetzt hast: gib die Zusammenfassung an Flo zurück mit den
Verifikations-Ergebnissen.
