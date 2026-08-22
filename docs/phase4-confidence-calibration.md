# Concilium — Phase 4: Konfidenz-Kalibrierung (Lernen harten)

> **Handoff-Dokument für eine frische, kontextfreie Session.**
> Lies diese Datei vollständig und folge ihr. Zielbild aus `docs/ROADMAP.md` (Phase 4).

---

## Kontext (für die Session, die das umsetzt)

Concilium ist ein Multi-Agenten-Fonds-Entscheidungssystem (Python-CLI, Repo
`/opt/data/Concilium`, GitHub `flokoko/Concilium`). Es führt ein
Entscheidungs-Journal (`journal/decisions.csv`) und ein Track-Record-Modul
(`--evaluate`), das Entscheidungen gegen tatsächliche Kurse abgleicht.

**Das Problem dieser Phase:** Beide Systeme (Concilium + Original TradingAgents)
haben Reflexion/Kontext-Feedback. Aber nur Concilium hat bereits `--evaluate`.
Wir können die Lern-Schleife **quantifizieren**: über viele Läufe prüfen, ob hohe
Konfidenz wirklich öfter richtig ist (Konfidenz-Kalibrierung) und ob Portfolio-Fit
mit Erfolg korreliert. Das Original misst das nicht — **Differenzierung**.

**Ziel:**
1. `evaluate.py`: neue Kennzahl **Konfidenz-Kalibrierungs-Score** (bspw. Brier-Score /
   Reliability-Bänder) über das bestehende Journal.
2. `feedback.py`: Kalibrierungs-Statistik als Kontext-Block injizieren, damit Agenten
   gezielt über-/unterkonfident korrigieren.
3. Report-Erweiterung in `--evaluate`.

**Konventionen (unbedingt einhalten):**
- Lauf: `/opt/data/depot-venv/bin/python` (NICHT System-Python).
- LLM-Key aus `OLLAMA_API_KEY` in `/opt/data/.env`.
- Ruff-Check: `/opt/data/depot-venv/bin/python -m ruff check src/ tests/ main.py`
- Tests: `/opt/data/depot-venv/bin/python -m pytest tests/ -q` (763 Tests, offline).
- Verifikationspflicht: ruff grün + pytest grün + `git push` + CI grün + Live-Test
  mit echtem LLM (Key aus .env) über `--evaluate`.
- `main.py` hat `sys.path.insert`-Workaround → `# noqa: E402, I001` auf Import belassen.

---

## Derzeitige Architektur (relevant)

**`src/concilium/evaluate.py`** — Track-Record-Evaluierung:
- `_evaluate_single(row, prices, lookback_days)` → bewertet EINE Journal-Zeile:
  `{hit, rendite_pct, ziel_erreicht, stop_gerissen, action, rating, rating_distance,
  confidence, portfolio_fit_score, ticker, timestamp}`.
  - `confidence` kommt aus `row["confidence"]` (final-PM-Confidence, 1-5).
  - `hit` ist bool (richtig/falsch) — Basis für Kalibrierung.
- `_aggregate(evaluations)` → aggregiert zu dict mit u.a.:
  - `hit_rate_gesamt`, `durchschnitt_rating_distanz`, `konfidenz_baende` (hoch/mittel/niedrig Hit-Rate), `portfolio_fit_hoch`.
- `evaluate_journal(journal_file, lookback_days, llm)` → Hauptfunktion, `_empty_result()` bei leer.
- `realised_return_for_row(row, lookback_days)` → für Reflexion.

**`src/concilium/feedback.py`** — Kontext-Feedback:
- `_compute_stats(rows)` → Track-Record-Statistiken aus Journal (actions, ratings, avg_confidence, avg_ensemble_confidence, avg_portfolio_fit, avg_ziel_gewichtung, kauf_genehmigt_pct).
- `build_feedback_context(journal_file, min_decisions=5)` → deutscher Kontext-Block für Trader/PM-Prompts.

**`src/concilium/report.py`** — `generate_track_record_report(eval_result)`:
- Sektion "## Konfidenz-Bänder (Trefferquote nach Confidence)" mit `| Band | n | Hit-Rate |`-Tabelle aus `konfidenz_baende`.

**`src/concilium/cli.py`** — `--evaluate [journal_datei]`, `--lookback`.

---

## Umsetzungsauftrag

### Schritt 1 — `evaluate.py`: Brier-Score & Konfidenz-Kalibrierungs-Kennzahlen

Ergänze in `_aggregate` (oder neue Hilfsfunktion) die Berechnung der
**Konfidenz-Kalibrierung**. Kernideen:

1. **Brier-Score** (klassisch, binär): Für jedes `hit` (True=1, False=0) und die
   normalisierte Konfidenz `p = confidence / 5` (0.2..1.0) gilt:
   `brier_i = (p - hit_int)^2`. Der Brier-Score ist der Durchschnitt über alle
   bewerteten Zeilen (niedriger = besser, 0 = perfekt, 0.25 = "immer 50%").
   - Nur Zeilen verwenden, wo `confidence` nicht None UND `hit` nicht None ist.
   - `math.isfinite`-Guards.
   - Neues Feld im Ergebnis: `konfidenz_kalibrierung` (dict) mit:
     - `brier_score` (float | None)
     - `n` (Anzahl bewertete Zeilen)
     - `durchschnittliche_konfidenz` (mean p)
     - `durchschnittliche_tatsaechliche_hit_rate` (mean hit)
     - `kalibrierungs_gap` = `durchschnittliche_konfidenz - hit_rate` (positiv =
       überkonfident, negativ = unterkonfident)

3. **Reliability-Bänder** (Kalibrierungs-Diagramm): Gruppiere die bewerteten Zeilen
   in Konfidenz-Intervalle (z.B. [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0])
   und berechne je Bin: `n`, `mittlere_konfidenz`, `hit_rate`. Das zeigt, ob die
   Hit-Rate mit der Konfidenz skaliert (ideale Kalibrierung: hit_rate ≈ konfidenz).
   Neues Feld: `reliability_bins` (list[dict]).
   Diese Bänder sind feiner als die bestehenden `konfidenz_baende` (hoch/mittel/niedrig).

4. **`_empty_result()`** um die neuen Felder erweitern (None/[]-Defaults), damit der
   Report robust bei fehlenden Daten ist.

**Design-Konvention:** Die neue Kennzahl soll die bestehenden Felder NICHT brechen.
`konfidenz_baende` bleibt. `konfidenz_kalibrierung` und `reliability_bins` sind neu.

### Schritt 2 — `feedback.py`: Kalibrierungs-Statistik injizieren

Erweitere `build_feedback_context` (bzw. die Statistik-Berechnung), sodass der
Kontext-Block zusätzlich eine **Kalibrierungs-Zeile** enthält, die den Agenten
sagt, ob sie tendenziell über- oder unterkonfident sind:

- Wenn Brier-/Kalibrierungs-Daten aus dem Journal berechenbar sind (hit + confidence
  vorhanden): füge eine Zeile hinzu, z.B.:
  ```
  Konfidenz-Kalibrierung: Ø Confidence X.X/5 vs. tatsächliche Trefferquote YY%.
  Tendenz: überkonfident / unterkonfident / gut kalibriert.
  ```
- Leite "Tendenz" aus `kalibrierungs_gap` ab (gap > +0.15 → überkonfident;
  gap < -0.15 → unterkonfident; sonst gut kalibriert). Schwellen sinnvoll wählen
  (du darfst sie justieren), aber dokumentieren.
- WICHTIG: `build_feedback_context` nutzt NUR Journal-CSV-Felder (kein yfinance),
  wie bisher. Die Kalibrierung muss aus `confidence` + einem Hit-Feld ableitbar sein.
  Das Problem: Das Journal hat `confidence` (final) und `hit` wird in evaluate berechnet
  (yfinance-basiert). Für feedback (ohne Netz) kannst du den `hit` NICHT direkt haben.
  → Entweder (a) eine einfache Proxy-Kalibrierung aus den Journal-Feldern (z.B.
  "wie oft wurde KAUFEN final GENEHMIGT" als Erfolgs-Proxi, oder die Verteilung von
  confidence), ODER (b) eine bestehende Statistik erweitern. Der Handoff verlangt:
  "Kalibrierungs-Statistik als Kontext-Block injizieren". Da feedback ohne Netz läuft,
  ist eine Näherung via Journal-Feldern akzeptabel — aber der Brier-Score-Server aus
  evaluate.py ist die präzise Messung für den Report. Du darfst entscheiden, ob
  feedback eine eigene leichte Kalibrierungs-Näherung macht oder auf eine
  evaluate-Hilfsfunktion zurückgreift, die KEIN Netz braucht. WICHTIG: feedback darf
  nicht crashen und muss ohne yfinance auskommen.

### Schritt 3 — `report.py`: Track-Record-Report erweitern

Erweitere `generate_track_record_report`:
- Neue Sektion "## Konfidenz-Kalibrierung" (nach den bestehenden Konfidenz-Bändern):
  - Brier-Score
  - Durchschnittliche Konfidenz vs. Hit-Rate
  - Kalibrierungs-Gap + Interpretation (über/unter/gut kalibriert)
  - Reliability-Bänder-Tabelle: | Konfidenz-Bin | n | Ø Konfidenz | Hit-Rate |
- Robust gegen fehlende Werte (N/A). `math.isfinite`-Guards.

### 4 — `cli.py` & `evaluate.py`
- `--evaluate` unverändert (nutzt dieselbe `generate_track_record_report`-Funktion).
- Keine neuen CLI-Flags nötig. Die Kennzahlen erscheinen automatisch im Report.

### 5 — Tests

- **`tests/test_calibration.py`** (neu):
  - Brier-Score: perfekt kalibriert (p==hit) → score ≈ 0.00xx klein; unpassend
    (p=0.8 aber hit=0) → höherer Brier.
  - Reliability-Bins: Gruppierung korrekt.
  - `_aggregate` mit konstruierten evaluations → `konfidenz_kalibrierung`-Keys present.
  - Robustheit: keine confidence / keine hit → None, kein Crash.
  - `build_feedback_context` mit konstruiertem Journal → enthält Kalibrierungs-Zeile,
    ohne Netzwerk.
  - `generate_track_record_report` mit kalibrierungs-dict → Sektion vorhanden, N/A bei fehlenden Werten.
- Bestehende 763 Tests grün halten.

## Verifikations-Checkliste (REIHENFOLGE ERZIELEN)
1. `/opt/data/depot-venv/bin/python -m ruff check src/ tests/ main.py` → 0 Fehler.
2. `/opt/data/depot-venv/bin/python -m pytest tests/ -q` → alle grün (763 + neue).
3. `git add -A && git commit` — aussagekräftige Message.
4. `git push origin main`.
5. GitHub Actions CI abwarten (API). CI grün.
6. **Live-Test** mit echtem LLM:
   ```
   cd /opt/data/Concilium
   export LLM_API_KEY=$(grep -oP '^OLLAMA_API_KEY=\K.*' /opt/data/.env | tr -d '"' | tr -d "'" | head -1)
   export LLM_BASE_URL="https://ollama.com/v1"
   export LLM_MODEL="glm-5.2:cloud"
   /opt/data/depot-venv/bin/python main.py --evaluate
   ```
   Report prüfen: "## Konfidenz-Kalibrierung"-Sektion mit Brier-Score, Gap, Tendenz,
   Reliability-Bänder; keine `+nan%`/`fehler`. (Wenn das Journal noch wenig Einträge
   hat, wird der Brier evtl. None/aus wenigen Zeilen — das ist ok.)
7. Abschluss-Meldung an Flo: was umgesetzt wurde, wie die Kalibrierung gemessen wird,
   was in feedback injiziert wird, Verifikations-Ergebnis.

---

## Fallstricke / Bewertung
- **Rückwärtskompatibilität**: neue Felder in `_empty_result()` und `_aggregate`; bestehende Felder unverändert. Report-Konfidenz-Bänder (hoch/mittel/niedrig) bleiben.
- **Feedback ohne Netz**: `build_feedback_context` darf NICHT yfinance laden (nur CSV-Felder). Die Kalibrierungs-Zeile muss aus Journal-Feldern ableitbar sein oder auf eine netzfreie Hilfsfunktion zurückgreifen.
- **`math.isfinite`** überall; `_fmt_pct2`/`_fmt` nutzen.
- **Kein `+nan%` / `fehler`** im Report aus Kalibrierungswerten.

---

## Done-Definition
- `evaluate.py`: `konfidenz_kalibrierung` (Brier-Score, Ø-Konfidenz vs Hit-Rate, Gap, Tendenz) + `reliability_bins` in `_aggregate` + `_empty_result`.
- `feedback.py`: `build_feedback_context` injiziert eine Kalibrierungs-/Tendenz-Zeile (netzfrei).
- `report.py`: `generate_track_record_report` hat eine "## Konfidenz-Kalibrierung"-Sektion (Brier, Gap, Tendenz, Reliability-Bänder-Tabelle).
- `--evaluate` zeigt die neuen Kennzahlen.
- 763+ Tests grün, ruff grün, CI grün, Live-Test `--evaluate` ohne `+nan%`/`fehler`.

Wenn du das umgesetzt hast: gib die Zusammenfassung an Flo zurück mit den Verifikations-Ergebnissen.
