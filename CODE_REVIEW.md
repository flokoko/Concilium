# 🔍 Concilium — Review, TradingAgents-Vergleich, Bugs & Optimierungen

**Datum:** 26.08.2026
**Projekt:** `/opt/data/Concilium` (Repo `flokoko/Concilium`, HEAD `99a41f4`)
**Vergleichs-Repo:** `TauricResearch/TradingAgents` (geklont nach `/opt/data/TradingAgents`)
**Tests:** 1045 passed, 2 skipped (Baseline grün)

---

## Zusammenfassung

Concilium ist als eigenständiges, schlankes System (kein LangGraph, kein SDK) bereits sehr
weit über seinen Ursprung TradingAgents hinausgewachsen. Es hat **bereits Funktionen, die
TradingAgents nicht hat**: echte Depot-Einbindung (Portfolio-Fit), Kalibrierungs-JSON,
kalibrierungs-gewichtetes Ensemble, Entscheidungs-Disziplin (Rating-Dämpfung), segmentierte
Brier-Scores, Track-Record-Evaluierung, Währungsrisiko. Der Vergleich zeigt dennoch
**6 bestätigte Bugs/Logikfehler** (2 davon sind Live-relevant) und einen klaren Katalog an
**Feature-Portierungen aus TradingAgents**, die Concilium noch fehlen.

---

# 🚨 BUGS (bestätigt, teils Live-wirksam)

## B1. 🔴 `portfolio_overlap`: `_idx` wird nie gesetzt → falscher Gesamt-Overlap
**Datei:** `src/concilium/portfolio_analysis.py` (Z. 268, 294) + `src/concilium/portfolio_fit.py`
**Problem:** `portfolio_overlap` dedupliziert überlappte Positionen über `pos.get("_idx", 0)`.
Aber `_parse_positions()` (portfolio_fit.py) **setzt `_idx` niemals** — jede Position fällt
auf Default `0`. Dadurch werden alle Positionen mit `_idx==0` als „eine überlappte Position“
gezählt, und `total_overlap_pct` summiert falsche Positionen.
**Live-Reproduktion (verifiziert):** Depot mit AAPL 5% + MSFT 30%, Analyse AAPL →
`total_overlap_pct = 35.0%` statt korrekt `5.0%`. Die Warnschwelle `>20%` wird also regelmäßig
fälschlich ausgelöst.
**Fix:** In `_parse_positions` beim Anhängen `positions.append({... "_idx": len(positions)})`
setzen, oder in `portfolio_overlap` auf `id(pos)` / enumerierte Indexe zurückgreifen.
**Risiko ungefixt:** Falsche Overlap- und Konzentrationswarnungen im Portfolio-Modus und
Portfolio-Blick-Sektion.

## B2. 🔴 Entscheidungs-Disziplin (Rating-Dämpfung) wird in `trade_revision` umgangen
**Datei:** `src/concilium/agents.py` (Ende von `trade_revision`, Z. 1704–1722)
**Problem:** `trader()` und `ensemble_trader()` dämpfen `STARK KAUFEN/VERKAUFEN`→einfache
Aktion bei überkonfidenter Historie (`_dampen_stark_rating` / `_final_dampen_ensemble`).
Der **Trade-Revision (5c)** ruht die Dämpfung aber **nicht** — er normalisiert nur
`rating`/`aktion`, ohne `_dampen_stark_rating` erneut aufzurufen. Da der revidierte Trade den
Original-Trade **ersetzt** (`result["trade"] = revised`) und damit ins Journal + zum PM geht,
wird die in Commit `6f7d892` eingebaute Überkonfidenz-Dämpfung **umgangen**, sobald die Revision
ein STARK-Rating ausgibt — genau in den Live-Messungen, die überkonfident waren (KAUFEN Gap +0.44).
**Fix:** Am Ende von `trade_revision` nach `_ensure_ziel_stop` ein
`_dampen_stark_rating(result, result["rating"])` einfügen (und `_final_dampen_ensemble` analog,
falls das Ensemble den finalen Trade bestimmt).

## B3. 🟡 `_momentum_score`: 52-Wochen-Hoch-Komponente ist toter Code
**Datei:** `src/concilium/factors.py` (Z. 116–146)
**Problem:** Der „52W-Nähe“-Zweig liest `f.get("current_price")` — aber `compute_multi_factor_score`
wird in `_build_data_text` mit `fundamentals` aufgerufen (data.py Z. 384), das **kein**
`current_price` enthält (der Kurs liegt in `technicals`). `current` ist daher immer `None`,
beide `if`-Zweige (Z. 122 und 134) greifen nie → Momentum-Score verliert dauerhaft seine
stärkste Komponente und besteht nur aus `recommendation_mean`.
**Fix:** `_build_data_text` aufruf ändern auf `compute_multi_factor_score({**f, "current_price": t.get("current_price")})`
oder den Kurs in `fundamentals` ergänzen.

## B4. 🟡 Prompt-Einheit: Margen/Renditen als 0.35 statt 35 % an die LLM
**Problem:** `_build_data_text` zeigt `profit_margin`, `dividend_yield`, `revenue_growth`,
`fcf_margin` via `_fmt_num` (ohne ×100) → die LLM sieht z. B. `Gewinnmarge 0.35` statt `35%`.
Der Report zeigt dagegen `_fmt_pct` (×100). Diese Inkonsistenz ist bei LLM-Modellen bekannt,
führt zu Fehlinterpretationen. Einheitlich auf Prozenttext umstellen.

## B5. 🟡 `max_drawdown_schaetzung` / `positionsgröße_empfohlen` sind Strings im Schema
**Problem:** `RISK_SCHEMA` erlaubt `anyOf` number/string für beide Felder. `positionsgröße_rechnerisch_pct`
bleibt float, aber die LLM-Antwort `positionsgröße_empfohlen` kann als `"5 %"`-String kommen →
downstream `float()`-Aufrufe schlagen fehl. In `risk_manager`/`report` defensiv normalisieren
(`_safe_float`, Prozent-Prefix entfernen) oder Schema strikt auf `number` setzen.

## B6. 🟡 `--portfolio` ignoriert `--peers`
**Problem:** `cli.py` parst `peers_list` und reicht es nur an `run_pipeline` (`--ticker`/`--tickers`),
**nicht** an `run_portfolio` (das keine `peers`-Param). Mit `--portfolio --peers X` wird peers
stumm ignoriert. Entweder in `run_portfolio` durchreichen oder einen CLI-Hinweis/Error.

---

# 🔍 VERGLEICH mit TradingAgents — was Concilium übernehmen kann

TradingAgents (LangGraph-basiert) hat Architektur-Stärken, die Concilium als Optimierungen
kopieren kann, **ohne** den (bewusst schlanken, SDK-freien) Ansatz aufzugeben:

## C1. 🔴 Multi-Runden-Debatte (Bull/Bear + Risk) statt Einzel-LLM-Calls
Concilium: je **ein** Bull-, ein Bear-, ein Risk-Aufruf. TradingAgents: Bull↔Bear ping-pong
bis `max_debate_rounds`, und Aggressive/Conservative/Neutral-Debatte bis `max_risk_discuss_rounds`,
mit „latest_speaker“-Routing (`conditional_logic.py`). Konkret umsetzbar: `debate()` in Schleife
laufen lassen, Trader bekommt konvergierte Argumente. Kostet mehr LLM-Tokens, verbessert aber die
Tiefe der Argumentation deutlich.

## C2. 🟡 Deterministisches „Verified Market Snapshot“-Tool für Analysten
TradingAgents liefert den Analysten ein `get_verified_market_snapshot()` (OHLCV + Indikatoren
als Ground-Truth, damit die LLM keine exakten Kurs-Behauptungen halluzinieren, vgl.
`market_data_validation_tools.py` + `build_verified_market_snapshot`). Concilium lässt den
Technik-Analysten nur Zahlen „im Kontext sehen“. **Portierung:** den `_build_data_text`-Technikblock
bereits als verbindlichen „Snapshot“ markieren und im Prompt als `source of truth` für exakte
Kurs-SMA/RSI-Angaben ausweisen (minimal-invasiv).

## 3. 🏠 Instrument-Identity-Ankerung
TradingAgents injiziert eine deterministisch aufgelöste Unternehmens-Identität
(`resolve_instrument_identity` + `build_instrument_context`) in **jeden** Agenten, damit keiner
das Unternehmen aus dem Chart halluziniert. Concilium hat nur ISIN/WKN-Auflösung; die echten
Firmen-Fakten (Land, Börse, Währung) gehen nicht in Analysten-Prompts. **Portierung:** ein
kurzer „Instrument-Kontext“-Block im `_build_data_text`-Prolog.

## 4. 🟡 Cross-Ticker-Gedächtnis (n_same + n_cross) statt nur Ticker-spezifischer Reflexion
TradingAgents `get_past_context(ticker)` liefert bis 5 Same-Ticker- + 3 Cross-Ticker-Lektionen
mit realisierten Returns in den PM-Prompt (memory.py Z. 70). Concilium hat nur Reflexion des
letzten Entscheids **desselben Tickers** (`build_reflection_context`). **Portierung:** ebenfalls
„recent cross-ticker lessons“ in den Track-Record-Block aufnehmen — gerade als Fondsmanager
wertvoll (generalisierte Lektionen).

## 5. 🟡 LangGraph-Checkpoint-Resume mit Graph-Shape-Signatur
Concilium hat bereits ein schlankes Checkpoint (`--resume`). TradingAgents erweitert es um eine
**Run-Signatur** (`_run_signature`), sodass ein Resume bei **geänderter Analystenauswahl/
Debatten-/Risk-Tiefe** ignoriert und frisch gestartet wird. **Optimierung:** analog
`_completed_steps` mit den Pipeline-Parametern (ensemble_runs, peers, backtest) stempeln, damit
ein `--resume` mit geänderten Flags nicht einen Checkpoint aus anderen Parametern fortschreibt.

## 6. 🔵 Deferred Reflection (Phase B)
TradingAgents resolved **pending**-Entries erst beim nächsten Run derselben Aktie (realisierten
Return + LLM-Reflexion), über `_resolve_pending_entries` + `batch_update_with_outcomes`.
Concilium berechnet Reflexion sofort mit lookback_days. Das TradingAgents-Modell ist fairer
(kein Look-ahead: der Ausgang existiert beim Originallauf noch gar nicht). **Portierung**
würde die Kalibrierungs-/Reflexions-Aussagekraft erhöhen.

## 7. 🔵 `max_entries`-Rotation im Entscheidungslog
TradingAgents' MemoryLog kappt bei `memory_log_max_entries` die ältesten **resolved**-Entries
(rotation in `_apply_rotation`), hält Pending immer. Concilium-Journal wächst unbegrenzt.
Klein aber sauber für langfristige Nutzung.

---

# ✅ STÄRKEN von Concilium (TradingAgents hat das NICHT)
- **Konfigurierte Mehrheitsabstimmung mit Vorrang-Hit-Definition** (Stop gerissen = Miss) — TradingAgents bewertet nur Rating ohne Hit-Definition.
- **Echte Depot-Integration** (Google-Sheet) mit Konzentrations-/Overlap/Währungsrisiko — TradingAgents hat gar kein Depot.
- **Netzfreie Kalibrierungs-JSON** + voronoi/segmentierte Brier + Entscheidungs-Disziplin — TradingAgents: nur Metriken im Log, keine Rückkopplung.
- **Rechnerisches Volatility-Targeting-Sizing** (min(2%/vol, 10%)) parallel zum LLM — TradingAgents: nur LLM.
- **Robuste Fallback-Kaskade** yfinance→Google→StockTwits→Reddit; deutscher Report mit Datenqualitäts-Warnung.

---

## ✅ PRIORISIERTE TODO-LISTE
### 🔴 Sofort
- [x] B1: `_idx` in `_parse_positions` setzen / `portfolio_overlap` fixen
- [x] B2: `_dampen_stark_rating` in `trade_revision` nachziehen (Disziplin nicht umgehen)
- [x] **Neu 02.09. N1:** Ziel-Gewichtungs-Dämpfung NACH Trade-Revision (Commit `628814a`) — vorher lief die Dämpfung mit der alten Aktion, obwohl die Revision die Aktion ändern kann
- [x] **Neu 02.09. N2:** `--review` schreibt kein Journal mehr (Commit `628814a`) — Exit-Review-Läufe verunreinigten die Kalibrierung der Neukauf-Analysen

### 🟡 Bald
- [x] B3: `current_price` in `compute_multi_factor_score` einspeisen
- [x] B4: Prompt-%-Einheit (×100) vereinheitlichen
- [x] B5: Positionsgrößen-String defensiv normalisieren
- [x] B6: `--portfolio --peers` durchreichen
- [x] C1: Multi-Runden-Debatte & Risk-Debatte (konfigurierbar)
- [x] C5: Resume-Signatur mit Konfigurations-Fingerprint (Commit `8f3db46`)

### 🔵 Nice-to-have
- [x] C2: Analysten-Snapshot als Ground-Truth-Anker (Commit `2f99e11`)
- [x] C3: Instrument-Identity-Kontext in alle Prompts (Commit `2f99e11`)
- [x] C4: Cross-Ticker-Lektionen im Track-Record-Block
  (Commit siehe git log — `build_cross_ticker_context` + `build_memory_context`
  in feedback.py; Pipeline ergänzt den Cross-Ticker-Block nur, wenn
  `build_reflection_context` ungepatcht ist — Rückwärtskompatibilität mit
  bestehenden Pipeline-Tests. Der Report-Abschnitt "Reflexion (Track-Record)"
  bleibt Ticker-spezifisch; der Cross-Ticker-Block geht via
  `_reflection_context` in die Trader-/Ensemble-/Risk-/PM-Prompts und ist
  zusätzlich in `result["_cross_ticker_context"]` abgelegt.)
- [x] C6: Pending-Entries-Rückwärts-Auflösung (Look-ahead-frei)
  (Commit siehe git log — JOURNAL_HEADER um reflection_status/resolved_at/
  realised_return_pct/alpha_pct/lesson erweitert; append_decision schreibt
  neue Entscheidungen als "pending". feedback.py::resolve_pending_reflections
  löst die jüngste Pending-Entry beim nächsten Lauf auf, sobald
  decision_date + lookback_days vollständig abgelaufen ist (atomarer,
  lock-gesicherter Zurückschreib ins Journal; Return + Lektion werden
  persistiert und von build_reflection_context wiederverwendet).
  build_reflection_context und build_cross_ticker_context liefern nur noch
  Reflexionen aus VOLLSTÄNDIG abgelaufenen Ausgangsfenstern — kein Look-ahead
  mehr. Pipeline ruft resolve_pending_reflections im Normal-Modus (journal=True)
  vor build_reflection_context auf; --review (journal=False) löst nichts auf.)
- [ ] C7: Journal-Rotation (max_entries)
