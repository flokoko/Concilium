"""Quantitativer Multi-Faktor-Score als deterministischer Anker für den LLM-Fundamental-Analysten.

Die Funktion ``compute_multi_factor_score`` berechnet aus den im
``data["fundamentals"]``-Dict vorhandenen Feldern einen neutralen, transparenten
Quant-Score. Dieser dient als Referenzwert (Anker) für den LLM-Fundamental-Analysten,
ist aber NICHT die finale Wahrheit — der LLM soll ihn kritisch einordnen.

Alle Sub-Scores sind defensiv implementiert (None-Handling), craschen nie und
liefern Werte im Bereich 1-5.
"""

from __future__ import annotations

from typing import Any


def _safe_float(val: Any) -> float | None:
    """Konvertiert val zu float oder gibt None zurück (defensiv)."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN-Check
        return None
    return f


# ---------------------------------------------------------------------------
# Sub-Score-Berechnungen (jeweils 1-5, defensiv)
# ---------------------------------------------------------------------------


def _value_score(f: dict[str, Any]) -> float | None:
    """Value-Score: KGV/PEG-Attraktivität + Analysten-Upside.

    Komponenten:
      - pe_ratio: <=10→5, <=18→4, <=25→3, <=40→2, >40→1, None→3 (neutral)
      - peg_ratio: <1→5, <1.5→4, <2→3, <3→2, >=3→1, None→neutral 3
      - analyst_upside_pct: >20→5, >10→4, >0→3, >-10→2, sonst→1, None→neutral 3

    Gesamt = Durchschnitt der verfügbaren Komponenten (mind. 1 benötigt).
    """
    components: list[float] = []

    # KGV
    pe = _safe_float(f.get("pe_ratio"))
    if pe is not None:
        if pe <= 10:
            components.append(5.0)
        elif pe <= 18:
            components.append(4.0)
        elif pe <= 25:
            components.append(3.0)
        elif pe <= 40:
            components.append(2.0)
        else:
            components.append(1.0)

    # PEG
    peg = _safe_float(f.get("peg_ratio"))
    if peg is not None:
        if peg < 1:
            components.append(5.0)
        elif peg < 1.5:
            components.append(4.0)
        elif peg < 2:
            components.append(3.0)
        elif peg < 3:
            components.append(2.0)
        else:
            components.append(1.0)

    # Analysten-Upside
    upside = _safe_float(f.get("analyst_upside_pct"))
    if upside is not None:
        if upside > 20:
            components.append(5.0)
        elif upside > 10:
            components.append(4.0)
        elif upside > 0:
            components.append(3.0)
        elif upside > -10:
            components.append(2.0)
        else:
            components.append(1.0)

    if not components:
        return None
    return round(sum(components) / len(components), 2)


def _momentum_score(f: dict[str, Any]) -> float | None:
    """Momentum-Score: Nähe zum 52-Wochen-Hoch + Analysten-Konsens.

    Komponenten:
      - 52W-Nähe: current_price wird aus technicals nicht direkt übergeben,
        aber falls fifty_two_week_high und fifty_two_week_low vorhanden:
        Wir approximieren die Nähe zum Hoch über das Verhältnis von
        current_price zu high. Da current_price in fundamentals nicht
        vorhanden ist, nutzen wir analyst_target_mean als Proxy falls
        vorhanden, sonst nur high/low-Relation.
        Pragmatisch: Wenn high und low vorhanden und current_price über
        technics ermittelt wird, übergeben wir es via f.get("current_price").
        Die Pipeline kann current_price zu fundamentals hinzufügen.
        Hier: wir prüfen f.get("_current_price") falls von außen gesetzt,
        sonst analyst_target_mean als Kurs-Proxy.
        Letztlich: nahe Hoch (>90% des Hochs)→5, mittel→3, nahe Tief→1.

      - recommendation_mean: <=1.8→5, <=2.5→4, <=3.2→3, <=3.9→2, sonst→1
    """
    components: list[float] = []

    # 52W-Nähe — wir brauchen einen aktuellen Kurs.
    # fundamentals enthält current_price nicht, aber die Pipeline kann es
    # ergänzen. Falls nicht vorhanden, überspringen wir diese Komponente.
    high = _safe_float(f.get("fifty_two_week_high"))
    low = _safe_float(f.get("fifty_two_week_low"))
    current = _safe_float(f.get("current_price"))

    if high is not None and current is not None and high > 0:
        ratio = current / high
        if ratio > 0.90:
            components.append(5.0)
        elif ratio > 0.75:
            components.append(4.0)
        elif ratio > 0.60:
            components.append(3.0)
        elif ratio > 0.45:
            components.append(2.0)
        else:
            components.append(1.0)
    elif low is not None and current is not None and high is not None and high > low:
        # Fallback: Position im 52W-Range
        pos_in_range = (current - low) / (high - low)
        if pos_in_range > 0.80:
            components.append(5.0)
        elif pos_in_range > 0.60:
            components.append(4.0)
        elif pos_in_range > 0.40:
            components.append(3.0)
        elif pos_in_range > 0.20:
            components.append(2.0)
        else:
            components.append(1.0)

    # Analysten-Konsens (recommendation_mean: 1=strong buy ... 5=sell)
    rec_mean = _safe_float(f.get("recommendation_mean"))
    if rec_mean is not None:
        if rec_mean <= 1.8:
            components.append(5.0)
        elif rec_mean <= 2.5:
            components.append(4.0)
        elif rec_mean <= 3.2:
            components.append(3.0)
        elif rec_mean <= 3.9:
            components.append(2.0)
        else:
            components.append(1.0)

    if not components:
        return None
    return round(sum(components) / len(components), 2)


def _quality_score(f: dict[str, Any]) -> float | None:
    """Qualitäts-Score: Gewinnmarge + Umsatzwachstum + Dividendenrendite (Stabilität).

    Komponenten:
      - profit_margin: >=0.25→5, >=0.15→4, >=0.08→3, >=0→2, negativ→1
      - revenue_growth: >=0.15→5, >=0.05→4, >=0→3, >=-0.05→2, sonst→1
      - dividend_yield: >0.04→5, >0.02→4, >0→3, ==0→2, None→neutral 3
      - fcf_margin: >=20→5, >=10→4, >0→3, ==0→2, negativ→1
      - net_debt_to_ebitda: <0→5, <1→5, <2→4, <3→3, <5→2, >=5→1
    """
    components: list[float] = []

    # Gewinnmarge
    margin = _safe_float(f.get("profit_margin"))
    if margin is not None:
        if margin >= 0.25:
            components.append(5.0)
        elif margin >= 0.15:
            components.append(4.0)
        elif margin >= 0.08:
            components.append(3.0)
        elif margin >= 0:
            components.append(2.0)
        else:
            components.append(1.0)

    # Umsatzwachstum
    growth = _safe_float(f.get("revenue_growth"))
    if growth is not None:
        if growth >= 0.15:
            components.append(5.0)
        elif growth >= 0.05:
            components.append(4.0)
        elif growth >= 0:
            components.append(3.0)
        elif growth >= -0.05:
            components.append(2.0)
        else:
            components.append(1.0)

    # Dividendenrendite (Stabilitäts-Indikator)
    div = _safe_float(f.get("dividend_yield"))
    if div is not None:
        if div > 0.04:
            components.append(5.0)
        elif div > 0.02:
            components.append(4.0)
        elif div > 0:
            components.append(3.0)
        elif div == 0:
            components.append(2.0)
    # None → keine Komponente (neutral, nicht 3 — da Dividende 0 info)

    # FCF-Marge (zusätzliches Qualitäts-Signal — positiv = besser)
    fcf_margin = _safe_float(f.get("fcf_margin"))
    if fcf_margin is not None:
        if fcf_margin >= 20:
            components.append(5.0)
        elif fcf_margin >= 10:
            components.append(4.0)
        elif fcf_margin > 0:
            components.append(3.0)
        elif fcf_margin == 0:
            components.append(2.0)
        else:
            components.append(1.0)

    # Net-Debt/EBITDA (Verschuldungs-Qualität — niedriger = besser)
    ndte = _safe_float(f.get("net_debt_to_ebitda"))
    if ndte is not None:
        if ndte < 0:
            # Nettoliquidität (negative Nettoverschuldung) → sehr stark
            components.append(5.0)
        elif ndte < 1.0:
            components.append(5.0)
        elif ndte < 2.0:
            components.append(4.0)
        elif ndte < 3.0:
            components.append(3.0)
        elif ndte < 5.0:
            components.append(2.0)
        else:
            components.append(1.0)

    if not components:
        return None
    return round(sum(components) / len(components), 2)


# ---------------------------------------------------------------------------
# Kurz-Einschätzung (deterministischer Text)
# ---------------------------------------------------------------------------

_LABELS = {
    5: "sehr stark",
    4: "stark",
    3: "solide",
    2: "schwach",
    1: "sehr schwach",
}

_VALUE_LABELS = {
    5: "sehr günstig",
    4: "günstig",
    3: "fair",
    2: "teuer",
    1: "sehr teuer",
}

_MOMENTUM_LABELS = {
    5: "sehr stark",
    4: "positiv",
    3: "neutral",
    2: "negativ",
    1: "sehr schwach",
}


def _label_for(score: float | None, mapping: dict[int, str]) -> str:
    """Mapt einen Score auf ein Label. None → 'N/A'."""
    if score is None:
        return "N/A"
    # Auf int runden für Mapping
    key = int(round(score))
    return mapping.get(key, "N/A")


def _build_kurzeinschaetzung(
    value: float | None,
    momentum: float | None,
    quality: float | None,
) -> str:
    """Erzeugt einen deterministischen deutschen Kurztext aus den Sub-Scores."""
    parts: list[str] = []

    if value is not None:
        parts.append(f"Value {_VALUE_LABELS.get(int(round(value)), 'N/A')}")
    if momentum is not None:
        parts.append(f"Momentum {_MOMENTUM_LABELS.get(int(round(momentum)), 'N/A')}")
    if quality is not None:
        parts.append(f"Qualität {_LABELS.get(int(round(quality)), 'N/A')}")

    if not parts:
        return "Keine Fundamentals verfügbar für quantitative Einschätzung."
    return ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Haupt-Funktion
# ---------------------------------------------------------------------------


def compute_multi_factor_score(fundamentals: dict[str, Any]) -> dict[str, Any]:
    """Berechnet einen quantitativen Multi-Faktor-Score als deterministischer Anker.

    Args:
        fundamentals: Dict mit Fundamentals-Feldern (aus data["fundamentals"]).
            Felder können None sein — die Funktion crasht nie.

    Returns:
        Dict mit:
          - value_score: float|None (1-5)
          - momentum_score: float|None (1-5)
          - quality_score: float|None (1-5)
          - overall_score: float|None (gerundeter Durchschnitt der verfügbaren)
          - subscores_available: int (Anzahl verfügbarer Sub-Scores, 0-3)
          - kurzeinschaetzung: str (deterministischer deutscher Text)
    """
    if not isinstance(fundamentals, dict):
        fundamentals = {}

    value = _value_score(fundamentals)
    momentum = _momentum_score(fundamentals)
    quality = _quality_score(fundamentals)

    available = [s for s in (value, momentum, quality) if s is not None]
    subscores_available = len(available)

    if subscores_available == 0:
        overall: float | None = None
    else:
        overall = round(sum(available) / len(available), 2)

    return {
        "value_score": value,
        "momentum_score": momentum,
        "quality_score": quality,
        "overall_score": overall,
        "subscores_available": subscores_available,
        "kurzeinschaetzung": _build_kurzeinschaetzung(value, momentum, quality),
    }
