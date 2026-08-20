#!/usr/bin/env python3
"""TradingAgents-Light CLI.

Nutzung:
    python main.py --ticker AAPL               # vollständige Analyse mit LLM
    python main.py --ticker AAPL --no-llm       # nur Datensnapshot, keine LLM-Aufrufe
    python main.py --ticker AAPL --backtest     # mit Backtest-Signalproxy

Dünner Wrapper um tradingagents_light.cli. Alternativ direkt nutzbar als
installierter Befehl `tradingagents-light`.
"""

from __future__ import annotations

import os
import sys

# src-Verzeichnis zum Pfad hinzufügen, damit tradingagents_light ohne
# Installation direkt aus dem Repo importierbar ist (python main.py ...)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from tradingagents_light.cli import main  # noqa: E402, I001


if __name__ == "__main__":
    sys.exit(main())
