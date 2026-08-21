"""LLM-Client — OpenAI-kompatibler Client via requests (kein SDK)."""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Umgebungsvariablen
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://ollama.com/v1"
DEFAULT_MODEL = "glm-5.2:cloud"
TIMEOUT_SECONDS = 120
MAX_RETRIES = 2
RETRY_BACKOFF = 2  # Sekunden

# Status-Codes, die einen Modell-Fallback rechtfertigen (überlasteter/ausgefallener
# primärer Endpunkt). Timeout/ConnectionError führen NICHT zum Fallback, da diese
# eher Netzwerk- als Modell-spezifische Probleme signalisieren.
_FALLBACK_STATUS_CODES = {429, *range(500, 600)}


def _jittered_backoff(attempt: int) -> float:
    """Berechnet den Backoff-Wert mit zufälligem Jitter (±30%).

    Verhindert Thundering-Herd bei parallelen/aufeinanderfolgenden Requests,
    die 429/5xx auslösen. Basis: RETRY_BACKOFF * (attempt + 1), multipliziert
    mit random.uniform(0.7, 1.3) → Jitter-Bereich ±30%.
    """
    return RETRY_BACKOFF * (attempt + 1) * random.uniform(0.7, 1.3)


class LLMClient:
    """OpenAI-kompatibler LLM-Client.

    Konfiguration via Umgebungsvariablen:
        LLM_BASE_URL  (default: https://ollama.com/v1)
        LLM_API_KEY   (default: aus OLLAMA_API_KEY)
        LLM_MODEL     (default: glm-5.2:cloud)
        LLM_FALLBACK_MODEL  (default: leer = kein Fallback)

    Fallback-Modell: Wenn der primäre Request nach allen Retries mit 429 oder
    5xx fehlschlägt und ein Fallback-Modell konfiguriert ist, wird EIN Versuch
    mit dem Fallback-Modell unternommen (gleiche Base-URL, anderes Modell-Feld).
    Bei Timeout/ConnectionError wird kein Fallback aktiviert.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
    ):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OLLAMA_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self.fallback_model = (
            fallback_model
            if fallback_model is not None
            else os.environ.get("LLM_FALLBACK_MODEL", "")
        ) or None

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Sendet Messages an /chat/completions und gibt den Text zurück.

        Retry bei 5xx und 429 (max. MAX_RETRIES Mal mit Backoff).
        Wenn ein Fallback-Modell konfiguriert ist und der primäre Request nach
        allen Retries mit 429/5xx fehlschlägt, wird ein Versuch mit dem
        Fallback-Modell unternommen.
        """
        url = f"{self.base_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            return self._send_with_retries(url, headers, payload)
        except _RetryableHTTPError as exc:
            # Primärer Request nach allen Retries mit 429/5xx fehlgesehen.
            # Fallback-Modell versuchen, falls konfiguriert.
            if self.fallback_model is not None:
                logger.warning(
                    "Primärmodell '%s' erschöpft (%s) — wechsle auf Fallback-Modell '%s'",
                    self.model,
                    exc,
                    self.fallback_model,
                )
                fallback_payload = dict(payload)
                fallback_payload["model"] = self.fallback_model
                try:
                    return self._send_with_retries(url, headers, fallback_payload, is_fallback=True)
                except _RetryableHTTPError as fb_exc:
                    raise RuntimeError(
                        f"LLM-Anfrage fehlgeschlagen: Primärmodell ({exc}), "
                        f"Fallback-Modell '{self.fallback_model}' ({fb_exc})"
                    ) from None
            # Ohne Fallback-Modell: Original-Fehler weitergeben
            raise RuntimeError(f"LLM-Anfrage fehlgeschlagen nach {MAX_RETRIES + 1} Versuchen: {exc}") from None

    def _send_with_retries(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        is_fallback: bool = False,
    ) -> str:
        """Sendet den Request mit Retries bei 429/5xx.

        Löst _RetryableHTTPError aus, wenn alle Retries mit 429/5xx fehlschlagen
        (damit der Aufrufer einen Modell-Fallback durchführen kann).
        Andere Fehler (Timeout, ConnectionError) werden als RuntimeError geworfen.
        """
        model_label = payload.get("model", "?")
        last_error: str | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                label = f"Fallback '{model_label}'" if is_fallback else f"Modell '{model_label}'"
                logger.debug("LLM-Request (%s, Versuch %d/%d) an %s", label, attempt + 1, MAX_RETRIES + 1, url)
                resp = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT_SECONDS)

                if resp.status_code in _FALLBACK_STATUS_CODES:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    logger.warning("LLM-Fehler %s — Versuch %d/%d", last_error, attempt + 1, MAX_RETRIES + 1)
                    if attempt < MAX_RETRIES:
                        time.sleep(_jittered_backoff(attempt))
                        continue
                    raise _RetryableHTTPError(last_error)

                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content if content is not None else ""

            except requests.Timeout:
                last_error = f"Timeout nach {TIMEOUT_SECONDS}s"
                logger.warning("LLM-Timeout — Versuch %d/%d", attempt + 1, MAX_RETRIES + 1)
                if attempt < MAX_RETRIES:
                    time.sleep(_jittered_backoff(attempt))
                    continue
                raise RuntimeError(f"LLM-Anfrage Timeout nach {MAX_RETRIES + 1} Versuchen") from None

            except requests.ConnectionError as exc:
                last_error = f"Verbindungsfehler: {exc}"
                logger.warning("LLM-Verbindungsfehler — Versuch %d/%d: %s", attempt + 1, MAX_RETRIES + 1, exc)
                if attempt < MAX_RETRIES:
                    time.sleep(_jittered_backoff(attempt))
                    continue
                raise RuntimeError(f"LLM-Verbindung fehlgeschlagen: {last_error}") from exc

        # Sollte nie erreicht werden, aber als Sicherung
        raise RuntimeError(f"LLM-Anfrage fehlgeschlagen: {last_error}")


class _RetryableHTTPError(Exception):
    """429/5xx-Fehler nach Erschöpfung aller Retries — signalisiert Fallback-Möglichkeit."""
    pass
