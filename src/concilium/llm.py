"""LLM-Client — OpenAI-kompatibler Client via requests (kein SDK)."""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, NamedTuple

import requests

logger = logging.getLogger(__name__)


class StructuredChatResult(NamedTuple):
    """Ergebnis eines strukturierten chat()-Aufrufs.

    Felder:
        text: Der Text-Inhalt der LLM-Antwort.
        response_format_used: True, wenn das response_format vom Provider
            akzeptiert wurde. False, wenn der Provider 400/4xx zurückgegeben
            hat und der Aufruf ohne response_format wiederholt wurde (Fallback).
    """

    text: str
    response_format_used: bool


# ---------------------------------------------------------------------------
# Umgebungsvariablen
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://ollama.com/v1"
DEFAULT_MODEL = "glm-5.3-flash"
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
        LLM_MODEL     (default: glm-5.3-flash)
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
        self.last_usage: dict | None = None
        self.total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        as_structured: bool = False,
    ) -> str | StructuredChatResult:
        """Sendet Messages an /chat/completions und gibt den Text zurück.

        Retry bei 5xx und 429 (max. MAX_RETRIES Mal mit Backoff).
        Wenn ein Fallback-Modell konfiguriert ist und der primäre Request nach
        allen Retries mit 429/5xx fehlschlägt, wird ein Versuch mit dem
        Fallback-Modell unternommen.

        Wenn ``response_format`` gesetzt ist, wird es in den Payload als
        ``"response_format"`` aufgenommen (OpenAI-kompatibel). Antwortet die
        API mit 400/4xx (invalid response_format), wird EINMALIG ohne
        response_format wiederholt und ``response_format_used=False`` gesetzt.

        Wenn ``as_structured=True`` und ``response_format`` gesetzt ist, wird
        ein :class:`StructuredChatResult` zurückgegeben. Sonst immer ``str``
        (Rückwärtskompatibilität).
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
        if response_format is not None:
            payload["response_format"] = response_format

        self.last_usage = None

        try:
            text, response_format_used, usage = self._send_with_retries(
                url, headers, payload, response_format=response_format,
            )
        except _RetryableHTTPError as exc:
            # Primärer Request nach allen Retries mit 429/5xx fehlgeschlagen.
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
                    text, response_format_used, usage = self._send_with_retries(
                        url, headers, fallback_payload,
                        is_fallback=True,
                        response_format=response_format,
                    )
                except _RetryableHTTPError as fb_exc:
                    raise RuntimeError(
                        f"LLM-Anfrage fehlgeschlagen: Primärmodell ({exc}), "
                        f"Fallback-Modell '{self.fallback_model}' ({fb_exc})"
                    ) from None
            else:
                # Ohne Fallback-Modell: Original-Fehler weitergeben
                raise RuntimeError(
                    f"LLM-Anfrage fehlgeschlagen nach {MAX_RETRIES + 1} Versuchen: {exc}"
                ) from None

        # usage erfassen (last_usage + kumulativ)
        self.last_usage = usage
        if usage is not None:
            self.total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
            self.total_usage["completion_tokens"] += usage.get("completion_tokens", 0) or 0
            self.total_usage["total_tokens"] += usage.get("total_tokens", 0) or 0

        if as_structured and response_format is not None:
            return StructuredChatResult(text=text, response_format_used=response_format_used)
        return text

    def _send_with_retries(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        is_fallback: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, bool, dict | None]:
        """Sendet den Request mit Retries bei 429/5xx.

        Löst _RetryableHTTPError aus, wenn alle Retries mit 429/5xx fehlschlagen
        (damit der Aufrufer einen Modell-Fallback durchführen kann).
        Andere Fehler (Timeout, ConnectionError) werden als RuntimeError geworfen.

        Wenn ``response_format`` gesetzt ist und die API mit 400/4xx antwortet
        (nicht 429/5xx), wird EINMALIG ohne response_format wiederholt und
        ``response_format_used=False`` zurückgegeben. Bei 429/5xx verhält sich
        die Methode wie bisher (retry + ggf. Fallback-Modell).

        Returns:
            Tuple (text, response_format_used, usage). response_format_used ist
            True, wenn response_format erfolgreich gesendet wurde (oder None war).
            usage ist das usage-dict aus der API-Antwort oder None.
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

                # 400/4xx-Fallback: wenn response_format im Payload und die API
                # 400/4xx (aber NICHT 429/5xx) zurückgibt → einmalig ohne
                # response_format wiederholen, response_format_used=False melden.
                if (
                    response_format is not None
                    and 400 <= resp.status_code < 500
                ):
                    logger.info(
                        "Provider lehnt response_format ab (HTTP %d) — "
                        "Fallback ohne response_format, Versuch %d/%d",
                        resp.status_code, attempt + 1, MAX_RETRIES + 1,
                    )
                    fallback_payload = dict(payload)
                    fallback_payload.pop("response_format", None)
                    resp2 = requests.post(
                        url, json=fallback_payload, headers=headers, timeout=TIMEOUT_SECONDS,
                    )
                    resp2.raise_for_status()
                    data2 = resp2.json()
                    content2 = data2["choices"][0]["message"]["content"]
                    usage2 = data2.get("usage")
                    return (content2 if content2 is not None else "", False, usage2)

                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage")
                return (content if content is not None else "", True, usage)

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
