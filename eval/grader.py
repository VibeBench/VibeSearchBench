"""Grading backend: unified prompt->text interface for LLM-as-judge.

Backends:
  - ``"openai"`` : OpenAI-compatible /v1/chat/completions (vLLM, etc.)
  - ``"gemini"`` : Google Gemini REST API (:generateContent, non-streaming)

Usage::

    from eval.grader import create_grader

    grader = create_grader({"type": "openai", "base_url": ..., "api_key": ..., "model": ...})
    text = grader.complete("Grade this response ...")
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

import httpx

from agent.llm import get_sync_client

logger = logging.getLogger(__name__)


# ============================================================================
# Abstract interface
# ============================================================================


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (e.g. Qwen3.5 reasoning traces)."""
    return _THINK_RE.sub("", text).strip()


class GraderClient(ABC):
    """Prompt-in -> text-out.  One method, no streaming, no tools."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send *prompt* and return the response text.

        Raises on HTTP / API errors — callers are responsible for retry.
        """


# ============================================================================
# OpenAI-compatible backend (vLLM, OpenAI, etc.)
# ============================================================================


class OpenAIGrader(GraderClient):
    """Grade via ``/v1/chat/completions``."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 32768,
        max_retries: int = 16,
    ):
        self.client = get_sync_client(base_url, api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._max_retries = max_retries

    def complete(self, prompt: str) -> str:
        last_err = None
        for attempt in range(self._max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = resp.choices[0].message.content or ""
                return _strip_thinking(text)
            except Exception as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    delay = min(2 ** attempt, 30)
                    logger.warning(
                        "OpenAI grader attempt %d/%d failed: %s, retrying in %ds",
                        attempt + 1, self._max_retries, e, delay,
                    )
                    time.sleep(delay)
        raise last_err


# ============================================================================
# Gemini REST backend
# ============================================================================


class GeminiGrader(GraderClient):
    """Grade via Gemini ``generateContent`` (non-streaming, no tools).

    Expected *api_url* example::

        https://<host>/v1beta/models/gemini-2.5-flash:generateContent
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        temperature: float = 0.8,
        max_tokens: int = 64000,
        top_p: float = 0.6,
        max_retries: int = 16,
    ):
        self.api_url = api_url
        self.headers = {"api-key": api_key, "Content-Type": "application/json"}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._max_retries = max_retries
        self._client = httpx.Client(
            timeout=httpx.Timeout(900.0),
            headers=self.headers,
            follow_redirects=True,
        )

    def complete(self, prompt: str) -> str:
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "topP": self.top_p,
            },
        }
        last_err = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.post(self.api_url, json=payload)
                resp.raise_for_status()
                data = resp.json()

                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError(f"Gemini: empty candidates – {data}")

                parts = candidates[0].get("content", {}).get("parts", [])
                texts = [
                    p["text"] for p in parts
                    if "text" in p and p.get("thought") is None
                ]
                if not texts:
                    raise ValueError("Gemini: no text parts in response")
                return "".join(texts)
            except Exception as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    delay = 2 ** attempt
                    logger.warning("Gemini attempt %d/%d failed: %s, retrying in %ds",
                                   attempt + 1, self._max_retries, e, delay)
                    time.sleep(delay)
        raise last_err

    def close(self):
        if hasattr(self, "_client") and not self._client.is_closed:
            self._client.close()


# ============================================================================
# Factory
# ============================================================================


def create_grader(config: Dict[str, Any]) -> GraderClient:
    """Build a :class:`GraderClient` from a config dict.

    Required keys by *type*:

    ======== ==============================
    openai   base_url, api_key, model
    gemini   api_url, api_key
    ======== ==============================
    """
    gtype = config.get("type", "openai")
    if gtype == "gemini":
        return GeminiGrader(
            api_url=config["api_url"],
            api_key=config["api_key"],
            temperature=config.get("temperature", 0.8),
            max_tokens=config.get("max_tokens", 64000),
            top_p=config.get("top_p", 0.6),
            max_retries=config.get("max_retries", 16),
        )
    return OpenAIGrader(
        base_url=config.get("base_url", ""),
        api_key=config.get("api_key", "EMPTY"),
        model=config.get("model", "gpt-4.1-2025-04-14"),
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 32768),
        max_retries=config.get("max_retries", 16),
    )
