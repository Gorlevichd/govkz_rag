from __future__ import annotations

from collections.abc import Sequence
import logging
import ssl
from time import perf_counter
from typing import Any, Protocol

import httpx


LOGGER = logging.getLogger("uvicorn.error")


class ChatModel(Protocol):
    async def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
        response_format: dict[str, Any] | str | None = None,
    ) -> str: ...


class GroqChatClient:
    def __init__(
        self,
        base_url: str,
        chat_model: str,
        api_key: str,
        *,
        timeout_seconds: float = 120.0,
        temperature: float = 0.1,
        max_tokens: int = 512,
        reasoning_effort: str = "none",
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Groq API key must not be empty")
        self._base_url = base_url.rstrip("/")
        self._chat_model = chat_model
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._ssl_context = ssl_context

    async def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
        response_format: dict[str, Any] | str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._chat_model,
            "messages": list(messages),
            "stream": False,
            "temperature": self._temperature if temperature is None else temperature,
            "max_completion_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "reasoning_effort": self._reasoning_effort,
            "include_reasoning": False,
        }
        if response_format is not None:
            payload["response_format"] = _groq_response_format(response_format)

        data = await self._post(payload)
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError("Groq returned an invalid chat response") from exc

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                verify=self._ssl_context if self._ssl_context is not None else True,
            ) as client:
                response = await client.post(
                    self._base_url + "/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                "Cannot connect to the Groq API. Ensure the corporate CA is "
                "trusted by the operating system or set GROQ_CA_BUNDLE to a "
                "PEM certificate bundle."
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Groq request timed out after {self._timeout:g} seconds"
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                raise RuntimeError(
                    "Groq rate limit exceeded. Retry after the limit resets or "
                    "reduce generation.max_tokens."
                ) from exc
            if exc.response.status_code in {401, 403}:
                raise RuntimeError(
                    "Groq authentication failed. Check GROQ_KEY and account access."
                ) from exc
            detail = exc.response.text[:500]
            raise RuntimeError(
                f"Groq request failed ({exc.response.status_code}): {detail}"
            ) from exc
        finally:
            LOGGER.info(
                "latency stage=groq_chat duration_ms=%.0f",
                (perf_counter() - started) * 1_000,
            )

        if not isinstance(data, dict):
            raise RuntimeError("Groq returned a non-object response")
        return data


def _groq_response_format(schema: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(schema, str):
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_response",
            "strict": True,
            "schema": schema,
        },
    }
