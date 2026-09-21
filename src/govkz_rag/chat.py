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


class OllamaChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 240.0,
        temperature: float = 0.1,
        max_tokens: int = 512,
        reasoning_effort: str = "none",
        keep_alive: str | int = "5m",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._keep_alive = keep_alive

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
            "model": self._model,
            "messages": list(messages),
            "stream": False,
            "think": _ollama_think(self._reasoning_effort, think),
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": self._temperature if temperature is None else temperature,
                "num_predict": self._max_tokens if max_tokens is None else max_tokens,
            },
        }
        if response_format is not None:
            payload["format"] = _ollama_response_format(response_format)

        data = await self._post(payload)
        try:
            return str(data["message"]["content"]).strip()
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Ollama returned an invalid chat response") from exc

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._base_url + "/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}. Start Ollama and retry."
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama chat request timed out after {self._timeout:g} seconds"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise RuntimeError(
                f"Ollama chat request failed ({exc.response.status_code}): {detail}"
            ) from exc
        finally:
            LOGGER.info(
                "latency stage=ollama_chat duration_ms=%.0f",
                (perf_counter() - started) * 1_000,
            )
        if not isinstance(data, dict):
            raise RuntimeError("Ollama returned a non-object chat response")
        return data


class OpenAIChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        timeout_seconds: float = 120.0,
        temperature: float = 0.1,
        max_tokens: int = 512,
        reasoning_effort: str = "none",
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("API key must not be empty")
        self._base_url = base_url.rstrip("/")
        self._model = model
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
            "model": self._model,
            "messages": list(messages),
            "stream": False,
            "temperature": self._temperature if temperature is None else temperature,
            "max_completion_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "reasoning_effort": (
                "none" if think is False else self._reasoning_effort
            ),
        }
        if response_format is not None:
            payload["response_format"] = _openai_response_format(response_format)

        data = await self._post(payload)
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "The OpenAI-compatible API returned an invalid chat response"
            ) from exc

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
                "Cannot connect to the OpenAI-compatible API. Check its base URL "
                "and ensure CA_BUNDLE points to the corporate proxy certificate "
                "when one is required."
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"OpenAI-compatible request timed out after {self._timeout:g} seconds"
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                raise RuntimeError(
                    "External model rate limit exceeded. Retry later or reduce "
                    "generation.max_tokens."
                ) from exc
            if exc.response.status_code in {401, 403}:
                raise RuntimeError(
                    "External model authentication failed. Check API_KEY."
                ) from exc
            detail = exc.response.text[:500]
            raise RuntimeError(
                "OpenAI-compatible request failed "
                f"({exc.response.status_code}): {detail}"
            ) from exc
        finally:
            LOGGER.info(
                "latency stage=openai_chat duration_ms=%.0f",
                (perf_counter() - started) * 1_000,
            )

        if not isinstance(data, dict):
            raise RuntimeError(
                "The OpenAI-compatible API returned a non-object response"
            )
        return data


def _ollama_think(
    reasoning_effort: str,
    think: bool | None,
) -> bool | str:
    if think is not None:
        return think
    if reasoning_effort == "none":
        return False
    return reasoning_effort


def _ollama_response_format(schema: dict[str, Any] | str) -> dict[str, Any] | str:
    if isinstance(schema, str):
        return "json"
    return schema


def _openai_response_format(schema: dict[str, Any] | str) -> dict[str, Any]:
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
