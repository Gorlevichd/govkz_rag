from __future__ import annotations

from collections.abc import Sequence
import logging
from time import perf_counter
from typing import Any, Protocol

import httpx


LOGGER = logging.getLogger("uvicorn.error")
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class EmbeddingModel(Protocol):
    @property
    def embedding_model(self) -> str: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OllamaEmbeddingClient:
    def __init__(
        self,
        base_url: str,
        embedding_model: str,
        timeout_seconds: float = 180.0,
        embedding_keep_alive: str | int = 0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._embedding_model = embedding_model
        self._timeout = timeout_seconds
        self._embedding_keep_alive = embedding_keep_alive

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        data = await self._post(
            "/api/embed",
            {
                "model": self._embedding_model,
                "input": list(texts),
                "truncate": True,
                "keep_alive": self._embedding_keep_alive,
            },
        )
        try:
            embeddings = data["embeddings"]
            if len(embeddings) != len(texts):
                raise ValueError("embedding count mismatch")
            return [[float(value) for value in vector] for vector in embeddings]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Ollama returned an invalid embedding response") from exc

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._base_url + endpoint, json=payload)
                response.raise_for_status()
                result = response.json()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}. Start Ollama and retry."
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self._timeout:g} seconds"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise RuntimeError(
                f"Ollama request failed ({exc.response.status_code}): {detail}"
            ) from exc
        finally:
            LOGGER.info(
                "latency stage=ollama%s duration_ms=%.0f",
                endpoint.replace("/api/", "_"),
                (perf_counter() - started) * 1_000,
            )
        if not isinstance(result, dict):
            raise RuntimeError("Ollama returned a non-object response")
        prompt_duration = result.get("prompt_eval_duration")
        generation_duration = result.get("eval_duration")
        if isinstance(prompt_duration, int) or isinstance(generation_duration, int):
            LOGGER.info(
                "ollama_metrics endpoint=%s prompt_tokens=%s output_tokens=%s "
                "prompt_ms=%.0f generation_ms=%.0f",
                endpoint,
                result.get("prompt_eval_count", 0),
                result.get("eval_count", 0),
                (prompt_duration or 0) / 1_000_000,
                (generation_duration or 0) / 1_000_000,
            )
        return result
