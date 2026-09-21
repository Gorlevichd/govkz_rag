from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

from dotenv import load_dotenv
import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from govkz_rag.config import Settings  # noqa: E402
from govkz_rag.ollama import OLLAMA_BASE_URL  # noqa: E402


def desired_models(settings: Settings) -> tuple[str, ...]:
    models = [settings.retrieval.embedding.model]
    if settings.agent.qa.provider == "ollama":
        models.append(settings.agent.qa.model)
    return tuple(dict.fromkeys(models))


def _wait_for_ollama(client: httpx.Client, timeout_seconds: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            return
        except httpx.HTTPError:
            time.sleep(1)
    raise RuntimeError(f"Ollama did not become ready at {OLLAMA_BASE_URL}")


def _installed_models(client: httpx.Client) -> set[str]:
    response = client.get(f"{OLLAMA_BASE_URL}/api/tags")
    response.raise_for_status()
    payload: Any = response.json()
    return {
        str(model["name"])
        for model in payload.get("models", [])
        if isinstance(model, dict) and model.get("name")
    }


def _pull_model(client: httpx.Client, model: str) -> None:
    print(f"Pulling Ollama model: {model}", flush=True)
    with client.stream(
        "POST",
        f"{OLLAMA_BASE_URL}/api/pull",
        json={"model": model, "stream": True},
    ) as response:
        response.raise_for_status()
        last_status = ""
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if error := event.get("error"):
                raise RuntimeError(f"Ollama could not pull {model}: {error}")
            status = str(event.get("status", ""))
            if status and status != last_status:
                print(f"  {status}", flush=True)
                last_status = status


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = Settings.from_yaml(PROJECT_ROOT / "config.yaml")
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)
    with httpx.Client(timeout=timeout) as client:
        _wait_for_ollama(client)
        installed = _installed_models(client)
        for model in desired_models(settings):
            if model in installed:
                print(f"Ollama model is ready: {model}", flush=True)
                continue
            _pull_model(client, model)


if __name__ == "__main__":
    main()
