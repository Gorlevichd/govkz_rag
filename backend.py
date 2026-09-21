from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from time import perf_counter

import uvicorn
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from govkz_rag.api.app import create_app  # noqa: E402
from govkz_rag.api.runnable import warmup_embedding  # noqa: E402
from govkz_rag.config import Settings  # noqa: E402


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_utf8_console()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = Settings.from_yaml(PROJECT_ROOT / "config.yaml")
    if settings.app.server.warmup_embedding:
        started = perf_counter()
        asyncio.run(warmup_embedding(settings))
        print(
            "latency stage=embedding_warmup "
            f"duration_ms={(perf_counter() - started) * 1_000:.0f}",
            flush=True,
        )
    uvicorn.run(
        create_app(settings=settings),
        host=settings.app.server.host,
        port=settings.app.server.port,
    )


if __name__ == "__main__":
    main()
