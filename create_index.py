from __future__ import annotations

import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from govkz_rag.config import Settings  # noqa: E402
from govkz_rag.ollama import OllamaEmbeddingClient  # noqa: E402
from govkz_rag.retrieval import (  # noqa: E402
    collection_exists,
    load_documents,
    rebuild_collection,
)


def _client(settings: Settings) -> OllamaEmbeddingClient:
    return OllamaEmbeddingClient(
        str(settings.ollama.base_url).rstrip("/"),
        settings.ollama.embedding_model,
        timeout_seconds=settings.ollama.request_timeout_seconds,
        embedding_keep_alive=settings.index.embedding_keep_alive,
    )


def _confirm_overwrite(settings: Settings) -> bool:
    try:
        response = input(
            f"Index '{settings.chroma.collection}' already exists at "
            f"{settings.chroma.path}. Overwrite it? [y/N]: "
        )
    except EOFError:
        return False
    return response.strip().lower() in {"y", "yes"}


async def _build_index(settings: Settings) -> None:
    documents = load_documents(settings.data.workbook)
    alias_count = sum(len(document.alternative_queries) for document in documents)

    def show_progress(completed: int, total: int) -> None:
        print(
            f"Embedding {completed}/{total} records ({completed / total:.1%})",
            flush=True,
        )

    count = await rebuild_collection(
        documents,
        _client(settings),
        settings.chroma.path,
        settings.chroma.collection,
        batch_size=settings.index.batch_size,
        progress=show_progress,
    )
    print(
        f"Indexed {count} canonical services from {alias_count} query aliases with "
        f"{settings.ollama.embedding_model} into "
        f"'{settings.chroma.collection}' at {settings.chroma.path}"
    )


def main() -> None:
    settings = Settings.from_yaml(PROJECT_ROOT / "config.yaml")
    if collection_exists(settings.chroma.path, settings.chroma.collection):
        if not _confirm_overwrite(settings):
            print("Index build cancelled. Existing index was not changed.")
            return
    try:
        asyncio.run(_build_index(settings))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
