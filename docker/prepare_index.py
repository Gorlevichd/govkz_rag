from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import chromadb
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from govkz_rag.config import Settings  # noqa: E402
from govkz_rag.ollama import OLLAMA_BASE_URL, OllamaEmbeddingClient  # noqa: E402
from govkz_rag.retrieval import load_documents, rebuild_collection  # noqa: E402
from govkz_rag.retrieval.loader import document_fingerprint  # noqa: E402
from govkz_rag.retrieval.models import ServiceDocument  # noqa: E402


def index_is_current(
    documents: list[ServiceDocument],
    settings: Settings,
) -> bool:
    client = chromadb.PersistentClient(path=str(settings.retrieval.chroma.path))
    collection_name = settings.retrieval.chroma.collection
    try:
        aliases = client.get_collection(collection_name, embedding_function=None)
        canonical = client.get_collection(
            f"{collection_name}_documents",
            embedding_function=None,
        )
    except Exception:
        return False

    fingerprint = document_fingerprint(documents)
    expected_alias_count = sum(
        len(document.alternative_queries) for document in documents
    )
    expected_metadata = {
        "embedding_model": settings.retrieval.embedding.model,
        "source_fingerprint": fingerprint,
        "index_complete": True,
        "canonical_document_count": len(documents),
        "alias_count": expected_alias_count,
    }
    return (
        all(
            all(
                (collection.metadata or {}).get(key) == value
                for key, value in expected_metadata.items()
            )
            for collection in (aliases, canonical)
        )
        and aliases.count() == expected_alias_count
        and canonical.count() == len(documents)
    )


async def prepare_index() -> None:
    settings = Settings.from_yaml(PROJECT_ROOT / "config.yaml")
    documents = load_documents(settings.retrieval.data.workbook)
    if index_is_current(documents, settings):
        print("Chroma index is current", flush=True)
        return

    embedding = settings.retrieval.embedding
    client = OllamaEmbeddingClient(
        OLLAMA_BASE_URL,
        embedding.model,
        timeout_seconds=embedding.request_timeout_seconds,
        embedding_keep_alive=settings.retrieval.index.keep_alive,
    )

    def show_progress(current: int, total: int) -> None:
        print(f"Indexing aliases: {current}/{total}", flush=True)

    count = await rebuild_collection(
        documents,
        client,
        settings.retrieval.chroma.path,
        settings.retrieval.chroma.collection,
        batch_size=settings.retrieval.index.batch_size,
        progress=show_progress,
    )
    print(f"Chroma index built: {count} canonical documents", flush=True)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    asyncio.run(prepare_index())


if __name__ == "__main__":
    main()
