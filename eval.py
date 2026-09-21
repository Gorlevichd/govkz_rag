from __future__ import annotations

import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from govkz_rag.config import Settings
from govkz_rag.evaluation import (
    evaluate_retrieval,
    format_evaluation_tables,
    load_eval_cases,
)
from govkz_rag.ollama import OLLAMA_BASE_URL, OllamaEmbeddingClient
from govkz_rag.retrieval import ChromaRetriever, CrossEncoderReranker


async def run() -> None:
    settings = Settings.from_yaml(PROJECT_ROOT / "config.yaml")
    client = OllamaEmbeddingClient(
        OLLAMA_BASE_URL,
        settings.retrieval.embedding.model,
        timeout_seconds=settings.retrieval.embedding.request_timeout_seconds,
        embedding_keep_alive=settings.retrieval.index.keep_alive,
    )
    reranker = None
    if settings.retrieval.reranker.enabled:
        reranker = CrossEncoderReranker(
            settings.retrieval.reranker.model,
            min_score=settings.retrieval.reranker.min_score,
            max_length=settings.retrieval.reranker.max_length,
            max_passage_chars=settings.retrieval.reranker.max_passage_chars,
            batch_size=settings.retrieval.reranker.batch_size,
            device=settings.retrieval.reranker.device,
        )
    retriever = ChromaRetriever.from_path(
        settings.retrieval.chroma.path,
        settings.retrieval.chroma.collection,
        client,
        candidate_k=settings.retrieval.search.candidate_k,
        rrf_k=settings.retrieval.search.rrf_k,
        alias_candidate_multiplier=settings.retrieval.search.alias_candidate_multiplier,
        min_semantic_similarity=settings.retrieval.search.min_semantic_similarity,
        reranker=reranker,
        reranker_candidate_k=settings.retrieval.reranker.candidate_k,
    )
    cases = load_eval_cases(settings.retrieval.evaluation.dataset)

    def show_progress(current: int, total: int, eval_id: str) -> None:
        print(f"\rEvaluating {current}/{total}: {eval_id}", end="", flush=True)

    results, summary = await evaluate_retrieval(
        retriever,
        cases,
        settings.retrieval.search.top_k,
        progress=show_progress,
    )
    print("\n")
    print(format_evaluation_tables(results, summary))


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")
