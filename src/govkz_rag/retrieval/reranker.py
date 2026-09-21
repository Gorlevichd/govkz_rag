from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from govkz_rag.retrieval.models import RetrievalHit


class RerankerInferenceError(RuntimeError):
    pass


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        k: int,
    ) -> list[RetrievalHit]: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str | Path,
        *,
        min_score: float = 0.0,
        max_length: int = 512,
        max_passage_chars: int = 4_000,
        batch_size: int = 8,
        device: str = "cpu",
    ) -> None:
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        if max_length <= 0 or max_passage_chars <= 0 or batch_size <= 0:
            raise ValueError("reranker size parameters must be positive")
        try:
            from sentence_transformers import CrossEncoder
            from torch.nn import Sigmoid
        except ImportError as exc:
            raise RuntimeError(
                "The reranker requires sentence-transformers. "
                "Install dependencies from requirements.txt."
            ) from exc

        self._model: Any = CrossEncoder(
            str(model_name),
            device=device,
            max_length=max_length,
            activation_fn=Sigmoid(),
            local_files_only=True,
        )
        self._min_score = min_score
        self._max_passage_chars = max_passage_chars
        self._batch_size = batch_size
        self._inference_lock = Lock()

    async def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        k: int,
    ) -> list[RetrievalHit]:
        if not query.strip() or not hits or k <= 0:
            return []
        pairs = [
            (query, hit.document.rerank_text(self._max_passage_chars))
            for hit in hits
        ]
        try:
            scores = await asyncio.to_thread(self._predict, pairs)
        except Exception as exc:
            raise RerankerInferenceError("Local reranker inference failed") from exc
        if len(scores) != len(hits):
            raise RuntimeError("Reranker returned an unexpected number of scores")
        ranked = sorted(
            (
                replace(hit, score=score, reranker_score=score)
                for hit, score in zip(hits, scores, strict=True)
                if score >= self._min_score
            ),
            key=lambda hit: hit.reranker_score or 0.0,
            reverse=True,
        )
        return ranked[:k]

    def _predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        with self._inference_lock:
            raw_scores = self._model.predict(
                list(pairs),
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [float(score) for score in raw_scores]
