from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, StringConstraints


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ServiceDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: PositiveInt
    source_ids: tuple[PositiveInt, ...] = Field(min_length=1)
    name: NonEmptyText
    answer: NonEmptyText
    chunks: NonEmptyText
    alternative_queries: tuple[NonEmptyText, ...] = Field(min_length=1)
    instruction: NonEmptyText | None = None
    egov_link: NonEmptyText | None = None
    egov_kaz_link: NonEmptyText | None = None
    apply_button: NonEmptyText | None = None

    @property
    def search_text(self) -> str:
        parts = [self.name, self.name, *self.alternative_queries, self.answer]
        if self.instruction:
            parts.append(self.instruction)
        return "\n".join(parts)

    def evidence_text(self, max_chars: int) -> str:
        parts = [f"Название: {self.name}", f"Краткий ответ: {self.answer}"]
        if self.instruction:
            parts.append(f"Инструкция: {self.instruction}")
        if self.chunks:
            parts.append(f"Материал: {self.chunks}")
        text = "\n".join(parts)
        return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"

    def rerank_text(self, max_chars: int) -> str:
        parts = [f"Название: {self.name}", f"Краткий ответ: {self.answer}"]
        if self.instruction:
            parts.append(f"Инструкция: {self.instruction}")
        parts.append(f"Материал: {self.chunks}")
        text = "\n".join(parts)
        return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"

    @property
    def preferred_url(self) -> str | None:
        return self.egov_link or self.egov_kaz_link or self.apply_button


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    document: ServiceDocument
    score: float
    semantic_similarity: float
    lexical_coverage: float = 0.0
    rrf_score: float | None = None
    reranker_score: float | None = None

    @property
    def relevance(self) -> float:
        value = (
            self.reranker_score
            if self.reranker_score is not None
            else self.semantic_similarity
        )
        return max(0.0, min(1.0, value))
