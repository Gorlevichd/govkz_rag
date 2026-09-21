from __future__ import annotations

import re
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from govkz_rag.retrieval.models import RetrievalHit


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval_query: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=160),
        Field(
            description=(
                "Краткий поисковый запрос на русском языке: услуга, действие и "
                "важные условия. Без рассуждений и пояснений."
            )
        ),
    ]

    @field_validator("retrieval_query")
    @classmethod
    def ensure_concise_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        lowered = normalized.casefold()
        reasoning_markers = (
            "мне нужно",
            "сначала ",
            "вопрос пользователя",
            "нужно преобразовать",
            "поэтому ",
            "таким образом",
        )
        if len(normalized.split()) > 18 or any(
            marker in lowered for marker in reasoning_markers
        ):
            raise ValueError("retrieval_query must be a concise search query")
        return normalized


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000),
        Field(
            description=(
                "Краткий итоговый ответ на русском языке, основанный только на "
                "источниках и содержащий ссылки [1], [2]. Без рассуждений, "
                "списка источников и URL. Строго 1–2 коротких предложения, не "
                "более 350 символов, без вступления и второстепенных деталей."
            )
        ),
    ]

    @field_validator("answer")
    @classmethod
    def reject_reasoning_trace(cls, value: str) -> str:
        lowered = value.casefold()
        reasoning_markers = (
            "<think",
            "</think>",
            "мне нужно ответить",
            "нам нужно ответить",
            "пользователь спрашивает",
            "проанализируем ",
            "проанализирую ",
            "сначала определю",
            "ход рассуждений",
            "промежуточный вывод",
            "нужно сформулировать ответ",
            "we need to answer",
            "i need to answer",
            "the user asks",
            "let's analyze",
            "let us analyze",
        )
        if not re.search(r"[а-яё]", lowered) or any(
            marker in lowered for marker in reasoning_markers
        ):
            raise ValueError("answer must not contain a reasoning trace")
        return value.strip()


class AgentInput(TypedDict):
    question: str


class AgentOutput(TypedDict):
    answer: str
    answer_body: str
    sources: list[dict[str, Any]]
    useful_links: list[str]
    grounded: bool


class AgentState(TypedDict, total=False):
    question: str
    retrieval_query: str
    hits: list[RetrievalHit]
    answer_body: str
    answer: str
    sources: list[dict[str, Any]]
    useful_links: list[str]
    grounded: bool
