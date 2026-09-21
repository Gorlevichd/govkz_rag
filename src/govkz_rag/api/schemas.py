from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, StringConstraints


QuestionText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: QuestionText


class AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[dict[str, Any]]
    useful_links: list[str]
    grounded: bool
