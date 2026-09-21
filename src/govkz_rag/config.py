from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class DataSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workbook: Path


class ChromaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    collection: str = Field(min_length=1)


class OllamaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_model: str = Field(min_length=1)
    request_timeout_seconds: float = Field(gt=0)
    embedding_keep_alive: str | int


class OllamaQaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama"]
    model: str = Field(min_length=1)
    request_timeout_seconds: float = Field(gt=0)
    reasoning_effort: str = Field(min_length=1)
    keep_alive: str | int = "5m"


class OpenAIQaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai"]
    model: str = Field(min_length=1)
    request_timeout_seconds: float = Field(gt=0)
    reasoning_effort: str = Field(min_length=1)


QaSettings = Annotated[
    OllamaQaSettings | OpenAIQaSettings,
    Field(discriminator="provider"),
]


class IndexSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: PositiveInt
    embedding_keep_alive: str | int


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: PositiveInt
    candidate_k: PositiveInt
    rrf_k: PositiveInt
    alias_candidate_multiplier: PositiveInt
    min_semantic_similarity: float = Field(ge=0, le=1)


class RerankerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    model: Path
    candidate_k: PositiveInt
    min_score: float = Field(ge=0, le=1)
    max_length: PositiveInt
    max_passage_chars: PositiveInt
    batch_size: PositiveInt
    device: str = Field(min_length=1)


class EvaluationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: Path


class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_context_chars: PositiveInt
    max_tokens: PositiveInt
    temperature: float = Field(ge=0, le=2)
    think: bool


class SummarizationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    max_tokens: PositiveInt
    temperature: float = Field(ge=0, le=2)
    think: bool


class ServerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65_535)
    warmup_embedding: bool


class FrontendSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_timeout_seconds: float = Field(gt=0)


class LangfuseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    run_name: str = Field(min_length=1)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataSettings
    chroma: ChromaSettings
    ollama: OllamaSettings
    qa: QaSettings
    index: IndexSettings
    retrieval: RetrievalSettings
    reranker: RerankerSettings
    evaluation: EvaluationSettings
    summarization: SummarizationSettings
    generation: GenerationSettings
    server: ServerSettings
    frontend: FrontendSettings
    langfuse: LangfuseSettings

    @classmethod
    def from_yaml(cls, path: Path = Path("config.yaml")) -> "Settings":
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        settings = cls.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        base = path.resolve().parent
        return settings.model_copy(
            update={
                "data": settings.data.model_copy(
                    update={"workbook": base / settings.data.workbook}
                ),
                "chroma": settings.chroma.model_copy(
                    update={"path": base / settings.chroma.path}
                ),
                "evaluation": settings.evaluation.model_copy(
                    update={"dataset": base / settings.evaluation.dataset}
                ),
                "reranker": settings.reranker.model_copy(
                    update={"model": base / settings.reranker.model}
                ),
            }
        )
