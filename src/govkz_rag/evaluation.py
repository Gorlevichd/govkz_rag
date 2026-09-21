from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, StringConstraints

from govkz_rag.retrieval import Retriever


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_id: NonEmptyText
    category: NonEmptyText
    question: NonEmptyText
    relevant_document_ids: tuple[PositiveInt, ...] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class RetrievalEvalResult:
    eval_id: str
    relevant_ids: tuple[int, ...]
    retrieved_ids: tuple[int, ...]
    first_relevant_rank: int | None
    hit_at_1: float
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    latency_ms: float


@dataclass(frozen=True, slots=True)
class RetrievalEvalSummary:
    question_count: int
    k: int
    hit_at_1: float
    recall_at_k: float
    precision_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    mean_latency_ms: float
    p95_latency_ms: float


def load_eval_cases(path: Path) -> list[EvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    cases: list[EvalCase] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(
                f"Invalid evaluation record at {path}:{line_number}"
            ) from exc
    if not cases:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    return cases


async def evaluate_retrieval(
    retriever: Retriever,
    cases: list[EvalCase],
    k: int,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[RetrievalEvalResult], RetrievalEvalSummary]:
    if k <= 0:
        raise ValueError("k must be positive")
    results: list[RetrievalEvalResult] = []
    for position, case in enumerate(cases, start=1):
        if progress:
            progress(position, len(cases), case.eval_id)
        started = perf_counter()
        hits = await retriever.retrieve(
            case.question,
            k,
            relevance_query=case.question,
        )
        latency_ms = (perf_counter() - started) * 1_000
        retrieved_ids = tuple(hit.document.id for hit in hits)
        relevant = set(case.relevant_document_ids)
        matched_ranks = [
            rank
            for rank, document_id in enumerate(retrieved_ids, start=1)
            if document_id in relevant
        ]
        first_rank = matched_ranks[0] if matched_ranks else None
        matched_count = len(matched_ranks)
        dcg = sum(1.0 / math.log2(rank + 1) for rank in matched_ranks)
        ideal_count = min(len(relevant), k)
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_count + 1)
        )
        results.append(
            RetrievalEvalResult(
                eval_id=case.eval_id,
                relevant_ids=case.relevant_document_ids,
                retrieved_ids=retrieved_ids,
                first_relevant_rank=first_rank,
                hit_at_1=float(first_rank == 1),
                recall_at_k=matched_count / len(relevant),
                precision_at_k=matched_count / k,
                reciprocal_rank=1.0 / first_rank if first_rank else 0.0,
                ndcg_at_k=dcg / ideal_dcg if ideal_dcg else 0.0,
                latency_ms=latency_ms,
            )
        )

    latencies = sorted(result.latency_ms for result in results)
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1)
    summary = RetrievalEvalSummary(
        question_count=len(results),
        k=k,
        hit_at_1=mean(result.hit_at_1 for result in results),
        recall_at_k=mean(result.recall_at_k for result in results),
        precision_at_k=mean(result.precision_at_k for result in results),
        mrr_at_k=mean(result.reciprocal_rank for result in results),
        ndcg_at_k=mean(result.ndcg_at_k for result in results),
        mean_latency_ms=mean(result.latency_ms for result in results),
        p95_latency_ms=latencies[p95_index],
    )
    return results, summary


def format_evaluation_tables(
    results: list[RetrievalEvalResult],
    summary: RetrievalEvalSummary,
) -> str:
    detail_rows = [
        [
            result.eval_id,
            _ids(result.relevant_ids),
            _ids(result.retrieved_ids),
            str(result.first_relevant_rank or "-") ,
            f"{result.recall_at_k:.2f}",
            f"{result.reciprocal_rank:.2f}",
            f"{result.ndcg_at_k:.2f}",
            f"{result.latency_ms:.0f}",
        ]
        for result in results
    ]
    details = _format_table(
        ["ID", "Relevant", "Retrieved", "Rank", f"R@{summary.k}", "RR", f"nDCG@{summary.k}", "ms"],
        detail_rows,
    )
    summary_rows = [
        ["Questions", str(summary.question_count)],
        ["Hit@1", _percent(summary.hit_at_1)],
        [f"Recall@{summary.k}", _percent(summary.recall_at_k)],
        [f"Precision@{summary.k}", _percent(summary.precision_at_k)],
        [f"MRR@{summary.k}", f"{summary.mrr_at_k:.3f}"],
        [f"nDCG@{summary.k}", f"{summary.ndcg_at_k:.3f}"],
        ["Mean latency", f"{summary.mean_latency_ms:.0f} ms"],
        ["P95 latency", f"{summary.p95_latency_ms:.0f} ms"],
    ]
    return f"Per-question results\n{details}\n\nSummary\n{_format_table(['Metric', 'Value'], summary_rows)}"


def _ids(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values) or "-"


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[column]) for column, value in enumerate(row)
        )

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])
