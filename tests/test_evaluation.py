import asyncio

import pytest

from govkz_rag.evaluation import (
    EvalCase,
    evaluate_retrieval,
    format_evaluation_tables,
)
from govkz_rag.retrieval.models import RetrievalHit, ServiceDocument


def _document(document_id: int) -> ServiceDocument:
    return ServiceDocument(
        id=document_id,
        source_ids=(document_id,),
        name=f"Услуга {document_id}",
        answer=f"Ответ {document_id}",
        chunks=f"Материал {document_id}",
        alternative_queries=(f"запрос {document_id}",),
    )


def _case(eval_id: str, relevant_id: int) -> EvalCase:
    return EvalCase(
        eval_id=eval_id,
        category="Тест",
        question=f"Вопрос {eval_id}",
        relevant_document_ids=(relevant_id,),
    )


class FakeRetriever:
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        relevance_query: str | None = None,
    ) -> list[RetrievalHit]:
        document_ids = [30, 20, 10] if query.endswith("1") else [30, 20, 10]
        return [
            RetrievalHit(
                document=_document(document_id),
                score=1.0,
                semantic_similarity=0.9,
            )
            for document_id in document_ids[:k]
        ]


def test_retrieval_metrics_and_table_formatting() -> None:
    results, summary = asyncio.run(
        evaluate_retrieval(
            FakeRetriever(),
            [_case("case-1", 20), _case("case-2", 99)],
            k=3,
        )
    )

    assert results[0].first_relevant_rank == 2
    assert results[1].first_relevant_rank is None
    assert summary.hit_at_1 == 0.0
    assert summary.recall_at_k == 0.5
    assert summary.precision_at_k == pytest.approx(1 / 6)
    assert summary.mrr_at_k == 0.25
    assert summary.ndcg_at_k == pytest.approx(0.3154648768)

    table = format_evaluation_tables(results, summary)
    assert "Per-question results" in table
    assert "Recall@3" in table
    assert "50.0%" in table
