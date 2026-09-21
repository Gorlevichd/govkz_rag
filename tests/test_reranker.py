import asyncio

import sentence_transformers

from govkz_rag.retrieval.models import RetrievalHit, ServiceDocument
from govkz_rag.retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self, *args, **kwargs) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs, **kwargs):
        self.pairs = list(pairs)
        return [0.2, 0.9, 0.6]


def _hit(document_id: int) -> RetrievalHit:
    document = ServiceDocument(
        id=document_id,
        source_ids=(document_id,),
        name=f"Услуга {document_id}",
        answer=f"Ответ {document_id}",
        chunks=f"Материал {document_id}",
        alternative_queries=(f"запрос {document_id}",),
    )
    return RetrievalHit(
        document=document,
        score=0.5,
        semantic_similarity=0.8,
        rrf_score=0.5,
    )


def test_cross_encoder_reranks_and_applies_threshold(monkeypatch) -> None:
    fake_model = FakeCrossEncoder()
    model_options: dict[str, object] = {}

    def build_model(*args, **kwargs):
        model_options.update(kwargs)
        return fake_model

    monkeypatch.setattr(
        sentence_transformers,
        "CrossEncoder",
        build_model,
    )
    reranker = CrossEncoderReranker(
        "test-model",
        min_score=0.5,
        batch_size=3,
    )

    result = asyncio.run(
        reranker.rerank("исходный вопрос", [_hit(1), _hit(2), _hit(3)], k=3)
    )

    assert [hit.document.id for hit in result] == [2, 3]
    assert [hit.reranker_score for hit in result] == [0.9, 0.6]
    assert result[0].rrf_score == 0.5
    assert model_options["local_files_only"] is True
    assert fake_model.pairs[0][0] == "исходный вопрос"
    assert "Название: Услуга 1" in fake_model.pairs[0][1]
