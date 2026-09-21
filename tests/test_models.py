import pytest
from pydantic import ValidationError

from govkz_rag.retrieval.models import ServiceDocument


def test_service_document_normalizes_and_validates_text() -> None:
    document = ServiceDocument(
        id=1,
        source_ids=(1,),
        name="  Название  ",
        answer=" Ответ ",
        chunks=" Материал ",
        alternative_queries=(" Поисковый текст ",),
    )

    assert document.name == "Название"
    assert document.answer == "Ответ"
    assert document.alternative_queries == ("Поисковый текст",)


def test_service_document_rejects_blank_required_text() -> None:
    with pytest.raises(ValidationError):
        ServiceDocument(
            id=1,
            source_ids=(1,),
            name="   ",
            answer="Ответ",
            chunks="Материал",
            alternative_queries=("Поисковый текст",),
        )
