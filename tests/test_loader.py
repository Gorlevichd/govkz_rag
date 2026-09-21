from pathlib import Path

import pytest
from openpyxl import Workbook

from govkz_rag.retrieval.loader import load_documents


def _write_workbook(path: Path, rows: list[list[object]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(
        [
            "id",
            "name",
            "egov_link",
            "apply_button",
            "chunks",
            "instruction",
            "for_embedding",
            "egov_kaz_link",
            "answer",
            "answer_web",
        ]
    )
    for row in rows:
        sheet.append(row)
    workbook.save(path)


def test_load_documents_maps_expected_fields(tmp_path: Path) -> None:
    path = tmp_path / "egov.xlsx"
    _write_workbook(
        path,
        [
            [
                7,
                "Регистрация брака",
                "https://egov.kz/service/7",
                None,
                "Полный материал",
                "Подайте заявление",
                "регистрация брака",
                None,
                "Краткий ответ",
                "Краткий ответ",
            ]
        ],
    )

    documents = load_documents(path)

    assert len(documents) == 1
    assert documents[0].id == 7
    assert documents[0].source_ids == (7,)
    assert documents[0].alternative_queries == ("регистрация брака",)
    assert documents[0].preferred_url == "https://egov.kz/service/7"
    assert documents[0].instruction == "Подайте заявление"


def test_load_documents_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "egov.xlsx"
    row = [1, "A", None, None, "chunk", None, "embed", None, "answer", "answer"]
    _write_workbook(path, [row, row])

    with pytest.raises(ValueError, match="Duplicate id"):
        load_documents(path)


def test_load_documents_groups_service_queries_and_deduplicates_aliases(
    tmp_path: Path,
) -> None:
    path = tmp_path / "egov.xlsx"
    common = [
        "Регистрация брака",
        "https://egov.kz/service/7",
        None,
        "Полный материал",
        None,
    ]
    _write_workbook(
        path,
        [
            [1, *common, "Как зарегистрировать брак?", None, "Краткий ответ", "web"],
            [2, *common, "Можно ли подать заявление онлайн", None, "Краткий ответ", "web"],
            [3, *common, "как зарегистрировать брак", None, "Краткий ответ", "web"],
        ],
    )

    documents = load_documents(path)

    assert len(documents) == 1
    assert documents[0].id == 1
    assert documents[0].source_ids == (1, 2, 3)
    assert documents[0].alternative_queries == (
        "Как зарегистрировать брак?",
        "Можно ли подать заявление онлайн",
    )
