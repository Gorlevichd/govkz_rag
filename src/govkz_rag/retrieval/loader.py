from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError

from govkz_rag.retrieval.models import NonEmptyText, ServiceDocument


REQUIRED_COLUMNS = {
    "id",
    "name",
    "chunks",
    "for_embedding",
    "answer",
}


class SourceServiceRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: PositiveInt
    name: NonEmptyText
    answer: NonEmptyText
    chunks: NonEmptyText
    for_embedding: NonEmptyText
    instruction: NonEmptyText | None = None
    egov_link: NonEmptyText | None = None
    egov_kaz_link: NonEmptyText | None = None
    apply_button: NonEmptyText | None = None


def load_documents(path: Path) -> list[ServiceDocument]:
    if not path.exists():
        raise FileNotFoundError(f"eGov workbook not found: {path}")

    frame = pd.read_excel(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            "Workbook is missing required columns: " + ", ".join(sorted(missing))
        )
    if frame.empty:
        raise ValueError(f"Workbook contains no service records: {path}")

    frame = frame.replace(r"^\s*$", None, regex=True)
    frame = frame.astype(object).where(frame.notna(), None)
    records: list[SourceServiceRecord] = []
    seen_ids: set[int] = set()
    field_names = SourceServiceRecord.model_fields.keys()
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
        try:
            record = SourceServiceRecord.model_validate(
                {field_name: row.get(field_name) for field_name in field_names}
            )
        except ValidationError as exc:
            raise ValueError(
                f"Invalid service data in workbook row {row_number}: {exc}"
            ) from exc
        if record.id in seen_ids:
            raise ValueError(f"Duplicate id {record.id} in workbook row {row_number}")
        seen_ids.add(record.id)
        records.append(record)
    return _canonical_documents(records)


def document_fingerprint(documents: Iterable[ServiceDocument]) -> str:
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item.id):
        digest.update(str(document.id).encode("utf-8"))
        digest.update(b"\0")
        for value in (
            document.name,
            document.answer,
            document.chunks,
            document.instruction or "",
            document.egov_link or "",
            document.egov_kaz_link or "",
            document.apply_button or "",
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        for query in document.alternative_queries:
            digest.update(query.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _canonical_documents(records: list[SourceServiceRecord]) -> list[ServiceDocument]:
    groups: dict[tuple[str, ...], list[SourceServiceRecord]] = {}
    for record in sorted(records, key=lambda item: item.id):
        groups.setdefault(_service_key(record), []).append(record)

    documents: list[ServiceDocument] = []
    for group in groups.values():
        representative = group[0]
        queries: list[str] = []
        seen_queries: set[str] = set()
        for record in group:
            query_key = _normalized_query(record.for_embedding)
            if query_key not in seen_queries:
                seen_queries.add(query_key)
                queries.append(record.for_embedding)
        documents.append(
            ServiceDocument(
                id=representative.id,
                source_ids=tuple(record.id for record in group),
                name=representative.name,
                answer=representative.answer,
                chunks=representative.chunks,
                alternative_queries=tuple(queries),
                instruction=representative.instruction,
                egov_link=representative.egov_link,
                egov_kaz_link=representative.egov_kaz_link,
                apply_button=representative.apply_button,
            )
        )
    return documents


def _service_key(record: SourceServiceRecord) -> tuple[str, ...]:
    return (
        _normalized_text(record.name),
        _normalized_text(record.answer),
        _normalized_text(record.chunks),
        _normalized_text(record.instruction),
        _normalized_url(record.egov_link),
        _normalized_url(record.egov_kaz_link),
        _normalized_url(record.apply_button),
    )


def _normalized_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold().replace("\u00a0", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def _normalized_query(value: str) -> str:
    return re.sub(
        r"[^0-9a-zа-яёәіңғүұқөһ]+",
        " ",
        _normalized_text(value),
        flags=re.IGNORECASE,
    ).strip()


def _normalized_url(value: str | None) -> str:
    normalized = _normalized_text(value)
    if not normalized:
        return ""
    parts = urlsplit(normalized)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, parts.query, "")
    )
