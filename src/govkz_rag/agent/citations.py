from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from govkz_rag.retrieval.models import RetrievalHit


CITATION_PATTERN = re.compile(r"\[(\d+)]")
APPENDED_SECTION_PATTERN = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:источники|полезные ссылки)\s*:\s*$"
)


def answer_body_only(answer: str) -> str:
    section = APPENDED_SECTION_PATTERN.search(answer)
    body = answer[: section.start()] if section else answer
    return body.strip()


def cited_numbers(answer: str, hit_count: int) -> list[int]:
    numbers = [int(match) for match in CITATION_PATTERN.findall(answer)]
    if not numbers or any(number < 1 or number > hit_count for number in numbers):
        return []
    return sorted(set(numbers))


def source_metadata(
    hits: list[RetrievalHit], numbers: list[int]
) -> list[dict[str, Any]]:
    return [
        {
            "number": number,
            "id": hits[number - 1].document.id,
            "source_ids": list(hits[number - 1].document.source_ids),
            "title": hits[number - 1].document.name,
            "answer": hits[number - 1].document.answer,
            "url": hits[number - 1].document.egov_link,
            "semantic_similarity": round(
                hits[number - 1].semantic_similarity,
                4,
            ),
            "lexical_coverage": round(hits[number - 1].lexical_coverage, 4),
            "rrf_score": (
                round(hits[number - 1].rrf_score, 4)
                if hits[number - 1].rrf_score is not None
                else None
            ),
            "reranker_score": (
                round(hits[number - 1].reranker_score, 4)
                if hits[number - 1].reranker_score is not None
                else None
            ),
        }
        for number in numbers
    ]


def format_final_answer(
    answer_body: str,
    hits: list[RetrievalHit],
    numbers: list[int],
) -> str:
    source_lines = [
        f"[{number}] - {_single_line(hits[number - 1].document.name)}"
        for number in numbers
    ]
    sections = [answer_body.strip(), "Источники:\n\n" + "\n".join(source_lines)]
    urls = deduplicated_urls(hits, numbers)
    if urls:
        sections.append("Полезные ссылки:\n\n" + "\n".join(f"- {url}" for url in urls))
    return "\n\n".join(sections)


def deduplicated_urls(hits: list[RetrievalHit], numbers: list[int]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for number in numbers:
        url = hits[number - 1].document.egov_link
        normalized = _normalized_url(url)
        if normalized is not None and normalized not in seen:
            seen.add(normalized)
            urls.append(url.strip())
    return urls


def _normalized_url(value: str | None) -> str | None:
    if not value:
        return None
    url = value.strip()
    parts = urlsplit(url)
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return None
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, parts.query, "")
    )


def _single_line(value: str) -> str:
    return " ".join(value.split())
