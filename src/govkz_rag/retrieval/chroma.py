from __future__ import annotations

import asyncio
import heapq
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError
from rank_bm25 import BM25Okapi
import snowballstemmer

from govkz_rag.retrieval.loader import document_fingerprint
from govkz_rag.retrieval.models import RetrievalHit, ServiceDocument
from govkz_rag.retrieval.reranker import Reranker, RerankerInferenceError
from govkz_rag.ollama import EmbeddingModel


class Retriever(Protocol):
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        relevance_query: str | None = None,
    ) -> list[RetrievalHit]: ...


LOGGER = logging.getLogger(__name__)


class ChromaRetriever:
    def __init__(
        self,
        alias_collection: Collection,
        document_collection: Collection,
        embedder: EmbeddingModel,
        *,
        candidate_k: int = 30,
        rrf_k: int = 60,
        alias_candidate_multiplier: int = 4,
        min_semantic_similarity: float = 0.0,
        reranker: Reranker | None = None,
        reranker_candidate_k: int = 10,
    ) -> None:
        if (
            candidate_k <= 0
            or rrf_k <= 0
            or alias_candidate_multiplier <= 0
            or reranker_candidate_k <= 0
        ):
            raise ValueError("retrieval parameters must be positive")
        self._alias_collection = alias_collection
        self._embedder = embedder
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k
        self._alias_candidate_multiplier = alias_candidate_multiplier
        self._reranker = reranker
        self._reranker_candidate_k = reranker_candidate_k
        if not 0.0 <= min_semantic_similarity <= 1.0:
            raise ValueError("min_semantic_similarity must be between 0 and 1")
        self._min_semantic_similarity = min_semantic_similarity
        _validate_collection(alias_collection, embedder.embedding_model, "alias")
        _validate_collection(document_collection, embedder.embedding_model, "document")
        alias_metadata = alias_collection.metadata or {}
        document_metadata = document_collection.metadata or {}
        if (
            alias_metadata.get("source_fingerprint")
            != document_metadata.get("source_fingerprint")
            or alias_metadata.get("canonical_document_count")
            != document_collection.count()
            or alias_metadata.get("alias_count") != alias_collection.count()
        ):
            raise ValueError(
                "Chroma alias and canonical-document collections are inconsistent. "
                "Run `python create_index.py`."
            )

        stored_documents = document_collection.get(include=["documents", "metadatas"])
        document_texts = stored_documents["documents"] or []
        document_metadatas = stored_documents["metadatas"] or []
        self._records = {
            str(record_id): _service_document(record_id, document_texts[position], metadata)
            for position, (record_id, metadata) in enumerate(
                zip(stored_documents["ids"], document_metadatas, strict=True)
            )
        }

        stored_aliases = alias_collection.get(include=["documents", "metadatas"])
        alias_queries = stored_aliases["documents"] or []
        alias_metadatas = stored_aliases["metadatas"] or []
        self._alias_ids = stored_aliases["ids"]
        self._alias_to_canonical = {
            alias_id: str(metadata["canonical_id"])
            for alias_id, metadata in zip(
                self._alias_ids, alias_metadatas, strict=True
            )
        }
        missing_documents = set(self._alias_to_canonical.values()).difference(self._records)
        if missing_documents:
            raise ValueError(
                "Chroma alias index references missing canonical documents. "
                "Run `python create_index.py`."
            )
        self._tokenized_corpus = [_tokenize(query) for query in alias_queries]
        self._token_sets = [set(tokens) for tokens in self._tokenized_corpus]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    @classmethod
    def from_path(
        cls,
        path: Path,
        collection_name: str,
        embedder: EmbeddingModel,
        *,
        candidate_k: int = 30,
        rrf_k: int = 60,
        alias_candidate_multiplier: int = 4,
        min_semantic_similarity: float = 0.0,
        reranker: Reranker | None = None,
        reranker_candidate_k: int = 10,
    ) -> "ChromaRetriever":
        client = chromadb.PersistentClient(path=str(path))
        try:
            alias_collection = client.get_collection(
                collection_name,
                embedding_function=None,
            )
            document_collection = client.get_collection(
                _document_collection_name(collection_name),
                embedding_function=None,
            )
        except Exception as exc:
            raise ValueError(
                f"Chroma index '{collection_name}' is missing or uses the old format. "
                "Run `python create_index.py`."
            ) from exc
        return cls(
            alias_collection,
            document_collection,
            embedder,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            alias_candidate_multiplier=alias_candidate_multiplier,
            min_semantic_similarity=min_semantic_similarity,
            reranker=reranker,
            reranker_candidate_k=reranker_candidate_k,
        )

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        relevance_query: str | None = None,
    ) -> list[RetrievalHit]:
        if not query.strip() or k <= 0:
            return []
        candidate_count = min(
            max(self._candidate_k, k) * self._alias_candidate_multiplier,
            len(self._alias_ids),
        )
        query_embedding = (await self._embedder.embed([query]))[0]
        semantic_result = await asyncio.to_thread(
            self._alias_collection.query,
            query_embeddings=[query_embedding],
            n_results=candidate_count,
            include=["distances"],
        )
        semantic_ids = semantic_result["ids"][0]
        distances = semantic_result["distances"][0] if semantic_result["distances"] else []

        query_tokens = _tokenize(query)
        lexical_scores = self._bm25.get_scores(query_tokens)
        lexical_positions = heapq.nlargest(
            candidate_count,
            range(len(lexical_scores)),
            key=lexical_scores.__getitem__,
        )
        lexical_positions = [
            position for position in lexical_positions if lexical_scores[position] > 0
        ]

        semantic_best: dict[str, tuple[int, float]] = {}
        for rank, (alias_id, distance) in enumerate(
            zip(semantic_ids, distances, strict=True), start=1
        ):
            canonical_id = self._alias_to_canonical[alias_id]
            if canonical_id not in semantic_best:
                semantic_best[canonical_id] = (
                    rank,
                    max(0.0, 1.0 - float(distance)),
                )

        query_terms = set(query_tokens)
        lexical_best: dict[str, tuple[int, float]] = {}
        for rank, position in enumerate(lexical_positions, start=1):
            canonical_id = self._alias_to_canonical[self._alias_ids[position]]
            if canonical_id not in lexical_best:
                coverage = (
                    len(query_terms.intersection(self._token_sets[position]))
                    / len(query_terms)
                    if query_terms
                    else 0.0
                )
                lexical_best[canonical_id] = (rank, coverage)

        fused: dict[str, dict[str, float]] = {}
        for canonical_id, (rank, similarity) in semantic_best.items():
            fused[canonical_id] = {
                "rrf": 1.0 / (self._rrf_k + rank),
                "semantic": similarity,
                "coverage": 0.0,
            }
        for canonical_id, (rank, coverage) in lexical_best.items():
            item = fused.setdefault(
                canonical_id,
                {"rrf": 0.0, "semantic": 0.0, "coverage": 0.0},
            )
            item["rrf"] += 1.0 / (self._rrf_k + rank)
            item["coverage"] = coverage

        max_rrf = 2.0 / (self._rrf_k + 1)
        ranked = sorted(fused.items(), key=lambda item: item[1]["rrf"], reverse=True)
        accepted = [
            item
            for item in ranked
            if item[1]["semantic"] >= self._min_semantic_similarity
        ]
        hit_limit = max(k, self._reranker_candidate_k) if self._reranker else k
        hits = [
            RetrievalHit(
                document=self._records[record_id],
                score=values["rrf"] / max_rrf,
                semantic_similarity=values["semantic"],
                lexical_coverage=values["coverage"],
                rrf_score=values["rrf"] / max_rrf,
            )
            for record_id, values in accepted[:hit_limit]
        ]
        if self._reranker is None:
            return hits[:k]
        try:
            return await self._reranker.rerank(relevance_query or query, hits, k)
        except RerankerInferenceError:
            LOGGER.exception("Reranker failed; refusing to return unchecked evidence")
            return []


def collection_exists(path: Path, collection_name: str) -> bool:
    if not path.exists():
        return False
    client = chromadb.PersistentClient(path=str(path))
    for name in (collection_name, _document_collection_name(collection_name)):
        try:
            client.get_collection(name, embedding_function=None)
            return True
        except NotFoundError:
            continue
    return False


async def rebuild_collection(
    documents: list[ServiceDocument],
    embedder: EmbeddingModel,
    path: Path,
    collection_name: str,
    *,
    batch_size: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not documents:
        raise ValueError("Cannot build a Chroma index without documents")
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    document_collection_name = _document_collection_name(collection_name)
    for name in (collection_name, document_collection_name):
        try:
            client.delete_collection(name)
        except Exception:
            pass

    fingerprint = document_fingerprint(documents)
    alias_count = sum(len(document.alternative_queries) for document in documents)
    common_metadata = {
        "embedding_model": embedder.embedding_model,
        "source_fingerprint": fingerprint,
        "index_complete": False,
    }
    alias_collection = client.create_collection(
        name=collection_name,
        embedding_function=None,
        metadata={
            **common_metadata,
            "index_kind": "query_aliases",
            "canonical_document_count": len(documents),
            "alias_count": alias_count,
        },
        configuration={"hnsw": {"space": "cosine"}},
    )
    document_collection = client.create_collection(
        name=document_collection_name,
        embedding_function=None,
        metadata={
            **common_metadata,
            "index_kind": "canonical_documents",
            "canonical_document_count": len(documents),
            "alias_count": alias_count,
        },
        configuration={"hnsw": {"space": "cosine"}},
    )

    aliases = [
        (f"{document.id}:{position}", document.id, query)
        for document in documents
        for position, query in enumerate(document.alternative_queries, start=1)
    ]
    canonical_embeddings: dict[int, list[float]] = {}
    for start in range(0, alias_count, batch_size):
        batch = aliases[start : start + batch_size]
        embeddings = await embedder.embed([query for _, _, query in batch])
        for (_, canonical_id, _), embedding in zip(batch, embeddings, strict=True):
            canonical_embeddings.setdefault(canonical_id, embedding)
        await asyncio.to_thread(
            alias_collection.upsert,
            ids=[alias_id for alias_id, _, _ in batch],
            embeddings=embeddings,
            documents=[query for _, _, query in batch],
            metadatas=[{"canonical_id": canonical_id} for _, canonical_id, _ in batch],
        )
        if progress:
            progress(min(start + batch_size, alias_count), alias_count)

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        await asyncio.to_thread(
            document_collection.upsert,
            ids=[str(document.id) for document in batch],
            embeddings=[canonical_embeddings[document.id] for document in batch],
            documents=[document.chunks for document in batch],
            metadatas=[_metadata(document) for document in batch],
        )

    completed_metadata = {
        "embedding_model": embedder.embedding_model,
        "source_fingerprint": fingerprint,
        "index_complete": True,
        "canonical_document_count": len(documents),
        "alias_count": alias_count,
    }
    alias_collection.modify(
        metadata={
            **completed_metadata,
            "index_kind": "query_aliases",
        }
    )
    document_collection.modify(
        metadata={
            **completed_metadata,
            "index_kind": "canonical_documents",
        }
    )
    return document_collection.count()


def _metadata(document: ServiceDocument) -> dict[str, str | int]:
    return {
        "source_id": document.id,
        "source_ids": json.dumps(document.source_ids),
        "name": document.name,
        "answer": document.answer,
        "alternative_queries": json.dumps(
            document.alternative_queries,
            ensure_ascii=False,
        ),
        "instruction": document.instruction or "",
        "egov_link": document.egov_link or "",
        "egov_kaz_link": document.egov_kaz_link or "",
        "apply_button": document.apply_button or "",
    }


def _service_document(
    record_id: str,
    chunks: str,
    metadata: dict[str, str | int | float | bool],
) -> ServiceDocument:
    return ServiceDocument.model_validate(
        {
            "id": metadata.get("source_id", record_id),
            "source_ids": json.loads(str(metadata["source_ids"])),
            "name": metadata["name"],
            "answer": metadata["answer"],
            "chunks": chunks,
            "alternative_queries": json.loads(str(metadata["alternative_queries"])),
            "instruction": metadata.get("instruction") or None,
            "egov_link": metadata.get("egov_link") or None,
            "egov_kaz_link": metadata.get("egov_kaz_link") or None,
            "apply_button": metadata.get("apply_button") or None,
        }
    )


def _document_collection_name(collection_name: str) -> str:
    return f"{collection_name}_documents"


def _validate_collection(
    collection: Collection,
    embedding_model: str,
    kind: str,
) -> None:
    metadata = collection.metadata or {}
    if metadata.get("embedding_model") != embedding_model:
        raise ValueError(
            "Chroma collection is missing or uses a different embedding model. "
            "Run `python create_index.py`."
        )
    if metadata.get("index_complete") is not True:
        raise ValueError(
            "Chroma collection is incomplete. Run `python create_index.py` "
            "and let it finish."
        )
    expected = "query_aliases" if kind == "alias" else "canonical_documents"
    if metadata.get("index_kind") != expected:
        raise ValueError(
            "Chroma collection uses an incompatible index format. "
            "Run `python create_index.py`."
        )
    if collection.count() == 0:
        raise ValueError("Chroma collection is empty. Run `python create_index.py`.")


TOKEN_PATTERN = re.compile(r"[0-9a-zа-яёәіңғүұқөһ]+", re.IGNORECASE)
CYRILLIC_PATTERN = re.compile(r"[а-яё]", re.IGNORECASE)
RUSSIAN_STEMMER = snowballstemmer.stemmer("russian")
STOPWORDS = {
    "а",
    "в",
    "и",
    "или",
    "как",
    "мне",
    "мой",
    "моя",
    "на",
    "нужно",
    "о",
    "по",
    "при",
    "с",
    "что",
    "это",
    "the",
    "and",
    "for",
    "how",
    "what",
    "with",
}


def _tokenize(text: str) -> list[str]:
    tokens = [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if len(token) >= 2 and token.lower() not in STOPWORDS
    ]
    return [
        RUSSIAN_STEMMER.stemWord(token) if CYRILLIC_PATTERN.search(token) else token
        for token in tokens
    ]
