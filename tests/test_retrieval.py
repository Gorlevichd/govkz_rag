import asyncio
from dataclasses import replace

import chromadb

from govkz_rag.retrieval.models import ServiceDocument
from govkz_rag.retrieval.chroma import (
    ChromaRetriever,
    _metadata,
    collection_exists,
    rebuild_collection,
)


class FakeEmbedder:
    embedding_model = "test-embedding"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "регистрация брака": [1.0, 0.0, 0.0],
            "заключить брак": [0.9, 0.1, 0.0],
            "получение паспорта": [0.0, 1.0, 0.0],
            "регистрация автомобиля": [0.0, 0.0, 1.0],
            # Semantic search alone favors the passport record. BM25 should
            # bring the explicitly matching marriage record to the top via RRF.
            "как заключить брак": [0.1, 0.9, 0.0],
        }
        return [vectors[text] for text in texts]


class FakeReranker:
    def __init__(self) -> None:
        self.query: str | None = None
        self.candidate_ids: list[int] = []

    async def rerank(
        self,
        query: str,
        hits: list,
        k: int,
    ) -> list:
        self.query = query
        self.candidate_ids = [hit.document.id for hit in hits]
        ranked = sorted(hits, key=lambda hit: hit.document.id == 2, reverse=True)
        return [
            replace(hit, score=0.9 - position / 10, reranker_score=0.9 - position / 10)
            for position, hit in enumerate(ranked[:k])
        ]


def _document(document_id: int, title: str) -> ServiceDocument:
    return ServiceDocument(
        id=document_id,
        source_ids=(document_id,),
        name=title,
        answer=f"Ответ: {title}",
        chunks=f"Материал: {title}",
        alternative_queries=(title.lower(),),
    )


def test_chroma_and_bm25_are_combined_with_rrf() -> None:
    client = chromadb.EphemeralClient()
    metadata = {
        "embedding_model": "test-embedding",
        "index_complete": True,
        "source_fingerprint": "test-fingerprint",
    }
    alias_collection = client.create_collection(
        "egov_test",
        embedding_function=None,
        metadata={
            **metadata,
            "index_kind": "query_aliases",
            "canonical_document_count": 3,
            "alias_count": 4,
        },
        configuration={"hnsw": {"space": "cosine"}},
    )
    document_collection = client.create_collection(
        "egov_test_documents",
        embedding_function=None,
        metadata={
            **metadata,
            "index_kind": "canonical_documents",
            "canonical_document_count": 3,
            "alias_count": 4,
        },
        configuration={"hnsw": {"space": "cosine"}},
    )
    documents = [
        _document(1, "Регистрация брака").model_copy(
            update={
                "alternative_queries": (
                    "регистрация брака",
                    "заключить брак",
                )
            }
        ),
        _document(2, "Получение паспорта"),
        _document(3, "Регистрация автомобиля"),
    ]
    alias_collection.add(
        ids=["1:1", "1:2", "2:1", "3:1"],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        documents=[
            "регистрация брака",
            "заключить брак",
            "получение паспорта",
            "регистрация автомобиля",
        ],
        metadatas=[
            {"canonical_id": 1},
            {"canonical_id": 1},
            {"canonical_id": 2},
            {"canonical_id": 3},
        ],
    )
    document_collection.add(
        ids=["1", "2", "3"],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        documents=[document.chunks for document in documents],
        metadatas=[_metadata(document) for document in documents],
    )
    retriever = ChromaRetriever(
        alias_collection,
        document_collection,
        FakeEmbedder(),
        candidate_k=4,
        rrf_k=60,
        alias_candidate_multiplier=1,
    )

    hits = asyncio.run(retriever.retrieve("как заключить брак", k=3))

    assert hits[0].document.id == 1
    assert hits[0].score > 0.9
    assert hits[0].semantic_similarity >= 0.0
    assert [hit.document.id for hit in hits].count(1) == 1
    assert hits[0].document.alternative_queries == (
        "регистрация брака",
        "заключить брак",
    )

    reranker = FakeReranker()
    reranking_retriever = ChromaRetriever(
        alias_collection,
        document_collection,
        FakeEmbedder(),
        candidate_k=4,
        rrf_k=60,
        alias_candidate_multiplier=1,
        reranker=reranker,
        reranker_candidate_k=3,
    )

    reranked = asyncio.run(
        reranking_retriever.retrieve(
            "как заключить брак",
            k=2,
            relevance_query="Исходный вопрос",
        )
    )

    assert reranker.query == "Исходный вопрос"
    assert len(reranker.candidate_ids) == 3
    assert reranked[0].document.id == 2
    assert reranked[0].reranker_score == 0.9
    assert reranked[0].rrf_score is not None

    gated_reranker = FakeReranker()
    gated_retriever = ChromaRetriever(
        alias_collection,
        document_collection,
        FakeEmbedder(),
        candidate_k=4,
        rrf_k=60,
        alias_candidate_multiplier=1,
        min_semantic_similarity=0.5,
        reranker=gated_reranker,
        reranker_candidate_k=3,
    )

    asyncio.run(gated_retriever.retrieve("как заключить брак", k=3))

    assert gated_reranker.candidate_ids == [2]


def test_collection_exists_checks_named_persistent_collection(tmp_path) -> None:
    path = tmp_path / "chroma"

    assert collection_exists(path, "egov_test") is False

    client = chromadb.PersistentClient(path=str(path))
    client.create_collection("egov_test", embedding_function=None)

    assert collection_exists(path, "egov_test") is True


def test_rebuild_stores_aliases_and_canonical_documents_separately(tmp_path) -> None:
    documents = [
        _document(1, "Регистрация брака").model_copy(
            update={"alternative_queries": ("регистрация брака", "заключить брак")}
        ),
        _document(2, "Получение паспорта"),
    ]

    count = asyncio.run(
        rebuild_collection(
            documents,
            FakeEmbedder(),
            tmp_path / "chroma",
            "egov_test",
            batch_size=2,
        )
    )

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    aliases = client.get_collection("egov_test", embedding_function=None)
    canonical = client.get_collection("egov_test_documents", embedding_function=None)
    assert count == 2
    assert aliases.count() == 3
    assert canonical.count() == 2
    assert aliases.metadata["index_kind"] == "query_aliases"
    assert canonical.metadata["index_kind"] == "canonical_documents"
