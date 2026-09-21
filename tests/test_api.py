from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from langchain_core.runnables import RunnableLambda

from govkz_rag.api import create_app
from govkz_rag.api import app as app_module
from govkz_rag.api import runnable as runnable_module
from govkz_rag.api.schemas import AnswerResponse, QuestionRequest
from govkz_rag.agent import build_agent
from govkz_rag.retrieval.models import RetrievalHit, ServiceDocument


async def fake_answer(payload: dict[str, str]) -> str:
    return f"Ответ на: {payload['question']}"


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    runnable = RunnableLambda(fake_answer).with_types(
        input_type=QuestionRequest,
        output_type=str,
    )
    app = create_app(runnable=runnable)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as test_client:
        yield test_client


@pytest.mark.asyncio
async def test_question_invoke_returns_grounded_answer(client: AsyncClient) -> None:
    response = await client.post(
        "/invoke",
        json={"input": {"question": "Как зарегистрировать брак?"}},
    )

    assert response.status_code == 200
    assert response.json()["output"].startswith("Ответ на:")


@pytest.mark.asyncio
async def test_question_invoke_rejects_blank_question(client: AsyncClient) -> None:
    response = await client.post(
        "/invoke",
        json={"input": {"question": "   "}},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_question_schema_endpoint(client: AsyncClient) -> None:
    response = await client.get("/input_schema")

    assert response.status_code == 200
    assert "question" in response.json()["properties"]


@pytest.mark.asyncio
async def test_default_app_exposes_structured_ask_endpoint(monkeypatch) -> None:
    text_runnable = RunnableLambda(fake_answer).with_types(
        input_type=QuestionRequest,
        output_type=str,
    )
    structured_runnable = RunnableLambda(
        lambda payload: {
            "answer": "Ответ [1].",
            "sources": [{"number": 1, "title": "Услуга", "url": None}],
            "useful_links": [],
            "grounded": True,
        }
    ).with_types(input_type=QuestionRequest, output_type=AnswerResponse)
    monkeypatch.setattr(
        app_module,
        "build_question_runnables",
        lambda settings: (text_runnable, structured_runnable),
    )
    app = app_module.create_app(settings=object())

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as test_client:
        response = await test_client.post(
            "/ask/invoke",
            json={"input": {"question": "Вопрос"}},
        )

    assert response.status_code == 200
    assert response.json()["output"]["answer"] == "Ответ [1]."


@pytest.mark.asyncio
async def test_stream_log_endpoint_is_registered(client: AsyncClient) -> None:
    response = await client.post(
        "/stream_log",
        json={"input": {"question": "Как зарегистрировать брак?"}},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


@pytest.mark.asyncio
async def test_public_runnable_returns_only_final_answer(monkeypatch) -> None:
    settings = SimpleNamespace(
        ollama=SimpleNamespace(
            base_url="http://localhost:11434",
            embedding_model="embedding",
            request_timeout_seconds=30,
            embedding_keep_alive=0,
        ),
        groq=SimpleNamespace(
            base_url="https://api.groq.com/openai/v1",
            chat_model="qwen/qwen3.8-27b",
            api_key_env="GROQ_KEY",
            ca_bundle_env="GROQ_CA_BUNDLE",
            request_timeout_seconds=30,
            reasoning_effort="none",
        ),
        generation=SimpleNamespace(
            temperature=0.1,
            max_context_chars=1000,
            max_tokens=128,
            think=False,
        ),
        chroma=SimpleNamespace(path="unused", collection="test"),
        retrieval=SimpleNamespace(
            candidate_k=5,
            rrf_k=60,
            alias_candidate_multiplier=4,
            top_k=3,
            min_semantic_similarity=0.6,
        ),
        reranker=SimpleNamespace(
            enabled=False,
            model="unused",
            candidate_k=10,
            min_score=0.0,
            max_length=512,
            max_passage_chars=4000,
            batch_size=8,
            device="cpu",
        ),
        summarization=SimpleNamespace(
            enabled=False,
            max_tokens=64,
            temperature=0.0,
            think=False,
        ),
        langfuse=SimpleNamespace(enabled=False, run_name="test-question"),
    )
    monkeypatch.setattr(
        runnable_module.ChromaRetriever,
        "from_path",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        runnable_module,
        "build_agent",
        lambda *args, **kwargs: RunnableLambda(
            lambda payload: {
                "answer": "Финальный ответ",
                "sources": [{"id": 1}],
                "grounded": True,
            }
        ),
    )
    monkeypatch.setenv("GROQ_KEY", "test-key")

    runnable = runnable_module.build_question_runnable(settings)
    result = await runnable.ainvoke({"question": "Вопрос"})

    assert result == "Финальный ответ"


@pytest.mark.asyncio
async def test_final_answer_stream_waits_for_completed_graph() -> None:
    document = ServiceDocument(
        id=1,
        source_ids=(1,),
        name="Регистрация брака",
        answer="Подайте заявление.",
        chunks="Заявление подается через портал.",
        alternative_queries=("регистрация брака",),
    )

    class Retriever:
        async def retrieve(
            self,
            query: str,
            k: int = 5,
            *,
            relevance_query: str | None = None,
        ) -> list[RetrievalHit]:
            return [
                RetrievalHit(
                    document=document,
                    score=0.9,
                    semantic_similarity=0.9,
                )
            ]

    class ChatModel:
        async def chat(self, messages, **kwargs) -> str:
            if "поисковые запросы" in messages[0]["content"]:
                return '{"retrieval_query":"подача заявления"}'
            return '{"answer":"Готовый ответ [1]."}'

    agent = build_agent(Retriever(), ChatModel())
    runnable = runnable_module.final_answer_runnable(agent)
    chunks = [
        chunk
        async for chunk in runnable.astream({"question": "Как подать заявление?"})
    ]

    assert chunks == [
        "Готовый ответ [1].\n\nИсточники:\n\n[1] - Регистрация брака"
    ]


@pytest.mark.asyncio
async def test_structured_runnable_keeps_answer_sections_separate() -> None:
    agent = RunnableLambda(
        lambda payload: {
            "answer": "Ответ [1].\n\nИсточники:\n\n[1] - Услуга",
            "answer_body": "Ответ [1].",
            "sources": [{"number": 1, "title": "Услуга", "url": None}],
            "useful_links": [],
            "grounded": True,
        }
    )

    runnable = runnable_module.structured_answer_runnable(agent)
    result = await runnable.ainvoke({"question": "Вопрос"})

    assert result == {
        "answer": "Ответ [1].",
        "sources": [{"number": 1, "title": "Услуга", "url": None}],
        "useful_links": [],
        "grounded": True,
    }
