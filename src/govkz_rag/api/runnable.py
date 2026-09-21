from __future__ import annotations

import logging
import os
from pathlib import Path
import ssl
from time import perf_counter
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
import truststore

from govkz_rag.agent import build_agent
from govkz_rag.chat import ChatModel, OllamaChatClient, OpenAIChatClient
from govkz_rag.config import Settings
from govkz_rag.observability import create_langfuse_handler
from govkz_rag.ollama import OLLAMA_BASE_URL, OllamaEmbeddingClient
from govkz_rag.retrieval import ChromaRetriever, CrossEncoderReranker
from govkz_rag.api.schemas import AnswerResponse, QuestionRequest


LOGGER = logging.getLogger("uvicorn.error")


def build_embedding_client(settings: Settings) -> OllamaEmbeddingClient:
    return OllamaEmbeddingClient(
        OLLAMA_BASE_URL,
        settings.retrieval.embedding.model,
        timeout_seconds=settings.retrieval.embedding.request_timeout_seconds,
        embedding_keep_alive=settings.retrieval.embedding.keep_alive,
    )


def build_chat_client(settings: Settings) -> ChatModel:
    if settings.agent.qa.provider == "ollama":
        return OllamaChatClient(
            OLLAMA_BASE_URL,
            settings.agent.qa.model,
            timeout_seconds=settings.agent.qa.request_timeout_seconds,
            temperature=settings.agent.generation.temperature,
            max_tokens=settings.agent.generation.max_tokens,
            reasoning_effort=settings.agent.qa.reasoning_effort,
            keep_alive=settings.agent.qa.keep_alive,
        )

    base_url = os.getenv("BASE_URL")
    if not base_url:
        raise ValueError("Required environment variable BASE_URL is not set")
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("Required environment variable API_KEY is not set")
    ca_bundle = os.getenv("CA_BUNDLE")
    ssl_context = _ssl_context(ca_bundle)
    return OpenAIChatClient(
        base_url,
        settings.agent.qa.model,
        api_key,
        timeout_seconds=settings.agent.qa.request_timeout_seconds,
        temperature=settings.agent.generation.temperature,
        max_tokens=settings.agent.generation.max_tokens,
        reasoning_effort=settings.agent.qa.reasoning_effort,
        ssl_context=ssl_context,
    )


def _ssl_context(ca_bundle: str | None) -> ssl.SSLContext:
    if ca_bundle:
        path = Path(ca_bundle).expanduser()
        if not path.is_file():
            raise ValueError(f"CA bundle from CA_BUNDLE does not exist: {path}")
        return ssl.create_default_context(cafile=str(path))
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


async def warmup_embedding(settings: Settings) -> None:
    started = perf_counter()
    client = build_embedding_client(settings)
    await client.embed(["государственные услуги Казахстана"])
    LOGGER.info(
        "latency stage=embedding_warmup duration_ms=%.0f",
        (perf_counter() - started) * 1_000,
    )


def final_answer_runnable(agent: Runnable[Any, dict[str, Any]]) -> Runnable:
    async def invoke_agent(
        payload: dict[str, str],
        config: RunnableConfig,
    ) -> str:
        result = await agent.ainvoke(payload, config=config)
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Agent completed without a final answer")
        return answer

    return RunnableLambda(invoke_agent)


def structured_answer_runnable(agent: Runnable[Any, dict[str, Any]]) -> Runnable:
    async def invoke_agent(
        payload: dict[str, str],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        result = await agent.ainvoke(payload, config=config)
        response = AnswerResponse(
            answer=result["answer_body"],
            sources=result["sources"],
            useful_links=result["useful_links"],
            grounded=result["grounded"],
        )
        return response.model_dump(mode="json")

    return RunnableLambda(invoke_agent)


def build_agent_runnable(settings: Settings) -> Runnable[Any, dict[str, Any]]:
    embedder = build_embedding_client(settings)
    chat_model = build_chat_client(settings)
    reranker = None
    if settings.retrieval.reranker.enabled:
        reranker = CrossEncoderReranker(
            settings.retrieval.reranker.model,
            min_score=settings.retrieval.reranker.min_score,
            max_length=settings.retrieval.reranker.max_length,
            max_passage_chars=settings.retrieval.reranker.max_passage_chars,
            batch_size=settings.retrieval.reranker.batch_size,
            device=settings.retrieval.reranker.device,
        )
    retriever = ChromaRetriever.from_path(
        settings.retrieval.chroma.path,
        settings.retrieval.chroma.collection,
        embedder,
        candidate_k=settings.retrieval.search.candidate_k,
        rrf_k=settings.retrieval.search.rrf_k,
        alias_candidate_multiplier=settings.retrieval.search.alias_candidate_multiplier,
        min_semantic_similarity=settings.retrieval.search.min_semantic_similarity,
        reranker=reranker,
        reranker_candidate_k=settings.retrieval.reranker.candidate_k,
    )
    agent = build_agent(
        retriever,
        chat_model,
        top_k=settings.retrieval.search.top_k,
        min_semantic_similarity=settings.retrieval.search.min_semantic_similarity,
        max_context_chars=settings.agent.generation.max_context_chars,
        summary_max_tokens=settings.agent.summarization.max_tokens,
        summary_temperature=settings.agent.summarization.temperature,
        summary_think=settings.agent.summarization.think,
        generation_think=settings.agent.generation.think,
        summarize_query=settings.agent.summarization.enabled,
    )
    return agent


def build_question_runnables(settings: Settings) -> tuple[Runnable, Runnable]:
    agent = build_agent_runnable(settings)
    final_answer = final_answer_runnable(agent)
    runnable = final_answer.with_types(input_type=QuestionRequest, output_type=str)
    structured = structured_answer_runnable(agent).with_types(
        input_type=QuestionRequest,
        output_type=AnswerResponse,
    )
    handler = create_langfuse_handler(settings.observability.langfuse.enabled)
    if handler is None:
        return runnable, structured
    config = {
        "callbacks": [handler],
        "run_name": settings.observability.langfuse.run_name,
    }
    return runnable.with_config(**config), structured.with_config(**config)


def build_question_runnable(settings: Settings) -> Runnable:
    runnable, _ = build_question_runnables(settings)
    return runnable
