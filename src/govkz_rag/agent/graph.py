from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any, Literal

from pydantic import ValidationError

from langgraph.graph import END, START, StateGraph

from govkz_rag.agent.citations import (
    answer_body_only,
    cited_numbers,
    deduplicated_urls,
    format_final_answer,
    source_metadata,
)
from govkz_rag.agent.prompts import (
    ANSWER_FEW_SHOTS,
    ANSWER_SYSTEM_PROMPT,
    SUMMARIZE_FEW_SHOTS,
    SUMMARIZE_SYSTEM_PROMPT,
)
from govkz_rag.agent.state import (
    AgentInput,
    AgentOutput,
    AgentState,
    GroundedAnswer,
    RetrievalQuery,
)
from govkz_rag.chat import ChatModel
from govkz_rag.retrieval import Retriever


LOGGER = logging.getLogger("uvicorn.error")


INSUFFICIENT_ANSWER = (
    "В доступных данных недостаточно надежной информации для ответа. "
    "Уточните государственную услугу, жизненную ситуацию или название документа."
)


def build_agent(
    retriever: Retriever,
    chat_model: ChatModel,
    *,
    top_k: int = 5,
    min_semantic_similarity: float = 0.60,
    max_context_chars: int = 18_000,
    summary_max_tokens: int = 64,
    summary_temperature: float = 0.0,
    summary_think: bool = False,
    generation_think: bool = False,
    summarize_query: bool = True,
) -> Any:
    async def summarize_node(state: AgentState) -> AgentState:
        started = perf_counter()
        if not summarize_query:
            LOGGER.info(
                "latency stage=query_preparation duration_ms=%.0f mode=direct",
                (perf_counter() - started) * 1_000,
            )
            return {"retrieval_query": state["question"].strip()}
        schema = RetrievalQuery.model_json_schema()
        messages = [
            {
                "role": "system",
                "content": (
                    f"{SUMMARIZE_SYSTEM_PROMPT}\n\nJSON Schema:\n"
                    f"{json.dumps(schema, ensure_ascii=False)}"
                ),
            }
        ]
        for example_question, example_answer in SUMMARIZE_FEW_SHOTS:
            messages.extend(
                [
                    {"role": "user", "content": example_question},
                    {"role": "assistant", "content": example_answer},
                ]
            )
        messages.append({"role": "user", "content": state["question"]})
        response = await chat_model.chat(
            messages,
            temperature=summary_temperature,
            max_tokens=summary_max_tokens,
            think=summary_think,
            response_format=schema,
        )
        try:
            query = RetrievalQuery.model_validate_json(response).retrieval_query
        except ValidationError:
            query = state["question"].strip()
        LOGGER.info(
            "latency stage=query_preparation duration_ms=%.0f mode=llm",
            (perf_counter() - started) * 1_000,
        )
        return {"retrieval_query": query}

    async def retrieve_node(state: AgentState) -> AgentState:
        started = perf_counter()
        hits = await retriever.retrieve(
            state["retrieval_query"],
            top_k,
            relevance_query=state["question"],
        )
        accepted_hits = [
            hit
            for hit in hits
            if hit.semantic_similarity >= min_semantic_similarity
        ]
        LOGGER.info(
            "latency stage=retrieval duration_ms=%.0f hits=%d",
            (perf_counter() - started) * 1_000,
            len(accepted_hits),
        )
        return {"hits": accepted_hits}

    def route_after_retrieval(state: AgentState) -> Literal["generate", "refuse"]:
        hits = state.get("hits", [])
        if not hits:
            return "refuse"
        return "generate"

    async def generate_node(state: AgentState) -> AgentState:
        started = perf_counter()
        hits = state["hits"]
        context = _format_context(hits, max_context_chars)
        schema = GroundedAnswer.model_json_schema()
        messages = [
            {
                "role": "system",
                "content": (
                    f"{ANSWER_SYSTEM_PROMPT}\n\nJSON Schema:\n"
                    f"{json.dumps(schema, ensure_ascii=False)}"
                ),
            }
        ]
        for example_input, example_output in ANSWER_FEW_SHOTS:
            messages.extend(
                [
                    {"role": "user", "content": example_input},
                    {"role": "assistant", "content": example_output},
                ]
            )
        messages.append(
            {
                "role": "user",
                "content": f"Вопрос пользователя:\n{state['question']}\n\nИсточники:\n{context}",
            }
        )
        response = await chat_model.chat(
            messages,
            think=generation_think,
            response_format=schema,
        )
        try:
            answer = GroundedAnswer.model_validate_json(response).answer
        except ValidationError:
            answer = ""
        LOGGER.info(
            "latency stage=answer_generation duration_ms=%.0f",
            (perf_counter() - started) * 1_000,
        )
        return {"answer_body": answer}

    def finalize_node(state: AgentState) -> AgentState:
        hits = state["hits"]
        answer_body = answer_body_only(state["answer_body"])
        numbers = cited_numbers(answer_body, len(hits))
        if not numbers:
            return {
                "answer": INSUFFICIENT_ANSWER,
                "answer_body": INSUFFICIENT_ANSWER,
                "sources": [],
                "useful_links": [],
                "grounded": False,
            }
        return {
            "answer": format_final_answer(answer_body, hits, numbers),
            "answer_body": answer_body,
            "sources": source_metadata(hits, numbers),
            "useful_links": deduplicated_urls(hits, numbers),
            "grounded": True,
        }

    def refuse_node(state: AgentState) -> AgentState:
        return {
            "answer": INSUFFICIENT_ANSWER,
            "answer_body": INSUFFICIENT_ANSWER,
            "sources": [],
            "useful_links": [],
            "grounded": False,
        }

    builder = StateGraph(
        AgentState,
        input_schema=AgentInput,
        output_schema=AgentOutput,
    )
    builder.add_node("summarize", summarize_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("refuse", refuse_node)
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {"generate": "generate", "refuse": "refuse"},
    )
    builder.add_edge("generate", "finalize")
    builder.add_edge("finalize", END)
    builder.add_edge("refuse", END)
    return builder.compile()


def _format_context(hits: list[RetrievalHit], max_context_chars: int) -> str:
    if not hits:
        return ""
    per_document = max(800, max_context_chars // len(hits))
    sections: list[str] = []
    used = 0
    for number, hit in enumerate(hits, start=1):
        body = hit.document.evidence_text(per_document)
        section = f"[{number}] ID {hit.document.id}\n{body}"
        remaining = max_context_chars - used
        if remaining <= 0:
            break
        sections.append(section[:remaining])
        used += len(sections[-1])
    return "\n\n".join(sections)
