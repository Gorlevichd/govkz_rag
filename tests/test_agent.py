import asyncio
from collections.abc import Sequence

from govkz_rag.agent import build_agent
from govkz_rag.retrieval.models import RetrievalHit, ServiceDocument


class FakeRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        relevance_query: str | None = None,
    ) -> list[RetrievalHit]:
        self.queries.append(query)
        self.relevance_query = relevance_query
        return self.hits[:k]


class FakeChatModel:
    def __init__(self) -> None:
        self.requests: list[Sequence[dict[str, str]]] = []
        self.options: list[dict] = []

    async def chat(self, messages: Sequence[dict[str, str]], **kwargs) -> str:
        self.requests.append(messages)
        self.options.append(kwargs)
        if len(self.requests) == 1:
            return '{"retrieval_query":"регистрация брака подача заявления"}'
        return '{"answer":"Подайте заявление через портал [1]."}'


def _hit(relevance: float) -> RetrievalHit:
    document = ServiceDocument(
        id=42,
        source_ids=(42,),
        name="Регистрация брака",
        answer="Заявление можно подать через портал.",
        chunks="Порядок регистрации брака.",
        alternative_queries=("регистрация брака",),
        egov_link="https://egov.kz/service/42",
    )
    return RetrievalHit(
        document=document,
        score=relevance,
        semantic_similarity=relevance,
    )


def test_agent_generates_only_after_sufficient_retrieval() -> None:
    model = FakeChatModel()
    retriever = FakeRetriever([_hit(0.8)])
    agent = build_agent(retriever, model, min_semantic_similarity=0.35)

    result = asyncio.run(agent.ainvoke({"question": "Как зарегистрировать брак?"}))

    assert result["grounded"] is True
    assert result["sources"][0]["id"] == 42
    assert result["sources"][0]["answer"] == "Заявление можно подать через портал."
    assert "[1]" in result["answer"]
    assert "Источники:\n\n[1] - Регистрация брака" in result["answer"]
    assert "Полезные ссылки:\n\n- https://egov.kz/service/42" in result["answer"]
    assert result["answer_body"] == "Подайте заявление через портал [1]."
    assert result["useful_links"] == ["https://egov.kz/service/42"]
    assert retriever.queries == ["регистрация брака подача заявления"]
    assert retriever.relevance_query == "Как зарегистрировать брак?"
    assert "Не показывай ход рассуждений" in model.requests[0][0]["content"]
    assert "/no_think" in model.requests[0][0]["content"]
    assert "Не показывай ход рассуждений" in model.requests[-1][0]["content"]
    assert "строго 1–2 коротких предложения" in model.requests[-1][0]["content"]
    assert "не более\n350 символов" in model.requests[-1][0]["content"]
    assert "не перечисляй" in model.requests[-1][0]["content"].casefold()
    assert model.options[-1]["response_format"]["required"] == ["answer"]
    assert "Заявление можно подать" in model.requests[-1][-1]["content"]


def test_agent_refuses_when_evidence_is_weak() -> None:
    model = FakeChatModel()
    agent = build_agent(
        FakeRetriever([_hit(0.1)]),
        model,
        min_semantic_similarity=0.35,
    )

    result = asyncio.run(agent.ainvoke({"question": "Какая погода на Марсе?"}))

    assert result["grounded"] is False
    assert result["sources"] == []
    assert len(model.requests) == 1


def test_agent_can_retrieve_original_question_without_llm_summary() -> None:
    model = FakeChatModel()

    async def answer_only_chat(messages, **kwargs) -> str:
        model.requests.append(messages)
        return '{"answer":"Подайте заявление через портал [1]."}'

    model.chat = answer_only_chat
    retriever = FakeRetriever([_hit(0.8)])
    agent = build_agent(
        retriever,
        model,
        min_semantic_similarity=0.35,
        summarize_query=False,
    )

    result = asyncio.run(agent.ainvoke({"question": "Как зарегистрировать брак?"}))

    assert result["grounded"] is True
    assert retriever.queries == ["Как зарегистрировать брак?"]
    assert len(model.requests) == 1


def test_agent_rejects_high_lexical_hit_with_low_semantic_similarity() -> None:
    model = FakeChatModel()
    weak_hit = _hit(0.2)
    weak_hit = RetrievalHit(
        document=weak_hit.document,
        score=1.0,
        semantic_similarity=0.2,
        lexical_coverage=1.0,
    )
    agent = build_agent(
        FakeRetriever([weak_hit]),
        model,
        min_semantic_similarity=0.6,
    )

    result = asyncio.run(agent.ainvoke({"question": "Как открыть банковский счет?"}))

    assert result["grounded"] is False
    assert result["sources"] == []
    assert len(model.requests) == 1


def test_agent_rejects_answer_without_valid_citation() -> None:
    model = FakeChatModel()

    async def uncited_chat(messages, **kwargs) -> str:
        if "поисковые запросы" in messages[0]["content"]:
            return '{"retrieval_query":"регистрация брака"}'
        return '{"answer":"Подайте заявление через портал."}'

    model.chat = uncited_chat
    agent = build_agent(FakeRetriever([_hit(0.8)]), model)

    result = asyncio.run(agent.ainvoke({"question": "Как зарегистрировать брак?"}))

    assert result["grounded"] is False
    assert result["sources"] == []
    assert "недостаточно" in result["answer"]


def test_agent_does_not_accept_citation_only_from_model_source_list() -> None:
    model = FakeChatModel()

    async def misplaced_citation_chat(messages, **kwargs) -> str:
        if "поисковые запросы" in messages[0]["content"]:
            return '{"retrieval_query":"регистрация брака"}'
        return '{"answer":"Подайте заявление через портал.\\n\\nИсточники:\\n[1] - Запись"}'

    model.chat = misplaced_citation_chat
    agent = build_agent(FakeRetriever([_hit(0.8)]), model)

    result = asyncio.run(agent.ainvoke({"question": "Как зарегистрировать брак?"}))

    assert result["grounded"] is False
    assert result["sources"] == []


def test_agent_lists_only_cited_sources_and_deduplicates_urls() -> None:
    first = _hit(0.9)
    duplicate = RetrievalHit(
        document=ServiceDocument(
            id=43,
            source_ids=(43,),
            name="Срок регистрации",
            answer="Срок указан в карточке услуги.",
            chunks="Срок регистрации.",
            alternative_queries=("срок регистрации брака",),
            egov_link="https://EGOV.kz/service/42#details",
        ),
        score=0.8,
        semantic_similarity=0.8,
    )
    unused = RetrievalHit(
        document=ServiceDocument(
            id=44,
            source_ids=(44,),
            name="Другой документ",
            answer="Другой ответ.",
            chunks="Другой материал.",
            alternative_queries=("другая услуга",),
            egov_link="https://egov.kz/service/44",
        ),
        score=0.7,
        semantic_similarity=0.7,
    )
    model = FakeChatModel()

    async def cited_chat(messages, **kwargs) -> str:
        if "поисковые запросы" in messages[0]["content"]:
            return '{"retrieval_query":"регистрация брака"}'
        return (
            '{"answer":"Заявление подается через портал [1]. '
            'Срок указан отдельно [2]."}'
        )

    model.chat = cited_chat
    agent = build_agent(FakeRetriever([first, duplicate, unused]), model)

    result = asyncio.run(agent.ainvoke({"question": "Как зарегистрировать брак?"}))

    assert [source["id"] for source in result["sources"]] == [42, 43]
    assert "[3] - Другой документ" not in result["answer"]
    assert result["answer"].count("- https://egov.kz/service/42") == 1
    assert result["useful_links"] == ["https://egov.kz/service/42"]


def test_agent_ignores_non_egov_link_fields() -> None:
    hit = RetrievalHit(
        document=ServiceDocument(
            id=50,
            source_ids=(50,),
            name="Услуга без русской ссылки",
            answer="Краткий ответ.",
            chunks="Материал.",
            alternative_queries=("услуга",),
            egov_kaz_link="https://egov.kz/cms/kz/service/50",
            apply_button="https://example.test/apply/50",
        ),
        score=0.9,
        semantic_similarity=0.9,
    )
    model = FakeChatModel()
    agent = build_agent(FakeRetriever([hit]), model)

    result = asyncio.run(agent.ainvoke({"question": "Как получить услугу?"}))

    assert result["useful_links"] == []
    assert "Полезные ссылки" not in result["answer"]


def test_agent_never_uses_unstructured_reasoning_as_retrieval_query() -> None:
    retriever = FakeRetriever([_hit(0.1)])
    model = FakeChatModel()

    async def reasoning_chat(messages, **kwargs) -> str:
        return (
            '{"retrieval_query":"Хорошо, мне нужно преобразовать вопрос пользователя '
            'как зарегистрировать брак в короткий поисковый запрос"}'
        )

    model.chat = reasoning_chat
    agent = build_agent(retriever, model)

    asyncio.run(agent.ainvoke({"question": "Как зарегистрировать брак?"}))

    assert retriever.queries == ["Как зарегистрировать брак?"]


def test_agent_rejects_reasoning_inside_structured_answer() -> None:
    model = FakeChatModel()

    async def reasoning_answer_chat(messages, **kwargs) -> str:
        if "поисковые запросы" in messages[0]["content"]:
            return '{"retrieval_query":"регистрация брака"}'
        return (
            '{"answer":"Пользователь спрашивает о регистрации брака. '
            'Мне нужно ответить по источнику. Подайте заявление [1]."}'
        )

    model.chat = reasoning_answer_chat
    agent = build_agent(FakeRetriever([_hit(0.8)]), model)

    result = asyncio.run(agent.ainvoke({"question": "Как зарегистрировать брак?"}))

    assert result["grounded"] is False
    assert result["sources"] == []
    assert "недостаточно" in result["answer"]
