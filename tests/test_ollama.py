import asyncio

from govkz_rag.groq import GroqChatClient
from govkz_rag.ollama import OllamaEmbeddingClient


def test_query_embedding_uses_configured_keep_alive(monkeypatch) -> None:
    client = OllamaEmbeddingClient(
        "http://localhost:11434",
        "embedding-model",
        embedding_keep_alive="5m",
    )
    request: dict[str, object] = {}

    async def fake_post(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        request.update(endpoint=endpoint, payload=payload)
        return {"embeddings": [[0.1, 0.2]]}

    monkeypatch.setattr(client, "_post", fake_post)

    embeddings = asyncio.run(client.embed(["Вопрос"]))

    assert embeddings == [[0.1, 0.2]]
    assert request["endpoint"] == "/api/embed"
    payload = request["payload"]
    assert isinstance(payload, dict)
    assert payload["keep_alive"] == "5m"


def test_groq_chat_uses_qwen_and_strict_structured_output(monkeypatch) -> None:
    client = GroqChatClient(
        "https://api.groq.com/openai/v1",
        "qwen/qwen3.8-27b",
        "test-key",
        max_tokens=128,
        reasoning_effort="none",
    )
    request: dict[str, object] = {}

    async def fake_post(payload: dict[str, object]) -> dict[str, object]:
        request.update(payload=payload)
        return {
            "choices": [
                {"message": {"content": '{"answer":"Финальный ответ [1]."}'}}
            ]
        }

    monkeypatch.setattr(client, "_post", fake_post)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    answer = asyncio.run(
        client.chat(
            [{"role": "user", "content": "Вопрос"}],
            temperature=0.0,
            max_tokens=64,
            think=False,
            response_format=schema,
        )
    )

    assert answer == '{"answer":"Финальный ответ [1]."}'
    payload = request["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen/qwen3.8-27b"
    assert payload["max_completion_tokens"] == 64
    assert payload["reasoning_effort"] == "none"
    assert payload["include_reasoning"] is False
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == schema
