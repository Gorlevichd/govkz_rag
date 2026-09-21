from pathlib import Path
from types import SimpleNamespace

from docker.prepare_assets import RERANKER_FILES, _reranker_is_ready
from docker.prepare_ollama import desired_models


def _settings(provider: str) -> SimpleNamespace:
    qa = SimpleNamespace(provider=provider, model="qa-model")
    embedding = SimpleNamespace(model="embedding-model")
    return SimpleNamespace(
        agent=SimpleNamespace(qa=qa),
        retrieval=SimpleNamespace(embedding=embedding),
    )


def test_local_qa_downloads_both_ollama_models() -> None:
    assert desired_models(_settings("ollama")) == (
        "embedding-model",
        "qa-model",
    )


def test_external_qa_downloads_only_embedding_model() -> None:
    assert desired_models(_settings("openai")) == ("embedding-model",)


def test_reranker_readiness_requires_complete_snapshot(tmp_path: Path) -> None:
    assert _reranker_is_ready(tmp_path) is False
    for name in RERANKER_FILES:
        (tmp_path / name).write_text("test", encoding="utf-8")
    assert _reranker_is_ready(tmp_path) is True
