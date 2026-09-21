from pathlib import Path

from govkz_rag.config import Settings


def test_settings_are_loaded_from_yaml_relative_to_config(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
retrieval:
  data:
    workbook: input/data.xlsx
  chroma:
    path: state/chroma
    collection: test_services
  embedding:
    model: embedding-model
    request_timeout_seconds: 90
    keep_alive: 0
  index:
    batch_size: 10
    keep_alive: 5m
  search:
    top_k: 3
    candidate_k: 12
    rrf_k: 42
    alias_candidate_multiplier: 5
    min_semantic_similarity: 0.6
  reranker:
    enabled: true
    model: test-reranker
    candidate_k: 8
    min_score: 0.25
    max_length: 256
    max_passage_chars: 2000
    batch_size: 4
    device: cpu
  evaluation:
    dataset: evals/test_eval.jsonl
agent:
  qa:
    provider: openai
    model: provider-model
    request_timeout_seconds: 60
    reasoning_effort: none
  summarization:
    enabled: false
    max_tokens: 48
    temperature: 0.0
    think: false
  generation:
    max_context_chars: 9000
    max_tokens: 256
    temperature: 0.2
    think: false
observability:
  langfuse:
    enabled: false
    run_name: test-egov-question
app:
  server:
    host: 0.0.0.0
    port: 9000
    warmup_embedding: true
  frontend:
    request_timeout_seconds: 120
""".strip(),
        encoding="utf-8",
    )

    settings = Settings.from_yaml(config)

    retrieval = settings.retrieval
    assert retrieval.data.workbook == tmp_path / "input/data.xlsx"
    assert retrieval.chroma.path == tmp_path / "state/chroma"
    assert retrieval.index.batch_size == 10
    assert retrieval.search.rrf_k == 42
    assert retrieval.search.alias_candidate_multiplier == 5
    assert retrieval.search.min_semantic_similarity == 0.6
    assert retrieval.reranker.enabled is True
    assert retrieval.reranker.model == tmp_path / "test-reranker"
    assert retrieval.reranker.candidate_k == 8
    assert retrieval.reranker.min_score == 0.25
    assert retrieval.evaluation.dataset == tmp_path / "evals/test_eval.jsonl"
    assert retrieval.embedding.request_timeout_seconds == 90
    assert retrieval.embedding.keep_alive == 0
    assert retrieval.index.keep_alive == "5m"

    assert settings.agent.qa.provider == "openai"
    assert settings.agent.qa.model == "provider-model"
    assert settings.agent.qa.reasoning_effort == "none"
    assert settings.agent.summarization.max_tokens == 48
    assert settings.agent.summarization.enabled is False
    assert settings.agent.generation.temperature == 0.2
    assert settings.agent.generation.max_tokens == 256
    assert settings.agent.generation.think is False

    assert settings.app.server.port == 9000
    assert settings.app.server.warmup_embedding is True
    assert settings.app.frontend.request_timeout_seconds == 120
    assert settings.observability.langfuse.enabled is False
    assert settings.observability.langfuse.run_name == "test-egov-question"
