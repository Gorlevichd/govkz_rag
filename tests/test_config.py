from pathlib import Path

from govkz_rag.config import Settings


def test_settings_are_loaded_from_yaml_relative_to_config(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
data:
  workbook: input/data.xlsx
chroma:
  path: state/chroma
  collection: test_services
ollama:
  base_url: http://localhost:11434/
  embedding_model: embedding-model
  request_timeout_seconds: 90
  embedding_keep_alive: 0
groq:
  base_url: https://api.groq.com/openai/v1
  chat_model: qwen/qwen3.8-27b
  api_key_env: GROQ_KEY
  ca_bundle_env: GROQ_CA_BUNDLE
  request_timeout_seconds: 60
  reasoning_effort: none
index:
  batch_size: 10
  embedding_keep_alive: 5m
retrieval:
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
server:
  host: 0.0.0.0
  port: 9000
  warmup_embedding: true
frontend:
  api_url: http://127.0.0.1:9000/ask/invoke
  request_timeout_seconds: 120
langfuse:
  enabled: false
  run_name: test-egov-question
""".strip(),
        encoding="utf-8",
    )

    settings = Settings.from_yaml(config)

    assert settings.data.workbook == tmp_path / "input/data.xlsx"
    assert settings.chroma.path == tmp_path / "state/chroma"
    assert settings.index.batch_size == 10
    assert settings.retrieval.rrf_k == 42
    assert settings.retrieval.alias_candidate_multiplier == 5
    assert settings.retrieval.min_semantic_similarity == 0.6
    assert settings.reranker.enabled is True
    assert settings.reranker.model == tmp_path / "test-reranker"
    assert settings.reranker.candidate_k == 8
    assert settings.reranker.min_score == 0.25
    assert settings.evaluation.dataset == tmp_path / "evals/test_eval.jsonl"
    assert str(settings.ollama.base_url) == "http://localhost:11434/"
    assert settings.ollama.request_timeout_seconds == 90
    assert settings.ollama.embedding_keep_alive == 0
    assert settings.groq.chat_model == "qwen/qwen3.8-27b"
    assert settings.groq.api_key_env == "GROQ_KEY"
    assert settings.groq.ca_bundle_env == "GROQ_CA_BUNDLE"
    assert settings.groq.reasoning_effort == "none"
    assert settings.index.embedding_keep_alive == "5m"
    assert settings.summarization.max_tokens == 48
    assert settings.summarization.enabled is False
    assert settings.generation.temperature == 0.2
    assert settings.generation.max_tokens == 256
    assert settings.generation.think is False
    assert settings.server.port == 9000
    assert settings.server.warmup_embedding is True
    assert str(settings.frontend.api_url) == "http://127.0.0.1:9000/ask/invoke"
    assert settings.frontend.request_timeout_seconds == 120
    assert settings.langfuse.enabled is False
    assert settings.langfuse.run_name == "test-egov-question"
