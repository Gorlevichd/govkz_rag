import pytest

from govkz_rag.observability.langfuse import (
    REQUIRED_ENVIRONMENT_VARIABLES,
    create_langfuse_handler,
)


def test_langfuse_is_optional() -> None:
    assert create_langfuse_handler(False) is None


def test_enabled_langfuse_requires_credentials(monkeypatch) -> None:
    for name in REQUIRED_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
        create_langfuse_handler(True)
