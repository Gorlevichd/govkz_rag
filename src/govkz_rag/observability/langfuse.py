from __future__ import annotations

import os

from langfuse.langchain import CallbackHandler


REQUIRED_ENVIRONMENT_VARIABLES = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
)


def create_langfuse_handler(enabled: bool) -> CallbackHandler | None:
    if not enabled:
        return None
    missing = [name for name in REQUIRED_ENVIRONMENT_VARIABLES if not os.getenv(name)]
    if missing:
        raise ValueError(
            "Langfuse monitoring is enabled but these environment variables are "
            f"missing: {', '.join(missing)}"
        )
    return CallbackHandler()
