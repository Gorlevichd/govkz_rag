from fastapi import FastAPI
from langserve import add_routes

from govkz_rag.config import Settings
from govkz_rag.api.runnable import build_question_runnables


def create_app(*, settings: Settings | None = None, runnable=None) -> FastAPI:
    application = FastAPI(
        title="Kazakhstan eGov RAG LangServe API",
        version="0.1.0",
    )
    structured_runnable = None
    if runnable is None:
        runnable, structured_runnable = build_question_runnables(
            settings or Settings.from_yaml()
        )
    add_routes(
        application,
        runnable,
        path="",
        enabled_endpoints=[
            "invoke",
            "batch",
            "stream",
            "stream_log",
            "input_schema",
            "output_schema",
            "playground",
        ],
    )
    if structured_runnable is not None:
        add_routes(
            application,
            structured_runnable,
            path="/ask",
            enabled_endpoints=["invoke", "input_schema", "output_schema"],
        )
    return application
