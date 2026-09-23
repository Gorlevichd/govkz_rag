from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from govkz_rag.api.schemas import AnswerResponse  # noqa: E402
from govkz_rag.config import Settings  # noqa: E402


class FrontendRequestError(RuntimeError):
    pass


REFERENCE_NUMBER_PATTERN = re.compile(r"(?:\s*\[\d+\])+")


def without_reference_numbers(answer: str) -> str:
    return REFERENCE_NUMBER_PATTERN.sub("", answer).strip()


def backend_api_url(host: str, port: int) -> str:
    client_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{client_host}:{port}/ask/invoke"


def configured_backend_url(host: str, port: int) -> str:
    return os.getenv("BACKEND_URL") or backend_api_url(host, port)


def _backend_health_url(api_url: str) -> str:
    url = httpx.URL(api_url)
    return str(url.copy_with(path="/openapi.json", query=None, fragment=None))


def backend_is_ready(api_url: str) -> bool:
    try:
        response = httpx.get(_backend_health_url(api_url), timeout=1.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return False
    return True


def request_answer(
    question: str,
    api_url: str,
    timeout_seconds: float,
) -> AnswerResponse:
    try:
        response = httpx.post(
            api_url,
            json={"input": {"question": question}},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except httpx.ConnectError as exc:
        raise FrontendRequestError(
            "Сервис ответов недоступен. Проверьте backend."
        ) from exc
    except httpx.TimeoutException as exc:
        raise FrontendRequestError(
            "Сервис не успел подготовить ответ. Попробуйте повторить запрос."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise FrontendRequestError(
            f"Сервис вернул ошибку HTTP {exc.response.status_code}."
        ) from exc

    try:
        payload: Any = response.json()
        return AnswerResponse.model_validate(payload["output"])
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise FrontendRequestError(
            "Сервис вернул ответ в неожиданном формате."
        ) from exc


def main() -> None:
    settings = Settings.from_yaml(PROJECT_ROOT / "config.yaml")
    api_url = configured_backend_url(
        settings.app.server.host, settings.app.server.port
    )
    st.set_page_config(
        page_title="Государственные услуги Казахстана",
        page_icon="🇰🇿",
        layout="centered",
    )
    st.markdown(
        """
        <style>
        .stApp { background: #ffffff; }
        .block-container { max-width: 850px; padding-top: 14vh; }
        .gov-title { text-align: center; font-size: 2.35rem; font-weight: 600; }
        .gov-subtitle { text-align: center; color: #5f6368; margin-bottom: 2rem; }
        div[data-testid="stTextInput"] input {
            border-radius: 999px;
            min-height: 3.4rem;
            padding-left: 1.4rem;
            font-size: 1.05rem;
        }
        div[data-testid="stFormSubmitButton"] button {
            border-radius: 999px;
            display: block;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="gov-title">Государственные услуги Казахстана</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="gov-subtitle">Задайте вопрос о государственной услуге</div>',
        unsafe_allow_html=True,
    )

    if not backend_is_ready(api_url):
        st.error("Сервис ответов недоступен. Проверьте backend.")
        return

    with st.form("question_form"):
        question = st.text_input(
            "Вопрос",
            placeholder="Например: Как зарегистрировать брак?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Найти ответ", use_container_width=False)

    if not submitted:
        return
    if not question.strip():
        st.warning("Введите вопрос.")
        return

    try:
        with st.spinner("Ищем информацию в базе государственных услуг…"):
            result = request_answer(
                question.strip(),
                api_url,
                settings.app.frontend.request_timeout_seconds,
            )
    except FrontendRequestError as exc:
        st.error(str(exc))
        return

    st.markdown(without_reference_numbers(result.answer))
    if result.sources:
        st.subheader("Источники")
        for source in result.sources:
            title = source.get("title", "Источник")
            with st.expander(title):
                answer = source.get("answer")
                if answer:
                    st.write(answer)
                else:
                    st.caption("Текст источника недоступен.")

if __name__ == "__main__":
    main()
