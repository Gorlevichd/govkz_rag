from __future__ import annotations

import atexit
import re
import subprocess
import sys
import time
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


class BackendStartupError(RuntimeError):
    pass


REFERENCE_NUMBER_PATTERN = re.compile(r"(?:\s*\[\d+\])+")


def without_reference_numbers(answer: str) -> str:
    return REFERENCE_NUMBER_PATTERN.sub("", answer).strip()


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


def _stop_backend(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


@st.cache_resource
def _start_backend() -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "backend.py")],
        cwd=PROJECT_ROOT,
    )
    atexit.register(_stop_backend, process)
    return process


def ensure_backend(api_url: str, timeout_seconds: float) -> None:
    if backend_is_ready(api_url):
        return

    process = _start_backend()
    if process.poll() is not None:
        _start_backend.clear()
        process = _start_backend()

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if backend_is_ready(api_url):
            return
        return_code = process.poll()
        if return_code is not None:
            raise BackendStartupError(
                f"Backend не запустился (код завершения {return_code})."
            )
        time.sleep(0.25)

    raise BackendStartupError(
        "Backend не успел запуститься. Проверьте Ollama и локальные модели."
    )


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
            "Сервис ответов недоступен. Перезапустите Streamlit-приложение."
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

    try:
        with st.spinner("Запускаем сервис и локальные модели…"):
            ensure_backend(
                str(settings.frontend.api_url),
                settings.frontend.request_timeout_seconds,
            )
    except BackendStartupError as exc:
        st.error(str(exc))
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
                str(settings.frontend.api_url),
                settings.frontend.request_timeout_seconds,
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

    if result.useful_links:
        with st.expander("Полезные ссылки"):
            for url in result.useful_links:
                st.markdown(f"- [{url}]({url})")


if __name__ == "__main__":
    main()
