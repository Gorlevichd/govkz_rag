import httpx
import pytest

import serve


def test_backend_api_url_uses_connectable_loopback_address() -> None:
    assert (
        serve.backend_api_url("0.0.0.0", 9000)
        == "http://127.0.0.1:9000/ask/invoke"
    )


class FakeProcess:
    def poll(self) -> None:
        return None


def test_without_reference_numbers_removes_citations() -> None:
    answer = "Подайте заявление через eGov [1]. Затем посетите ЦОН. [1][2]"

    assert (
        serve.without_reference_numbers(answer)
        == "Подайте заявление через eGov. Затем посетите ЦОН."
    )


def test_request_answer_reads_structured_langserve_output(monkeypatch) -> None:
    request = httpx.Request("POST", "http://test/ask/invoke")
    response = httpx.Response(
        200,
        request=request,
        json={
            "output": {
                "answer": "Подайте заявление. [1]",
                "sources": [
                    {
                        "number": 1,
                        "title": "Регистрация брака",
                        "answer": "Заявление можно подать через портал.",
                        "url": "https://egov.kz/service/42",
                    }
                ],
                "useful_links": ["https://egov.kz/service/42"],
                "grounded": True,
            }
        },
    )
    monkeypatch.setattr(serve.httpx, "post", lambda *args, **kwargs: response)

    result = serve.request_answer(
        "Как зарегистрировать брак?",
        "http://test/ask/invoke",
        30,
    )

    assert result.grounded is True
    assert result.sources[0]["title"] == "Регистрация брака"
    assert result.sources[0]["answer"] == "Заявление можно подать через портал."
    assert result.useful_links == ["https://egov.kz/service/42"]


def test_request_answer_reports_unavailable_backend(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(serve.httpx, "post", fail)

    with pytest.raises(serve.FrontendRequestError, match="Перезапустите"):
        serve.request_answer("Вопрос", "http://test/ask/invoke", 30)


def test_ensure_backend_reuses_running_service(monkeypatch) -> None:
    monkeypatch.setattr(serve, "backend_is_ready", lambda url: True)

    def unexpected_start() -> FakeProcess:
        raise AssertionError("a running backend must be reused")

    monkeypatch.setattr(serve, "_start_backend", unexpected_start)

    serve.ensure_backend("http://127.0.0.1:8000/ask/invoke", 1)


def test_ensure_backend_starts_and_waits_for_service(monkeypatch) -> None:
    readiness = iter([False, False, True])
    process = FakeProcess()
    monkeypatch.setattr(serve, "backend_is_ready", lambda url: next(readiness))
    monkeypatch.setattr(serve, "_start_backend", lambda: process)
    monkeypatch.setattr(serve.time, "sleep", lambda seconds: None)

    serve.ensure_backend("http://127.0.0.1:8000/ask/invoke", 1)
