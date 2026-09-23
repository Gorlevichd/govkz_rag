from types import SimpleNamespace

import create_index
import backend


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        retrieval=SimpleNamespace(
            chroma=SimpleNamespace(path="data/chroma", collection="egov_services"),
        ),
        app=SimpleNamespace(
            server=SimpleNamespace(
                host="127.0.0.1",
                port=8000,
                warmup_embedding=True,
            ),
        ),
    )


def test_create_index_requires_confirmation_before_overwrite(monkeypatch) -> None:
    settings = _settings()
    monkeypatch.setattr(create_index.Settings, "from_yaml", lambda path: settings)
    monkeypatch.setattr(create_index, "collection_exists", lambda path, name: True)
    monkeypatch.setattr(create_index, "_confirm_overwrite", lambda current: False)

    async def unexpected_build(current) -> None:
        raise AssertionError("existing index must not be overwritten")

    monkeypatch.setattr(create_index, "_build_index", unexpected_build)

    create_index.main()


def test_backend_uses_configured_host_and_port(monkeypatch) -> None:
    settings = _settings()
    app = object()
    called: dict[str, object] = {}
    monkeypatch.setattr(backend.Settings, "from_yaml", lambda path: settings)
    monkeypatch.setattr(backend, "_configure_utf8_console", lambda: None)
    monkeypatch.setattr(backend, "load_dotenv", lambda *args, **kwargs: None)
    warmed_up: list[object] = []

    async def fake_warmup(current) -> None:
        warmed_up.append(current)

    monkeypatch.setattr(backend, "warmup_embedding", fake_warmup)
    monkeypatch.setattr(backend, "create_app", lambda settings: app)
    monkeypatch.setattr(
        backend.uvicorn,
        "run",
        lambda application, **kwargs: called.update(app=application, **kwargs),
    )

    backend.main()

    assert called == {"app": app, "host": "127.0.0.1", "port": 8000}
    assert warmed_up == [settings]


def test_backend_accepts_container_bind_address(monkeypatch) -> None:
    settings = _settings()
    monkeypatch.setenv("SERVER_HOST", "0.0.0.0")
    monkeypatch.setattr(backend.Settings, "from_yaml", lambda path: settings)
    monkeypatch.setattr(backend, "_configure_utf8_console", lambda: None)
    monkeypatch.setattr(backend, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "create_app", lambda settings: object())
    called: dict[str, object] = {}
    monkeypatch.setattr(backend.uvicorn, "run", lambda app, **kwargs: called.update(kwargs))

    async def fake_warmup(current) -> None:
        pass

    monkeypatch.setattr(backend, "warmup_embedding", fake_warmup)
    backend.main()

    assert called["host"] == "0.0.0.0"
