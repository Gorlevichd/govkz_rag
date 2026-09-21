from types import SimpleNamespace

import create_index
import backend


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        chroma=SimpleNamespace(path="data/chroma", collection="egov_services"),
        server=SimpleNamespace(
            host="127.0.0.1",
            port=8000,
            warmup_embedding=True,
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
