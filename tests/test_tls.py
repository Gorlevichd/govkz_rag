import ssl

import pytest

from govkz_rag.api.runnable import _ssl_context


def test_external_api_uses_verified_system_certificate_store() -> None:
    context = _ssl_context(None)

    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_external_api_rejects_missing_custom_ca_bundle(tmp_path) -> None:
    missing = tmp_path / "missing-ca.pem"

    with pytest.raises(ValueError, match="does not exist"):
        _ssl_context(str(missing))
