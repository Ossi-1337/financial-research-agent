from __future__ import annotations

from dataclasses import dataclass, field

import keyring
import pytest

from financial_research_agent.credentials import (
    CredentialStoreError,
    KeyringCredentialStore,
    UnavailableCredentialStore,
    create_system_credential_store,
    credential_status,
    resolve_provider_credentials,
)
from financial_research_agent.settings import Settings


@dataclass
class MemoryCredentialStore:
    values: dict[tuple[str, str], str] = field(default_factory=dict)

    def get_password(self, service: str, username: str) -> str | None:
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.values[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.values.pop((service, username), None)


class FailingCredentialStore(MemoryCredentialStore):
    def set_password(self, service: str, username: str, password: str) -> None:
        raise RuntimeError(password)


class UnknownCredentialBackend(MemoryCredentialStore):
    priority = 100


def test_keyring_store_saves_reads_and_deletes_provider_credentials() -> None:
    backend = MemoryCredentialStore()
    store = KeyringCredentialStore(backend=backend)

    store.set("openai", " saved-secret ")
    saved = store.get("openai")
    store.delete("openai")

    assert saved == "saved-secret"
    assert store.get("openai") is None
    assert backend.values == {}


def test_environment_credentials_take_precedence_over_keyring() -> None:
    backend = MemoryCredentialStore()
    store = KeyringCredentialStore(backend=backend)
    store.set("openai", "keyring-secret")
    store.set("gemini", "gemini-keyring-secret")
    environment = Settings.from_env({"FRA_OPENAI_API_KEY": "environment-secret"})

    resolved = resolve_provider_credentials(environment, store)
    openai_status = credential_status("openai", environment_settings=environment, store=store)
    gemini_status = credential_status("gemini", environment_settings=environment, store=store)

    assert resolved.provider.openai_api_key == "environment-secret"
    assert resolved.provider.gemini_api_key == "gemini-keyring-secret"
    assert openai_status.source == "environment"
    assert openai_status.writable is False
    assert gemini_status.source == "keyring"


def test_unavailable_store_never_persists_or_resolves_credentials() -> None:
    store = UnavailableCredentialStore()
    settings = Settings.from_env({})

    resolved = resolve_provider_credentials(settings, store)
    status = credential_status("anthropic", environment_settings=settings, store=store)

    assert resolved is settings
    assert status.source == "not_configured"
    assert status.writable is False
    with pytest.raises(CredentialStoreError, match="Secure OS credential storage is unavailable"):
        store.set("anthropic", "secret")


def test_credential_errors_do_not_include_secret_values() -> None:
    store = KeyringCredentialStore(backend=FailingCredentialStore())
    secret = "sensitive-value"

    with pytest.raises(CredentialStoreError) as captured:
        store.set("openai", secret)

    assert secret not in str(captured.value)


def test_unknown_keyring_backend_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = UnknownCredentialBackend()
    monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
    monkeypatch.setattr(keyring.backend, "get_all_keyring", lambda: [backend])

    store = create_system_credential_store()

    assert isinstance(store, UnavailableCredentialStore)


def test_unsupported_provider_is_rejected() -> None:
    store = KeyringCredentialStore(backend=MemoryCredentialStore())

    with pytest.raises(CredentialStoreError, match="does not support"):
        store.get("unknown")
