from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from financial_research_agent.settings import Settings

SERVICE_NAME = "financial-research-agent"
SUPPORTED_CREDENTIAL_PROVIDERS = ("openai", "anthropic", "gemini", "litellm")

_PROVIDER_FIELDS = {
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
    "gemini": "gemini_api_key",
    "litellm": "litellm_api_key",
}
_SECURE_BACKENDS = {
    ("keyring.backends.Windows", "WinVaultKeyring"),
    ("keyring.backends.macOS", "Keyring"),
    ("keyring.backends.SecretService", "Keyring"),
    ("keyring.backends.kwallet", "DBusKeyring"),
}


class CredentialStoreError(RuntimeError):
    pass


class CredentialStore(Protocol):
    @property
    def available(self) -> bool: ...

    @property
    def backend_name(self) -> str | None: ...

    def get(self, provider: str) -> str | None: ...

    def set(self, provider: str, api_key: str) -> None: ...

    def delete(self, provider: str) -> None: ...


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    provider: str
    configured: bool
    source: str
    writable: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "configured": self.configured,
            "source": self.source,
            "writable": self.writable,
        }


@dataclass(frozen=True, slots=True)
class UnavailableCredentialStore:
    reason: str = "Secure OS credential storage is unavailable."

    @property
    def available(self) -> bool:
        return False

    @property
    def backend_name(self) -> str | None:
        return None

    def get(self, provider: str) -> str | None:
        _require_provider(provider)
        return None

    def set(self, provider: str, api_key: str) -> None:
        _require_provider(provider)
        raise CredentialStoreError(self.reason)

    def delete(self, provider: str) -> None:
        _require_provider(provider)
        raise CredentialStoreError(self.reason)


@dataclass(frozen=True, slots=True)
class KeyringCredentialStore:
    backend: object

    @property
    def available(self) -> bool:
        return True

    @property
    def backend_name(self) -> str:
        backend_type = type(self.backend)
        return f"{backend_type.__module__}.{backend_type.__name__}"

    def get(self, provider: str) -> str | None:
        normalized = _require_provider(provider)
        try:
            value = self.backend.get_password(SERVICE_NAME, normalized)
        except Exception:
            raise CredentialStoreError("Could not read the saved provider credential.") from None
        return value.strip() if isinstance(value, str) and value.strip() else None

    def set(self, provider: str, api_key: str) -> None:
        normalized = _require_provider(provider)
        value = api_key.strip()
        if not value or len(value) > 500:
            raise CredentialStoreError("API key must contain between 1 and 500 characters.")
        try:
            self.backend.set_password(SERVICE_NAME, normalized, value)
        except Exception:
            raise CredentialStoreError("Could not save the provider credential securely.") from None

    def delete(self, provider: str) -> None:
        normalized = _require_provider(provider)
        try:
            if self.backend.get_password(SERVICE_NAME, normalized) is not None:
                self.backend.delete_password(SERVICE_NAME, normalized)
        except Exception:
            raise CredentialStoreError("Could not remove the saved provider credential.") from None


def create_system_credential_store() -> CredentialStore:
    try:
        import keyring
        from keyring.backend import get_all_keyring

        candidates = [keyring.get_keyring(), *get_all_keyring()]
    except Exception:
        return UnavailableCredentialStore()
    secure = [candidate for candidate in candidates if _is_secure_backend(candidate)]
    if not secure:
        return UnavailableCredentialStore()
    backend = max(secure, key=lambda candidate: float(getattr(candidate, "priority", 0)))
    return KeyringCredentialStore(backend=backend)


def resolve_provider_credentials(settings: Settings, store: CredentialStore) -> Settings:
    provider_settings = settings.provider
    updates: dict[str, str] = {}
    for provider, field_name in _PROVIDER_FIELDS.items():
        if getattr(provider_settings, field_name) is not None:
            continue
        try:
            saved = store.get(provider)
        except CredentialStoreError:
            continue
        if saved is not None:
            updates[field_name] = saved
    if not updates:
        return settings
    return replace(settings, provider=replace(provider_settings, **updates))


def credential_status(
    provider: str,
    *,
    environment_settings: Settings,
    store: CredentialStore,
) -> CredentialStatus:
    normalized = _require_provider(provider)
    field_name = _PROVIDER_FIELDS[normalized]
    if getattr(environment_settings.provider, field_name) is not None:
        return CredentialStatus(
            provider=normalized,
            configured=True,
            source="environment",
            writable=False,
        )
    try:
        saved = store.get(normalized)
    except CredentialStoreError:
        return CredentialStatus(
            provider=normalized,
            configured=False,
            source="not_configured",
            writable=False,
        )
    return CredentialStatus(
        provider=normalized,
        configured=saved is not None,
        source="keyring" if saved is not None else "not_configured",
        writable=store.available,
    )


def _require_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in _PROVIDER_FIELDS:
        raise CredentialStoreError("Provider does not support stored API credentials.")
    return normalized


def _is_secure_backend(backend: object) -> bool:
    backend_type = type(backend)
    return (backend_type.__module__, backend_type.__name__) in _SECURE_BACKENDS
