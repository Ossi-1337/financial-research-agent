# LLM Providers

Financial Research Agent keeps provider-specific HTTP formats behind one async contract. The
default remains `offline-test`; selecting a network provider is always explicit.

## Capability Matrix

| Provider | Chat | Streaming | Tools | Structured output | Embeddings | Credential |
| --- | --- | --- | --- | --- | --- | --- |
| `offline-test` | Yes | Yes | Yes | Yes | Yes | None |
| `local-openai` | Yes | Yes | Model-dependent | Model-dependent | Runtime-dependent | None |
| `openai` | Yes | Yes | Model-dependent | Model-dependent | Yes | OpenAI key |
| `anthropic` | Yes | Yes | Model-dependent | Model-dependent | No | Anthropic key |
| `gemini` | Yes | Yes | Model-dependent | Model-dependent | Yes | Gemini key |
| `litellm` | Yes | Yes | Upstream-dependent | Upstream-dependent | Upstream-dependent | Gateway-dependent |

Capability flags describe adapter support, not a guarantee that every model or account supports
that feature. Provider health shows configured credentials, endpoint reachability, visible
models, and limitations without returning secret values.

## Direct Providers

Configure Anthropic:

```dotenv
FRA_LLM_PROVIDER=anthropic
FRA_LLM_MODEL=anthropic-model-id
FRA_ANTHROPIC_API_KEY=
```

Configure Gemini:

```dotenv
FRA_LLM_PROVIDER=gemini
FRA_LLM_MODEL=gemini-model-id
FRA_GEMINI_API_KEY=
```

`ANTHROPIC_API_KEY` and `GEMINI_API_KEY` are accepted as fallback aliases. The `FRA_` names take
precedence. Keys stay in the process environment or ignored `.env`; the settings UI never stores
them.

Anthropic uses the native Messages API and is chat-only in this app. Gemini uses stateless
`generateContent`, SSE streaming, function calling, structured responses, and batch embeddings.
Conversation state remains local and app-owned.

## LiteLLM Gateway

Run LiteLLM separately, then configure its OpenAI-compatible endpoint:

```dotenv
FRA_LLM_PROVIDER=litellm
FRA_LLM_MODEL=gateway-model-id
FRA_LITELLM_BASE_URL=http://127.0.0.1:4000/v1
FRA_LITELLM_API_KEY=
```

The key may stay empty for a trusted loopback proxy without authentication. This project does not
install, start, route, retry, meter, or administer LiteLLM. Upstream provider terms, retention,
cost, and model capabilities still apply.

## Verification

Default tests use `httpx.MockTransport` and make no network calls. Live smoke tests are opt-in:

```powershell
$env:FRA_ANTHROPIC_SMOKE_TEST = "1"
$env:FRA_GEMINI_SMOKE_TEST = "1"
$env:FRA_LITELLM_SMOKE_TEST = "1"
python -m pytest tests/test_hosted_provider_live.py
```

Enable one provider at a time with its matching model, endpoint, and credential configuration.
