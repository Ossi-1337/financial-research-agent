# Troubleshooting

## Docker Is Unavailable

`Docker CLI is not installed or not on PATH` means the command cannot find Docker.
`Docker daemon is unavailable` means Docker Desktop is installed but not running.

Start Docker Desktop, wait until its engine is ready, then run:

```powershell
docker info
python scripts/dev.py docker-up --runtime offline
```

## CUDA Profile Fails

The CUDA profile requires Docker NVIDIA runtime support. Check:

```powershell
docker info --format "{{json .Runtimes}}"
nvidia-smi
```

Use `--runtime cpu` when `nvidia` is absent. The application itself does not install GPU
drivers, CUDA, or NVIDIA Container Toolkit.

## Model Is Downloading Or Loading

First profile startup may take minutes. llama.cpp can return `503 Service Unavailable` until
the model is loaded. Follow logs:

```powershell
docker compose --profile cuda logs -f llama-cpp-cuda
docker compose --profile cpu logs -f llama-cpp-cpu
```

Check `FRA_HUGGINGFACE_CACHE`, free disk space, model name, and network access when download
progress stops.

## Provider Unavailable

Confirm app and model use different ports and the app container uses the internal endpoint:

```text
FRA_LLM_BASE_URL=http://llama-cpp:8080/v1
```

For direct Python outside Docker, use `http://127.0.0.1:8080/v1` instead. Inspect configured
provider/model through `GET /api/status` without exposing secrets.

## Port Already In Use

Override host ports in `.env`:

```dotenv
FRA_APP_PORT=8010
FRA_LLAMA_PORT=8081
```

Changing the llama host port does not change the internal Compose endpoint.

## Data Provider Failures

- SEC EDGAR requires a descriptive `FRA_SEC_USER_AGENT` with real contact information.
- Alpha Vantage requires `FRA_ALPHA_VANTAGE_API_KEY` and may return rate-limit errors.
- Hosted OpenAI requires `FRA_OPENAI_API_KEY` only when `openai` is selected.
- Local/offline startup needs none of these credentials.

## SQLite Migration Required

Legacy JSON is never imported automatically:

```powershell
python -m financial_research_agent storage-migrate --pretty
python -m financial_research_agent storage-check --full --pretty
```

Keep the generated backup until migrated data has been inspected.

## SQLite Busy Or Maintenance Lock

Stop app, background CLI, and other processes using the same `FRA_HOME`, then retry. Restore
uses a sidecar maintenance lock so concurrent writes cannot be lost.

Only after confirming no restore or app process is active, a stale lock from an interrupted
process may be removed from:

```text
FRA_HOME/data/financial_research_agent.sqlite3.maintenance.lock
```

Never remove this file while restore is running. Run `storage-check --full --pretty` after
recovering from an interrupted maintenance operation.

## Reset Local State

Preview cleanup where supported. Full reset requires explicit confirmation:

```powershell
python scripts/dev.py reset --yes
```

Docker data persists after `docker compose down`. Deleting the `fra-data` volume is a separate,
destructive operation and is not performed by project scripts.
