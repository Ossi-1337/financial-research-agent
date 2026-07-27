# Local Development

Financial Research Agent supports an offline Docker path, optional llama.cpp profiles, and a
direct Python development path. Python 3.14 is required.

## Docker: Offline UI

```powershell
docker compose up --build
```

Open `http://127.0.0.1:8000`. This starts only the app with `offline-test`; no model is
downloaded. Structured state is stored in the `fra-data` volume.

Stop containers without deleting local data:

```powershell
docker compose down
```

## Docker: Local LLM

Use one profile at a time:

```powershell
python scripts/dev.py docker-up --runtime cpu
python scripts/dev.py docker-up --runtime cuda
```

Add `--detach` to run in the background. Both profiles expose llama.cpp at
`http://127.0.0.1:8080`; the app uses `http://llama-cpp:8080/v1` inside Compose.

The CPU default is `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M`. It is suitable for connection and
chat smoke tests, not high-quality financial research. The CUDA default is
`unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL`.

Override models or cache location in `.env`:

```dotenv
FRA_CPU_MODEL=ggml-org/gemma-3-1b-it-GGUF:Q4_K_M
FRA_CUDA_MODEL=unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL
```

Model files are downloaded by llama.cpp only when a model profile starts. They are never
bundled in the app image or repository. Set `FRA_HUGGINGFACE_CACHE` to an absolute local cache
directory when the Compose default should be overridden.

## Direct Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python scripts/dev.py install
python scripts/dev.py run
```

The UI is available at `http://127.0.0.1:8000`. Direct Python starts with `offline-test`
unless environment or `.env` settings select another provider.

Common developer commands:

```powershell
python scripts/dev.py lint
python scripts/dev.py test
python scripts/dev.py check
python scripts/dev.py reset --yes
```

`check` runs lint, format validation, tests, compilation, deterministic evaluations, and a
source/wheel build. Docker validation remains separate so Python-only development does not
require Docker Desktop.

## Environment Precedence

Copy `.env.example` to `.env` only when local overrides are needed. CLI commands load the
`.env` in the current working directory with `override=False`:

1. Existing process or container environment variables win.
2. Missing values may be read from `.env`.
3. Code defaults apply last.

Set `PYTHON_DOTENV_DISABLED=1` to disable `.env` loading. Secrets must remain outside Git.

## Package Build

```powershell
python -m build
```

This creates a source distribution and wheel under `dist/`. The installed command is:

```powershell
financial-research-agent --pretty
```

Package publishing is intentionally not configured.
