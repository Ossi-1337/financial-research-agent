# Demo Walkthrough

This walkthrough demonstrates one real-data Novo Nordisk research run in roughly 5-10 minutes
after dependencies and any local model are ready. It does not demonstrate general financial
accuracy.

## 1. Prepare An Isolated Environment

Create and activate a Python 3.14 virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python scripts/dev.py install
```

Copy `.env.example` to `.env`. Set:

- `FRA_ALPHA_VANTAGE_API_KEY` to a valid key.
- `FRA_SEC_USER_AGENT` to an application identifier plus a real contact email.
- Optional local LLM settings when source-bounded Q&A is part of the demo.

Keep `.env` local. Never paste credentials into chat, screenshots, issues, or commits.

Use an isolated data root for the demo:

```powershell
$env:FRA_HOME = Join-Path $PWD ".demo-data"
python -m financial_research_agent storage-check --pretty
```

## 2. Run Real-Data Preflight

Run the deterministic scenario:

```powershell
python -m financial_research_agent scenario-run novo-nordisk --pretty
```

Inspect the scenario checks. A useful run must show:

- CIK `0000353278`.
- Primary security `NVO` on `NYSE`.
- Non-empty NVO and SPY histories.
- At least two annual IFRS/DKK statement periods.
- One stored `20-F` and one stored `6-K`.
- Specialist and synthesis handoffs.
- Resolved evidence.
- Markdown, HTML, and PDF artifacts.

Freshness or source limitations may produce a partial result. Failed required checks must not be
treated as complete.

## 3. Start The UI

```powershell
python scripts/dev.py run
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Enter:

```text
/scenario novo-nordisk
```

The UI queues a background run, renders progress, then shows the timestamped deterministic
report. Material findings have source markers. The NVO/SPY chart uses only shared trading dates
and indexes both series to 100 at the first shared date.

![Deterministic report](assets/demo/novo-report-desktop.jpg)

## 4. Inspect Evidence

Open source markers or the evidence panel. Verify source URL, publication/data-as-of date,
retrieval date, section, bounded quote, and original evidence ID.

![Source evidence panel](assets/demo/novo-evidence-desktop.jpg)

Do not treat a marker as proof by itself. Open the linked official source before relying on a
material claim.

## 5. Optional Local Q&A

Start a local OpenAI-compatible endpoint such as llama.cpp, then configure the app for
`local-openai` before starting the direct Python UI. The required provider, model, and endpoint
settings are documented in the repository `.env.example`.

Alternatively, stop the direct Python server and start the complete app plus llama.cpp stack
with `python scripts/dev.py docker-up --runtime cuda --detach`. Use the CPU profile when CUDA is
unavailable. Do not run the direct and Docker app servers on port `8000` at the same time.

After the UI reports `local-openai`, enter:

```text
/scenario novo-nordisk --with-local-qa
```

The answer is labeled `LLM-generated source-bounded Q&A`, cites available source markers, and
does not modify deterministic report content or acceptance status.

![Mobile source-bounded Q&A](assets/demo/novo-local-qa-mobile.jpg)

## 6. Open Exports

Select `Export` on the synthesis report. Open all three artifacts:

- Markdown for source review and versionable text.
- Self-contained HTML for local sharing.
- PDF for a fixed document snapshot.

Each immutable export has its own ID, hashes, timestamps, limitations, no-advice notice, and
source appendix.

## 7. Cleanup

Stop direct Python with `Ctrl+C`. Stop Docker containers without deleting the data volume:

```powershell
python scripts/dev.py docker-down
```

Reset the isolated application data only when it is no longer needed:

```powershell
python scripts/dev.py reset --yes
```

Do not publish the local database, cached filing documents, provider data, logs, or `.env`.
