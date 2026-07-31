CREATE TABLE narrative_presentations (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES orchestrator_runs(id) ON DELETE CASCADE,
    report_id TEXT NOT NULL,
    synthesis_sha256 TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE UNIQUE INDEX narrative_presentations_cache_key_idx
ON narrative_presentations(
    run_id,
    synthesis_sha256,
    prompt_id,
    prompt_version,
    provider,
    model
);

CREATE INDEX narrative_presentations_run_created_idx
ON narrative_presentations(run_id, created_at DESC);
