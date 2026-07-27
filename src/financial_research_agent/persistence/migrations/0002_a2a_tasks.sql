CREATE TABLE a2a_tasks (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    context_id TEXT NOT NULL,
    state INTEGER NOT NULL,
    initial_message_id TEXT NOT NULL UNIQUE,
    background_job_id TEXT,
    orchestrator_run_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload BLOB NOT NULL
);
CREATE INDEX a2a_tasks_updated_at_idx ON a2a_tasks(updated_at DESC, id DESC);

CREATE TABLE a2a_task_events (
    task_id TEXT NOT NULL REFERENCES a2a_tasks(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL CHECK(sequence >= 0),
    event_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(task_id, sequence)
);
CREATE INDEX a2a_task_events_task_sequence_idx
    ON a2a_task_events(task_id, sequence);
