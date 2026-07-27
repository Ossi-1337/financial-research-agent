UPDATE a2a_tasks
SET owner = 'company-research'
WHERE owner = 'local';

CREATE INDEX IF NOT EXISTS idx_a2a_tasks_owner_state_updated
ON a2a_tasks(owner, state, updated_at DESC);

CREATE TABLE IF NOT EXISTS a2a_delegations (
    id TEXT PRIMARY KEY,
    orchestrator_run_id TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    agent_role TEXT NOT NULL,
    service_id TEXT NOT NULL,
    remote_task_id TEXT,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL CHECK(attempt_count > 0),
    error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(orchestrator_run_id) REFERENCES orchestrator_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_a2a_delegations_run_role
ON a2a_delegations(orchestrator_run_id, agent_role, updated_at);
