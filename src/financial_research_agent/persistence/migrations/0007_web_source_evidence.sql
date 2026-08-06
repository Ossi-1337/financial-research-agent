CREATE TABLE web_source_evidence (
    id TEXT PRIMARY KEY,
    canonical_url TEXT NOT NULL UNIQUE,
    retrieved_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    payload_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE INDEX web_source_evidence_expires_at_idx
ON web_source_evidence(expires_at);
