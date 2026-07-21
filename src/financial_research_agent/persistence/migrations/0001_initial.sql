CREATE TABLE app_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE companies (
    id TEXT PRIMARY KEY,
    legal_name TEXT,
    cik TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX companies_cik_unique ON companies(cik) WHERE cik IS NOT NULL;

CREATE TABLE company_identifiers (
    company_id TEXT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    scheme TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (scheme, value)
);

CREATE TABLE securities (
    id TEXT PRIMARY KEY,
    company_id TEXT REFERENCES companies(id) ON DELETE SET NULL,
    symbol TEXT NOT NULL,
    exchange_mic TEXT,
    currency TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE security_identifiers (
    security_id TEXT NOT NULL REFERENCES securities(id) ON DELETE CASCADE,
    scheme TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (scheme, value)
);

CREATE TABLE chat_sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    summary TEXT
);
CREATE INDEX chat_sessions_updated_at_idx ON chat_sessions(updated_at DESC);

CREATE TABLE chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL CHECK(sequence >= 0),
    role TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(session_id, sequence)
);

CREATE TABLE market_series (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    symbol TEXT NOT NULL,
    security_id TEXT REFERENCES securities(id) ON DELETE SET NULL,
    retrieved_at TEXT NOT NULL,
    payload_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(provider, symbol)
);

CREATE TABLE price_bars (
    series_id TEXT NOT NULL REFERENCES market_series(id) ON DELETE CASCADE,
    priced_at TEXT NOT NULL,
    open_value TEXT NOT NULL,
    high_value TEXT NOT NULL,
    low_value TEXT NOT NULL,
    close_value TEXT NOT NULL,
    adjusted_close_value TEXT,
    volume INTEGER NOT NULL,
    PRIMARY KEY(series_id, priced_at)
);

CREATE TABLE statement_results (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    cik TEXT NOT NULL,
    company_id TEXT REFERENCES companies(id) ON DELETE SET NULL,
    retrieved_at TEXT NOT NULL,
    payload_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(provider, cik)
);

CREATE TABLE financial_statements (
    id TEXT PRIMARY KEY,
    result_id TEXT NOT NULL REFERENCES statement_results(id) ON DELETE CASCADE,
    statement_type TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    period_end TEXT NOT NULL,
    currency TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE filing_results (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    cik TEXT NOT NULL,
    company_id TEXT REFERENCES companies(id) ON DELETE SET NULL,
    retrieved_at TEXT NOT NULL,
    payload_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(provider, cik)
);

CREATE TABLE filings (
    id TEXT PRIMARY KEY,
    result_id TEXT NOT NULL REFERENCES filing_results(id) ON DELETE CASCADE,
    accession_number TEXT NOT NULL UNIQUE,
    form_type TEXT NOT NULL,
    filing_date TEXT NOT NULL,
    local_raw_path TEXT NOT NULL,
    local_text_path TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE filing_chunks (
    id TEXT PRIMARY KEY,
    filing_id TEXT NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    section_heading TEXT,
    metadata_json TEXT NOT NULL,
    UNIQUE(filing_id, chunk_index)
);

CREATE TABLE cited_runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    query TEXT NOT NULL,
    payload_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE citations (
    run_id TEXT NOT NULL REFERENCES cited_runs(id) ON DELETE CASCADE,
    citation_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(run_id, citation_id)
);

CREATE TABLE evidence_snippets (
    run_id TEXT NOT NULL REFERENCES cited_runs(id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(run_id, evidence_id)
);

CREATE TABLE orchestrator_runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    company_id TEXT REFERENCES companies(id) ON DELETE SET NULL,
    security_id TEXT REFERENCES securities(id) ON DELETE SET NULL,
    payload_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX orchestrator_runs_updated_at_idx ON orchestrator_runs(updated_at DESC);

CREATE TABLE agent_handoffs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES orchestrator_runs(id) ON DELETE CASCADE,
    step_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE background_jobs (
    id TEXT PRIMARY KEY,
    orchestrator_run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    payload_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX background_jobs_updated_at_idx ON background_jobs(updated_at DESC);

CREATE TABLE runtime_settings (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    updated_at TEXT NOT NULL,
    payload_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE import_records (
    id TEXT PRIMARY KEY,
    imported_at TEXT NOT NULL,
    backup_id TEXT NOT NULL,
    source_manifest_json TEXT NOT NULL,
    counts_json TEXT NOT NULL
);
