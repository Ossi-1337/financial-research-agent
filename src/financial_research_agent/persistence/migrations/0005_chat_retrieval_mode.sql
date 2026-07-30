ALTER TABLE chat_sessions
ADD COLUMN retrieval_mode TEXT NOT NULL DEFAULT 'auto'
CHECK(retrieval_mode IN ('auto', 'off', 'required'));
