ALTER TABLE filings ADD COLUMN extraction_status TEXT;
ALTER TABLE filings ADD COLUMN extraction_method TEXT;
ALTER TABLE filings ADD COLUMN page_count INTEGER;

ALTER TABLE filing_chunks ADD COLUMN page_number INTEGER;
ALTER TABLE filing_chunks ADD COLUMN region_json TEXT;
ALTER TABLE filing_chunks ADD COLUMN extraction_method TEXT;

CREATE INDEX filing_chunks_page_idx
ON filing_chunks(filing_id, page_number, chunk_index);
