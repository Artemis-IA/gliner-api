-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop the documents table and any objects that depend on it
DROP TABLE IF EXISTS documents CASCADE;

-- Create the documents table with vector support for embedding
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    company TEXT,
    publication_date DATE,
    pdf_link TEXT NOT NULL,
    local_path TEXT,
    vector VECTOR(768)
);

-- Table for storing search terms
CREATE TABLE IF NOT EXISTS search_terms (
    id SERIAL PRIMARY KEY,
    term TEXT NOT NULL
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_pdf_link ON documents(pdf_link);
CREATE INDEX IF NOT EXISTS idx_title ON documents USING GIN (to_tsvector('french', title));
