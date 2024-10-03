-- init.sql

CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    data JSON NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS inferences (
    id SERIAL PRIMARY KEY,
    file_path VARCHAR(255) NOT NULL,
    entities JSON NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) UNIQUE NOT NULL,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    epochs INTEGER DEFAULT 10,
    batch_size INTEGER DEFAULT 32,
    status VARCHAR(50) DEFAULT 'Started',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
