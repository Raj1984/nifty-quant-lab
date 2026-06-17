-- ============================================================
-- NIFTY Quant Lab — PostgreSQL Initialization
-- Runs once on first container start
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;         -- text search
CREATE EXTENSION IF NOT EXISTS btree_gin;        -- GIN index on btree types

-- Schemas (optional: separate raw vs computed data)
-- CREATE SCHEMA IF NOT EXISTS market_data;
-- CREATE SCHEMA IF NOT EXISTS analytics;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE nifty_quant_lab TO postgres;

-- Performance tuning hints (applied at session level in .env or postgresql.conf)
-- work_mem = 256MB
-- shared_buffers = 1GB
-- effective_cache_size = 3GB
-- maintenance_work_mem = 256MB

SELECT 'NIFTY Quant Lab DB initialized.' AS status;
