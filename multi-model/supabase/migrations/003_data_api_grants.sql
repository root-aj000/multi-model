-- ============================================================
-- Migration 003: Data API Grants
-- ============================================================
-- Target: Supabase PostgreSQL 15+
-- Purpose: Grant Data API role permissions so that supabase-py
--          can access tables and functions through the Data API.
--
-- Per supabase-py docs §"Enable Data API access":
--   "supabase-py uses the Data API to query and mutate your
--    Postgres data. You first need to grant Data API roles
--    permissions to access your tables and functions."
--
-- These grants work alongside RLS policies — RLS still enforces
-- row-level access. The grants only enable the Data API layer
-- to reach the tables at all.
-- ============================================================

-- ============================================================
-- Table grants
-- ============================================================
-- RLS policies control WHICH ROWS each role can see.
-- These GRANTs control WHETHER the Data API can reach the table.
-- Without these, supabase-py queries return permission errors
-- even if RLS policies would allow the rows.
-- ============================================================

-- Tenants
GRANT SELECT ON tenants TO anon, authenticated;
GRANT ALL ON tenants TO service_role;

-- Users
GRANT SELECT ON users TO anon, authenticated;
GRANT ALL ON users TO service_role;

-- Predictions
GRANT SELECT, INSERT ON predictions TO anon, authenticated;
GRANT ALL ON predictions TO service_role;

-- API Keys
GRANT SELECT, INSERT, UPDATE ON api_keys TO anon, authenticated;
GRANT ALL ON api_keys TO service_role;

-- Key Usage Daily
GRANT SELECT, INSERT ON key_usage_daily TO anon, authenticated;
GRANT ALL ON key_usage_daily TO service_role;

-- Healthcheck (from migration 002)
GRANT SELECT ON _healthcheck TO anon, authenticated, service_role;

-- ============================================================
-- Function grants
-- ============================================================
-- Functions must also be granted execute access for the Data API
-- to call them via supabase-py's .rpc() method.
-- ============================================================

-- increment_key_usage (from migration 001)
GRANT EXECUTE ON FUNCTION increment_key_usage(UUID, DATE) TO anon, authenticated, service_role;

-- ping (from migration 002)
GRANT EXECUTE ON FUNCTION ping() TO anon, authenticated, service_role;
