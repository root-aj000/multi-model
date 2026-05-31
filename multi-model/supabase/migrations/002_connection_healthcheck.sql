-- ============================================================
-- Migration 002: Connection healthcheck helpers
-- ============================================================
-- Target: Supabase PostgreSQL 15+
-- Purpose: Provides lightweight query targets for the
--          check_supabase_connection.py script so it can
--          verify database reachability without depending
--          on application tables that may not exist yet.
-- ============================================================

-- ============================================================
-- Table: _healthcheck
-- ============================================================
-- A single-row, single-column table. The cheapest possible
-- database round-trip. The CHECK constraint ensures only
-- row id=1 can ever exist.
-- ============================================================
CREATE TABLE IF NOT EXISTS _healthcheck (
    id    INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    alive BOOLEAN NOT NULL DEFAULT TRUE
);

-- Insert the singleton row (idempotent — safe to re-run)
INSERT INTO _healthcheck (id, alive)
VALUES (1, TRUE)
ON CONFLICT (id) DO NOTHING;

-- ============================================================
-- RLS: Allow everyone to read _healthcheck
-- ============================================================
-- This table exists solely for connectivity testing.
-- All roles (anon, authenticated, service_role) must be
-- able to SELECT from it, otherwise the anon-key check
-- in the script would fail.
-- ============================================================
ALTER TABLE _healthcheck ENABLE ROW LEVEL SECURITY;

CREATE POLICY "_healthcheck_select_anyone"
    ON _healthcheck FOR SELECT
    TO anon, authenticated, service_role
    USING (TRUE);

-- ============================================================
-- RPC: ping()
-- ============================================================
-- Even lighter than a table scan — no disk access at all.
-- Returns the string 'pong' so the caller can distinguish
-- a real response from a cached/proxy response.
-- ============================================================
CREATE OR REPLACE FUNCTION ping()
RETURNS TEXT AS $$
    SELECT 'pong';
$$ LANGUAGE SQL SECURITY DEFINER;
