-- ============================================================
-- Migration 001: Initial multi-tenant schema
-- ============================================================
-- Target: Supabase PostgreSQL 15+
-- Author: Prototype Specification
-- Review: Senior Engineering Review Required
-- ============================================================

-- Enable UUID generation (Supabase has this by default)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- Table: tenants
-- ============================================================
CREATE TABLE tenants (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          VARCHAR(255) NOT NULL,
    slug          VARCHAR(100) NOT NULL UNIQUE,
    plan          VARCHAR(50) NOT NULL DEFAULT 'free'
                    CHECK (plan IN ('free', 'pro', 'enterprise')),
    settings      JSONB NOT NULL DEFAULT '{}',
    monthly_limit INTEGER NOT NULL DEFAULT 100
                    CHECK (monthly_limit > 0),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index for slug lookups (login flow resolves tenant by slug)
CREATE INDEX idx_tenants_slug ON tenants (slug);

-- Trigger: auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_tenants_updated_at
    BEFORE UPDATE ON tenants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- Table: users
-- ============================================================
-- NOTE: Supabase Auth provides its own auth.users table.
-- We create a public.users table that references auth.users
-- via a 1:1 relationship. This is the Supabase-recommended pattern.
-- See: https://supabase.com/docs/guides/auth/managing-user-data
-- ============================================================
CREATE TABLE users (
    id            UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    tenant_id     UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email         VARCHAR(255) NOT NULL,
    display_name  VARCHAR(255),
    role          VARCHAR(50) NOT NULL DEFAULT 'member'
                    CHECK (role IN ('owner', 'admin', 'member')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- One email per tenant
    UNIQUE (tenant_id, email)
);

-- Index: find all users in a tenant
CREATE INDEX idx_users_tenant_id ON users (tenant_id);

-- Index: find a user by email within a tenant
CREATE INDEX idx_users_tenant_email ON users (tenant_id, email);

-- Trigger: auto-update updated_at
CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- Table: predictions
-- ============================================================
CREATE TABLE predictions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id     UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename      VARCHAR(500),
    ocr_text      TEXT,
    result        JSONB NOT NULL,
    processing_ms INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index: list predictions for a tenant (paginated, sorted by created_at desc)
CREATE INDEX idx_predictions_tenant_created
    ON predictions (tenant_id, created_at DESC);

-- Index: find predictions by user
CREATE INDEX idx_predictions_user_id
    ON predictions (user_id, created_at DESC);

-- Index: GIN index for JSONB result queries (attribute filtering)
CREATE INDEX idx_predictions_result_gin
    ON predictions USING GIN (result jsonb_path_ops);

-- ============================================================
-- Table: api_keys
-- ============================================================
CREATE TABLE api_keys (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id     UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash      VARCHAR(255) NOT NULL UNIQUE,
    key_prefix    VARCHAR(10) NOT NULL,
    name          VARCHAR(255) NOT NULL,
    permissions   JSONB NOT NULL DEFAULT '["predict"]'::jsonb,
    expires_at    TIMESTAMPTZ,
    last_used_at  TIMESTAMPTZ,
    revoked_at    TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index: look up an API key by its hash (auth middleware hot path)
CREATE INDEX idx_api_keys_key_hash ON api_keys (key_hash);

-- Index: list keys for a tenant
CREATE INDEX idx_api_keys_tenant_id ON api_keys (tenant_id);

-- Index: find active (non-revoked, non-expired) keys
CREATE INDEX idx_api_keys_active
    ON api_keys (tenant_id, revoked_at)
    WHERE revoked_at IS NULL;

-- ============================================================
-- Table: key_usage_daily
-- ============================================================
-- Aggregated daily usage per API key.
-- Why aggregated? Because a high-traffic key could generate
-- millions of individual request rows. Daily aggregates keep
-- the table small and queries fast.
-- ============================================================
CREATE TABLE key_usage_daily (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_id    UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    date          DATE NOT NULL DEFAULT CURRENT_DATE,
    request_count INTEGER NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- One row per key per date
    UNIQUE (api_key_id, date)
);

-- Index: usage chart for a specific key
CREATE INDEX idx_key_usage_api_key_date
    ON key_usage_daily (api_key_id, date DESC);

-- ============================================================
-- RPC: Increment key usage (atomic upsert)
-- ============================================================
CREATE OR REPLACE FUNCTION increment_key_usage(
    key_id UUID,
    usage_date DATE
)
RETURNS void AS $$
BEGIN
    INSERT INTO key_usage_daily (api_key_id, date, request_count)
    VALUES (key_id, usage_date, 1)
    ON CONFLICT (api_key_id, date)
    DO UPDATE SET request_count = key_usage_daily.request_count + 1;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================
-- RLS: Enable RLS on all tenant-scoped tables
-- ============================================================
-- By default, RLS DENIES all access. Policies must explicitly
-- grant access. This is the "default deny" security posture.
-- ============================================================

ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE key_usage_daily ENABLE ROW LEVEL SECURITY;

-- ============================================================
-- RLS Helper: Extract tenant_id from JWT
-- ============================================================
-- Supabase Auth stores custom claims in auth.jwt().
-- We inject tenant_id during login via a trigger on auth.users.
--
-- NOTE: Created in the `public` schema (not `auth`) because the
-- `auth` schema is owned by supabase_auth_admin and cannot be
-- written to from the dashboard SQL editor. The function still
-- calls auth.jwt() which is a built-in Supabase function
-- accessible from any schema — behavior is identical.
-- ============================================================

CREATE OR REPLACE FUNCTION public.tenant_id() RETURNS UUID AS $$
  -- Extract tenant_id from the JWT claims set by Supabase Auth
  SELECT (auth.jwt() -> 'app_metadata' ->> 'tenant_id')::UUID;
$$ LANGUAGE SQL STABLE;

-- ============================================================
-- RLS: tenants
-- ============================================================
-- Users can read their own tenant. Owners/admins can update.
-- No one can delete a tenant (soft-delete via status column in future).
-- ============================================================

CREATE POLICY "Users can read own tenant"
    ON tenants FOR SELECT
    USING (id = public.tenant_id());

CREATE POLICY "Owners and admins can update own tenant"
    ON tenants FOR UPDATE
    USING (
        id = public.tenant_id()
        AND EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid()
            AND users.tenant_id = public.tenant_id()
            AND users.role IN ('owner', 'admin')
        )
    );

-- ============================================================
-- RLS: users
-- ============================================================
-- Users can read other users in their tenant (needed for team page).
-- Users can update their own profile.
-- Only owners/admins can insert (invite) or delete (remove) users.
-- ============================================================

CREATE POLICY "Users can read users in own tenant"
    ON users FOR SELECT
    USING (tenant_id = public.tenant_id());

CREATE POLICY "Users can update own profile"
    ON users FOR UPDATE
    USING (id = auth.uid() AND tenant_id = public.tenant_id());

CREATE POLICY "Owners and admins can invite users"
    ON users FOR INSERT
    WITH CHECK (
        tenant_id = public.tenant_id()
        AND EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid()
            AND users.tenant_id = public.tenant_id()
            AND users.role IN ('owner', 'admin')
        )
    );

CREATE POLICY "Owners and admins can remove users"
    ON users FOR DELETE
    USING (
        tenant_id = public.tenant_id()
        AND EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid()
            AND users.tenant_id = public.tenant_id()
            AND users.role IN ('owner', 'admin')
        )
        -- Cannot delete yourself
        AND id != auth.uid()
    );

-- ============================================================
-- RLS: predictions
-- ============================================================
-- Users can read predictions in their tenant.
-- Users can insert (create) predictions in their tenant.
-- Users can delete their own predictions.
-- No one can update predictions (immutable record).
-- ============================================================

CREATE POLICY "Users can read predictions in own tenant"
    ON predictions FOR SELECT
    USING (tenant_id = public.tenant_id());

CREATE POLICY "Users can create predictions in own tenant"
    ON predictions FOR INSERT
    WITH CHECK (tenant_id = public.tenant_id());

CREATE POLICY "Users can delete own predictions"
    ON predictions FOR DELETE
    USING (
        tenant_id = public.tenant_id()
        AND user_id = auth.uid()
    );

-- Predictions are immutable — no UPDATE policy

-- ============================================================
-- RLS: api_keys
-- ============================================================
-- Users can read API keys in their tenant.
-- Users can create API keys in their tenant.
-- Users can update (revoke) their own API keys.
-- Owners/admins can revoke any key in the tenant.
-- ============================================================

CREATE POLICY "Users can read API keys in own tenant"
    ON api_keys FOR SELECT
    USING (tenant_id = public.tenant_id());

CREATE POLICY "Users can create API keys in own tenant"
    ON api_keys FOR INSERT
    WITH CHECK (tenant_id = public.tenant_id());

CREATE POLICY "Users can revoke own API keys"
    ON api_keys FOR UPDATE
    USING (
        tenant_id = public.tenant_id()
        AND (user_id = auth.uid()
             OR EXISTS (
                SELECT 1 FROM users
                WHERE users.id = auth.uid()
                AND users.tenant_id = public.tenant_id()
                AND users.role IN ('owner', 'admin')
             ))
    );

-- ============================================================
-- RLS: key_usage_daily
-- ============================================================
-- Users can read usage data for keys in their tenant.
-- The backend service role can insert usage data (not user-facing).
-- ============================================================

CREATE POLICY "Users can read key usage in own tenant"
    ON key_usage_daily FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM api_keys
            WHERE api_keys.id = key_usage_daily.api_key_id
            AND api_keys.tenant_id = public.tenant_id()
        )
    );

-- ============================================================
-- Service Role Bypass
-- ============================================================
-- The Supabase service_role key bypasses ALL RLS policies.
-- This is used ONLY by the backend for:
-- 1. API key hash lookup during auth middleware (performance)
-- 2. Admin endpoints that need cross-tenant visibility
-- 3. Key usage incrementing (happens after the request completes)
--
-- NEVER expose the service_role key to the frontend.
-- ============================================================

-- ============================================================
-- Supabase Auth Trigger — Auto-Populate tenant_id in JWT Claims
-- ============================================================

-- Step 1: Auto-create public.users row when auth.users row is created
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.users (id, tenant_id, email, display_name, role)
    VALUES (
        NEW.id,
        (NEW.raw_user_meta_data->>'tenant_id')::UUID,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'display_name', split_part(NEW.email, '@', 1)),
        COALESCE(NEW.raw_user_meta_data->>'role', 'member')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Step 2: When a user is created in public.users, update
-- their auth.jwt() claims to include tenant_id
CREATE OR REPLACE FUNCTION public.set_tenant_claim()
RETURNS TRIGGER AS $$
BEGIN
    -- Update the user's app_metadata to include tenant_id
    -- This makes tenant_id available in auth.jwt() -> 'app_metadata'
    UPDATE auth.users
    SET app_metadata = app_metadata || jsonb_build_object('tenant_id', NEW.tenant_id)
    WHERE id = NEW.id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_public_user_created
    AFTER INSERT ON public.users
    FOR EACH ROW EXECUTE FUNCTION public.set_tenant_claim();
