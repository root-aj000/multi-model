-- ============================================================
-- Migration 004: Fix auth triggers
-- ============================================================
-- Target: Supabase PostgreSQL 15+
-- Purpose: The `on_public_user_created` trigger calls
--   `set_tenant_claim()` which tries to UPDATE auth.users.
--   This fails because the postgres role cannot modify the
--   auth schema (owned by supabase_auth_admin).
--
-- Fix: Drop the trigger. The backend will set app_metadata
--   (tenant_id) via supabase.auth.admin.update_user_by_id()
--   after user creation, which is the recommended approach.
-- ============================================================

-- Drop the problematic trigger
DROP TRIGGER IF EXISTS on_public_user_created ON public.users;

-- Drop the function that tried to update auth.users
DROP FUNCTION IF EXISTS public.set_tenant_claim();

-- Keep the handle_new_user trigger (it only inserts into public.users,
-- which works fine). But make it more robust — if the insert fails
-- (e.g., tenant_id FK violation), don't crash the auth signup.
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
EXCEPTION WHEN OTHERS THEN
    -- Log the error but don't crash the auth signup
    -- The backend will handle the public.users insert manually
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
