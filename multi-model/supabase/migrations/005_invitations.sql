-- ============================================================================
-- Migration 005: Invitations table
-- ============================================================================
-- Creates the invitations table for email-based team invitations.
-- Owners/admins can invite users by email; invited users register via
-- a special /invite?token=xxx page and join the inviter's tenant as members.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.invitations (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    tenant_id   UUID NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    email       TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'member',
    token       TEXT NOT NULL UNIQUE,
    invited_by  UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    status      TEXT NOT NULL DEFAULT 'pending',
    expires_at  TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Index for fast token lookup (the verify endpoint hits this)
CREATE INDEX IF NOT EXISTS idx_invitations_token ON public.invitations(token);

-- Index for listing invites by tenant
CREATE INDEX IF NOT EXISTS idx_invitations_tenant_id ON public.invitations(tenant_id);

-- Prevent duplicate pending invites for same email+tenant
CREATE UNIQUE INDEX IF NOT EXISTS idx_invitations_pending_unique
    ON public.invitations(email, tenant_id)
    WHERE status = 'pending';

-- Enable RLS
ALTER TABLE public.invitations ENABLE ROW LEVEL SECURITY;

-- RLS policies: only admin/owner users in the tenant can manage invitations
CREATE POLICY "Admins can view invitations in their tenant"
    ON public.invitations FOR SELECT
    USING (
        tenant_id IN (
            SELECT u.tenant_id FROM public.users u
            WHERE u.id = auth.uid() AND u.role IN ('owner', 'admin', 'platform_admin')
        )
    );

CREATE POLICY "Admins can insert invitations in their tenant"
    ON public.invitations FOR INSERT
    WITH CHECK (
        tenant_id IN (
            SELECT u.tenant_id FROM public.users u
            WHERE u.id = auth.uid() AND u.role IN ('owner', 'admin', 'platform_admin')
        )
    );

CREATE POLICY "Admins can update invitations in their tenant"
    ON public.invitations FOR UPDATE
    USING (
        tenant_id IN (
            SELECT u.tenant_id FROM public.users u
            WHERE u.id = auth.uid() AND u.role IN ('owner', 'admin', 'platform_admin')
        )
    );

CREATE POLICY "Admins can delete invitations in their tenant"
    ON public.invitations FOR DELETE
    USING (
        tenant_id IN (
            SELECT u.tenant_id FROM public.users u
            WHERE u.id = auth.uid() AND u.role IN ('owner', 'admin', 'platform_admin')
        )
    );

-- Grant access to the service role (backend uses admin client)
GRANT ALL ON public.invitations TO service_role;
GRANT SELECT ON public.invitations TO anon;
GRANT SELECT ON public.invitations TO authenticated;
