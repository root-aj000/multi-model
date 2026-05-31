"""
Invitation API router.

Provides endpoints for managing team invitations:
- Create invite (owner/admin sends email invite)
- List invites (owner/admin views pending invites)
- Cancel invite (owner/admin cancels a pending invite)
- Verify invite token (public — validates token and returns invite info)
- List team members (owner/admin views all members)
- Remove team member (owner/admin removes a member)

Invited users register via /auth/register-invite (in auth_router.py).
"""

import logging
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, field_validator

from lib.auth.config import (
    get_supabase_admin_client,
    is_auth_enabled,
    get_frontend_url,
)
from lib.auth.deps import require_admin
from lib.auth.middleware import RequestContext
from lib.auth.email import send_invite_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/invites", tags=["Invitations"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
INVITE_EXPIRY_DAYS = 7

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class CreateInviteRequest(BaseModel):
    """Request to invite a new member by email."""
    email: str
    role: str = "member"

    @field_validator("email")
    @classmethod
    def email_format(cls, v: str) -> str:
        if not _EMAIL_RE.match(v):
            raise ValueError("Invalid email format")
        return v

    @field_validator("role")
    @classmethod
    def role_value(cls, v: str) -> str:
        if v not in ("member", "admin"):
            raise ValueError("Role must be 'member' or 'admin'")
        return v


class InvitationResponse(BaseModel):
    """Invitation data returned in API responses."""
    id: str
    email: str
    role: str
    status: str
    expires_at: str
    created_at: str
    invite_link: str | None = None


class InviteVerifyResponse(BaseModel):
    """Response from verifying an invite token."""
    valid: bool
    email: Optional[str] = None
    tenant_name: Optional[str] = None
    role: Optional[str] = None
    error: Optional[str] = None


class TeamMemberResponse(BaseModel):
    """Team member data."""
    id: str
    email: str
    display_name: Optional[str] = None
    role: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_auth_enabled() -> None:
    """Raise 503 if auth is disabled."""
    if not is_auth_enabled():
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=InvitationResponse, status_code=status.HTTP_201_CREATED)
async def create_invite(
    request: CreateInviteRequest,
    ctx: RequestContext = Depends(require_admin),
) -> InvitationResponse:
    """
    Create an invitation for a new team member.

    Generates a secure token, stores the invitation in the database,
    and sends an email with the invite link.

    Args:
        request: Email and role for the invited user.
        ctx: Authenticated admin/owner context.

    Returns:
        InvitationResponse with the created invitation details.

    Raises:
        HTTPException: 409 if a pending invite already exists for this email+tenant.
        HTTPException: 409 if the email is already a member of this tenant.
    """
    _require_auth_enabled()

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured.",
        )

    # ── Check if user is already a member ────────────────────────────────
    existing_user = admin_client.table("users").select("id").eq(
        "email", request.email
    ).eq("tenant_id", ctx.tenant_id).execute()

    if existing_user.data:
        raise HTTPException(
            status_code=409,
            detail="This email is already a member of your workspace.",
        )

    # ── Check for existing pending invite ────────────────────────────────
    existing_invite = admin_client.table("invitations").select("id").eq(
        "email", request.email
    ).eq("tenant_id", ctx.tenant_id).eq("status", "pending").execute()

    if existing_invite.data:
        raise HTTPException(
            status_code=409,
            detail="A pending invitation already exists for this email.",
        )

    # ── Create invitation ───────────────────────────────────────────────
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=INVITE_EXPIRY_DAYS)

    invite_data = {
        "tenant_id": ctx.tenant_id,
        "email": request.email,
        "role": request.role,
        "token": token,
        "invited_by": ctx.user_id,
        "status": "pending",
        "expires_at": expires_at.isoformat(),
    }

    try:
        result = admin_client.table("invitations").insert(invite_data).execute()
    except Exception as exc:
        # Handle unique constraint violation (race condition)
        if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
            raise HTTPException(
                status_code=409,
                detail="A pending invitation already exists for this email.",
            )
        logger.error("Failed to create invitation: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to create invitation.",
        )

    if not result.data:
        raise HTTPException(
            status_code=500,
            detail="Failed to create invitation.",
        )

    invite = result.data[0]

    # ── Send invite email ───────────────────────────────────────────────
    frontend_url = get_frontend_url()
    invite_link = f"{frontend_url}/invite?token={token}"

    # Look up inviter's display name
    inviter_name = "A team member"
    try:
        inviter_result = admin_client.table("users").select(
            "display_name"
        ).eq("id", ctx.user_id).execute()
        if inviter_result.data and inviter_result.data[0].get("display_name"):
            inviter_name = inviter_result.data[0]["display_name"]
    except Exception:
        pass

    # Look up tenant name
    tenant_name = "the workspace"
    try:
        tenant_result = admin_client.table("tenants").select(
            "name"
        ).eq("id", ctx.tenant_id).execute()
        if tenant_result.data:
            tenant_name = tenant_result.data[0]["name"]
    except Exception:
        pass

    email_sent = False
    try:
        send_invite_email(
            to_email=request.email,
            invite_link=invite_link,
            tenant_name=tenant_name,
            inviter_name=inviter_name,
        )
        email_sent = True
    except Exception as exc:
        logger.error("Failed to send invite email: %s", exc)
        # Don't fail the request — the invitation was created successfully.
        # The link can be manually shared.

    # Include invite_link in response when email was NOT sent,
    # so the frontend can display it for manual sharing.
    return InvitationResponse(
        id=invite["id"],
        email=invite["email"],
        role=invite["role"],
        status=invite["status"],
        expires_at=invite["expires_at"],
        created_at=invite["created_at"],
        invite_link=invite_link if not email_sent else None,
    )


@router.get("", response_model=List[InvitationResponse])
async def list_invites(
    ctx: RequestContext = Depends(require_admin),
) -> List[InvitationResponse]:
    """
    List all invitations for the current tenant.

    Returns both pending and past invitations (most recent first).

    Args:
        ctx: Authenticated admin/owner context.

    Returns:
        List of InvitationResponse objects.
    """
    _require_auth_enabled()

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured.",
        )

    try:
        result = admin_client.table("invitations").select("*").eq(
            "tenant_id", ctx.tenant_id
        ).order("created_at", desc=True).execute()

        invites = result.data or []
        return [
            InvitationResponse(
                id=inv["id"],
                email=inv["email"],
                role=inv["role"],
                status=inv["status"],
                expires_at=inv["expires_at"],
                created_at=inv["created_at"],
            )
            for inv in invites
        ]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list invitations: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to load invitations.",
        )


@router.delete("/{invite_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_invite(
    invite_id: str,
    ctx: RequestContext = Depends(require_admin),
) -> None:
    """
    Cancel a pending invitation.

    Only pending invitations can be cancelled. The invitation must
    belong to the current user's tenant.

    Args:
        invite_id: UUID of the invitation to cancel.
        ctx: Authenticated admin/owner context.

    Raises:
        HTTPException: 404 if invitation not found or not in this tenant.
        HTTPException: 409 if invitation is not pending.
    """
    _require_auth_enabled()

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured.",
        )

    # ── Lookup invitation ────────────────────────────────────────────────
    result = admin_client.table("invitations").select("*").eq("id", invite_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Invitation not found.")

    invite = result.data[0]

    # ── Verify tenant ownership ─────────────────────────────────────────
    if invite["tenant_id"] != ctx.tenant_id:
        raise HTTPException(status_code=404, detail="Invitation not found.")

    # ── Only cancel pending invites ─────────────────────────────────────
    if invite["status"] != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel invitation with status '{invite['status']}'.",
        )

    # ── Update status ───────────────────────────────────────────────────
    admin_client.table("invitations").update({
        "status": "cancelled",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", invite_id).execute()


@router.get("/verify", response_model=InviteVerifyResponse)
async def verify_invite(token: str) -> InviteVerifyResponse:
    """
    Verify an invitation token (public endpoint — no auth required).

    Validates that the token exists, is pending, and has not expired.
    Returns the invited email and tenant name for pre-filling the
    registration form.

    Args:
        token: The invitation token from the URL.

    Returns:
        InviteVerifyResponse with validity status and invite details.
    """
    if not is_auth_enabled():
        return InviteVerifyResponse(valid=False, error="Authentication is not configured.")

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        return InviteVerifyResponse(valid=False, error="Service not available.")

    # ── Lookup invitation by token ──────────────────────────────────────
    try:
        result = admin_client.table("invitations").select("*").eq(
            "token", token
        ).execute()

        if not result.data:
            return InviteVerifyResponse(valid=False, error="Invalid invitation token.")

        invite = result.data[0]
    except Exception as exc:
        logger.error("Failed to verify invitation: %s", exc)
        return InviteVerifyResponse(valid=False, error="Failed to verify invitation.")

    # ── Check status ────────────────────────────────────────────────────
    if invite["status"] != "pending":
        if invite["status"] == "accepted":
            return InviteVerifyResponse(valid=False, error="This invitation has already been used.")
        if invite["status"] == "expired":
            return InviteVerifyResponse(valid=False, error="This invitation has expired.")
        if invite["status"] == "cancelled":
            return InviteVerifyResponse(valid=False, error="This invitation has been cancelled.")
        return InviteVerifyResponse(valid=False, error="This invitation is no longer valid.")

    # ── Check expiry ────────────────────────────────────────────────────
    expires_at = invite["expires_at"]
    if isinstance(expires_at, str):
        expires_at_dt = datetime.fromisoformat(expires_at)
        if expires_at_dt.tzinfo is None:
            expires_at_dt = expires_at_dt.replace(tzinfo=timezone.utc)
    else:
        expires_at_dt = expires_at

    if datetime.now(timezone.utc) > expires_at_dt:
        # Mark as expired in the database
        try:
            admin_client.table("invitations").update({
                "status": "expired",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", invite["id"]).execute()
        except Exception:
            pass
        return InviteVerifyResponse(valid=False, error="This invitation has expired.")

    # ── Look up tenant name ─────────────────────────────────────────────
    tenant_name = "a workspace"
    try:
        tenant_result = admin_client.table("tenants").select("name").eq(
            "id", invite["tenant_id"]
        ).execute()
        if tenant_result.data:
            tenant_name = tenant_result.data[0]["name"]
    except Exception:
        pass

    return InviteVerifyResponse(
        valid=True,
        email=invite["email"],
        tenant_name=tenant_name,
        role=invite["role"],
    )


@router.get("/members", response_model=List[TeamMemberResponse])
async def list_members(
    ctx: RequestContext = Depends(require_admin),
) -> List[TeamMemberResponse]:
    """
    List all members of the current tenant.

    Args:
        ctx: Authenticated admin/owner context.

    Returns:
        List of TeamMemberResponse objects.
    """
    _require_auth_enabled()

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured.",
        )

    try:
        result = admin_client.table("users").select(
            "id, email, display_name, role"
        ).eq("tenant_id", ctx.tenant_id).order("role").execute()

        members = result.data or []
        return [
            TeamMemberResponse(
                id=m["id"],
                email=m["email"],
                display_name=m.get("display_name"),
                role=m["role"],
            )
            for m in members
        ]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list members: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to load team members.",
        )


@router.delete("/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    user_id: str,
    ctx: RequestContext = Depends(require_admin),
) -> None:
    """
    Remove a team member from the current tenant.

    Cannot remove the owner or yourself. Deletes the user from
    public.users and auth.users.

    Args:
        user_id: UUID of the user to remove.
        ctx: Authenticated admin/owner context.

    Raises:
        HTTPException: 404 if user not found or not in this tenant.
        HTTPException: 403 if trying to remove the owner.
        HTTPException: 403 if trying to remove yourself.
    """
    _require_auth_enabled()

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured.",
        )

    # ── Lookup user ─────────────────────────────────────────────────────
    result = admin_client.table("users").select("*").eq("id", user_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="User not found.")

    user = result.data[0]

    # ── Verify tenant ───────────────────────────────────────────────────
    if user["tenant_id"] != ctx.tenant_id:
        raise HTTPException(status_code=404, detail="User not found.")

    # ── Cannot remove owner ─────────────────────────────────────────────
    if user["role"] == "owner":
        raise HTTPException(
            status_code=403,
            detail="Cannot remove the workspace owner.",
        )

    # ── Cannot remove yourself ──────────────────────────────────────────
    if user_id == ctx.user_id:
        raise HTTPException(
            status_code=403,
            detail="Cannot remove yourself from the workspace.",
        )

    # ── Delete from public.users ────────────────────────────────────────
    try:
        admin_client.table("users").delete().eq("id", user_id).execute()
    except Exception as exc:
        logger.error("Failed to delete user from public.users: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to remove team member.",
        )

    # ── Delete from auth.users via admin API ───────────────────────────
    try:
        admin_client.auth.admin.delete_user(user_id)
    except Exception as exc:
        logger.warning(
            "Failed to delete user from auth.users: %s. "
            "The public.users record was deleted.",
            exc,
        )
        # Don't fail — the user is effectively removed from the tenant
