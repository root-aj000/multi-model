"""
Authentication API router.

Provides endpoints for user registration, login, token refresh, and logout.
All auth operations are proxied through the FastAPI backend (not direct
Supabase Auth from frontend) to allow custom validation, rate limiting,
and audit logging.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, field_validator

from lib.auth.config import (
    get_supabase_client,
    get_supabase_admin_client,
    is_auth_enabled,
    create_temp_client,
)
from lib.auth.deps import require_auth
from lib.auth.middleware import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

# Simple email regex (avoids dependency on email-validator package)
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


class RegisterRequest(BaseModel):
    """Registration request with tenant creation."""
    email: str
    password: str
    display_name: str
    tenant_name: str
    tenant_slug: str

    @field_validator("email")
    @classmethod
    def email_format(cls, v: str) -> str:
        if not _EMAIL_RE.match(v):
            raise ValueError("Invalid email format")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("tenant_slug")
    @classmethod
    def slug_format(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9][a-z0-9-]{1,98}[a-z0-9]$", v):
            raise ValueError(
                "Slug must be 3-100 chars, lowercase alphanumeric and hyphens, "
                "cannot start/end with hyphen"
            )
        return v


class LoginRequest(BaseModel):
    """Login request."""
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def email_format(cls, v: str) -> str:
        if not _EMAIL_RE.match(v):
            raise ValueError("Invalid email format")
        return v


class RegisterInviteRequest(BaseModel):
    """Registration request via invitation token."""
    email: str
    password: str
    display_name: str
    token: str

    @field_validator("email")
    @classmethod
    def email_format(cls, v: str) -> str:
        if not _EMAIL_RE.match(v):
            raise ValueError("Invalid email format")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class RefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class UserResponse(BaseModel):
    """User data returned in auth responses."""
    id: str
    email: str
    display_name: Optional[str] = None
    role: str = "member"


class TenantResponse(BaseModel):
    """Tenant data returned in auth responses."""
    id: str
    name: str
    slug: str
    plan: str = "free"


class AuthResponse(BaseModel):
    """Successful authentication response."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = 900  # 15 minutes
    user: UserResponse
    tenant: TenantResponse


class RefreshResponse(BaseModel):
    """Token refresh response."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = 900


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_auth_enabled() -> None:
    """Raise 503 if auth is disabled (Supabase not configured)."""
    if not is_auth_enabled():
        raise HTTPException(
            status_code=503,
            detail=(
                "Authentication is not configured. "
                "Set SUPABASE_URL, SUPABASE_KEY, and AUTH_ENABLED=true, "
                "or use the app in open mode (AUTH_ENABLED=false)."
            ),
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest) -> AuthResponse:
    """
    Register a new user and create a new tenant.

    This endpoint:
    1. Creates a tenant in the public.tenants table
    2. Signs up the user via Supabase Auth with tenant_id in metadata
    3. The Supabase trigger auto-creates public.users row and sets JWT claims

    Args:
        request: Registration data including email, password, and tenant info.

    Returns:
        AuthResponse with tokens, user, and tenant data.

    Raises:
        HTTPException: 503 if auth is not configured.
        HTTPException: 409 if email or slug is already taken.
        HTTPException: 500 if registration fails unexpectedly.
    """
    _require_auth_enabled()

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured. Cannot register users.",
        )

    # ── Check slug uniqueness ────────────────────────────────────────────
    existing_slug = admin_client.table("tenants").select("id").eq(
        "slug", request.tenant_slug
    ).execute()
    if existing_slug.data:
        raise HTTPException(
            status_code=409,
            detail=f"Tenant slug '{request.tenant_slug}' is already taken",
        )

    # ── Create tenant ───────────────────────────────────────────────────
    tenant_result = admin_client.table("tenants").insert({
        "name": request.tenant_name,
        "slug": request.tenant_slug,
        "plan": "free",
        "monthly_limit": 100,
    }).execute()

    if not tenant_result.data:
        raise HTTPException(
            status_code=500,
            detail="Failed to create tenant",
        )

    tenant = tenant_result.data[0]
    tenant_id = tenant["id"]

    # ── Sign up user via Supabase Auth ──────────────────────────────────
    try:
        # Use a temporary client so the signup session doesn't pollute
        # the singleton used by resolve_identity().
        client = create_temp_client()
        auth_response = client.auth.sign_up({
            "email": request.email,
            "password": request.password,
            "options": {
                "data": {
                    "display_name": request.display_name,
                    "tenant_id": tenant_id,
                    "role": "owner",  # First user is always owner
                },
            },
        })
    except Exception as exc:
        # Clean up tenant if user creation fails
        admin_client.table("tenants").delete().eq("id", tenant_id).execute()
        error_msg = str(exc)
        if "already registered" in error_msg.lower():
            raise HTTPException(
                status_code=409,
                detail="A user with this email already exists",
            )
        logger.error("Registration failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {exc}",
        )

    if not auth_response.user:
        # Clean up tenant
        admin_client.table("tenants").delete().eq("id", tenant_id).execute()
        raise HTTPException(
            status_code=500,
            detail="Registration failed — no user returned from auth service",
        )

    # ── Set tenant_id in app_metadata via admin API ─────────────────────
    # The set_tenant_claim trigger was removed (migration 004) because
    # the postgres role cannot UPDATE auth.users. Instead, we set the
    # app_metadata here using the admin API, which has the necessary
    # permissions.
    try:
        admin_client.auth.admin.update_user_by_id(
            auth_response.user.id,
            {"app_metadata": {"tenant_id": tenant_id, "role": "owner"}},
        )
    except Exception as exc:
        logger.warning(
            "Failed to set tenant_id in app_metadata for user %s: %s. "
            "The user was created but may not have tenant context in JWT.",
            auth_response.user.id,
            exc,
        )

    # ── Build response ──────────────────────────────────────────────────
    user_id = auth_response.user.id
    access_token = auth_response.session.access_token if auth_response.session else ""
    refresh_token = auth_response.session.refresh_token if auth_response.session else ""

    # ── Refresh session to get a JWT with updated app_metadata ─────────
    # The JWT from sign_up() was issued BEFORE admin.update_user_by_id()
    # set tenant_id/role in app_metadata. Refreshing the session produces
    # a fresh JWT that includes the updated claims, so the middleware can
    # resolve the user's tenant on subsequent requests.
    if refresh_token:
        try:
            temp_client = create_temp_client()
            refresh_result = temp_client.auth.refresh_session(
                refresh_token=refresh_token
            )
            if refresh_result and refresh_result.session:
                access_token = refresh_result.session.access_token
                refresh_token = refresh_result.session.refresh_token
                logger.info(
                    "Refreshed session after registration for user %s — "
                    "JWT now includes updated app_metadata",
                    user_id,
                )
        except Exception as exc:
            # Non-fatal: the original token still works, but the user
            # may need to re-login to get full tenant context.
            logger.warning(
                "Failed to refresh session after registration for user %s: %s. "
                "The original JWT (without tenant_id in claims) will be used.",
                user_id,
                exc,
            )

    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=900,
        user=UserResponse(
            id=user_id,
            email=request.email,
            display_name=request.display_name,
            role="owner",
        ),
        tenant=TenantResponse(
            id=tenant_id,
            name=tenant["name"],
            slug=tenant["slug"],
            plan=tenant["plan"],
        ),
    )


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest) -> AuthResponse:
    """
    Authenticate a user with email and password.

    Uses Supabase Auth to verify credentials, then looks up
    the user's role and tenant information.

    Args:
        request: Login credentials (email + password).

    Returns:
        AuthResponse with tokens, user, and tenant data.

    Raises:
        HTTPException: 503 if auth is not configured.
        HTTPException: 401 if credentials are invalid.
    """
    _require_auth_enabled()

    try:
        # Use a temporary client so the login session doesn't pollute
        # the singleton used by resolve_identity().
        client = create_temp_client()
        auth_response = client.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password,
        })
    except Exception as exc:
        logger.warning("Login failed for %s: %s", request.email, exc)
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )

    if not auth_response.user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )

    user_id = auth_response.user.id
    access_token = auth_response.session.access_token if auth_response.session else ""
    refresh_token = auth_response.session.refresh_token if auth_response.session else ""

    # ── Look up user role and tenant ────────────────────────────────────
    try:
        admin_client = get_supabase_admin_client()
        user_result = admin_client.table("users").select(
            "role, display_name, tenant_id"
        ).eq("id", user_id).execute()

        if not user_result.data:
            raise HTTPException(
                status_code=403,
                detail="User profile not found. Contact your administrator.",
            )

        user_data = user_result.data[0]
        tenant_id = user_data["tenant_id"]

        # Look up tenant
        tenant_result = admin_client.table("tenants").select("*").eq(
            "id", tenant_id
        ).execute()

        if not tenant_result.data:
            raise HTTPException(
                status_code=403,
                detail="Tenant not found. Contact your administrator.",
            )

        tenant = tenant_result.data[0]

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to look up user/tenant after login: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Login succeeded but failed to load profile",
        )

    # ── Ensure app_metadata has tenant_id and role ────────────────────
    # This is needed for users created before app_metadata was populated.
    # After this update, the next JWT issued (via refresh) will include
    # tenant_id in its claims, avoiding the DB fallback in the middleware.
    try:
        existing_meta = getattr(auth_response.user, "app_metadata", {}) or {}
        if not existing_meta.get("tenant_id"):
            admin_client.auth.admin.update_user_by_id(
                user_id,
                {"app_metadata": {
                    "tenant_id": tenant_id,
                    "role": user_data["role"],
                }},
            )
            logger.info(
                "Updated app_metadata for user %s with tenant_id=%s",
                user_id, tenant_id,
            )
    except Exception as exc:
        # Non-fatal: the middleware has a DB fallback for missing tenant_id
        logger.warning(
            "Failed to update app_metadata for user %s: %s",
            user_id, exc,
        )

    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=900,
        user=UserResponse(
            id=user_id,
            email=request.email,
            display_name=user_data.get("display_name"),
            role=user_data["role"],
        ),
        tenant=TenantResponse(
            id=tenant_id,
            name=tenant["name"],
            slug=tenant["slug"],
            plan=tenant["plan"],
        ),
    )


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(request: RefreshRequest) -> RefreshResponse:
    """
    Refresh an expired access token.

    Uses the refresh token to obtain a new access token from Supabase Auth.

    Args:
        request: Refresh token from a previous login.

    Returns:
        RefreshResponse with new access and refresh tokens.

    Raises:
        HTTPException: 503 if auth is not configured.
        HTTPException: 401 if the refresh token is invalid or expired.
    """
    _require_auth_enabled()

    try:
        # Use a temporary client so the refresh doesn't pollute the
        # singleton used by resolve_identity().
        client = create_temp_client()
        auth_response = client.auth.refresh_session(refresh_token=request.refresh_token)

        if not auth_response.session:
            raise HTTPException(
                status_code=401,
                detail="Refresh token is invalid or expired",
            )

        return RefreshResponse(
            access_token=auth_response.session.access_token,
            refresh_token=auth_response.session.refresh_token,
            token_type="bearer",
            expires_in=900,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Token refresh failed: %s", exc)
        raise HTTPException(
            status_code=401,
            detail="Refresh token is invalid or expired",
        )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout() -> None:
    """
    Logout the current user.

    Signs out from Supabase Auth, invalidating the current session.
    The frontend should also clear its local token storage.

    Note: This endpoint doesn't require authentication because the
    frontend should clear tokens locally regardless of whether the
    server-side logout succeeds.
    """
    if not is_auth_enabled():
        # Nothing to do server-side when auth is disabled
        return

    try:
        client = get_supabase_client()
        client.auth.sign_out()
    except Exception as exc:
        logger.warning("Server-side logout failed: %s", exc)
        # Don't fail the request — client should clear tokens anyway


@router.get("/me", response_model=AuthResponse)
async def get_current_user(ctx: RequestContext = Depends(require_auth)) -> AuthResponse:
    """
    Get the current authenticated user's profile and tenant.

    Used by the frontend on page reload to rehydrate the auth store
    with user/tenant data from the JWT. The middleware resolves the
    identity from the Bearer token, then this endpoint looks up the
    full user profile from the database.

    Args:
        ctx: Authenticated user context (injected by require_auth).

    Returns:
        AuthResponse with user and tenant data.

    Raises:
        HTTPException: 401 if not authenticated.
        HTTPException: 403 if user profile not found in database.
    """
    try:
        admin_client = get_supabase_admin_client()
        user_result = admin_client.table("users").select(
            "id, email, display_name, role, tenant_id"
        ).eq("id", ctx.user_id).execute()

        if not user_result.data:
            raise HTTPException(
                status_code=403,
                detail="User profile not found. Contact your administrator.",
            )

        user_data = user_result.data[0]
        tenant_id = user_data["tenant_id"]

        # Look up tenant
        tenant_result = admin_client.table("tenants").select("*").eq(
            "id", tenant_id
        ).execute()

        if not tenant_result.data:
            raise HTTPException(
                status_code=403,
                detail="Tenant not found. Contact your administrator.",
            )

        tenant = tenant_result.data[0]

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to look up user/tenant for /auth/me: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to load user profile",
        )

    return AuthResponse(
        access_token="",  # Not issued here — frontend already has it
        refresh_token="",
        token_type="bearer",
        expires_in=0,
        user=UserResponse(
            id=user_data["id"],
            email=user_data["email"],
            display_name=user_data.get("display_name"),
            role=user_data["role"],
        ),
        tenant=TenantResponse(
            id=tenant_id,
            name=tenant["name"],
            slug=tenant["slug"],
            plan=tenant["plan"],
        ),
    )


@router.post("/register-invite", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register_via_invite(request: RegisterInviteRequest) -> AuthResponse:
    """
    Register a new user via an invitation token.

    This endpoint is used when a user clicks an invite link from their email.
    It validates the invitation token, creates the user in Supabase Auth
    with the tenant_id and role from the invitation, and marks the invitation
    as accepted.

    Unlike the normal register endpoint, this does NOT create a new tenant —
    the user joins the inviter's existing tenant as a member.

    Args:
        request: Registration data including email, password, display_name, and token.

    Returns:
        AuthResponse with tokens, user, and tenant data.

    Raises:
        HTTPException: 503 if auth is not configured.
        HTTPException: 400 if the invitation token is invalid, expired, or email mismatch.
        HTTPException: 409 if a user with this email already exists.
        HTTPException: 500 if registration fails unexpectedly.
    """
    _require_auth_enabled()

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured. Cannot register users.",
        )

    # ── Validate invitation token ────────────────────────────────────────
    invite_result = admin_client.table("invitations").select("*").eq(
        "token", request.token
    ).execute()

    if not invite_result.data:
        raise HTTPException(
            status_code=400,
            detail="Invalid invitation token.",
        )

    invite = invite_result.data[0]

    # ── Check invitation status ─────────────────────────────────────────
    if invite["status"] != "pending":
        if invite["status"] == "accepted":
            raise HTTPException(
                status_code=400,
                detail="This invitation has already been used.",
            )
        if invite["status"] == "expired":
            raise HTTPException(
                status_code=400,
                detail="This invitation has expired.",
            )
        if invite["status"] == "cancelled":
            raise HTTPException(
                status_code=400,
                detail="This invitation has been cancelled.",
            )
        raise HTTPException(
            status_code=400,
            detail="This invitation is no longer valid.",
        )

    # ── Check expiry ────────────────────────────────────────────────────
    expires_at = invite["expires_at"]
    if isinstance(expires_at, str):
        from datetime import timezone as _tz
        expires_at_dt = datetime.fromisoformat(expires_at)
        if expires_at_dt.tzinfo is None:
            expires_at_dt = expires_at_dt.replace(tzinfo=_tz.utc)
    else:
        expires_at_dt = expires_at

    if datetime.now(timezone.utc) > expires_at_dt:
        # Mark as expired
        admin_client.table("invitations").update({
            "status": "expired",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", invite["id"]).execute()
        raise HTTPException(
            status_code=400,
            detail="This invitation has expired.",
        )

    # ── Verify email matches invitation ─────────────────────────────────
    if request.email.lower() != invite["email"].lower():
        raise HTTPException(
            status_code=400,
            detail="The email address does not match the invitation.",
        )

    tenant_id = invite["tenant_id"]
    invite_role = invite["role"]

    # ── Sign up user via Supabase Auth ──────────────────────────────────
    try:
        client = create_temp_client()
        auth_response = client.auth.sign_up({
            "email": request.email,
            "password": request.password,
            "options": {
                "data": {
                    "display_name": request.display_name,
                    "tenant_id": tenant_id,
                    "role": invite_role,
                },
            },
        })
    except Exception as exc:
        error_msg = str(exc)
        if "already registered" in error_msg.lower():
            raise HTTPException(
                status_code=409,
                detail="A user with this email already exists. Please log in instead.",
            )
        logger.error("Invite registration failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {exc}",
        )

    if not auth_response.user:
        raise HTTPException(
            status_code=500,
            detail="Registration failed — no user returned from auth service",
        )

    # ── Set tenant_id in app_metadata via admin API ─────────────────────
    try:
        admin_client.auth.admin.update_user_by_id(
            auth_response.user.id,
            {"app_metadata": {"tenant_id": tenant_id, "role": invite_role}},
        )
    except Exception as exc:
        logger.warning(
            "Failed to set app_metadata for invited user %s: %s",
            auth_response.user.id, exc,
        )

    # ── Mark invitation as accepted ─────────────────────────────────────
    try:
        admin_client.table("invitations").update({
            "status": "accepted",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", invite["id"]).execute()
    except Exception as exc:
        logger.warning(
            "Failed to mark invitation %s as accepted: %s",
            invite["id"], exc,
        )

    # ── Build response ──────────────────────────────────────────────────
    user_id = auth_response.user.id
    access_token = auth_response.session.access_token if auth_response.session else ""
    refresh_token = auth_response.session.refresh_token if auth_response.session else ""

    # ── Refresh session to get JWT with updated app_metadata ───────────
    if refresh_token:
        try:
            temp_client = create_temp_client()
            refresh_result = temp_client.auth.refresh_session(
                refresh_token=refresh_token
            )
            if refresh_result and refresh_result.session:
                access_token = refresh_result.session.access_token
                refresh_token = refresh_result.session.refresh_token
        except Exception as exc:
            logger.warning(
                "Failed to refresh session after invite registration for %s: %s",
                user_id, exc,
            )

    # ── Look up tenant info ─────────────────────────────────────────────
    try:
        tenant_result = admin_client.table("tenants").select("*").eq(
            "id", tenant_id
        ).execute()
        if not tenant_result.data:
            raise HTTPException(
                status_code=500,
                detail="Tenant not found after registration.",
            )
        tenant = tenant_result.data[0]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to look up tenant after invite registration: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Registration succeeded but failed to load tenant info.",
        )

    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=900,
        user=UserResponse(
            id=user_id,
            email=request.email,
            display_name=request.display_name,
            role=invite_role,
        ),
        tenant=TenantResponse(
            id=tenant_id,
            name=tenant["name"],
            slug=tenant["slug"],
            plan=tenant["plan"],
        ),
    )
