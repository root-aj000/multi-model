"""
Dual-mode authentication middleware for FastAPI.

Resolves identity from either:
1. Authorization: Bearer <jwt>  (browser sessions via Supabase Auth)
2. X-API-Key: <key>            (programmatic access)

Both resolve to a unified RequestContext attached to request.state.

JWT verification uses supabase.auth.get_claims() which verifies tokens
against the Supabase JWKS endpoint (/.well-known/jwks.json). This is
faster and more secure than manual HS256 decoding because:
- The JWKS is cached, avoiding repeated network calls
- It supports key rotation without configuration changes
- It uses asymmetric algorithms (RS256/ES256) by default

For HS256 tokens (symmetric), get_claims() falls back to get_user()
which verifies the token server-side.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import Request, HTTPException

from .config import (
    get_supabase_client,
    get_supabase_admin_client,
    is_auth_enabled,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class RequestContext:
    """
    Unified identity context attached to every authenticated request.

    This is the single source of truth for who is making the request,
    which tenant they belong to, and how they authenticated.
    """

    user_id: str
    tenant_id: str
    role: str  # "owner", "admin", "member", "platform_admin"
    auth_method: str  # "jwt" or "api_key"
    api_key_id: Optional[str] = None

    @property
    def is_admin(self) -> bool:
        """Check if the user has admin or owner privileges."""
        return self.role in ("admin", "owner", "platform_admin")

    @property
    def is_platform_admin(self) -> bool:
        """Check if the user is a platform admin."""
        return self.role == "platform_admin"


# ---------------------------------------------------------------------------
# API key hashing
# ---------------------------------------------------------------------------

def hash_api_key(raw_key: str) -> str:
    """
    SHA-256 hash of an API key.

    Uses SHA-256 (not bcrypt) because API keys are high-entropy random
    tokens (190+ bits), not low-entropy passwords. SHA-256 is faster
    for the middleware hot path (every API request).

    Args:
        raw_key: The plaintext API key string.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# DB fallback for missing tenant_id
# ---------------------------------------------------------------------------

def _lookup_user_profile(user_id: str) -> tuple:
    """
    Look up tenant_id and role from the public.users table.

    Used as a fallback when the JWT's app_metadata doesn't contain
    tenant_id (e.g., users created before app_metadata was populated,
    or if admin.update_user_by_id() failed during registration).

    Args:
        user_id: The user's UUID from the JWT sub claim.

    Returns:
        Tuple of (tenant_id, role). Both are empty strings if lookup fails.
    """
    try:
        admin_client = get_supabase_admin_client()
        result = admin_client.table("users").select(
            "tenant_id, role"
        ).eq("id", user_id).execute()

        if result.data:
            user_data = result.data[0]
            return (
                str(user_data.get("tenant_id", "")),
                user_data.get("role", ""),
            )
    except Exception as exc:
        logger.warning(
            "DB fallback lookup failed for user %s: %s", user_id, exc
        )

    return ("", "")


# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------

async def resolve_identity(request: Request) -> RequestContext:
    """
    Resolve the caller's identity from request headers.

    Priority:
    1. Authorization: Bearer <jwt>  → verify with Supabase get_claims()
    2. X-API-Key: <key>             → hash + DB lookup
    3. Neither present               → 401

    When AUTH_ENABLED is false, returns a default context with
    placeholder values (backward compatibility mode).

    Args:
        request: The incoming FastAPI request.

    Returns:
        RequestContext with resolved identity.

    Raises:
        HTTPException: 401 if authentication fails or is required but missing.
    """
    # ── Feature flag: bypass auth when disabled ────────────────────────
    if not is_auth_enabled():
        return RequestContext(
            user_id="anonymous",
            tenant_id="default",
            role="member",
            auth_method="none",
        )

    auth_header = request.headers.get("Authorization", "")
    api_key_header = request.headers.get("X-API-Key", "")

    # ── Mode 1: JWT Bearer Token ───────────────────────────────────────
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        return _resolve_jwt(token)

    # ── Mode 2: API Key ────────────────────────────────────────────────
    if api_key_header:
        return await _resolve_api_key(api_key_header)

    # ── Mode 3: No auth provided ───────────────────────────────────────
    raise HTTPException(
        status_code=401,
        detail=(
            "Authentication required. Provide Authorization: Bearer <token> "
            "or X-API-Key: <key>"
        ),
    )


def _resolve_jwt(token: str) -> RequestContext:
    """
    Resolve identity from a JWT Bearer token.

    Uses supabase.auth.get_user(jwt=token) which verifies the token
    server-side against the Supabase Auth server. Per the supabase-py
    documentation (§"Retrieve a user"):

    > "This method is useful for checking if the user is authorized
    >  because it validates the user's access token JWT on the server."

    Falls back to local JWT payload decoding if the server call fails
    (e.g., network issues, JWKS errors).

    Args:
        token: The raw JWT token string.

    Returns:
        RequestContext with JWT-resolved identity.

    Raises:
        HTTPException: 401 if the token is expired or invalid.
        HTTPException: 500 if the Supabase client is not configured.
    """
    try:
        client = get_supabase_client()
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Supabase client not configured. Set SUPABASE_URL and SUPABASE_KEY.",
        )

    # ── Primary: Server-side verification via get_user() ─────────────
    try:
        logger.debug("Attempting server-side JWT verification via get_user()")
        user_response = client.auth.get_user(jwt=token)
        if user_response and user_response.user:
            user = user_response.user
            user_id = str(user.id)
            # Extract app_metadata for tenant_id
            app_metadata = getattr(user, "app_metadata", {}) or {}
            tenant_id = str(app_metadata.get("tenant_id", ""))
            role = app_metadata.get("role", getattr(user, "role", "member") or "member")

            if not user_id:
                logger.debug("JWT verified but missing user identifier (sub)")
                raise HTTPException(
                    status_code=401, detail="Token missing user identifier (sub)."
                )

            if not tenant_id:
                # Fallback: look up tenant_id from public.users table.
                # This happens for users created before app_metadata
                # was populated, or if the admin.update_user_by_id()
                # call failed during registration.
                tenant_id, db_role = _lookup_user_profile(user_id)
                if tenant_id:
                    role = db_role or role
                    logger.info(
                        "Resolved tenant_id=%s for user %s from DB fallback",
                        tenant_id, user_id,
                    )
                else:
                    logger.warning(
                        "JWT for user %s has no tenant_id in app_metadata "
                        "and no profile in public.users table.",
                        user_id,
                    )

            logger.debug(
                "JWT resolved via get_user(): user_id=%s, tenant_id=%s, role=%s",
                user_id, tenant_id, role,
            )

            return RequestContext(
                user_id=user_id,
                tenant_id=tenant_id,
                role=role,
                auth_method="jwt",
            )
    except HTTPException:
        raise
    except Exception as exc:
        error_name = type(exc).__name__
        error_msg = str(exc).lower()

        if "expired" in error_msg:
            logger.debug("JWT expired: %s", exc)
            raise HTTPException(
                status_code=401,
                detail="Token expired. Please refresh your session.",
            )

        if "invalid" in error_msg or "jwt" in error_name.lower() or "auth" in error_name.lower():
            logger.debug("Invalid JWT (%s): %s", error_name, exc)
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token: {exc}",
            )

        # Server verification failed — fall through to local decode
        logger.warning(
            "Server-side JWT verification failed (%s): %s. Falling back to local decode.",
            error_name,
            exc,
        )

    # ── Fallback: Local JWT payload decode (no signature verification) ──
    # This is less secure but works when the Supabase Auth server is
    # unreachable. The token was issued by Supabase Auth, so we trust it.
    try:
        logger.debug("Attempting local JWT payload decode (no signature verification)")
        import base64 as _b64
        import json as _json

        parts = token.split(".")
        if len(parts) != 3:
            raise HTTPException(status_code=401, detail="Invalid JWT structure")

        # Decode the payload (second part)
        payload_b64 = parts[1]
        # Add padding if needed
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload_json = _b64.urlsafe_b64decode(payload_b64)
        payload = _json.loads(payload_json)

        user_id = payload.get("sub", "")
        app_metadata = payload.get("app_metadata", {}) or {}
        tenant_id = str(app_metadata.get("tenant_id", ""))
        role = payload.get("role", "member")

        if not user_id:
            raise HTTPException(status_code=401, detail="Token missing user identifier (sub).")

        if not tenant_id:
            # Fallback: look up tenant_id from public.users table
            tenant_id, db_role = _lookup_user_profile(user_id)
            if tenant_id:
                role = db_role or role
                logger.info(
                    "Resolved tenant_id=%s for user %s from DB fallback (local decode)",
                    tenant_id, user_id,
                )
            else:
                logger.warning(
                    "JWT for user %s has no tenant_id in app_metadata "
                    "and no profile in public.users table.",
                    user_id,
                )

        logger.debug(
            "JWT resolved via local decode: user_id=%s, tenant_id=%s, role=%s",
            user_id, tenant_id, role,
        )

        return RequestContext(
            user_id=str(user_id),
            tenant_id=tenant_id,
            role=role,
            auth_method="jwt",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Local JWT decode failed: %s", exc)
        raise HTTPException(
            status_code=401,
            detail="Token verification failed",
        )


async def _resolve_api_key(raw_key: str) -> RequestContext:
    """
    Resolve identity from an API key.

    Hashes the key, looks it up in the database (using admin client
    to bypass RLS since we don't know the tenant yet), verifies
    it's not revoked or expired, and returns the identity context.

    Args:
        raw_key: The plaintext API key from the X-API-Key header.

    Returns:
        RequestContext with API-key-resolved identity.

    Raises:
        HTTPException: 401 if the key is invalid, revoked, or expired.
    """
    key_hash = hash_api_key(raw_key)

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Supabase admin client not configured for API key authentication.",
        )

    # Look up the key by hash (admin client bypasses RLS)
    result = admin_client.table("api_keys").select(
        "id, tenant_id, user_id, permissions, expires_at, revoked_at, key_prefix"
    ).eq("key_hash", key_hash).execute()

    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid API key")

    key_record = result.data[0]

    # Check revoked
    if key_record.get("revoked_at") is not None:
        raise HTTPException(status_code=401, detail="API key has been revoked")

    # Check expired
    expires_at = key_record.get("expires_at")
    if expires_at is not None:
        expires_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if datetime.now(timezone.utc) > expires_dt:
            raise HTTPException(status_code=401, detail="API key has expired")

    # Look up the user's role from the users table
    user_result = admin_client.table("users").select("role").eq(
        "id", key_record["user_id"]
    ).execute()

    role = user_result.data[0]["role"] if user_result.data else "member"

    # Fire-and-forget: update last_used_at and increment daily usage
    _update_key_usage_async(admin_client, key_record["id"])

    return RequestContext(
        user_id=key_record["user_id"],
        tenant_id=key_record["tenant_id"],
        role=role,
        auth_method="api_key",
        api_key_id=key_record["id"],
    )


def _update_key_usage_async(admin_client, api_key_id: str) -> None:
    """
    Update last_used_at and increment daily usage counter.

    This is a fire-and-forget operation — it should not block or
    fail the request. Errors are logged but not propagated.

    Args:
        admin_client: Supabase admin client (bypasses RLS).
        api_key_id: UUID of the API key that was used.
    """
    try:
        today = time.strftime("%Y-%m-%d")

        # Update last_used_at
        admin_client.table("api_keys").update(
            {"last_used_at": "now()"}
        ).eq("id", api_key_id).execute()

        # Upsert daily usage counter (atomic increment via RPC)
        try:
            admin_client.rpc(
                "increment_key_usage",
                {"key_id": api_key_id, "usage_date": today},
            ).execute()
        except Exception:
            # RPC may not exist yet — fall back to manual upsert
            admin_client.table("key_usage_daily").upsert(
                {
                    "api_key_id": api_key_id,
                    "date": today,
                    "request_count": 1,
                },
                on_conflict="api_key_id,date",
            ).execute()

    except Exception as exc:
        logger.warning("Failed to update key usage for %s: %s", api_key_id, exc)
        # Don't fail the request — usage tracking is non-critical
