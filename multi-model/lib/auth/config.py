"""
Supabase client configuration and initialization.

Provides two client instances:
- Regular client: respects RLS policies (for user-facing operations)
- Admin client: uses service_role key, bypasses RLS (for internal operations)

Authentication is auto-detected: when both SUPABASE_URL and SUPABASE_KEY
are present, auth is enabled. Set AUTH_ENABLED=false to explicitly disable.

Per supabase-py docs, server-side clients should use ClientOptions with:
    auto_refresh_token=False, persist_session=False

Environment variables:
    SUPABASE_URL:             Supabase project URL
    SUPABASE_KEY:             Supabase anon/public key
    SUPABASE_SERVICE_ROLE_KEY: Supabase service_role key (bypasses RLS)
    SUPABASE_JWT_SECRET:      Supabase JWT secret (base64-encoded)
    AUTH_ENABLED:             Optional override: "false" to force-disable auth
    FRONTEND_URL:             Frontend URL for CORS configuration
"""

import os
import logging
from typing import Optional

from supabase import create_client, Client

logger = logging.getLogger(__name__)

ENV_SUPABASE_URL = "SUPABASE_URL"
ENV_SUPABASE_KEY = "SUPABASE_KEY"
ENV_SUPABASE_SERVICE_ROLE_KEY = "SUPABASE_SERVICE_ROLE_KEY"
ENV_SUPABASE_JWT_SECRET = "SUPABASE_JWT_SECRET"
ENV_AUTH_ENABLED = "AUTH_ENABLED"
ENV_FRONTEND_URL = "FRONTEND_URL"

_supabase_client: Optional[Client] = None
_supabase_admin_client: Optional[Client] = None


def is_auth_enabled() -> bool:
    """
    Check if authentication is enabled.

    Auth is enabled when both SUPABASE_URL and SUPABASE_KEY are set,
    unless AUTH_ENABLED is explicitly set to "false".
    """
    explicit_disable = os.environ.get(ENV_AUTH_ENABLED, "").lower() in ("false", "0", "no")
    if explicit_disable:
        return False
    url = os.environ.get(ENV_SUPABASE_URL, "")
    key = os.environ.get(ENV_SUPABASE_KEY, "")
    return bool(url and key)


def get_frontend_url() -> str:
    """Return the frontend URL for CORS configuration."""
    return os.environ.get(ENV_FRONTEND_URL, "http://localhost:3000")


def get_supabase_jwt_secret() -> str:
    """
    Get the Supabase JWT secret for token verification.

    Returns:
        The base64-encoded JWT secret string.

    Raises:
        ValueError: If SUPABASE_JWT_SECRET is not configured.
    """
    secret = os.environ.get(ENV_SUPABASE_JWT_SECRET, "")
    if not secret:
        raise ValueError(
            f"JWT secret not configured. Set {ENV_SUPABASE_JWT_SECRET} "
            "environment variable."
        )
    return secret


def get_supabase_client() -> Client:
    """
    Get or create the regular Supabase client (RLS enforced).

    Uses the anon key and respects all Row-Level Security policies.
    Use for user-facing operations where the user's JWT context
    determines data access.

    Server-side options are applied (no session persistence, no auto-refresh)
    by modifying client.options after creation. This avoids compatibility
    issues with ClientOptions across different supabase-py versions that
    may have missing fields (storage, httpx_client).

    Returns:
        Supabase Client instance with RLS enforced.

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_KEY is not configured.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.environ.get(ENV_SUPABASE_URL, "")
    key = os.environ.get(ENV_SUPABASE_KEY, "")

    if not url or not key:
        raise ValueError(
            f"Supabase not configured. Set {ENV_SUPABASE_URL} and {ENV_SUPABASE_KEY} "
            "environment variables, or set AUTH_ENABLED=false to run without auth."
        )

    _supabase_client = create_client(url, key)
    # Disable session persistence and auto-refresh for server-side use.
    # Done post-creation to avoid ClientOptions field compatibility issues.
    _supabase_client.options.auto_refresh_token = False
    _supabase_client.options.persist_session = False
    logger.info("Supabase client initialized (RLS enforced, no session persistence)")
    return _supabase_client


def get_supabase_admin_client() -> Client:
    """
    Get or create the admin Supabase client (bypasses RLS).

    Uses the service_role key which bypasses ALL Row-Level Security
    policies. Use ONLY for:
    1. API key hash lookup during auth middleware (before tenant_id is known)
    2. Admin endpoints that need cross-tenant visibility
    3. Key usage incrementing (happens after request completes)

    Server-side options are applied (no session persistence, no auto-refresh)
    by modifying client.options after creation. This avoids compatibility
    issues with ClientOptions across different supabase-py versions that
    may have missing fields (storage, httpx_client).

    NEVER expose this client or its key to the frontend.

    Returns:
        Supabase Client instance with service_role access (RLS bypassed).

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is not configured.
    """
    global _supabase_admin_client
    if _supabase_admin_client is not None:
        return _supabase_admin_client

    url = os.environ.get(ENV_SUPABASE_URL, "")
    service_role_key = os.environ.get(ENV_SUPABASE_SERVICE_ROLE_KEY, "")

    if not url or not service_role_key:
        raise ValueError(
            f"Supabase admin not configured. Set {ENV_SUPABASE_URL} and "
            f"{ENV_SUPABASE_SERVICE_ROLE_KEY} environment variables."
        )

    _supabase_admin_client = create_client(url, service_role_key)
    # Disable session persistence and auto-refresh for server-side use.
    # Done post-creation to avoid ClientOptions field compatibility issues.
    _supabase_admin_client.options.auto_refresh_token = False
    _supabase_admin_client.options.persist_session = False
    logger.warning(
        "Supabase admin client initialized (RLS BYPASSED, no session persistence). "
        "Use only for internal operations — never expose to frontend."
    )
    return _supabase_admin_client


def create_temp_client() -> Client:
    """
    Create a one-time Supabase client for auth operations.

    Unlike the singleton from get_supabase_client(), this client is NOT
    cached. Each call creates a fresh instance with no stored session.
    Use for sign_in_with_password(), sign_up(), and refresh_session()
    to prevent session pollution of the shared singleton client.

    Returns:
        A fresh Supabase Client instance with RLS enforced.

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_KEY is not configured.
    """
    url = os.environ.get(ENV_SUPABASE_URL, "")
    key = os.environ.get(ENV_SUPABASE_KEY, "")

    if not url or not key:
        raise ValueError(
            f"Supabase not configured. Set {ENV_SUPABASE_URL} and {ENV_SUPABASE_KEY} "
            "environment variables, or set AUTH_ENABLED=false to run without auth."
        )

    client = create_client(url, key)
    # Disable session persistence and auto-refresh for server-side use.
    # Done post-creation to avoid ClientOptions field compatibility issues.
    client.options.auto_refresh_token = False
    client.options.persist_session = False
    return client


def reset_clients() -> None:
    """
    Reset cached Supabase clients.

    Useful for testing or when environment variables change at runtime.
    """
    global _supabase_client, _supabase_admin_client
    _supabase_client = None
    _supabase_admin_client = None
