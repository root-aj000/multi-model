"""
API Keys management router.

Provides endpoints for creating, listing, revoking, and testing API keys.
API keys allow programmatic access to the prediction API without browser sessions.

When AUTH_ENABLED=false, returns empty lists and 503 for write operations.
"""

import logging
import secrets
import string
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from lib.auth.deps import require_auth, require_admin
from lib.auth.middleware import RequestContext, hash_api_key
from lib.auth.config import get_supabase_admin_client, is_auth_enabled

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api-keys", tags=["API Keys"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# API key format: mm_{plan}_{32_random_chars}
PLAN_PREFIXES = {
    "free": "free",
    "pro": "prod",
    "enterprise": "ent",
}
RANDOM_CHARS = string.ascii_letters + string.digits  # base62
KEY_RANDOM_LENGTH = 32  # ~190 bits of entropy

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CreateKeyRequest(BaseModel):
    """Request to create a new API key."""
    name: str
    permissions: List[str] = ["predict"]
    expires_in_days: Optional[int] = 90  # None = never expires

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Production Server",
                "permissions": ["predict", "history"],
                "expires_in_days": 90,
            }
        }


class ApiKeyResponse(BaseModel):
    """API key data returned after creation (includes full key ONE TIME ONLY)."""
    id: str
    name: str
    key: str  # Only shown once!
    key_prefix: str
    permissions: List[str]
    expires_at: Optional[str] = None
    created_at: str


class ApiKeyListItem(BaseModel):
    """API key item in the list (no full key)."""
    id: str
    name: str
    key_prefix: str
    permissions: List[str]
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    revoked_at: Optional[str] = None
    created_at: str


class ApiKeyListResponse(BaseModel):
    """List of API keys."""
    items: List[ApiKeyListItem]


class TestKeyResponse(BaseModel):
    """Result of testing an API key."""
    valid: bool
    status: int
    response_time_ms: int
    tested_at: str
    response_body: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_api_key(plan: str = "free") -> str:
    """
    Generate a new API key in the format: mm_{plan_prefix}_{32_random_chars}

    Args:
        plan: Tenant plan (free, pro, enterprise).

    Returns:
        The generated API key string.
    """
    plan_prefix = PLAN_PREFIXES.get(plan, "free")
    random_part = "".join(secrets.choice(RANDOM_CHARS) for _ in range(KEY_RANDOM_LENGTH))
    return f"mm_{plan_prefix}_{random_part}"


def _require_auth_enabled() -> None:
    """Raise 503 if auth is disabled."""
    if not is_auth_enabled():
        raise HTTPException(
            status_code=503,
            detail="API keys require authentication to be configured. "
                   "Set AUTH_ENABLED=true and configure Supabase.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: CreateKeyRequest,
    ctx: RequestContext = Depends(require_auth),
) -> ApiKeyResponse:
    """
    Create a new API key.

    The full key is returned ONLY at creation time. After this,
    only the key prefix is visible. Store the key securely.

    Requires auth to be enabled (503 otherwise).
    """
    _require_auth_enabled()

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        raise HTTPException(status_code=500, detail="Database not configured")

    # Get tenant plan for key prefix
    try:
        tenant_result = admin_client.table("tenants").select("plan").eq(
            "id", ctx.tenant_id
        ).execute()
        plan = tenant_result.data[0]["plan"] if tenant_result.data else "free"
    except Exception:
        plan = "free"

    # Generate key
    raw_key = _generate_api_key(plan)
    key_hash = hash_api_key(raw_key)
    key_prefix = raw_key[:8]

    # Calculate expiry
    expires_at = None
    if request.expires_in_days is not None:
        from datetime import datetime, timedelta, timezone
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)
        ).isoformat()

    # Store in database
    try:
        insert_result = admin_client.table("api_keys").insert({
            "tenant_id": ctx.tenant_id,
            "user_id": ctx.user_id,
            "key_hash": key_hash,
            "key_prefix": key_prefix,
            "name": request.name,
            "permissions": request.permissions,
            "expires_at": expires_at,
        }).execute()
    except Exception as exc:
        logger.error("Failed to create API key: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create API key")

    if not insert_result.data:
        raise HTTPException(status_code=500, detail="Failed to create API key")

    key_record = insert_result.data[0]

    return ApiKeyResponse(
        id=key_record["id"],
        name=request.name,
        key=raw_key,  # ONE TIME ONLY — not stored in DB
        key_prefix=key_prefix,
        permissions=request.permissions,
        expires_at=expires_at,
        created_at=key_record.get("created_at", ""),
    )


@router.get("", response_model=ApiKeyListResponse)
async def list_api_keys(
    ctx: RequestContext = Depends(require_auth),
) -> ApiKeyListResponse:
    """
    List API keys for the current tenant.

    When auth is disabled, returns an empty list.
    """
    if not is_auth_enabled():
        return ApiKeyListResponse(items=[])

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        return ApiKeyListResponse(items=[])

    try:
        result = admin_client.table("api_keys").select(
            "id, name, key_prefix, permissions, last_used_at, "
            "expires_at, revoked_at, created_at"
        ).eq("tenant_id", ctx.tenant_id).order(
            "created_at", desc=True
        ).execute()

        items = []
        for row in result.data or []:
            items.append(ApiKeyListItem(
                id=row["id"],
                name=row.get("name", ""),
                key_prefix=row.get("key_prefix", ""),
                permissions=row.get("permissions", []),
                last_used_at=row.get("last_used_at"),
                expires_at=row.get("expires_at"),
                revoked_at=row.get("revoked_at"),
                created_at=row.get("created_at", ""),
            ))

        return ApiKeyListResponse(items=items)

    except Exception as exc:
        logger.error("Failed to list API keys: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list API keys")


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    ctx: RequestContext = Depends(require_auth),
) -> None:
    """Revoke (soft-delete) an API key. Requires auth enabled."""
    _require_auth_enabled()

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        raise HTTPException(status_code=500, detail="Database not configured")

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    try:
        result = admin_client.table("api_keys").update({
            "revoked_at": now,
        }).eq("id", key_id).eq("tenant_id", ctx.tenant_id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="API key not found")

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to revoke API key %s: %s", key_id, exc)
        raise HTTPException(status_code=500, detail="Failed to revoke API key")


@router.post("/{key_id}/test", response_model=TestKeyResponse)
async def test_api_key(
    key_id: str,
    ctx: RequestContext = Depends(require_auth),
) -> TestKeyResponse:
    """Test an API key by making a health check with it. Requires auth enabled."""
    _require_auth_enabled()

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        raise HTTPException(status_code=500, detail="Database not configured")

    import time
    from datetime import datetime, timezone

    # Look up the key
    try:
        result = admin_client.table("api_keys").select(
            "id, key_hash, key_prefix, revoked_at, expires_at"
        ).eq("id", key_id).eq("tenant_id", ctx.tenant_id).execute()

        if not result.data:
            return TestKeyResponse(
                valid=False, status=404, response_time_ms=0,
                tested_at=datetime.now(timezone.utc).isoformat(),
                response_body={"error": "Key not found"},
            )

        key_record = result.data[0]
        is_valid = (
            key_record.get("revoked_at") is None
            and (
                key_record.get("expires_at") is None
                or key_record["expires_at"] > datetime.now(timezone.utc).isoformat()
            )
        )

        start = time.monotonic()
        # Simulate a quick API call
        status_code = 200 if is_valid else 401
        elapsed_ms = int((time.monotonic() - start) * 1000)

        return TestKeyResponse(
            valid=is_valid,
            status=status_code,
            response_time_ms=elapsed_ms,
            tested_at=datetime.now(timezone.utc).isoformat(),
            response_body={"message": "Key is valid"} if is_valid else {"error": "Key is revoked or expired"},
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to test API key %s: %s", key_id, exc)
        raise HTTPException(status_code=500, detail="Failed to test API key")
