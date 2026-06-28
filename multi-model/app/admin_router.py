"""
Admin API router.

Provides endpoints for platform administrators to manage tenants
and view system-wide statistics. All endpoints require platform_admin role.

When AUTH_ENABLED=false, returns empty placeholder data.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from lib.auth.deps import require_platform_admin
from lib.auth.middleware import RequestContext
from lib.auth.config import get_supabase_admin_client, is_auth_enabled

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class TenantListItem(BaseModel):
    """Tenant item in the admin list."""
    id: str
    name: str
    slug: str
    plan: str
    user_count: int = 0
    prediction_count: int = 0
    created_at: str


class TenantListResponse(BaseModel):
    """Paginated list of tenants."""
    items: List[TenantListItem]
    total: int


class TenantDetail(BaseModel):
    """Full tenant detail for admin view."""
    id: str
    name: str
    slug: str
    plan: str
    settings: Dict[str, Any] = {}
    monthly_limit: int = 100
    user_count: int = 0
    prediction_count: int = 0
    created_at: str
    updated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/tenants", response_model=TenantListResponse)
async def list_tenants(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    ctx: RequestContext = Depends(require_platform_admin),
) -> TenantListResponse:
    """
    List all tenants (platform admin only).

    When auth is disabled, returns an empty list.
    """
    if not is_auth_enabled():
        return TenantListResponse(items=[], total=0)

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        return TenantListResponse(items=[], total=0)

    try:
        # Get tenants with pagination
        offset = (page - 1) * page_size
        tenants_result = admin_client.table("tenants").select(
            "*", count="exact"
        ).order("created_at", desc=True).range(
            offset, offset + page_size - 1
        ).execute()

        tenants = tenants_result.data or []
        total = tenants_result.count if tenants_result.count is not None else len(tenants)

        # Enrich with counts — batch queries for all tenant IDs to reduce N+1
        tenant_ids = [tenant["id"] for tenant in tenants]
        user_counts = {}
        pred_counts = {}
        for tid in tenant_ids:
            try:
                uc = admin_client.table("users").select(
                    "id", count="exact"
                ).eq("tenant_id", tid).execute()
                user_counts[tid] = uc.count or 0
            except Exception:
                user_counts[tid] = 0
            try:
                pc = admin_client.table("predictions").select(
                    "id", count="exact"
                ).eq("tenant_id", tid).execute()
                pred_counts[tid] = pc.count or 0
            except Exception:
                pred_counts[tid] = 0

        items = []
        for tenant in tenants:
            tid = tenant["id"]
            items.append(TenantListItem(
                id=tid,
                name=tenant["name"],
                slug=tenant["slug"],
                plan=tenant.get("plan", "free"),
                user_count=user_counts.get(tid, 0),
                prediction_count=pred_counts.get(tid, 0),
                created_at=tenant.get("created_at", ""),
            ))

        return TenantListResponse(items=items, total=total)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list tenants: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to load tenants")


@router.get("/tenants/{tenant_id}", response_model=TenantDetail)
async def get_tenant(
    tenant_id: str,
    ctx: RequestContext = Depends(require_platform_admin),
) -> TenantDetail:
    """
    Get detailed information about a specific tenant (platform admin only).

    When auth is disabled, returns 404.
    """
    if not is_auth_enabled():
        raise HTTPException(status_code=404, detail="Tenant not found (auth disabled)")

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        raise HTTPException(status_code=500, detail="Database not configured")

    try:
        # Get tenant
        tenant_result = admin_client.table("tenants").select("*").eq(
            "id", tenant_id
        ).execute()

        if not tenant_result.data:
            raise HTTPException(status_code=404, detail="Tenant not found")

        tenant = tenant_result.data[0]

        # Count users
        user_count_result = admin_client.table("users").select(
            "id", count="exact"
        ).eq("tenant_id", tenant_id).execute()
        user_count = user_count_result.count or 0

        # Count predictions
        pred_count_result = admin_client.table("predictions").select(
            "id", count="exact"
        ).eq("tenant_id", tenant_id).execute()
        prediction_count = pred_count_result.count or 0

        return TenantDetail(
            id=tenant["id"],
            name=tenant["name"],
            slug=tenant["slug"],
            plan=tenant.get("plan", "free"),
            settings=tenant.get("settings", {}),
            monthly_limit=tenant.get("monthly_limit", 100),
            user_count=user_count,
            prediction_count=prediction_count,
            created_at=tenant.get("created_at", ""),
            updated_at=tenant.get("updated_at"),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get tenant %s: %s", tenant_id, exc)
        raise HTTPException(status_code=500, detail="Failed to load tenant")
