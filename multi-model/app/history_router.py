"""
History API router.

Provides endpoints for viewing and managing prediction history.
All endpoints are tenant-scoped via the auth middleware.

When AUTH_ENABLED=false, returns empty placeholder data.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from lib.auth.deps import require_auth, require_admin
from lib.auth.middleware import RequestContext
from lib.auth.config import get_supabase_admin_client, is_auth_enabled

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["History"])

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PredictionListItem(BaseModel):
    """Prediction item in the history list."""
    id: str
    filename: Optional[str] = None
    predicted_label: Optional[str] = None
    theme: Optional[str] = None
    sentiment: Optional[str] = None
    processing_ms: Optional[int] = None
    created_at: str


class PredictionDetail(BaseModel):
    """Full prediction detail."""
    id: str
    filename: Optional[str] = None
    ocr_text: Optional[str] = None
    result: Dict[str, Any]
    processing_ms: Optional[int] = None
    user_id: str
    created_at: str


class PaginatedResponse(BaseModel):
    """Paginated list response."""
    items: List[PredictionListItem]
    total: int
    page: int
    page_size: int
    total_pages: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=PaginatedResponse)
async def list_predictions(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    attribute: Optional[str] = Query(None, description="Filter by attribute name"),
    value: Optional[str] = Query(None, description="Filter by attribute value"),
    search: Optional[str] = Query(None, description="Search in filename or OCR text"),
    ctx: RequestContext = Depends(require_auth),
) -> PaginatedResponse:
    """
    List prediction history for the current tenant.

    When auth is disabled, returns an empty list.
    """
    # Return empty data when auth is disabled (no database)
    if not is_auth_enabled():
        return PaginatedResponse(
            items=[], total=0, page=page, page_size=page_size, total_pages=0,
        )

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        return PaginatedResponse(
            items=[], total=0, page=page, page_size=page_size, total_pages=0,
        )

    # Build query
    query = admin_client.table("predictions").select(
        "id, filename, result, processing_ms, created_at, user_id",
        count="exact",
    ).eq("tenant_id", ctx.tenant_id)

    # Members see only their own predictions
    if ctx.role == "member":
        query = query.eq("user_id", ctx.user_id)

    # Attribute filter (JSONB)
    if attribute and value:
        query = query.contains("result", {attribute: value})

    # Text search
    if search:
        query = query.or_(
            f"filename.ilike.%{search}%,ocr_text.ilike.%{search}%"
        )

    # Order by newest first
    query = query.order("created_at", desc=True)

    # Pagination
    offset = (page - 1) * page_size
    query = query.range(offset, offset + page_size - 1)

    try:
        result = query.execute()
    except Exception as exc:
        logger.error("Failed to query predictions: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to load history")

    items = []
    for row in result.data or []:
        prediction_result = row.get("result", {})
        items.append(PredictionListItem(
            id=row["id"],
            filename=row.get("filename"),
            predicted_label=prediction_result.get("predicted_label"),
            theme=prediction_result.get("theme"),
            sentiment=prediction_result.get("sentiment"),
            processing_ms=row.get("processing_ms"),
            created_at=row.get("created_at", ""),
        ))

    total = result.count if result.count is not None else len(items)
    total_pages = max(1, (total + page_size - 1) // page_size)

    return PaginatedResponse(
        items=items, total=total, page=page,
        page_size=page_size, total_pages=total_pages,
    )


@router.get("/{prediction_id}", response_model=PredictionDetail)
async def get_prediction(
    prediction_id: str,
    ctx: RequestContext = Depends(require_auth),
) -> PredictionDetail:
    """
    Get a single prediction by ID.

    When auth is disabled, returns 404.
    """
    if not is_auth_enabled():
        raise HTTPException(status_code=404, detail="Prediction not found (auth disabled)")

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        raise HTTPException(status_code=500, detail="Database not configured")

    query = admin_client.table("predictions").select("*").eq(
        "id", prediction_id
    ).eq("tenant_id", ctx.tenant_id)

    # Members can only see their own
    if ctx.role == "member":
        query = query.eq("user_id", ctx.user_id)

    try:
        result = query.execute()
    except Exception as exc:
        logger.error("Failed to query prediction %s: %s", prediction_id, exc)
        raise HTTPException(status_code=500, detail="Failed to load prediction")

    if not result.data:
        raise HTTPException(status_code=404, detail="Prediction not found")

    row = result.data[0]
    return PredictionDetail(
        id=row["id"],
        filename=row.get("filename"),
        ocr_text=row.get("ocr_text"),
        result=row.get("result", {}),
        processing_ms=row.get("processing_ms"),
        user_id=row.get("user_id", ""),
        created_at=row.get("created_at", ""),
    )


@router.delete("/{prediction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prediction(
    prediction_id: str,
    ctx: RequestContext = Depends(require_auth),
) -> None:
    """Delete a prediction. When auth is disabled, returns 404."""
    if not is_auth_enabled():
        raise HTTPException(status_code=404, detail="Prediction not found (auth disabled)")

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        raise HTTPException(status_code=500, detail="Database not configured")

    query = admin_client.table("predictions").delete().eq(
        "id", prediction_id
    ).eq("tenant_id", ctx.tenant_id)

    # Members can only delete their own
    if ctx.role == "member":
        query = query.eq("user_id", ctx.user_id)

    try:
        result = query.execute()
    except Exception as exc:
        logger.error("Failed to delete prediction %s: %s", prediction_id, exc)
        raise HTTPException(status_code=500, detail="Failed to delete prediction")
