"""
Analytics API router.

Provides endpoints for viewing prediction analytics and attribute distributions.
All endpoints are tenant-scoped via the auth middleware.

When AUTH_ENABLED=false, returns zero-value placeholder data.
"""

import logging
from typing import Any, Dict, List, Optional
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from lib.auth.deps import require_auth
from lib.auth.middleware import RequestContext
from lib.auth.config import get_supabase_admin_client, is_auth_enabled

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class AnalyticsSummary(BaseModel):
    """Summary statistics for the tenant."""
    total_predictions: int
    predictions_this_week: int
    predictions_this_month: int
    avg_processing_ms: Optional[float] = None
    most_common_theme: Optional[str] = None
    most_common_sentiment: Optional[str] = None
    quota_used: int
    quota_limit: int


class AttributeDistributions(BaseModel):
    """Distribution counts for each attribute."""
    theme: Dict[str, int] = {}
    sentiment: Dict[str, int] = {}
    emotion: Dict[str, int] = {}
    dominant_colour: Dict[str, int] = {}
    attention_score: Dict[str, int] = {}
    trust_safety: Dict[str, int] = {}
    target_audience: Dict[str, int] = {}
    predicted_ctr: Dict[str, int] = {}
    likelihood_shares: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/summary", response_model=AnalyticsSummary)
async def get_analytics_summary(
    ctx: RequestContext = Depends(require_auth),
) -> AnalyticsSummary:
    """
    Get analytics summary for the current tenant.

    When auth is disabled, returns zero-value placeholder data.
    """
    if not is_auth_enabled():
        return AnalyticsSummary(
            total_predictions=0,
            predictions_this_week=0,
            predictions_this_month=0,
            avg_processing_ms=None,
            most_common_theme=None,
            most_common_sentiment=None,
            quota_used=0,
            quota_limit=100,
        )

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        return AnalyticsSummary(
            total_predictions=0,
            predictions_this_week=0,
            predictions_this_month=0,
            avg_processing_ms=None,
            most_common_theme=None,
            most_common_sentiment=None,
            quota_used=0,
            quota_limit=100,
        )

    try:
        # Get predictions count for the tenant
        count_result = admin_client.table("predictions").select(
            "id", count="exact"
        ).eq("tenant_id", ctx.tenant_id).execute()

        total = count_result.count or 0

        # Get this month's count using server-side filter
        from datetime import datetime, timezone
        month_start = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        month_result = admin_client.table("predictions").select(
            "id", count="exact"
        ).eq("tenant_id", ctx.tenant_id).gte("created_at", month_start).execute()
        this_month = month_result.count or 0

        # Get this week's count
        from datetime import timedelta
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        week_result = admin_client.table("predictions").select(
            "id", count="exact"
        ).eq("tenant_id", ctx.tenant_id).gte("created_at", week_ago).execute()
        this_week = week_result.count or 0

        # Get recent predictions for aggregation (limit to prevent OOM)
        result = admin_client.table("predictions").select(
            "result, processing_ms"
        ).eq("tenant_id", ctx.tenant_id).order("created_at", desc=True).limit(1000).execute()

        predictions = result.data or []

        # Average processing time
        processing_times = [
            p["processing_ms"] for p in predictions
            if p.get("processing_ms") is not None
        ]
        avg_ms = sum(processing_times) / len(processing_times) if processing_times else None

        # Most common theme and sentiment
        themes = Counter()
        sentiments = Counter()
        for p in predictions:
            r = p.get("result", {})
            if isinstance(r, dict):
                if r.get("theme"):
                    themes[r["theme"]] += 1
                if r.get("sentiment"):
                    sentiments[r["sentiment"]] += 1

        most_common_theme = themes.most_common(1)[0][0] if themes else None
        most_common_sentiment = sentiments.most_common(1)[0][0] if sentiments else None

        # Quota info
        tenant_result = admin_client.table("tenants").select(
            "monthly_limit"
        ).eq("id", ctx.tenant_id).execute()

        quota_limit = (
            tenant_result.data[0]["monthly_limit"]
            if tenant_result.data
            else 100
        )

        return AnalyticsSummary(
            total_predictions=total,
            predictions_this_week=this_week,
            predictions_this_month=this_month,
            avg_processing_ms=avg_ms,
            most_common_theme=most_common_theme,
            most_common_sentiment=most_common_sentiment,
            quota_used=this_month,
            quota_limit=quota_limit,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to compute analytics summary: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to load analytics")


@router.get("/attributes", response_model=AttributeDistributions)
async def get_attribute_distributions(
    ctx: RequestContext = Depends(require_auth),
) -> AttributeDistributions:
    """
    Get attribute distribution counts for the current tenant.

    When auth is disabled, returns empty distributions.
    """
    if not is_auth_enabled():
        return AttributeDistributions()

    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        return AttributeDistributions()

    try:
        result = admin_client.table("predictions").select(
            "result"
        ).eq("tenant_id", ctx.tenant_id).execute()

        predictions = result.data or []

        # Count distributions for each attribute
        attribute_keys = [
            "theme", "sentiment", "emotion", "dominant_colour",
            "attention_score", "trust_safety", "target_audience",
            "predicted_ctr", "likelihood_shares",
        ]

        distributions = {}
        for key in attribute_keys:
            counter = Counter()
            for p in predictions:
                r = p.get("result", {})
                if isinstance(r, dict) and r.get(key):
                    counter[r[key]] += 1
            distributions[key] = dict(counter.most_common())

        return AttributeDistributions(**distributions)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to compute attribute distributions: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to load attribute distributions")
