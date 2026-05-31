"""
Prediction API router.

Provides FastAPI endpoints for single and batch image prediction,
wired to the prediction use cases.

Modified to support:
- Authentication via JWT or API key (feature-flagged)
- Prediction persistence to Supabase
- Tenant quota enforcement
"""

import logging
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from PIL import Image

from lib.ocr.engine import OCREngine
from lib.services.predictor import Predictor
from lib.auth.deps import require_auth
from lib.auth.middleware import RequestContext
from lib.auth.config import is_auth_enabled, get_supabase_admin_client
from use_cases.prediction.predict_image import predict_image

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level references set during application startup
_predictor: Optional[Predictor] = None
_ocr_engine: Optional[OCREngine] = None


def configure_predictor(predictor: Predictor, ocr_engine: OCREngine) -> None:
    """
    Set the module-level predictor and OCR engine references.

    Called once during application startup to wire the prediction
    dependencies into the router.

    Args:
        predictor: A configured Predictor instance.
        ocr_engine: A configured OCREngine instance.
    """
    global _predictor, _ocr_engine
    _predictor = predictor
    _ocr_engine = ocr_engine
    logger.info("Prediction router configured with predictor and OCR engine")


def _load_image_from_upload(upload_file: UploadFile) -> Image.Image:
    """
    Load a PIL Image from an uploaded file.

    Args:
        upload_file: The FastAPI UploadFile object.

    Returns:
        A PIL Image in RGB mode.

    Raises:
        HTTPException: If the file cannot be read or is not a valid image.
    """
    try:
        file_content = upload_file.file.read()
        upload_file.file.seek(0)  # Reset file pointer for potential re-reads
        pil_image = Image.open(BytesIO(file_content)).convert("RGB")
        return pil_image
    except IOError as io_error:
        logger.error("Failed to load image from upload: %s", io_error)
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image file: {io_error}",
        ) from io_error


def _check_quota(tenant_id: str) -> Dict[str, Any]:
    """
    Check the tenant's monthly prediction quota.

    Args:
        tenant_id: UUID of the tenant.

    Returns:
        Dict with 'used', 'limit', 'remaining', and 'reset_seconds'.

    Raises:
        HTTPException: 500 if quota check fails.
    """
    try:
        admin_client = get_supabase_admin_client()
    except ValueError:
        # If Supabase not configured, allow unlimited
        return {"used": 0, "limit": 999999, "remaining": 999999, "reset_seconds": 0}

    try:
        # Get tenant's monthly limit
        tenant_result = admin_client.table("tenants").select(
            "monthly_limit"
        ).eq("id", tenant_id).execute()

        monthly_limit = (
            tenant_result.data[0]["monthly_limit"]
            if tenant_result.data
            else 100
        )

        # Count predictions this month
        from datetime import datetime, timezone
        month_start = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ).isoformat()

        predictions_result = admin_client.table("predictions").select(
            "id", count="exact"
        ).eq("tenant_id", tenant_id).gte("created_at", month_start).execute()

        used = predictions_result.count or 0
        remaining = max(0, monthly_limit - used)

        # Calculate seconds until next month
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        reset_seconds = int((next_month - now).total_seconds())

        return {
            "used": used,
            "limit": monthly_limit,
            "remaining": remaining,
            "reset_seconds": reset_seconds,
        }

    except Exception as exc:
        logger.warning("Quota check failed for tenant %s: %s", tenant_id, exc)
        # Fail open — allow the request if quota check fails
        return {"used": 0, "limit": 999999, "remaining": 999999, "reset_seconds": 0}


def _save_prediction(
    tenant_id: str,
    user_id: str,
    filename: str,
    ocr_text: str,
    result: Dict[str, Any],
    processing_ms: int,
) -> Optional[str]:
    """
    Save a prediction to Supabase.

    Args:
        tenant_id: UUID of the tenant.
        user_id: UUID of the user.
        filename: Original filename.
        ocr_text: Extracted OCR text.
        result: Full prediction result dict.
        processing_ms: Processing time in milliseconds.

    Returns:
        The prediction UUID, or None if save fails.
    """
    try:
        admin_client = get_supabase_admin_client()
        insert_result = admin_client.table("predictions").insert({
            "tenant_id": tenant_id,
            "user_id": user_id,
            "filename": filename,
            "ocr_text": ocr_text,
            "result": result,
            "processing_ms": processing_ms,
        }).execute()

        if insert_result.data:
            return insert_result.data[0]["id"]
        return None

    except Exception as exc:
        logger.warning("Failed to save prediction: %s", exc)
        # Don't fail the request — persistence is non-critical
        return None


@router.post("/predict")
async def predict_endpoint(
    files: List[UploadFile] = File(...),
    ctx: RequestContext = Depends(require_auth),
) -> Dict[str, Any]:
    """
    Prediction endpoint for uploaded images.

    Accepts one or more image files and returns multi-attribute
    predictions for each image.

    When AUTH_ENABLED is true:
    - Requires authentication (JWT or API key)
    - Enforces tenant quota
    - Persists predictions to Supabase
    - Includes prediction.id in response

    When AUTH_ENABLED is false:
    - Works without authentication (backward compatible)
    - No quota enforcement
    - No persistence

    Args:
        files: List of uploaded image files.
        ctx: Authenticated user context (or default when auth disabled).

    Returns:
        Dictionary containing 'predictions' list, 'total_images' count,
        and 'processing_time_ms' duration.

    Raises:
        HTTPException: If the predictor is not configured or prediction fails.
        HTTPException: 429 if tenant quota is exceeded.
    """
    if _predictor is None or _ocr_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not configured. "
            "Ensure the model and OCR engine are loaded.",
        )

    # ── Quota check (only when auth is enabled) ────────────────────────
    if is_auth_enabled():
        quota = _check_quota(ctx.tenant_id)
        if quota["remaining"] <= 0:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Monthly prediction limit reached "
                    f"({quota['used']}/{quota['limit']}). "
                    f"Upgrade your plan at /settings"
                ),
                headers={"Retry-After": str(quota["reset_seconds"])},
            )

    start_time = time.monotonic()

    results = []
    for upload_file in files:
        pil_image = _load_image_from_upload(upload_file)
        try:
            result = predict_image(
                image=pil_image,
                model=_predictor.model,
                ocr_engine=_ocr_engine,
                label_maps=_predictor.label_maps,
                filename=upload_file.filename or "",
                predictor=_predictor,
            )
            results.append(result)
        except RuntimeError as prediction_error:
            logger.error(
                "Prediction failed for file '%s': %s",
                upload_file.filename, prediction_error,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {prediction_error}",
            ) from prediction_error

    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    # ── Persist predictions to Supabase (only when auth is enabled) ────
    if is_auth_enabled():
        for result in results:
            prediction_id = _save_prediction(
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                filename=result.get("filename", ""),
                ocr_text=result.get("ocr_text", ""),
                result=result,
                processing_ms=elapsed_ms,
            )
            if prediction_id:
                result["id"] = prediction_id

    return {
        "predictions": results,
        "total_images": len(results),
        "processing_time_ms": elapsed_ms,
    }
