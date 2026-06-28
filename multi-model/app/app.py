"""
FastAPI application for the Multi-Model Prediction API.

Provides endpoints for health checks, model info, image prediction,
authentication, history, analytics, API key management, and admin operations.
The prediction pipeline is configured during application startup.

Modified to support:
- Multi-tenant authentication via Supabase
- CORS lockdown to configured frontend URL
- Feature-flagged auth (AUTH_ENABLED env var)
- Additional routers for auth, history, analytics, api-keys, admin
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Set

from dotenv import load_dotenv

# Load .env file BEFORE importing auth config (which reads env vars at import time)
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from lib.utils.lifecycle import setup_directories, cleanup_upload_directory
from lib.auth.config import is_auth_enabled, get_frontend_url

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS: Set[str] = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"}
MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
UPLOAD_DIR: Path = Path("uploads")

# Default config path for the prediction pipeline
DEFAULT_CONFIG_PATH = "backup/multi-model/configs/model/model_config.json"


# ---------------------------------------------------------------------------
# Lifespan handler (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Run setup on startup, cleanup on shutdown.

    Initializes directories and configures the prediction pipeline
    if a configuration file is available.
    """
    setup_directories()

    config_path = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)
    config_file = Path(config_path)
    if config_file.exists():
        try:
            from use_cases.prediction.pipeline import build_prediction_pipeline
            from app.predict import configure_predictor
            from lib.ocr.factory import create_ocr_engine
            from lib.utils.config import load_config

            # Resolve checkpoint path:
            #   1. CHECKPOINT_PATH env var (explicit override)
            #   2. config["checkpoint_path"] if set
            #   3. config["checkpoint_dir"]/<best-or-last>.pt
            #   4. fallback: most recent .pt in saved_models/
            checkpoint_path = os.environ.get("CHECKPOINT_PATH")
            config_data = load_config(config_path)

            if not checkpoint_path:
                checkpoint_path = config_data.get("checkpoint_path")

            if not checkpoint_path:
                checkpoint_dir = Path(
                    config_data.get("checkpoint_dir", "saved_models")
                )
                # Prefer best_model_*.pt (sorted by accuracy in filename),
                # fall back to last_model.pt, then any .pt file.
                best_candidates = sorted(checkpoint_dir.glob("best_model_*.pt"))
                last_candidate = checkpoint_dir / "last_model.pt"
                if best_candidates:
                    checkpoint_path = str(best_candidates[-1])
                elif last_candidate.exists():
                    checkpoint_path = str(last_candidate)
                else:
                    any_candidates = sorted(
                        checkpoint_dir.glob("*.pt"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if any_candidates:
                        checkpoint_path = str(any_candidates[0])

            if not checkpoint_path:
                raise FileNotFoundError(
                    "No checkpoint found. Set CHECKPOINT_PATH env var, "
                    "set 'checkpoint_path' in config, or place a .pt file "
                    f"in {config_data.get('checkpoint_dir', 'saved_models')}/."
                )

            predictor = build_prediction_pipeline(config_path, checkpoint_path)
            ocr_model_dir = Path("local/ocr")
            ocr_engine = create_ocr_engine("easyocr", ocr_model_dir)
            configure_predictor(predictor, ocr_engine)
            logger.info("Prediction pipeline configured from %s", config_path)
        except RuntimeError as config_error:
            logger.warning(
                "Could not configure prediction pipeline: %s", config_error,
            )
    else:
        logger.warning(
            "Config file not found at %s; prediction endpoint will be unavailable",
            config_path,
        )

    # Log auth status
    if is_auth_enabled():
        logger.info("Authentication ENABLED — all endpoints require auth")
    else:
        logger.warning(
            "Authentication DISABLED — running in open mode. "
            "Set AUTH_ENABLED=true for production."
        )

    yield

    cleanup_upload_directory(UPLOAD_DIR)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Multi-Model Prediction API",
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware — locked down to configured frontend URL
# ---------------------------------------------------------------------------
frontend_url = get_frontend_url()
allowed_origins = [frontend_url]

# In development, also allow localhost variants
if os.environ.get("ENV") == "dev" or not is_auth_enabled():
    allowed_origins = [
        frontend_url,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key", "Authorization"],  # Required for API key auth
)

logger.info("CORS allowed origins: %s", allowed_origins)

# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

# Original prediction router
from app.predict import router as predict_router  # noqa: E402
app.include_router(predict_router)

# Auth router — registration, login, refresh, logout
from app.auth_router import router as auth_router  # noqa: E402
app.include_router(auth_router)

# History router — prediction history with search/filter
from app.history_router import router as history_router  # noqa: E402
app.include_router(history_router)

# Analytics router — attribute distributions and summary stats
from app.analytics_router import router as analytics_router  # noqa: E402
app.include_router(analytics_router)

# API Keys router — create, list, revoke, test API keys
from app.api_keys_router import router as api_keys_router  # noqa: E402
app.include_router(api_keys_router)

# Admin router — platform admin tenant management
from app.admin_router import router as admin_router  # noqa: E402
app.include_router(admin_router)

# Invites router — team invitation management
from app.invites_router import router as invites_router  # noqa: E402
app.include_router(invites_router)


# ---------------------------------------------------------------------------
# Root endpoint — basic service info (avoids 404 on GET /)
# ---------------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root() -> Dict[str, Any]:
    """Return basic service info and a list of mounted routes."""
    return {
        "service": "Multi-Model Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "auth_enabled": is_auth_enabled(),
        "endpoints": [route.path for route in app.routes],
    }


# ---------------------------------------------------------------------------
# HTTP boundary helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    """
    Check whether the uploaded file has an allowed extension.

    Args:
        filename: The original filename string from the upload.

    Returns:
        True if the file extension is in ALLOWED_EXTENSIONS, False otherwise.
    """
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def save_upload_file(upload_file: UploadFile, dest_folder: Path) -> str:
    """
    Save an uploaded file to the destination folder.

    Args:
        upload_file: The FastAPI UploadFile object.
        dest_folder: Directory where the file should be stored.

    Returns:
        The full path (as a string) to the saved file.

    Raises:
        HTTPException: If the file extension is not allowed or save fails.
    """
    if not allowed_file(upload_file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed extensions: {ALLOWED_EXTENSIONS}",
        )

    dest_folder.mkdir(parents=True, exist_ok=True)
    # Sanitize filename to prevent path traversal attacks
    safe_filename = Path(upload_file.filename).name
    # Append UUID to prevent filename collisions
    stem = Path(safe_filename).stem
    suffix = Path(safe_filename).suffix
    unique_filename = f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
    dest_path = dest_folder / unique_filename

    # Read in chunks to check size before loading entire file into memory
    file_content = bytearray()
    chunk_size = 64 * 1024  # 64KB chunks
    while True:
        chunk = upload_file.file.read(chunk_size)
        if not chunk:
            break
        file_content.extend(chunk)
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE} bytes.",
            )

    with open(dest_path, "wb") as buffer:
        buffer.write(file_content)

    return str(dest_path)


# ---------------------------------------------------------------------------
# Route definitions
# ---------------------------------------------------------------------------
@app.get("/health", tags=["meta"])
def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Dictionary with a status message and version.
    """
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    """
    Model information endpoint.

    Returns:
        Dictionary with model name and version.
    """
    return {"model": "fg_mfn", "version": "1.0"}


if __name__ == "__main__":
    import uvicorn

    is_dev = os.environ.get("ENV") == "dev"
    uvicorn.run(
        "app.app:app",
        host="0.0.0.0",
        port=8000,
        reload=is_dev,
        log_level="info",
    )
