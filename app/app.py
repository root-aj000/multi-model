"""
FastAPI Server for Multi-Modal Classification
==============================================
This module provides a REST API for the FG_MFN model.
It handles image uploads, runs predictions, and returns results
for multiple attributes.

Features:
- Multi-file upload support
- Multi-attribute prediction (9 attributes)
- Automatic file cleanup
- CORS support for web clients
- Comprehensive error handling
- Detailed logging
- Health check endpoint
- Proper lifespan management (startup/shutdown)

Endpoints:
- POST /predict - Upload images and get predictions
- GET /health - Check if server is running
- GET /model/info - Get model information

Author: [Your Name]
Date: [Date]
"""

import os
import sys
import shutil
import logging
import traceback
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import torch
import uvicorn

# Import project modules
from app.old_predict import predict, IMAGE_UPLOAD_DIR
from utils.path import LOG_DIR


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

# Maximum file size in bytes (10 MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Maximum number of files per request
MAX_FILES_PER_REQUEST = 10

# CORS allowed origins
ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8000",
]


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Create logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Prevent duplicate handlers
if not logger.handlers:
    # File handler
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "server.log"),
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


# =============================================================================
# GLOBAL STATE (initialized during lifespan)
# =============================================================================

# These will be set during startup
app_state = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "model_loaded": False,
    "startup_time": None
}


# =============================================================================
# DIRECTORY SETUP
# =============================================================================

def setup_directories() -> None:
    """Create and clean necessary directories."""
    try:
        logger.info("Setting up server directories...")
        
        # Create log directory
        os.makedirs(LOG_DIR, exist_ok=True)
        logger.info(f"✓ Log directory: {LOG_DIR}")
        
        # Create image upload directory
        os.makedirs(IMAGE_UPLOAD_DIR, exist_ok=True)
        logger.info(f"✓ Upload directory: {IMAGE_UPLOAD_DIR}")
        
        # Clean existing files in upload directory
        existing_files = os.listdir(IMAGE_UPLOAD_DIR)
        if existing_files:
            logger.info(f"Cleaning {len(existing_files)} existing files...")
            for filename in existing_files:
                try:
                    file_path = os.path.join(IMAGE_UPLOAD_DIR, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove {filename}: {e}")
            logger.info("✓ Upload directory cleaned")
        
        logger.info("✓ Directory setup complete")
        
    except OSError as e:
        logger.error(f"Failed to create directories: {e}")
        raise


# =============================================================================
# LIFESPAN CONTEXT MANAGER (Startup/Shutdown Events)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    This handles:
    - Startup: Load model, tokenizer, initialize resources
    - Shutdown: Cleanup resources, clear GPU memory
    
    Usage:
        app = FastAPI(lifespan=lifespan)
    """
    global app_state
    
    # =========================================================================
    # STARTUP
    # =========================================================================
    logger.info("=" * 80)
    logger.info("🚀 SERVER STARTUP")
    logger.info("=" * 80)
    
    try:
        # Record startup time
        app_state["startup_time"] = datetime.now()
        
        # Setup directories
        setup_directories()
        
        # Set device
        app_state["device"] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Device: {app_state['device']}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Load model (optional - uncomment if you want to preload)
        # await load_model()
        
        logger.info("=" * 80)
        logger.info("✅ SERVER READY")
        logger.info("=" * 80)
        logger.info(f"CORS origins: {ALLOWED_ORIGINS}")
        logger.info(f"Max file size: {MAX_FILE_SIZE // (1024*1024)} MB")
        logger.info(f"Max files per request: {MAX_FILES_PER_REQUEST}")
        logger.info(f"Allowed extensions: {ALLOWED_EXTENSIONS}")
        logger.info("=" * 80)
        logger.info("API Documentation:")
        logger.info("  📚 Swagger UI: http://localhost:8000/docs")
        logger.info("  📖 ReDoc: http://localhost:8000/redoc")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # =========================================================================
    # APPLICATION RUNS HERE
    # =========================================================================
    yield
    
    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    logger.info("=" * 80)
    logger.info("🛑 SERVER SHUTDOWN")
    logger.info("=" * 80)
    
    try:
        # Clean up upload directory
        cleanup_upload_directory()
        
        # Clear model from memory
        app_state["model"] = None
        app_state["tokenizer"] = None
        app_state["model_loaded"] = False
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✓ GPU cache cleared")
        
        # Calculate uptime
        if app_state["startup_time"]:
            uptime = datetime.now() - app_state["startup_time"]
            logger.info(f"Total uptime: {uptime}")
        
        logger.info("✓ Shutdown complete")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


def cleanup_upload_directory() -> None:
    """Clean up all files in the upload directory."""
    try:
        if os.path.exists(IMAGE_UPLOAD_DIR):
            files = os.listdir(IMAGE_UPLOAD_DIR)
            if files:
                logger.info(f"Cleaning up {len(files)} remaining file(s)...")
                for filename in files:
                    try:
                        file_path = os.path.join(IMAGE_UPLOAD_DIR, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove {filename}: {e}")
                logger.info("✓ Cleanup complete")
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")


async def load_model() -> None:
    """
    Load the FG_MFN model and tokenizer.
    
    This is called during startup to preload the model,
    or can be called on first request (lazy loading).
    """
    global app_state
    
    if app_state["model_loaded"]:
        logger.info("Model already loaded")
        return
    
    try:
        logger.info("Loading model...")
        
        import json
        from models.fg_mfn import FG_MFN
        from transformers import AutoTokenizer
        
        # Load config
        config_path = "models/configs/model_config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        
        logger.info(f"Config loaded from: {config_path}")
        
        # Create model
        model = FG_MFN(cfg)
        
        # Load trained weights
        model_path = "saved_models/model_best.pt"
        try:
            state_dict = torch.load(
                model_path,
                map_location=app_state["device"]
            )
            model.load_state_dict(state_dict)
            logger.info(f"✓ Loaded weights from: {model_path}")
        except FileNotFoundError:
            logger.warning(f"⚠️ No trained weights found at: {model_path}")
            logger.warning("Model will use random weights")
        
        # Move to device and set to eval mode
        model = model.to(app_state["device"])
        model.eval()
        
        app_state["model"] = model
        logger.info("✓ Model loaded and ready")
        
        # Load tokenizer
        encoder_name = cfg.get("TEXT_ENCODER", "distilbert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained(
            encoder_name,
            cache_dir="local/BERT_MODELS/text_models"
        )
        app_state["tokenizer"] = tokenizer
        logger.info(f"✓ Tokenizer loaded: {encoder_name}")
        
        app_state["model_loaded"] = True
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable:,} trainable")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise


# =============================================================================
# CREATE FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Multi-Modal Classification API",
    description="""
    API for predicting multiple attributes from advertisement images 
    using the Fine-Grained Multi-Modal Fusion Network (FG_MFN).
    
    ## Features
    - Upload single or multiple images
    - Get predictions for 9 different attributes
    - OCR text extraction
    - Keyword and feature extraction
    
    ## Attributes Predicted
    - **theme**: Topic/theme of the content
    - **sentiment**: Positive/Negative/Neutral
    - **emotion**: Specific emotion detected
    - **dominant_colour**: Main color in image
    - **attention_score**: How attention-grabbing
    - **trust_safety**: Safety level
    - **target_audience**: Intended audience
    - **predicted_ctr**: Click-through rate prediction
    - **likelihood_shares**: Share likelihood
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # ← Use lifespan context manager
)

logger.info("✓ FastAPI application created")


# =============================================================================
# CORS MIDDLEWARE
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"✓ CORS configured for origins: {ALLOWED_ORIGINS}")


# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================

class PredictionResult(BaseModel):
    """Schema for a single prediction result."""
    
    filename: str = Field(..., description="Original filename")
    
    # Legacy fields
    predicted_label_text: str = Field(..., description="Primary prediction label")
    predicted_label_num: int = Field(..., description="Primary prediction class")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    
    # OCR
    ocr_text: str = Field(default="", description="Extracted text")
    
    # Multi-attribute predictions (all optional)
    theme: Optional[str] = None
    theme_confidence: Optional[float] = None
    
    sentiment: Optional[str] = None
    sentiment_confidence: Optional[float] = None
    
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    
    dominant_colour: Optional[str] = None
    dominant_colour_confidence: Optional[float] = None
    
    attention_score: Optional[str] = None
    attention_score_confidence: Optional[float] = None
    
    trust_safety: Optional[str] = None
    trust_safety_confidence: Optional[float] = None
    
    target_audience: Optional[str] = None
    target_audience_confidence: Optional[float] = None
    
    predicted_ctr: Optional[str] = None
    predicted_ctr_confidence: Optional[float] = None
    
    likelihood_shares: Optional[str] = None
    likelihood_shares_confidence: Optional[float] = None
    
    # Extracted features
    keywords: Optional[str] = None
    monetary_mention: Optional[str] = None
    call_to_action: Optional[str] = None
    object_detected: Optional[str] = None
    
    class Config:
        extra = "allow"


class PredictionResponse(BaseModel):
    """Schema for prediction endpoint response."""
    
    predictions: List[PredictionResult]
    total_images: int = Field(..., ge=0)
    processing_time_ms: Optional[float] = Field(None, ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [{
                    "filename": "ad.jpg",
                    "predicted_label_text": "positive",
                    "predicted_label_num": 2,
                    "confidence_score": 0.95,
                    "sentiment": "positive",
                    "sentiment_confidence": 0.95
                }],
                "total_images": 1,
                "processing_time_ms": 1250.5
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response."""
    
    status: str
    message: str
    timestamp: str
    model_loaded: bool = False
    device: Optional[str] = None
    uptime_seconds: Optional[float] = None


class ModelInfoResponse(BaseModel):
    """Schema for model info response."""
    
    model_loaded: bool
    attributes: Optional[List[str]] = None
    device: Optional[str] = None
    total_params: Optional[int] = None
    trainable_params: Optional[int] = None
    fusion_type: Optional[str] = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def allowed_file(filename: str) -> bool:
    """Check if file has an allowed extension."""
    if not filename:
        return False
    
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in ALLOWED_EXTENSIONS


def save_upload_file(upload_file: UploadFile, dest_folder: str) -> str:
    """
    Save uploaded file with unique name.
    
    Args:
        upload_file: FastAPI UploadFile
        dest_folder: Destination directory
    
    Returns:
        Path to saved file
    
    Raises:
        HTTPException: If file is invalid
    """
    if not upload_file or not upload_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate extension
    ext = upload_file.filename.rsplit('.', 1)[-1].lower() if '.' in upload_file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '.{ext}'. Allowed: {list(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate unique filename
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(dest_folder, unique_name)
    
    # Ensure directory exists
    os.makedirs(dest_folder, exist_ok=True)
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Validate saved file
    file_size = os.path.getsize(file_path)
    
    if file_size == 0:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    
    if file_size > MAX_FILE_SIZE:
        os.remove(file_path)
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({file_size} bytes). Max: {MAX_FILE_SIZE // (1024*1024)} MB"
        )
    
    logger.debug(f"Saved: {upload_file.filename} -> {unique_name} ({file_size} bytes)")
    return file_path


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint - basic API info."""
    return {
        "name": "Multi-Modal Classification API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns server status, model state, and uptime.
    """
    uptime = None
    if app_state["startup_time"]:
        uptime = (datetime.now() - app_state["startup_time"]).total_seconds()
    
    return HealthResponse(
        status="ok",
        message="API is running",
        timestamp=datetime.now().isoformat(),
        model_loaded=app_state["model_loaded"],
        device=str(app_state["device"]) if app_state["device"] else None,
        uptime_seconds=uptime
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about the loaded model."""
    
    if not app_state["model_loaded"] or app_state["model"] is None:
        return ModelInfoResponse(
            model_loaded=False,
            attributes=None,
            device=str(app_state["device"]) if app_state["device"] else None
        )
    
    model = app_state["model"]
    
    return ModelInfoResponse(
        model_loaded=True,
        attributes=list(model.attribute_heads.keys()),
        device=str(app_state["device"]),
        total_params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        fusion_type=getattr(model, 'fusion_type', None)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    files: List[UploadFile] = File(
        ...,
        description=f"Images to analyze (max {MAX_FILES_PER_REQUEST}, max {MAX_FILE_SIZE//(1024*1024)}MB each)"
    )
):
    """
    Predict attributes from uploaded images.
    
    Upload one or more images to get predictions for:
    - Sentiment, emotion, theme
    - Dominant color, attention score
    - Trust/safety, target audience
    - Predicted CTR, share likelihood
    
    Also extracts:
    - OCR text from image
    - Keywords, monetary mentions, CTAs
    """
    start_time = datetime.now()
    
    images = []
    filenames = []
    uploaded_paths = []
    
    try:
        # Validate request
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        if len(files) > MAX_FILES_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files ({len(files)}). Max: {MAX_FILES_PER_REQUEST}"
            )
        
        logger.info(f"Processing {len(files)} file(s)...")
        
        # Process each file
        for idx, file in enumerate(files, 1):
            try:
                # Validate filename
                if not file.filename:
                    logger.warning(f"File {idx} has no filename, skipping")
                    continue
                
                # Check extension
                if not allowed_file(file.filename):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid file type: {file.filename}. Allowed: {list(ALLOWED_EXTENSIONS)}"
                    )
                
                # Save file
                file_path = save_upload_file(file, IMAGE_UPLOAD_DIR)
                uploaded_paths.append(file_path)
                
                # Open and validate image
                try:
                    img = Image.open(file_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                except Exception as e:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        uploaded_paths.remove(file_path)
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot open image '{file.filename}': {e}"
                    )
                
                images.append(img)
                filenames.append(file.filename)
                
                logger.debug(f"Processed: {file.filename} ({img.size[0]}x{img.size[1]})")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error processing file {idx}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        if not images:
            raise HTTPException(status_code=400, detail="No valid images to process")
        
        # Run predictions
        logger.info(f"Running predictions on {len(images)} image(s)...")
        
        try:
            results = predict(images)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
        
        # Attach filenames
        for i, result in enumerate(results):
            result["filename"] = filenames[i] if i < len(filenames) else f"image_{i}.jpg"
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"✓ Predictions complete: {len(results)} results in {processing_time:.1f}ms")
        
        return PredictionResponse(
            predictions=results,
            total_images=len(results),
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")
        
    finally:
        # Cleanup uploaded files
        for path in uploaded_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Failed to cleanup {path}: {e}")


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting server...")
    
    uvicorn.run(
        "app.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )