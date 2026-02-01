"""
FastAPI Server for Multi-Modal Classification
==============================================
This module provides a REST API for the FG_MFN model.
It handles image uploads, runs predictions, and returns results
for multiple attributes WITHOUT showing confidence scores.

Features:
- Multi-file upload support
- Multi-attribute prediction (9 attributes)
- Automatic file cleanup
- CORS support for web clients
- Comprehensive error handling
- Detailed logging
- Health check endpoint
- No accuracy/confidence scores exposed to clients

Endpoints:
- POST /predict - Upload images and get predictions
- GET /health - Check if server is running

Author: [Your Name]
Date: [Date]
"""

import os
import shutil
import logging
import traceback
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from app.predict import predict, IMAGE_UPLOAD_DIR
from utils.path import LOG_DIR


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Allowed file extensions for image uploads
# Only these formats are accepted to prevent security issues
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"]

# Maximum file size in bytes (10 MB)
# This prevents denial-of-service attacks via huge file uploads
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Maximum number of files per request
# Prevents server overload from too many simultaneous predictions
MAX_FILES_PER_REQUEST = 10

# CORS allowed origins
# Add your frontend URLs here
# Use ["*"] to allow all origins (not recommended for production)
ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",  # Local development
    "http://localhost:5500",   # Local development
    "http://localhost:3000",   # React default port
    "http://localhost:8080",   # Vue default port
]


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure logging for the server
# Logs are written to both file and console for debugging

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Create logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler - saves all logs to file
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

# Console handler - prints logs to terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("=" * 80)
logger.info("FASTAPI SERVER STARTING")
logger.info("=" * 80)


# =============================================================================
# DIRECTORY SETUP
# =============================================================================

def setup_directories() -> None:
    """
    Create necessary directories for server operation.
    
    This function ensures all required directories exist before
    the server starts accepting requests.
    
    Raises:
        OSError: If directory creation fails
    """
    try:
        logger.info("Setting up server directories...")
        
        # Create log directory
        os.makedirs(LOG_DIR, exist_ok=True)
        logger.info(f"✓ Log directory: {LOG_DIR}")
        
        # Create image upload directory
        os.makedirs(IMAGE_UPLOAD_DIR, exist_ok=True)
        logger.info(f"✓ Upload directory: {IMAGE_UPLOAD_DIR}")
        
        # Clean any existing files in upload directory
        # This prevents leftover files from previous runs
        existing_files = os.listdir(IMAGE_UPLOAD_DIR)
        if existing_files:
            logger.info(f"Cleaning {len(existing_files)} existing files from upload directory...")
            for filename in existing_files:
                try:
                    file_path = os.path.join(IMAGE_UPLOAD_DIR, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove {filename}: {str(e)}")
            logger.info("✓ Upload directory cleaned")
        
        logger.info("✓ Directory setup complete")
        
    except OSError as e:
        error_msg = f"Failed to create directories: {str(e)}"
        logger.error(error_msg)
        raise OSError(error_msg)


# Initialize directories
setup_directories()


# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

# Create FastAPI application instance
# The title and description appear in the auto-generated API documentation
app = FastAPI(
    title="Multi-Modal Classification API",
    description="API for predicting multiple attributes from advertisement images using FG_MFN model",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc"  # ReDoc UI at http://localhost:8000/redoc
)

logger.info("✓ FastAPI application created")


# =============================================================================
# CORS MIDDLEWARE SETUP
# =============================================================================

# Configure CORS (Cross-Origin Resource Sharing)
# This allows web browsers to make requests from different domains
# Without CORS, browsers block requests from other origins for security

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Which domains can access the API
    allow_credentials=True,          # Allow cookies and authentication
    allow_methods=["*"],             # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],             # Allow all headers
)

logger.info(f"✓ CORS configured for origins: {ALLOWED_ORIGINS}")


# =============================================================================
# PYDANTIC MODELS (REQUEST/RESPONSE SCHEMAS)
# =============================================================================

class PredictionResult(BaseModel):
    """
    Schema for a single prediction result.
    
    This defines the structure of prediction data returned for each image.
    It includes all 9 attributes plus additional extracted features.
    
    NOTE: Confidence scores and accuracy metrics are NOT included
    to keep the interface simple and focused on predictions.
    
    Attributes:
        filename: Original filename of the uploaded image
        
        # Primary prediction (main result)
        predicted_label: Main predicted label
        
        # Multi-attribute predictions (9 attributes)
        theme: Detected theme/topic
        sentiment: Sentiment (positive/negative/neutral)
        emotion: Specific emotion detected
        dominant_colour: Main color in the image
        attention_score: How attention-grabbing the content is
        trust_safety: Safety and trustworthiness level
        target_audience: Intended audience type
        predicted_ctr: Click-through rate prediction
        likelihood_shares: Likelihood of being shared
        
        # OCR and extracted features
        ocr_text: Text extracted from image
        keywords: Important keywords from text
        monetary_mention: Price/discount information
        call_to_action: CTA phrases detected
        object_detected: Product categories detected
    """
    
    # Required fields
    filename: str = Field(..., description="Original filename of uploaded image")
    
    # Primary prediction (for backward compatibility)
    predicted_label: str = Field(..., description="Main predicted label")
    
    # OCR result
    ocr_text: str = Field(default="", description="Text extracted from image via OCR")
    
    # Optional fields - multi-attribute predictions (9 attributes)
    # Note: No confidence scores included - just the predictions
    theme: Optional[str] = Field(None, description="Detected theme/topic")
    sentiment: Optional[str] = Field(None, description="Sentiment classification")
    emotion: Optional[str] = Field(None, description="Specific emotion detected")
    dominant_colour: Optional[str] = Field(None, description="Main color in image")
    attention_score: Optional[str] = Field(None, description="Attention-grabbing level")
    trust_safety: Optional[str] = Field(None, description="Safety/trustworthiness level")
    target_audience: Optional[str] = Field(None, description="Intended audience type")
    predicted_ctr: Optional[str] = Field(None, description="Click-through rate prediction")
    likelihood_shares: Optional[str] = Field(None, description="Share likelihood")
    
    # Extracted text features
    keywords: Optional[str] = Field(None, description="Extracted keywords from text")
    monetary_mention: Optional[str] = Field(None, description="Price/discount mentions")
    call_to_action: Optional[str] = Field(None, description="Call-to-action phrases")
    object_detected: Optional[str] = Field(None, description="Detected product categories")
    
    class Config:
        # Allow additional fields that aren't defined in the schema
        # This makes the model flexible for future attributes
        extra = "allow"
        
        # Example for documentation
        schema_extra = {
            "example": {
                "filename": "ad_image.jpg",
                "predicted_label": "positive",
                "ocr_text": "Buy Now 50% OFF",
                "theme": "sales",
                "sentiment": "positive",
                "emotion": "excited",
                "dominant_colour": "red",
                "attention_score": "high",
                "trust_safety": "safe",
                "target_audience": "adults",
                "predicted_ctr": "high",
                "likelihood_shares": "likely",
                "keywords": "Buy Now Discount",
                "monetary_mention": "50% OFF",
                "call_to_action": "Buy Now",
                "object_detected": "General"
            }
        }


class PredictionResponse(BaseModel):
    """
    Schema for the prediction endpoint response.
    
    This wraps multiple prediction results in a single response.
    
    Attributes:
        predictions: List of prediction results, one per uploaded image
        total_images: Total number of images processed
        processing_time_ms: Time taken to process all images (milliseconds)
    """
    
    predictions: List[PredictionResult] = Field(..., description="List of predictions")
    total_images: int = Field(..., ge=0, description="Total number of images processed")
    processing_time_ms: Optional[float] = Field(None, ge=0, description="Processing time in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "filename": "image1.jpg",
                        "predicted_label": "positive",
                        "ocr_text": "Buy Now 50% OFF",
                        "sentiment": "positive",
                        "theme": "sales",
                        "emotion": "excited"
                    }
                ],
                "total_images": 1,
                "processing_time_ms": 1250.5
            }
        }


class ErrorResponse(BaseModel):
    """
    Schema for error responses.
    
    This provides consistent error messages to clients.
    
    Attributes:
        detail: Human-readable error message
        error_code: Optional error code for programmatic handling
        timestamp: When the error occurred
    """
    
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code for programmatic handling")
    timestamp: str = Field(..., description="ISO format timestamp of error")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "File size exceeds maximum allowed size of 10 MB",
                "error_code": "FILE_TOO_LARGE",
                "timestamp": "2024-01-15T14:30:22.123456"
            }
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def allowed_file(filename: str) -> bool:
    """
    Check if a file has an allowed extension.
    
    This function validates that uploaded files are images with
    allowed extensions. This prevents users from uploading
    executable files or other non-image formats.
    
    Args:
        filename (str): Name of the file to check
    
    Returns:
        bool: True if file extension is allowed, False otherwise
    
    Example:
        >>> allowed_file("image.jpg")
        True
        >>> allowed_file("document.pdf")
        False
        >>> allowed_file("photo.PNG")
        True
        >>> allowed_file("archive.zip")
        False
    
    Note:
        Extension check is case-insensitive (jpg = JPG = JpG)
    """
    try:
        # Validate input
        if not filename:
            logger.warning("Empty filename provided to allowed_file()")
            return False
        
        if not isinstance(filename, str):
            logger.warning(f"Non-string filename: {type(filename)}")
            return False
        
        # Extract file extension
        # os.path.splitext returns ('filename', '.ext')
        # We take the extension and remove the leading dot
        file_extension = os.path.splitext(filename)[1].lower().lstrip('.')
        
        # Check if extension is in allowed list
        is_allowed = file_extension in ALLOWED_EXTENSIONS
        
        if not is_allowed:
            logger.debug(f"File '{filename}' has disallowed extension: {file_extension}")
        
        return is_allowed
        
    except Exception as e:
        logger.error(f"Error checking file extension: {str(e)}")
        logger.error(traceback.format_exc())
        # If we can't determine, reject the file for safety
        return False


def save_upload_file(upload_file: UploadFile, dest_folder: str) -> str:
    """
    Save an uploaded file to the destination folder with a unique name.
    
    This function:
    1. Validates the file extension
    2. Generates a unique filename to prevent collisions
    3. Saves the file to disk
    4. Returns the path to the saved file
    
    Args:
        upload_file (UploadFile): FastAPI UploadFile object
        dest_folder (str): Directory to save the file
    
    Returns:
        str: Full path to the saved file
    
    Raises:
        HTTPException: If file type is not allowed
        OSError: If file saving fails
    
    Example:
        >>> file = await request.files['image']
        >>> path = save_upload_file(file, '/tmp/uploads')
        >>> print(path)
        '/tmp/uploads/a1b2c3d4e5f6.jpg'
    
    Note:
        - Generates UUID-based filenames to avoid conflicts
        - Original extension is preserved
        - File is saved in binary mode to preserve data
    """
    try:
        # Step 1: Validate upload_file object
        if not upload_file:
            error_msg = "No upload file provided"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        if not upload_file.filename:
            error_msg = "Upload file has no filename"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.debug(f"Processing upload: {upload_file.filename}")
        
        # Step 2: Extract and validate file extension
        # os.path.splitext returns ('filename', '.ext')
        file_extension = os.path.splitext(upload_file.filename)[1].lower()
        
        # Remove the dot and check if allowed
        extension_without_dot = file_extension.lstrip('.')
        
        if extension_without_dot not in ALLOWED_EXTENSIONS:
            error_msg = f"Unsupported file type: {file_extension}. Allowed types: {ALLOWED_EXTENSIONS}"
            logger.warning(error_msg)
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
        
        logger.debug(f"File extension validated: {file_extension}")
        
        # Step 3: Generate unique filename
        # UUID ensures no filename collisions even with concurrent uploads
        # Format: {uuid}.{ext} (e.g., a1b2c3d4e5f6.jpg)
        unique_name = f"{uuid.uuid4().hex}{file_extension}"
        file_path = os.path.join(dest_folder, unique_name)
        
        logger.debug(f"Generated unique path: {file_path}")
        
        # Step 4: Ensure destination folder exists
        if not os.path.exists(dest_folder):
            logger.info(f"Creating destination folder: {dest_folder}")
            os.makedirs(dest_folder, exist_ok=True)
        
        # Step 5: Save file to disk
        # We use binary mode ('wb') to preserve the exact file data
        # shutil.copyfileobj efficiently copies the file content
        try:
            with open(file_path, "wb") as buffer:
                # Copy uploaded file content to destination
                shutil.copyfileobj(upload_file.file, buffer)
            
            logger.debug(f"✓ File saved: {file_path}")
            
        except OSError as e:
            error_msg = f"Failed to save file: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Step 6: Validate saved file
        if not os.path.exists(file_path):
            error_msg = f"File was not saved properly: {file_path}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Check file size (basic sanity check)
        file_size = os.path.getsize(file_path)
        logger.debug(f"Saved file size: {file_size} bytes")
        
        if file_size == 0:
            error_msg = "Saved file is empty (0 bytes)"
            logger.error(error_msg)
            # Clean up empty file
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=error_msg)
        
        if file_size > MAX_FILE_SIZE:
            error_msg = f"File size ({file_size} bytes) exceeds maximum ({MAX_FILE_SIZE} bytes)"
            logger.error(error_msg)
            # Clean up oversized file
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)} MB"
            )
        
        logger.info(f"✓ Successfully saved: {upload_file.filename} -> {unique_name} ({file_size} bytes)")
        
        return file_path
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        # Catch any unexpected errors
        error_msg = f"Unexpected error saving file: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


def clean_prediction_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove confidence scores and accuracy metrics from prediction results.
    
    This function filters out all confidence-related fields to provide
    a cleaner, simpler response to clients. Only the predictions
    themselves are kept.
    
    Args:
        result (dict): Raw prediction result with confidence scores
    
    Returns:
        dict: Cleaned prediction result without confidence scores
    
    Example:
        >>> raw = {
        ...     "sentiment": "positive",
        ...     "sentiment_confidence": 0.95,
        ...     "theme": "sales",
        ...     "theme_confidence": 0.92
        ... }
        >>> clean = clean_prediction_result(raw)
        >>> print(clean)
        {'sentiment': 'positive', 'theme': 'sales'}
    
    Note:
        - All fields ending with '_confidence' are removed
        - 'confidence_score' field is removed
        - 'predicted_label_num' is removed (internal numeric code)
        - All other fields are preserved
    """
    try:
        # Create a copy to avoid modifying the original
        cleaned = {}
        
        # Fields to exclude (confidence and accuracy related)
        excluded_suffixes = ['_confidence', '_score', '_accuracy', '_f1']
        excluded_fields = {'confidence_score', 'predicted_label_num'}
        
        # Filter out unwanted fields
        for key, value in result.items():
            # Skip if it's an excluded field
            if key in excluded_fields:
                logger.debug(f"Removing field: {key}")
                continue
            
            # Skip if it ends with an excluded suffix
            if any(key.endswith(suffix) for suffix in excluded_suffixes):
                logger.debug(f"Removing field: {key}")
                continue
            
            # Keep this field
            cleaned[key] = value
        
        # Rename 'predicted_label_text' to 'predicted_label' for simplicity
        if 'predicted_label_text' in cleaned:
            cleaned['predicted_label'] = cleaned.pop('predicted_label_text')
        
        logger.debug(f"Cleaned result: {len(result)} -> {len(cleaned)} fields")
        
        return cleaned
        
    except Exception as e:
        logger.error(f"Error cleaning prediction result: {str(e)}")
        logger.error(traceback.format_exc())
        # Return original if cleaning fails
        return result


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    summary="Predict attributes from images",
    description="Upload one or more images to get predictions for multiple attributes including sentiment, emotion, theme, etc. (without confidence scores)"
)
async def predict_endpoint(
    files: List[UploadFile] = File(
        ...,
        description=f"Image files to analyze (max {MAX_FILES_PER_REQUEST} files, max {MAX_FILE_SIZE//(1024*1024)}MB each)"
    )
) -> PredictionResponse:
    """
    Predict multiple attributes from uploaded images.
    
    This endpoint:
    1. Accepts multiple image uploads
    2. Validates file types and sizes
    3. Extracts text using OCR
    4. Predicts 9 different attributes using the FG_MFN model
    5. Extracts additional features (keywords, prices, CTAs)
    6. Returns predictions WITHOUT confidence scores
    7. Automatically cleans up uploaded files
    
    Args:
        files: List of uploaded image files
    
    Returns:
        PredictionResponse: Predictions for all uploaded images
    
    Raises:
        HTTPException 400: Invalid input (wrong file type, too many files, etc.)
        HTTPException 500: Server error during processing
    
    Example Request (curl):
        curl -X POST "http://localhost:8000/predict" \\
             -F "files=@image1.jpg" \\
             -F "files=@image2.png"
    
    Example Response:
        {
            "predictions": [
                {
                    "filename": "image1.jpg",
                    "predicted_label": "positive",
                    "ocr_text": "Buy Now 50% OFF",
                    "sentiment": "positive",
                    "theme": "sales",
                    "emotion": "excited",
                    "dominant_colour": "red",
                    "attention_score": "high",
                    "trust_safety": "safe",
                    "target_audience": "adults",
                    "predicted_ctr": "high",
                    "likelihood_shares": "likely",
                    "keywords": "Buy Now Discount",
                    "monetary_mention": "50% OFF",
                    "call_to_action": "Buy Now",
                    "object_detected": "General"
                }
            ],
            "total_images": 1,
            "processing_time_ms": 1250.5
        }
    """
    # Start timing
    start_time = datetime.now()
    
    # Initialize tracking variables
    images = []           # PIL Image objects
    filenames = []        # Original filenames
    uploaded_paths = []   # Paths to saved files (for cleanup)
    
    try:
        logger.info("=" * 80)
        logger.info("NEW PREDICTION REQUEST")
        logger.info("=" * 80)
        
        # =====================================================================
        # STEP 1: VALIDATE REQUEST
        # =====================================================================
        
        logger.info(f"[Step 1/5] Validating request...")
        
        # Check if files were uploaded
        if not files:
            error_msg = "No files uploaded. Please upload at least one image."
            logger.warning(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Check number of files
        if len(files) > MAX_FILES_PER_REQUEST:
            error_msg = f"Too many files. Maximum {MAX_FILES_PER_REQUEST} files allowed, got {len(files)}"
            logger.warning(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info(f"✓ Request validation passed: {len(files)} file(s)")
        
        # =====================================================================
        # STEP 2: SAVE AND VALIDATE UPLOADED FILES
        # =====================================================================
        
        logger.info(f"\n[Step 2/5] Processing uploaded files...")
        
        for idx, file in enumerate(files, 1):
            try:
                logger.info(f"\nProcessing file {idx}/{len(files)}: {file.filename}")
                
                # Step 2a: Validate filename
                if not file.filename:
                    logger.warning(f"File {idx} has no filename, skipping")
                    continue
                
                # Step 2b: Check file extension
                if not allowed_file(file.filename):
                    error_msg = f"File '{file.filename}' has invalid extension. Allowed: {ALLOWED_EXTENSIONS}"
                    logger.warning(error_msg)
                    raise HTTPException(status_code=400, detail=error_msg)
                
                # Step 2c: Save file temporarily
                logger.debug(f"  Saving file...")
                file_path = save_upload_file(file, IMAGE_UPLOAD_DIR)
                uploaded_paths.append(file_path)
                logger.debug(f"  ✓ Saved to: {file_path}")
                
                # Step 2d: Open and validate image
                logger.debug(f"  Opening image...")
                try:
                    img = Image.open(file_path)
                    
                    # Convert to RGB if needed
                    # Some images are grayscale or RGBA, we need RGB
                    if img.mode != 'RGB':
                        logger.debug(f"  Converting from {img.mode} to RGB")
                        img = img.convert('RGB')
                    
                    logger.debug(f"  ✓ Image loaded: {img.size[0]}x{img.size[1]} pixels")
                    
                except Exception as e:
                    error_msg = f"Failed to open image '{file.filename}': {str(e)}"
                    logger.error(error_msg)
                    # Clean up the saved file
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        uploaded_paths.remove(file_path)
                    raise HTTPException(status_code=400, detail=error_msg)
                
                # Step 2e: Store image and filename
                images.append(img)
                filenames.append(file.filename)
                
                logger.info(f"✓ File {idx} processed successfully")
                
            except HTTPException:
                # Re-raise HTTP exceptions
                raise
                
            except Exception as e:
                error_msg = f"Error processing file {idx} ('{file.filename}'): {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=error_msg)
        
        # Check if we have any valid images
        if not images:
            error_msg = "No valid images to process"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info(f"\n✓ All files processed: {len(images)} valid image(s)")
        
        # =====================================================================
        # STEP 3: RUN PREDICTIONS
        # =====================================================================
        
        logger.info(f"\n[Step 3/5] Running predictions...")
        
        try:
            # Call the predict function from server.predict
            # This handles OCR, model inference, and feature extraction
            raw_results = predict(images)
            
            logger.info(f"✓ Predictions complete: {len(raw_results)} result(s)")
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Internal error during prediction: {str(e)}"
            )
        
        # =====================================================================
        # STEP 4: CLEAN AND FORMAT RESULTS
        # =====================================================================
        
        logger.info(f"\n[Step 4/5] Formatting results...")
        
        try:
            # Remove confidence scores from results
            cleaned_results = []
            
            for i, raw_result in enumerate(raw_results):
                # Remove all confidence scores and accuracy metrics
                cleaned = clean_prediction_result(raw_result)
                
                # Add original filename
                if i < len(filenames):
                    cleaned["filename"] = filenames[i]
                else:
                    # Shouldn't happen, but handle gracefully
                    cleaned["filename"] = f"unknown_{i}.jpg"
                    logger.warning(f"Missing filename for result {i}")
                
                cleaned_results.append(cleaned)
            
            logger.info("✓ Results cleaned and formatted")
            logger.info(f"  Removed all confidence scores and accuracy metrics")
            
        except Exception as e:
            error_msg = f"Error formatting results: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            # Use raw results as fallback
            cleaned_results = raw_results
        
        # =====================================================================
        # STEP 5: PREPARE RESPONSE
        # =====================================================================
        
        logger.info(f"\n[Step 5/5] Preparing response...")
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds
        
        # Create response object
        response = {
            "predictions": cleaned_results,
            "total_images": len(cleaned_results),
            "processing_time_ms": round(processing_time, 2)
        }
        
        # Log success
        logger.info("=" * 80)
        logger.info("PREDICTION SUCCESSFUL")
        logger.info("=" * 80)
        logger.info(f"Images processed: {len(cleaned_results)}")
        logger.info(f"Processing time: {processing_time:.2f} ms ({processing_time/1000:.2f} seconds)")
        logger.info(f"Average per image: {processing_time/len(cleaned_results):.2f} ms")
        logger.info("=" * 80)
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        # These are intentional errors with proper status codes
        raise
        
    except Exception as e:
        # Catch any unexpected errors
        error_msg = f"Unexpected error in prediction endpoint: {str(e)}"
        logger.error("=" * 80)
        logger.error("PREDICTION FAILED")
        logger.error("=" * 80)
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        
        # Return generic error to client (don't expose internal details)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction. Please try again."
        )
        
    finally:
        # =====================================================================
        # CLEANUP: DELETE UPLOADED FILES
        # =====================================================================
        
        # This runs whether the request succeeded or failed
        # We always clean up temporary files to prevent disk filling up
        
        logger.info("\nCleaning up temporary files...")
        
        deleted_count = 0
        failed_count = 0
        
        for file_path in uploaded_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
                    logger.debug(f"  ✓ Deleted: {file_path}")
                else:
                    logger.debug(f"  File already removed: {file_path}")
                    
            except Exception as e:
                failed_count += 1
                logger.warning(f"  Failed to delete {file_path}: {str(e)}")
        
        if deleted_count > 0:
            logger.info(f"✓ Cleanup complete: {deleted_count} file(s) deleted")
        
        if failed_count > 0:
            logger.warning(f"⚠ Failed to delete {failed_count} file(s)")


@app.get(
    "/health",
    response_model=Dict[str, str],
    summary="Health check",
    description="Check if the API server is running and responsive"
)
def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    This simple endpoint confirms the server is running.
    It's useful for:
    - Load balancers to check if server is healthy
    - Monitoring systems to detect downtime
    - Quick manual testing
    
    Returns:
        dict: Status message
    
    Example Request:
        GET http://localhost:8000/health
    
    Example Response:
        {
            "status": "ok",
            "message": "API is running",
            "timestamp": "2024-01-15T14:30:22.123456"
        }
    """
    try:
        logger.debug("Health check requested")
        
        # Get current timestamp
        timestamp = datetime.now().isoformat()
        
        # Return success response
        return {
            "status": "ok",
            "message": "API is running",
            "timestamp": timestamp
        }
        
    except Exception as e:
        # Even health check can fail in extreme cases
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom handler for HTTP exceptions.
    
    This provides consistent error responses with timestamps
    and detailed logging.
    
    Args:
        request: The request that caused the error
        exc: The HTTP exception
    
    Returns:
        JSONResponse: Formatted error response
    """
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
    """
    Handler for unexpected exceptions.
    
    This catches any unhandled errors and returns a generic
    500 error to the client while logging detailed information.
    
    Args:
        request: The request that caused the error
        exc: The exception
    
    Returns:
        JSONResponse: Generic error response
    """
    logger.error("=" * 80)
    logger.error("UNHANDLED EXCEPTION")
    logger.error("=" * 80)
    logger.error(f"Request: {request.method} {request.url}")
    logger.error(f"Error: {str(exc)}")
    logger.error(traceback.format_exc())
    logger.error("=" * 80)
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat()
        }
    )


# =============================================================================
# STARTUP AND SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Run tasks when server starts.
    
    This is called once when the server starts up.
    Good for:
    - Loading models
    - Connecting to databases
    - Initializing resources
    """
    logger.info("=" * 80)
    logger.info("SERVER STARTUP")
    logger.info("=" * 80)
    logger.info("FastAPI server is starting up...")
    logger.info(f"CORS origins: {ALLOWED_ORIGINS}")
    logger.info(f"Max file size: {MAX_FILE_SIZE // (1024*1024)} MB")
    logger.info(f"Max files per request: {MAX_FILES_PER_REQUEST}")
    logger.info(f"Allowed extensions: {ALLOWED_EXTENSIONS}")
    logger.info("=" * 80)
    logger.info("✓ Server ready to accept requests")
    logger.info("=" * 80)
    logger.info("API Documentation:")
    logger.info("  - Swagger UI: http://localhost:8000/docs")
    logger.info("  - ReDoc: http://localhost:8000/redoc")
    logger.info("=" * 80)
    logger.info("NOTE: All confidence scores and accuracy metrics are hidden from responses")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Run cleanup tasks when server shuts down.
    
    This is called when the server is stopping.
    Good for:
    - Closing database connections
    - Saving state
    - Cleaning up resources
    """
    logger.info("=" * 80)
    logger.info("SERVER SHUTDOWN")
    logger.info("=" * 80)
    logger.info("FastAPI server is shutting down...")
    
    # Clean up any remaining files in upload directory
    try:
        files = os.listdir(IMAGE_UPLOAD_DIR)
        if files:
            logger.info(f"Cleaning up {len(files)} remaining file(s)...")
            for filename in files:
                try:
                    file_path = os.path.join(IMAGE_UPLOAD_DIR, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove {filename}: {str(e)}")
            logger.info("✓ Cleanup complete")
    except Exception as e:
        logger.warning(f"Cleanup error: {str(e)}")
    
    logger.info("✓ Server shutdown complete")
    logger.info("=" * 80)


# =============================================================================
# MAIN - RUN SERVER
# =============================================================================

if __name__ == "__main__":
    """
    Run the server directly.
    
    This block runs when you execute: python server/app.py
    
    For production, use:
        uvicorn server.app:app --host 0.0.0.0 --port 8000
    """
    import uvicorn
    
    logger.info("Starting server via __main__...")
    
    try:
        # Run server with auto-reload for development
        uvicorn.run(
            "server.app:app",  # Module:app_instance
            host="0.0.0.0",    # Listen on all interfaces
            port=8000,          # Port number
            reload=True,        # Auto-reload on code changes (development only)
            log_level="info"    # Logging level
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        logger.error(traceback.format_exc())


# =============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# =============================================================================

"""
FUNCTION SUMMARY
================

1. setup_directories()
   Purpose: Create and clean necessary directories
   Use Cases:
     - Initialize server environment
     - Clean leftover files from previous runs
     - Ensure upload directory exists
   
   Called: During module import


2. allowed_file(filename)
   Purpose: Validate file extension
   Use Cases:
     - Check if uploaded file is an image
     - Prevent non-image file uploads
     - Security validation
   
   Example:
     if allowed_file("photo.jpg"):
         # Process file


3. save_upload_file(upload_file, dest_folder)
   Purpose: Save uploaded file with unique name
   Use Cases:
     - Save temporary copy of uploaded file
     - Generate unique filename to avoid conflicts
     - Validate file size
   
   Returns:
     str: Path to saved file
   
   Example:
     path = save_upload_file(file, IMAGE_UPLOAD_DIR)


4. clean_prediction_result(result)
   Purpose: Remove confidence scores from predictions
   Use Cases:
     - Clean up raw predictions
     - Hide accuracy metrics from clients
     - Simplify response format
   
   Returns:
     dict: Cleaned result without confidence scores
   
   Example:
     cleaned = clean_prediction_result(raw_result)


5. predict_endpoint(files)
   Purpose: Main prediction API endpoint
   Use Cases:
     - Process uploaded images
     - Run OCR and model inference
     - Return predictions WITHOUT confidence scores
   
   HTTP Method: POST
   URL: /predict
   
   Returns:
     PredictionResponse with all predictions (no confidence)
   
   Example Request (Python):
     import requests
     
     files = [
         ('files', open('image1.jpg', 'rb')),
         ('files', open('image2.jpg', 'rb'))
     ]
     
     response = requests.post(
         'http://localhost:8000/predict',
         files=files
     )
     
     result = response.json()
     for pred in result['predictions']:
         print(f"Sentiment: {pred['sentiment']}")
         # Note: No confidence score!


6. health_check()
   Purpose: Simple health check endpoint
   Use Cases:
     - Verify server is running
     - Load balancer health checks
     - Monitoring systems
   
   HTTP Method: GET
   URL: /health
   
   Example Request:
     curl http://localhost:8000/health


TYPICAL USAGE WORKFLOW
======================

1. Start Server:
   python server/app.py
   # Or in production:
   uvicorn server.app:app --host 0.0.0.0 --port 8000

2. Check Health:
   curl http://localhost:8000/health

3. Make Prediction:
   curl -X POST "http://localhost:8000/predict" \\
        -F "files=@image.jpg"

4. View API Docs:
   Open browser: http://localhost:8000/docs


RESPONSE FORMAT (WITHOUT CONFIDENCE)
====================================

Success Response (200):
{
    "predictions": [
        {
            "filename": "ad_image.jpg",
            
            // Primary prediction
            "predicted_label": "positive",
            
            // OCR result
            "ocr_text": "Buy Now 50% OFF",
            
            // All 9 attributes (NO CONFIDENCE SCORES)
            "theme": "sales",
            "sentiment": "positive",
            "emotion": "excited",
            "dominant_colour": "red",
            "attention_score": "high",
            "trust_safety": "safe",
            "target_audience": "adults",
            "predicted_ctr": "high",
            "likelihood_shares": "likely",
            
            // Extracted features
            "keywords": "Buy Now Discount",
            "monetary_mention": "50% OFF",
            "call_to_action": "Buy Now",
            "object_detected": "General"
        }
    ],
    "total_images": 1,
    "processing_time_ms": 1250.5
}


Error Response (400/500):
{
    "detail": "Error message",
    "timestamp": "2024-01-15T14:30:22.123456"
}


KEY DIFFERENCES FROM ORIGINAL
==============================

✓ All confidence scores REMOVED
✓ All accuracy metrics REMOVED
✓ predicted_label_num REMOVED (internal code)
✓ Cleaner, simpler response format
✓ Only predictions shown, no uncertainty metrics
✓ Response is easier to read and use
✓ All 9 attributes still predicted
✓ OCR and feature extraction still work
✓ Everything else remains the same


WHY HIDE CONFIDENCE SCORES?
============================

1. Simpler Interface:
   - Clients don't need to interpret confidence values
   - Less cognitive load for users
   - Cleaner API responses

2. Business Logic:
   - Sometimes you just need the prediction
   - Confidence can confuse non-technical users
   - Allows internal quality control without exposing it

3. Flexibility:
   - Can always add confidence back later
   - Model improvements don't change API
   - Easier to version the API


NOTE FOR DEVELOPERS
===================

If you need confidence scores for internal use:
- They are still logged in server.log
- Raw predictions still contain them
- clean_prediction_result() removes them before sending to client
- You can comment out the cleaning step for debugging
"""