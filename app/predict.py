"""
Prediction Module for Multi-Modal Classification
=================================================
This module handles the complete prediction pipeline:
1. OCR text extraction (supports EasyOCR and PaddleOCR)
2. Image preprocessing
3. Model inference
4. Feature extraction (keywords, prices, CTAs)
5. Result formatting

Features:
- Dual OCR support (EasyOCR as primary, PaddleOCR as fallback)
- All models cached locally in ./local/predict/
- Lazy model loading for fast server startup
- Batch processing for efficiency
- Comprehensive error handling
- Detailed logging

Author: [Your Name]
Date: [Date]
"""

import os
import re
import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms


# ============================================================================
# DIRECTORY SETUP - All downloads go to ./local/predict/
# ============================================================================

# Get project root directory (where this script's parent's parent is)
SCRIPT_DIR = Path(__file__).parent  # app/
PROJECT_ROOT = SCRIPT_DIR.parent     # project root

# Create local cache directories
LOCAL_DIR = PROJECT_ROOT / "local"
PREDICT_CACHE_DIR = LOCAL_DIR / "predict"

# Subdirectories for different components
EASYOCR_CACHE_DIR = PREDICT_CACHE_DIR / "easyocr_models"
PADDLEOCR_CACHE_DIR = PREDICT_CACHE_DIR / "paddleocr_models"
TORCH_CACHE_DIR = PREDICT_CACHE_DIR / "torch_cache"
# Create temp directory for OCR
TEMP_DIR = PREDICT_CACHE_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Create all directories
for dir_path in [LOCAL_DIR, PREDICT_CACHE_DIR, EASYOCR_CACHE_DIR, PADDLEOCR_CACHE_DIR, TORCH_CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set environment variables BEFORE importing libraries
# This ensures all downloads go to our local directories

# EasyOCR cache
os.environ['EASYOCR_MODULE_PATH'] = str(EASYOCR_CACHE_DIR)

# PaddleOCR/PaddleX cache
os.environ['PADDLE_HOME'] = str(PADDLEOCR_CACHE_DIR)
os.environ['PADDLEX_HOME'] = str(PADDLEOCR_CACHE_DIR)
os.environ['HUB_HOME'] = str(PADDLEOCR_CACHE_DIR)

# PyTorch cache (for any torch hub downloads)
os.environ['TORCH_HOME'] = str(TORCH_CACHE_DIR)

# HuggingFace cache (if used)
os.environ['HF_HOME'] = str(PREDICT_CACHE_DIR / "huggingface")
os.environ['TRANSFORMERS_CACHE'] = str(PREDICT_CACHE_DIR / "huggingface")


# ============================================================================
# SUPPRESS WARNINGS
# ============================================================================

import warnings
import sys

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PADDLE_LOG_LEVEL'] = '0'  # Suppress Paddle logs
os.environ['GLOG_v'] = '0'

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for noisy_logger in ['tensorflow', 'tensorboard', 'torch', 'torchvision', 
                      'paddle', 'paddlex', 'ppocr', 'matplotlib']:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)


# ============================================================================
# LOG CACHE DIRECTORIES
# ============================================================================

logger.info("=" * 80)
logger.info("PREDICTION MODULE - Cache Configuration")
logger.info("=" * 80)
logger.info(f"Project Root: {PROJECT_ROOT}")
logger.info(f"Cache Directory: {PREDICT_CACHE_DIR}")
logger.info(f"  └── EasyOCR Models: {EASYOCR_CACHE_DIR}")
logger.info(f"  └── PaddleOCR Models: {PADDLEOCR_CACHE_DIR}")
logger.info(f"  └── Torch Cache: {TORCH_CACHE_DIR}")
logger.info("=" * 80)


# ============================================================================
# OCR ENGINE DETECTION AND IMPORTS
# ============================================================================

class OCREngine(Enum):
    """Available OCR engines."""
    NONE = "none"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"


# Detect available OCR engines
EASYOCR_AVAILABLE = False
PADDLEOCR_AVAILABLE = False

# Try importing EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    logger.info("✓ EasyOCR available")
    logger.info(f"  Models will be cached in: {EASYOCR_CACHE_DIR}")
except ImportError:
    logger.info(" EasyOCR not installed (pip install easyocr)")

# Try importing PaddleOCR
try:
    from paddlex import create_pipeline
    PADDLEOCR_AVAILABLE = True
    logger.info("✓ PaddleOCR available")
    logger.info(f"  Models will be cached in: {PADDLEOCR_CACHE_DIR}")
except ImportError:
    logger.info("ℹ PaddleOCR not installed (pip install paddlex[ocr])")

# Determine default OCR engine
if EASYOCR_AVAILABLE:
    DEFAULT_OCR_ENGINE = OCREngine.EASYOCR
    logger.info("Using EasyOCR as primary OCR engine")
elif PADDLEOCR_AVAILABLE:
    DEFAULT_OCR_ENGINE = OCREngine.PADDLEOCR
    logger.info("Using PaddleOCR as primary OCR engine")
else:
    DEFAULT_OCR_ENGINE = OCREngine.NONE
    logger.warning("=" * 80)
    logger.warning("⚠ NO OCR ENGINE AVAILABLE")
    logger.warning("Install one of the following:")
    logger.warning("  pip install easyocr          (Recommended, easier)")
    logger.warning("  pip install paddlex[ocr]     (Alternative)")
    logger.warning("=" * 80)


# ============================================================================
# IMPORT PROJECT MODULES
# ============================================================================

try:
    from preprocessing.text_preprocessing import tokenize_text
    from models.fg_mfn import FG_MFN, ATTRIBUTE_NAMES
    from utils.path import SAVED_MODEL_PATH, MODEL_CONFIG, IMAGE_UPLOAD_DIR
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}")
    logger.error("Make sure you're running from the project root directory")
    raise


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Path configurations
MODEL_PATH = SAVED_MODEL_PATH
MODEL_CONFIG_PATH = MODEL_CONFIG

# Processing hyperparameters
BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)
MAX_TEXT_LEN = 128

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("=" * 80)
logger.info("PREDICTION MODULE CONFIGURATION")
logger.info("=" * 80)
logger.info(f"Device: {DEVICE}")
if DEVICE == "cuda":
    logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
logger.info(f"Batch Size: {BATCH_SIZE}")
logger.info(f"Image Size: {IMAGE_SIZE}")
logger.info(f"Model Path: {MODEL_PATH}")
logger.info(f"OCR Engine: {DEFAULT_OCR_ENGINE.value}")
logger.info("=" * 80)


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================================
# GLOBAL STATE
# ============================================================================

# Configuration
CFG: Dict[str, Any] = {}

# Model instance (lazy loaded)
model: Optional[FG_MFN] = None

# Label mappings
LABEL_MAPS: Dict[str, List[str]] = {}

# Global OCR manager
ocr_manager: Optional['OCRManager'] = None


# ============================================================================
# OCR MANAGER CLASS
# ============================================================================

class OCRManager:
    """
    Manages OCR engines with automatic fallback and local caching.
    
    All models are downloaded to:
    - EasyOCR: ./local/predict/easyocr_models/
    - PaddleOCR: ./local/predict/paddleocr_models/
    
    Supports:
    - EasyOCR (primary, easier to install)
    - PaddleOCR (fallback, more accurate for some cases)
    
    Usage:
        ocr = OCRManager()
        text, confidence = ocr.extract_text(image)
    """
    
    def __init__(
        self,
        preferred_engine: OCREngine = DEFAULT_OCR_ENGINE,
        languages: List[str] = ['en'],
        gpu: bool = None
    ):
        """
        Initialize OCR manager.
        
        Args:
            preferred_engine: Preferred OCR engine to use
            languages: List of language codes for OCR
            gpu: Use GPU if available (None = auto-detect)
        """
        self.preferred_engine = preferred_engine
        self.languages = languages
        self.use_gpu = gpu if gpu is not None else torch.cuda.is_available()
        
        self.easyocr_reader = None
        self.paddleocr_pipeline = None
        self._initialized = False
        
        # Cache directories
        self.easyocr_model_dir = EASYOCR_CACHE_DIR
        self.paddleocr_model_dir = PADDLEOCR_CACHE_DIR
        
        logger.info(f"OCR Manager created")
        logger.info(f"  Preferred engine: {preferred_engine.value}")
        logger.info(f"  Languages: {languages}")
        logger.info(f"  GPU: {self.use_gpu}")
    
    def initialize(self) -> bool:
        """
        Initialize the OCR engine(s).
        
        Returns:
            bool: True if at least one engine initialized successfully
        """
        if self._initialized:
            return True
        
        success = False
        
        # Try to initialize preferred engine first
        if self.preferred_engine == OCREngine.EASYOCR:
            if self._init_easyocr():
                success = True
            elif self._init_paddleocr():
                success = True
                logger.info("Falling back to PaddleOCR")
        
        elif self.preferred_engine == OCREngine.PADDLEOCR:
            if self._init_paddleocr():
                success = True
            elif self._init_easyocr():
                success = True
                logger.info("Falling back to EasyOCR")
        
        else:
            # Try both
            success = self._init_easyocr() or self._init_paddleocr()
        
        self._initialized = success
        
        if success:
            logger.info("✓ OCR Manager initialized successfully")
        else:
            logger.warning("⚠ OCR Manager initialization failed")
        
        return success
    
    def _init_easyocr(self) -> bool:
        """Initialize EasyOCR reader with local model storage."""
        if not EASYOCR_AVAILABLE:
            return False
        
        try:
            logger.info(f"Initializing EasyOCR...")
            logger.info(f"  Languages: {self.languages}")
            logger.info(f"  Model directory: {self.easyocr_model_dir}")
            logger.info(f"  GPU: {self.use_gpu}")
            
            # Check if models already cached
            cached_models = list(self.easyocr_model_dir.glob("*.pth")) + \
                           list(self.easyocr_model_dir.glob("*.pt")) + \
                           list(self.easyocr_model_dir.glob("*.zip"))
            
            if cached_models:
                logger.info(f"  Found {len(cached_models)} cached model file(s)")
            else:
                logger.info("  No cached models found - will download on first use")
            
            # Initialize EasyOCR with custom model storage directory
            self.easyocr_reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                model_storage_directory=str(self.easyocr_model_dir),
                download_enabled=True,  # Allow downloading
                verbose=False
            )
            
            # Check models after initialization
            cached_models = list(self.easyocr_model_dir.glob("*.pth")) + \
                           list(self.easyocr_model_dir.glob("*.pt"))
            logger.info(f"✓ EasyOCR initialized successfully")
            logger.info(f"  Cached models: {len(cached_models)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _init_paddleocr(self) -> bool:
        """Initialize PaddleOCR pipeline with local model storage."""
        if not PADDLEOCR_AVAILABLE:
            return False
        
        try:
            logger.info(f"Initializing PaddleOCR...")
            logger.info(f"  Model directory: {self.paddleocr_model_dir}")
            
            # Check if models already cached
            cached_items = list(self.paddleocr_model_dir.iterdir()) if self.paddleocr_model_dir.exists() else []
            
            if cached_items:
                logger.info(f"  Found {len(cached_items)} cached item(s)")
            else:
                logger.info("  No cached models found - will download on first use")
            
            # Initialize PaddleOCR
            # PaddleX uses PADDLEX_HOME environment variable for caching
            self.paddleocr_pipeline = create_pipeline(pipeline="ocr")
            
            logger.info("✓ PaddleOCR initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            logger.error("You may need to install: pip install paddlex[ocr]")
            return False
    
    def extract_text(
        self,
        image: Union[Image.Image, np.ndarray, str],
        use_engine: Optional[OCREngine] = None
    ) -> Tuple[str, float]:
        """
        Extract text from image using OCR.
        
        Args:
            image: PIL Image, numpy array, or path to image file
            use_engine: Specific engine to use (optional)
        
        Returns:
            tuple: (extracted_text, average_confidence)
        """
        # Initialize if needed
        if not self._initialized:
            if not self.initialize():
                logger.warning("No OCR engine available")
                return "", 0.0
        
        # Determine which engine to use
        engine = use_engine or self.preferred_engine
        
        # Try preferred engine first, then fallback
        if engine == OCREngine.EASYOCR and self.easyocr_reader is not None:
            result = self._extract_with_easyocr(image)
            if result[0]:  # If text was extracted
                return result
            # Try fallback
            if self.paddleocr_pipeline is not None:
                logger.debug("EasyOCR returned empty, trying PaddleOCR")
                return self._extract_with_paddleocr(image)
            return result
        
        elif engine == OCREngine.PADDLEOCR and self.paddleocr_pipeline is not None:
            result = self._extract_with_paddleocr(image)
            if result[0]:
                return result
            # Try fallback
            if self.easyocr_reader is not None:
                logger.debug("PaddleOCR returned empty, trying EasyOCR")
                return self._extract_with_easyocr(image)
            return result
        
        # Use whatever is available
        if self.easyocr_reader is not None:
            return self._extract_with_easyocr(image)
        elif self.paddleocr_pipeline is not None:
            return self._extract_with_paddleocr(image)
        
        logger.warning("No OCR engine initialized")
        return "", 0.0
    
    def _extract_with_easyocr(
        self,
        image: Union[Image.Image, np.ndarray, str]
    ) -> Tuple[str, float]:
        """Extract text using EasyOCR."""
        try:
            # Convert to numpy array
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            elif isinstance(image, str):
                if not os.path.exists(image):
                    logger.error(f"Image file not found: {image}")
                    return "", 0.0
                image_array = np.array(Image.open(image))
            else:
                image_array = image
            
            # Run OCR
            results = self.easyocr_reader.readtext(image_array)
            
            if not results:
                return "", 0.0
            
            # Extract text and confidences
            texts = [r[1] for r in results]
            confidences = [r[2] for r in results]
            
            # Join texts and calculate average confidence
            extracted_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            logger.debug(f"EasyOCR: '{extracted_text[:50]}...' (conf: {avg_confidence:.3f})")
            
            return extracted_text, avg_confidence
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return "", 0.0
    
    def _extract_with_paddleocr(
        self,
        image: Union[Image.Image, np.ndarray, str]
    ) -> Tuple[str, float]:
        """Extract text using PaddleOCR."""
        try:
            import tempfile
            
            # Convert to file path (PaddleOCR requires file path)
            temp_file = None
            
            if isinstance(image, Image.Image):
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=".png",
                    dir=str(PREDICT_CACHE_DIR / "temp")
                )
                # Ensure temp directory exists
                (PREDICT_CACHE_DIR / "temp").mkdir(exist_ok=True)
                image.save(temp_file.name)
                image_path = temp_file.name
                
            elif isinstance(image, str):
                if not os.path.exists(image):
                    logger.error(f"Image file not found: {image}")
                    return "", 0.0
                image_path = image
                
            else:
                # numpy array
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=".png",
                    dir=str(PREDICT_CACHE_DIR / "temp")
                )
                (PREDICT_CACHE_DIR / "temp").mkdir(exist_ok=True)
                Image.fromarray(image).save(temp_file.name)
                image_path = temp_file.name
            
            try:
                # Run OCR
                result = list(self.paddleocr_pipeline.predict(image_path))
                
                if not result or not isinstance(result[0], dict):
                    return "", 0.0
                
                # Extract texts and scores
                texts = result[0].get("rec_texts", [])
                scores = result[0].get("rec_scores", [])
                
                if not texts:
                    return "", 0.0
                
                # Join texts and calculate average confidence
                extracted_text = " ".join(texts)
                avg_confidence = sum(scores) / len(scores) if scores else 0.0
                
                logger.debug(f"PaddleOCR: '{extracted_text[:50]}...' (conf: {avg_confidence:.3f})")
                
                return extracted_text, avg_confidence
                
            finally:
                # Clean up temp file
                if temp_file is not None:
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return "", 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get OCR manager status."""
        # Get cache sizes
        easyocr_size = self._get_dir_size(self.easyocr_model_dir)
        paddleocr_size = self._get_dir_size(self.paddleocr_model_dir)
        
        return {
            "initialized": self._initialized,
            "preferred_engine": self.preferred_engine.value,
            "languages": self.languages,
            "gpu": self.use_gpu,
            "easyocr": {
                "available": EASYOCR_AVAILABLE,
                "initialized": self.easyocr_reader is not None,
                "cache_dir": str(self.easyocr_model_dir),
                "cache_size_mb": easyocr_size
            },
            "paddleocr": {
                "available": PADDLEOCR_AVAILABLE,
                "initialized": self.paddleocr_pipeline is not None,
                "cache_dir": str(self.paddleocr_model_dir),
                "cache_size_mb": paddleocr_size
            }
        }
    
    def _get_dir_size(self, path: Path) -> float:
        """Get directory size in MB."""
        try:
            total = 0
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
            return round(total / (1024 * 1024), 2)
        except:
            return 0.0
    
    def clear_cache(self, engine: Optional[str] = None, confirm: bool = False) -> bool:
        """
        Clear cached OCR models.
        
        Args:
            engine: 'easyocr', 'paddleocr', or None (both)
            confirm: Must be True to actually delete
        
        Returns:
            bool: True if cache was cleared
        """
        if not confirm:
            logger.warning("clear_cache() requires confirm=True")
            return False
        
        import shutil
        
        try:
            if engine is None or engine.lower() == 'easyocr':
                if self.easyocr_model_dir.exists():
                    shutil.rmtree(self.easyocr_model_dir)
                    self.easyocr_model_dir.mkdir(parents=True)
                    logger.info(f"✓ Cleared EasyOCR cache")
                    self.easyocr_reader = None
            
            if engine is None or engine.lower() in ('paddleocr', 'paddle'):
                if self.paddleocr_model_dir.exists():
                    shutil.rmtree(self.paddleocr_model_dir)
                    self.paddleocr_model_dir.mkdir(parents=True)
                    logger.info(f"✓ Cleared PaddleOCR cache")
                    self.paddleocr_pipeline = None
            
            self._initialized = False
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


# ============================================================================
# GLOBAL OCR MANAGER ACCESSOR
# ============================================================================

def get_ocr_manager() -> OCRManager:
    """Get or create the global OCR manager instance."""
    global ocr_manager
    
    if ocr_manager is None:
        ocr_manager = OCRManager(
            preferred_engine=DEFAULT_OCR_ENGINE,
            languages=['en']
        )
    
    return ocr_manager


# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

def _load_config() -> Dict[str, Any]:
    """Load model configuration from JSON file."""
    try:
        logger.info("Loading model configuration...")
        
        if not os.path.exists(MODEL_CONFIG_PATH):
            logger.warning(f"Config file not found: {MODEL_CONFIG_PATH}")
            return {}
        
        with open(MODEL_CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        
        logger.info("✓ Configuration loaded")
        attrs = list(cfg.get('ATTRIBUTES', {}).keys())
        logger.info(f"  Attributes: {attrs}")
        
        return cfg
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def get_label_maps() -> Dict[str, List[str]]:
    """Get label mappings for all attributes."""
    try:
        label_maps = {}
        attributes = CFG.get("ATTRIBUTES", {})
        
        for attr_name, attr_config in attributes.items():
            labels = attr_config.get("labels", [])
            label_maps[attr_name] = labels
        
        return label_maps
        
    except Exception as e:
        logger.error(f"Failed to extract label maps: {e}")
        return {}


# Load configuration and label maps
CFG = _load_config()
LABEL_MAPS = get_label_maps()


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model() -> FG_MFN:
    """
    Load the trained FG_MFN model (lazy loading).
    
    Returns:
        FG_MFN: Loaded model in evaluation mode
    """
    global model
    
    if model is not None:
        return model
    
    try:
        logger.info("=" * 80)
        logger.info("Loading FG_MFN Model")
        logger.info("=" * 80)
        
        # Create model
        loaded_model = FG_MFN(CFG).to(DEVICE)
        
        total_params = sum(p.numel() for p in loaded_model.parameters())
        trainable = sum(p.numel() for p in loaded_model.parameters() if p.requires_grad)
        logger.info(f"Model: {total_params:,} params ({trainable:,} trainable)")
        
        # Load weights
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading weights: {MODEL_PATH}")
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            loaded_model.load_state_dict(state_dict)
            logger.info("✓ Weights loaded")
        else:
            logger.warning(f"⚠ Checkpoint not found: {MODEL_PATH}")
            logger.warning("Using random weights!")
        
        loaded_model.eval()
        model = loaded_model
        
        logger.info("✓ Model ready")
        logger.info("=" * 80)
        
        return model
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.error(traceback.format_exc())
        
        # Fallback
        try:
            loaded_model = FG_MFN(CFG).to(DEVICE)
            loaded_model.eval()
            model = loaded_model
            logger.warning("✓ Fallback model created (random weights)")
            return model
        except Exception as fallback_error:
            raise RuntimeError(f"Failed to create model: {fallback_error}")


# ============================================================================
# TEXT EXTRACTION UTILITIES
# ============================================================================

def extract_keywords(text: str) -> str:
    """Extract important keywords from text."""
    if not text or not isinstance(text, str):
        return ""
    
    try:
        stopwords = {
            "the", "a", "an", "is", "are", "and", "or", "to", "for",
            "of", "in", "on", "at", "with", "your", "you", "we",
            "our", "this", "that", "it", "be", "by", "from", "as",
            "was", "were", "been", "have", "has", "had", "do", "does"
        }
        
        words = re.findall(r"\b[A-Za-z]{3,}\b", text)
        keywords = [w.capitalize() for w in words if w.lower() not in stopwords]
        
        seen = set()
        unique = []
        for w in keywords:
            if w.lower() not in seen:
                seen.add(w.lower())
                unique.append(w)
        
        return " ".join(unique[:5])
        
    except Exception as e:
        logger.error(f"Keyword extraction error: {e}")
        return ""


def extract_monetary_mention(text: str) -> str:
    """Extract price, discount, or promotional information."""
    if not text or not isinstance(text, str):
        return "None"
    
    try:
        patterns = [
            r"\d+%\s*(?:OFF|off|discount|Discount|DISCOUNT)",
            r"(?:Rs\.?|INR|USD|\$|€|£|₹)\s*\d+(?:,\d{3})*(?:\.\d{2})?",
            r"(?:FREE|Free|free)(?:\s+\w+)?",
            r"\d+(?:\.\d{2})?\s*(?:dollars?|USD|EUR|GBP)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return "None"
        
    except Exception:
        return "None"


def extract_call_to_action(text: str) -> str:
    """Extract call-to-action (CTA) phrases."""
    if not text or not isinstance(text, str):
        return "None"
    
    try:
        patterns = [
            r"(?:Buy|Shop|Order|Get|Grab|Claim|Download|Subscribe|Sign\s*Up)\s*(?:Now|Today|Here|it)?",
            r"(?:Limited\s*(?:Time\s*)?Offer|Hurry|Act\s*Now|Don't\s*Miss|Last\s*Chance)",
            r"(?:Learn|Find\s*Out|Discover)\s*More",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "None"
        
    except Exception:
        return "None"


def extract_objects_mentioned(text: str) -> str:
    """Detect product categories mentioned in text."""
    if not text or not isinstance(text, str):
        return "Unknown"
    
    try:
        categories = {
            "Phone": r"\b(phone|iphone|smartphone|mobile|android|samsung)\b",
            "Laptop": r"\b(laptop|computer|pc|notebook|macbook)\b",
            "Food": r"\b(food|burger|pizza|coffee|drink|restaurant)\b",
            "Clothing": r"\b(shirt|dress|jeans|clothes|fashion|shoes)\b",
            "Electronics": r"\b(tv|camera|headphone|speaker|tablet)\b",
            "Beauty": r"\b(makeup|cosmetic|perfume|skincare|beauty)\b",
            "Travel": r"\b(travel|flight|hotel|vacation|booking)\b",
            "Finance": r"\b(loan|credit|bank|insurance|invest)\b",
        }
        
        found = []
        text_lower = text.lower()
        
        for cat, pattern in categories.items():
            if re.search(pattern, text_lower):
                found.append(cat)
        
        return ", ".join(found) if found else "General"
        
    except Exception:
        return "Unknown"


# ============================================================================
# OCR WRAPPER FUNCTION
# ============================================================================

def extract_text(
    image: Union[Image.Image, np.ndarray, str],
    engine: Optional[str] = None
) -> Tuple[str, float]:
    """
    Extract text from image using OCR.
    
    Args:
        image: PIL Image, numpy array, or path to image file
        engine: Optional engine name ('easyocr' or 'paddleocr')
    
    Returns:
        tuple: (extracted_text, confidence_score)
    """
    try:
        ocr = get_ocr_manager()
        
        use_engine = None
        if engine:
            if engine.lower() == 'easyocr':
                use_engine = OCREngine.EASYOCR
            elif engine.lower() in ('paddleocr', 'paddle'):
                use_engine = OCREngine.PADDLEOCR
        
        return ocr.extract_text(image, use_engine=use_engine)
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return "", 0.0


# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================

def predict(
    images: List[Any],
    ocr_engine: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Predict attributes for a list of images.
    
    Args:
        images: List of images (PIL Image, path, or numpy array)
        ocr_engine: Optional OCR engine ('easyocr' or 'paddleocr')
    
    Returns:
        List of prediction dictionaries
    """
    try:
        logger.info("=" * 80)
        logger.info(f"Starting Prediction for {len(images)} Image(s)")
        logger.info("=" * 80)
        
        if not images:
            return []
        
        if not isinstance(images, list):
            images = [images]
        
        results = []
        
        # Step 1: Extract text with OCR
        logger.info("\n[Step 1/3] OCR Text Extraction...")
        
        ocr_texts = []
        ocr_confidences = []
        
        for idx, img in enumerate(images):
            try:
                text, conf = extract_text(img, engine=ocr_engine)
                ocr_texts.append(text)
                ocr_confidences.append(conf)
                
                if text:
                    preview = text[:40] + "..." if len(text) > 40 else text
                    logger.debug(f"  [{idx+1}] '{preview}' (conf: {conf:.2f})")
                    
            except Exception as e:
                logger.error(f"OCR failed for image {idx+1}: {e}")
                ocr_texts.append("")
                ocr_confidences.append(0.0)
        
        avg_conf = np.mean(ocr_confidences) if ocr_confidences else 0.0
        logger.info(f"✓ OCR complete (avg conf: {avg_conf:.2f})")
        
        # Step 2: Model Inference
        logger.info("\n[Step 2/3] Model Inference...")
        
        num_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_start in range(0, len(images), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(images))
            batch_imgs = images[batch_start:batch_end]
            batch_texts = ocr_texts[batch_start:batch_end]
            
            batch_num = (batch_start // BATCH_SIZE) + 1
            logger.info(f"  Batch {batch_num}/{num_batches} ({len(batch_imgs)} images)")
            
            try:
                # Preprocess images
                img_tensors = []
                for img in batch_imgs:
                    if not isinstance(img, Image.Image):
                        if isinstance(img, str):
                            img = Image.open(img)
                        else:
                            img = Image.fromarray(img)
                    
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img_tensors.append(transform(img))
                
                img_tensor = torch.stack(img_tensors).to(DEVICE)
                
                # Tokenize text
                tokens_list = [tokenize_text(t) for t in batch_texts]
                text_ids = torch.stack([t["input_ids"] for t in tokens_list]).to(DEVICE)
                masks = torch.stack([t["attention_mask"] for t in tokens_list]).to(DEVICE)
                
                # Run inference
                model_instance = load_model()
                
                with torch.no_grad():
                    outputs = model_instance(img_tensor, text_ids, attention_mask=masks)
                
                # Process results
                for j in range(len(batch_imgs)):
                    result = {"ocr_text": batch_texts[j]}
                    
                    primary_label = None
                    primary_idx = None
                    primary_conf = None
                    
                    for attr in ATTRIBUTE_NAMES:
                        if attr not in outputs:
                            continue
                        
                        try:
                            logits = outputs[attr][j]
                            probs = torch.softmax(logits, dim=0)
                            pred_idx = int(torch.argmax(probs))
                            conf = float(torch.max(probs))
                            
                            labels = LABEL_MAPS.get(attr, [])
                            label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
                            
                            result[attr] = label
                            result[f"{attr}_confidence"] = round(conf, 4)
                            
                            if primary_label is None:
                                primary_label = label
                                primary_idx = pred_idx
                                primary_conf = conf
                                
                        except Exception as e:
                            logger.error(f"Error processing {attr}: {e}")
                            result[attr] = "Unknown"
                            result[f"{attr}_confidence"] = 0.0
                    
                    # Legacy fields
                    result["predicted_label_text"] = primary_label or "Unknown"
                    result["predicted_label_num"] = primary_idx if primary_idx is not None else -1
                    result["confidence_score"] = round(primary_conf, 4) if primary_conf else 0.0
                    
                    # Text features
                    result["keywords"] = extract_keywords(batch_texts[j])
                    result["monetary_mention"] = extract_monetary_mention(batch_texts[j])
                    result["call_to_action"] = extract_call_to_action(batch_texts[j])
                    result["object_detected"] = extract_objects_mentioned(batch_texts[j])
                    
                    results.append(result)
                
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                logger.error(traceback.format_exc())
                
                # Add error results
                for j in range(len(batch_imgs)):
                    results.append({
                        "ocr_text": batch_texts[j] if j < len(batch_texts) else "",
                        "predicted_label_text": "Error",
                        "predicted_label_num": -1,
                        "confidence_score": 0.0,
                        "error": str(e)
                    })
        
        # Step 3: Summary
        logger.info("\n[Step 3/3] Summary")
        logger.info(f"  Processed: {len(results)} images")
        
        if results:
            avg = np.mean([r.get("confidence_score", 0) for r in results])
            logger.info(f"  Avg confidence: {avg:.2f}")
        
        logger.info("=" * 80)
        logger.info(f"✓ Prediction Complete!")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Prediction failed: {e}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_ocr_status() -> Dict[str, Any]:
    """Get current OCR engine status."""
    return get_ocr_manager().get_status()


def set_ocr_engine(engine: str) -> bool:
    """Set the preferred OCR engine."""
    global ocr_manager
    
    if engine.lower() == 'easyocr':
        if not EASYOCR_AVAILABLE:
            logger.error("EasyOCR not installed")
            return False
        ocr_manager = OCRManager(OCREngine.EASYOCR)
        return True
    
    elif engine.lower() in ('paddleocr', 'paddle'):
        if not PADDLEOCR_AVAILABLE:
            logger.error("PaddleOCR not installed")
            return False
        ocr_manager = OCRManager(OCREngine.PADDLEOCR)
        return True
    
    logger.error(f"Unknown engine: {engine}")
    return False


def get_cache_info() -> Dict[str, Any]:
    """Get information about cached models."""
    def get_size(path: Path) -> float:
        try:
            total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            return round(total / (1024 * 1024), 2)
        except:
            return 0.0
    
    def count_files(path: Path) -> int:
        try:
            return sum(1 for f in path.rglob('*') if f.is_file())
        except:
            return 0
    
    return {
        "cache_root": str(PREDICT_CACHE_DIR),
        "easyocr": {
            "path": str(EASYOCR_CACHE_DIR),
            "size_mb": get_size(EASYOCR_CACHE_DIR),
            "files": count_files(EASYOCR_CACHE_DIR)
        },
        "paddleocr": {
            "path": str(PADDLEOCR_CACHE_DIR),
            "size_mb": get_size(PADDLEOCR_CACHE_DIR),
            "files": count_files(PADDLEOCR_CACHE_DIR)
        },
        "torch": {
            "path": str(TORCH_CACHE_DIR),
            "size_mb": get_size(TORCH_CACHE_DIR),
            "files": count_files(TORCH_CACHE_DIR)
        },
        "total_size_mb": (
            get_size(EASYOCR_CACHE_DIR) + 
            get_size(PADDLEOCR_CACHE_DIR) + 
            get_size(TORCH_CACHE_DIR)
        )
    }


def clear_cache(engine: Optional[str] = None, confirm: bool = False) -> bool:
    """
    Clear cached OCR models.
    
    Args:
        engine: 'easyocr', 'paddleocr', or None (all)
        confirm: Must be True to actually delete
    
    Returns:
        bool: True if cleared successfully
    """
    return get_ocr_manager().clear_cache(engine, confirm)


# ============================================================================
# TEST AND EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PREDICTION MODULE - Test Suite")
    print("=" * 80)
    
    # Test 1: Cache Info
    print("\n[Test 1] Cache Information")
    print("-" * 80)
    cache_info = get_cache_info()
    print(f"  Cache Root: {cache_info['cache_root']}")
    print(f"  EasyOCR: {cache_info['easyocr']['size_mb']} MB ({cache_info['easyocr']['files']} files)")
    print(f"  PaddleOCR: {cache_info['paddleocr']['size_mb']} MB ({cache_info['paddleocr']['files']} files)")
    print(f"  Total: {cache_info['total_size_mb']} MB")
    
    # Test 2: OCR Status
    print("\n[Test 2] OCR Status")
    print("-" * 80)
    status = get_ocr_status()
    print(f"  Preferred Engine: {status['preferred_engine']}")
    print(f"  EasyOCR: {'✓' if status['easyocr']['available'] else '✗'} available, "
          f"{'✓' if status['easyocr']['initialized'] else '✗'} initialized")
    print(f"  PaddleOCR: {'✓' if status['paddleocr']['available'] else '✗'} available, "
          f"{'✓' if status['paddleocr']['initialized'] else '✗'} initialized")
    
    # Test 3: Prediction
    print("\n[Test 3] Prediction Test")
    print("-" * 80)
    
    try:
        dummy_img = Image.new('RGB', (224, 224), color='white')
        results = predict([dummy_img])
        
        print("  Result keys:")
        for key in sorted(results[0].keys()):
            if not key.endswith('_confidence'):
                value = results[0][key]
                if isinstance(value, str) and len(value) > 30:
                    value = value[:30] + "..."
                print(f"    {key}: {value}")
                
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 80)
    print("✓ Tests Complete")
    print("=" * 80)
    
    print("""

CACHE STRUCTURE
===============

./local/predict/
├── easyocr_models/      # EasyOCR models (~100-300 MB per language)
│   ├── craft_mlt_25k.pth
│   ├── english_g2.pth
│   └── ...
├── paddleocr_models/    # PaddleOCR models
│   └── ...
├── torch_cache/         # PyTorch hub cache
│   └── ...
└── temp/               # Temporary files (auto-cleaned)


USAGE
=====

# Basic prediction
>>> from app.predict import predict
>>> results = predict([image1, image2])

# With specific OCR engine
>>> results = predict([image], ocr_engine='easyocr')
>>> results = predict([image], ocr_engine='paddleocr')

# Direct OCR
>>> from app.predict import extract_text
>>> text, confidence = extract_text(image)

# Check status
>>> from app.predict import get_ocr_status, get_cache_info
>>> print(get_ocr_status())
>>> print(get_cache_info())

# Change engine
>>> from app.predict import set_ocr_engine
>>> set_ocr_engine('paddleocr')

# Clear cache
>>> from app.predict import clear_cache
>>> clear_cache('easyocr', confirm=True)


INSTALLATION
============

For EasyOCR (recommended):
    pip install easyocr

For PaddleOCR:
    pip install paddlex[ocr]

For both:
    pip install easyocr paddlex[ocr]

    """)