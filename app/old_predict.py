

import os
import re
import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# ============================================================================
# SUPPRESS WARNINGS AND SETUP LOGGING
# ============================================================================

import warnings



# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops

# Suppress torchvision warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Configure logging to only show ERROR level and above for specific libraries
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorboard').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('torchvision').setLevel(logging.ERROR)

# PaddleOCR for text extraction
try:
    from paddlex import create_pipeline
    PADDLEX_AVAILABLE = True
except ImportError:
    PADDLEX_AVAILABLE = False
    logging.warning("PaddleX not available. OCR functionality will be limited.")

from preprocessing.text_preprocessing import tokenize_text
from models.fg_mfn import FG_MFN, ATTRIBUTE_NAMES
from utils.path import SAVED_MODEL_PATH, MODEL_CONFIG


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Path configurations
MODEL_PATH = SAVED_MODEL_PATH           # Path to saved model weights
MODEL_CONFIG_PATH = MODEL_CONFIG         # Path to model configuration
IMAGE_UPLOAD_DIR = "data/images/tmp_uploads/"  # Temporary upload directory

# Processing hyperparameters
BATCH_SIZE = 16          # Number of images to process at once
                         # Larger = faster but needs more memory
                         # Smaller = slower but safer

IMAGE_SIZE = (224, 224)  # Input size for the model (height, width)
                         # ResNet expects 224x224 images

MAX_TEXT_LEN = 128       # Maximum text sequence length
                         # Should match training configuration

# Device configuration - use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("=" * 80)
logger.info("PREDICTION SERVER CONFIGURATION")
logger.info("=" * 80)
logger.info(f"Device: {DEVICE}")
logger.info(f"Batch Size: {BATCH_SIZE}")
logger.info(f"Image Size: {IMAGE_SIZE}")
logger.info(f"Model Path: {MODEL_PATH}")
logger.info(f"Config Path: {MODEL_CONFIG_PATH}")
logger.info("=" * 80)


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

# Image transformation pipeline
# This converts PIL images to tensors and normalizes them
# The normalization values are ImageNet statistics (standard for ResNet)
transform = transforms.Compose([
    # Step 1: Resize image to model's expected size
    transforms.Resize(IMAGE_SIZE),
    
    # Step 2: Convert PIL image to PyTorch tensor
    # This converts from (H, W, C) to (C, H, W) format
    # and scales pixel values from [0, 255] to [0, 1]
    transforms.ToTensor(),
    
    # Step 3: Normalize with ImageNet statistics
    # mean and std are per-channel (R, G, B)
    # This helps the model generalize better
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Configuration dictionary - loaded from JSON
CFG = {}

# Model instance - lazy loaded on first prediction
# This allows the server to start faster
model = None

# OCR model - lazy loaded when needed
ocr_model = None

# Label mappings - attribute names to human-readable labels
LABEL_MAPS = {}


# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================

def _load_config() -> Dict[str, Any]:
    """
    Load model configuration from JSON file.
    
    This is an internal initialization function that loads the
    model architecture configuration needed for inference.
    
    Returns:
        dict: Model configuration dictionary
    
    Note:
        This function is called automatically during module import.
        Errors are logged but don't crash the module.
    """
    try:
        logger.info("Loading model configuration...")
        
        # Check if config file exists
        if not os.path.exists(MODEL_CONFIG_PATH):
            logger.warning(f"Config file not found: {MODEL_CONFIG_PATH}")
            logger.warning("Using default empty configuration")
            return {}
        
        # Load JSON file
        with open(MODEL_CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        
        logger.info("✓ Configuration loaded successfully")
        logger.info(f"  - Attributes defined: {list(cfg.get('ATTRIBUTES', {}).keys())}")
        
        return cfg
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {str(e)}")
        logger.warning("Using default empty configuration")
        return {}
        
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        logger.error(traceback.format_exc())
        logger.warning("Using default empty configuration")
        return {}


def _initialize_ocr() -> Optional[Any]:
    """
    Initialize OCR model for text extraction.
    
    This function creates the PaddleOCR pipeline for extracting
    text from images. OCR is loaded lazily to avoid startup delays.
    
    Returns:
        OCR pipeline instance or None if unavailable
    
    Note:
        If PaddleX is not installed, this returns None and OCR
        functionality will be disabled.
    """
    global ocr_model
    
    # Return existing instance if already loaded
    if ocr_model is not None:
        return ocr_model
    
    # Check if PaddleX is available
    if not PADDLEX_AVAILABLE:
        logger.warning("PaddleX not available. OCR disabled.")
        return None
    
    try:
        logger.info("Initializing OCR model...")
        
        # Create OCR pipeline
        # This may download model weights on first run
        ocr_model = create_pipeline(pipeline="ocr")
        
        logger.info("✓ OCR model initialized successfully")
        return ocr_model
        
    except Exception as e:
        logger.error(f"Failed to initialize OCR model: {str(e)}")
        logger.error(traceback.format_exc())
        logger.warning("OCR functionality will be disabled")
        return None


# Load configuration during module import
CFG = _load_config()


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model() -> FG_MFN:
    """
    Load the trained FG_MFN model (lazy loading).
    
    This function loads the model only when first needed, not during
    module import. This makes the server start faster. Subsequent calls
    return the cached model instance.
    
    Returns:
        FG_MFN: Loaded model in evaluation mode
    
    Raises:
        RuntimeError: If model loading fails critically
    
    Example:
        >>> model = load_model()
        >>> # First call loads model
        >>> model = load_model()
        >>> # Second call returns cached instance
    
    Note:
        The model is automatically set to evaluation mode (model.eval())
        which disables dropout and batch normalization training behavior.
    """
    global model
    
    # Step 1: Return cached model if already loaded
    if model is not None:
        logger.debug("Using cached model instance")
        return model
    
    try:
        logger.info("=" * 80)
        logger.info("Loading FG_MFN Model")
        logger.info("=" * 80)
        
        # Step 2: Validate configuration
        if not CFG:
            logger.warning("Empty configuration. Model may not work correctly.")
        
        # Step 3: Create model architecture
        logger.info("Creating model architecture...")
        
        try:
            loaded_model = FG_MFN(CFG).to(DEVICE)
            logger.info("✓ Model architecture created")
            
            # Log model details
            total_params = sum(p.numel() for p in loaded_model.parameters())
            logger.info(f"  - Total parameters: {total_params:,}")
            logger.info(f"  - Attribute heads: {list(loaded_model.attribute_heads.keys())}")
            
        except Exception as e:
            error_msg = f"Failed to create model architecture: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
        
        # Step 4: Load trained weights
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading weights from: {MODEL_PATH}")
            
            try:
                # Load state dictionary
                # map_location ensures weights are loaded to correct device
                state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                loaded_model.load_state_dict(state_dict)
                logger.info("✓ Weights loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load weights: {str(e)}")
                logger.error(traceback.format_exc())
                logger.warning("⚠ Using randomly initialized weights!")
                logger.warning("Model predictions will be meaningless!")
        else:
            logger.warning(f"Model checkpoint not found: {MODEL_PATH}")
            logger.warning("⚠ Using randomly initialized weights!")
            logger.warning("Model predictions will be meaningless!")
        
        # Step 5: Set model to evaluation mode
        # This disables dropout and sets batch norm to use running stats
        loaded_model.eval()
        logger.info("✓ Model set to evaluation mode")
        
        # Step 6: Cache the model for future use
        model = loaded_model
        
        logger.info("=" * 80)
        logger.info("Model Loading Complete")
        logger.info("=" * 80)
        
        return model
        
    except Exception as e:
        # If loading fails, try to return a model with random weights
        # This prevents the server from crashing completely
        logger.error(f"Critical error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        logger.warning("Attempting to create model with random weights as fallback...")
        
        try:
            loaded_model = FG_MFN(CFG).to(DEVICE)
            loaded_model.eval()
            model = loaded_model
            logger.warning("✓ Fallback model created (with random weights)")
            return model
            
        except Exception as fallback_error:
            error_msg = f"Failed to create fallback model: {str(fallback_error)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)


def get_label_maps() -> Dict[str, List[str]]:
    """
    Get label mappings for all attributes.
    
    This function extracts human-readable label names from the
    configuration. These are used to convert predicted class indices
    to meaningful text labels.
    
    Returns:
        dict: Mapping from attribute names to lists of label names
              Format: {
                  'sentiment': ['negative', 'neutral', 'positive'],
                  'emotion': ['happy', 'sad', 'angry', 'surprised'],
                  ...
              }
    
    Example:
        >>> label_maps = get_label_maps()
        >>> print(label_maps['sentiment'])
        ['negative', 'neutral', 'positive']
        
        >>> # Convert predicted index to label
        >>> predicted_index = 2
        >>> label = label_maps['sentiment'][predicted_index]
        >>> print(label)  # 'positive'
    
    Note:
        If an attribute doesn't have labels defined in the config,
        it will have an empty list in the returned dictionary.
    """
    try:
        logger.debug("Extracting label mappings from configuration...")
        
        # Initialize empty dictionary
        label_maps = {}
        
        # Get attributes from configuration
        # ATTRIBUTES is a dict: {attr_name: {num_classes: X, labels: [...]}}
        attributes = CFG.get("ATTRIBUTES", {})
        
        if not attributes:
            logger.warning("No attributes defined in configuration")
            logger.warning("Label mapping will be empty")
            return label_maps
        
        # Extract labels for each attribute
        for attr_name, attr_config in attributes.items():
            # Get the labels list, default to empty list if not found
            labels = attr_config.get("labels", [])
            label_maps[attr_name] = labels
            
            logger.debug(f"  {attr_name}: {len(labels)} labels")
        
        logger.debug(f"✓ Extracted label maps for {len(label_maps)} attributes")
        return label_maps
        
    except Exception as e:
        logger.error(f"Failed to extract label maps: {str(e)}")
        logger.error(traceback.format_exc())
        logger.warning("Returning empty label maps")
        return {}


# Initialize label maps during module import
LABEL_MAPS = get_label_maps()


# =============================================================================
# TEXT EXTRACTION UTILITIES
# =============================================================================

def extract_keywords(text: str) -> str:
    """
    Extract important keywords from text.
    
    This function identifies and extracts meaningful keywords from
    the input text by:
    1. Removing common stopwords (the, a, is, etc.)
    2. Extracting words 3+ characters long
    3. Removing duplicates while preserving order
    4. Capitalizing for consistency
    5. Limiting to top 5 keywords
    
    Args:
        text (str): Input text to extract keywords from
    
    Returns:
        str: Space-separated keywords (up to 5)
             Returns empty string if no keywords found
    
    Example:
        >>> text = "Buy the best smartphone at amazing discount today"
        >>> keywords = extract_keywords(text)
        >>> print(keywords)
        'Buy Best Smartphone Amazing Discount'
        
        >>> text = "The a is and"  # All stopwords
        >>> keywords = extract_keywords(text)
        >>> print(keywords)
        ''
    
    Note:
        This is a simple keyword extraction method. For production,
        consider using more sophisticated NLP techniques like TF-IDF
        or named entity recognition (NER).
    """
    try:
        # Step 1: Validate input
        if not text:
            logger.debug("Empty text provided for keyword extraction")
            return ""
        
        if not isinstance(text, str):
            logger.warning(f"Expected string, got {type(text)}. Converting...")
            text = str(text)
        
        # Step 2: Define stopwords
        # These are common words that don't carry much meaning
        stopwords = {
            "the", "a", "an", "is", "are", "and", "or", "to", "for",
            "of", "in", "on", "at", "with", "your", "you", "we",
            "our", "this", "that", "it", "be", "by", "from", "as",
            "was", "were", "been", "have", "has", "had", "do", "does"
        }
        
        # Step 3: Extract words using regex
        # \b = word boundary
        # [A-Za-z]{3,} = alphabetic characters, 3 or more
        words = re.findall(r"\b[A-Za-z]{3,}\b", text)
        
        logger.debug(f"Extracted {len(words)} words from text")
        
        # Step 4: Filter out stopwords and capitalize
        # Capitalize makes keywords look more professional
        keywords = [
            word.capitalize() 
            for word in words 
            if word.lower() not in stopwords
        ]
        
        # Step 5: Remove duplicates while preserving order
        # Using a set to track what we've seen
        seen = set()
        unique_keywords = []
        
        for word in keywords:
            # Use lowercase for comparison to catch duplicates
            # like "Phone" and "phone"
            word_lower = word.lower()
            
            if word_lower not in seen:
                seen.add(word_lower)
                unique_keywords.append(word)
        
        # Step 6: Limit to top 5 keywords
        # First 5 are usually the most relevant
        top_keywords = unique_keywords[:5]
        
        # Step 7: Join with spaces
        result = " ".join(top_keywords)
        
        logger.debug(f"Extracted keywords: '{result}'")
        return result
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        logger.error(traceback.format_exc())
        return ""


def extract_monetary_mention(text: str) -> str:
    """
    Extract price, discount, or promotional information from text.
    
    This function searches for monetary mentions like:
    - Discounts: "50% OFF", "30% discount"
    - Prices: "$99.99", "Rs. 1,999", "₹500"
    - Free offers: "FREE shipping"
    
    Args:
        text (str): Input text to search
    
    Returns:
        str: First monetary mention found, or "None" if nothing found
    
    Example:
        >>> text = "Get 50% OFF on iPhone today!"
        >>> mention = extract_monetary_mention(text)
        >>> print(mention)
        '50% OFF'
        
        >>> text = "Price: $99.99"
        >>> mention = extract_monetary_mention(text)
        >>> print(mention)
        '$99.99'
        
        >>> text = "No price mentioned"
        >>> mention = extract_monetary_mention(text)
        >>> print(mention)
        'None'
    
    Note:
        Patterns are checked in order. First match wins.
        Add more patterns as needed for different formats.
    """
    try:
        # Step 1: Validate input
        if not text:
            logger.debug("Empty text for monetary extraction")
            return "None"
        
        if not isinstance(text, str):
            logger.warning(f"Expected string, got {type(text)}. Converting...")
            text = str(text)
        
        # Step 2: Define regex patterns for different monetary mentions
        # Order matters - more specific patterns first
        patterns = [
            # Pattern 1: Percentage discounts
            # Matches: "50% OFF", "30% discount", "20% off"
            r"\d+%\s*(?:OFF|off|discount|Discount|DISCOUNT)",
            
            # Pattern 2: Currency amounts
            # Matches: "Rs. 1,999", "$99.99", "₹500", "INR 1000"
            r"(?:Rs\.?|INR|USD|\$|₹)\s*\d+(?:,\d{3})*(?:\.\d{2})?",
            
            # Pattern 3: Free offers
            # Matches: "FREE", "Free shipping"
            r"(?:FREE|Free|free)(?:\s+\w+)?",
        ]
        
        # Step 3: Try each pattern
        for pattern in patterns:
            match = re.search(pattern, text)
            
            if match:
                # Found a match!
                result = match.group(0)
                logger.debug(f"Found monetary mention: '{result}'")
                return result
        
        # Step 4: No matches found
        logger.debug("No monetary mention found in text")
        return "None"
        
    except Exception as e:
        logger.error(f"Error extracting monetary mention: {str(e)}")
        logger.error(traceback.format_exc())
        return "None"


def extract_call_to_action(text: str) -> str:
    """
    Extract call-to-action (CTA) phrases from text.
    
    CTAs are phrases that encourage the user to take action:
    - "Buy Now", "Shop Today", "Order Now"
    - "Limited Offer", "Hurry", "Act Now"
    
    Args:
        text (str): Input text to search
    
    Returns:
        str: First CTA found, or "None" if nothing found
    
    Example:
        >>> text = "Buy Now and save 50%!"
        >>> cta = extract_call_to_action(text)
        >>> print(cta)
        'Buy Now'
        
        >>> text = "Limited Offer - Shop Today"
        >>> cta = extract_call_to_action(text)
        >>> print(cta)
        'Limited Offer'
        
        >>> text = "Just an advertisement"
        >>> cta = extract_call_to_action(text)
        >>> print(cta)
        'None'
    
    Note:
        Case-insensitive matching is used to catch variations
        like "buy now", "Buy Now", "BUY NOW".
    """
    try:
        # Step 1: Validate input
        if not text:
            logger.debug("Empty text for CTA extraction")
            return "None"
        
        if not isinstance(text, str):
            logger.warning(f"Expected string, got {type(text)}. Converting...")
            text = str(text)
        
        # Step 2: Define CTA patterns
        patterns = [
            # Pattern 1: Action verbs with optional time modifier
            # Matches: "Buy Now", "Shop Today", "Order", "Get it"
            r"(?:Buy|Shop|Order|Get|Grab|Claim)\s*(?:Now|Today|it)?",
            
            # Pattern 2: Urgency phrases
            # Matches: "Limited Offer", "Hurry Up", "Act Now"
            r"(?:Limited\s*Offer|Hurry|Act\s*Now|Don't\s*Miss)",
        ]
        
        # Step 3: Try each pattern (case-insensitive)
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                # Found a CTA!
                result = match.group(0)
                logger.debug(f"Found CTA: '{result}'")
                return result
        
        # Step 4: No CTA found
        logger.debug("No call-to-action found in text")
        return "None"
        
    except Exception as e:
        logger.error(f"Error extracting CTA: {str(e)}")
        logger.error(traceback.format_exc())
        return "None"


def extract_objects_mentioned(text: str) -> str:
    """
    Detect product categories or objects mentioned in text.
    
    This function identifies what products or items are mentioned:
    - Electronics: phone, laptop, computer
    - Food & Beverage: burger, coffee, drink
    - Clothing: shirt, dress, jeans
    
    Args:
        text (str): Input text to analyze
    
    Returns:
        str: Comma-separated list of detected categories
             Returns "General" if no specific category detected
             Returns "Unknown" if text is empty
    
    Example:
        >>> text = "Buy the latest iPhone and laptop"
        >>> objects = extract_objects_mentioned(text)
        >>> print(objects)
        'Phone, Laptop'
        
        >>> text = "Get your burger and coffee here"
        >>> objects = extract_objects_mentioned(text)
        >>> print(objects)
        'Food'
        
        >>> text = "Amazing product available"
        >>> objects = extract_objects_mentioned(text)
        >>> print(objects)
        'General'
    
    Note:
        This uses a simple keyword matching approach. For better
        results, consider using named entity recognition (NER) or
        a trained product classifier.
    """
    try:
        # Step 1: Validate input
        if not text:
            logger.debug("Empty text for object detection")
            return "Unknown"
        
        if not isinstance(text, str):
            logger.warning(f"Expected string, got {type(text)}. Converting...")
            text = str(text)
        
        # Step 2: Define category mappings
        # Each category has a regex pattern for keywords
        category_mapping = {
            "Phone": r"\b(phone|iphone|smartphone|mobile|android)\b",
            "Laptop": r"\b(laptop|computer|pc|notebook|macbook)\b",
            "Food": r"\b(food|burger|pizza|sandwich|meal|coffee|drink|beverage)\b",
            "Clothing": r"\b(shirt|dress|jeans|pants|clothes|apparel|fashion)\b",
            "Electronics": r"\b(tv|television|camera|headphone|speaker)\b",
            "Beauty": r"\b(makeup|cosmetic|perfume|skincare|beauty)\b",
        }
        
        # Step 3: Search for each category
        found_categories = []
        text_lower = text.lower()  # Convert once for efficiency
        
        for category, pattern in category_mapping.items():
            if re.search(pattern, text_lower):
                found_categories.append(category)
                logger.debug(f"Detected category: {category}")
        
        # Step 4: Return results
        if found_categories:
            # Found specific categories
            result = ", ".join(found_categories)
            logger.debug(f"Detected objects: {result}")
            return result
        else:
            # No specific category detected
            logger.debug("No specific category detected, using 'General'")
            return "General"
        
    except Exception as e:
        logger.error(f"Error detecting objects: {str(e)}")
        logger.error(traceback.format_exc())
        return "Unknown"


# =============================================================================
# OCR TEXT EXTRACTION
# =============================================================================

def extract_text(image: Any) -> Tuple[str, float]:
    """
    Extract text from image using OCR.
    
    This function uses PaddleOCR to detect and recognize text in images.
    It returns both the extracted text and a confidence score.
    
    Args:
        image: Input image, can be:
               - PIL.Image.Image object
               - str (file path)
               - np.ndarray (image array)
    
    Returns:
        tuple: (extracted_text, confidence_score)
               - extracted_text (str): All detected text joined by spaces
               - confidence_score (float): Average confidence (0.0 to 1.0)
               
               Returns ("", 0.0) if OCR fails or no text detected
    
    Example:
        >>> from PIL import Image
        >>> img = Image.open("ad_image.png")
        >>> text, confidence = extract_text(img)
        >>> print(f"Text: {text}")
        Text: Buy Now 50% OFF
        >>> print(f"Confidence: {confidence:.2f}")
        Confidence: 0.95
    
    Note:
        - If PaddleX is not installed, returns empty text
        - Temporary files are created for PIL images
        - Higher confidence = more reliable OCR results
    """
    try:
        # Step 1: Check if OCR is available
        ocr = _initialize_ocr()
        
        if ocr is None:
            logger.warning("OCR not available. Returning empty text.")
            return "", 0.0
        
        # Step 2: Handle different image input types
        image_path = None
        temp_file = None
        
        if isinstance(image, Image.Image):
            # PIL Image - save to temporary file
            import tempfile
            
            logger.debug("Converting PIL Image to temporary file...")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            image.save(temp_file.name)
            image_path = temp_file.name
            logger.debug(f"Saved to: {image_path}")
            
        elif isinstance(image, str):
            # File path
            if not os.path.exists(image):
                logger.error(f"Image file not found: {image}")
                return "", 0.0
            image_path = image
            logger.debug(f"Using image path: {image_path}")
            
        else:
            # Assume it's a numpy array or similar
            import tempfile
            import cv2
            
            logger.debug("Converting array to temporary file...")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(temp_file.name, image)
            image_path = temp_file.name
            logger.debug(f"Saved to: {image_path}")
        
        # Step 3: Run OCR
        logger.debug(f"Running OCR on: {image_path}")
        
        try:
            # Predict returns a generator, convert to list
            result = list(ocr.predict(image_path))
            
        except Exception as e:
            logger.error(f"OCR prediction failed: {str(e)}")
            logger.error(traceback.format_exc())
            return "", 0.0
        
        finally:
            # Clean up temporary file if created
            if temp_file is not None:
                try:
                    os.unlink(temp_file.name)
                    logger.debug("Cleaned up temporary file")
                except:
                    pass
        
        # Step 4: Extract text and scores from result
        texts = []
        scores = []
        
        if result and isinstance(result[0], dict):
            # Result format: [{'rec_texts': [...], 'rec_scores': [...]}]
            texts = result[0].get("rec_texts", [])
            scores = result[0].get("rec_scores", [])
            
            logger.debug(f"OCR found {len(texts)} text regions")
        else:
            logger.debug("OCR returned no text")
        
        # Step 5: Calculate average confidence
        avg_score = 0.0
        if scores:
            avg_score = sum(scores) / len(scores)
            logger.debug(f"Average OCR confidence: {avg_score:.3f}")
        
        # Step 6: Join all text with spaces
        extracted_text = " ".join(texts).strip()
        
        logger.debug(f"Extracted text: '{extracted_text}'")
        
        return extracted_text, avg_score
        
    except Exception as e:
        logger.error(f"Error in OCR text extraction: {str(e)}")
        logger.error(traceback.format_exc())
        return "", 0.0


# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

def predict(images: List[Any]) -> List[Dict[str, Any]]:
    """
    Predict attributes for a list of images.
    
    This is the main inference function. It:
    1. Extracts text from images using OCR
    2. Processes images and text through the model
    3. Predicts multiple attributes (sentiment, emotion, etc.)
    4. Extracts additional features (keywords, prices, CTAs)
    5. Returns comprehensive results
    
    Args:
        images (list): List of images to process
                      Each image can be:
                      - PIL.Image.Image object
                      - str (file path)
                      - np.ndarray (image array)
    
    Returns:
        list: List of prediction dictionaries, one per image
              Each dictionary contains:
              {
                  # OCR results
                  'ocr_text': str,
                  
                  # Per-attribute predictions
                  'sentiment': str,
                  'sentiment_confidence': float,
                  'emotion': str,
                  'emotion_confidence': float,
                  # ... (for all attributes)
                  
                  # Legacy fields (for backward compatibility)
                  'predicted_label_text': str,
                  'predicted_label_num': int,
                  'confidence_score': float,
                  
                  # Extracted features
                  'keywords': str,
                  'monetary_mention': str,
                  'call_to_action': str,
                  'object_detected': str
              }
    
    Example:
        >>> from PIL import Image
        >>> images = [Image.open("ad1.png"), Image.open("ad2.png")]
        >>> results = predict(images)
        >>> for i, result in enumerate(results):
        ...     print(f"Image {i+1}:")
        ...     print(f"  Sentiment: {result['sentiment']}")
        ...     print(f"  OCR Text: {result['ocr_text']}")
    
    Raises:
        RuntimeError: If prediction fails completely
    
    Note:
        - Images are processed in batches for efficiency
        - OCR is run on all images first
        - Results maintain input order
        - Empty images or OCR failures are handled gracefully
    """
    try:
        logger.info("=" * 80)
        logger.info(f"Starting Prediction for {len(images)} Image(s)")
        logger.info("=" * 80)
        
        # Step 1: Validate input
        if not images:
            logger.warning("Empty image list provided")
            return []
        
        if not isinstance(images, list):
            logger.warning(f"Expected list, got {type(images)}. Converting...")
            images = [images]
        
        logger.info(f"Processing {len(images)} images in batches of {BATCH_SIZE}")
        
        # Step 2: Initialize results storage
        results = []
        
        # Step 3: Extract text from all images using OCR
        logger.info("\n[Step 1/4] Extracting text with OCR...")
        ocr_texts = []
        ocr_confidences = []
        
        for idx, img in enumerate(images):
            try:
                text, confidence = extract_text(img)
                ocr_texts.append(text)
                ocr_confidences.append(confidence)
                
                logger.debug(f"  Image {idx+1}: '{text[:50]}...' (conf: {confidence:.3f})")
                
            except Exception as e:
                logger.error(f"OCR failed for image {idx+1}: {str(e)}")
                ocr_texts.append("")
                ocr_confidences.append(0.0)
        
        logger.info(f"✓ OCR complete. Average confidence: {np.mean(ocr_confidences):.3f}")
        
        # Step 4: Process images in batches
        logger.info("\n[Step 2/4] Running model inference...")
        
        num_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"Processing {num_batches} batch(es)...")
        
        for batch_idx in range(0, len(images), BATCH_SIZE):
            try:
                # Step 4a: Get current batch
                batch_end = min(batch_idx + BATCH_SIZE, len(images))
                batch_imgs = images[batch_idx:batch_end]
                batch_texts = ocr_texts[batch_idx:batch_end]
                
                batch_num = (batch_idx // BATCH_SIZE) + 1
                logger.info(f"\nProcessing batch {batch_num}/{num_batches} ({len(batch_imgs)} images)...")
                
                # Step 4b: Preprocess images
                logger.debug("  Preprocessing images...")
                try:
                    # Apply transforms to each image
                    img_tensors = []
                    for img in batch_imgs:
                        # Ensure image is PIL Image
                        if not isinstance(img, Image.Image):
                            if isinstance(img, str):
                                img = Image.open(img)
                            else:
                                # Assume numpy array
                                img = Image.fromarray(img)
                        
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Apply transformation
                        img_tensor = transform(img)
                        img_tensors.append(img_tensor)
                    
                    # Stack into batch tensor
                    img_tensor = torch.stack(img_tensors).to(DEVICE)
                    logger.debug(f"  Image tensor shape: {img_tensor.shape}")
                    
                except Exception as e:
                    error_msg = f"Image preprocessing failed: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    # Skip this batch
                    continue
                
                # Step 4c: Tokenize text
                logger.debug("  Tokenizing text...")
                try:
                    tokens_list = [tokenize_text(t) for t in batch_texts]
                    
                    # Stack tensors
                    text_ids = torch.stack([t["input_ids"] for t in tokens_list]).to(DEVICE)
                    masks = torch.stack([t["attention_mask"] for t in tokens_list]).to(DEVICE)
                    
                    logger.debug(f"  Text tensor shape: {text_ids.shape}")
                    
                except Exception as e:
                    error_msg = f"Text tokenization failed: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    # Skip this batch
                    continue
                
                # Step 4d: Run model inference
                logger.debug("  Running model forward pass...")
                
                try:
                    # Load model (lazy loading)
                    model_instance = load_model()
                    
                    # Disable gradient computation (inference only)
                    with torch.no_grad():
                        outputs = model_instance(
                            img_tensor,
                            text_ids,
                            attention_mask=masks
                        )
                    
                    logger.debug(f"  Got predictions for {len(outputs)} attributes")
                    
                except Exception as e:
                    error_msg = f"Model inference failed: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    # Skip this batch
                    continue
                
                # Step 4e: Process each image in batch
                logger.debug("  Processing predictions...")
                
                for j in range(len(batch_imgs)):
                    try:
                        # Initialize result dictionary for this image
                        result = {
                            "ocr_text": batch_texts[j]
                        }
                        
                        # Track primary prediction for legacy compatibility
                        primary_label = None
                        primary_idx = None
                        primary_conf = None
                        
                        # Process each attribute
                        for attr in ATTRIBUTE_NAMES:
                            # Check if this attribute has predictions
                            if attr not in outputs:
                                logger.debug(f"    Skipping {attr} - not in outputs")
                                continue
                            
                            try:
                                # Get logits for this sample
                                logits = outputs[attr][j]
                                
                                # Convert to probabilities
                                probs = torch.softmax(logits, dim=0)
                                
                                # Get predicted class
                                pred_idx = int(torch.argmax(probs))
                                confidence = float(torch.max(probs))
                                
                                # Convert index to label name
                                labels = LABEL_MAPS.get(attr, [])
                                if pred_idx < len(labels):
                                    label = labels[pred_idx]
                                else:
                                    # Fallback to numeric label
                                    label = str(pred_idx)
                                    logger.warning(f"    Label index {pred_idx} out of range for {attr}")
                                
                                # Store in result
                                result[attr] = label
                                result[f"{attr}_confidence"] = confidence
                                
                                logger.debug(f"    {attr}: {label} (conf: {confidence:.3f})")
                                
                                # Track first prediction as primary
                                if primary_label is None:
                                    primary_label = label
                                    primary_idx = pred_idx
                                    primary_conf = confidence
                                
                            except Exception as e:
                                logger.error(f"    Error processing {attr}: {str(e)}")
                                result[attr] = "Unknown"
                                result[f"{attr}_confidence"] = 0.0
                        
                        # Step 4f: Add legacy fields for backward compatibility
                        result["predicted_label_text"] = primary_label or "Unknown"
                        result["predicted_label_num"] = primary_idx if primary_idx is not None else -1
                        result["confidence_score"] = primary_conf if primary_conf else 0.0
                        
                        # Step 4g: Extract additional text features
                        logger.debug("  Extracting text features...")
                        
                        result["keywords"] = extract_keywords(batch_texts[j])
                        result["monetary_mention"] = extract_monetary_mention(batch_texts[j])
                        result["call_to_action"] = extract_call_to_action(batch_texts[j])
                        result["object_detected"] = extract_objects_mentioned(batch_texts[j])
                        
                        # Add result to list
                        results.append(result)
                        
                    except Exception as e:
                        logger.error(f"  Error processing image {j} in batch: {str(e)}")
                        logger.error(traceback.format_exc())
                        
                        # Add default result to maintain order
                        results.append({
                            "ocr_text": batch_texts[j] if j < len(batch_texts) else "",
                            "predicted_label_text": "Error",
                            "predicted_label_num": -1,
                            "confidence_score": 0.0,
                            "keywords": "",
                            "monetary_mention": "None",
                            "call_to_action": "None",
                            "object_detected": "Unknown"
                        })
                
                logger.info(f"✓ Batch {batch_num} complete")
                
            except Exception as e:
                logger.error(f"Batch {batch_idx // BATCH_SIZE + 1} failed: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue with next batch
                continue
        
        # Step 5: Validate results
        if len(results) != len(images):
            logger.warning(f"Result count mismatch: {len(results)} results for {len(images)} images")
        
        # Step 6: Log summary
        logger.info("\n[Step 3/4] Prediction Summary")
        logger.info("-" * 80)
        logger.info(f"Total images processed: {len(results)}")
        
        if results:
            # Calculate average confidence
            avg_conf = np.mean([r.get("confidence_score", 0.0) for r in results])
            logger.info(f"Average confidence: {avg_conf:.3f}")
            
            # Count predictions per attribute
            for attr in ATTRIBUTE_NAMES:
                count = sum(1 for r in results if attr in r)
                if count > 0:
                    logger.info(f"  {attr}: {count} predictions")
        
        logger.info("=" * 80)
        logger.info(f"Prediction Complete! Returning {len(results)} Results")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg)


# =============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# =============================================================================

if __name__ == "__main__":
    """
    Example usage and test cases for the prediction server.
    """
    
    print("=" * 80)
    print("PREDICTION SERVER - Usage Examples")
    print("=" * 80)
    
    # Example 1: Single Image Prediction
    print("\n[Example 1] Single Image Prediction")
    print("-" * 80)
    
    try:
        from PIL import Image
        import numpy as np
        
        # Create a dummy image
        dummy_img = Image.new('RGB', (224, 224), color='red')
        
        # Run prediction
        results = predict([dummy_img])
        
        print(f"Prediction result:")
        for key, value in results[0].items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Example 2: Batch Prediction
    print("\n[Example 2] Batch Prediction")
    print("-" * 80)
    
    try:
        # Create multiple dummy images
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='blue'),
            Image.new('RGB', (224, 224), color='green')
        ]
        
        # Run batch prediction
        results = predict(images)
        
        print(f"Processed {len(results)} images")
        for i, result in enumerate(results):
            print(f"\nImage {i+1}:")
            print(f"  Primary prediction: {result['predicted_label_text']}")
            print(f"  Confidence: {result['confidence_score']:.3f}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "=" * 80)
    print("FUNCTION SUMMARY")
    print("=" * 80)
    
    print("""
    1. load_model()
       Purpose: Lazy load the trained model
       Use Cases:
         - First prediction call loads model
         - Subsequent calls use cached instance
         - Allows fast server startup
       
       Example:
         model = load_model()
    
    
    2. get_label_maps()
       Purpose: Extract label names from configuration
       Use Cases:
         - Convert predicted indices to text labels
         - Map class numbers to human-readable names
       
       Example:
         labels = get_label_maps()
         sentiment_label = labels['sentiment'][predicted_idx]
    
    
    3. extract_keywords(text)
       Purpose: Extract important keywords from text
       Use Cases:
         - Identify main topics in OCR text
         - Generate text summaries
         - Feature extraction for analysis
       
       Example:
         text = "Buy the best smartphone today"
         keywords = extract_keywords(text)
         # Returns: "Buy Best Smartphone Today"
    
    
    4. extract_monetary_mention(text)
       Purpose: Find price/discount information
       Use Cases:
         - Detect promotional offers
         - Extract pricing information
         - Identify discount mentions
       
       Example:
         text = "Get 50% OFF on all items"
         mention = extract_monetary_mention(text)
         # Returns: "50% OFF"
    
    
    5. extract_call_to_action(text)
       Purpose: Identify CTA phrases
       Use Cases:
         - Detect action-oriented text
         - Identify urgency markers
         - Classify ad intent
       
       Example:
         text = "Buy Now before stock runs out"
         cta = extract_call_to_action(text)
         # Returns: "Buy Now"
    
    
    6. extract_objects_mentioned(text)
       Purpose: Detect product categories
       Use Cases:
         - Classify product types
         - Identify advertised items
         - Categorize content
       
       Example:
         text = "Latest iPhone and MacBook deals"
         objects = extract_objects_mentioned(text)
         # Returns: "Phone, Laptop"
    
    
    7. extract_text(image)
       Purpose: OCR text extraction from images
       Use Cases:
         - Read text from advertisement images
         - Extract information from graphics
         - Digitize image content
       
       Returns:
         tuple: (text, confidence)
       
       Example:
         img = Image.open("ad.png")
         text, conf = extract_text(img)
    
    
    8. predict(images)
       Purpose: Main prediction function
       Use Cases:
         - Predict multiple attributes from images
         - Batch process multiple images
         - Generate comprehensive results
       
       Returns:
         list: List of result dictionaries
       
       Example:
         images = [img1, img2, img3]
         results = predict(images)
         for result in results:
             print(result['sentiment'])
    
    
    TYPICAL API WORKFLOW:
    ====================
    
    1. Receive image(s) from client
    2. Call predict(images)
    3. Get results with all predictions
    4. Return JSON response to client
    
    Example Flask endpoint:
    
    @app.route('/predict', methods=['POST'])
    def predict_endpoint():
        # Get uploaded images
        files = request.files.getlist('images')
        images = [Image.open(f) for f in files]
        
        # Run prediction
        results = predict(images)
        
        # Return JSON
        return jsonify(results)
    
    
    RESULT FORMAT:
    ==============
    
    Each prediction result contains:
    
    {
        // OCR Results
        "ocr_text": "Buy Now 50% OFF",
        
        // Multi-Attribute Predictions
        "sentiment": "positive",
        "sentiment_confidence": 0.95,
        "emotion": "excited",
        "emotion_confidence": 0.87,
        "theme": "sales",
        "theme_confidence": 0.92,
        
        // Legacy Fields (for backward compatibility)
        "predicted_label_text": "positive",
        "predicted_label_num": 2,
        "confidence_score": 0.95,
        
        // Extracted Features
        "keywords": "Buy Now Discount",
        "monetary_mention": "50% OFF",
        "call_to_action": "Buy Now",
        "object_detected": "General"
    }
    
    
    ERROR HANDLING:
    ===============
    
    The module handles errors gracefully:
    
    - Missing model file: Uses random weights (logs warning)
    - OCR failure: Returns empty text
    - Invalid image: Skips and continues
    - Batch failure: Continues with next batch
    - Missing config: Uses defaults
    
    
    PERFORMANCE TIPS:
    =================
    
    1. Use batch processing for multiple images
    2. Adjust BATCH_SIZE based on GPU memory
    3. Model is loaded once and cached
    4. OCR is the slowest step (consider caching)
    5. Use GPU for faster inference
    """)
    
    print("=" * 80)
    print("End of Examples")
    print("=" * 80)