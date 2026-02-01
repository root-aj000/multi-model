import re
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch

# Import BERT tokenizer from transformers library
from transformers import AutoTokenizer, BertTokenizer

# Configure logging for this module
# This helps us track what's happening and debug issues in production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Default maximum length for tokenized sequences
# BERT has a maximum of 512 tokens, but 128 is often sufficient
DEFAULT_MAX_LENGTH = 128

# Maximum reasonable text length before cleaning
# Extremely long texts might indicate data issues
MAX_RAW_TEXT_LENGTH = 100000

# BERT model name for tokenizer
# Using base uncased model (lowercase, 12 layers)
# BERT_MODEL_NAME = "bert-base-uncased"
BERT_MODEL_NAME = "distilbert-base-uncased"

# Local cache directory for models
# This will store the downloaded tokenizer files locally
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
# Go up one level to project root, then into models folder
PROJECT_ROOT = SCRIPT_DIR.parent
# LOCAL_MODEL_DIR = PROJECT_ROOT / "d_models" / "bert-base-uncased"
LOCAL_MODEL_DIR = PROJECT_ROOT / "local" /"tokenizer" / BERT_MODEL_NAME

# ============================================================================
# SETUP LOCAL MODEL DIRECTORY
# ============================================================================
# Create the local directory if it doesn't exist

try:
    # Create the models directory structure
    # parents=True creates all parent directories if needed
    # exist_ok=True doesn't raise error if directory already exists
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model cache directory: {LOCAL_MODEL_DIR}")
    
except Exception as e:
    logger.error(f"Failed to create model directory: {str(e)}")
    logger.warning("Will attempt to use default HuggingFace cache")
    LOCAL_MODEL_DIR = None


# ============================================================================
# INITIALIZE TOKENIZER
# ============================================================================
# Load the BERT tokenizer once when the module is imported
# This avoids reloading it every time we tokenize text

try:
    logger.info(f"Loading BERT tokenizer: {BERT_MODEL_NAME}")
    
    # Check if we should use local cache
    if LOCAL_MODEL_DIR is not None:
        # Check if tokenizer files already exist locally
        config_file = LOCAL_MODEL_DIR / "tokenizer_config.json"
        vocab_file = LOCAL_MODEL_DIR / "vocab.txt"
        
        if config_file.exists() and vocab_file.exists():
            # Load from local cache
            logger.info(f"Loading tokenizer from local cache: {LOCAL_MODEL_DIR}")
            tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_DIR))
            logger.info("✓ Loaded tokenizer from local cache")
        else:
            # Download and save to local cache
            logger.info(
                f"Tokenizer not found locally. "
                f"Downloading from HuggingFace and saving to: {LOCAL_MODEL_DIR}"
            )
            
            # Download from HuggingFace
            tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            
            # Save to local directory for future use
            tokenizer.save_pretrained(str(LOCAL_MODEL_DIR))
            
            logger.info(
                f"✓ Tokenizer downloaded and saved to: {LOCAL_MODEL_DIR}"
            )
    else:
        # Use default HuggingFace cache (fallback)
        logger.warning("Using default HuggingFace cache location")
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    logger.info(
        f"✓ BERT tokenizer loaded successfully. "
        f"Vocab size: {tokenizer.vocab_size}"
    )
    
    # Log tokenizer details
    logger.info(f"Tokenizer details:")
    logger.info(f"  - Model max length: {tokenizer.model_max_length}")
    logger.info(f"  - Padding token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    logger.info(f"  - CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    logger.info(f"  - SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    
except Exception as e:
    logger.error(f"Failed to load BERT tokenizer: {str(e)}")
    logger.error("Text preprocessing functions may not work correctly")
    tokenizer = None


def clean_text(text):
    """
    Clean OCR text: lowercase, strip, remove special characters.
    
    This function standardizes text by:
    1. Converting to lowercase (for consistency)
    2. Removing leading/trailing whitespace
    3. Removing special characters (keeping only letters, numbers, spaces)
    4. Normalizing multiple spaces to single space
    
    Why we need this:
    - OCR text often has noise and inconsistencies
    - Lowercase makes matching easier (e.g., "The" and "the" are same)
    - Special characters don't usually help with meaning
    - Normalized spacing makes tokenization better
    
    Args:
        text: Input text string (can be messy OCR output)
              Can also be non-string types (will be converted or handled)
    
    Returns:
        text: Cleaned text string, or empty string if input is invalid
        
    Example:
        >>> messy = "  Hello!!! This is   TEXT@123  "
        >>> clean = clean_text(messy)
        >>> print(clean)  # "hello this is text123"
    """
    
    # ========================================================================
    # STEP 1: VALIDATE AND CONVERT INPUT TO STRING
    # ========================================================================
    # Handle various input types gracefully
    
    try:
        # Check if text is None
        if text is None:
            logger.debug("Input text is None. Returning empty string.")
            return ""
        
        # Check if text is already a string
        if not isinstance(text, str):
            logger.debug(
                f"Input is not a string (type: {type(text)}). "
                f"Converting to string."
            )
            
            # Try to convert to string
            try:
                text = str(text)
                logger.debug("Successfully converted to string")
            except Exception as e:
                logger.warning(
                    f"Failed to convert input to string: {str(e)}. "
                    f"Returning empty string."
                )
                return ""
        
        # Log original text length
        original_length = len(text)
        logger.debug(f"Original text length: {original_length} characters")
        
        # Check for extremely long text (might indicate data issue)
        if original_length > MAX_RAW_TEXT_LENGTH:
            logger.warning(
                f"Text is very long ({original_length} chars). "
                f"Maximum recommended: {MAX_RAW_TEXT_LENGTH}. "
                f"This might indicate a data issue."
            )
        
        # Check if text is empty after initial validation
        if len(text) == 0:
            logger.debug("Input text is empty. Returning empty string.")
            return ""
        
    except Exception as e:
        logger.error(f"Error during input validation: {str(e)}")
        return ""
    
    
    # ========================================================================
    # STEP 2: CONVERT TO LOWERCASE
    # ========================================================================
    # Standardize case for consistency
    
    try:
        # Convert entire text to lowercase
        # This makes "Hello", "HELLO", and "hello" all the same
        # Important for text matching and consistency
        text = text.lower()
        
        logger.debug("Converted text to lowercase")
        
    except Exception as e:
        logger.error(f"Error converting to lowercase: {str(e)}")
        # If lowercase fails, continue with original text
        logger.warning("Continuing with original case")
    
    
    # ========================================================================
    # STEP 3: STRIP LEADING AND TRAILING WHITESPACE
    # ========================================================================
    # Remove spaces, tabs, newlines from start and end
    
    try:
        # .strip() removes whitespace from both ends
        # Whitespace includes: spaces, tabs (\t), newlines (\n), etc.
        text = text.strip()
        
        logger.debug(f"Stripped whitespace. New length: {len(text)}")
        
        # Check if text became empty after stripping
        if len(text) == 0:
            logger.debug("Text is empty after stripping. Returning empty string.")
            return ""
        
    except Exception as e:
        logger.error(f"Error stripping whitespace: {str(e)}")
        # Continue with unstripped text
    
    
    # ========================================================================
    # STEP 4: REMOVE SPECIAL CHARACTERS
    # ========================================================================
    # Keep only letters (a-z), numbers (0-9), and spaces
    
    try:
        # Regular expression pattern explanation:
        # [^a-z0-9\s] means "anything that is NOT":
        #   - a-z: lowercase letters
        #   - 0-9: digits
        #   - \s: whitespace (spaces, tabs, newlines)
        # ^ at the start means "NOT" (negation)
        # So this matches all special characters, punctuation, etc.
        
        logger.debug("Removing special characters...")
        
        # Count special characters before removal (for logging)
        special_chars_count = len(re.findall(r"[^a-z0-9\s]", text))
        
        if special_chars_count > 0:
            logger.debug(f"Found {special_chars_count} special characters to remove")
        
        # re.sub(pattern, replacement, text) replaces matches with replacement
        # We replace all special characters with empty string (remove them)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        
        logger.debug(f"Special characters removed. New length: {len(text)}")
        
        # Check if text became empty
        if len(text) == 0:
            logger.debug("Text is empty after removing special chars. Returning empty string.")
            return ""
        
    except Exception as e:
        logger.error(f"Error removing special characters: {str(e)}")
        logger.error(f"Pattern used: [^a-z0-9\\s]")
        # Continue with text as-is if regex fails
    
    
    # ========================================================================
    # STEP 5: NORMALIZE MULTIPLE SPACES TO SINGLE SPACE
    # ========================================================================
    # Replace multiple consecutive spaces with single space
    
    try:
        # Regular expression pattern explanation:
        # \s+ means "one or more whitespace characters"
        # This matches: "  " (2 spaces), "   " (3 spaces), "\t\t" (tabs), etc.
        # We replace all of these with a single space " "
        
        logger.debug("Normalizing whitespace...")
        
        # Count sequences of multiple spaces
        multi_space_count = len(re.findall(r"\s{2,}", text))
        
        if multi_space_count > 0:
            logger.debug(f"Found {multi_space_count} sequences of multiple spaces")
        
        # Replace all sequences of whitespace with single space
        text = re.sub(r"\s+", " ", text)
        
        logger.debug(f"Whitespace normalized. Final length: {len(text)}")
        
        # Final strip to remove any edge spaces created
        text = text.strip()
        
    except Exception as e:
        logger.error(f"Error normalizing whitespace: {str(e)}")
        logger.error(f"Pattern used: \\s+")
        # Continue with text as-is
    
    
    # ========================================================================
    # STEP 6: FINAL VALIDATION AND RETURN
    # ========================================================================
    
    try:
        # Final check
        if text is None:
            logger.warning("Text became None during processing. Returning empty string.")
            return ""
        
        # Log the cleaning summary
        final_length = len(text)
        
        logger.debug(
            f"Text cleaning complete. "
            f"Original: {original_length} chars, Final: {final_length} chars"
        )
        
        # Show sample of cleaned text (first 50 chars)
        if final_length > 0:
            sample = text[:50] + "..." if final_length > 50 else text
            logger.debug(f"Cleaned text sample: '{sample}'")
        
    except Exception as e:
        logger.error(f"Error during final validation: {str(e)}")
    
    # Return the cleaned text
    # Variable name is kept as 'text' as requested
    return text


def tokenize_text(text, max_length=DEFAULT_MAX_LENGTH):
    """
    Tokenize text using BERT tokenizer and return tensor of token IDs.
    
    This function converts text to numerical tokens that BERT understands:
    1. Splits text into subword tokens (BERT's vocabulary)
    2. Converts tokens to IDs (numbers)
    3. Adds special tokens ([CLS], [SEP])
    4. Pads or truncates to fixed length
    5. Creates attention mask (which tokens are real vs padding)
    
    Why we need this:
    - Neural networks need numbers, not text
    - BERT uses subword tokenization (handles unknown words better)
    - Fixed length allows batching multiple texts together
    - Attention mask tells model which tokens to focus on
    
    Args:
        text: Input text string (ideally cleaned with clean_text first)
        max_length: Maximum sequence length (default: 128)
                   Sequences longer than this will be truncated
                   Shorter sequences will be padded
    
    Returns:
        encoding: Dictionary containing:
                 - input_ids: Token IDs as tensor [seq_len]
                 - attention_mask: Attention mask as tensor [seq_len]
                 Or None if tokenization fails
        
    Example:
        >>> text = "hello world"
        >>> tokens = tokenize_text(text, max_length=10)
        >>> print(tokens['input_ids'].shape)  # torch.Size([10])
        >>> print(tokens['attention_mask'].shape)  # torch.Size([10])
    """
    
    # ========================================================================
    # STEP 1: VALIDATE TOKENIZER IS LOADED
    # ========================================================================
    # Check that the global tokenizer was loaded successfully
    
    try:
        # Check if tokenizer exists
        if tokenizer is None:
            logger.error(
                "BERT tokenizer is not loaded. "
                "Cannot tokenize text. "
                "Check tokenizer initialization at module import."
            )
            return None
        
        logger.debug("Tokenizer validation passed")
        
    except Exception as e:
        logger.error(f"Error validating tokenizer: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 2: VALIDATE AND PREPARE INPUT TEXT
    # ========================================================================
    # Ensure text is valid and in correct format
    
    try:
        # Check if text is None
        if text is None:
            logger.warning("Input text is None. Using empty string.")
            text = ""
        
        # Convert to string if not already
        if not isinstance(text, str):
            logger.warning(
                f"Input is not a string (type: {type(text)}). "
                f"Converting to string."
            )
            try:
                text = str(text)
            except Exception as e:
                logger.error(
                    f"Failed to convert input to string: {str(e)}. "
                    f"Using empty string."
                )
                text = ""
        
        # Log text information
        text_length = len(text)
        logger.debug(f"Input text length: {text_length} characters")
        
        # Show sample of text being tokenized
        if text_length > 0:
            sample = text[:50] + "..." if text_length > 50 else text
            logger.debug(f"Text to tokenize: '{sample}'")
        else:
            logger.debug("Text is empty (will tokenize to special tokens only)")
        
    except Exception as e:
        logger.error(f"Error validating input text: {str(e)}")
        text = ""
    
    
    # ========================================================================
    # STEP 3: VALIDATE MAX_LENGTH PARAMETER
    # ========================================================================
    # Ensure max_length is valid
    
    try:
        # Check if max_length is None
        if max_length is None:
            logger.warning(
                f"max_length is None. Using default: {DEFAULT_MAX_LENGTH}"
            )
            max_length = DEFAULT_MAX_LENGTH
        
        # Convert to integer if needed
        try:
            max_length = int(max_length)
        except (ValueError, TypeError) as e:
            logger.error(
                f"Cannot convert max_length to integer: {max_length}. "
                f"Error: {str(e)}. Using default: {DEFAULT_MAX_LENGTH}"
            )
            max_length = DEFAULT_MAX_LENGTH
        
        # Validate max_length is positive
        if max_length <= 0:
            logger.error(
                f"max_length must be positive: {max_length}. "
                f"Using default: {DEFAULT_MAX_LENGTH}"
            )
            max_length = DEFAULT_MAX_LENGTH
        
        # Check against BERT's maximum
        BERT_MAX_LENGTH = 512
        if max_length > BERT_MAX_LENGTH:
            logger.warning(
                f"max_length ({max_length}) exceeds BERT's maximum ({BERT_MAX_LENGTH}). "
                f"Using {BERT_MAX_LENGTH} instead."
            )
            max_length = BERT_MAX_LENGTH
        
        logger.debug(f"max_length validated: {max_length}")
        
    except Exception as e:
        logger.error(f"Error validating max_length: {str(e)}")
        max_length = DEFAULT_MAX_LENGTH
    
    
    # ========================================================================
    # STEP 4: TOKENIZE THE TEXT
    # ========================================================================
    # Use BERT tokenizer to convert text to tokens
    
    try:
        logger.debug("Starting tokenization...")
        
        # Call the BERT tokenizer
        # Parameters explained:
        #   text: The input text to tokenize
        #   add_special_tokens=True: Add [CLS] at start and [SEP] at end
        #   max_length: Maximum sequence length
        #   padding="max_length": Pad sequences shorter than max_length
        #                         This ensures all sequences have same length
        #   truncation=True: Cut off sequences longer than max_length
        #                    This prevents errors with long texts
        #   return_tensors="pt": Return PyTorch tensors (not lists)
        #                        "pt" stands for PyTorch
        
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        logger.debug("Tokenization successful")
        
        # Validate encoding result
        if encoding is None:
            logger.error("Tokenizer returned None")
            return None
        
        # Check if encoding has the required keys
        # BatchEncoding is dictionary-like, so we check for key access
        try:
            # Try to access the keys - this will work for both dict and BatchEncoding
            _ = encoding["input_ids"]
            _ = encoding["attention_mask"]
            logger.debug("Encoding has required keys")
        except (KeyError, TypeError) as e:
            logger.error(
                f"Encoding is missing required keys or is not dict-like: {str(e)}"
            )
            return None
        
        logger.debug(f"Encoding keys: {list(encoding.keys())}")
        
    except Exception as e:
        logger.error(f"Error during tokenization: {str(e)}")
        logger.error(f"Text length: {len(text)}, max_length: {max_length}")
        return None
    
    
    # ========================================================================
    # STEP 5: EXTRACT AND VALIDATE INPUT_IDS
    # ========================================================================
    # Get the token IDs from the encoding
    
    try:
        # Check if input_ids exists in encoding
        if "input_ids" not in encoding:
            logger.error("'input_ids' not found in tokenizer output")
            return None
        
        # Get input_ids tensor
        input_ids = encoding["input_ids"]
        
        # Validate it's a tensor
        if not isinstance(input_ids, torch.Tensor):
            logger.error(
                f"input_ids is not a tensor. Got type: {type(input_ids)}"
            )
            return None
        
        logger.debug(f"input_ids shape before squeeze: {input_ids.shape}")
        
        # Squeeze to remove batch dimension
        # Tokenizer returns shape [1, seq_len] (batch of 1)
        # We want [seq_len] for single sample
        # .squeeze(0) removes the first dimension if it's size 1
        input_ids = input_ids.squeeze(0)
        
        logger.debug(f"input_ids shape after squeeze: {input_ids.shape}")
        
        # Validate final shape
        if len(input_ids.shape) != 1:
            logger.error(
                f"input_ids has wrong dimensions after squeeze: {input_ids.shape}. "
                f"Expected 1D tensor."
            )
            return None
        
        # Validate length matches max_length
        actual_length = input_ids.shape[0]
        if actual_length != max_length:
            logger.warning(
                f"input_ids length ({actual_length}) doesn't match "
                f"max_length ({max_length})"
            )
        
        # Count non-padding tokens
        # Padding token ID is usually 0 in BERT
        non_padding = (input_ids != tokenizer.pad_token_id).sum().item()
        logger.debug(
            f"input_ids: {non_padding} real tokens, "
            f"{actual_length - non_padding} padding tokens"
        )
        
        # Show first few token IDs for debugging
        sample_ids = input_ids[:10].tolist()
        logger.debug(f"First 10 token IDs: {sample_ids}")
        
    except Exception as e:
        logger.error(f"Error processing input_ids: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 6: EXTRACT AND VALIDATE ATTENTION_MASK
    # ========================================================================
    # Get the attention mask from the encoding
    
    try:
        # Check if attention_mask exists in encoding
        if "attention_mask" not in encoding:
            logger.error("'attention_mask' not found in tokenizer output")
            return None
        
        # Get attention_mask tensor
        attention_mask = encoding["attention_mask"]
        
        # Validate it's a tensor
        if not isinstance(attention_mask, torch.Tensor):
            logger.error(
                f"attention_mask is not a tensor. Got type: {type(attention_mask)}"
            )
            return None
        
        logger.debug(f"attention_mask shape before squeeze: {attention_mask.shape}")
        
        # Squeeze to remove batch dimension
        # Same as input_ids, we want [seq_len] not [1, seq_len]
        attention_mask = attention_mask.squeeze(0)
        
        logger.debug(f"attention_mask shape after squeeze: {attention_mask.shape}")
        
        # Validate final shape
        if len(attention_mask.shape) != 1:
            logger.error(
                f"attention_mask has wrong dimensions after squeeze: {attention_mask.shape}. "
                f"Expected 1D tensor."
            )
            return None
        
        # Validate length matches input_ids
        if attention_mask.shape[0] != input_ids.shape[0]:
            logger.error(
                f"attention_mask length ({attention_mask.shape[0]}) doesn't match "
                f"input_ids length ({input_ids.shape[0]})"
            )
            return None
        
        # Count attended tokens (should match non-padding in input_ids)
        attended = attention_mask.sum().item()
        logger.debug(f"attention_mask: {attended} tokens to attend to")
        
        # Show first few mask values for debugging
        sample_mask = attention_mask[:10].tolist()
        logger.debug(f"First 10 attention mask values: {sample_mask}")
        
        # Validate attention_mask values are 0 or 1
        unique_values = attention_mask.unique().tolist()
        if not all(v in [0, 1] for v in unique_values):
            logger.warning(
                f"attention_mask contains unexpected values: {unique_values}. "
                f"Expected only 0 and 1."
            )
        
    except Exception as e:
        logger.error(f"Error processing attention_mask: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 7: CREATE OUTPUT DICTIONARY
    # ========================================================================
    # Combine the processed tensors into a dictionary
    
    try:
        # Create the output dictionary
        # This format matches what the model expects
        encoding = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        logger.debug("Created output dictionary")
        
        # Validate the dictionary
        for key, value in encoding.items():
            if not isinstance(value, torch.Tensor):
                logger.error(
                    f"Output {key} is not a tensor: {type(value)}"
                )
                return None
            
            logger.debug(
                f"  {key}: shape {value.shape}, dtype {value.dtype}"
            )
        
    except Exception as e:
        logger.error(f"Error creating output dictionary: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 8: FINAL VALIDATION AND RETURN
    # ========================================================================
    
    try:
        # Final checks
        if encoding is None:
            logger.error("Final encoding is None")
            return None
        
        # Check for required keys
        required_keys = ["input_ids", "attention_mask"]
        missing_keys = [k for k in required_keys if k not in encoding]
        
        if missing_keys:
            logger.error(f"Missing required keys in output: {missing_keys}")
            return None
        
        # Log success
        logger.info(
            f"Text tokenization completed successfully. "
            f"Sequence length: {encoding['input_ids'].shape[0]}, "
            f"Active tokens: {encoding['attention_mask'].sum().item()}"
        )
        
    except Exception as e:
        logger.error(f"Error during final validation: {str(e)}")
        return None
    
    # Return the encoding dictionary
    # Variable name is kept as 'encoding' as requested
    return encoding


# ============================================================================
# USAGE EXAMPLES AND TEST CASES
# ============================================================================

if __name__ == "__main__":
    """
    This section demonstrates how to use the text preprocessing functions.
    It only runs when you execute this file directly (not when imported).
    """
    
    print("=" * 70)
    print("TEXT PREPROCESSING - USAGE EXAMPLES")
    print("=" * 70)
    print()
    
    # Show where tokenizer is cached
    print("TOKENIZER LOCATION")
    print("-" * 70)
    if LOCAL_MODEL_DIR:
        print(f"Local cache: {LOCAL_MODEL_DIR}")
        print(f"Exists: {LOCAL_MODEL_DIR.exists()}")
        if LOCAL_MODEL_DIR.exists():
            files = list(LOCAL_MODEL_DIR.glob("*"))
            print(f"Files in cache: {len(files)}")
            for f in files[:5]:  # Show first 5 files
                print(f"  - {f.name}")
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 1: Basic Text Cleaning
    # ------------------------------------------------------------------------
    print("USE CASE 1: Basic text cleaning")
    print("-" * 70)
    
    try:
        # Test with messy text (like OCR output)
        messy_text = "  Hello!!! This is   MESSY text@2024...  "
        print(f"Original text: '{messy_text}'")
        
        # Clean the text
        clean = clean_text(messy_text)
        
        print(f"Cleaned text:  '{clean}'")
        print(f"Length change: {len(messy_text)} -> {len(clean)} characters")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 2: Cleaning Various Text Types
    # ------------------------------------------------------------------------
    print("USE CASE 2: Cleaning various text formats")
    print("-" * 70)
    
    test_cases = [
        "UPPERCASE TEXT",
        "Special!@#$%Chars",
        "Multiple   Spaces",
        "  Leading and trailing  ",
        "Mix123OF456Everything!!!",
        "",  # Empty string
        "   ",  # Only spaces
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        clean = clean_text(test_text)
        print(f"{i}. '{test_text}' -> '{clean}'")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 3: Basic Text Tokenization
    # ------------------------------------------------------------------------
    print("USE CASE 3: Basic text tokenization")
    print("-" * 70)
    
    try:
        # Test with simple text
        test_text = "hello world this is a test"
        print(f"Text: '{test_text}'")
        
        # Tokenize with default max_length
        tokens = tokenize_text(test_text)
        
        if tokens is not None:
            print("✓ Tokenization successful!")
            print(f"  Keys: {list(tokens.keys())}")
            print(f"  input_ids shape: {tokens['input_ids'].shape}")
            print(f"  attention_mask shape: {tokens['attention_mask'].shape}")
            print(f"  Number of real tokens: {tokens['attention_mask'].sum().item()}")
            
            # Show first 15 tokens
            print(f"  First 15 token IDs: {tokens['input_ids'][:15].tolist()}")
            print(f"  First 15 attention: {tokens['attention_mask'][:15].tolist()}")
        else:
            print("✗ Tokenization failed")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 4: Complete Pipeline (Clean + Tokenize)
    # ------------------------------------------------------------------------
    print("USE CASE 4: Complete preprocessing pipeline")
    print("-" * 70)
    
    try:
        # Start with messy OCR-like text
        raw_text = "  Product#123:  SALE!!!   50% OFF   Today Only...  "
        print(f"Step 1 - Raw OCR text:")
        print(f"  '{raw_text}'")
        print()
        
        # Step 1: Clean the text
        cleaned = clean_text(raw_text)
        print(f"Step 2 - After cleaning:")
        print(f"  '{cleaned}'")
        print()
        
        # Step 2: Tokenize the cleaned text
        tokens = tokenize_text(cleaned, max_length=20)
        
        if tokens is not None:
            print(f"Step 3 - After tokenization:")
            print(f"  ✓ Ready for model!")
            print(f"  Token IDs shape: {tokens['input_ids'].shape}")
            print(f"  Active tokens: {tokens['attention_mask'].sum().item()}")
        else:
            print(f"Step 3 - Tokenization failed")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)