import logging
import os
from pathlib import Path
from typing import Optional
import torch

# Import BERT tokenizer from transformers library
from transformers import BertTokenizer

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

# BERT model name for tokenizer
# Using base uncased model (lowercase, 12 layers)
BERT_MODEL_NAME = "bert-base-uncased"

# Default maximum sequence length
# BERT supports up to 512, but 128 is often sufficient for most tasks
DEFAULT_MAX_LENGTH = 128

# Local cache directory for models
# This will store the downloaded tokenizer files locally
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
# Go up one level to project root, then into models folder
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_MODEL_DIR = PROJECT_ROOT / "local" /"tokenizer" / BERT_MODEL_NAME

# ============================================================================
# SETUP LOCAL MODEL DIRECTORY
# ============================================================================
# Create the local directory if it doesn't exist
# This ensures we have a place to cache the tokenizer

try:
    # Create the models directory structure
    # parents=True creates all parent directories if needed
    # exist_ok=True doesn't raise error if directory already exists
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Model cache directory ready: {LOCAL_MODEL_DIR}")
    logger.debug(f"Directory exists: {LOCAL_MODEL_DIR.exists()}")
    
except Exception as e:
    logger.error(f"Failed to create model directory: {str(e)}")
    logger.warning("Will attempt to use default HuggingFace cache instead")
    # Set to None so we know to use default cache
    LOCAL_MODEL_DIR = None


# ============================================================================
# INITIALIZE BERT TOKENIZER
# ============================================================================
# Load the BERT tokenizer once when the module is imported
# This is more efficient than loading it every time we tokenize

try:
    logger.info("=" * 70)
    logger.info("Initializing BERT Tokenizer")
    logger.info("=" * 70)
    logger.info(f"Model: {BERT_MODEL_NAME}")
    
    # Check if we have a local cache directory
    if LOCAL_MODEL_DIR is not None:
        
        # Check if tokenizer files already exist locally
        # These are the essential files for BERT tokenizer
        config_file = LOCAL_MODEL_DIR / "tokenizer_config.json"
        vocab_file = LOCAL_MODEL_DIR / "vocab.txt"
        
        logger.debug(f"Checking for local tokenizer files...")
        logger.debug(f"  Config file: {config_file}")
        logger.debug(f"  Vocab file: {vocab_file}")
        
        # Check if both essential files exist
        if config_file.exists() and vocab_file.exists():
            # ================================================================
            # LOAD FROM LOCAL CACHE
            # ================================================================
            # Files exist locally, so we can load from there (faster)
            
            logger.info("✓ Found tokenizer in local cache")
            logger.info(f"Loading from: {LOCAL_MODEL_DIR}")
            
            try:
                # Load tokenizer from local directory
                # str() converts Path to string (required by transformers)
                tokenizer = BertTokenizer.from_pretrained(str(LOCAL_MODEL_DIR))
                
                logger.info("✓ Successfully loaded tokenizer from local cache")
                logger.info("  (No internet connection needed)")
                
            except Exception as e:
                logger.error(f"Failed to load from local cache: {str(e)}")
                logger.info("Falling back to downloading from HuggingFace...")
                
                # Try downloading instead
                tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
                logger.info("✓ Downloaded tokenizer from HuggingFace")
        
        else:
            # ================================================================
            # DOWNLOAD AND SAVE TO LOCAL CACHE
            # ================================================================
            # Files don't exist locally, need to download
            
            logger.info("Tokenizer not found in local cache")
            logger.info(f"Downloading from HuggingFace: {BERT_MODEL_NAME}")
            logger.info("(This is a one-time download, ~230KB)")
            
            try:
                # Download tokenizer from HuggingFace
                tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
                
                logger.info("✓ Tokenizer downloaded successfully")
                
                # Save to local directory for future use
                logger.info(f"Saving to local cache: {LOCAL_MODEL_DIR}")
                
                tokenizer.save_pretrained(str(LOCAL_MODEL_DIR))
                
                logger.info("✓ Tokenizer saved to local cache")
                logger.info("  (Future runs will load from local cache)")
                
            except Exception as e:
                logger.error(f"Failed to download and save tokenizer: {str(e)}")
                raise
    
    else:
        # ====================================================================
        # USE DEFAULT HUGGINGFACE CACHE
        # ====================================================================
        # Local directory couldn't be created, use default cache
        
        logger.warning("Using default HuggingFace cache location")
        logger.info(f"Downloading tokenizer: {BERT_MODEL_NAME}")
        
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        
        logger.info("✓ Tokenizer loaded using default cache")
    
    # ========================================================================
    # LOG TOKENIZER INFORMATION
    # ========================================================================
    # Display useful information about the loaded tokenizer
    
    logger.info("-" * 70)
    logger.info("Tokenizer Details:")
    logger.info(f"  Vocabulary size: {tokenizer.vocab_size:,}")
    logger.info(f"  Model max length: {tokenizer.model_max_length:,}")
    logger.info(f"  Padding token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    logger.info(f"  CLS token: '{tokenizer.cls_token}' (ID: {tokenizer.cls_token_id})")
    logger.info(f"  SEP token: '{tokenizer.sep_token}' (ID: {tokenizer.sep_token_id})")
    logger.info(f"  UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    logger.info(f"  MASK token: '{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")
    logger.info("=" * 70)
    
except Exception as e:
    # ========================================================================
    # HANDLE TOKENIZER INITIALIZATION FAILURE
    # ========================================================================
    
    logger.error("=" * 70)
    logger.error("CRITICAL ERROR: Failed to initialize BERT tokenizer")
    logger.error("=" * 70)
    logger.error(f"Error details: {str(e)}")
    logger.error("The tokenize_text function will not work correctly")
    logger.error("Please check your internet connection or HuggingFace access")
    logger.error("=" * 70)
    
    # Set tokenizer to None so we can check for it later
    tokenizer = None


def tokenize_text(text, max_length=DEFAULT_MAX_LENGTH):
    """
    Tokenize text using BERT tokenizer and return tensor of token IDs.
    
    This function converts text into numerical tokens that BERT understands.
    It handles the complete tokenization pipeline:
    1. Splits text into subword tokens (BERT vocabulary)
    2. Converts tokens to numerical IDs
    3. Adds special tokens ([CLS] at start, [SEP] at end)
    4. Pads sequences to fixed length (for batching)
    5. Truncates sequences longer than max_length
    6. Returns as PyTorch tensor
    
    Why we need this:
    - BERT models need numerical input, not text
    - Fixed length allows batching multiple texts together
    - Special tokens help BERT understand sentence structure
    - Subword tokenization handles unknown words better
    
    Args:
        text: Input text string to tokenize
              Can be any text (will be converted to string if needed)
        max_length: Maximum sequence length (default: 128)
                   - Longer sequences will be truncated
                   - Shorter sequences will be padded
                   - Must be between 1 and 512 for BERT
    
    Returns:
        encoding: PyTorch tensor of token IDs with shape [max_length]
                 Or None if tokenization fails
        
    Example:
        >>> tokens = tokenize_text("Hello world", max_length=10)
        >>> print(tokens.shape)  # torch.Size([10])
        >>> print(tokens)  # tensor([101, 7592, 2088, 102, 0, 0, 0, 0, 0, 0])
        >>> # 101 = [CLS], 7592 = "hello", 2088 = "world", 102 = [SEP], 0 = [PAD]
    """
    
    # ========================================================================
    # STEP 1: VALIDATE TOKENIZER IS LOADED
    # ========================================================================
    # Check that the global tokenizer was initialized successfully
    
    try:
        # Check if tokenizer exists
        if tokenizer is None:
            logger.error(
                "BERT tokenizer is not initialized. "
                "Cannot tokenize text. "
                "Please check tokenizer initialization logs above."
            )
            return None
        
        logger.debug("Tokenizer validation passed")
        
    except Exception as e:
        logger.error(f"Error validating tokenizer: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 2: VALIDATE AND PREPARE INPUT TEXT
    # ========================================================================
    # Ensure text is valid and in the correct format
    
    try:
        # Check if text is None
        if text is None:
            logger.warning("Input text is None. Using empty string.")
            text = ""
        
        # Check if text is already a string
        if not isinstance(text, str):
            logger.warning(
                f"Input is not a string (type: {type(text)}). "
                f"Converting to string."
            )
            
            # Try to convert to string
            try:
                text = str(text)
                logger.debug("Successfully converted input to string")
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
            # Show first 100 characters if text is long
            sample = text[:100] + "..." if text_length > 100 else text
            logger.debug(f"Text to tokenize: '{sample}'")
        else:
            logger.debug("Text is empty (will tokenize to [CLS] [SEP] + padding)")
        
        # Warn if text is very long
        if text_length > 10000:
            logger.warning(
                f"Text is very long ({text_length} chars). "
                f"This may be slow to tokenize. "
                f"Consider splitting into smaller chunks."
            )
        
    except Exception as e:
        logger.error(f"Error validating input text: {str(e)}")
        # Use empty string as fallback
        text = ""
    
    
    # ========================================================================
    # STEP 3: VALIDATE MAX_LENGTH PARAMETER
    # ========================================================================
    # Ensure max_length is valid for BERT
    
    try:
        # Check if max_length is None
        if max_length is None:
            logger.warning(
                f"max_length is None. Using default: {DEFAULT_MAX_LENGTH}"
            )
            max_length = DEFAULT_MAX_LENGTH
        
        # Try to convert to integer
        try:
            max_length = int(max_length)
            logger.debug(f"max_length: {max_length}")
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
        
        # Check against BERT's maximum (512 tokens)
        BERT_MAX_LENGTH = 512
        if max_length > BERT_MAX_LENGTH:
            logger.warning(
                f"max_length ({max_length}) exceeds BERT's maximum ({BERT_MAX_LENGTH}). "
                f"Using {BERT_MAX_LENGTH} instead."
            )
            max_length = BERT_MAX_LENGTH
        
        # Warn if max_length is very small
        if max_length < 10:
            logger.warning(
                f"max_length is very small ({max_length}). "
                f"Most text will be truncated. "
                f"Consider using at least 64."
            )
        
        logger.debug(f"max_length validated: {max_length}")
        
    except Exception as e:
        logger.error(f"Error validating max_length: {str(e)}")
        max_length = DEFAULT_MAX_LENGTH
    
    
    # ========================================================================
    # STEP 4: TOKENIZE THE TEXT
    # ========================================================================
    # Use the BERT tokenizer to convert text to tokens
    
    try:
        logger.debug("Starting tokenization process...")
        
        # Call the BERT tokenizer
        # This does multiple things in one call:
        # 1. Splits text into tokens (subword tokenization)
        # 2. Converts tokens to IDs (numbers)
        # 3. Adds special tokens ([CLS], [SEP])
        # 4. Pads or truncates to max_length
        # 5. Returns as PyTorch tensor
        
        # Parameters explained:
        #   text: The input text to tokenize
        #
        #   add_special_tokens=True:
        #     Adds [CLS] token at the beginning (ID: 101)
        #     Adds [SEP] token at the end (ID: 102)
        #     These help BERT understand sentence boundaries
        #
        #   max_length: Maximum sequence length
        #     If text produces more tokens, they'll be cut off
        #     If text produces fewer tokens, padding will be added
        #
        #   padding="max_length":
        #     Always pad to max_length (not just to longest in batch)
        #     Padding token is [PAD] (ID: 0)
        #     This ensures consistent tensor size
        #
        #   truncation=True:
        #     If text is too long, cut it off at max_length
        #     Without this, long text would cause an error
        #     Truncation happens before adding [SEP] token
        #
        #   return_tensors="pt":
        #     Return PyTorch tensors instead of Python lists
        #     "pt" stands for PyTorch
        #     Makes it ready to use with PyTorch models
        
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        logger.debug("Tokenization completed successfully")
        
        # Validate encoding result
        if encoding is None:
            logger.error("Tokenizer returned None")
            return None
        
        # Check if encoding has the expected structure
        # The tokenizer returns a BatchEncoding object (dictionary-like)
        try:
            # Try to access input_ids - this should always exist
            _ = encoding["input_ids"]
            logger.debug("Encoding structure validated")
        except (KeyError, TypeError) as e:
            logger.error(
                f"Encoding is missing 'input_ids' or is not dict-like: {str(e)}"
            )
            return None
        
        logger.debug(f"Encoding keys: {list(encoding.keys())}")
        
    except Exception as e:
        logger.error(f"Error during tokenization: {str(e)}")
        logger.error(f"Text length: {len(text)}, max_length: {max_length}")
        return None
    
    
    # ========================================================================
    # STEP 5: EXTRACT INPUT_IDS TENSOR
    # ========================================================================
    # Get the token IDs from the encoding dictionary
    
    try:
        # Check if input_ids exists in encoding
        if "input_ids" not in encoding:
            logger.error("'input_ids' not found in tokenizer output")
            return None
        
        # Get the input_ids tensor
        # This contains the numerical token IDs
        input_ids = encoding["input_ids"]
        
        # Validate it's a tensor
        if not isinstance(input_ids, torch.Tensor):
            logger.error(
                f"input_ids is not a PyTorch tensor. "
                f"Got type: {type(input_ids)}"
            )
            return None
        
        logger.debug(f"input_ids tensor shape (before squeeze): {input_ids.shape}")
        
        # The tokenizer returns shape [1, seq_len] (batch size of 1)
        # We need to remove this batch dimension to get [seq_len]
        # because we're processing a single text, not a batch
        
        # Example:
        #   Before squeeze: torch.Size([1, 128])
        #   After squeeze:  torch.Size([128])
        
        # .squeeze(0) removes the first dimension if it has size 1
        input_ids = input_ids.squeeze(0)
        
        logger.debug(f"input_ids tensor shape (after squeeze): {input_ids.shape}")
        
        # Validate the final shape
        if len(input_ids.shape) != 1:
            logger.error(
                f"input_ids has wrong number of dimensions: {input_ids.shape}. "
                f"Expected 1D tensor (shape: [max_length])"
            )
            return None
        
        # Validate the length matches what we expected
        actual_length = input_ids.shape[0]
        
        if actual_length != max_length:
            logger.warning(
                f"input_ids length ({actual_length}) doesn't match "
                f"max_length ({max_length}). "
                f"This might indicate a tokenization issue."
            )
        
        logger.debug(f"Final tensor length: {actual_length}")
        
    except Exception as e:
        logger.error(f"Error extracting input_ids: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 6: ANALYZE TOKEN STATISTICS (FOR LOGGING)
    # ========================================================================
    # Gather information about the tokens for debugging
    
    try:
        # Count different types of tokens
        
        # Count padding tokens (token ID = 0)
        num_padding = (input_ids == tokenizer.pad_token_id).sum().item()
        
        # Count non-padding tokens (real content)
        num_real_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
        
        # Count special tokens
        num_cls = (input_ids == tokenizer.cls_token_id).sum().item()
        num_sep = (input_ids == tokenizer.sep_token_id).sum().item()
        
        # Calculate content tokens (excluding special tokens and padding)
        num_content = num_real_tokens - num_cls - num_sep
        
        logger.debug("Token statistics:")
        logger.debug(f"  Total length: {actual_length}")
        logger.debug(f"  Real tokens: {num_real_tokens}")
        logger.debug(f"  Content tokens: {num_content}")
        logger.debug(f"  Special tokens: {num_cls + num_sep} ([CLS]: {num_cls}, [SEP]: {num_sep})")
        logger.debug(f"  Padding tokens: {num_padding}")
        
        # Show first few token IDs for debugging
        sample_size = min(15, actual_length)
        sample_ids = input_ids[:sample_size].tolist()
        logger.debug(f"First {sample_size} token IDs: {sample_ids}")
        
        # Decode first few tokens to show what they are
        if num_content > 0:
            # Decode without special tokens to see the actual text
            sample_text = tokenizer.decode(
                input_ids[:sample_size],
                skip_special_tokens=True
            )
            logger.debug(f"Decoded sample: '{sample_text}'")
        
    except Exception as e:
        logger.warning(f"Error analyzing token statistics: {str(e)}")
        # This is just for logging, so we continue even if it fails
    
    
    # ========================================================================
    # STEP 7: VALIDATE OUTPUT TENSOR
    # ========================================================================
    # Final checks before returning
    
    try:
        # Check for None
        if input_ids is None:
            logger.error("Final input_ids tensor is None")
            return None
        
        # Check tensor has data
        if input_ids.numel() == 0:
            logger.error("Final input_ids tensor is empty (no elements)")
            return None
        
        # Check for invalid values (NaN or Inf)
        if torch.isnan(input_ids).any():
            logger.error("input_ids tensor contains NaN values")
            return None
        
        if torch.isinf(input_ids).any():
            logger.error("input_ids tensor contains infinite values")
            return None
        
        # Check data type is correct
        # Token IDs should be long integers
        if input_ids.dtype not in [torch.long, torch.int64]:
            logger.warning(
                f"input_ids has unexpected dtype: {input_ids.dtype}. "
                f"Expected torch.long or torch.int64"
            )
        
        # Log success
        logger.info(
            f"Text tokenization successful. "
            f"Output: tensor with {actual_length} token IDs "
            f"({num_real_tokens} real, {num_padding} padding)"
        )
        
    except Exception as e:
        logger.error(f"Error during final validation: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 8: RETURN THE RESULT
    # ========================================================================
    
    # Return the token IDs tensor
    # Variable name is kept as 'encoding' as requested
    # (we reuse the name for the final output)
    encoding = input_ids
    
    return encoding


# ============================================================================
# USAGE EXAMPLES AND TEST CASES
# ============================================================================

if __name__ == "__main__":
    """
    This section demonstrates how to use the tokenize_text function.
    It only runs when you execute this file directly (not when imported).
    """
    
    print("\n" + "=" * 70)
    print("BERT TOKENIZER - USAGE EXAMPLES")
    print("=" * 70)
    print()
    
    # ------------------------------------------------------------------------
    # DISPLAY TOKENIZER LOCATION
    # ------------------------------------------------------------------------
    print("TOKENIZER CACHE LOCATION")
    print("-" * 70)
    
    if LOCAL_MODEL_DIR and LOCAL_MODEL_DIR.exists():
        print(f"✓ Local cache: {LOCAL_MODEL_DIR}")
        
        # List files in cache
        files = list(LOCAL_MODEL_DIR.glob("*"))
        print(f"✓ Files in cache: {len(files)}")
        
        for f in sorted(files):
            size = f.stat().st_size / 1024  # Size in KB
            print(f"  - {f.name} ({size:.1f} KB)")
    else:
        print("Using HuggingFace default cache")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 1: Basic Tokenization
    # ------------------------------------------------------------------------
    print("USE CASE 1: Basic tokenization")
    print("-" * 70)
    
    try:
        # Simple text
        text = "Hello world"
        print(f"Text: '{text}'")
        
        # Tokenize with default max_length (128)
        tokens = tokenize_text(text)
        
        if tokens is not None:
            print(f"✓ Success!")
            print(f"  Output type: {type(tokens)}")
            print(f"  Output shape: {tokens.shape}")
            print(f"  Token IDs: {tokens.tolist()[:20]}")  # Show first 20
            
            # Decode to see what we got
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"  Decoded: '{decoded}'")
        else:
            print("✗ Tokenization failed")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 2: Different Max Lengths
    # ------------------------------------------------------------------------
    print("USE CASE 2: Tokenization with different max_length values")
    print("-" * 70)
    
    try:
        text = "This is a sample sentence for testing tokenization"
        print(f"Text: '{text}'")
        print()
        
        # Try different max_length values
        for max_len in [10, 20, 50]:
            tokens = tokenize_text(text, max_length=max_len)
            
            if tokens is not None:
                # Count real tokens (non-padding)
                num_real = (tokens != tokenizer.pad_token_id).sum().item()
                num_pad = max_len - num_real
                
                print(f"max_length={max_len}:")
                print(f"  Shape: {tokens.shape}")
                print(f"  Real tokens: {num_real}, Padding: {num_pad}")
                print(f"  Token IDs: {tokens.tolist()}")
            else:
                print(f"max_length={max_len}: Failed")
            
            print()
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 3: Comparing Original vs Tokenized
    # ------------------------------------------------------------------------
    print("USE CASE 3: Understanding tokenization")
    print("-" * 70)
    
    try:
        text = "BERT tokenization is amazing!"
        print(f"Original text: '{text}'")
        
        # Tokenize
        tokens = tokenize_text(text, max_length=20)
        
        if tokens is not None:
            print(f"Token IDs: {tokens.tolist()}")
            
            # Decode each token individually to understand what they are
            print("\nToken-by-token breakdown:")
            for i, token_id in enumerate(tokens[:10]):  # First 10 tokens
                token_id = token_id.item()
                
                # Get the string representation
                token_str = tokenizer.decode([token_id])
                
                # Identify special tokens
                if token_id == tokenizer.cls_token_id:
                    token_type = "[CLS]"
                elif token_id == tokenizer.sep_token_id:
                    token_type = "[SEP]"
                elif token_id == tokenizer.pad_token_id:
                    token_type = "[PAD]"
                else:
                    token_type = "WORD"
                
                print(f"  Position {i}: ID={token_id:5d} -> '{token_str}' ({token_type})")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 4: Batch Processing Multiple Texts
    # ------------------------------------------------------------------------
    print("USE CASE 4: Tokenizing multiple texts for batching")
    print("-" * 70)
    
    try:
        # Multiple texts to tokenize
        texts = [
            "First example",
            "Second example with more words",
            "Third",
            "Fourth example text here",
        ]
        
        print(f"Tokenizing {len(texts)} texts...")
        print()
        
        tokenized_list = []
        
        for i, text in enumerate(texts, 1):
            tokens = tokenize_text(text, max_length=16)
            
            if tokens is not None:
                tokenized_list.append(tokens)
                num_real = (tokens != tokenizer.pad_token_id).sum().item()
                print(f"{i}. '{text}' -> {num_real} real tokens")
            else:
                print(f"{i}. '{text}' -> Failed")
        
        # Stack into batch tensor
        if tokenized_list:
            print()
            batch = torch.stack(tokenized_list)
            print(f"✓ Created batch tensor")
            print(f"  Shape: {batch.shape}")
            print(f"  (batch_size={batch.shape[0]}, sequence_length={batch.shape[1]})")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 5: Handling Edge Cases
    # ------------------------------------------------------------------------
    print("USE CASE 5: Testing edge cases")
    print("-" * 70)
    
    edge_cases = [
        ("Empty string", ""),
        ("None input", None),
        ("Number", 12345),
        ("Very long text", "word " * 200),
        ("Special characters", "!@#$%^&*()"),
        ("Unicode", "Hello 世界 🌍"),
    ]
    
    for name, test_input in edge_cases:
        print(f"{name}: ", end="")
        
        tokens = tokenize_text(test_input, max_length=20)
        
        if tokens is not None:
            num_real = (tokens != tokenizer.pad_token_id).sum().item()
            print(f"✓ ({num_real} tokens)")
        else:
            print("✗ Failed")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 6: Token ID to Text Conversion
    # ------------------------------------------------------------------------
    print("USE CASE 6: Converting tokens back to text")
    print("-" * 70)
    
    try:
        original = "Machine learning is fun"
        print(f"Original: '{original}'")
        
        # Tokenize
        tokens = tokenize_text(original, max_length=15)
        
        if tokens is not None:
            # Decode back to text
            # skip_special_tokens=True removes [CLS], [SEP], [PAD]
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"After tokenize->decode: '{decoded}'")
            
            # Check if they match (should be the same)
            match = original.lower() == decoded.lower()
            print(f"Match: {match}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 7: Analyzing Token Distribution
    # ------------------------------------------------------------------------
    print("USE CASE 7: Analyzing token usage")
    print("-" * 70)
    
    try:
        test_texts = [
            "short",
            "this is medium length text",
            "this is a much longer piece of text that will use more tokens",
        ]
        
        for text in test_texts:
            tokens = tokenize_text(text, max_length=30)
            
            if tokens is not None:
                total = len(tokens)
                real = (tokens != tokenizer.pad_token_id).sum().item()
                pad = total - real
                
                print(f"Text: '{text[:40]}...'")
                print(f"  Total: {total}, Real: {real} ({real/total*100:.1f}%), "
                      f"Padding: {pad} ({pad/total*100:.1f}%)")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 8: Performance Test
    # ------------------------------------------------------------------------
    print("USE CASE 8: Performance testing")
    print("-" * 70)
    
    try:
        import time
        
        test_text = "This is a test sentence " * 10
        num_iterations = 100
        
        print(f"Tokenizing {num_iterations} times...")
        
        start = time.time()
        for _ in range(num_iterations):
            tokenize_text(test_text, max_length=64)
        elapsed = time.time() - start
        
        avg_time = (elapsed / num_iterations) * 1000  # Convert to ms
        
        print(f"✓ Average time: {avg_time:.2f} ms per tokenization")
        print(f"  Total time: {elapsed:.2f} seconds")
        print(f"  Throughput: {num_iterations/elapsed:.1f} tokenizations/second")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Function: tokenize_text(text, max_length=128)")
    print()
    print("What it does:")
    print("  - Converts text to BERT token IDs")
    print("  - Adds [CLS] and [SEP] special tokens")
    print("  - Pads to max_length with [PAD] tokens")
    print("  - Truncates if text is too long")
    print("  - Returns PyTorch tensor ready for BERT model")
    print()
    print("Input:")
    print("  - text: String (or convertible to string)")
    print("  - max_length: Integer (1-512, default 128)")
    print()
    print("Output:")
    print("  - PyTorch tensor of shape [max_length]")
    print("  - Contains token IDs (integers)")
    print("  - None if tokenization fails")
    print()
    print("Tokenizer info:")
    if tokenizer:
        print(f"  - Vocabulary: {tokenizer.vocab_size:,} tokens")
        print(f"  - Model: {BERT_MODEL_NAME}")
        print(f"  - Cache: {LOCAL_MODEL_DIR if LOCAL_MODEL_DIR else 'HuggingFace default'}")
    print()
    print("All examples completed!")
    print("=" * 70)