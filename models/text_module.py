"""
Text Module for Feature Extraction
===================================
This module extracts meaningful features from text using transformer models.
It uses BERT (or similar models) to convert text into fixed-size feature vectors.

Author: [Your Name]
Date: [Date]
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from transformers import BertModel, AutoModel, BertTokenizer, AutoTokenizer
import os
from pathlib import Path


# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL CACHE DIRECTORY SETUP
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.parent

# Create a folder for downloaded models in the same directory as the script
MODEL_CACHE_DIR = SCRIPT_DIR / "local" / "BERT_MODELS"

# Create subdirectories for different types of models
TEXT_MODEL_CACHE_DIR = MODEL_CACHE_DIR / "text_models"
TEXT_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Text model cache directory: {TEXT_MODEL_CACHE_DIR}")

# Set environment variable to tell HuggingFace where to cache models
# This makes transformers download models to our custom directory
os.environ['TRANSFORMERS_CACHE'] = str(TEXT_MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(TEXT_MODEL_CACHE_DIR)
os.environ['HF_HUB_CACHE'] = str(TEXT_MODEL_CACHE_DIR)


# Supported encoder architectures
SUPPORTED_ENCODERS = {
    "bert-base-uncased": {
        "description": "BERT Base (110M params, good for general text)",
        "hidden_size": 768,
        "max_length": 512
    },
    # "bert-base-cased": {
    #     "description": "BERT Base Cased (110M params, preserves case)",
    #     "hidden_size": 768,
    #     "max_length": 512
    # },
    # "bert-large-uncased": {
    #     "description": "BERT Large (340M params, higher accuracy)",
    #     "hidden_size": 1024,
    #     "max_length": 512
    # },
    "distilbert-base-uncased": {
        "description": "DistilBERT (66M params, faster, 95% of BERT performance)",
        "hidden_size": 768,
        "max_length": 512
    },
    # "roberta-base": {
    #     "description": "RoBERTa Base (125M params, improved BERT)",
    #     "hidden_size": 768,
    #     "max_length": 512
    # },
    # "roberta-large": {
    #     "description": "RoBERTa Large (355M params, highest accuracy)",
    #     "hidden_size": 1024,
    #     "max_length": 512
    # }
}


class TextModule(nn.Module):
    """
    Text Feature Extraction Module
    
    This module processes tokenized text through a transformer model (BERT)
    and converts it into fixed-size feature vectors suitable for downstream tasks.
    
    Architecture:
    1. Text Tokens → BERT Encoder → Contextual Embeddings
    2. Extract [CLS] Token (represents entire sentence)
    3. Apply Dropout (for regularization)
    4. Linear Projection → Output Features
    
    The [CLS] token is used because BERT is trained to encode the entire
    sentence meaning into this special token.
    
    All pre-trained models are downloaded to: downloaded_models/text_models/
    """
    
    def __init__(
        self, 
        encoder_name: str = "bert-base-uncased", 
        out_features: int = 512, 
        freeze: bool = False
    ):
        """
        Initialize the Text Module with a transformer encoder.
        
        Args:
            encoder_name (str): Name of the pre-trained transformer model
                               Examples: 'bert-base-uncased', 'bert-large-uncased',
                                        'roberta-base', 'distilbert-base-uncased'
                               Default: 'bert-base-uncased'
            
            out_features (int): Dimension of the output feature vector
                               This is the size of features passed to fusion module
                               Default: 512
            
            freeze (bool): Whether to freeze the transformer weights
                          True = Don't update BERT during training (faster, less memory)
                          False = Fine-tune BERT weights (better performance, slower)
                          Default: False
        
        Raises:
            ValueError: If out_features is not positive
            RuntimeError: If model loading fails
        
        Note:
            Pre-trained models will be downloaded to: downloaded_models/text_models/
            The download happens only once, then cached for future use.
        """
        super(TextModule, self).__init__()
        
        # Step 1: Log initialization
        logger.info("=" * 60)
        logger.info("Initializing Text Module")
        logger.info("=" * 60)
        logger.info(f"Encoder: {encoder_name}")
        logger.info(f"Output Features: {out_features}")
        logger.info(f"Freeze Backbone: {freeze}")
        logger.info(f"Model Cache Directory: {TEXT_MODEL_CACHE_DIR}")
        
        # Step 2: Validate parameters
        self._validate_parameters(encoder_name, out_features, freeze)
        
        # Step 3: Store configuration
        self.encoder_name = encoder_name
        self.out_features = out_features
        self.freeze = freeze
        self.model_cache_dir = TEXT_MODEL_CACHE_DIR
        
        # Step 4: Load pre-trained transformer model
        # This is the main component that understands text
        try:
            logger.info(f"Loading pre-trained model: {encoder_name}")
            logger.info(f"Models will be downloaded/cached to: {TEXT_MODEL_CACHE_DIR}")
            
            # Check if model is already cached
            self._check_cached_models()
            
            # Try to load as BERT first, fallback to AutoModel for other types
            try:
                self.bert = BertModel.from_pretrained(
                    encoder_name,
                    cache_dir=str(TEXT_MODEL_CACHE_DIR)
                )
                logger.info("✓ Loaded as BertModel")
            except Exception as e:
                logger.warning(f"Failed to load as BertModel: {str(e)}")
                logger.info("Attempting to load with AutoModel...")
                self.bert = AutoModel.from_pretrained(
                    encoder_name,
                    cache_dir=str(TEXT_MODEL_CACHE_DIR)
                )
                logger.info("✓ Loaded with AutoModel")
            
            # Log model information
            logger.info(f"Model hidden size: {self.bert.config.hidden_size}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.bert.parameters()):,}")
            
            # Log cache location
            logger.info(f"✓ Model cached in: {TEXT_MODEL_CACHE_DIR}")
            
        except Exception as e:
            error_msg = f"Failed to load transformer model '{encoder_name}': {str(e)}"
            logger.error(error_msg)
            logger.error("Please check if the model name is correct and you have internet connection")
            raise RuntimeError(error_msg)
        
        # Step 5: Freeze transformer weights if requested
        # Freezing means the weights won't be updated during training
        # This is useful for:
        # - Faster training
        # - Less memory usage
        # - Preventing overfitting on small datasets
        if freeze:
            logger.info("Freezing transformer backbone weights...")
            frozen_params = 0
            
            for param in self.bert.parameters():
                # Set requires_grad to False means gradients won't be computed
                param.requires_grad = False
                frozen_params += param.numel()
            
            logger.info(f"✓ Frozen {frozen_params:,} parameters")
            logger.info("Note: Transformer weights will not be updated during training")
        else:
            trainable_params = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
            logger.info(f"Transformer is trainable with {trainable_params:,} parameters")
        
        # Step 6: Create dropout layer
        # Dropout randomly sets some activations to zero during training
        # This prevents overfitting by forcing the network to learn robust features
        # 0.3 means 30% of neurons are randomly turned off during training
        dropout_rate = 0.3
        self.dropout = nn.Dropout(dropout_rate)
        logger.info(f"Added dropout layer with rate: {dropout_rate}")
        
        # Step 7: Create projection layer
        # This layer transforms BERT's hidden size to our desired output size
        # For example: BERT-base has 768 dimensions, we might want 512
        try:
            bert_hidden_size = self.bert.config.hidden_size
            self.fc = nn.Linear(bert_hidden_size, out_features)
            logger.info(f"Created projection layer: {bert_hidden_size} → {out_features}")
            
            # Initialize weights properly for better training
            # Xavier initialization helps with gradient flow
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
            logger.info("✓ Initialized projection layer with Xavier uniform")
            
        except Exception as e:
            error_msg = f"Failed to create projection layer: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Step 8: Log total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info("=" * 60)
        logger.info("Text Module Initialization Complete")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info(f"Frozen Parameters: {total_params - trainable_params:,}")
        logger.info("=" * 60)
    
    
    def _check_cached_models(self) -> None:
        """
        Check if models are already cached and log information.
        
        This helps users understand if a download will occur or if
        the model will be loaded from cache.
        """
        if TEXT_MODEL_CACHE_DIR.exists():
            cached_items = list(TEXT_MODEL_CACHE_DIR.iterdir())
            if cached_items:
                logger.info(f"Found {len(cached_items)} item(s) in cache directory")
                
                # Check for model-specific directories
                model_dirs = [d for d in cached_items if d.is_dir() and 'model' in d.name.lower()]
                if model_dirs:
                    logger.info(f"Found {len(model_dirs)} model cache director(ies)")
                    logger.info("Model may load from cache (faster)")
                else:
                    logger.info("Model will be downloaded (first time)")
            else:
                logger.info("Cache directory is empty - model will be downloaded")
        else:
            logger.info("Cache directory doesn't exist - model will be downloaded")
    
    
    def _validate_parameters(
        self, 
        encoder_name: str, 
        out_features: int, 
        freeze: bool
    ) -> None:
        """
        Validate initialization parameters.
        
        This method checks if all parameters are valid before proceeding
        with model initialization. It's better to catch errors early.
        
        Args:
            encoder_name (str): Name of the encoder to validate
            out_features (int): Output feature dimension to validate
            freeze (bool): Freeze flag to validate
        
        Raises:
            TypeError: If parameters have wrong type
            ValueError: If parameters have invalid values
        """
        # Validate encoder_name
        if not isinstance(encoder_name, str):
            error_msg = f"encoder_name must be a string, got {type(encoder_name)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        if not encoder_name.strip():
            error_msg = "encoder_name cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log information about known encoders
        if encoder_name in SUPPORTED_ENCODERS:
            info = SUPPORTED_ENCODERS[encoder_name]
            logger.info(f"Encoder info: {info['description']}")
            logger.info(f"Hidden size: {info['hidden_size']}")
            logger.info(f"Max sequence length: {info['max_length']}")
        else:
            logger.warning(f"Encoder '{encoder_name}' is not in the list of known encoders")
            logger.warning("It may still work if it's a valid HuggingFace model")
            logger.warning(f"Known encoders: {list(SUPPORTED_ENCODERS.keys())}")
        
        # Validate out_features
        if not isinstance(out_features, int):
            error_msg = f"out_features must be an integer, got {type(out_features)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        if out_features <= 0:
            error_msg = f"out_features must be positive, got {out_features}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Common feature dimensions for sanity check
        if out_features > 4096:
            logger.warning(f"out_features={out_features} is unusually large. Are you sure?")
        
        if out_features < 64:
            logger.warning(f"out_features={out_features} is quite small. This may lose information.")
        
        # Validate freeze
        if not isinstance(freeze, bool):
            error_msg = f"freeze must be a boolean, got {type(freeze)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        logger.debug("✓ Parameter validation passed")
    

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to extract text features.
        
        This method processes tokenized text through the transformer and
        produces fixed-size feature vectors.
        
        Args:
            input_ids (torch.Tensor): Tokenized text input
                                     Shape: [batch_size, sequence_length]
                                     Each value is a token ID (integer)
                                     Example: [[101, 2023, 2003, 102, 0, 0],  # "this is"
                                              [101, 7592, 2088, 102, 0, 0]]  # "hello world"
            
            attention_mask (torch.Tensor, optional): Mask to ignore padding tokens
                                                    Shape: [batch_size, sequence_length]
                                                    1 = real token, 0 = padding token
                                                    Example: [[1, 1, 1, 1, 0, 0],
                                                             [1, 1, 1, 1, 0, 0]]
                                                    Default: None (all tokens are real)
        
        Returns:
            torch.Tensor: Extracted text features
                         Shape: [batch_size, out_features]
                         These features encode the meaning of the input text
        
        Raises:
            ValueError: If input_ids has invalid shape or type
            RuntimeError: If forward pass fails
        
        Example:
            >>> text_module = TextModule(out_features=512)
            >>> input_ids = torch.randint(0, 1000, (2, 128))  # 2 samples, 128 tokens
            >>> attention_mask = torch.ones(2, 128)
            >>> features = text_module(input_ids, attention_mask)
            >>> print(features.shape)  # torch.Size([2, 512])
        """
        try:
            # Step 1: Validate inputs
            logger.debug(f"Processing text batch. Input shape: {input_ids.shape}")
            self._validate_forward_inputs(input_ids, attention_mask)
            
            # Step 2: Pass through BERT/transformer encoder
            # The encoder processes all tokens and generates contextual embeddings
            # Each token's embedding is influenced by all other tokens (attention mechanism)
            logger.debug("Passing input through transformer encoder...")
            
            try:
                # Run the transformer model
                # outputs contains multiple things: hidden states, attentions, etc.
                outputs = self.bert(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
                logger.debug("✓ Transformer encoding complete")
                
            except Exception as e:
                error_msg = f"Transformer forward pass failed: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Input IDs shape: {input_ids.shape}")
                if attention_mask is not None:
                    logger.error(f"Attention mask shape: {attention_mask.shape}")
                raise RuntimeError(error_msg)
            
            # Step 3: Extract [CLS] token representation
            # The [CLS] token is the first token (index 0) in BERT
            # It's trained to represent the meaning of the entire sentence
            # 
            # outputs.last_hidden_state shape: [batch_size, sequence_length, hidden_size]
            # We want: [batch_size, hidden_size]
            # So we take [:, 0, :] which means:
            #   - : = all samples in batch
            #   - 0 = first token ([CLS])
            #   - : = all hidden dimensions
            try:
                cls_output = outputs.last_hidden_state[:, 0, :]
                logger.debug(f"Extracted [CLS] token. Shape: {cls_output.shape}")
                
                # Verify we got the expected shape
                batch_size = input_ids.size(0)
                expected_hidden_size = self.bert.config.hidden_size
                
                if cls_output.shape != (batch_size, expected_hidden_size):
                    error_msg = f"Unexpected [CLS] shape: {cls_output.shape}, expected: ({batch_size}, {expected_hidden_size})"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
            except Exception as e:
                error_msg = f"Failed to extract [CLS] token: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Step 4: Apply dropout for regularization
            # During training: randomly zeros some elements
            # During evaluation: does nothing (automatically handled by PyTorch)
            cls_output = self.dropout(cls_output)
            logger.debug("Applied dropout regularization")
            
            # Step 5: Project to output dimension
            # Transform from BERT's hidden size to our desired output size
            # This allows us to control the feature dimension for downstream tasks
            try:
                output_features = self.fc(cls_output)
                logger.debug(f"Projected to output features. Shape: {output_features.shape}")
                
                # Verify output shape
                if output_features.shape != (batch_size, self.out_features):
                    error_msg = f"Unexpected output shape: {output_features.shape}, expected: ({batch_size}, {self.out_features})"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Check for NaN or Inf values (indicates numerical instability)
                if torch.isnan(output_features).any():
                    error_msg = "NaN detected in output features"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                if torch.isinf(output_features).any():
                    error_msg = "Inf detected in output features"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
            except Exception as e:
                error_msg = f"Projection layer failed: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            logger.debug(f"✓ Text feature extraction complete. Output: {output_features.shape}")
            return output_features
            
        except Exception as e:
            # Catch any unexpected errors
            error_msg = f"Text module forward pass failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    
    def _validate_forward_inputs(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]
    ) -> None:
        """
        Validate inputs for the forward pass.
        
        This method ensures that input tensors have the correct type,
        shape, and values before processing.
        
        Args:
            input_ids (torch.Tensor): Input token IDs to validate
            attention_mask (torch.Tensor, optional): Attention mask to validate
        
        Raises:
            TypeError: If inputs are not tensors
            ValueError: If inputs have invalid shapes or values
        """
        # Validate input_ids type
        if not isinstance(input_ids, torch.Tensor):
            error_msg = f"input_ids must be torch.Tensor, got {type(input_ids)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Validate input_ids dimensions
        # Expected: 2D tensor [batch_size, sequence_length]
        if input_ids.dim() != 2:
            error_msg = f"input_ids must be 2D [batch_size, seq_len], got shape {input_ids.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check batch size is positive
        batch_size, seq_length = input_ids.shape
        if batch_size <= 0:
            error_msg = f"Batch size must be positive, got {batch_size}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check sequence length is reasonable
        if seq_length <= 0:
            error_msg = f"Sequence length must be positive, got {seq_length}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Warn if sequence length exceeds BERT's maximum (usually 512)
        max_seq_length = getattr(self.bert.config, 'max_position_embeddings', 512)
        if seq_length > max_seq_length:
            logger.warning(f"Sequence length {seq_length} exceeds model's maximum {max_seq_length}")
            logger.warning("This may cause errors or unexpected behavior")
        
        # Validate input_ids values are non-negative (token IDs can't be negative)
        if (input_ids < 0).any():
            error_msg = "input_ids contains negative values (token IDs must be non-negative)"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate input_ids are integers
        if not input_ids.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            logger.warning(f"input_ids should be integer type, got {input_ids.dtype}")
            logger.warning("Attempting to convert to long...")
            try:
                input_ids = input_ids.long()
            except Exception as e:
                error_msg = f"Failed to convert input_ids to long: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Validate attention_mask if provided
        if attention_mask is not None:
            # Check type
            if not isinstance(attention_mask, torch.Tensor):
                error_msg = f"attention_mask must be torch.Tensor, got {type(attention_mask)}"
                logger.error(error_msg)
                raise TypeError(error_msg)
            
            # Check shape matches input_ids
            if attention_mask.shape != input_ids.shape:
                error_msg = f"attention_mask shape {attention_mask.shape} doesn't match input_ids shape {input_ids.shape}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Check values are 0 or 1
            unique_values = torch.unique(attention_mask)
            if not all(v in [0, 1] for v in unique_values.tolist()):
                logger.warning(f"attention_mask should contain only 0s and 1s, got unique values: {unique_values.tolist()}")
        
        logger.debug("✓ Forward input validation passed")
    
    
    def get_model_cache_info(self) -> dict:
        """
        Get information about cached models in the download directory.
        
        Returns:
            dict: Information about cached models including:
                  - cache_dir: Path to cache directory
                  - exists: Whether directory exists
                  - cached_items: List of items in cache
                  - total_size_mb: Total size of cache in MB
        
        Example:
            >>> text_module = TextModule()
            >>> cache_info = text_module.get_model_cache_info()
            >>> print(f"Cache directory: {cache_info['cache_dir']}")
            >>> print(f"Total size: {cache_info['total_size_mb']:.2f} MB")
        """
        info = {
            'cache_dir': str(TEXT_MODEL_CACHE_DIR),
            'exists': TEXT_MODEL_CACHE_DIR.exists(),
            'cached_items': [],
            'total_size_mb': 0.0
        }
        
        if TEXT_MODEL_CACHE_DIR.exists():
            cached_items = list(TEXT_MODEL_CACHE_DIR.iterdir())
            info['cached_items'] = [item.name for item in cached_items]
            
            # Calculate total size
            total_size = 0
            for item in cached_items:
                if item.is_file():
                    total_size += item.stat().st_size
                elif item.is_dir():
                    # Recursively get size of directories
                    for subitem in item.rglob('*'):
                        if subitem.is_file():
                            total_size += subitem.stat().st_size
            
            info['total_size_mb'] = total_size / (1024 * 1024)
        
        return info


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_text_model_cache_directory() -> Path:
    """
    Get the path to the text model cache directory.
    
    Returns:
        Path: Path object pointing to the downloaded_models/text_models directory
    
    Example:
        >>> cache_dir = get_text_model_cache_directory()
        >>> print(f"Text models are cached in: {cache_dir}")
    """
    return TEXT_MODEL_CACHE_DIR


def list_cached_text_models() -> List[str]:
    """
    List all cached text model items in the download directory.
    
    Returns:
        List[str]: List of cached items (directories and files)
    
    Example:
        >>> cached_items = list_cached_text_models()
        >>> print(f"Found {len(cached_items)} cached item(s):")
        >>> for item in cached_items:
        ...     print(f"  - {item}")
    """
    if not TEXT_MODEL_CACHE_DIR.exists():
        return []
    
    cached_items = list(TEXT_MODEL_CACHE_DIR.iterdir())
    return [item.name for item in cached_items]


def get_text_cache_size() -> float:
    """
    Get the total size of cached text models in MB.
    
    Returns:
        float: Total size in megabytes
    
    Example:
        >>> cache_size = get_text_cache_size()
        >>> print(f"Total text model cache size: {cache_size:.2f} MB")
    """
    if not TEXT_MODEL_CACHE_DIR.exists():
        return 0.0
    
    total_size = 0
    for item in TEXT_MODEL_CACHE_DIR.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size
    
    return total_size / (1024 * 1024)


def clear_text_model_cache(confirm: bool = False) -> bool:
    """
    Clear all cached text models from the download directory.
    
    Args:
        confirm (bool): Must be True to actually delete files (safety measure)
    
    Returns:
        bool: True if cache was cleared, False if not confirmed
    
    Warning:
        This will delete all downloaded text model files!
        They will need to be re-downloaded next time.
    
    Example:
        >>> # Clear cache (requires confirmation)
        >>> cleared = clear_text_model_cache(confirm=True)
        >>> if cleared:
        ...     print("Text model cache cleared successfully")
    """
    if not confirm:
        logger.warning("clear_text_model_cache() called without confirmation")
        logger.warning("Set confirm=True to actually clear the cache")
        return False
    
    if not TEXT_MODEL_CACHE_DIR.exists():
        logger.info("Cache directory doesn't exist, nothing to clear")
        return True
    
    try:
        import shutil
        
        cached_items = list(TEXT_MODEL_CACHE_DIR.iterdir())
        logger.info(f"Deleting {len(cached_items)} cached item(s)...")
        
        for item in cached_items:
            if item.is_file():
                item.unlink()
                logger.info(f"  Deleted file: {item.name}")
            elif item.is_dir():
                shutil.rmtree(item)
                logger.info(f"  Deleted directory: {item.name}")
        
        logger.info("✓ Text model cache cleared successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return False


def load_tokenizer(encoder_name: str = "bert-base-uncased"):
    """
    Load a tokenizer with caching to the local directory.
    
    Args:
        encoder_name (str): Name of the encoder/tokenizer to load
    
    Returns:
        Tokenizer: HuggingFace tokenizer instance
    
    Example:
        >>> tokenizer = load_tokenizer("bert-base-uncased")
        >>> encoded = tokenizer("Hello world", return_tensors="pt")
    """
    logger.info(f"Loading tokenizer: {encoder_name}")
    logger.info(f"Tokenizer will be cached to: {TEXT_MODEL_CACHE_DIR}")
    
    try:
        # Try BERT tokenizer first
        tokenizer = BertTokenizer.from_pretrained(
            encoder_name,
            cache_dir=str(TEXT_MODEL_CACHE_DIR)
        )
        logger.info("✓ Loaded BertTokenizer")
    except Exception:
        # Fallback to AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            encoder_name,
            cache_dir=str(TEXT_MODEL_CACHE_DIR)
        )
        logger.info("✓ Loaded with AutoTokenizer")
    
    return tokenizer


# ============================================================================
# USAGE EXAMPLES AND TEST CASES
# ============================================================================

if __name__ == "__main__":
    """
    Example usage and test cases for the TextModule.
    This section demonstrates how to:
    1. Initialize the module with different configurations
    2. Process text inputs
    3. Handle errors
    4. Use with different transformer models
    5. Check cached models
    """
    
    print("=" * 80)
    print("TEXT MODULE - Usage Examples")
    print("=" * 80)
    
    # ------------------------------------------------------------------------
    # Example 0: Show Model Cache Information
    # ------------------------------------------------------------------------
    print("\n[Example 0] Model Cache Information")
    print("-" * 80)
    
    print(f"\nText model cache directory: {TEXT_MODEL_CACHE_DIR}")
    print(f"Directory exists: {TEXT_MODEL_CACHE_DIR.exists()}")
    
    cached_items = list_cached_text_models()
    cache_size = get_text_cache_size()
    
    print(f"\nCached items: {len(cached_items)}")
    if cached_items:
        for item in cached_items[:5]:  # Show first 5
            print(f"  - {item}")
        if len(cached_items) > 5:
            print(f"  ... and {len(cached_items) - 5} more")
        print(f"\nTotal cache size: {cache_size:.2f} MB")
    else:
        print("  (No models cached yet - they will be downloaded on first use)")
    
    # ------------------------------------------------------------------------
    # Example 1: Basic Usage with BERT
    # ------------------------------------------------------------------------
    print("\n[Example 1] Basic Usage with BERT-base")
    print("-" * 80)
    
    try:
        # Initialize text module
        print("Initializing TextModule...")
        print("(This will download the model if not already cached)")
        text_module = TextModule(
            encoder_name="bert-base-uncased",
            out_features=512,
            freeze=False
        )
        print("✓ TextModule initialized successfully\n")
        
        # Show cache info after initialization
        cache_info = text_module.get_model_cache_info()
        print(f"Cache information:")
        print(f"  - Cache directory: {cache_info['cache_dir']}")
        print(f"  - Items cached: {len(cache_info['cached_items'])}")
        print(f"  - Total size: {cache_info['total_size_mb']:.2f} MB")
        
        # Create dummy input data
        # In real usage, these would come from a tokenizer
        batch_size = 4
        seq_length = 128
        
        print(f"\nCreating dummy input:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Sequence length: {seq_length}")
        
        # Random token IDs (in real usage, these come from tokenizer)
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        # Attention mask (1 for real tokens, 0 for padding)
        # Let's simulate that each sample has different length
        attention_mask = torch.ones(batch_size, seq_length)
        attention_mask[0, 100:] = 0  # First sample: 100 real tokens
        attention_mask[1, 80:] = 0   # Second sample: 80 real tokens
        attention_mask[2, 120:] = 0  # Third sample: 120 real tokens
        attention_mask[3, 90:] = 0   # Fourth sample: 90 real tokens
        
        print(f"\nInput shapes:")
        print(f"  - input_ids: {input_ids.shape}")
        print(f"  - attention_mask: {attention_mask.shape}")
        
        # Forward pass
        print("\nRunning forward pass...")
        with torch.no_grad():  # No gradient computation for inference
            features = text_module(input_ids, attention_mask=attention_mask)
        
        print(f"✓ Forward pass complete")
        print(f"\nOutput:")
        print(f"  - Shape: {features.shape}")
        print(f"  - Data type: {features.dtype}")
        print(f"  - Device: {features.device}")
        print(f"  - Value range: [{features.min().item():.4f}, {features.max().item():.4f}]")
        print(f"  - Mean: {features.mean().item():.4f}")
        print(f"  - Std: {features.std().item():.4f}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    
    # ------------------------------------------------------------------------
    # Example 2: Check Cache After Download
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Example 2] Verify Models are Cached Locally")
    print("-" * 80)
    
    print(f"\nChecking cache directory: {TEXT_MODEL_CACHE_DIR}")
    
    cached_items = list_cached_text_models()
    cache_size = get_text_cache_size()
    
    print(f"\nCached items: {len(cached_items)}")
    if cached_items:
        for item in cached_items[:5]:
            print(f"  ✓ {item}")
        if len(cached_items) > 5:
            print(f"  ... and {len(cached_items) - 5} more")
        print(f"\nTotal cache size: {cache_size:.2f} MB")
        print(f"\n✓ Models are now cached locally in: {TEXT_MODEL_CACHE_DIR}")
        print("  Next time you run the code, models will load from cache (much faster!)")
    else:
        print("  (No models cached - this shouldn't happen after Example 1)")
    
    
    # ------------------------------------------------------------------------
    # Example 3: Using with Real Tokenizer
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Example 3] Integration with Real Tokenizer")
    print("-" * 80)
    
    try:
        print("Loading tokenizer and model...")
        print("(Tokenizer will also be cached locally)")
        
        tokenizer = load_tokenizer("bert-base-uncased")
        text_module = TextModule(
            encoder_name="bert-base-uncased",
            out_features=512,
            freeze=False
        )
        print("✓ Loaded successfully\n")
        
        # Example texts
        texts = [
            "This is a great example of text processing.",
            "The model extracts meaningful features from text.",
            "BERT understands context and semantics."
        ]
        
        print("Input texts:")
        for i, text in enumerate(texts, 1):
            print(f"  {i}. {text}")
        
        # Tokenize texts
        print("\nTokenizing...")
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        print(f"Tokenized shapes:")
        print(f"  - input_ids: {encoded['input_ids'].shape}")
        print(f"  - attention_mask: {encoded['attention_mask'].shape}")
        
        # Extract features
        print("\nExtracting features...")
        with torch.no_grad():
            features = text_module(
                encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
        
        print(f"✓ Features extracted. Shape: {features.shape}")
        
        # Compute similarity between texts
        print("\nComputing pairwise cosine similarity:")
        from torch.nn.functional import cosine_similarity
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = cosine_similarity(
                    features[i].unsqueeze(0),
                    features[j].unsqueeze(0)
                ).item()
                print(f"  Text {i+1} ↔ Text {j+1}: {sim:.4f}")
        
        # Show updated cache info
        print("\nUpdated cache information:")
        cache_size = get_text_cache_size()
        print(f"  Total cache size: {cache_size:.2f} MB")
        
    except ImportError:
        print("✗ transformers library not available for this example")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    
    # ------------------------------------------------------------------------
    # Example 4: Different Encoder Architectures
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Example 4] Different Encoder Architectures")
    print("-" * 80)
    
    try:
        # Test just one lightweight model to save time
        encoder = "distilbert-base-uncased"
        
        print(f"Testing {encoder}...")
        print("(This will download if not cached)\n")
        
        module = TextModule(
            encoder_name=encoder,
            out_features=512,
            freeze=True
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in module.parameters())
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 64))
        with torch.no_grad():
            features = module(input_ids)
        
        print(f"\n✓ {encoder}:")
        print(f"  - Parameters: {total_params:,}")
        print(f"  - Output shape: {features.shape}")
        
        # Show cache growth
        print(f"\nCache size after loading {encoder}:")
        cache_size = get_text_cache_size()
        print(f"  Total: {cache_size:.2f} MB")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    
    # ------------------------------------------------------------------------
    # Summary with Cache Information
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODEL CACHE SUMMARY")
    print("=" * 80)
    
    print(f"""
Cache Directory: {TEXT_MODEL_CACHE_DIR}

Structure:
  downloaded_models/
    └── text_models/
        ├── models--bert-base-uncased/
        ├── models--distilbert-base-uncased/
        └── ... (other models and tokenizers)

Current Status:
  - Cached items: {len(list_cached_text_models())}
  - Total size: {get_text_cache_size():.2f} MB

Helper Functions:
  1. get_text_model_cache_directory() - Get cache directory path
  2. list_cached_text_models() - List all cached items
  3. get_text_cache_size() - Get total cache size in MB
  4. clear_text_model_cache(confirm=True) - Delete all cached models
  5. load_tokenizer(name) - Load tokenizer with caching

Usage:
  # Check what's cached
  >>> cached = list_cached_text_models()
  >>> print(f"Cached items: {{cached}}")
  
  # Get cache size
  >>> size = get_text_cache_size()
  >>> print(f"Cache size: {{size:.2f}} MB")
  
  # Load tokenizer (will be cached)
  >>> tokenizer = load_tokenizer("bert-base-uncased")
  
  # Clear cache (be careful!)
  >>> clear_text_model_cache(confirm=True)

Environment Variables Set:
  - TRANSFORMERS_CACHE = {TEXT_MODEL_CACHE_DIR}
  - HF_HOME = {TEXT_MODEL_CACHE_DIR}
  - HF_HUB_CACHE = {TEXT_MODEL_CACHE_DIR}

Notes:
  - Models and tokenizers are downloaded only once
  - Subsequent runs load from cache (much faster)
  - Cache persists between runs
  - Can be shared across different scripts in the same directory
  - Safe to delete - models will re-download if needed
  - HuggingFace models are typically larger than vision models (200MB - 1GB+)
    """)
    
    print("=" * 80)
    print("End of Examples")
    print("=" * 80)