"""
Text Module for Feature Extraction
===================================
This module extracts meaningful features from text using transformer models.
It uses BERT, DistilBERT, RoBERTa, or other transformers to convert text 
into fixed-size feature vectors.

Key Fix: Uses AutoModel to automatically select the correct architecture
for each model type (BERT, DistilBERT, RoBERTa, etc.)

Author: [Your Name]
Date: [Date]
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from pathlib import Path
import os

# Import transformers with proper error handling
try:
    from transformers import (
        AutoModel,      # Automatically selects correct model class
        AutoTokenizer,  # Automatically selects correct tokenizer
        AutoConfig      # For getting model config without loading weights
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers library not installed. Install with: pip install transformers")


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

# Create a folder for downloaded models in the project directory
MODEL_CACHE_DIR = SCRIPT_DIR / "local" / "BERT_MODELS"

# Create subdirectories for different types of models
TEXT_MODEL_CACHE_DIR = MODEL_CACHE_DIR / "text_models"
TEXT_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Text model cache directory: {TEXT_MODEL_CACHE_DIR}")

# Set environment variables to tell HuggingFace where to cache models
os.environ['TRANSFORMERS_CACHE'] = str(TEXT_MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(TEXT_MODEL_CACHE_DIR)
os.environ['HF_HUB_CACHE'] = str(TEXT_MODEL_CACHE_DIR)


# ============================================================================
# SUPPORTED ENCODER CONFIGURATIONS
# ============================================================================

SUPPORTED_ENCODERS: Dict[str, Dict[str, Any]] = {
    # BERT Models
    "bert-base-uncased": {
        "description": "BERT Base Uncased (110M params, good for general text)",
        "hidden_size": 768,
        "max_length": 512,
        "has_pooler": True,
        "model_type": "bert"
    },
    "distilbert-base-uncased": {
        "description": "DistilBERT (66M params, 60% faster, 97% of BERT performance)",
        "hidden_size": 768,
        "max_length": 512,
        "has_pooler": False,  # DistilBERT doesn't have pooler_output
        "model_type": "distilbert"
    },
    # "bert-base-cased": {
    #     "description": "BERT Base Cased (110M params, preserves case)",
    #     "hidden_size": 768,
    #     "max_length": 512,
    #     "has_pooler": True,
    #     "model_type": "bert"
    # },
    # "distilbert-base-cased": {
    #     "description": "DistilBERT Cased (66M params, preserves case)",
    #     "hidden_size": 768,
    #     "max_length": 512,
    #     "has_pooler": False,
    #     "model_type": "distilbert"
    # },
    # "bert-large-uncased": {
    #     "description": "BERT Large (340M params, higher accuracy)",
    #     "hidden_size": 1024,
    #     "max_length": 512,
    #     "has_pooler": True,
    #     "model_type": "bert"
    # },
    
    # DistilBERT Models (Smaller, Faster)
    
    # RoBERTa Models (Improved BERT training)
    # "roberta-base": {
    #     "description": "RoBERTa Base (125M params, improved BERT)",
    #     "hidden_size": 768,
    #     "max_length": 512,
    #     "has_pooler": True,
    #     "model_type": "roberta"
    # },
    # "roberta-large": {
    #     "description": "RoBERTa Large (355M params, highest accuracy)",
    #     "hidden_size": 1024,
    #     "max_length": 512,
    #     "has_pooler": True,
    #     "model_type": "roberta"
    # },
    
    # ALBERT Models (Parameter Efficient)
    # "albert-base-v2": {
    #     "description": "ALBERT Base v2 (12M params, smallest, parameter sharing)",
    #     "hidden_size": 768,
    #     "max_length": 512,
    #     "has_pooler": True,
    #     "model_type": "albert"
    # },
    
    # Smaller/Faster Models
    # "prajjwal1/bert-tiny": {
    #     "description": "BERT Tiny (4.4M params, very fast, lower accuracy)",
    #     "hidden_size": 128,
    #     "max_length": 512,
    #     "has_pooler": True,
    #     "model_type": "bert"
    # },
    # "prajjwal1/bert-mini": {
    #     "description": "BERT Mini (11M params, fast, moderate accuracy)",
    #     "hidden_size": 256,
    #     "max_length": 512,
    #     "has_pooler": True,
    #     "model_type": "bert"
    # },
    # "prajjwal1/bert-small": {
    #     "description": "BERT Small (29M params, balanced speed/accuracy)",
    #     "hidden_size": 512,
    #     "max_length": 512,
    #     "has_pooler": True,
    #     "model_type": "bert"
    # }
}


class TextModule(nn.Module):
    """
    Text Feature Extraction Module
    
    This module processes tokenized text through a transformer model
    and converts it into fixed-size feature vectors suitable for downstream tasks.
    
    Architecture:
    1. Text Tokens → Transformer Encoder → Contextual Embeddings
    2. Extract [CLS] Token (represents entire sentence)
    3. Apply Dropout (for regularization)
    4. Linear Projection → Output Features
    
    Key Features:
    - Uses AutoModel for automatic architecture selection
    - Properly handles different model types (BERT, DistilBERT, RoBERTa, etc.)
    - Caches models locally for faster subsequent loads
    - Comprehensive logging and error handling
    
    Supported Models:
    - bert-base-uncased, bert-base-cased, bert-large-uncased
    - distilbert-base-uncased, distilbert-base-cased
    - roberta-base, roberta-large
    - albert-base-v2
    - prajjwal1/bert-tiny, bert-mini, bert-small (smaller/faster)
    """
    
    def __init__(
        self, 
        encoder_name: str = "distilbert-base-uncased", 
        out_features: int = 256, 
        freeze: bool = False,
        dropout_rate: float = 0.3
    ):
        """
        Initialize the Text Module with a transformer encoder.
        
        Args:
            encoder_name (str): Name of the pre-trained transformer model.
                Options include:
                - 'bert-base-uncased' (110M params, good balance)
                - 'distilbert-base-uncased' (66M params, faster) [RECOMMENDED]
                - 'roberta-base' (125M params, slightly better)
                - 'albert-base-v2' (12M params, smallest)
                - 'prajjwal1/bert-tiny' (4.4M params, fastest)
                Default: 'distilbert-base-uncased'
            
            out_features (int): Dimension of the output feature vector.
                This is the size of features passed to fusion module.
                Common values: 128, 256, 512
                Default: 256
            
            freeze (bool): Whether to freeze the transformer weights.
                True = Don't update transformer during training (faster, less memory)
                False = Fine-tune transformer weights (better performance, slower)
                Default: False
            
            dropout_rate (float): Dropout probability for regularization.
                Higher values (0.3-0.5) help prevent overfitting.
                Default: 0.3
        
        Raises:
            ImportError: If transformers library is not installed
            ValueError: If parameters are invalid
            RuntimeError: If model loading fails
        """
        super(TextModule, self).__init__()
        
        # Check if transformers is available
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required. "
                "Install with: pip install transformers"
            )
        
        # ====================================================================
        # STEP 1: Log initialization
        # ====================================================================
        logger.info("=" * 60)
        logger.info("Initializing Text Module")
        logger.info("=" * 60)
        logger.info(f"Encoder: {encoder_name}")
        logger.info(f"Output Features: {out_features}")
        logger.info(f"Freeze Backbone: {freeze}")
        logger.info(f"Dropout Rate: {dropout_rate}")
        logger.info(f"Model Cache Directory: {TEXT_MODEL_CACHE_DIR}")
        
        # ====================================================================
        # STEP 2: Validate parameters
        # ====================================================================
        self._validate_parameters(encoder_name, out_features, freeze, dropout_rate)
        
        # ====================================================================
        # STEP 3: Store configuration
        # ====================================================================
        self.encoder_name = encoder_name
        self.out_features = out_features
        self.freeze = freeze
        self.dropout_rate = dropout_rate
        self.model_cache_dir = TEXT_MODEL_CACHE_DIR
        
        # Determine if this model has a pooler output
        if encoder_name in SUPPORTED_ENCODERS:
            self.has_pooler = SUPPORTED_ENCODERS[encoder_name].get("has_pooler", True)
            self.model_type = SUPPORTED_ENCODERS[encoder_name].get("model_type", "bert")
        else:
            # For unknown models, assume no pooler (safer)
            self.has_pooler = False
            self.model_type = "unknown"
            logger.warning(f"Unknown model type for '{encoder_name}', assuming no pooler output")
        
        # ====================================================================
        # STEP 4: Load pre-trained transformer model using AutoModel
        # ====================================================================
        try:
            logger.info(f"Loading pre-trained model: {encoder_name}")
            logger.info(f"Models will be downloaded/cached to: {TEXT_MODEL_CACHE_DIR}")
            
            # Check if model might be cached
            self._check_cached_models()
            
            # ================================================================
            # KEY FIX: Use AutoModel instead of BertModel
            # AutoModel automatically selects the correct architecture:
            # - BertModel for bert-*
            # - DistilBertModel for distilbert-*
            # - RobertaModel for roberta-*
            # - AlbertModel for albert-*
            # ================================================================
            self.transformer = AutoModel.from_pretrained(
                encoder_name,
                cache_dir=str(TEXT_MODEL_CACHE_DIR)
            )
            
            # Log successful load
            model_class_name = type(self.transformer).__name__
            logger.info(f"✓ Loaded as {model_class_name}")
            
            # Get hidden size from the loaded model's config
            self.hidden_size = self.transformer.config.hidden_size
            logger.info(f"Model hidden size: {self.hidden_size}")
            
            # Count and log parameters
            total_params = sum(p.numel() for p in self.transformer.parameters())
            logger.info(f"Model parameters: {total_params:,}")
            logger.info(f"✓ Model cached in: {TEXT_MODEL_CACHE_DIR}")
            
        except Exception as e:
            error_msg = f"Failed to load transformer model '{encoder_name}': {str(e)}"
            logger.error(error_msg)
            logger.error("Please check:")
            logger.error("  1. Model name is correct")
            logger.error("  2. Internet connection is available")
            logger.error("  3. Sufficient disk space for model download")
            logger.error(f"  4. Cache directory is writable: {TEXT_MODEL_CACHE_DIR}")
            raise RuntimeError(error_msg)
        
        # ====================================================================
        # STEP 5: Freeze transformer weights if requested
        # ====================================================================
        if freeze:
            logger.info("Freezing transformer backbone weights...")
            frozen_params = 0
            
            for param in self.transformer.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            
            logger.info(f"✓ Frozen {frozen_params:,} parameters")
            logger.info("Note: Transformer weights will not be updated during training")
        else:
            trainable_params = sum(
                p.numel() for p in self.transformer.parameters() if p.requires_grad
            )
            logger.info(f"Transformer is trainable with {trainable_params:,} parameters")
        
        # ====================================================================
        # STEP 6: Create dropout layer
        # ====================================================================
        self.dropout = nn.Dropout(dropout_rate)
        logger.info(f"Added dropout layer with rate: {dropout_rate}")
        
        # ====================================================================
        # STEP 7: Create projection layer
        # ====================================================================
        try:
            self.projection = nn.Linear(self.hidden_size, out_features)
            logger.info(f"Created projection layer: {self.hidden_size} → {out_features}")
            
            # Initialize weights with Xavier uniform for better training
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            logger.info("✓ Initialized projection layer with Xavier uniform")
            
        except Exception as e:
            error_msg = f"Failed to create projection layer: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # ====================================================================
        # STEP 8: Log final statistics
        # ====================================================================
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logger.info("=" * 60)
        logger.info("Text Module Initialization Complete")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info(f"Frozen Parameters: {frozen_params:,}")
        logger.info("=" * 60)
    
    
    def _validate_parameters(
        self, 
        encoder_name: str, 
        out_features: int, 
        freeze: bool,
        dropout_rate: float
    ) -> None:
        """
    
        Validate initialization parameters.
    
        Args:
            encoder_name: Name of the encoder to validate
            out_features: Output feature dimension to validate
            freeze: Freeze flag to validate
            dropout_rate: Dropout rate to validate
        
        Raises:
            TypeError: If parameters have wrong type
            ValueError: If parameters have invalid values
        """
        # Validate encoder_name
        if not isinstance(encoder_name, str):
            raise TypeError(f"encoder_name must be a string, got {type(encoder_name)}")
        
        if not encoder_name.strip():
            raise ValueError("encoder_name cannot be empty")
        
        # Log info about known encoders
        if encoder_name in SUPPORTED_ENCODERS:
            info = SUPPORTED_ENCODERS[encoder_name]
            logger.info(f"Encoder info: {info['description']}")
            logger.info(f"Hidden size: {info['hidden_size']}")
            logger.info(f"Max sequence length: {info['max_length']}")
            logger.info(f"Has pooler output: {info['has_pooler']}")
        else:
            logger.warning(f"Encoder '{encoder_name}' is not in the list of known encoders")
            logger.warning("It may still work if it's a valid HuggingFace model")
            logger.info(f"Known encoders: {list(SUPPORTED_ENCODERS.keys())}")
        
        # Validate out_features
        if not isinstance(out_features, int):
            raise TypeError(f"out_features must be an integer, got {type(out_features)}")
        
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        
        if out_features > 4096:
            logger.warning(f"out_features={out_features} is unusually large")
        
        if out_features < 64:
            logger.warning(f"out_features={out_features} is quite small, may lose information")
        
        # Validate freeze
        if not isinstance(freeze, bool):
            raise TypeError(f"freeze must be a boolean, got {type(freeze)}")
        
        # Validate dropout_rate
        if not isinstance(dropout_rate, (int, float)):
            raise TypeError(f"dropout_rate must be a number, got {type(dropout_rate)}")
        
        if not 0 <= dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        
        logger.debug("✓ Parameter validation passed")
    
    
    def _check_cached_models(self) -> None:
        """Check if models are already cached and log information."""
        if TEXT_MODEL_CACHE_DIR.exists():
            cached_items = list(TEXT_MODEL_CACHE_DIR.iterdir())
            if cached_items:
                logger.info(f"Found {len(cached_items)} item(s) in cache directory")
                
                # Check for model-specific directories
                model_dirs = [
                    d for d in cached_items 
                    if d.is_dir() and ('model' in d.name.lower() or 'hub' in d.name.lower())
                ]
                if model_dirs:
                    logger.info(f"Found {len(model_dirs)} model cache director(ies)")
                    logger.info("Model may load from cache (faster)")
            else:
                logger.info("Cache directory is empty - model will be downloaded")
        else:
            logger.info("Cache directory doesn't exist - model will be downloaded")
    
    
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
            input_ids (torch.Tensor): Tokenized text input.
                Shape: [batch_size, sequence_length]
                Each value is a token ID (integer).
            
            attention_mask (torch.Tensor, optional): Mask to ignore padding tokens.
                Shape: [batch_size, sequence_length]
                1 = real token, 0 = padding token.
                Default: None (all tokens treated as real)
        
        Returns:
            torch.Tensor: Extracted text features.
                Shape: [batch_size, out_features]
        
        Raises:
            ValueError: If input tensors have invalid shapes
            RuntimeError: If forward pass fails
        """
        try:
            # Step 1: Validate inputs
            logger.debug(f"Processing text batch. Input shape: {input_ids.shape}")
            self._validate_forward_inputs(input_ids, attention_mask)
            
            # Step 2: Pass through transformer encoder
            logger.debug("Passing input through transformer encoder...")
            
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logger.debug("✓ Transformer encoding complete")
            
            # Step 3: Extract sentence representation
            # ================================================================
            # KEY FIX: Handle different model types correctly
            # 
            # - BERT, RoBERTa, ALBERT: Have pooler_output (processed [CLS])
            # - DistilBERT: No pooler_output, use raw [CLS] token
            # ================================================================
            
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                # Models with pooler (BERT, RoBERTa, ALBERT)
                # pooler_output is the [CLS] token passed through a linear layer + tanh
                pooled_output = outputs.pooler_output
                logger.debug("Using pooler_output for sentence representation")
            else:
                # Models without pooler (DistilBERT)
                # Use the raw [CLS] token (first token) from last hidden state
                # last_hidden_state shape: [batch_size, seq_len, hidden_size]
                pooled_output = outputs.last_hidden_state[:, 0, :]
                logger.debug("Using last_hidden_state[:, 0] (CLS token) for sentence representation")
            
            logger.debug(f"Pooled output shape: {pooled_output.shape}")
            
            # Step 4: Apply dropout for regularization
            pooled_output = self.dropout(pooled_output)
            logger.debug("Applied dropout regularization")
            
            # Step 5: Project to output dimension
            output_features = self.projection(pooled_output)
            logger.debug(f"Projected to output features. Shape: {output_features.shape}")
            
            # Step 6: Validate output
            batch_size = input_ids.size(0)
            expected_shape = (batch_size, self.out_features)
            
            if output_features.shape != expected_shape:
                raise RuntimeError(
                    f"Unexpected output shape: {output_features.shape}, "
                    f"expected: {expected_shape}"
                )
            
            # Check for numerical issues
            if torch.isnan(output_features).any():
                raise RuntimeError("NaN detected in output features")
            
            if torch.isinf(output_features).any():
                raise RuntimeError("Inf detected in output features")
            
            logger.debug(f"✓ Text feature extraction complete. Output: {output_features.shape}")
            return output_features
            
        except Exception as e:
            error_msg = f"Text module forward pass failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    
    def _validate_forward_inputs(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]
    ) -> None:
        """Validate inputs for the forward pass."""
        # Validate input_ids type
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids must be torch.Tensor, got {type(input_ids)}")
        
        # Validate input_ids dimensions (should be 2D)
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D [batch_size, seq_len], got shape {input_ids.shape}"
            )
        
        batch_size, seq_length = input_ids.shape
        
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        
        if seq_length <= 0:
            raise ValueError(f"Sequence length must be positive, got {seq_length}")
        
        # Check max sequence length
        max_seq_length = getattr(self.transformer.config, 'max_position_embeddings', 512)
        if seq_length > max_seq_length:
            logger.warning(
                f"Sequence length {seq_length} exceeds model's maximum {max_seq_length}"
            )
        
        # Validate input_ids values
        if (input_ids < 0).any():
            raise ValueError("input_ids contains negative values (token IDs must be non-negative)")
        
        # Validate attention_mask if provided
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError(
                    f"attention_mask must be torch.Tensor, got {type(attention_mask)}"
                )
            
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    f"attention_mask shape {attention_mask.shape} doesn't match "
                    f"input_ids shape {input_ids.shape}"
                )
        
        logger.debug("✓ Forward input validation passed")
    
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including:
                - encoder_name: Name of the encoder
                - model_class: Class name of the loaded model
                - hidden_size: Hidden dimension of the transformer
                - out_features: Output feature dimension
                - frozen: Whether backbone is frozen
                - total_params: Total number of parameters
                - trainable_params: Number of trainable parameters
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'encoder_name': self.encoder_name,
            'model_class': type(self.transformer).__name__,
            'model_type': self.model_type,
            'hidden_size': self.hidden_size,
            'out_features': self.out_features,
            'has_pooler': self.has_pooler,
            'frozen': self.freeze,
            'dropout_rate': self.dropout_rate,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params
        }
    
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached models.
        
        Returns:
            dict: Cache information including directory path and size
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
                    for subitem in item.rglob('*'):
                        if subitem.is_file():
                            total_size += subitem.stat().st_size
            
            info['total_size_mb'] = total_size / (1024 * 1024)
        
        return info


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_text_model_cache_directory() -> Path:
    """Get the path to the text model cache directory."""
    return TEXT_MODEL_CACHE_DIR


def list_cached_text_models() -> List[str]:
    """List all cached text model items in the download directory."""
    if not TEXT_MODEL_CACHE_DIR.exists():
        return []
    return [item.name for item in TEXT_MODEL_CACHE_DIR.iterdir()]


def get_text_cache_size() -> float:
    """Get the total size of cached text models in MB."""
    if not TEXT_MODEL_CACHE_DIR.exists():
        return 0.0
    
    total_size = 0
    for item in TEXT_MODEL_CACHE_DIR.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size
    
    return total_size / (1024 * 1024)


def list_supported_encoders() -> Dict[str, str]:
    """
    List all supported encoder models with their descriptions.
    
    Returns:
        dict: Mapping of encoder names to their descriptions
    """
    return {
        name: info['description'] 
        for name, info in SUPPORTED_ENCODERS.items()
    }


def load_tokenizer(encoder_name: str = "distilbert-base-uncased"):
    """
    Load a tokenizer with caching to the local directory.
    
    Args:
        encoder_name: Name of the encoder/tokenizer to load
    
    Returns:
        Tokenizer: HuggingFace tokenizer instance
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required")
    
    logger.info(f"Loading tokenizer: {encoder_name}")
    logger.info(f"Tokenizer will be cached to: {TEXT_MODEL_CACHE_DIR}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_name,
        cache_dir=str(TEXT_MODEL_CACHE_DIR)
    )
    
    logger.info(f"✓ Loaded {type(tokenizer).__name__}")
    return tokenizer


def clear_text_model_cache(confirm: bool = False) -> bool:
    """
    Clear all cached text models from the download directory.
    
    Args:
        confirm: Must be True to actually delete files (safety measure)
    
    Returns:
        bool: True if cache was cleared, False if not confirmed
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


# ============================================================================
# TEST AND EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """Test the TextModule with different configurations."""
    
    print("=" * 80)
    print("TEXT MODULE - Test Suite")
    print("=" * 80)
    
    # Test 1: List supported encoders
    print("\n[Test 1] Supported Encoders")
    print("-" * 80)
    encoders = list_supported_encoders()
    for name, description in encoders.items():
        print(f"  • {name}")
        print(f"    {description}")
    
    # Test 2: Load DistilBERT (most common use case)
    print("\n[Test 2] Load DistilBERT (Recommended for 10K samples)")
    print("-" * 80)
    
    try:
        text_module = TextModule(
            encoder_name="distilbert-base-uncased",
            out_features=128,
            freeze=True,
            dropout_rate=0.5
        )
        
        # Print model info
        info = text_module.get_model_info()
        print(f"\nModel Information:")
        print(f"  • Encoder: {info['encoder_name']}")
        print(f"  • Model Class: {info['model_class']}")
        print(f"  • Model Type: {info['model_type']}")
        print(f"  • Hidden Size: {info['hidden_size']}")
        print(f"  • Output Features: {info['out_features']}")
        print(f"  • Has Pooler: {info['has_pooler']}")
        print(f"  • Frozen: {info['frozen']}")
        print(f"  • Total Params: {info['total_params']:,}")
        print(f"  • Trainable Params: {info['trainable_params']:,}")
        print(f"  • Frozen Params: {info['frozen_params']:,}")
        
        # Test forward pass
        print("\n[Test 3] Forward Pass")
        print("-" * 80)
        
        batch_size = 4
        seq_length = 64
        
        input_ids = torch.randint(0, 30000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        
        with torch.no_grad():
            features = text_module(input_ids, attention_mask)
        
        print(f"✓ Output shape: {features.shape}")
        print(f"  Output dtype: {features.dtype}")
        print(f"  Value range: [{features.min().item():.4f}, {features.max().item():.4f}]")
        
        # Test with real tokenizer
        print("\n[Test 4] Integration with Tokenizer")
        print("-" * 80)
        
        tokenizer = load_tokenizer("distilbert-base-uncased")
        
        texts = [
            "This is a great product!",
            "I'm not happy with this purchase.",
            "The quality is average."
        ]
        
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        
        print(f"Tokenized {len(texts)} texts")
        print(f"  input_ids shape: {encoded['input_ids'].shape}")
        print(f"  attention_mask shape: {encoded['attention_mask'].shape}")
        
        with torch.no_grad():
            features = text_module(
                encoded['input_ids'],
                encoded['attention_mask']
            )
        
        print(f"✓ Extracted features shape: {features.shape}")
        
        # Cache info
        print("\n[Test 5] Cache Information")
        print("-" * 80)
        
        cache_info = text_module.get_cache_info()
        print(f"  Cache directory: {cache_info['cache_dir']}")
        print(f"  Cache exists: {cache_info['exists']}")
        print(f"  Cached items: {len(cache_info['cached_items'])}")
        print(f"  Total size: {cache_info['total_size_mb']:.2f} MB")
        
        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()