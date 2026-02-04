"""
Visual Module for Image Feature Extraction
===========================================
This module extracts meaningful features from images using CNN backbones.
It supports various pre-trained models like ResNet, and can be extended
to support other architectures.

Author: [Your Name]
Date: [Date]
"""

import logging
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple, List
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
MODEL_CACHE_DIR = SCRIPT_DIR / "local" / "RESNET_MODELS"

# Create the directory if it doesn't exist
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Model cache directory: {MODEL_CACHE_DIR}")

# Set environment variable to tell PyTorch where to cache models
# This makes torchvision download models to our custom directory
os.environ['TORCH_HOME'] = str(MODEL_CACHE_DIR)


# Also set the hub directory (used by newer versions)
os.environ['TORCH_HUB'] = str(MODEL_CACHE_DIR / 'hub')


# Supported backbone architectures
# This makes it easy to add new backbones in the future
SUPPORTED_BACKBONES = {
    "resnet18": {
        "model_fn": models.resnet18,
        "description": "ResNet-18 (11M params, faster, less accurate)"
    },
    "resnet34": {
        "model_fn": models.resnet34,
        "description": "ResNet-34 (21M params, balanced)"
    },
    "resnet50": {
        "model_fn": models.resnet50,
        "description": "ResNet-50 (25M params, good accuracy)"
    },
    "resnet101": {
        "model_fn": models.resnet101,
        "description": "ResNet-101 (44M params, high accuracy)"
    },
    "resnet152": {
        "model_fn": models.resnet152,
        "description": "ResNet-152 (60M params, highest accuracy, slower)"
    }
}


class VisualModule(nn.Module):
    """
    Visual Feature Extraction Module
    
    This module processes images through a CNN backbone (like ResNet)
    and extracts fixed-size feature vectors suitable for downstream tasks.
    
    Architecture:
    1. Input Image → CNN Backbone → Feature Maps
    2. Global Average Pooling → Fixed-size Features
    3. Fully Connected Layer → Output Features
    
    The backbone can be frozen to use only pre-trained features,
    or fine-tuned for better performance on specific tasks.
    
    All pre-trained models are downloaded to: downloaded_models/
    """
    
    def __init__(
        self, 
        backbone: str = "resnet18", 
        pretrained: bool = True,
        out_features: int = 512, 
        freeze: bool = False
    ):
        """
        Initialize the Visual Module with a CNN backbone.
        
        Args:
            backbone (str): Name of the CNN architecture to use
                           Supported: 'resnet18', 'resnet34', 'resnet50', 
                                     'resnet101', 'resnet152'
                           Default: 'resnet50'
            
            pretrained (bool): Whether to load ImageNet pre-trained weights
                              True = Use pre-trained weights (recommended)
                              False = Random initialization (train from scratch)
                              Default: True
            
            out_features (int): Dimension of the output feature vector
                               This is the size of features passed to fusion module
                               Default: 512
            
            freeze (bool): Whether to freeze the backbone weights
                          True = Don't update backbone during training (faster)
                          False = Fine-tune backbone weights (better performance)
                          Default: False
        
        Raises:
            ValueError: If parameters are invalid
            NotImplementedError: If backbone is not supported
            RuntimeError: If model loading fails
        
        Note:
            Pre-trained models will be downloaded to: downloaded_models/
            The download happens only once, then cached for future use.
        
        Example:
            >>> # Create with default ResNet-50
            >>> visual_module = VisualModule()
            
            >>> # Create with frozen ResNet-101
            >>> visual_module = VisualModule(
            ...     backbone="resnet101",
            ...     pretrained=True,
            ...     out_features=1024,
            ...     freeze=True
            ... )
        """
        super(VisualModule, self).__init__()
        
        # Step 1: Log initialization
        logger.info("=" * 60)
        logger.info("Initializing Visual Module")
        logger.info("=" * 60)
        logger.info(f"Backbone: {backbone}")
        logger.info(f"Pretrained: {pretrained}")
        logger.info(f"Output Features: {out_features}")
        logger.info(f"Freeze Backbone: {freeze}")
        logger.info(f"Model Cache Directory: {MODEL_CACHE_DIR}")
        
        # Step 2: Validate parameters
        self._validate_parameters(backbone, pretrained, out_features, freeze)
        
        # Step 3: Store configuration
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.out_features = out_features
        self.freeze = freeze
        self.model_cache_dir = MODEL_CACHE_DIR
        
        # Step 4: Load the backbone model
        # This is the main CNN that extracts visual features
        try:
            logger.info(f"Loading {backbone} backbone...")
            if pretrained:
                logger.info(f"Models will be downloaded/cached to: {MODEL_CACHE_DIR}")
            self.model = self._load_backbone(backbone, pretrained)
            logger.info("✓ Backbone loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load backbone '{backbone}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Step 5: Freeze backbone if requested
        # Freezing means the weights won't be updated during training
        # This is useful for:
        # - Faster training
        # - Less memory usage
        # - Preventing overfitting on small datasets
        # - Using as a feature extractor only
        if freeze:
            logger.info("Freezing backbone weights...")
            frozen_params = self._freeze_backbone()
            logger.info(f"✓ Frozen {frozen_params:,} parameters")
            logger.info("Note: Only the final projection layer will be trained")
        else:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Backbone is trainable with {trainable_params:,} parameters")
        
        # Step 6: Replace the final classification layer
        # Original: designed for 1000 ImageNet classes
        # Modified: outputs features of desired dimension
        try:
            logger.info("Replacing final classification layer...")
            self._replace_final_layer(out_features)
            logger.info(f"✓ Final layer replaced: ? → {out_features}")
            
        except Exception as e:
            error_msg = f"Failed to replace final layer: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Step 7: Log total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info("=" * 60)
        logger.info("Visual Module Initialization Complete")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info(f"Frozen Parameters: {total_params - trainable_params:,}")
        logger.info("=" * 60)
    
    
    def _validate_parameters(
        self,
        backbone: str,
        pretrained: bool,
        out_features: int,
        freeze: bool
    ) -> None:
        """
        Validate initialization parameters.
        
        This method checks if all parameters are valid before proceeding
        with model initialization. It's better to catch errors early.
        
        Args:
            backbone (str): Backbone name to validate
            pretrained (bool): Pretrained flag to validate
            out_features (int): Output feature dimension to validate
            freeze (bool): Freeze flag to validate
        
        Raises:
            TypeError: If parameters have wrong type
            ValueError: If parameters have invalid values
            NotImplementedError: If backbone is not supported
        """
        # Validate backbone
        if not isinstance(backbone, str):
            error_msg = f"backbone must be a string, got {type(backbone)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        if not backbone.strip():
            error_msg = "backbone cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check if backbone is supported
        if backbone not in SUPPORTED_BACKBONES:
            error_msg = f"Backbone '{backbone}' is not supported. "
            error_msg += f"Supported backbones: {list(SUPPORTED_BACKBONES.keys())}"
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
        
        # Validate pretrained
        if not isinstance(pretrained, bool):
            error_msg = f"pretrained must be a boolean, got {type(pretrained)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Validate out_features
        if not isinstance(out_features, int):
            error_msg = f"out_features must be an integer, got {type(out_features)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        if out_features <= 0:
            error_msg = f"out_features must be positive, got {out_features}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Sanity check for reasonable feature dimension
        if out_features > 4096:
            logger.warning(f"out_features={out_features} is unusually large. Are you sure?")
        
        if out_features < 64:
            logger.warning(f"out_features={out_features} is quite small. This may lose information.")
        
        # Validate freeze
        if not isinstance(freeze, bool):
            error_msg = f"freeze must be a boolean, got {type(freeze)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Warning if using freeze without pretrained weights
        if freeze and not pretrained:
            logger.warning("freeze=True with pretrained=False: freezing random weights!")
            logger.warning("This is unusual. Consider using pretrained=True with freeze=True")
        
        logger.debug("✓ Parameter validation passed")
    
    
    def _load_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """
        Load the CNN backbone model.
        
        This method loads the specified backbone architecture with or
        without pre-trained weights. Models are downloaded to the
        downloaded_models/ directory.
        
        Args:
            backbone (str): Name of the backbone to load
            pretrained (bool): Whether to load pre-trained weights
        
        Returns:
            nn.Module: The loaded backbone model
        
        Raises:
            NotImplementedError: If backbone is not in SUPPORTED_BACKBONES
            RuntimeError: If model loading fails
        
        Note:
            Pre-trained models are downloaded to: downloaded_models/
            Once downloaded, they are cached for future use.
        """
        try:
            # Get the model loading function from our supported backbones
            if backbone not in SUPPORTED_BACKBONES:
                raise NotImplementedError(f"Backbone '{backbone}' is not supported")
            
            model_fn = SUPPORTED_BACKBONES[backbone]["model_fn"]
            description = SUPPORTED_BACKBONES[backbone]["description"]
            
            logger.info(f"Architecture: {description}")
            
            # Ensure TORCH_HOME is set to our custom directory
            # This is redundant but ensures it's set even if called directly
            os.environ['TORCH_HOME'] = str(MODEL_CACHE_DIR)
            
            # Check if model files exist
            checkpoints_dir = MODEL_CACHE_DIR / 'hub' / 'checkpoints'
            if pretrained and checkpoints_dir.exists():
                existing_models = list(checkpoints_dir.glob('*.pth'))
                if existing_models:
                    logger.info(f"Found {len(existing_models)} cached model(s) in {checkpoints_dir}")
            
            # Load the model
            # For newer versions of torchvision, use 'weights' parameter
            # For older versions, use 'pretrained' parameter
            try:
                if pretrained:
                    # Try new API first (torchvision >= 0.13)
                    logger.debug("Attempting to load with new torchvision API...")
                    logger.info(f"Downloading/loading pre-trained weights to: {MODEL_CACHE_DIR}")
                    weights = "DEFAULT"  # Use default pre-trained weights
                    model = model_fn(weights=weights)
                    logger.debug("✓ Loaded with weights parameter (new API)")
                else:
                    model = model_fn(weights=None)
                    logger.debug("✓ Loaded without pre-trained weights (new API)")
                    
            except TypeError:
                # Fall back to old API (torchvision < 0.13)
                logger.debug("Falling back to old torchvision API...")
                if pretrained:
                    logger.info(f"Downloading/loading pre-trained weights to: {MODEL_CACHE_DIR}")
                model = model_fn(pretrained=pretrained)
                logger.debug("✓ Loaded with pretrained parameter (old API)")
            
            # Log model information
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model parameters: {total_params:,}")
            
            if pretrained:
                logger.info("✓ Loaded ImageNet pre-trained weights")
                # Log where the model was saved
                checkpoints_dir = MODEL_CACHE_DIR / 'hub' / 'checkpoints'
                if checkpoints_dir.exists():
                    logger.info(f"✓ Model cached in: {checkpoints_dir}")
            else:
                logger.info("✓ Initialized with random weights")
            
            return model
            
        except Exception as e:
            error_msg = f"Error loading backbone: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    
    def _freeze_backbone(self) -> int:
        """
        Freeze all parameters in the backbone.
        
        This method sets requires_grad=False for all backbone parameters,
        preventing them from being updated during training. The final
        projection layer will remain trainable.
        
        Returns:
            int: Number of parameters that were frozen
        
        Note:
            This is called during __init__ if freeze=True
        """
        frozen_count = 0
        
        try:
            # Freeze all parameters in the model
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                frozen_count += param.numel()
                logger.debug(f"Frozen parameter: {name} ({param.numel():,} values)")
            
            logger.debug(f"Total frozen parameters: {frozen_count:,}")
            return frozen_count
            
        except Exception as e:
            error_msg = f"Error freezing backbone: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    
    def _replace_final_layer(self, out_features: int) -> None:
        """
        Replace the final classification layer with a new projection layer.
        
        ResNet models have a final 'fc' (fully connected) layer that outputs
        1000 classes for ImageNet. We replace this with our own layer that
        outputs the desired feature dimension.
        
        Args:
            out_features (int): Desired output feature dimension
        
        Raises:
            RuntimeError: If final layer replacement fails
        
        Note:
            The new layer is always trainable, even if backbone is frozen.
            This allows learning task-specific features.
        """
        try:
            # For ResNet models, the final layer is called 'fc'
            if hasattr(self.model, 'fc'):
                # Get the input dimension of the current final layer
                in_features = self.model.fc.in_features
                logger.info(f"Original final layer: {in_features} → 1000 (ImageNet classes)")
                
                # Create new linear layer
                # This layer will project from backbone features to our desired dimension
                new_fc = nn.Linear(in_features, out_features)
                
                # Initialize weights properly for better training
                # Xavier/Glorot initialization helps with gradient flow
                nn.init.xavier_uniform_(new_fc.weight)
                nn.init.zeros_(new_fc.bias)
                logger.debug("✓ Initialized new layer with Xavier uniform")
                
                # Replace the final layer
                self.model.fc = new_fc
                
                # Ensure the new layer is trainable
                # (Important when backbone is frozen)
                for param in new_fc.parameters():
                    param.requires_grad = True
                
                trainable_params = sum(p.numel() for p in new_fc.parameters())
                logger.info(f"New final layer: {in_features} → {out_features}")
                logger.info(f"New layer parameters (always trainable): {trainable_params:,}")
                
            else:
                # This shouldn't happen with ResNet, but handle it
                error_msg = "Model doesn't have 'fc' attribute. Cannot replace final layer."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            error_msg = f"Error replacing final layer: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract visual features from images.
        
        This method processes images through the CNN backbone and
        produces fixed-size feature vectors.
        
        Args:
            x (torch.Tensor): Batch of images
                             Shape: [batch_size, channels, height, width]
                             Channels: typically 3 (RGB)
                             Height/Width: typically 224x224 for ResNet
                             
                             Example shapes:
                             - [32, 3, 224, 224] = 32 RGB images of 224x224
                             - [1, 3, 256, 256] = 1 RGB image of 256x256
        
        Returns:
            torch.Tensor: Extracted visual features
                         Shape: [batch_size, out_features]
                         These features encode the visual content
                         
                         Example: [32, 512] = 512-dim features for 32 images
        
        Raises:
            ValueError: If input tensor has invalid shape
            RuntimeError: If forward pass fails
        
        Example:
            >>> visual_module = VisualModule(out_features=512)
            >>> images = torch.randn(16, 3, 224, 224)  # 16 images
            >>> features = visual_module(images)
            >>> print(features.shape)  # torch.Size([16, 512])
        """
        try:
            # Step 1: Validate input
            logger.debug(f"Processing visual input. Shape: {x.shape}")
            self._validate_forward_input(x)
            
            # Step 2: Pass through the CNN backbone
            # The backbone extracts hierarchical visual features
            # Early layers detect edges, colors
            # Middle layers detect textures, patterns
            # Late layers detect objects, concepts
            try:
                output = self.model(x)
                logger.debug(f"Backbone forward pass complete. Output shape: {output.shape}")
                
            except Exception as e:
                error_msg = f"Backbone forward pass failed: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Input shape: {x.shape}")
                logger.error(f"Input dtype: {x.dtype}")
                logger.error(f"Input device: {x.device}")
                logger.error(f"Input value range: [{x.min().item():.4f}, {x.max().item():.4f}]")
                raise RuntimeError(error_msg)
            
            # Step 3: Validate output
            batch_size = x.size(0)
            expected_shape = (batch_size, self.out_features)
            
            if output.shape != expected_shape:
                error_msg = f"Unexpected output shape: {output.shape}, expected: {expected_shape}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Step 4: Check for NaN or Inf values
            # These indicate numerical instability
            if torch.isnan(output).any():
                error_msg = "NaN detected in output features"
                logger.error(error_msg)
                logger.error(f"Input had NaN: {torch.isnan(x).any()}")
                raise RuntimeError(error_msg)
            
            if torch.isinf(output).any():
                error_msg = "Inf detected in output features"
                logger.error(error_msg)
                logger.error(f"Input had Inf: {torch.isinf(x).any()}")
                raise RuntimeError(error_msg)
            
            logger.debug(f"✓ Visual feature extraction complete. Output: {output.shape}")
            return output
            
        except Exception as e:
            error_msg = f"Visual module forward pass failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    
    def _validate_forward_input(self, x: torch.Tensor) -> None:
        """
        Validate input tensor for the forward pass.
        
        This method ensures that the input image tensor has the correct
        shape, type, and values before processing.
        
        Args:
            x (torch.Tensor): Input image tensor to validate
        
        Raises:
            TypeError: If input is not a tensor
            ValueError: If input has invalid shape or values
        """
        # Validate type
        if not isinstance(x, torch.Tensor):
            error_msg = f"Input must be torch.Tensor, got {type(x)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Validate dimensions
        # Expected: 4D tensor [batch_size, channels, height, width]
        if x.dim() != 4:
            error_msg = f"Input must be 4D [batch, channels, height, width], got shape {x.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Extract dimensions
        batch_size, channels, height, width = x.shape
        
        # Validate batch size
        if batch_size <= 0:
            error_msg = f"Batch size must be positive, got {batch_size}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate number of channels
        # Most models expect 3 channels (RGB)
        if channels != 3:
            logger.warning(f"Expected 3 channels (RGB), got {channels}")
            logger.warning("Model may not work correctly with non-RGB images")
        
        # Validate spatial dimensions
        if height <= 0 or width <= 0:
            error_msg = f"Image dimensions must be positive, got height={height}, width={width}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check if dimensions are too small
        min_size = 32
        if height < min_size or width < min_size:
            logger.warning(f"Image size {height}x{width} is very small (minimum recommended: {min_size}x{min_size})")
            logger.warning("This may result in poor feature extraction")
        
        # Check if dimensions are unusual
        # ResNet is typically trained on 224x224 images
        expected_size = 224
        if height != expected_size or width != expected_size:
            logger.debug(f"Input size {height}x{width} differs from typical {expected_size}x{expected_size}")
            logger.debug("The model will resize internally, but aspect ratio may change")
        
        # Validate data type
        # Models typically expect float32
        if x.dtype not in [torch.float32, torch.float16, torch.float64]:
            logger.warning(f"Input dtype is {x.dtype}, expected float32")
            logger.warning("Consider converting to float: tensor.float()")
        
        # Check value range
        # Typically images are normalized to [0, 1] or [-1, 1]
        min_val = x.min().item()
        max_val = x.max().item()
        
        if min_val < -10 or max_val > 10:
            logger.warning(f"Unusual value range: [{min_val:.4f}, {max_val:.4f}]")
            logger.warning("Images are typically normalized to [0, 1] or [-1, 1]")
        
        # Check for NaN or Inf in input
        if torch.isnan(x).any():
            error_msg = "Input contains NaN values"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if torch.isinf(x).any():
            error_msg = "Input contains Inf values"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug("✓ Forward input validation passed")
    
    
    def get_model_cache_info(self) -> dict:
        """
        Get information about cached models in the download directory.
        
        Returns:
            dict: Information about cached models including:
                  - cache_dir: Path to cache directory
                  - exists: Whether directory exists
                  - model_files: List of downloaded model files
                  - total_size_mb: Total size of cached models in MB
        
        Example:
            >>> visual_module = VisualModule()
            >>> cache_info = visual_module.get_model_cache_info()
            >>> print(f"Cache directory: {cache_info['cache_dir']}")
            >>> print(f"Cached models: {cache_info['model_files']}")
            >>> print(f"Total size: {cache_info['total_size_mb']:.2f} MB")
        """
        checkpoints_dir = MODEL_CACHE_DIR / 'hub' / 'checkpoints'
        
        info = {
            'cache_dir': str(MODEL_CACHE_DIR),
            'checkpoints_dir': str(checkpoints_dir),
            'exists': checkpoints_dir.exists(),
            'model_files': [],
            'total_size_mb': 0.0
        }
        
        if checkpoints_dir.exists():
            model_files = list(checkpoints_dir.glob('*.pth'))
            info['model_files'] = [f.name for f in model_files]
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in model_files)
            info['total_size_mb'] = total_size / (1024 * 1024)
        
        return info


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_cache_directory() -> Path:
    """
    Get the path to the model cache directory.
    
    Returns:
        Path: Path object pointing to the downloaded_models directory
    
    Example:
        >>> cache_dir = get_model_cache_directory()
        >>> print(f"Models are cached in: {cache_dir}")
    """
    return MODEL_CACHE_DIR


def list_cached_models() -> List[str]:
    """
    List all cached model files in the download directory.
    
    Returns:
        List[str]: List of cached model filenames
    
    Example:
        >>> cached_models = list_cached_models()
        >>> print(f"Found {len(cached_models)} cached model(s):")
        >>> for model in cached_models:
        ...     print(f"  - {model}")
    """
    checkpoints_dir = MODEL_CACHE_DIR / 'hub' / 'checkpoints'
    
    if not checkpoints_dir.exists():
        return []
    
    model_files = list(checkpoints_dir.glob('*.pth'))
    return [f.name for f in model_files]


def get_cache_size() -> float:
    """
    Get the total size of cached models in MB.
    
    Returns:
        float: Total size in megabytes
    
    Example:
        >>> cache_size = get_cache_size()
        >>> print(f"Total cache size: {cache_size:.2f} MB")
    """
    checkpoints_dir = MODEL_CACHE_DIR / 'hub' / 'checkpoints'
    
    if not checkpoints_dir.exists():
        return 0.0
    
    model_files = list(checkpoints_dir.glob('*.pth'))
    total_size = sum(f.stat().st_size for f in model_files)
    
    return total_size / (1024 * 1024)


def clear_model_cache(confirm: bool = False) -> bool:
    """
    Clear all cached models from the download directory.
    
    Args:
        confirm (bool): Must be True to actually delete files (safety measure)
    
    Returns:
        bool: True if cache was cleared, False if not confirmed
    
    Warning:
        This will delete all downloaded model files!
        They will need to be re-downloaded next time.
    
    Example:
        >>> # Clear cache (requires confirmation)
        >>> cleared = clear_model_cache(confirm=True)
        >>> if cleared:
        ...     print("Cache cleared successfully")
    """
    if not confirm:
        logger.warning("clear_model_cache() called without confirmation")
        logger.warning("Set confirm=True to actually clear the cache")
        return False
    
    checkpoints_dir = MODEL_CACHE_DIR / 'hub' / 'checkpoints'
    
    if not checkpoints_dir.exists():
        logger.info("Cache directory doesn't exist, nothing to clear")
        return True
    
    try:
        import shutil
        
        model_files = list(checkpoints_dir.glob('*.pth'))
        logger.info(f"Deleting {len(model_files)} cached model file(s)...")
        
        for model_file in model_files:
            model_file.unlink()
            logger.info(f"  Deleted: {model_file.name}")
        
        logger.info("✓ Cache cleared successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return False


# ============================================================================
# USAGE EXAMPLES AND TEST CASES
# ============================================================================

if __name__ == "__main__":
    """
    Example usage and test cases for the VisualModule.
    This section demonstrates how to:
    1. Initialize the module with different configurations
    2. Process image inputs
    3. Handle errors
    4. Use different backbones
    5. Compare frozen vs trainable modes
    6. Check cached models
    """
    
    print("=" * 80)
    print("VISUAL MODULE - Usage Examples")
    print("=" * 80)
    
    # ------------------------------------------------------------------------
    # Example 0: Show Model Cache Information
    # ------------------------------------------------------------------------
    print("\n[Example 0] Model Cache Information")
    print("-" * 80)
    
    print(f"\nModel cache directory: {MODEL_CACHE_DIR}")
    print(f"Directory exists: {MODEL_CACHE_DIR.exists()}")
    
    cached_models = list_cached_models()
    cache_size = get_cache_size()
    
    print(f"\nCached models: {len(cached_models)}")
    if cached_models:
        for model in cached_models:
            print(f"  - {model}")
        print(f"\nTotal cache size: {cache_size:.2f} MB")
    else:
        print("  (No models cached yet - they will be downloaded on first use)")
    
    # ------------------------------------------------------------------------
    # Example 1: Basic Usage with ResNet-50
    # ------------------------------------------------------------------------
    print("\n[Example 1] Basic Usage with ResNet-50")
    print("-" * 80)
    
    try:
        # Initialize visual module
        print("Initializing VisualModule with ResNet-50...")
        print("(This will download the model if not already cached)")
        visual_module = VisualModule(
            backbone="resnet50",
            pretrained=True,
            out_features=512,
            freeze=False
        )
        print("✓ VisualModule initialized successfully\n")
        
        # Show cache info after initialization
        cache_info = visual_module.get_model_cache_info()
        print(f"Cache information:")
        print(f"  - Cache directory: {cache_info['cache_dir']}")
        print(f"  - Models cached: {len(cache_info['model_files'])}")
        print(f"  - Total size: {cache_info['total_size_mb']:.2f} MB")
        
        # Create dummy image data
        # In real usage, these would come from a DataLoader
        batch_size = 8
        channels = 3  # RGB
        height = 224
        width = 224
        
        print(f"\nCreating dummy image batch:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Channels: {channels} (RGB)")
        print(f"  - Height: {height}px")
        print(f"  - Width: {width}px")
        
        # Random image tensor (normally comes from real images)
        # Values should be normalized (e.g., mean=[0.485, 0.456, 0.406])
        images = torch.randn(batch_size, channels, height, width)
        
        print(f"\nInput shape: {images.shape}")
        print(f"Input dtype: {images.dtype}")
        print(f"Input value range: [{images.min().item():.4f}, {images.max().item():.4f}]")
        
        # Forward pass
        print("\nRunning forward pass...")
        with torch.no_grad():  # No gradient computation for inference
            features = visual_module(images)
        
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
    
    print(f"\nChecking cache directory: {MODEL_CACHE_DIR}")
    
    cached_models = list_cached_models()
    cache_size = get_cache_size()
    
    print(f"\nCached models: {len(cached_models)}")
    if cached_models:
        for model in cached_models:
            print(f"  ✓ {model}")
        print(f"\nTotal cache size: {cache_size:.2f} MB")
        print(f"\n✓ Models are now cached locally in: {MODEL_CACHE_DIR}")
        print("  Next time you run the code, models will load from cache (much faster!)")
    else:
        print("  (No models cached - this shouldn't happen after Example 1)")
    
    
    # ------------------------------------------------------------------------
    # Example 3: Different Backbone Architectures
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Example 3] Different Backbone Architectures")
    print("-" * 80)
    
    try:
        backbones_to_test = ["resnet18", "resnet34"]  # Test just 2 to save time
        
        print("Testing different ResNet architectures:")
        print("(Each will be downloaded and cached if not already present)\n")
        
        for backbone in backbones_to_test:
            print(f"Testing {backbone}...")
            
            # Initialize module
            module = VisualModule(
                backbone=backbone,
                pretrained=True,
                out_features=512,
                freeze=True
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in module.parameters())
            
            # Test forward pass
            images = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                features = module(images)
            
            print(f"  ✓ {backbone}: {total_params:,} params, output: {features.shape}")
            print()
        
        # Show updated cache
        print("Updated cache:")
        cached_models = list_cached_models()
        cache_size = get_cache_size()
        print(f"  - Total models cached: {len(cached_models)}")
        print(f"  - Total cache size: {cache_size:.2f} MB")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    
    # ------------------------------------------------------------------------
    # Summary with Cache Information
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODEL CACHE SUMMARY")
    print("=" * 80)
    
    print(f"""
Cache Directory: {MODEL_CACHE_DIR}

Structure:
  downloaded_models/
    └── hub/
        └── checkpoints/
            ├── resnet18-*.pth
            ├── resnet34-*.pth
            ├── resnet50-*.pth
            └── ... (other models)

Current Status:
  - Cached models: {len(list_cached_models())}
  - Total size: {get_cache_size():.2f} MB

Helper Functions:
  1. get_model_cache_directory() - Get cache directory path
  2. list_cached_models() - List all cached model files
  3. get_cache_size() - Get total cache size in MB
  4. clear_model_cache(confirm=True) - Delete all cached models

Usage:
  # Check what's cached
  >>> cached = list_cached_models()
  >>> print(f"Cached models: {{cached}}")
  
  # Get cache size
  >>> size = get_cache_size()
  >>> print(f"Cache size: {{size:.2f}} MB")
  
  # Clear cache (be careful!)
  >>> clear_model_cache(confirm=True)

Notes:
  - Models are downloaded only once
  - Subsequent runs load from cache (much faster)
  - Cache persists between runs
  - Can be shared across different scripts in the same directory
  - Safe to delete - models will re-download if needed
    """)
    
    print("=" * 80)
    print("End of Examples")
    print("=" * 80)