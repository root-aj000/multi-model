"""
Fine-Grained Multi-Modal Fusion Network (FG_MFN)
================================================
This module combines visual and text features for multi-attribute classification.
It supports multiple classification heads for different attributes like sentiment, 
emotion, theme, etc.

Author: [Your Name]
Date: [Date]
"""

import json
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

from models.visual_module import VisualModule
from models.text_module import TextModule
from utils.path import MODEL_CONFIG



# Configure logging for better debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# List of supported attribute names for multi-head classification
# These are the possible attributes the model can predict
ATTRIBUTE_NAMES = [
    "theme",              # Topic or theme of the content
    "sentiment",          # Positive/Negative/Neutral sentiment
    "emotion",            # Specific emotion (happy, sad, angry, etc.)
    "dominant_colour",    # Main color in the image
    "attention_score",    # How attention-grabbing the content is
    "trust_safety",       # Safety and trustworthiness level
    "target_audience",    # Intended audience type
    "predicted_ctr",      # Click-through rate prediction
    "likelihood_shares"   # Likelihood of being shared
]


class FG_MFN(nn.Module):
    """
    Fine-Grained Multi-Modal Fusion Network
    
    This network combines visual features (from images) and text features
    to predict multiple attributes simultaneously. It uses separate 
    classification heads for each attribute.
    
    Architecture Flow:
    1. Image → Visual Module → Visual Features
    2. Text → Text Module → Text Features  
    3. Visual + Text → Fusion → Shared Features
    4. Shared Features → Multiple Heads → Predictions for each attribute
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize the FG_MFN model with configuration.
        
        Args:
            cfg (dict): Configuration dictionary containing:
                - IMAGE_BACKBONE: Name of the image model (e.g., 'resnet50')
                - TEXT_ENCODER: Name of the text encoder (e.g., 'bert-base')
                - HIDDEN_DIM: Dimension of hidden layers
                - FUSION_TYPE: How to combine features ('concat' or 'add')
                - DROPOUT: Dropout probability
                - ATTRIBUTES: Dict of attributes with num_classes and labels
                - FREEZE_BACKBONE: Whether to freeze pretrained weights
                - NUM_CLASSES: (Optional) For backward compatibility
        
        Raises:
            ValueError: If required config parameters are missing
            TypeError: If config is not a dictionary
        """
        super(FG_MFN, self).__init__()
        
        # Step 1: Validate input configuration
        logger.info("Initializing FG_MFN model...")
        self._validate_config(cfg)
        
        # Step 2: Store configuration for later use
        self.cfg = cfg
        
        # Step 3: Extract configuration parameters with defaults
        # freeze=True means pretrained weights won't be updated during training
        freeze = cfg.get("FREEZE_BACKBONE", False)
        logger.info(f"Backbone freeze mode: {freeze}")
        
        # Step 4: Initialize the visual processing module
        # This module extracts features from images
        try:
            logger.info(f"Loading visual module with backbone: {cfg['IMAGE_BACKBONE']}")
            self.visual_module = VisualModule(
                backbone=cfg["IMAGE_BACKBONE"],
                out_features=cfg["HIDDEN_DIM"],
                freeze=freeze
            )
            logger.info("Visual module loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load visual module: {str(e)}")
            raise RuntimeError(f"Visual module initialization failed: {str(e)}")
        
        # Step 5: Initialize the text processing module
        # This module extracts features from text
        try:
            logger.info(f"Loading text module with encoder: {cfg['TEXT_ENCODER']}")
            self.text_module = TextModule(
                encoder_name=cfg["TEXT_ENCODER"],
                out_features=cfg["HIDDEN_DIM"],
                freeze=freeze
            )
            logger.info("Text module loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load text module: {str(e)}")
            raise RuntimeError(f"Text module initialization failed: {str(e)}")
        
        # Step 6: Determine fusion strategy
        # 'concat' = concatenate features (dimension doubles)
        # 'add' = element-wise addition (dimension stays same)
        self.fusion_type = cfg.get("FUSION_TYPE", "concat")
        logger.info(f"Fusion type: {self.fusion_type}")
        
        # Calculate the dimension after fusion
        if self.fusion_type == "concat":
            # Concatenation doubles the feature dimension
            fusion_dim = cfg["HIDDEN_DIM"] * 2
        else:
            # Addition keeps the same dimension
            fusion_dim = cfg["HIDDEN_DIM"]
        
        logger.info(f"Fusion dimension: {fusion_dim}")
        
        # Step 7: Create shared processing layer
        # This layer processes the fused features before splitting to different heads
        # Why shared layer? It learns common patterns useful for all attributes
        try:
            self.shared_fc = nn.Sequential(
                # Linear transformation to compress/process fused features
                nn.Linear(fusion_dim, cfg["HIDDEN_DIM"]),
                # ReLU activation for non-linearity
                nn.ReLU(),
                # Dropout for regularization (prevents overfitting)
                nn.Dropout(cfg["DROPOUT"])
            )
            logger.info(f"Shared layer created: {fusion_dim} -> {cfg['HIDDEN_DIM']}")
        except Exception as e:
            logger.error(f"Failed to create shared layer: {str(e)}")
            raise RuntimeError(f"Shared layer creation failed: {str(e)}")
        
        # Step 8: Create classification heads for each attribute
        # Each head is a separate classifier for one specific attribute
        self.attribute_heads = nn.ModuleDict()
        
        if "ATTRIBUTES" in cfg:
            # New multi-attribute configuration format
            logger.info("Using multi-attribute configuration")
            self._create_multi_attribute_heads(cfg)
        else:
            # Backward compatibility: single sentiment classifier
            # This supports old configs that only had sentiment classification
            logger.warning("Using legacy single-attribute mode (sentiment only)")
            self._create_legacy_sentiment_head(cfg)
        
        logger.info(f"Model initialization complete. Total attribute heads: {len(self.attribute_heads)}")
    
    
    def _validate_config(self, cfg: Any) -> None:
        """
        Validate that the configuration has all required parameters.
        
        Args:
            cfg: Configuration object to validate
            
        Raises:
            TypeError: If cfg is not a dictionary
            ValueError: If required keys are missing
        """
        # Check if config is a dictionary
        if not isinstance(cfg, dict):
            error_msg = f"Config must be a dictionary, got {type(cfg)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Define required configuration keys
        required_keys = ["IMAGE_BACKBONE", "TEXT_ENCODER", "HIDDEN_DIM", "DROPOUT"]
        
        # Check each required key
        missing_keys = []
        for key in required_keys:
            if key not in cfg:
                missing_keys.append(key)
        
        # If any keys are missing, raise an error
        if missing_keys:
            error_msg = f"Missing required config keys: {missing_keys}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate value types and ranges
        if not isinstance(cfg["HIDDEN_DIM"], int) or cfg["HIDDEN_DIM"] <= 0:
            error_msg = f"HIDDEN_DIM must be positive integer, got {cfg['HIDDEN_DIM']}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not isinstance(cfg["DROPOUT"], (int, float)) or not 0 <= cfg["DROPOUT"] < 1:
            error_msg = f"DROPOUT must be in [0, 1), got {cfg['DROPOUT']}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    
    def _create_multi_attribute_heads(self, cfg: Dict[str, Any]) -> None:
        """
        Create classification heads for multiple attributes.
        
        This method creates a separate neural network head (classifier) for
        each attribute defined in the configuration.
        
        Args:
            cfg (dict): Configuration containing ATTRIBUTES definition
        """
        created_heads = 0
        
        # Loop through all possible attribute names
        for attr_name in ATTRIBUTE_NAMES:
            # Check if this attribute is defined in config
            if attr_name in cfg["ATTRIBUTES"]:
                try:
                    # Get the number of classes for this attribute
                    num_classes = cfg["ATTRIBUTES"][attr_name]["num_classes"]
                    
                    # Validate num_classes
                    if not isinstance(num_classes, int) or num_classes <= 0:
                        logger.warning(f"Invalid num_classes for {attr_name}: {num_classes}. Skipping.")
                        continue
                    
                    # Create a linear classification layer for this attribute
                    # Input: shared features (HIDDEN_DIM)
                    # Output: logits for each class
                    self.attribute_heads[attr_name] = nn.Linear(
                        cfg["HIDDEN_DIM"], 
                        num_classes
                    )
                    
                    created_heads += 1
                    logger.info(f"Created head for '{attr_name}' with {num_classes} classes")
                    
                except KeyError as e:
                    logger.error(f"Missing configuration for attribute '{attr_name}': {str(e)}")
                except Exception as e:
                    logger.error(f"Failed to create head for '{attr_name}': {str(e)}")
        
        # Check if at least one head was created
        if created_heads == 0:
            error_msg = "No attribute heads were created. Check ATTRIBUTES configuration."
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    
    def _create_legacy_sentiment_head(self, cfg: Dict[str, Any]) -> None:
        """
        Create a single sentiment classification head for backward compatibility.
        
        This supports older configurations that only had sentiment classification
        instead of multiple attributes.
        
        Args:
            cfg (dict): Configuration containing NUM_CLASSES (optional)
        """
        try:
            # Default to 2 classes (positive/negative) if not specified
            num_classes = cfg.get("NUM_CLASSES", 2)
            
            # Create sentiment classifier
            self.attribute_heads["sentiment"] = nn.Linear(
                cfg["HIDDEN_DIM"], 
                num_classes
            )
            
            logger.info(f"Created legacy sentiment head with {num_classes} classes")
            
        except Exception as e:
            error_msg = f"Failed to create legacy sentiment head: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)


    def forward(
        self, 
        image_tensor: torch.Tensor, 
        text_tensor: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        This method processes images and text through their respective modules,
        fuses the features, and produces predictions for all attributes.
        
        Args:
            image_tensor (torch.Tensor): Batch of images 
                                        Shape: [batch_size, channels, height, width]
                                        Example: [32, 3, 224, 224]
            
            text_tensor (torch.Tensor): Batch of tokenized text
                                       Shape: [batch_size, sequence_length]
                                       Example: [32, 128]
            
            attention_mask (torch.Tensor, optional): Mask for text padding
                                                    Shape: [batch_size, sequence_length]
                                                    1 = real token, 0 = padding
        
        Returns:
            dict: Dictionary mapping attribute names to their logits
                  Format: {attr_name: torch.Tensor of shape [batch_size, num_classes]}
                  Example: {
                      'sentiment': tensor([[0.2, 0.8], [0.6, 0.4]]),  # 2 samples, 2 classes
                      'emotion': tensor([[0.1, 0.3, 0.6], [0.5, 0.3, 0.2]])  # 2 samples, 3 classes
                  }
        
        Raises:
            ValueError: If input tensors have invalid shapes
            RuntimeError: If forward pass fails
        """
        try:
            # Step 1: Validate input tensors
            self._validate_inputs(image_tensor, text_tensor, attention_mask)
            
            # Step 2: Extract visual features from images
            # Visual module processes images and outputs fixed-size feature vectors
            logger.debug(f"Processing visual features. Input shape: {image_tensor.shape}")
            visual_feat = self.visual_module(image_tensor)
            logger.debug(f"Visual features extracted. Shape: {visual_feat.shape}")
            
            # Step 3: Extract text features from tokenized text
            # Text module processes text tokens and outputs fixed-size feature vectors
            logger.debug(f"Processing text features. Input shape: {text_tensor.shape}")
            text_feat = self.text_module(text_tensor, attention_mask=attention_mask)
            logger.debug(f"Text features extracted. Shape: {text_feat.shape}")
            
            # Step 4: Fuse visual and text features
            # Fusion combines information from both modalities
            if self.fusion_type == "concat":
                # Concatenation: [v1, v2, v3, t1, t2, t3] -> [v1, v2, v3, t1, t2, t3]
                # This preserves all information from both modalities
                fused = torch.cat([visual_feat, text_feat], dim=1)
                logger.debug(f"Features fused via concatenation. Shape: {fused.shape}")
            else:
                # Addition: [v1, v2, v3] + [t1, t2, t3] -> [v1+t1, v2+t2, v3+t3]
                # This combines features element-wise
                fused = visual_feat + text_feat
                logger.debug(f"Features fused via addition. Shape: {fused.shape}")
            
            # Step 5: Process fused features through shared layer
            # This creates a common representation useful for all attributes
            shared_out = self.shared_fc(fused)
            logger.debug(f"Shared layer output. Shape: {shared_out.shape}")
            
            # Step 6: Generate predictions for each attribute
            # Each head produces logits (unnormalized scores) for its attribute
            outputs = {}
            for attr_name, head in self.attribute_heads.items():
                # Pass shared features through attribute-specific classifier
                logits = head(shared_out)
                outputs[attr_name] = logits
                logger.debug(f"Predictions for '{attr_name}'. Shape: {logits.shape}")
            
            logger.debug(f"Forward pass complete. Generated predictions for {len(outputs)} attributes")
            return outputs
            
        except Exception as e:
            error_msg = f"Forward pass failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    
    def _validate_inputs(
        self, 
        image_tensor: torch.Tensor, 
        text_tensor: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> None:
        """
        Validate input tensors before processing.
        
        Args:
            image_tensor (torch.Tensor): Image batch tensor
            text_tensor (torch.Tensor): Text token tensor
            attention_mask (torch.Tensor, optional): Attention mask tensor
            
        Raises:
            ValueError: If tensors have invalid shapes or types
        """
        # Check if inputs are tensors
        if not isinstance(image_tensor, torch.Tensor):
            error_msg = f"image_tensor must be torch.Tensor, got {type(image_tensor)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not isinstance(text_tensor, torch.Tensor):
            error_msg = f"text_tensor must be torch.Tensor, got {type(text_tensor)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check image tensor dimensions
        # Expected: [batch_size, channels, height, width]
        if image_tensor.dim() != 4:
            error_msg = f"image_tensor must be 4D [batch, channels, height, width], got shape {image_tensor.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check text tensor dimensions
        # Expected: [batch_size, sequence_length]
        if text_tensor.dim() != 2:
            error_msg = f"text_tensor must be 2D [batch, seq_length], got shape {text_tensor.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check batch sizes match
        if image_tensor.size(0) != text_tensor.size(0):
            error_msg = f"Batch size mismatch: images={image_tensor.size(0)}, text={text_tensor.size(0)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate attention mask if provided
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                error_msg = f"attention_mask must be torch.Tensor, got {type(attention_mask)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Attention mask should have same shape as text_tensor
            if attention_mask.shape != text_tensor.shape:
                error_msg = f"attention_mask shape {attention_mask.shape} doesn't match text_tensor shape {text_tensor.shape}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        logger.debug("Input validation passed")
    
    
    def get_label_names(self, attr_name: str) -> Optional[List[str]]:
        """
        Get the human-readable label names for a specific attribute.
        
        This is useful for interpreting model predictions. For example,
        instead of knowing the model predicted class 0, you can know it
        predicted "positive" sentiment.
        
        Args:
            attr_name (str): Name of the attribute (e.g., 'sentiment', 'emotion')
        
        Returns:
            list of str or None: List of label names in order of class indices
                                Returns None if attribute not found or no labels defined
                                
        Example:
            >>> model.get_label_names('sentiment')
            ['negative', 'neutral', 'positive']
            >>> # This means class 0 = negative, class 1 = neutral, class 2 = positive
        """
        try:
            # Check if we have the new multi-attribute configuration
            if "ATTRIBUTES" in self.cfg:
                # Check if the requested attribute exists
                if attr_name in self.cfg["ATTRIBUTES"]:
                    # Get the labels list from config
                    labels = self.cfg["ATTRIBUTES"][attr_name].get("labels", None)
                    
                    if labels is not None:
                        logger.debug(f"Retrieved {len(labels)} labels for '{attr_name}'")
                        return labels
                    else:
                        logger.warning(f"No labels defined for attribute '{attr_name}'")
                        return None
                else:
                    logger.warning(f"Attribute '{attr_name}' not found in configuration")
                    return None
            else:
                # Legacy mode: no label names available
                logger.warning("Legacy configuration mode: no label names available")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving label names for '{attr_name}': {str(e)}")
            return None


# ============================================================================
# USAGE EXAMPLES AND TEST CASES
# ============================================================================

if __name__ == "__main__":
    """
    Example usage and test cases for the FG_MFN model.
    This section demonstrates how to:
    1. Load configuration
    2. Initialize the model
    3. Create dummy data
    4. Run forward pass
    5. Interpret outputs
    """
    
    print("=" * 80)
    print("FG_MFN Model - Usage Examples")
    print("=" * 80)
    
    # ------------------------------------------------------------------------
    # Example 1: Basic Model Initialization and Forward Pass
    # ------------------------------------------------------------------------
    print("\n[Example 1] Basic Usage with Multi-Attribute Configuration")
    print("-" * 80)
    
    try:
        # Load configuration from JSON file
        print("Loading configuration from:", MODEL_CONFIG)
        with open(MODEL_CONFIG, "r") as f:
            cfg = json.load(f)
        print("✓ Configuration loaded successfully")
        
        # Initialize the model
        print("\nInitializing FG_MFN model...")
        model = FG_MFN(cfg)
        print("✓ Model initialized successfully")
        
        # Print model information
        print(f"\nModel Information:")
        print(f"  - Fusion Type: {model.fusion_type}")
        print(f"  - Number of Attribute Heads: {len(model.attribute_heads)}")
        print(f"  - Attributes: {list(model.attribute_heads.keys())}")
        
        # Create dummy input data
        batch_size = 2
        img = torch.randn(batch_size, 3, 224, 224)  # 2 images, RGB, 224x224
        text = torch.randint(0, 1000, (batch_size, 128))  # 2 texts, 128 tokens each
        attention_mask = torch.ones(batch_size, 128)  # All tokens are real (no padding)
        
        print(f"\nInput Shapes:")
        print(f"  - Images: {img.shape}")
        print(f"  - Text: {text.shape}")
        print(f"  - Attention Mask: {attention_mask.shape}")
        
        # Run forward pass
        print("\nRunning forward pass...")
        with torch.no_grad():  # No gradient computation for inference
            outputs = model(img, text, attention_mask=attention_mask)
        print("✓ Forward pass completed successfully")
        
        # Display outputs
        print(f"\nOutput Information:")
        print(f"  - Number of attributes predicted: {len(outputs)}")
        for attr_name, logits in outputs.items():
            print(f"  - {attr_name}: shape={logits.shape}, dtype={logits.dtype}")
            
            # Get label names if available
            labels = model.get_label_names(attr_name)
            if labels:
                print(f"    Labels: {labels}")
                # Get predicted classes
                predicted_classes = torch.argmax(logits, dim=1)
                print(f"    Predictions: {[labels[idx] for idx in predicted_classes.tolist()]}")
        
    except FileNotFoundError:
        print("✗ Error: Configuration file not found")
        print("  Please ensure MODEL_CONFIG path is correct")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    
    # ------------------------------------------------------------------------
    # Example 2: Legacy Mode (Single Sentiment Classification)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Example 2] Legacy Mode - Single Sentiment Classification")
    print("-" * 80)
    
    try:
        # Create legacy configuration (no ATTRIBUTES key)
        legacy_cfg = {
            "IMAGE_BACKBONE": "resnet50",
            "TEXT_ENCODER": "bert-base-uncased",
            "HIDDEN_DIM": 256,
            "DROPOUT": 0.3,
            "FUSION_TYPE": "concat",
            "NUM_CLASSES": 3,  # negative, neutral, positive
            "FREEZE_BACKBONE": False
        }
        
        print("Creating model with legacy configuration...")
        legacy_model = FG_MFN(legacy_cfg)
        print("✓ Legacy model created successfully")
        
        print(f"\nLegacy Model Information:")
        print(f"  - Attributes: {list(legacy_model.attribute_heads.keys())}")
        print(f"  - Sentiment classes: {legacy_cfg['NUM_CLASSES']}")
        
        # Test with dummy data
        img = torch.randn(1, 3, 224, 224)
        text = torch.randint(0, 1000, (1, 64))
        
        with torch.no_grad():
            outputs = legacy_model(img, text)
        
        print(f"\nLegacy Model Output:")
        for attr_name, logits in outputs.items():
            print(f"  - {attr_name}: {logits.shape}")
        
    except Exception as e:
        print(f"✗ Error in legacy mode: {str(e)}")
    
    
    # ------------------------------------------------------------------------
    # Example 3: Error Handling - Invalid Inputs
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Example 3] Error Handling - Invalid Inputs")
    print("-" * 80)
    
    try:
        # Create a simple model
        simple_cfg = {
            "IMAGE_BACKBONE": "resnet50",
            "TEXT_ENCODER": "bert-base-uncased",
            "HIDDEN_DIM": 128,
            "DROPOUT": 0.2,
            "NUM_CLASSES": 2
        }
        model = FG_MFN(simple_cfg)
        
        # Test 1: Mismatched batch sizes
        print("\nTest 1: Mismatched batch sizes")
        try:
            img = torch.randn(2, 3, 224, 224)  # batch size = 2
            text = torch.randint(0, 1000, (3, 128))  # batch size = 3
            outputs = model(img, text)
            print("✗ Should have raised an error!")
        except ValueError as e:
            print(f"✓ Correctly caught error: {str(e)}")
        
        # Test 2: Wrong tensor dimensions
        print("\nTest 2: Wrong image tensor dimensions")
        try:
            img = torch.randn(2, 224, 224)  # Missing channel dimension
            text = torch.randint(0, 1000, (2, 128))
            outputs = model(img, text)
            print("✗ Should have raised an error!")
        except ValueError as e:
            print(f"✓ Correctly caught error: {str(e)}")
        
        # Test 3: Invalid config
        print("\nTest 3: Invalid configuration")
        try:
            invalid_cfg = {"HIDDEN_DIM": -100}  # Missing required keys, invalid value
            model = FG_MFN(invalid_cfg)
            print("✗ Should have raised an error!")
        except (ValueError, TypeError) as e:
            print(f"✓ Correctly caught error: {type(e).__name__}")
        
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
    
    
    # ------------------------------------------------------------------------
    # Example 4: Using Label Names for Interpretation
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Example 4] Using Label Names for Interpretation")
    print("-" * 80)
    
    try:
        # Create config with label names
        cfg_with_labels = {
            "IMAGE_BACKBONE": "resnet50",
            "TEXT_ENCODER": "bert-base-uncased",
            "HIDDEN_DIM": 256,
            "DROPOUT": 0.3,
            "FUSION_TYPE": "add",
            "ATTRIBUTES": {
                "sentiment": {
                    "num_classes": 3,
                    "labels": ["negative", "neutral", "positive"]
                },
                "emotion": {
                    "num_classes": 4,
                    "labels": ["happy", "sad", "angry", "surprised"]
                }
            }
        }
        
        model = FG_MFN(cfg_with_labels)
        
        # Run prediction
        img = torch.randn(1, 3, 224, 224)
        text = torch.randint(0, 1000, (1, 128))
        
        with torch.no_grad():
            outputs = model(img, text)
        
        # Interpret predictions using label names
        print("\nPrediction Interpretation:")
        for attr_name, logits in outputs.items():
            # Get label names
            labels = model.get_label_names(attr_name)
            
            if labels:
                # Get predicted class
                predicted_idx = torch.argmax(logits, dim=1).item()
                predicted_label = labels[predicted_idx]
                
                # Get confidence (softmax probability)
                probabilities = torch.softmax(logits, dim=1)[0]
                confidence = probabilities[predicted_idx].item()
                
                print(f"\n  {attr_name.upper()}:")
                print(f"    Predicted: {predicted_label}")
                print(f"    Confidence: {confidence:.2%}")
                print(f"    All probabilities:")
                for label, prob in zip(labels, probabilities):
                    print(f"      - {label}: {prob:.2%}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    
    # ------------------------------------------------------------------------
    # Summary of Functions
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FUNCTION SUMMARY")
    print("=" * 80)
    
    print("""
    1. FG_MFN.__init__(cfg)
       Purpose: Initialize the model with configuration
       Use Case: Create model instance at the start of training/inference
       
    2. FG_MFN._validate_config(cfg)
       Purpose: Validate configuration has all required parameters
       Use Case: Called internally during __init__ to catch config errors early
       
    3. FG_MFN._create_multi_attribute_heads(cfg)
       Purpose: Create classification heads for multiple attributes
       Use Case: Called internally during __init__ for new config format
       
    4. FG_MFN._create_legacy_sentiment_head(cfg)
       Purpose: Create single sentiment head for backward compatibility
       Use Case: Called internally during __init__ for old config format
       
    5. FG_MFN.forward(image_tensor, text_tensor, attention_mask)
       Purpose: Process inputs and generate predictions
       Use Case: Called during training/inference to get model outputs
       
    6. FG_MFN._validate_inputs(image_tensor, text_tensor, attention_mask)
       Purpose: Validate input tensors before processing
       Use Case: Called internally during forward() to catch input errors
       
    7. FG_MFN.get_label_names(attr_name)
       Purpose: Get human-readable labels for an attribute
       Use Case: Interpret model predictions by mapping class indices to names
    """)
    
    print("=" * 80)
    print("End of Examples")
    print("=" * 80)