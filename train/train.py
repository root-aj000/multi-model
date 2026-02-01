"""
Multi-Modal Training Script for FG_MFN
======================================
This script trains the Fine-Grained Multi-Modal Fusion Network (FG_MFN)
on image-text data for multi-attribute classification.

Features:
- Multi-attribute training (theme, sentiment, emotion, etc.)
- Mixed precision training for faster performance
- Early stopping to prevent overfitting
- Learning rate scheduling
- Comprehensive logging and metrics tracking
- Checkpoint saving (best and last models)

Author: [Your Name]
Date: [Date]
"""

import os
import sys
import json
import logging
import traceback
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

from models.fg_mfn import FG_MFN, ATTRIBUTE_NAMES
from train.logger import Logger
from utils.path import TRAIN_CSV, VAL_CSV, SAVED_MODEL_DIR, MODEL_CONFIG
from preprocessing.dataset import CustomDataset

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


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure logging for detailed debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),  # Save logs to file
        logging.StreamHandler(sys.stdout)      # Print logs to console
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# These are the hyperparameters that control the training process
# Adjust these based on your dataset size and computational resources

BATCH_SIZE = 32  # Number of samples processed together
                 # Larger = faster but needs more memory
                 # Smaller = slower but more stable gradients

EPOCHS = 50  # Maximum number of complete passes through the dataset
             # Training may stop earlier due to early stopping

LEARNING_RATE = 1e-4  # Step size for weight updates
                      # Too high = unstable training
                      # Too low = very slow learning

WEIGHT_DECAY = 1e-5  # L2 regularization strength
                     # Helps prevent overfitting

EARLY_STOPPING_PATIENCE = 5  # Number of epochs to wait for improvement
                             # before stopping training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

RANDOM_SEED = 42  # For reproducible results

# Set random seeds for reproducibility
# This ensures you get the same results when running multiple times
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(RANDOM_SEED)

logger.info("=" * 80)
logger.info("TRAINING CONFIGURATION")
logger.info("=" * 80)
logger.info(f"Device: {DEVICE}")
logger.info(f"Batch Size: {BATCH_SIZE}")
logger.info(f"Epochs: {EPOCHS}")
logger.info(f"Learning Rate: {LEARNING_RATE}")
logger.info(f"Weight Decay: {WEIGHT_DECAY}")
logger.info(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
logger.info(f"Random Seed: {RANDOM_SEED}")
logger.info("=" * 80)


# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================

def setup_directories() -> None:
    """
    Create necessary directories for saving models and logs.
    
    This function ensures all required directories exist before training starts.
    If directories don't exist, they will be created.
    
    Raises:
        OSError: If directories cannot be created
    """
    try:
        # Create directory for saved models
        os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
        logger.info(f"✓ Model directory: {SAVED_MODEL_DIR}")
        
        # Create directory for logs
        log_dir = os.path.join(SAVED_MODEL_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"✓ Log directory: {log_dir}")
        
        return log_dir
        
    except OSError as e:
        error_msg = f"Failed to create directories: {str(e)}"
        logger.error(error_msg)
        raise OSError(error_msg)


def load_model_config() -> Dict[str, Any]:
    """
    Load model configuration from JSON file.
    
    The configuration file contains model architecture settings like
    backbone types, hidden dimensions, fusion strategies, etc.
    
    Returns:
        dict: Model configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
        ValueError: If config is missing required fields
    """
    try:
        logger.info(f"Loading model configuration from: {MODEL_CONFIG}")
        
        # Check if config file exists
        if not os.path.exists(MODEL_CONFIG):
            error_msg = f"Model config file not found: {MODEL_CONFIG}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Load JSON configuration
        with open(MODEL_CONFIG, "r") as f:
            cfg = json.load(f)
        
        # Validate configuration has required fields
        required_fields = ["IMAGE_BACKBONE", "TEXT_ENCODER", "HIDDEN_DIM", "DROPOUT"]
        missing_fields = [field for field in required_fields if field not in cfg]
        
        if missing_fields:
            error_msg = f"Config missing required fields: {missing_fields}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("✓ Model configuration loaded successfully")
        logger.info(f"  - Image Backbone: {cfg.get('IMAGE_BACKBONE')}")
        logger.info(f"  - Text Encoder: {cfg.get('TEXT_ENCODER')}")
        logger.info(f"  - Hidden Dimension: {cfg.get('HIDDEN_DIM')}")
        logger.info(f"  - Fusion Type: {cfg.get('FUSION_TYPE', 'concat')}")
        
        return cfg
        
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in config file: {str(e)}"
        logger.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Error loading config: {str(e)}"
        logger.error(error_msg)
        raise


def load_datasets() -> Tuple[CustomDataset, CustomDataset, bool, list]:
    """
    Load training and validation datasets.
    
    This function creates dataset objects for both training and validation.
    It also determines whether we're in legacy mode (single label) or
    multi-attribute mode.
    
    Returns:
        tuple: (train_dataset, val_dataset, legacy_mode, available_attributes)
    
    Raises:
        FileNotFoundError: If CSV files don't exist
        ValueError: If datasets are empty
        RuntimeError: If dataset loading fails
    """
    try:
        logger.info("Loading datasets...")
        
        # Check if CSV files exist
        if not os.path.exists(TRAIN_CSV):
            error_msg = f"Training CSV not found: {TRAIN_CSV}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not os.path.exists(VAL_CSV):
            error_msg = f"Validation CSV not found: {VAL_CSV}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Load training dataset
        logger.info(f"Loading training data from: {TRAIN_CSV}")
        train_dataset = CustomDataset(TRAIN_CSV)
        logger.info(f"✓ Training samples: {len(train_dataset)}")
        
        # Load validation dataset
        logger.info(f"Loading validation data from: {VAL_CSV}")
        val_dataset = CustomDataset(VAL_CSV)
        logger.info(f"✓ Validation samples: {len(val_dataset)}")
        
        # Validate dataset sizes
        if len(train_dataset) == 0:
            error_msg = "Training dataset is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if len(val_dataset) == 0:
            error_msg = "Validation dataset is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Determine training mode (legacy vs multi-attribute)
        legacy_mode = getattr(train_dataset, 'legacy_mode', True)
        available_attributes = getattr(train_dataset, 'available_attributes', [])
        
        logger.info(f"Training mode: {'Legacy (single label)' if legacy_mode else 'Multi-attribute'}")
        
        if not legacy_mode:
            logger.info(f"Available attributes: {available_attributes}")
            if not available_attributes:
                logger.warning("Multi-attribute mode but no attributes found!")
        
        return train_dataset, val_dataset, legacy_mode, available_attributes
        
    except FileNotFoundError:
        raise
    except Exception as e:
        error_msg = f"Failed to load datasets: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg)


def create_data_loaders(
    train_dataset: CustomDataset,
    val_dataset: CustomDataset,
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.
    
    DataLoaders handle batching, shuffling, and parallel data loading.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Number of samples per batch
    
    Returns:
        tuple: (train_loader, val_loader)
    
    Raises:
        ValueError: If batch_size is invalid
        RuntimeError: If data loader creation fails
    """
    try:
        # Validate batch size
        if batch_size <= 0:
            error_msg = f"Batch size must be positive, got {batch_size}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if batch_size > len(train_dataset):
            logger.warning(f"Batch size ({batch_size}) is larger than training dataset ({len(train_dataset)})")
            logger.warning(f"Reducing batch size to {len(train_dataset)}")
            batch_size = len(train_dataset)
        
        logger.info(f"Creating data loaders with batch size: {batch_size}")
        
        # Create training data loader
        # shuffle=True: Randomize sample order each epoch (helps generalization)
        # num_workers=4: Use 4 parallel processes for data loading (faster)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if DEVICE == "cuda" else False  # Faster GPU transfer
        )
        
        # Create validation data loader
        # shuffle=False: Keep order consistent for reproducible validation
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if DEVICE == "cuda" else False
        )
        
        logger.info(f"✓ Training batches: {len(train_loader)}")
        logger.info(f"✓ Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader
        
    except Exception as e:
        error_msg = f"Failed to create data loaders: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def initialize_model(cfg: Dict[str, Any], device: str) -> FG_MFN:
    """
    Initialize the FG_MFN model and move it to the specified device.
    
    Args:
        cfg: Model configuration dictionary
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        FG_MFN: Initialized model
    
    Raises:
        RuntimeError: If model initialization fails
    """
    try:
        logger.info("Initializing FG_MFN model...")
        
        # Create model
        model = FG_MFN(cfg)
        
        # Move model to device (GPU or CPU)
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info("✓ Model initialized successfully")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Model device: {device}")
        logger.info(f"  - Number of attribute heads: {len(model.attribute_heads)}")
        logger.info(f"  - Attributes: {list(model.attribute_heads.keys())}")
        
        return model
        
    except Exception as e:
        error_msg = f"Failed to initialize model: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg)


def setup_training_components(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float
) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """
    Set up loss function, optimizer, and learning rate scheduler.
    
    Args:
        model: The model to train
        learning_rate: Initial learning rate
        weight_decay: L2 regularization coefficient
    
    Returns:
        tuple: (criterion, optimizer, scheduler)
    
    Raises:
        ValueError: If parameters are invalid
    """
    try:
        # Validate parameters
        if learning_rate <= 0:
            error_msg = f"Learning rate must be positive, got {learning_rate}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if weight_decay < 0:
            error_msg = f"Weight decay must be non-negative, got {weight_decay}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Setting up training components...")
        
        # Loss function: CrossEntropyLoss for classification
        # Combines LogSoftmax and NLLLoss
        # Expects raw logits (not probabilities) as input
        criterion = nn.CrossEntropyLoss()
        logger.info("✓ Loss function: CrossEntropyLoss")
        
        # Optimizer: AdamW (Adam with weight decay)
        # AdamW is better than Adam for most tasks
        # It properly decouples weight decay from gradient updates
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        logger.info(f"✓ Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
        
        # Learning rate scheduler: ReduceLROnPlateau
        # Automatically reduces learning rate when validation loss stops improving
        # This helps fine-tune the model in later epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',      # Minimize validation loss
            patience=3,      # Wait 3 epochs before reducing
            factor=0.5,      # Multiply LR by 0.5 when reducing
            verbose=True     # Print when LR is reduced
        )
        logger.info("✓ Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)")
        
        return criterion, optimizer, scheduler
        
    except Exception as e:
        error_msg = f"Failed to setup training components: {str(e)}"
        logger.error(error_msg)
        raise


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    This function performs one complete pass through the training dataset,
    updating model weights based on the computed loss.
    
    Args:
        model: The model to train
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for weight updates
        device: Device to use ('cuda' or 'cpu')
        scaler: Gradient scaler for mixed precision training (optional)
    
    Returns:
        dict: Dictionary of training metrics
              Format: {
                  'loss': average_loss,
                  'attribute_acc': accuracy_for_attribute,
                  'attribute_f1': f1_score_for_attribute,
                  ...
              }
    
    Raises:
        RuntimeError: If training fails
    
    Note:
        This function expects the model to return a dictionary of outputs,
        one for each attribute being predicted.
    """
    try:
        # Set model to training mode
        # This enables dropout and batch normalization in training mode
        model.train()
        
        # Lists to accumulate losses and predictions
        train_losses = []
        
        # Dictionary to store predictions and labels for each attribute
        all_preds = {attr: [] for attr in ATTRIBUTE_NAMES}
        all_labels = {attr: [] for attr in ATTRIBUTE_NAMES}
        
        # Progress bar for visual feedback
        progress_bar = tqdm(loader, desc="Training", leave=False)
        
        # Iterate through batches
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Step 1: Move data to device (GPU/CPU)
                # batch is a dictionary with keys: 'visual', 'text', 'attention_mask', labels
                images = batch["visual"].to(device)
                texts = batch["text"].to(device)
                masks = batch["attention_mask"].to(device)
                
                # Step 2: Zero gradients from previous iteration
                # PyTorch accumulates gradients by default, so we need to clear them
                optimizer.zero_grad()
                
                # Step 3: Forward pass with mixed precision (if using GPU)
                # Mixed precision uses float16 for faster computation
                # and float32 for numerical stability where needed
                with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
                    # Get model predictions
                    # outputs is a dict: {attr_name: logits_tensor}
                    outputs = model(images, texts, attention_mask=masks)
                    
                    # Step 4: Compute loss
                    # We need to handle both legacy mode (single label) and
                    # multi-attribute mode differently
                    
                    # Determine which mode we're in based on the batch
                    legacy_mode = "label" in batch and len(batch.get("label", [])) > 0
                    
                    total_loss = 0
                    num_attrs = 0
                    
                    if legacy_mode :
                        # Legacy mode: single sentiment classification
                        # Labels are expected to be 1, 2, 3 (convert to 0, 1, 2)
                        labels = (batch["label"] - 1).to(device)
                        
                        # Use sentiment head if available, otherwise use first head
                        if "sentiment" in outputs:
                            loss = criterion(outputs["sentiment"], labels)
                        else:
                            # Fallback to first available head
                            first_key = list(outputs.keys())[0]
                            loss = criterion(outputs[first_key], labels)
                        
                        total_loss = loss
                        num_attrs = 1
                        
                    else:
                        # Multi-attribute mode: compute loss for each attribute
                        for attr in ATTRIBUTE_NAMES:
                            # Check if this attribute has both predictions and labels
                            if attr in outputs and attr in batch:
                                # Get labels for this attribute
                                labels = batch[attr].to(device)
                                
                                # Compute loss for this attribute
                                loss = criterion(outputs[attr], labels)
                                total_loss += loss
                                num_attrs += 1
                                
                                # Track predictions for metric calculation
                                preds = torch.argmax(outputs[attr], dim=1).cpu().numpy()
                                all_preds[attr].extend(preds)
                                all_labels[attr].extend(labels.cpu().numpy())
                        
                        # Average the loss across all attributes
                        # This gives equal weight to each attribute
                        if num_attrs > 0:
                            total_loss = total_loss / num_attrs
                        else:
                            # No valid attributes found
                            logger.warning(f"Batch {batch_idx}: No valid attributes for loss computation")
                            continue
                
                # Step 5: Backward pass
                # Compute gradients with respect to model parameters
                if scaler is not None:
                    # Mixed precision: scale loss to prevent underflow
                    scaler.scale(total_loss).backward()
                    # Update weights with scaled gradients
                    scaler.step(optimizer)
                    # Update scaler for next iteration
                    scaler.update()
                else:
                    # Standard precision
                    total_loss.backward()
                    optimizer.step()
                
                # Step 6: Record loss
                train_losses.append(total_loss.item())
                
                # Update progress bar with current loss
                progress_bar.set_postfix({'loss': total_loss.item()})
                
            except Exception as e:
                error_msg = f"Error in training batch {batch_idx}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                # Continue with next batch instead of stopping entirely
                continue
        
        # Step 7: Compute epoch-level metrics
        # Calculate average loss over all batches
        if not train_losses:
            error_msg = "No valid training batches processed"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        avg_loss = np.mean(train_losses)
        
        # Initialize metrics dictionary with loss
        metrics = {"loss": avg_loss}
        
        # Step 8: Compute per-attribute metrics
        # Calculate accuracy and F1 score for each attribute
        for attr in ATTRIBUTE_NAMES:
            if all_preds[attr] and all_labels[attr]:
                try:
                    # Accuracy: percentage of correct predictions
                    acc = accuracy_score(all_labels[attr], all_preds[attr])
                    metrics[f"{attr}_acc"] = acc
                    
                    # F1 score: harmonic mean of precision and recall
                    # weighted: accounts for class imbalance
                    f1 = f1_score(
                        all_labels[attr],
                        all_preds[attr],
                        average='weighted',
                        zero_division=0  # Return 0 if no predictions
                    )
                    metrics[f"{attr}_f1"] = f1
                    
                except Exception as e:
                    logger.warning(f"Failed to compute metrics for {attr}: {str(e)}")
                    # Continue with other attributes
                    continue
        
        return metrics
        
    except Exception as e:
        error_msg = f"Training epoch failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg)


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """
    Validate the model for one epoch.
    
    This function evaluates the model on the validation dataset
    without updating weights. Used to monitor overfitting and
    guide early stopping.
    
    Args:
        model: The model to validate
        loader: DataLoader for validation data
        criterion: Loss function
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        dict: Dictionary of validation metrics
              Format same as train_epoch()
    
    Raises:
        RuntimeError: If validation fails
    
    Note:
        This function runs with torch.no_grad() to save memory
        and computation since we don't need gradients.
    """
    try:
        # Set model to evaluation mode
        # This disables dropout and sets batch normalization to use
        # running statistics instead of batch statistics
        model.eval()
        
        # Lists to accumulate losses and predictions
        val_losses = []
        
        # Dictionary to store predictions and labels for each attribute
        all_preds = {attr: [] for attr in ATTRIBUTE_NAMES}
        all_labels = {attr: [] for attr in ATTRIBUTE_NAMES}
        
        # Progress bar for visual feedback
        progress_bar = tqdm(loader, desc="Validation", leave=False)
        
        # Disable gradient computation
        # This saves memory and speeds up computation
        # since we don't need gradients during validation
        with torch.no_grad():
            # Iterate through batches
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Step 1: Move data to device
                    images = batch["visual"].to(device)
                    texts = batch["text"].to(device)
                    masks = batch["attention_mask"].to(device)
                    
                    # Step 2: Forward pass (no mixed precision needed for validation)
                    outputs = model(images, texts, attention_mask=masks)
                    
                    # Step 3: Compute loss
                    # Same logic as training, but without backward pass
                    
                    # Determine mode
                    legacy_mode = "label" in batch and len(batch.get("label", [])) > 0
                    
                    total_loss = 0
                    num_attrs = 0
                    
                    if legacy_mode:
                        # Legacy mode: single label
                        labels = (batch["label"] - 1).to(device)
                        
                        if "sentiment" in outputs:
                            loss = criterion(outputs["sentiment"], labels)
                        else:
                            first_key = list(outputs.keys())[0]
                            loss = criterion(outputs[first_key], labels)
                        
                        total_loss = loss
                        num_attrs = 1
                        
                    else:
                        # Multi-attribute mode
                        for attr in ATTRIBUTE_NAMES:
                            if attr in outputs and attr in batch:
                                labels = batch[attr].to(device)
                                loss = criterion(outputs[attr], labels)
                                total_loss += loss
                                num_attrs += 1
                                
                                # Track predictions
                                preds = torch.argmax(outputs[attr], dim=1).cpu().numpy()
                                all_preds[attr].extend(preds)
                                all_labels[attr].extend(labels.cpu().numpy())
                        
                        # Average loss
                        if num_attrs > 0:
                            total_loss = total_loss / num_attrs
                        else:
                            logger.warning(f"Validation batch {batch_idx}: No valid attributes")
                            continue
                    
                    # Step 4: Record loss
                    val_losses.append(total_loss.item())
                    
                    # Update progress bar
                    progress_bar.set_postfix({'loss': total_loss.item()})
                    
                except Exception as e:
                    error_msg = f"Error in validation batch {batch_idx}: {str(e)}"
                    logger.error(error_msg)
                    # Continue with next batch
                    continue
        
        # Step 5: Compute epoch-level metrics
        if not val_losses:
            error_msg = "No valid validation batches processed"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        avg_loss = np.mean(val_losses)
        
        # Initialize metrics dictionary
        metrics = {"loss": avg_loss}
        
        # Step 6: Compute per-attribute metrics
        for attr in ATTRIBUTE_NAMES:
            if all_preds[attr] and all_labels[attr]:
                try:
                    # Accuracy
                    acc = accuracy_score(all_labels[attr], all_preds[attr])
                    metrics[f"{attr}_acc"] = acc
                    
                    # F1 score
                    f1 = f1_score(
                        all_labels[attr],
                        all_preds[attr],
                        average='weighted',
                        zero_division=0
                    )
                    metrics[f"{attr}_f1"] = f1
                    
                except Exception as e:
                    logger.warning(f"Failed to compute validation metrics for {attr}: {str(e)}")
                    continue
        
        return metrics
        
    except Exception as e:
        error_msg = f"Validation epoch failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg)


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    """
    Main training function.
    
    This is the entry point for the training script. It:
    1. Sets up all necessary components (data, model, optimizer, etc.)
    2. Runs the training loop for multiple epochs
    3. Validates after each epoch
    4. Saves checkpoints
    5. Implements early stopping
    
    The function handles all error cases and ensures proper cleanup.
    
    Raises:
        RuntimeError: If training setup or execution fails
    """
    try:
        # Record training start time
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # =====================================================================
        # STEP 1: SETUP - Initialize all components
        # =====================================================================
        
        logger.info("\n[STEP 1/6] Setting up training environment...")
        
        # Create necessary directories
        log_dir = setup_directories()
        
        # Load model configuration
        cfg = load_model_config()
        
        # Load datasets
        train_dataset, val_dataset, legacy_mode, available_attributes = load_datasets()
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_dataset,
            val_dataset,
            BATCH_SIZE
        )
        
        # Initialize model
        model = initialize_model(cfg, DEVICE)
        
        # Setup training components
        criterion, optimizer, scheduler = setup_training_components(
            model,
            LEARNING_RATE,
            WEIGHT_DECAY
        )
        
        # Initialize logger for metrics
        metrics_logger = Logger(log_dir)
        logger.info("✓ Metrics logger initialized")
        
        # Setup mixed precision training (if using GPU)
        scaler = None
        if DEVICE == "cuda":
            scaler = torch.cuda.amp.GradScaler("cuda")
            logger.info("✓ Mixed precision training enabled")
        else:
            logger.info("ℹ Mixed precision disabled (CPU mode)")
        
        logger.info("\n✓ Setup complete!\n")
        
        # =====================================================================
        # STEP 2: TRAINING PREPARATION
        # =====================================================================
        
        logger.info("[STEP 2/6] Preparing for training...")
        
        # Early stopping variables
        best_val_loss = float('inf')  # Initialize to infinity
        patience_counter = 0           # Count epochs without improvement
        
        # Training statistics
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        logger.info(f"✓ Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")
        logger.info(f"✓ Best model will be saved based on validation loss")
        logger.info("\n")
        
        # =====================================================================
        # STEP 3: TRAINING LOOP
        # =====================================================================
        
        logger.info("[STEP 3/6] Starting training loop...")
        logger.info("=" * 80)
        
        for epoch in range(1, EPOCHS + 1):
            try:
                logger.info(f"\nEPOCH {epoch}/{EPOCHS}")
                logger.info("-" * 80)
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Learning Rate: {current_lr:.6f}")
                
                # ============================================================
                # STEP 3A: TRAINING PHASE
                # ============================================================
                
                logger.info("\nTraining phase...")
                train_metrics = train_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    DEVICE,
                    scaler
                )
                
                logger.info(f"✓ Training complete - Loss: {train_metrics['loss']:.4f}")
                
                # ============================================================
                # STEP 3B: VALIDATION PHASE
                # ============================================================
                
                logger.info("\nValidation phase...")
                val_metrics = validate_epoch(
                    model,
                    val_loader,
                    criterion,
                    DEVICE
                )
                
                logger.info(f"✓ Validation complete - Loss: {val_metrics['loss']:.4f}")
                
                # ============================================================
                # STEP 3C: LOG METRICS
                # ============================================================
                
                # Combine all metrics for logging
                log_data = {}
                
                # Add training metrics with 'train_' prefix
                for key, value in train_metrics.items():
                    log_data[f"train_{key}"] = value
                
                # Add validation metrics with 'val_' prefix
                for key, value in val_metrics.items():
                    log_data[f"val_{key}"] = value
                
                # Add learning rate
                log_data['learning_rate'] = current_lr
                
                # Log to file
                metrics_logger.log_metrics(log_data, epoch)
                
                # ============================================================
                # STEP 3D: PRINT SUMMARY
                # ============================================================
                
                train_loss = train_metrics["loss"]
                val_loss = val_metrics["loss"]
                
                logger.info("\n" + "=" * 80)
                logger.info(f"EPOCH {epoch} SUMMARY")
                logger.info("=" * 80)
                logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                # Print per-attribute metrics if available
                logger.info("\nPer-Attribute Performance:")
                for attr in ATTRIBUTE_NAMES:
                    train_acc_key = f"{attr}_acc"
                    val_acc_key = f"{attr}_acc"
                    
                    if train_acc_key in train_metrics:
                        train_acc = train_metrics[train_acc_key]
                        val_acc = val_metrics.get(val_acc_key, 0.0)
                        
                        train_f1 = train_metrics.get(f"{attr}_f1", 0.0)
                        val_f1 = val_metrics.get(f"{attr}_f1", 0.0)
                        
                        logger.info(f"  {attr:20s}: Train Acc={train_acc:.3f} | Val Acc={val_acc:.3f} | "
                                  f"Train F1={train_f1:.3f} | Val F1={val_f1:.3f}")
                
                logger.info("=" * 80)
                
                # ============================================================
                # STEP 3E: LEARNING RATE SCHEDULING
                # ============================================================
                
                # Update learning rate based on validation loss
                # The scheduler will reduce LR if val_loss doesn't improve
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                
                if new_lr != old_lr:
                    logger.info(f"\n📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
                
                # ============================================================
                # STEP 3F: SAVE CHECKPOINTS
                # ============================================================
                
                # Always save the latest model
                last_model_path = os.path.join(SAVED_MODEL_DIR, "model_last.pt")
                try:
                    torch.save(model.state_dict(), last_model_path)
                    logger.info(f"\n💾 Saved latest model: {last_model_path}")
                except Exception as e:
                    logger.error(f"Failed to save last model: {str(e)}")
                
                # ============================================================
                # STEP 3G: EARLY STOPPING CHECK
                # ============================================================
                
                # Check if validation loss improved
                if val_loss < best_val_loss:
                    # New best model!
                    improvement = best_val_loss - val_loss
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    best_model_path = os.path.join(SAVED_MODEL_DIR, "model_best.pt")
                    try:
                        torch.save(model.state_dict(), best_model_path)
                        logger.info(f"🌟 New best model! Validation loss improved by {improvement:.4f}")
                        logger.info(f"💾 Saved best model: {best_model_path}")
                    except Exception as e:
                        logger.error(f"Failed to save best model: {str(e)}")
                    
                else:
                    # No improvement
                    patience_counter += 1
                    logger.info(f"\n⚠ No improvement in validation loss for {patience_counter} epoch(s)")
                    logger.info(f"   Best val loss: {best_val_loss:.4f} | Current: {val_loss:.4f}")
                    
                    # Check if we should stop early
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        logger.info("\n" + "=" * 80)
                        logger.info(f"🛑 EARLY STOPPING at epoch {epoch}")
                        logger.info(f"   No improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs")
                        logger.info(f"   Best validation loss: {best_val_loss:.4f}")
                        logger.info("=" * 80)
                        break
                
                # Update training history
                training_history['train_loss'].append(train_loss)
                training_history['val_loss'].append(val_loss)
                training_history['learning_rates'].append(current_lr)
                
                logger.info("\n")  # Add spacing between epochs
                
            except Exception as e:
                error_msg = f"Error in epoch {epoch}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                
                # Save emergency checkpoint
                emergency_path = os.path.join(SAVED_MODEL_DIR, f"model_epoch_{epoch}_emergency.pt")
                try:
                    torch.save(model.state_dict(), emergency_path)
                    logger.info(f"💾 Saved emergency checkpoint: {emergency_path}")
                except:
                    logger.error("Failed to save emergency checkpoint")
                
                # Decide whether to continue or stop
                logger.info("Attempting to continue with next epoch...")
                continue
        
        # =====================================================================
        # STEP 4: TRAINING COMPLETE
        # =====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        
        # Calculate training duration
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {duration}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # =====================================================================
        # STEP 5: SAVE TRAINING HISTORY
        # =====================================================================
        
        logger.info("\n[STEP 5/6] Saving training history...")
        
        try:
            history_path = os.path.join(SAVED_MODEL_DIR, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)
            logger.info(f"✓ Training history saved: {history_path}")
        except Exception as e:
            logger.error(f"Failed to save training history: {str(e)}")
        
        # =====================================================================
        # STEP 6: CLEANUP
        # =====================================================================
        
        logger.info("\n[STEP 6/6] Cleaning up...")
        
        try:
            # Close metrics logger
            metrics_logger.close()
            logger.info("✓ Metrics logger closed")
            
            # Clear GPU cache if using CUDA
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                logger.info("✓ GPU cache cleared")
            
        except Exception as e:
            logger.error(f"Cleanup warning: {str(e)}")
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL DONE! 🎉")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        logger.info("\n" + "=" * 80)
        logger.info("⚠ Training interrupted by user")
        logger.info("=" * 80)
        
        # Try to save current state
        try:
            interrupt_path = os.path.join(SAVED_MODEL_DIR, "model_interrupted.pt")
            torch.save(model.state_dict(), interrupt_path)
            logger.info(f"💾 Saved interrupted model: {interrupt_path}")
        except:
            logger.error("Failed to save interrupted model")
        
        sys.exit(0)
        
    except Exception as e:
        # Handle any unexpected errors
        logger.error("\n" + "=" * 80)
        logger.error("❌ TRAINING FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        
        # Try to save current state for debugging
        try:
            error_path = os.path.join(SAVED_MODEL_DIR, "model_error_state.pt")
            torch.save(model.state_dict(), error_path)
            logger.info(f"💾 Saved error state model: {error_path}")
        except:
            logger.error("Failed to save error state model")
        
        raise


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Script entry point.
    
    This block only runs when the script is executed directly
    (not when imported as a module).
    """
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


# =============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# =============================================================================

"""
FUNCTION SUMMARY
================

1. setup_directories()
   Purpose: Create directories for models and logs
   Use Cases:
     - Initialize training environment
     - Ensure output directories exist
     - Called at start of training
   
   Example:
     log_dir = setup_directories()


2. load_model_config()
   Purpose: Load model configuration from JSON file
   Use Cases:
     - Read model architecture settings
     - Validate configuration completeness
     - Get hyperparameters for model initialization
   
   Example:
     cfg = load_model_config()
     print(cfg['HIDDEN_DIM'])  # Access configuration


3. load_datasets()
   Purpose: Load training and validation datasets
   Use Cases:
     - Read CSV files with training data
     - Determine training mode (legacy vs multi-attribute)
     - Validate dataset integrity
   
   Returns:
     train_dataset, val_dataset, legacy_mode, available_attributes
   
   Example:
     train_ds, val_ds, legacy, attrs = load_datasets()
     print(f"Training samples: {len(train_ds)}")


4. create_data_loaders(train_dataset, val_dataset, batch_size)
   Purpose: Create PyTorch DataLoaders for batching
   Use Cases:
     - Enable batch processing
     - Shuffle training data
     - Parallel data loading
   
   Example:
     train_loader, val_loader = create_data_loaders(
         train_dataset, val_dataset, batch_size=32
     )


5. initialize_model(cfg, device)
   Purpose: Create and initialize the FG_MFN model
   Use Cases:
     - Build model architecture
     - Move model to GPU/CPU
     - Log model statistics
   
   Example:
     model = initialize_model(cfg, 'cuda')
     print(f"Parameters: {sum(p.numel() for p in model.parameters())}")


6. setup_training_components(model, learning_rate, weight_decay)
   Purpose: Create loss function, optimizer, and scheduler
   Use Cases:
     - Initialize training components
     - Configure optimization strategy
     - Setup learning rate scheduling
   
   Returns:
     criterion, optimizer, scheduler
   
   Example:
     criterion, optimizer, scheduler = setup_training_components(
         model, lr=1e-4, weight_decay=1e-5
     )


7. train_epoch(model, loader, criterion, optimizer, device, scaler)
   Purpose: Train model for one complete epoch
   Use Cases:
     - Update model weights on training data
     - Compute training metrics
     - Support mixed precision training
   
   Returns:
     dict of metrics: {'loss': float, 'attr_acc': float, ...}
   
   Example:
     metrics = train_epoch(model, train_loader, criterion, optimizer, 'cuda')
     print(f"Training loss: {metrics['loss']:.4f}")


8. validate_epoch(model, loader, criterion, device)
   Purpose: Evaluate model on validation data
   Use Cases:
     - Monitor overfitting
     - Guide early stopping
     - Compare different models
   
   Returns:
     dict of metrics: {'loss': float, 'attr_acc': float, ...}
   
   Example:
     val_metrics = validate_epoch(model, val_loader, criterion, 'cuda')
     print(f"Validation accuracy: {val_metrics['sentiment_acc']:.3f}")


9. main()
   Purpose: Main training orchestration function
   Use Cases:
     - Complete training pipeline
     - Handle all setup and cleanup
     - Implement training loop with early stopping
   
   Example:
     # Simply run main to start training
     main()


TYPICAL WORKFLOW
================

1. Prepare Data:
   - Create train.csv and val.csv with your data
   - Ensure images are in correct directories
   - Set up MODEL_CONFIG.json with architecture settings

2. Configure Training:
   # Edit hyperparameters at top of script
   BATCH_SIZE = 32
   EPOCHS = 50
   LEARNING_RATE = 1e-4

3. Run Training:
   python training/train.py

4. Monitor Progress:
   - Watch console output for epoch summaries
   - Check training.log for detailed logs
   - Review metrics in logs/ directory

5. Use Trained Model:
   # Load best model
   model = FG_MFN(cfg)
   model.load_state_dict(torch.load('saved_models/model_best.pt'))
   model.eval()


COMMAND LINE USAGE
==================

Basic usage:
  python training/train.py

With logging redirect:
  python training/train.py 2>&1 | tee training_output.txt

Background process:
  nohup python training/train.py > training.out 2>&1 &


TROUBLESHOOTING
===============

1. Out of Memory:
   - Reduce BATCH_SIZE (try 16, 8, or 4)
   - Reduce image size in dataset
   - Use freeze=True in model config

2. Training Not Improving:
   - Check learning rate (try 1e-5 or 1e-3)
   - Verify data quality and labels
   - Try different backbone (resnet18 vs resnet50)
   - Check for data imbalance

3. Slow Training:
   - Ensure CUDA is available
   - Reduce num_workers if CPU is bottleneck
   - Use smaller backbone (resnet18)
   - Enable mixed precision (automatic on GPU)

4. NaN Loss:
   - Reduce learning rate
   - Check for invalid data (NaN, Inf)
   - Verify normalization is correct
   - Try gradient clipping


SAVED FILES
===========

After training, you'll find:

saved_models/
  ├── model_best.pt           # Best model based on validation loss
  ├── model_last.pt           # Latest model from last epoch
  ├── training_history.json   # Loss and metric history
  └── logs/
      └── events.out.tfevents.*  # TensorBoard logs

training.log                  # Detailed text logs
"""