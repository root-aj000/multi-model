import os
import sys
import json
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Any

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessing.dataset import CustomDataset
from models.fg_mfn import FG_MFN, ATTRIBUTE_NAMES
from utils.path import TEST_CSV, SAVED_MODEL_PATH, MODEL_CONFIG, LOG_DIR


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure logging for detailed debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),  # Save logs to file
        logging.StreamHandler(sys.stdout)        # Print logs to console
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Path configurations
# These specify where to find test data, model, and config files
TEST_CSV = TEST_CSV              # Path to test dataset CSV
MODEL_PATH = SAVED_MODEL_PATH    # Path to saved model weights
MODEL_CONFIG = MODEL_CONFIG       # Path to model configuration JSON
EVAL_LOG_DIR = LOG_DIR           # Directory to save evaluation results

# Evaluation hyperparameters
BATCH_SIZE = 1      # Number of samples to process at once
                    # Usually set to 1 for evaluation to process one at a time
                    # Increase (e.g., 32) if you have many test samples

MAX_TEXT_LEN = 128  # Maximum text sequence length
                    # Should match the length used during training

# Device configuration
# Use GPU if available for faster evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("=" * 80)
logger.info("EVALUATION CONFIGURATION")
logger.info("=" * 80)
logger.info(f"Device: {DEVICE}")
logger.info(f"Batch Size: {BATCH_SIZE}")
logger.info(f"Test CSV: {TEST_CSV}")
logger.info(f"Model Path: {MODEL_PATH}")
logger.info(f"Config Path: {MODEL_CONFIG}")
logger.info(f"Output Directory: {EVAL_LOG_DIR}")
logger.info("=" * 80)


# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================

def setup_directories() -> None:
    """
    Create necessary directories for evaluation outputs.
    
    This function ensures the evaluation log directory exists
    for saving confusion matrices and reports.
    
    Raises:
        OSError: If directory creation fails
    """
    try:
        logger.info("Setting up evaluation directories...")
        
        # Create evaluation log directory
        os.makedirs(EVAL_LOG_DIR, exist_ok=True)
        logger.info(f"✓ Evaluation directory ready: {EVAL_LOG_DIR}")
        
    except OSError as e:
        error_msg = f"Failed to create evaluation directory: {str(e)}"
        logger.error(error_msg)
        raise OSError(error_msg)


def load_config() -> Dict[str, Any]:
    """
    Load model configuration from JSON file.
    
    The configuration contains model architecture settings that were
    used during training. We need this to reconstruct the model.
    
    Returns:
        dict: Model configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
        ValueError: If config is missing required fields
    """
    try:
        logger.info("Loading model configuration...")
        
        # Step 1: Check if config file exists
        if not os.path.exists(MODEL_CONFIG):
            error_msg = f"Config file not found: {MODEL_CONFIG}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Step 2: Load JSON file
        logger.info(f"Reading config from: {MODEL_CONFIG}")
        with open(MODEL_CONFIG, "r") as f:
            cfg = json.load(f)
        
        # Step 3: Validate required fields
        required_fields = ["IMAGE_BACKBONE", "TEXT_ENCODER", "HIDDEN_DIM", "DROPOUT"]
        missing_fields = [field for field in required_fields if field not in cfg]
        
        if missing_fields:
            error_msg = f"Config missing required fields: {missing_fields}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Step 4: Log configuration details
        logger.info("✓ Configuration loaded successfully")
        logger.info(f"  - Image Backbone: {cfg.get('IMAGE_BACKBONE')}")
        logger.info(f"  - Text Encoder: {cfg.get('TEXT_ENCODER')}")
        logger.info(f"  - Hidden Dim: {cfg.get('HIDDEN_DIM')}")
        logger.info(f"  - Fusion Type: {cfg.get('FUSION_TYPE', 'concat')}")
        
        return cfg
        
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in config file: {str(e)}"
        logger.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Failed to load config: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg)


def load_dataset() -> Tuple[torch.utils.data.DataLoader, bool, CustomDataset]:
    """
    Load test dataset and create data loader.
    
    This function loads the test data and determines whether we're
    in legacy mode (single label) or multi-attribute mode.
    
    Returns:
        tuple: (test_loader, legacy_mode, test_dataset)
    
    Raises:
        FileNotFoundError: If test CSV doesn't exist
        ValueError: If dataset is empty
        RuntimeError: If data loading fails
    """
    try:
        logger.info("Loading test dataset...")
        
        # Step 1: Check if test CSV exists
        if not os.path.exists(TEST_CSV):
            error_msg = f"Test CSV not found: {TEST_CSV}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Step 2: Load dataset
        logger.info(f"Reading from: {TEST_CSV}")
        test_dataset = CustomDataset(TEST_CSV)
        
        # Step 3: Validate dataset
        if len(test_dataset) == 0:
            error_msg = "Test dataset is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"✓ Loaded {len(test_dataset)} test samples")
        
        # Step 4: Determine evaluation mode
        # Legacy mode: single sentiment label
        # Multi-attribute mode: multiple attributes per sample
        legacy_mode = getattr(test_dataset, 'legacy_mode', True)
        
        if legacy_mode:
            logger.info("Evaluation mode: Legacy (single label)")
        else:
            available_attributes = getattr(test_dataset, 'available_attributes', [])
            logger.info("Evaluation mode: Multi-attribute")
            logger.info(f"Available attributes: {available_attributes}")
        
        # Step 5: Create data loader
        # shuffle=False: Keep order consistent for reproducible evaluation
        # num_workers=0: Avoid multiprocessing issues during evaluation
        logger.info(f"Creating data loader with batch size: {BATCH_SIZE}")
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,  # Single process for evaluation
            pin_memory=True if DEVICE == "cuda" else False
        )
        
        logger.info(f"✓ Data loader created with {len(test_loader)} batches")
        
        return test_loader, legacy_mode, test_dataset
        
    except FileNotFoundError:
        raise
    except Exception as e:
        error_msg = f"Failed to load dataset: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg)


def load_model(cfg: Dict[str, Any], device: str) -> FG_MFN:
    """
    Load trained model from checkpoint.
    
    This function creates a model instance and loads the trained
    weights from a checkpoint file.
    
    Args:
        cfg: Model configuration dictionary
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        FG_MFN: Loaded model in evaluation mode
    
    Raises:
        FileNotFoundError: If model checkpoint doesn't exist
        RuntimeError: If model loading fails
    """
    try:
        logger.info("Loading trained model...")
        
        # Step 1: Check if model checkpoint exists
        if not os.path.exists(MODEL_PATH):
            error_msg = f"Model checkpoint not found: {MODEL_PATH}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Step 2: Create model architecture
        logger.info("Creating model architecture...")
        model = FG_MFN(cfg)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        logger.info(f"Attribute heads: {list(model.attribute_heads.keys())}")
        
        # Step 3: Load trained weights
        logger.info(f"Loading weights from: {MODEL_PATH}")
        
        try:
            # map_location ensures weights are loaded to correct device
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("✓ Weights loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load model weights: {str(e)}"
            logger.error(error_msg)
            logger.error("Make sure the model architecture matches the checkpoint")
            raise RuntimeError(error_msg)
        
        # Step 4: Move model to device
        model = model.to(device)
        logger.info(f"✓ Model moved to device: {device}")
        
        # Step 5: Set model to evaluation mode
        # This disables dropout and batch normalization training behavior
        model.eval()
        logger.info("✓ Model set to evaluation mode")
        
        return model
        
    except FileNotFoundError:
        raise
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(
    model: FG_MFN,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    legacy_mode: bool
) -> Tuple[Dict[str, List], Dict[str, List]]:
    """
    Run model evaluation on test dataset.
    
    This function processes all test samples through the model
    and collects predictions and ground truth labels.
    
    Args:
        model: Trained model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        legacy_mode: Whether to use legacy single-label mode
    
    Returns:
        tuple: (all_preds, all_labels) - dictionaries mapping attributes to lists
    
    Raises:
        RuntimeError: If evaluation fails
    """
    try:
        logger.info("=" * 80)
        logger.info("Starting Model Evaluation")
        logger.info("=" * 80)
        
        # Step 1: Initialize storage for predictions and labels
        # We store predictions and labels for each attribute separately
        all_preds = {attr: [] for attr in ATTRIBUTE_NAMES}
        all_labels = {attr: [] for attr in ATTRIBUTE_NAMES}
        
        # Step 2: Disable gradient computation
        # We don't need gradients during evaluation (saves memory and computation)
        with torch.no_grad():
            
            # Step 3: Create progress bar for visual feedback
            progress_bar = tqdm(
                test_loader,
                desc="Evaluating",
                total=len(test_loader),
                unit="batch"
            )
            
            # Step 4: Process each batch
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Step 4a: Move data to device
                    images = batch["visual"].to(device)
                    texts = batch["text"].to(device)
                    masks = batch["attention_mask"].to(device)
                    
                    # Step 4b: Forward pass through model
                    # outputs is a dict: {attr_name: logits_tensor}
                    outputs = model(images, texts, attention_mask=masks)
                    
                    # Step 4c: Process predictions based on mode
                    if legacy_mode:
                        # Legacy mode: single sentiment label
                        # Labels are expected as-is (not adjusted)
                        labels = batch["label"]
                        
                        # Get predictions from sentiment head (or first available head)
                        if "sentiment" in outputs:
                            preds = torch.argmax(outputs["sentiment"], dim=1).cpu().numpy()
                        else:
                            # Fallback to first available head
                            first_key = list(outputs.keys())[0]
                            preds = torch.argmax(outputs[first_key], dim=1).cpu().numpy()
                            logger.debug(f"Using {first_key} head for legacy mode")
                        
                        # Store predictions and labels
                        all_preds["sentiment"].extend(preds)
                        all_labels["sentiment"].extend(labels.cpu().numpy())
                        
                    else:
                        # Multi-attribute mode: process each attribute
                        for attr in ATTRIBUTE_NAMES:
                            # Check if this attribute is available in both outputs and batch
                            if attr in outputs and attr in batch:
                                # Get predicted class (argmax of logits)
                                preds = torch.argmax(outputs[attr], dim=1).cpu().numpy()
                                
                                # Get ground truth labels
                                labels = batch[attr].cpu().numpy()
                                
                                # Store predictions and labels
                                all_preds[attr].extend(preds)
                                all_labels[attr].extend(labels)
                    
                    # Step 4d: Update progress bar
                    progress_bar.set_postfix({
                        'batch': f'{batch_idx + 1}/{len(test_loader)}'
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing batch {batch_idx}: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    # Continue with next batch instead of failing completely
                    continue
            
            progress_bar.close()
        
        # Step 5: Validate results
        # Check that we collected predictions for at least one attribute
        has_predictions = any(len(preds) > 0 for preds in all_preds.values())
        
        if not has_predictions:
            error_msg = "No predictions collected during evaluation"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Step 6: Log summary
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Complete")
        logger.info("=" * 80)
        
        for attr in ATTRIBUTE_NAMES:
            if all_preds[attr]:
                logger.info(f"{attr}: {len(all_preds[attr])} predictions collected")
        
        logger.info("=" * 80)
        
        return all_preds, all_labels
        
    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg)


def compute_metrics(
    all_preds: Dict[str, List],
    all_labels: Dict[str, List],
    cfg: Dict[str, Any]
) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    Compute evaluation metrics for all attributes.
    
    This function calculates accuracy, F1 score, and confusion matrices
    for each attribute that has predictions.
    
    Args:
        all_preds: Dictionary mapping attributes to prediction lists
        all_labels: Dictionary mapping attributes to label lists
        cfg: Model configuration (for label names)
    
    Returns:
        tuple: (report_data, confusion_matrices)
               report_data: List of metric dictionaries
               confusion_matrices: Dict mapping attributes to confusion matrices
    
    Raises:
        RuntimeError: If metric computation fails
    """
    try:
        logger.info("\n" + "=" * 80)
        logger.info("Computing Metrics")
        logger.info("=" * 80)
        
        # Step 1: Initialize storage
        report_data = []  # List of metric dictionaries
        confusion_matrices = {}  # Store confusion matrices for visualization
        
        # Step 2: Process each attribute
        for attr in ATTRIBUTE_NAMES:
            # Check if we have predictions for this attribute
            if not all_preds[attr]:
                logger.debug(f"Skipping {attr} - no predictions")
                continue
            
            logger.info(f"\nComputing metrics for: {attr}")
            logger.info("-" * 40)
            
            try:
                # Step 2a: Convert to numpy arrays for sklearn
                preds = np.array(all_preds[attr])
                labels = np.array(all_labels[attr])
                
                # Step 2b: Validate data
                if len(preds) != len(labels):
                    logger.error(f"{attr}: Mismatch in predictions ({len(preds)}) and labels ({len(labels)})")
                    continue
                
                logger.info(f"  Samples: {len(preds)}")
                
                # Step 2c: Compute accuracy
                # Accuracy = (number of correct predictions) / (total predictions)
                accuracy = accuracy_score(labels, preds)
                logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                
                # Step 2d: Compute F1 score
                # F1 = harmonic mean of precision and recall
                # macro = unweighted mean (treats all classes equally)
                f1 = f1_score(
                    labels,
                    preds,
                    average='macro',
                    zero_division=0  # Return 0 if no predictions for a class
                )
                logger.info(f"  Macro F1: {f1:.4f}")
                
                # Step 2e: Store metrics for report
                report_data.append({
                    "attribute": attr,
                    "accuracy": accuracy,
                    "macro_f1": f1
                })
                
                # Step 2f: Get label information
                # Try to get label names from config for better visualization
                if "ATTRIBUTES" in cfg and attr in cfg["ATTRIBUTES"]:
                    label_names = cfg["ATTRIBUTES"][attr]["labels"]
                    num_classes = cfg["ATTRIBUTES"][attr]["num_classes"]
                    logger.info(f"  Classes: {label_names}")
                else:
                    # Fallback: use numeric labels
                    label_names = None
                    num_classes = len(set(labels))
                    logger.info(f"  Number of classes: {num_classes}")
                
                # Step 2g: Compute confusion matrix
                # Confusion matrix shows true vs predicted labels
                # Rows = true labels, Columns = predicted labels
                cm = confusion_matrix(
                    labels,
                    preds,
                    labels=range(num_classes)
                )
                
                # Store confusion matrix with metadata
                confusion_matrices[attr] = {
                    'matrix': cm,
                    'label_names': label_names,
                    'num_classes': num_classes
                }
                
                logger.info(f"  Confusion matrix shape: {cm.shape}")
                
                # Step 2h: Log per-class accuracy
                logger.info("  Per-class accuracy:")
                for i in range(num_classes):
                    class_total = np.sum(labels == i)
                    class_correct = cm[i, i]
                    class_acc = class_correct / class_total if class_total > 0 else 0
                    
                    class_name = label_names[i] if label_names else f"Class {i}"
                    logger.info(f"    {class_name}: {class_acc:.4f} ({class_correct}/{class_total})")
                
            except Exception as e:
                error_msg = f"Failed to compute metrics for {attr}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                # Continue with other attributes
                continue
        
        # Step 3: Validate results
        if not report_data:
            error_msg = "No metrics were computed"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✓ Computed metrics for {len(report_data)} attributes")
        logger.info("=" * 80)
        
        return report_data, confusion_matrices
        
    except Exception as e:
        error_msg = f"Failed to compute metrics: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg)


def save_results(
    report_data: List[Dict],
    confusion_matrices: Dict[str, Dict],
    output_dir: str
) -> None:
    """
    Save evaluation results including metrics report and confusion matrices.
    
    This function saves:
    1. CSV report with accuracy and F1 scores for each attribute
    2. PNG images of confusion matrices for visualization
    
    Args:
        report_data: List of metric dictionaries
        confusion_matrices: Dict mapping attributes to confusion matrix data
        output_dir: Directory to save results
    
    Raises:
        OSError: If file writing fails
        RuntimeError: If saving fails
    """
    try:
        logger.info("\n" + "=" * 80)
        logger.info("Saving Evaluation Results")
        logger.info("=" * 80)
        
        # Step 1: Save metrics report as CSV
        logger.info("\nSaving metrics report...")
        
        try:
            # Create DataFrame from report data
            report_df = pd.DataFrame(report_data)
            
            # Save to CSV
            report_path = os.path.join(output_dir, "evaluation_report.csv")
            report_df.to_csv(report_path, index=False)
            
            logger.info(f"✓ Metrics report saved: {report_path}")
            logger.info(f"  Columns: {list(report_df.columns)}")
            logger.info(f"  Rows: {len(report_df)}")
            
        except Exception as e:
            error_msg = f"Failed to save metrics report: {str(e)}"
            logger.error(error_msg)
            raise OSError(error_msg)
        
        # Step 2: Generate and save confusion matrix visualizations
        logger.info("\nGenerating confusion matrices...")
        
        saved_count = 0
        for attr, cm_data in confusion_matrices.items():
            try:
                # Extract data
                cm = cm_data['matrix']
                label_names = cm_data['label_names']
                num_classes = cm_data['num_classes']
                
                logger.info(f"  Creating visualization for: {attr}")
                
                # Step 2a: Create figure
                # Size depends on number of classes
                fig_size = max(8, num_classes * 1.5)
                cm_fig, ax = plt.subplots(figsize=(fig_size, fig_size))
                
                # Step 2b: Create confusion matrix display
                if label_names:
                    # Use custom label names if available
                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=cm,
                        display_labels=label_names
                    )
                else:
                    # Use default numeric labels
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                
                # Step 2c: Plot the confusion matrix
                # cmap=plt.cm.Blues creates a blue color scheme
                # Darker blue = more samples
                disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
                
                # Step 2d: Customize plot
                plt.title(f"Confusion Matrix - {attr.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
                
                # Rotate x-axis labels if they're long
                if label_names and any(len(label) > 10 for label in label_names):
                    plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                
                # Step 2e: Save figure
                save_path = os.path.join(output_dir, f"confusion_matrix_{attr}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(cm_fig)
                
                logger.info(f"    ✓ Saved: {save_path}")
                saved_count += 1
                
            except Exception as e:
                error_msg = f"Failed to save confusion matrix for {attr}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                # Continue with other attributes
                continue
        
        logger.info(f"\n✓ Saved {saved_count} confusion matrices")
        
        # Step 3: Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Results Summary")
        logger.info("=" * 80)
        
        if report_data:
            # Calculate overall average metrics
            avg_accuracy = np.mean([r["accuracy"] for r in report_data])
            avg_f1 = np.mean([r["macro_f1"] for r in report_data])
            
            logger.info(f"\n=== OVERALL PERFORMANCE ===")
            logger.info(f"Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
            logger.info(f"Average Macro F1: {avg_f1:.4f}")
            
            logger.info(f"\n=== PER-ATTRIBUTE RESULTS ===")
            for item in report_data:
                attr = item['attribute']
                acc = item['accuracy']
                f1 = item['macro_f1']
                logger.info(f"{attr:20s}: Acc={acc:.4f} ({acc*100:.2f}%), F1={f1:.4f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("All results saved successfully!")
        logger.info("=" * 80)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Files created:")
        logger.info(f"  - evaluation_report.csv")
        logger.info(f"  - confusion_matrix_*.png ({saved_count} files)")
        logger.info("=" * 80)
        
    except Exception as e:
        error_msg = f"Failed to save results: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg)


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def main():
    """
    Main evaluation function.
    
    This is the entry point for the evaluation script. It:
    1. Sets up the environment
    2. Loads configuration, dataset, and model
    3. Runs evaluation
    4. Computes metrics
    5. Saves results
    
    The function handles all error cases and ensures proper cleanup.
    
    Raises:
        RuntimeError: If evaluation setup or execution fails
    """
    try:
        # Print header
        logger.info("\n" + "=" * 80)
        logger.info("MODEL EVALUATION SCRIPT")
        logger.info("=" * 80)
        logger.info("This script evaluates a trained model on test data.")
        logger.info("=" * 80)
        
        # =====================================================================
        # STEP 1: SETUP
        # =====================================================================
        
        logger.info("\n[STEP 1/6] Setting up evaluation environment...")
        
        # Create output directories
        setup_directories()
        
        logger.info("✓ Setup complete\n")
        
        # =====================================================================
        # STEP 2: LOAD CONFIGURATION
        # =====================================================================
        
        logger.info("[STEP 2/6] Loading model configuration...")
        
        cfg = load_config()
        
        logger.info("✓ Configuration loaded\n")
        
        # =====================================================================
        # STEP 3: LOAD DATASET
        # =====================================================================
        
        logger.info("[STEP 3/6] Loading test dataset...")
        
        test_loader, legacy_mode, test_dataset = load_dataset()
        
        logger.info("✓ Dataset loaded\n")
        
        # =====================================================================
        # STEP 4: LOAD MODEL
        # =====================================================================
        
        logger.info("[STEP 4/6] Loading trained model...")
        
        model = load_model(cfg, DEVICE)
        
        logger.info("✓ Model loaded\n")
        
        # =====================================================================
        # STEP 5: RUN EVALUATION
        # =====================================================================
        
        logger.info("[STEP 5/6] Running model evaluation...")
        
        all_preds, all_labels = evaluate_model(
            model,
            test_loader,
            DEVICE,
            legacy_mode
        )
        
        logger.info("✓ Evaluation complete\n")
        
        # =====================================================================
        # STEP 6: COMPUTE METRICS AND SAVE RESULTS
        # =====================================================================
        
        logger.info("[STEP 6/6] Computing metrics and saving results...")
        
        # Compute metrics
        report_data, confusion_matrices = compute_metrics(
            all_preds,
            all_labels,
            cfg
        )
        
        # Save results
        save_results(
            report_data,
            confusion_matrices,
            EVAL_LOG_DIR
        )
        
        logger.info("✓ Results saved\n")
        
        # =====================================================================
        # COMPLETION
        # =====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY! 🎉")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        logger.info("\n" + "=" * 80)
        logger.info("⚠ Evaluation interrupted by user")
        logger.info("=" * 80)
        sys.exit(0)
        
    except Exception as e:
        # Handle any unexpected errors
        logger.error("\n" + "=" * 80)
        logger.error("❌ EVALUATION FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
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
   Purpose: Create necessary directories for evaluation outputs
   Use Cases:
     - Initialize evaluation environment
     - Ensure output directory exists
     - Called at start of evaluation
   
   Example:
     setup_directories()


2. load_config()
   Purpose: Load model configuration from JSON file
   Use Cases:
     - Read model architecture settings
     - Get attribute definitions
     - Reconstruct model for evaluation
   
   Returns:
     dict: Model configuration
   
   Example:
     cfg = load_config()
     print(cfg['HIDDEN_DIM'])


3. load_dataset()
   Purpose: Load test dataset and create data loader
   Use Cases:
     - Read test data from CSV
     - Determine evaluation mode (legacy vs multi-attribute)
     - Create batched data loader
   
   Returns:
     tuple: (test_loader, legacy_mode, test_dataset)
   
   Example:
     loader, is_legacy, dataset = load_dataset()
     print(f"Test samples: {len(dataset)}")


4. load_model(cfg, device)
   Purpose: Load trained model from checkpoint
   Use Cases:
     - Reconstruct model architecture
     - Load trained weights
     - Prepare model for evaluation
   
   Returns:
     FG_MFN: Loaded model in eval mode
   
   Example:
     model = load_model(cfg, 'cuda')


5. evaluate_model(model, test_loader, device, legacy_mode)
   Purpose: Run model evaluation on test dataset
   Use Cases:
     - Process all test samples
     - Collect predictions and labels
     - Generate model outputs
   
   Returns:
     tuple: (all_preds, all_labels) - dicts of predictions and labels
   
   Example:
     preds, labels = evaluate_model(model, loader, 'cuda', False)


6. compute_metrics(all_preds, all_labels, cfg)
   Purpose: Calculate evaluation metrics
   Use Cases:
     - Compute accuracy and F1 scores
     - Generate confusion matrices
     - Analyze per-class performance
   
   Returns:
     tuple: (report_data, confusion_matrices)
   
   Example:
     metrics, cm_data = compute_metrics(preds, labels, cfg)


7. save_results(report_data, confusion_matrices, output_dir)
   Purpose: Save evaluation results to disk
   Use Cases:
     - Save metrics as CSV report
     - Generate confusion matrix visualizations
     - Create evaluation documentation
   
   Example:
     save_results(metrics, cm_data, 'logs/')


8. main()
   Purpose: Main evaluation orchestration
   Use Cases:
     - Complete evaluation pipeline
     - Handle all setup and execution
     - Coordinate all evaluation steps
   
   Example:
     main()  # Run full evaluation


TYPICAL WORKFLOW
================

1. Prepare for Evaluation:
   - Train a model and save checkpoint
   - Prepare test.csv with test data
   - Ensure images are in correct directories
   - Set paths in utils/path.py

2. Configure Evaluation:
   # Edit paths at top of script if needed
   TEST_CSV = "data/test.csv"
   MODEL_PATH = "saved_models/model_best.pt"
   MODEL_CONFIG = "configs/model_config.json"

3. Run Evaluation:
   python training/evaluate.py

4. View Results:
   - Check evaluation_report.csv for metrics
   - View confusion_matrix_*.png for visualizations
   - Review evaluation.log for detailed logs

5. Analyze Results:
   import pandas as pd
   
   # Read metrics
   df = pd.read_csv('logs/evaluation_report.csv')
   print(df)
   
   # Find best/worst attributes
   best = df.loc[df['accuracy'].idxmax()]
   worst = df.loc[df['accuracy'].idxmin()]


COMMAND LINE USAGE
==================

Basic usage:
  python training/evaluate.py

With logging redirect:
  python training/evaluate.py 2>&1 | tee evaluation_output.txt

Background process:
  nohup python training/evaluate.py > evaluation.out 2>&1 &


UNDERSTANDING CONFUSION MATRIX
==============================

A confusion matrix shows prediction accuracy per class:

              Predicted
              Neg  Neu  Pos
Actual Neg  [ 90   5    5 ]  ← 90 correct, 5 confused with Neutral, 5 with Positive
       Neu  [  3  85   12 ]  ← 3 confused with Negative, 85 correct, etc.
       Pos  [  2   8   90 ]

- Diagonal values (90, 85, 90) = correct predictions
- Off-diagonal values = misclassifications
- Rows sum to total samples per true class
- Columns sum to total predictions per class

Good model: High values on diagonal, low elsewhere
Poor model: Values spread across matrix


METRICS EXPLAINED
==================

1. Accuracy:
   - Percentage of correct predictions
   - Formula: (Correct Predictions) / (Total Predictions)
   - Range: 0.0 (0%) to 1.0 (100%)
   - Higher is better

2. Macro F1 Score:
   - Average of F1 scores for each class
   - Treats all classes equally (good for imbalanced data)
   - Range: 0.0 (worst) to 1.0 (best)
   - Higher is better
   
   F1 for each class = 2 * (Precision * Recall) / (Precision + Recall)
   Macro F1 = Average of all class F1 scores


TROUBLESHOOTING
===============

1. Model Not Found:
   - Check MODEL_PATH points to correct checkpoint
   - Ensure training completed and saved model
   - Verify file permissions

2. Test Data Issues:
   - Ensure TEST_CSV exists and is readable
   - Check CSV format matches training data
   - Verify image paths are correct

3. Out of Memory:
   - Reduce BATCH_SIZE (try 1)
   - Use CPU instead of GPU (slower but uses less memory)
   - Process fewer samples at a time

4. Poor Performance:
   - Check if using correct model checkpoint (best vs last)
   - Verify test data is from same distribution as training
   - Check for data preprocessing consistency
   - Review confusion matrix to identify problem classes

5. Missing Attributes:
   - Ensure test data has all required attributes
   - Check attribute names match model configuration
   - Verify dataset mode (legacy vs multi-attribute)


OUTPUT FILES
============

After evaluation, you'll find:

logs/
  ├── evaluation_report.csv        # Metrics summary
  ├── confusion_matrix_*.png        # One per attribute
  └── evaluation.log                # Detailed logs

evaluation_report.csv contains:
  - attribute: Name of predicted attribute
  - accuracy: Overall accuracy for that attribute
  - macro_f1: Macro-averaged F1 score


INTERPRETING RESULTS
====================

Good Performance:
  - Accuracy > 0.80 (80%)
  - Macro F1 > 0.75
  - Diagonal-dominant confusion matrix

Average Performance:
  - Accuracy: 0.60-0.80
  - Macro F1: 0.50-0.75
  - Some off-diagonal confusion

Poor Performance:
  - Accuracy < 0.60
  - Macro F1 < 0.50
  - Highly scattered confusion matrix

If performance is poor:
  1. Collect more training data
  2. Try different model architecture
  3. Adjust learning rate / training epochs
  4. Check data quality and labels
  5. Consider data augmentation
"""
