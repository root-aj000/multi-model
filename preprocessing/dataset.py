import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any

# Import our preprocessing functions
from preprocessing.image_preprocessing import resize_image, normalize_image
from preprocessing.text_preprocessing import clean_text, tokenize_text
from preprocessing.augmentation import augment_image
from utils.path import IMAGE_DIR, TRAIN_CSV

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

# Define paths
IMAGE_DIR = IMAGE_DIR

# Maximum length for tokenized text
# This ensures all text sequences have the same length for batching
MAX_TEXT_LEN = 128

# Default image size if we need to create a placeholder
# This matches the standard input size for many neural networks
DEFAULT_IMAGE_SIZE = (224, 224, 3)

# Attribute names that we train on (must match model config)
# These are the different labels/attributes we want to predict
ATTRIBUTE_NAMES = [
    "theme",              # Topic or theme of the content
    "sentiment",          # Positive, negative, or neutral sentiment
    "emotion",            # Specific emotion (happy, sad, angry, etc.)
    "dominant_colour",    # Main color in the image
    "attention_score",    # How attention-grabbing the content is
    "trust_safety",       # Trust and safety rating
    "target_audience",    # Intended audience demographic
    "predicted_ctr",      # Predicted click-through rate category
    "likelihood_shares"   # Likelihood of being shared
]


# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================

class CustomDataset(Dataset):
    """
    PyTorch Dataset class for loading and preprocessing multimodal data.
    
    This dataset handles both images and text, and supports multiple labels.
    It can work in two modes:
    1. Legacy mode: Single label called "label_num"
    2. Multi-attribute mode: Multiple attributes from ATTRIBUTE_NAMES
    
    Why we need this:
    - PyTorch DataLoader needs a Dataset class to load data in batches
    - This class handles all preprocessing automatically
    - It supports data augmentation for training
    - It's memory efficient (loads data on-demand, not all at once)
    
    Example:
        >>> dataset = CustomDataset('train.csv', augment=True)
        >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        >>> for batch in dataloader:
        >>>     images = batch['visual']
        >>>     text = batch['text']
        >>>     labels = batch['theme']  # or any other attribute
    """
    
    def __init__(self, csv_path, image_dir=IMAGE_DIR, augment=False):
        """
        Initialize the dataset.
        
        Args:
            csv_path (str): Path to CSV file with columns: 
                           - image_path: filename of the image
                           - text: text content
                           - attribute columns (e.g., theme_num, sentiment_num)
            image_dir (str): Directory containing the images
            augment (bool): Whether to apply image augmentation
                           Set to True for training, False for validation/test
        """
        
        logger.info("=" * 70)
        logger.info("Initializing CustomDataset")
        logger.info("=" * 70)
        
        # ====================================================================
        # STEP 1: VALIDATE AND STORE INPUT PARAMETERS
        # ====================================================================
        
        # Validate csv_path
        try:
            if csv_path is None or csv_path == "":
                error_msg = "csv_path is None or empty"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not os.path.exists(csv_path):
                error_msg = f"CSV file does not exist: {csv_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            if not os.path.isfile(csv_path):
                error_msg = f"CSV path is not a file: {csv_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"CSV path validated: {csv_path}")
            
        except Exception as e:
            logger.error(f"Error validating csv_path: {str(e)}")
            raise
        
        # Validate image_dir
        try:
            if image_dir is None or image_dir == "":
                error_msg = "image_dir is None or empty"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not os.path.exists(image_dir):
                error_msg = f"Image directory does not exist: {image_dir}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            if not os.path.isdir(image_dir):
                error_msg = f"Image path is not a directory: {image_dir}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"Image directory validated: {image_dir}")
            
        except Exception as e:
            logger.error(f"Error validating image_dir: {str(e)}")
            raise
        
        # Store parameters
        self.image_dir = image_dir
        self.augment = augment
        
        logger.info(f"Augmentation enabled: {self.augment}")
        
        # ====================================================================
        # STEP 2: LOAD CSV FILE
        # ====================================================================
        
        try:
            # Read CSV file into pandas DataFrame
            self.df = pd.read_csv(csv_path)
            
            logger.info(f"Successfully loaded CSV with {len(self.df)} rows")
            logger.debug(f"CSV columns: {list(self.df.columns)}")
            
            # Check if DataFrame is empty
            if len(self.df) == 0:
                logger.warning("CSV file is empty (no data rows)")
            
            # Validate required columns
            required_columns = ["image_path", "text"]
            missing_columns = []
            
            for col in required_columns:
                if col not in self.df.columns:
                    missing_columns.append(col)
            
            if missing_columns:
                error_msg = (
                    f"CSV is missing required columns: {missing_columns}. "
                    f"Available columns: {list(self.df.columns)}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info("Required columns validated")
            
        except pd.errors.EmptyDataError:
            error_msg = f"CSV file is empty: {csv_path}"
            logger.error(error_msg)
            raise
            
        except pd.errors.ParserError as e:
            error_msg = f"Error parsing CSV file: {str(e)}"
            logger.error(error_msg)
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error loading CSV: {str(e)}")
            raise
        
        # ====================================================================
        # STEP 3: DETERMINE AVAILABLE ATTRIBUTES
        # ====================================================================
        # Check which attribute columns are present in the CSV
        # This allows the dataset to work with different CSV formats
        
        self.available_attributes = []
        
        logger.info("Checking for attribute columns...")
        
        # Loop through all possible attributes
        for attr in ATTRIBUTE_NAMES:
            # Attribute columns should be named like "theme_num", "sentiment_num"
            col_name = f"{attr}_num"
            
            # Check if this column exists in the CSV
            if col_name in self.df.columns:
                self.available_attributes.append(attr)
                logger.debug(f"  Found attribute: {attr} (column: {col_name})")
        
        logger.info(
            f"Found {len(self.available_attributes)} attributes: "
            f"{self.available_attributes}"
        )
        
        # ====================================================================
        # STEP 4: CHECK FOR LEGACY MODE
        # ====================================================================
        # Backwards compatibility with older CSV format
        # Old format had a single "label_num" column instead of multiple attributes
        
        # We're in legacy mode if:
        # 1. The old "label_num" column exists, AND
        # 2. No new attribute columns were found
        self.legacy_mode = (
            "label_num" in self.df.columns and 
            len(self.available_attributes) == 0
        )
        
        if self.legacy_mode:
            logger.info("Operating in LEGACY MODE (using single label_num column)")
        else:
            logger.info("Operating in MULTI-ATTRIBUTE MODE")
        
        # Log warning if no labels found at all
        if not self.legacy_mode and len(self.available_attributes) == 0:
            logger.warning(
                "No attribute columns found and not in legacy mode. "
                "Dataset will use default values (0) for all attributes."
            )
        
        logger.info("=" * 70)
        logger.info("Dataset initialization complete")
        logger.info("=" * 70)
    
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        This is called by PyTorch DataLoader to know how many batches to create.
        
        Returns:
            int: Number of samples (rows in the CSV)
        
        Example:
            >>> dataset = CustomDataset('train.csv')
            >>> print(f"Dataset has {len(dataset)} samples")
        """
        try:
            # Simply return the number of rows in the DataFrame
            dataset_length = len(self.df)
            
            logger.debug(f"Dataset length requested: {dataset_length}")
            
            return dataset_length
            
        except Exception as e:
            logger.error(f"Error getting dataset length: {str(e)}")
            # Return 0 if something goes wrong
            return 0
    
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        This is called by PyTorch DataLoader to load individual samples.
        It loads and preprocesses the image and text for the given index.
        
        Args:
            idx (int): Index of the sample to load (0 to len(dataset)-1)
        
        Returns:
            output (dict): Dictionary containing:
                - visual: Image tensor [C, H, W]
                - text: Text token IDs [seq_len]
                - attention_mask: Attention mask for text [seq_len]
                - labels: Either single 'label' or multiple attribute tensors
        
        Example:
            >>> dataset = CustomDataset('train.csv')
            >>> sample = dataset[0]  # Get first sample
            >>> image = sample['visual']
            >>> text = sample['text']
        """
        
        logger.debug(f"Loading sample at index: {idx}")
        
        # ====================================================================
        # STEP 1: VALIDATE INDEX
        # ====================================================================
        
        try:
            # Check if index is valid
            if idx < 0 or idx >= len(self.df):
                error_msg = (
                    f"Index {idx} out of bounds. "
                    f"Valid range: 0 to {len(self.df)-1}"
                )
                logger.error(error_msg)
                raise IndexError(error_msg)
            
        except Exception as e:
            logger.error(f"Error validating index: {str(e)}")
            raise
        
        # ====================================================================
        # STEP 2: GET DATA ROW FROM DATAFRAME
        # ====================================================================
        
        try:
            # Get the row at the specified index
            # iloc is used for integer-based indexing
            row = self.df.iloc[idx]
            
            logger.debug(f"Retrieved row {idx} from DataFrame")
            
        except Exception as e:
            logger.error(f"Error retrieving row {idx}: {str(e)}")
            raise
        
        # ====================================================================
        # STEP 3: LOAD AND PREPROCESS IMAGE
        # ====================================================================
        
        try:
            # Get image filename from the row
            image_filename = row["image_path"]
            
            # Build full path to the image
            image_path = os.path.join(self.image_dir, image_filename)
            
            logger.debug(f"Loading image: {image_path}")
            
            # Try to load the image
            img = cv2.imread(image_path)
            
            # Handle case where image fails to load
            if img is None:
                logger.warning(
                    f"Failed to load image: {image_path}. "
                    f"Using black placeholder image."
                )
                
                # Create a black placeholder image
                # This prevents the training from crashing due to missing images
                # Shape: (height, width, channels)
                img = np.zeros(DEFAULT_IMAGE_SIZE, dtype=np.uint8)
                
            else:
                logger.debug(f"Image loaded successfully. Original shape: {img.shape}")
                
                # Resize image to standard size
                # This ensures all images have the same dimensions
                img = resize_image(img)
                
                if img is None:
                    logger.warning(
                        f"resize_image returned None for {image_path}. "
                        f"Using black placeholder."
                    )
                    img = np.zeros(DEFAULT_IMAGE_SIZE, dtype=np.uint8)
                else:
                    logger.debug(f"Image resized. New shape: {img.shape}")
                
                # Convert from BGR to RGB
                # OpenCV loads images in BGR format, but most models expect RGB
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    logger.debug("Converted image from BGR to RGB")
                except Exception as e:
                    logger.warning(
                        f"Failed to convert BGR to RGB: {str(e)}. "
                        f"Continuing with original format."
                    )
                
                # Apply augmentation if enabled
                # Only augment during training, not validation/test
                if self.augment:
                    augmented_img = augment_image(img)
                    
                    if augmented_img is None:
                        logger.warning(
                            f"augment_image returned None. "
                            f"Using non-augmented image."
                        )
                    else:
                        img = augmented_img
                        logger.debug("Image augmentation applied")
            
            # Normalize image and convert to tensor
            # This scales pixel values and converts to PyTorch format
            img_tensor = normalize_image(img)
            
            if img_tensor is None:
                logger.error(
                    f"normalize_image returned None for index {idx}. "
                    f"Creating zero tensor."
                )
                # Create a zero tensor as fallback
                # Shape: [3, 224, 224] (channels first)
                img_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
            else:
                logger.debug(f"Image normalized to tensor. Shape: {img_tensor.shape}")
            
        except KeyError as e:
            logger.error(f"Missing 'image_path' column in row {idx}: {str(e)}")
            # Create zero tensor as fallback
            img_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"Error processing image for index {idx}: {str(e)}")
            # Create zero tensor as fallback
            img_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
        
        # ====================================================================
        # STEP 4: LOAD AND PREPROCESS TEXT
        # ====================================================================
        
        try:
            # Get text from the row
            # Use .get() with default empty string to handle missing values
            text = row.get("text", "")
            
            # Handle NaN or None values
            if pd.isna(text):
                logger.debug(f"Text is NaN for index {idx}. Using empty string.")
                text = ""
            
            # Convert to string (in case it's not already)
            text = str(text)
            
            logger.debug(
                f"Raw text: '{text[:50]}...'" if len(text) > 50 
                else f"Raw text: '{text}'"
            )
            
            # Clean the text (remove special characters, lowercase, etc.)
            text = clean_text(text)
            
            if text is None:
                logger.warning(
                    f"clean_text returned None for index {idx}. "
                    f"Using empty string."
                )
                text = ""
            
            logger.debug("Text cleaned")
            
            # Tokenize text (convert to token IDs)
            # max_length ensures all sequences are the same length
            text_tensor = tokenize_text(text, max_length=MAX_TEXT_LEN)
            
            if text_tensor is None:
                logger.error(
                    f"tokenize_text returned None for index {idx}. "
                    f"Creating default tensors."
                )
                # Create default tensors as fallback
                text_tensor = {
                    "input_ids": torch.zeros(MAX_TEXT_LEN, dtype=torch.long),
                    "attention_mask": torch.zeros(MAX_TEXT_LEN, dtype=torch.long)
                }
            else:
                logger.debug("Text tokenized successfully")
            
            # Validate text_tensor structure
            if not isinstance(text_tensor, dict):
                logger.error(
                    f"tokenize_text did not return a dict for index {idx}. "
                    f"Got type: {type(text_tensor)}. Creating default tensors."
                )
                text_tensor = {
                    "input_ids": torch.zeros(MAX_TEXT_LEN, dtype=torch.long),
                    "attention_mask": torch.zeros(MAX_TEXT_LEN, dtype=torch.long)
                }
            
            # Validate required keys
            if "input_ids" not in text_tensor:
                logger.error(f"Missing 'input_ids' in text_tensor for index {idx}")
                text_tensor["input_ids"] = torch.zeros(MAX_TEXT_LEN, dtype=torch.long)
            
            if "attention_mask" not in text_tensor:
                logger.error(f"Missing 'attention_mask' in text_tensor for index {idx}")
                text_tensor["attention_mask"] = torch.zeros(MAX_TEXT_LEN, dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Error processing text for index {idx}: {str(e)}")
            # Create default tensors as fallback
            text_tensor = {
                "input_ids": torch.zeros(MAX_TEXT_LEN, dtype=torch.long),
                "attention_mask": torch.zeros(MAX_TEXT_LEN, dtype=torch.long)
            }
        
        # ====================================================================
        # STEP 5: BUILD OUTPUT DICTIONARY
        # ====================================================================
        
        try:
            # Create the output dictionary with image and text data
            # These keys are used by the model during training
            output = {
                "visual": img_tensor,                           # Image tensor [C,H,W]
                "text": text_tensor["input_ids"],              # Token IDs [seq_len]
                "attention_mask": text_tensor["attention_mask"] # Attention mask [seq_len]
            }
            
            logger.debug("Created base output dictionary")
            
        except Exception as e:
            logger.error(f"Error creating output dictionary for index {idx}: {str(e)}")
            raise
        
        # ====================================================================
        # STEP 6: ADD LABELS TO OUTPUT
        # ====================================================================
        
        try:
            if self.legacy_mode:
                # -----------------------------------------------------------
                # Legacy Mode: Single label
                # -----------------------------------------------------------
                # Use the old "label_num" column for backwards compatibility
                
                logger.debug("Processing label in legacy mode")
                
                # Get label value, default to 0 if missing
                label_value = row.get("label_num", 0)
                
                # Handle NaN values
                if pd.isna(label_value):
                    logger.warning(
                        f"Label is NaN for index {idx}. Using default value 0."
                    )
                    label_value = 0
                
                # Convert to integer
                try:
                    label_value = int(label_value)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Cannot convert label to int for index {idx}: {label_value}. "
                        f"Using default value 0. Error: {str(e)}"
                    )
                    label_value = 0
                
                # Add to output as tensor
                output["label"] = torch.tensor(label_value, dtype=torch.long)
                
                logger.debug(f"Added legacy label: {label_value}")
                
            else:
                # -----------------------------------------------------------
                # Multi-attribute Mode: Multiple labels
                # -----------------------------------------------------------
                # Use separate columns for each attribute
                
                logger.debug("Processing labels in multi-attribute mode")
                
                # Loop through all possible attributes
                for attr in ATTRIBUTE_NAMES:
                    # Column name format: "theme_num", "sentiment_num", etc.
                    col_name = f"{attr}_num"
                    
                    # Check if this column exists in the CSV
                    if col_name in self.df.columns:
                        # Get the value from the row
                        attr_value = row.get(col_name, 0)
                        
                        # Handle NaN values
                        if pd.isna(attr_value):
                            logger.debug(
                                f"Attribute {attr} is NaN for index {idx}. "
                                f"Using default value 0."
                            )
                            attr_value = 0
                        
                        # Convert to integer
                        try:
                            attr_value = int(attr_value)
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Cannot convert {attr} to int for index {idx}: "
                                f"{attr_value}. Using default value 0. Error: {str(e)}"
                            )
                            attr_value = 0
                        
                        # Add to output as tensor
                        # Key is the attribute name (e.g., "theme", "sentiment")
                        output[attr] = torch.tensor(attr_value, dtype=torch.long)
                        
                        logger.debug(f"Added attribute {attr}: {attr_value}")
                        
                    else:
                        # Column doesn't exist, use default value 0
                        logger.debug(
                            f"Attribute {attr} not in CSV for index {idx}. "
                            f"Using default value 0."
                        )
                        output[attr] = torch.tensor(0, dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Error processing labels for index {idx}: {str(e)}")
            # Add default label to prevent crash
            if self.legacy_mode:
                output["label"] = torch.tensor(0, dtype=torch.long)
            else:
                for attr in ATTRIBUTE_NAMES:
                    output[attr] = torch.tensor(0, dtype=torch.long)
        
        # ====================================================================
        # STEP 7: FINAL VALIDATION AND RETURN
        # ====================================================================
        
        try:
            # Validate output dictionary
            logger.debug(f"Sample {idx} keys: {output.keys()}")
            
            # Log tensor shapes for debugging
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    logger.debug(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    logger.debug(f"  {key}: type {type(value)}")
            
        except Exception as e:
            logger.error(f"Error during final validation for index {idx}: {str(e)}")
        
        # Return the output dictionary
        # Variable name is kept as 'output' as requested
        return output


# ============================================================================
# USAGE EXAMPLES AND TEST CASES
# ============================================================================

if __name__ == "__main__":
    """
    This section demonstrates how to use the CustomDataset class.
    It only runs when you execute this file directly (not when imported).
    """
    
    print("=" * 70)
    print("CUSTOM DATASET - USAGE EXAMPLES")
    print("=" * 70)
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 1: Basic Dataset Creation and Information
    # ------------------------------------------------------------------------
    print("USE CASE 1: Creating a basic dataset")
    print("-" * 70)
    
    try:
        train_csv = TRAIN_CSV
        print(f"Loading dataset from: {train_csv}")
        
        # Create dataset with augmentation enabled
        dataset = CustomDataset(train_csv, augment=True)
        
        print(f"✓ Dataset created successfully")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Available attributes: {dataset.available_attributes}")
        print(f"  Legacy mode: {dataset.legacy_mode}")
        print(f"  Augmentation: {dataset.augment}")
        
    except Exception as e:
        print(f"✗ Error creating dataset: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 2: Loading and Inspecting Individual Samples
    # ------------------------------------------------------------------------
    print("USE CASE 2: Loading individual samples")
    print("-" * 70)
    
    try:
        if 'dataset' in locals() and len(dataset) > 0:
            # Load first sample
            print("Loading sample at index 0...")
            sample = dataset[0]
            
            print(f"✓ Sample loaded successfully")
            print(f"  Keys in sample: {list(sample.keys())}")
            print()
            print("  Tensor shapes and types:")
            
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key:20s}: shape {str(value.shape):20s} dtype {value.dtype}")
                else:
                    print(f"    {key:20s}: type {type(value)}")
            
            # Show actual values for labels
            print()
            print("  Label values:")
            for key, value in sample.items():
                if key not in ["visual", "text", "attention_mask"]:
                    if isinstance(value, torch.Tensor):
                        print(f"    {key:20s}: {value.item()}")
        else:
            print("  No dataset available to load samples from")
        
    except Exception as e:
        print(f"✗ Error loading sample: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 3: Creating Dataset with Different Configurations
    # ------------------------------------------------------------------------
    print("USE CASE 3: Different dataset configurations")
    print("-" * 70)
    
    try:
        # Configuration 1: Training dataset with augmentation
        print("Configuration 1: Training dataset (with augmentation)")
        train_dataset = CustomDataset(TRAIN_CSV, augment=True)
        print(f"  ✓ Created training dataset: {len(train_dataset)} samples")
        
        # Configuration 2: Validation dataset without augmentation
        print("Configuration 2: Validation dataset (without augmentation)")
        val_dataset = CustomDataset(TRAIN_CSV, augment=False)  # Using TRAIN_CSV as example
        print(f"  ✓ Created validation dataset: {len(val_dataset)} samples")
        
        # Configuration 3: Custom image directory
        print("Configuration 3: Custom image directory")
        custom_dataset = CustomDataset(
            TRAIN_CSV, 
            image_dir=IMAGE_DIR,
            augment=False
        )
        print(f"  ✓ Created custom dataset: {len(custom_dataset)} samples")
        print(f"    Image directory: {custom_dataset.image_dir}")
        
    except Exception as e:
        print(f"✗ Error creating datasets: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 4: Using Dataset with PyTorch DataLoader
    # ------------------------------------------------------------------------
    print("USE CASE 4: Using with PyTorch DataLoader")
    print("-" * 70)
    
    try:
        from torch.utils.data import DataLoader
        
        if 'dataset' in locals() and len(dataset) > 0:
            # Create a DataLoader
            # DataLoader batches samples and handles shuffling
            batch_size = 4
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,      # Shuffle for training
                num_workers=0,     # Number of processes for loading data
                drop_last=False    # Keep last batch even if smaller
            )
            
            print(f"✓ Created DataLoader")
            print(f"  Batch size: {batch_size}")
            print(f"  Total batches: {len(dataloader)}")
            
            # Get one batch
            print()
            print("Loading first batch...")
            batch = next(iter(dataloader))
            
            print(f"  ✓ Batch loaded successfully")
            print(f"  Batch keys: {list(batch.keys())}")
            print()
            print("  Batch tensor shapes:")
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key:20s}: {value.shape}")
            
            # Show how to access data in the batch
            print()
            print("  Accessing batch data:")
            print(f"    Batch visual shape: {batch['visual'].shape}")
            print(f"    - Batch size: {batch['visual'].shape[0]}")
            print(f"    - Channels: {batch['visual'].shape[1]}")
            print(f"    - Height: {batch['visual'].shape[2]}")
            print(f"    - Width: {batch['visual'].shape[3]}")
            
        else:
            print("  No dataset available")
        
    except Exception as e:
        print(f"✗ Error with DataLoader: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 5: Iterating Through Multiple Samples
    # ------------------------------------------------------------------------
    print("USE CASE 5: Iterating through multiple samples")
    print("-" * 70)
    
    try:
        if 'dataset' in locals() and len(dataset) > 0:
            # Get first 3 samples (or fewer if dataset is small)
            num_samples = min(3, len(dataset))
            
            print(f"Loading first {num_samples} samples...")
            
            for i in range(num_samples):
                sample = dataset[i]
                
                # Get image statistics
                img_tensor = sample['visual']
                img_min = img_tensor.min().item()
                img_max = img_tensor.max().item()
                img_mean = img_tensor.mean().item()
                
                print(f"  Sample {i}:")
                print(f"    Visual: min={img_min:.3f}, max={img_max:.3f}, mean={img_mean:.3f}")
                
                # Get text length
                text_tensor = sample['text']
                non_zero = (text_tensor != 0).sum().item()
                print(f"    Text: {non_zero} non-zero tokens out of {len(text_tensor)}")
                
                # Show label(s)
                if 'label' in sample:
                    print(f"    Label: {sample['label'].item()}")
                else:
                    labels = {k: v.item() for k, v in sample.items() 
                             if k not in ['visual', 'text', 'attention_mask']}
                    print(f"    Labels: {labels}")
        else:
            print("  No dataset available")
        
    except Exception as e:
        print(f"✗ Error iterating samples: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 6: Checking Dataset Attributes Distribution
    # ------------------------------------------------------------------------
    print("USE CASE 6: Analyzing attribute distribution")
    print("-" * 70)
    
    try:
        if 'dataset' in locals() and len(dataset) > 0:
            # Analyze up to 100 samples or entire dataset if smaller
            num_samples = min(100, len(dataset))
            
            print(f"Analyzing {num_samples} samples...")
            
            if dataset.legacy_mode:
                # Legacy mode: analyze single label
                labels = []
                for i in range(num_samples):
                    sample = dataset[i]
                    labels.append(sample['label'].item())
                
                unique_labels = sorted(set(labels))
                print(f"  Unique labels: {unique_labels}")
                print(f"  Label distribution:")
                
                for label in unique_labels:
                    count = labels.count(label)
                    percentage = (count / len(labels)) * 100
                    print(f"    Label {label}: {count:3d} samples ({percentage:5.1f}%)")
            
            else:
                # Multi-attribute mode: analyze available attributes
                if dataset.available_attributes:
                    print(f"  Analyzing attributes: {dataset.available_attributes[:3]}...")
                    
                    for attr in dataset.available_attributes[:3]:  # Show first 3
                        values = []
                        for i in range(num_samples):
                            sample = dataset[i]
                            if attr in sample:
                                values.append(sample[attr].item())
                        
                        unique_values = sorted(set(values))
                        print(f"  {attr}:")
                        print(f"    Unique values: {unique_values}")
                        
                        # Show top 3 most common
                        value_counts = {v: values.count(v) for v in unique_values}
                        top_3 = sorted(value_counts.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)[:3]
                        
                        for value, count in top_3:
                            percentage = (count / len(values)) * 100
                            print(f"      Value {value}: {count} ({percentage:.1f}%)")
                else:
                    print("  No attributes available")
        else:
            print("  No dataset available")
        
    except Exception as e:
        print(f"✗ Error analyzing distribution: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 7: Error Handling Demo
    # ------------------------------------------------------------------------
    print("USE CASE 7: Testing error handling")
    print("-" * 70)
    
    # Test 1: Non-existent CSV
    print("Test 1: Non-existent CSV file")
    try:
        bad_dataset = CustomDataset("non_existent.csv")
        print(f"  ✗ Should have raised an error")
    except Exception as e:
        print(f"  ✓ Correctly raised error: {type(e).__name__}")
    
    # Test 2: Invalid index
    print("Test 2: Invalid index access")
    try:
        if 'dataset' in locals() and len(dataset) > 0:
            invalid_idx = len(dataset) + 100
            sample = dataset[invalid_idx]
            print(f"  ✗ Should have raised an error")
    except Exception as e:
        print(f"  ✓ Correctly raised error: {type(e).__name__}")
    
    print()
    
    # ------------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if 'dataset' in locals():
        print(f"Dataset size: {len(dataset)}")
        print(f"Available attributes: {dataset.available_attributes}")
        print(f"Legacy mode: {dataset.legacy_mode}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            print()
            print("Sample structure:")
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape} ({v.dtype})")
    else:
        print("No dataset was created")
    
    print()
    print("All examples completed!")
    print("=" * 70)