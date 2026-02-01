import os
import cv2
import torch
import pandas as pd
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

# Import our preprocessing functions
from preprocessing.image_preprocessing import resize_image, normalize_image
from preprocessing.text_preprocessing import clean_text, tokenize_text
from preprocessing.augmentation import augment_image
from utils.path import PROCESSED_IMAGE_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV

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

# Define paths for processed images and CSV files
# Using constants makes it easy to change paths in one place
PROCESSED_IMAGE_DIR = PROCESSED_IMAGE_DIR
TRAIN_CSV = TRAIN_CSV
VAL_CSV = VAL_CSV
TEST_CSV = TEST_CSV

# Define default values for image processing
# These can be adjusted based on your needs
DEFAULT_IMAGE_SIZE = (224, 224)  # Standard size for many neural networks
MAX_TEXT_LENGTH = 512  # Maximum number of tokens for text


def process_dataset(csv_path, augment=False):
    """
    Load CSV file, preprocess images and text, return dataset ready for PyTorch.
    
    This function is the main data pipeline that:
    1. Reads a CSV file containing image names, text, and labels
    2. Loads and preprocesses each image (resize, augment, normalize)
    3. Cleans and tokenizes text data
    4. Combines everything into PyTorch-ready tensors
    
    Why we need this:
    - Neural networks need data in specific formats (tensors)
    - Images need to be same size for batch processing
    - Text needs to be converted to numbers (tokens)
    - Data augmentation helps prevent overfitting (only for training)
    
    Args:
        csv_path: Path to CSV file containing dataset information
                 CSV should have columns: image_name, text, label_num
        augment: Whether to apply data augmentation to images
                True for training data, False for validation/test
    
    Returns:
        processed_data: List of dictionaries, each containing:
                       - image: normalized image tensor
                       - text_tokens: tokenized text as tensor
                       - label: label as tensor
    
    Example:
        >>> train_data = process_dataset('train.csv', augment=True)
        >>> print(f"Loaded {len(train_data)} training samples")
    """
    
    # ========================================================================
    # STEP 1: VALIDATE INPUT PARAMETERS
    # ========================================================================
    # Check if inputs are valid before processing
    # This prevents errors and provides clear feedback
    
    logger.info(f"Starting dataset processing for: {csv_path}")
    logger.info(f"Augmentation enabled: {augment}")
    
    # Initialize the list that will hold our processed data
    # We'll add items to this list as we process each row
    processed_data = []
    
    # Validate csv_path parameter
    try:
        if csv_path is None or csv_path == "":
            logger.error("CSV path is None or empty")
            return processed_data
        
        # Check if the CSV file actually exists
        if not os.path.exists(csv_path):
            logger.error(f"CSV file does not exist: {csv_path}")
            return processed_data
        
        # Check if it's actually a file (not a directory)
        if not os.path.isfile(csv_path):
            logger.error(f"Path is not a file: {csv_path}")
            return processed_data
        
        logger.debug(f"CSV file validation passed: {csv_path}")
        
    except Exception as e:
        logger.error(f"Error validating CSV path: {str(e)}")
        return processed_data
    
    
    # ========================================================================
    # STEP 2: LOAD CSV FILE
    # ========================================================================
    # Read the CSV file into a pandas DataFrame
    # DataFrame is like a table with rows and columns
    
    try:
        # Read CSV file into memory
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded CSV file with {len(df)} rows")
        
        # Validate that CSV has required columns
        # We need: image_name, text, label_num
        required_columns = ["image_name", "text", "label_num"]
        missing_columns = []
        
        for col in required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            logger.error(
                f"CSV is missing required columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )
            return processed_data
        
        # Check if DataFrame is empty
        if len(df) == 0:
            logger.warning("CSV file is empty (no data rows)")
            return processed_data
        
        logger.debug(f"CSV columns: {list(df.columns)}")
        
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {csv_path}")
        return processed_data
        
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {str(e)}")
        return processed_data
        
    except Exception as e:
        logger.error(f"Unexpected error loading CSV: {str(e)}")
        return processed_data
    
    
    # ========================================================================
    # STEP 3: VALIDATE IMAGE DIRECTORY
    # ========================================================================
    # Make sure the directory containing images exists
    
    try:
        if not os.path.exists(PROCESSED_IMAGE_DIR):
            logger.error(f"Image directory does not exist: {PROCESSED_IMAGE_DIR}")
            return processed_data
        
        if not os.path.isdir(PROCESSED_IMAGE_DIR):
            logger.error(f"Image path is not a directory: {PROCESSED_IMAGE_DIR}")
            return processed_data
        
        logger.debug(f"Image directory validated: {PROCESSED_IMAGE_DIR}")
        
    except Exception as e:
        logger.error(f"Error validating image directory: {str(e)}")
        return processed_data
    
    
    # ========================================================================
    # STEP 4: PROCESS EACH ROW IN THE CSV
    # ========================================================================
    # Loop through each row and process the image and text
    
    # Keep track of statistics for logging
    total_rows = len(df)
    successful_count = 0
    failed_count = 0
    failed_reasons = {
        "image_not_found": 0,
        "image_load_error": 0,
        "image_processing_error": 0,
        "text_processing_error": 0,
        "label_error": 0,
        "other_error": 0
    }
    
    logger.info(f"Processing {total_rows} rows from CSV...")
    
    # Use iterrows() to loop through each row
    # _ is the index (we don't need it), row is the data
    for row_idx, row in df.iterrows():
        
        # Log progress every 100 rows to track processing
        if (row_idx + 1) % 100 == 0:
            logger.info(f"Processing row {row_idx + 1}/{total_rows}...")
        
        try:
            # ================================================================
            # STEP 4.1: LOAD AND VALIDATE IMAGE
            # ================================================================
            
            # Get the image filename from the current row
            image_name = row["image_name"]
            
            # Build full path to the image file
            # os.path.join handles different OS path separators (/ or \)
            image_path = os.path.join(PROCESSED_IMAGE_DIR, image_name)
            
            logger.debug(f"Row {row_idx}: Loading image from {image_path}")
            
            # Check if image file exists before trying to load it
            if not os.path.exists(image_path):
                logger.warning(
                    f"Row {row_idx}: Image file not found: {image_path}. "
                    f"Skipping this row."
                )
                failed_count += 1
                failed_reasons["image_not_found"] += 1
                continue  # Skip to next row
            
            # Load the image using OpenCV
            # cv2.imread returns None if the file can't be read
            img = cv2.imread(image_path)
            
            # Check if image was loaded successfully
            if img is None:
                logger.warning(
                    f"Row {row_idx}: Failed to load image: {image_path}. "
                    f"File might be corrupted or not a valid image. "
                    f"Skipping this row."
                )
                failed_count += 1
                failed_reasons["image_load_error"] += 1
                continue  # Skip to next row
            
            # Validate image has data
            if img.size == 0:
                logger.warning(
                    f"Row {row_idx}: Image is empty: {image_path}. "
                    f"Skipping this row."
                )
                failed_count += 1
                failed_reasons["image_load_error"] += 1
                continue
            
            logger.debug(
                f"Row {row_idx}: Image loaded successfully. "
                f"Shape: {img.shape}"
            )
            
        except KeyError:
            logger.error(f"Row {row_idx}: 'image_name' column not found in row")
            failed_count += 1
            failed_reasons["other_error"] += 1
            continue
            
        except Exception as e:
            logger.error(
                f"Row {row_idx}: Unexpected error loading image: {str(e)}"
            )
            failed_count += 1
            failed_reasons["image_load_error"] += 1
            continue
        
        
        try:
            # ================================================================
            # STEP 4.2: PREPROCESS IMAGE
            # ================================================================
            # Resize, augment (if needed), and normalize the image
            
            # Resize image to standard size
            # This ensures all images have the same dimensions for batching
            img = resize_image(img)
            
            if img is None:
                logger.warning(
                    f"Row {row_idx}: resize_image returned None. "
                    f"Skipping this row."
                )
                failed_count += 1
                failed_reasons["image_processing_error"] += 1
                continue
            
            logger.debug(f"Row {row_idx}: Image resized")
            
            # Apply augmentation if requested
            # Only do this for training data, not validation/test
            if augment:
                img = augment_image(img)
                
                if img is None:
                    logger.warning(
                        f"Row {row_idx}: augment_image returned None. "
                        f"Skipping this row."
                    )
                    failed_count += 1
                    failed_reasons["image_processing_error"] += 1
                    continue
                
                logger.debug(f"Row {row_idx}: Image augmented")
            
            # Normalize image and convert to tensor
            # This scales pixel values and converts to PyTorch format
            img_tensor = normalize_image(img)
            
            if img_tensor is None:
                logger.warning(
                    f"Row {row_idx}: normalize_image returned None. "
                    f"Skipping this row."
                )
                failed_count += 1
                failed_reasons["image_processing_error"] += 1
                continue
            
            # Validate that we got a proper tensor
            if not isinstance(img_tensor, torch.Tensor):
                logger.warning(
                    f"Row {row_idx}: normalize_image did not return a tensor. "
                    f"Got type: {type(img_tensor)}. Skipping this row."
                )
                failed_count += 1
                failed_reasons["image_processing_error"] += 1
                continue
            
            logger.debug(
                f"Row {row_idx}: Image normalized to tensor. "
                f"Shape: {img_tensor.shape}"
            )
            
        except Exception as e:
            logger.error(
                f"Row {row_idx}: Error during image preprocessing: {str(e)}"
            )
            failed_count += 1
            failed_reasons["image_processing_error"] += 1
            continue
        
        
        try:
            # ================================================================
            # STEP 4.3: PREPROCESS TEXT
            # ================================================================
            # Clean and tokenize the text data
            
            # Get text from the current row
            text = row["text"]
            
            # Handle missing or NaN text
            if pd.isna(text):
                logger.warning(
                    f"Row {row_idx}: Text is NaN/missing. "
                    f"Using empty string."
                )
                text = ""
            
            # Convert to string if it's not already
            text = str(text)
            
            logger.debug(
                f"Row {row_idx}: Raw text: '{text[:50]}...'" 
                if len(text) > 50 else 
                f"Row {row_idx}: Raw text: '{text}'"
            )
            
            # Clean the text (remove special characters, lowercase, etc.)
            text = clean_text(text)
            
            if text is None:
                logger.warning(
                    f"Row {row_idx}: clean_text returned None. "
                    f"Using empty string."
                )
                text = ""
            
            logger.debug(f"Row {row_idx}: Text cleaned")
            
            # Tokenize text (convert words to numbers)
            token_ids = tokenize_text(text)
            
            if token_ids is None:
                logger.warning(
                    f"Row {row_idx}: tokenize_text returned None. "
                    f"Skipping this row."
                )
                failed_count += 1
                failed_reasons["text_processing_error"] += 1
                continue
            
            # Validate that we got a proper tensor or list
            if not isinstance(token_ids, (torch.Tensor, list)):
                logger.warning(
                    f"Row {row_idx}: tokenize_text returned unexpected type: "
                    f"{type(token_ids)}. Skipping this row."
                )
                failed_count += 1
                failed_reasons["text_processing_error"] += 1
                continue
            
            logger.debug(
                f"Row {row_idx}: Text tokenized. "
                f"Tokens: {len(token_ids) if hasattr(token_ids, '__len__') else 'unknown'}"
            )
            
        except KeyError:
            logger.error(f"Row {row_idx}: 'text' column not found in row")
            failed_count += 1
            failed_reasons["text_processing_error"] += 1
            continue
            
        except Exception as e:
            logger.error(
                f"Row {row_idx}: Error during text preprocessing: {str(e)}"
            )
            failed_count += 1
            failed_reasons["text_processing_error"] += 1
            continue
        
        
        try:
            # ================================================================
            # STEP 4.4: PROCESS LABEL
            # ================================================================
            # Convert label to PyTorch tensor
            
            # Get label from the current row
            label_value = row["label_num"]
            
            # Handle missing or NaN labels
            if pd.isna(label_value):
                logger.warning(
                    f"Row {row_idx}: Label is NaN/missing. "
                    f"Skipping this row."
                )
                failed_count += 1
                failed_reasons["label_error"] += 1
                continue
            
            # Convert to integer (in case it's a float like 1.0)
            try:
                label_value = int(label_value)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Row {row_idx}: Cannot convert label to integer: {label_value}. "
                    f"Error: {str(e)}. Skipping this row."
                )
                failed_count += 1
                failed_reasons["label_error"] += 1
                continue
            
            # Validate label is non-negative
            if label_value < 0:
                logger.warning(
                    f"Row {row_idx}: Label is negative: {label_value}. "
                    f"Skipping this row."
                )
                failed_count += 1
                failed_reasons["label_error"] += 1
                continue
            
            # Convert to PyTorch tensor
            # dtype=torch.long is used for classification labels
            label_tensor = torch.tensor(label_value, dtype=torch.long)
            
            logger.debug(f"Row {row_idx}: Label converted to tensor: {label_value}")
            
        except KeyError:
            logger.error(f"Row {row_idx}: 'label_num' column not found in row")
            failed_count += 1
            failed_reasons["label_error"] += 1
            continue
            
        except Exception as e:
            logger.error(
                f"Row {row_idx}: Error processing label: {str(e)}"
            )
            failed_count += 1
            failed_reasons["label_error"] += 1
            continue
        
        
        try:
            # ================================================================
            # STEP 4.5: CREATE DATA DICTIONARY AND ADD TO LIST
            # ================================================================
            # Combine all processed data into a dictionary
            
            # Create dictionary with processed data
            # This format is convenient for PyTorch DataLoader
            data_dict = {
                "image": img_tensor,
                "text_tokens": token_ids,
                "label": label_tensor
            }
            
            # Add to our list of processed data
            processed_data.append(data_dict)
            
            successful_count += 1
            
            logger.debug(
                f"Row {row_idx}: Successfully processed and added to dataset"
            )
            
        except Exception as e:
            logger.error(
                f"Row {row_idx}: Error creating data dictionary: {str(e)}"
            )
            failed_count += 1
            failed_reasons["other_error"] += 1
            continue
    
    
    # ========================================================================
    # STEP 5: LOG FINAL STATISTICS
    # ========================================================================
    # Provide summary of processing results
    
    logger.info("=" * 70)
    logger.info("DATASET PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"CSV File: {csv_path}")
    logger.info(f"Total rows in CSV: {total_rows}")
    logger.info(f"Successfully processed: {successful_count}")
    logger.info(f"Failed to process: {failed_count}")
    
    if failed_count > 0:
        logger.info("Failure breakdown:")
        for reason, count in failed_reasons.items():
            if count > 0:
                logger.info(f"  - {reason}: {count}")
    
    success_rate = (successful_count / total_rows * 100) if total_rows > 0 else 0
    logger.info(f"Success rate: {success_rate:.2f}%")
    logger.info("=" * 70)
    
    # Warn if success rate is too low
    if success_rate < 50:
        logger.warning(
            "Success rate is below 50%. Please check your data and error logs."
        )
    
    # Return the list of processed data
    # Variable name is kept as 'processed_data' as requested
    return processed_data


# ============================================================================
# USAGE EXAMPLES AND TEST CASES
# ============================================================================

if __name__ == "__main__":
    """
    This section demonstrates how to use the process_dataset function.
    It only runs when you execute this file directly (not when imported).
    """
    
    print("=" * 70)
    print("DATASET PREPROCESSING - USAGE EXAMPLES")
    print("=" * 70)
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 1: Processing Training Data with Augmentation
    # ------------------------------------------------------------------------
    print("USE CASE 1: Processing training dataset with augmentation")
    print("-" * 70)
    
    try:
        # Process training data with augmentation enabled
        # Augmentation helps prevent overfitting by creating variations
        train_data = process_dataset(TRAIN_CSV, augment=True)
        
        print(f"✓ Training data processed")
        print(f"  Total samples: {len(train_data)}")
        
        # Show sample data structure
        if len(train_data) > 0:
            sample = train_data[0]
            print(f"  Sample data structure:")
            print(f"    - image shape: {sample['image'].shape}")
            print(f"    - text_tokens length: {len(sample['text_tokens'])}")
            print(f"    - label: {sample['label'].item()}")
        
    except Exception as e:
        print(f"✗ Error processing training data: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 2: Processing Validation Data without Augmentation
    # ------------------------------------------------------------------------
    print("USE CASE 2: Processing validation dataset without augmentation")
    print("-" * 70)
    
    try:
        # Process validation data WITHOUT augmentation
        # We don't augment validation data to get true performance metrics
        val_data = process_dataset(VAL_CSV, augment=False)
        
        print(f"✓ Validation data processed")
        print(f"  Total samples: {len(val_data)}")
        
        # Check for label distribution
        if len(val_data) > 0:
            labels = [item['label'].item() for item in val_data]
            unique_labels = set(labels)
            print(f"  Unique labels: {sorted(unique_labels)}")
            print(f"  Label distribution:")
            for label in sorted(unique_labels):
                count = labels.count(label)
                percentage = (count / len(labels)) * 100
                print(f"    Label {label}: {count} samples ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"✗ Error processing validation data: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 3: Processing Test Data
    # ------------------------------------------------------------------------
    print("USE CASE 3: Processing test dataset")
    print("-" * 70)
    
    try:
        # Process test data WITHOUT augmentation
        # Test data should never be augmented
        test_data = process_dataset(TEST_CSV, augment=False)
        
        print(f"✓ Test data processed")
        print(f"  Total samples: {len(test_data)}")
        
    except Exception as e:
        print(f"✗ Error processing test data: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 4: Error Handling - Non-existent CSV File
    # ------------------------------------------------------------------------
    print("USE CASE 4: Testing error handling with non-existent file")
    print("-" * 70)
    
    try:
        fake_csv = "non_existent_file.csv"
        print(f"Attempting to load: {fake_csv}")
        
        result = process_dataset(fake_csv, augment=False)
        
        print(f"  Result: {len(result)} samples")
        print(f"  (Should be 0 because file doesn't exist)")
        
    except Exception as e:
        print(f"  Caught exception: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 5: Comparing Augmented vs Non-Augmented Data
    # ------------------------------------------------------------------------
    print("USE CASE 5: Comparing augmented vs non-augmented processing")
    print("-" * 70)
    
    try:
        # Create a small test CSV for demonstration
        import tempfile
        
        # Create temporary CSV file
        test_csv_content = """image_name,text,label_num
sample1.jpg,This is test text one,0
sample2.jpg,This is test text two,1
sample3.jpg,This is test text three,0
"""
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(test_csv_content)
            temp_csv_path = f.name
        
        print(f"Created temporary CSV: {temp_csv_path}")
        
        # Process without augmentation
        data_no_aug = process_dataset(temp_csv_path, augment=False)
        print(f"  Without augmentation: {len(data_no_aug)} samples")
        
        # Process with augmentation
        data_with_aug = process_dataset(temp_csv_path, augment=True)
        print(f"  With augmentation: {len(data_with_aug)} samples")
        
        # Clean up
        os.unlink(temp_csv_path)
        print(f"  Cleaned up temporary file")
        
    except Exception as e:
        print(f"  Error in comparison test: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 6: Batch Processing and Memory Check
    # ------------------------------------------------------------------------
    print("USE CASE 6: Processing and checking memory usage")
    print("-" * 70)
    
    try:
        import sys
        
        # Process a dataset
        data = process_dataset(TRAIN_CSV, augment=False)
        
        if len(data) > 0:
            # Calculate approximate memory usage
            # This is a rough estimate
            sample_size = sys.getsizeof(data[0])
            total_size = sample_size * len(data)
            
            print(f"  Approximate memory usage:")
            print(f"    Per sample: {sample_size / 1024:.2f} KB")
            print(f"    Total dataset: {total_size / (1024 * 1024):.2f} MB")
            
            # Show tensor sizes
            if len(data) > 0:
                sample = data[0]
                img_size = sample['image'].element_size() * sample['image'].nelement()
                print(f"    Image tensor per sample: {img_size / 1024:.2f} KB")
        else:
            print("  No data to analyze")
        
    except Exception as e:
        print(f"  Error in memory check: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Processed train samples: {len(train_data) if 'train_data' in locals() else 'N/A'}")
    print(f"Processed val samples: {len(val_data) if 'val_data' in locals() else 'N/A'}")
    print(f"Processed test samples: {len(test_data) if 'test_data' in locals() else 'N/A'}")
    print()
    print("All examples completed!")
    print("=" * 70)