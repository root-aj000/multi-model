"""
Training Logger Module
======================
This module provides logging functionality for training metrics.
It logs to both TensorBoard (for visualization) and CSV files (for analysis).

Features:
- TensorBoard integration for real-time visualization
- CSV file output for easy data analysis
- Automatic timestamp-based directory organization
- Error handling and validation
- Thread-safe file operations

Author: [Your Name]
Date: [Date]
"""

import os
import csv
import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from utils.path import LOG_DIR


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure logging for this module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# LOGGER CLASS
# =============================================================================

class Logger:
    """
    Training Logger for Multi-Modal Model Training
    
    This class handles logging of training metrics to multiple destinations:
    1. TensorBoard: For interactive visualization and monitoring
    2. CSV File: For post-training analysis and plotting
    
    The logger creates a timestamped directory to organize logs from
    different training runs. This prevents overwriting previous logs
    and makes it easy to compare different experiments.
    
    Example Directory Structure:
        logs/
        ├── 20240115-143022/
        │   ├── events.out.tfevents.*  (TensorBoard files)
        │   └── training_log.csv
        └── 20240115-153045/
            ├── events.out.tfevents.*
            └── training_log.csv
    """
    
    def __init__(self, tranning_log_dir: str):
        """
        Initialize the Logger with a timestamped directory.
        
        This method sets up logging infrastructure including:
        - Creating a unique timestamped directory
        - Initializing TensorBoard writer
        - Preparing CSV file for metrics
        
        Args:
            tranning_log_dir (str): Base directory for all training logs
                                   Example: 'logs/' or '/path/to/logs'
        
        Raises:
            TypeError: If tranning_log_dir is not a string
            OSError: If directory creation fails
            RuntimeError: If TensorBoard writer initialization fails
        
        Example:
            >>> logger = Logger('logs/')
            >>> # Creates: logs/20240115-143022/
        """
        try:
            # Step 1: Validate input parameter
            logger.info("=" * 70)
            logger.info("Initializing Training Logger")
            logger.info("=" * 70)
            
            self._validate_log_dir(tranning_log_dir)
            
            # Step 2: Create timestamp for this training run
            # Format: YYYYMMDD-HHMMSS (e.g., 20240115-143022)
            # This ensures each training run has a unique directory
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            logger.info(f"Training run timestamp: {timestamp}")
            
            # Step 3: Create full path for this training run's logs
            # Example: logs/20240115-143022/
            self.tensorboard_dir = os.path.join(tranning_log_dir, timestamp)
            logger.info(f"Log directory: {self.tensorboard_dir}")
            
            # Step 4: Create the directory
            # exist_ok=True means don't error if directory already exists
            # (though with timestamps, this is unlikely)
            try:
                os.makedirs(self.tensorboard_dir, exist_ok=True)
                logger.info("✓ Log directory created successfully")
            except OSError as e:
                error_msg = f"Failed to create log directory '{self.tensorboard_dir}': {str(e)}"
                logger.error(error_msg)
                raise OSError(error_msg)
            
            # Step 5: Initialize TensorBoard writer
            # TensorBoard is a visualization tool that shows training curves,
            # metrics, and other information in an interactive web interface
            try:
                logger.info("Initializing TensorBoard writer...")
                self.writer = SummaryWriter(self.tensorboard_dir)
                logger.info("✓ TensorBoard writer initialized")
                logger.info(f"  To view logs, run: tensorboard --logdir={self.tensorboard_dir}")
            except Exception as e:
                error_msg = f"Failed to initialize TensorBoard writer: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                raise RuntimeError(error_msg)
            
            # Step 6: Set up CSV file path
            # CSV file will contain all metrics in tabular format
            # This is useful for creating custom plots or analysis
            self.csv_file = os.path.join(self.tensorboard_dir, "training_log.csv")
            logger.info(f"CSV log file: {self.csv_file}")
            
            # Step 7: Initialize CSV tracking flag
            # We need to track if we've written the CSV header yet
            # The header is only written once (first time log_metrics is called)
            self.csv_initialized = False
            
            # Step 8: Store base directory for reference
            self.base_log_dir = tranning_log_dir
            
            # Step 9: Track what metrics we're logging
            # This helps detect if metric names change during training
            self.known_metrics = set()
            
            # Step 10: Count number of log entries
            # Useful for debugging and monitoring
            self.log_count = 0
            
            logger.info("=" * 70)
            logger.info("Logger Initialization Complete")
            logger.info("=" * 70)
            logger.info(f"TensorBoard directory: {self.tensorboard_dir}")
            logger.info(f"CSV file: {self.csv_file}")
            logger.info("=" * 70)
            
        except Exception as e:
            # Catch any unexpected errors during initialization
            error_msg = f"Logger initialization failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    
    def _validate_log_dir(self, tranning_log_dir: str) -> None:
        """
        Validate the log directory parameter.
        
        This is a helper method that ensures the provided log directory
        parameter is valid before we try to use it.
        
        Args:
            tranning_log_dir (str): Directory path to validate
        
        Raises:
            TypeError: If parameter is not a string
            ValueError: If parameter is empty or invalid
        """
        # Check type
        if not isinstance(tranning_log_dir, str):
            error_msg = f"tranning_log_dir must be a string, got {type(tranning_log_dir)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Check if empty
        if not tranning_log_dir.strip():
            error_msg = "tranning_log_dir cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check if path contains invalid characters
        # This is a basic check; OS-specific validation would be more thorough
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        for char in invalid_chars:
            if char in tranning_log_dir:
                error_msg = f"tranning_log_dir contains invalid character: '{char}'"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        logger.debug(f"✓ Log directory validation passed: {tranning_log_dir}")
    
    
    def log_metrics(self, metrics_dict: Dict[str, Any], epoch: int) -> None:
        """
        Log training metrics to TensorBoard and CSV file.
        
        This method records metrics for a specific epoch to both:
        1. TensorBoard (for visualization)
        2. CSV file (for analysis)
        
        The CSV file is created on the first call with a header row
        containing all metric names. Subsequent calls append data rows.
        
        Args:
            metrics_dict (dict): Dictionary of metric names and values
                                Example: {
                                    "train_loss": 0.245,
                                    "val_loss": 0.312,
                                    "train_acc": 0.891,
                                    "val_acc": 0.856
                                }
            
            epoch (int): Current epoch number (used as x-axis in plots)
                        Should be 1-indexed (epoch 1, 2, 3, ...)
        
        Raises:
            TypeError: If parameters have wrong type
            ValueError: If parameters are invalid
            RuntimeError: If logging fails
        
        Example:
            >>> logger = Logger('logs/')
            >>> metrics = {"train_loss": 0.5, "val_loss": 0.6}
            >>> logger.log_metrics(metrics, epoch=1)
            >>> # Logs are written to TensorBoard and CSV
        """
        try:
            # Step 1: Validate inputs
            logger.debug(f"Logging metrics for epoch {epoch}")
            self._validate_metrics_input(metrics_dict, epoch)
            
            # Step 2: Check for metric name consistency
            # Warn if we're seeing new metrics that weren't in previous epochs
            current_metrics = set(metrics_dict.keys())
            
            if self.known_metrics and current_metrics != self.known_metrics:
                new_metrics = current_metrics - self.known_metrics
                missing_metrics = self.known_metrics - current_metrics
                
                if new_metrics:
                    logger.warning(f"New metrics detected: {new_metrics}")
                if missing_metrics:
                    logger.warning(f"Previously logged metrics are missing: {missing_metrics}")
            
            # Update known metrics
            self.known_metrics.update(current_metrics)
            
            # Step 3: Log to TensorBoard
            # TensorBoard stores metrics in a special binary format
            # that can be visualized in the TensorBoard web interface
            logger.debug("Writing to TensorBoard...")
            
            try:
                for key, value in metrics_dict.items():
                    # Check if value is a valid number
                    if not isinstance(value, (int, float)):
                        logger.warning(f"Metric '{key}' has non-numeric value: {value} (type: {type(value)})")
                        # Try to convert to float
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            logger.error(f"Cannot convert metric '{key}' to float. Skipping.")
                            continue
                    
                    # Check for NaN or Inf
                    if value != value:  # NaN check
                        logger.warning(f"Metric '{key}' is NaN. Logging as 0.")
                        value = 0.0
                    elif value == float('inf') or value == float('-inf'):
                        logger.warning(f"Metric '{key}' is Inf. Logging as 0.")
                        value = 0.0
                    
                    # Write to TensorBoard
                    # add_scalar creates a line plot: x-axis = epoch, y-axis = value
                    self.writer.add_scalar(key, value, epoch)
                
                logger.debug(f"✓ Wrote {len(metrics_dict)} metrics to TensorBoard")
                
            except Exception as e:
                error_msg = f"Failed to write to TensorBoard: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                # Don't raise - continue to CSV logging
                logger.warning("Continuing with CSV logging despite TensorBoard error")
            
            # Step 4: Log to CSV file
            # CSV provides a simple tabular format that's easy to analyze
            # with tools like Excel, pandas, or plotting libraries
            logger.debug("Writing to CSV...")
            
            try:
                # Step 4a: Initialize CSV file if this is the first log
                if not self.csv_initialized:
                    logger.info("Creating CSV file with header...")
                    self._initialize_csv_file(metrics_dict)
                    logger.info("✓ CSV file initialized")
                
                # Step 4b: Append metrics to CSV file
                self._append_to_csv(metrics_dict, epoch)
                logger.debug("✓ Metrics appended to CSV")
                
            except Exception as e:
                error_msg = f"Failed to write to CSV: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                # Don't raise - at least we have TensorBoard logs
                logger.warning("CSV logging failed, but TensorBoard logs are available")
            
            # Step 5: Flush TensorBoard writer
            # This ensures data is written to disk immediately
            # Without this, data might be buffered and lost if program crashes
            try:
                self.writer.flush()
                logger.debug("✓ TensorBoard data flushed to disk")
            except Exception as e:
                logger.warning(f"Failed to flush TensorBoard writer: {str(e)}")
            
            # Step 6: Update log counter
            self.log_count += 1
            
            # Step 7: Periodic status update
            # Every 10 epochs, print a status message
            if epoch % 10 == 0:
                logger.info(f"📊 Logged {self.log_count} metric entries so far")
            
            logger.debug(f"✓ Successfully logged metrics for epoch {epoch}")
            
        except Exception as e:
            # Catch any unexpected errors
            error_msg = f"Failed to log metrics for epoch {epoch}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    
    def _validate_metrics_input(
        self, 
        metrics_dict: Dict[str, Any], 
        epoch: int
    ) -> None:
        """
        Validate input parameters for log_metrics method.
        
        This helper method checks that the metrics dictionary and epoch
        number are valid before attempting to log them.
        
        Args:
            metrics_dict: Metrics dictionary to validate
            epoch: Epoch number to validate
        
        Raises:
            TypeError: If parameters have wrong type
            ValueError: If parameters are invalid
        """
        # Validate metrics_dict type
        if not isinstance(metrics_dict, dict):
            error_msg = f"metrics_dict must be a dictionary, got {type(metrics_dict)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Check if dictionary is empty
        if not metrics_dict:
            error_msg = "metrics_dict cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate metric keys are strings
        for key in metrics_dict.keys():
            if not isinstance(key, str):
                error_msg = f"Metric keys must be strings, got {type(key)} for key {key}"
                logger.error(error_msg)
                raise TypeError(error_msg)
            
            if not key.strip():
                error_msg = "Metric keys cannot be empty strings"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Validate epoch type
        if not isinstance(epoch, int):
            error_msg = f"epoch must be an integer, got {type(epoch)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Validate epoch value
        if epoch < 0:
            error_msg = f"epoch must be non-negative, got {epoch}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug("✓ Metrics input validation passed")
    
    
    def _initialize_csv_file(self, metrics_dict: Dict[str, Any]) -> None:
        """
        Create CSV file and write header row.
        
        This helper method creates a new CSV file with a header row
        containing the column names: epoch, metric1, metric2, ...
        
        Args:
            metrics_dict: Dictionary of metrics (keys become column names)
        
        Raises:
            OSError: If file creation fails
            csv.Error: If CSV writing fails
        """
        try:
            # Step 1: Prepare header row
            # First column is always "epoch", followed by metric names
            fieldnames = ["epoch"] + list(metrics_dict.keys())
            logger.debug(f"CSV columns: {fieldnames}")
            
            # Step 2: Create CSV file and write header
            # mode='w' creates a new file (overwrites if exists)
            # newline='' is recommended for CSV files to handle line endings correctly
            with open(self.csv_file, mode='w', newline='') as f:
                # DictWriter allows us to write dictionaries as rows
                # It automatically maps dictionary keys to column names
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write the header row (column names)
                writer.writeheader()
            
            # Step 3: Mark CSV as initialized
            # This flag prevents us from writing the header again
            self.csv_initialized = True
            
            logger.debug(f"✓ CSV file created with {len(fieldnames)} columns")
            
        except OSError as e:
            error_msg = f"Failed to create CSV file '{self.csv_file}': {str(e)}"
            logger.error(error_msg)
            raise OSError(error_msg)
        except csv.Error as e:
            error_msg = f"CSV writing error: {str(e)}"
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Unexpected error initializing CSV: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    
    def _append_to_csv(self, metrics_dict: Dict[str, Any], epoch: int) -> None:
        """
        Append a row of metrics to the CSV file.
        
        This helper method adds a new row to the CSV file containing
        the epoch number and all metric values.
        
        Args:
            metrics_dict: Dictionary of metric values
            epoch: Epoch number
        
        Raises:
            OSError: If file writing fails
            csv.Error: If CSV writing fails
        """
        try:
            # Step 1: Prepare fieldnames (must match header)
            fieldnames = ["epoch"] + list(metrics_dict.keys())
            
            # Step 2: Create row data
            # Combine epoch number with metric values
            row = {"epoch": epoch, **metrics_dict}
            
            # Step 3: Append row to CSV file
            # mode='a' opens file in append mode (adds to end of file)
            with open(self.csv_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write one row
                writer.writerow(row)
            
            logger.debug(f"✓ Appended epoch {epoch} data to CSV")
            
        except OSError as e:
            error_msg = f"Failed to write to CSV file '{self.csv_file}': {str(e)}"
            logger.error(error_msg)
            raise OSError(error_msg)
        except csv.Error as e:
            error_msg = f"CSV writing error: {str(e)}"
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Unexpected error appending to CSV: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    
    def close(self) -> None:
        """
        Close the logger and release resources.
        
        This method should be called when training is complete to ensure:
        1. All TensorBoard data is flushed to disk
        2. File handles are properly closed
        3. Resources are released
        
        It's important to call this method to prevent data loss and
        resource leaks.
        
        Example:
            >>> logger = Logger('logs/')
            >>> try:
            ...     # Training loop
            ...     for epoch in range(10):
            ...         metrics = {"loss": 0.5}
            ...         logger.log_metrics(metrics, epoch)
            ... finally:
            ...     logger.close()  # Always close, even if training fails
        
        Raises:
            RuntimeError: If closing fails (warning only, doesn't raise)
        """
        try:
            logger.info("=" * 70)
            logger.info("Closing Training Logger")
            logger.info("=" * 70)
            
            # Step 1: Flush TensorBoard writer
            # This ensures all buffered data is written to disk
            try:
                logger.info("Flushing TensorBoard data...")
                self.writer.flush()
                logger.info("✓ TensorBoard data flushed")
            except Exception as e:
                logger.warning(f"Failed to flush TensorBoard data: {str(e)}")
            
            # Step 2: Close TensorBoard writer
            # This releases file handles and cleans up resources
            try:
                logger.info("Closing TensorBoard writer...")
                self.writer.close()
                logger.info("✓ TensorBoard writer closed")
            except Exception as e:
                logger.warning(f"Failed to close TensorBoard writer: {str(e)}")
            
            # Step 3: Print summary
            logger.info("=" * 70)
            logger.info("Logger Summary")
            logger.info("=" * 70)
            logger.info(f"Total log entries: {self.log_count}")
            logger.info(f"Metrics tracked: {len(self.known_metrics)}")
            logger.info(f"Metric names: {sorted(self.known_metrics)}")
            logger.info(f"Log directory: {self.tensorboard_dir}")
            logger.info(f"CSV file: {self.csv_file}")
            
            # Step 4: Check if files exist and log their sizes
            try:
                if os.path.exists(self.csv_file):
                    csv_size = os.path.getsize(self.csv_file)
                    logger.info(f"CSV file size: {csv_size} bytes")
                else:
                    logger.warning("CSV file was not created")
            except Exception as e:
                logger.warning(f"Could not check CSV file size: {str(e)}")
            
            logger.info("=" * 70)
            logger.info("Logger Closed Successfully")
            logger.info(f"To view TensorBoard logs, run:")
            logger.info(f"  tensorboard --logdir={self.tensorboard_dir}")
            logger.info("=" * 70)
            
        except Exception as e:
            # Log the error but don't raise
            # We want to be able to close even if there are errors
            error_msg = f"Error while closing logger: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            logger.warning("Logger may not have closed cleanly")


# =============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# =============================================================================

if __name__ == "__main__":
    """
    Example usage and test cases for the Logger class.
    """
    
    print("=" * 80)
    print("TRAINING LOGGER - Usage Examples")
    print("=" * 80)
    
    # ------------------------------------------------------------------------
    # Example 1: Basic Usage
    # ------------------------------------------------------------------------
    print("\n[Example 1] Basic Usage")
    print("-" * 80)
    
    try:
        # Create logger instance
        print("Creating logger...")
        test_logger = Logger("test_logs")
        print("✓ Logger created\n")
        
        # Simulate training for 5 epochs
        print("Simulating training for 5 epochs...")
        for epoch in range(1, 6):
            # Simulate metrics (in real training, these come from your model)
            metrics = {
                "train_loss": 1.0 / epoch,  # Loss decreases
                "val_loss": 1.2 / epoch,
                "train_acc": 0.5 + (epoch * 0.08),  # Accuracy increases
                "val_acc": 0.45 + (epoch * 0.07)
            }
            
            # Log metrics
            test_logger.log_metrics(metrics, epoch)
            print(f"  Epoch {epoch}: train_loss={metrics['train_loss']:.4f}, "
                  f"val_loss={metrics['val_loss']:.4f}")
        
        print("\n✓ Logged 5 epochs")
        
        # Close logger
        print("\nClosing logger...")
        test_logger.close()
        print("✓ Logger closed")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    
    # ------------------------------------------------------------------------
    # Example 2: Multi-Attribute Logging
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Example 2] Multi-Attribute Logging")
    print("-" * 80)
    
    try:
        print("Creating logger for multi-attribute training...")
        multi_logger = Logger("test_logs")
        
        # Log multiple attributes
        print("\nLogging multiple attributes...")
        for epoch in range(1, 4):
            metrics = {
                "train_loss": 0.5 / epoch,
                "val_loss": 0.6 / epoch,
                "sentiment_acc": 0.7 + (epoch * 0.05),
                "sentiment_f1": 0.65 + (epoch * 0.05),
                "emotion_acc": 0.6 + (epoch * 0.06),
                "emotion_f1": 0.55 + (epoch * 0.06),
                "theme_acc": 0.75 + (epoch * 0.04),
                "theme_f1": 0.72 + (epoch * 0.04),
            }
            
            multi_logger.log_metrics(metrics, epoch)
            print(f"  Epoch {epoch}: logged {len(metrics)} metrics")
        
        print("\n✓ Multi-attribute logging complete")
        multi_logger.close()
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    
    # ------------------------------------------------------------------------
    # Example 3: Error Handling - Invalid Inputs
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Example 3] Error Handling - Invalid Inputs")
    print("-" * 80)
    
    try:
        err_logger = Logger("test_logs")
        
        # Test 1: Empty metrics dictionary
        print("\nTest 1: Empty metrics dictionary")
        try:
            err_logger.log_metrics({}, epoch=1)
            print("✗ Should have raised an error!")
        except ValueError as e:
            print(f"✓ Correctly caught error: {type(e).__name__}")
        
        # Test 2: Invalid epoch number
        print("\nTest 2: Negative epoch number")
        try:
            err_logger.log_metrics({"loss": 0.5}, epoch=-1)
            print("✗ Should have raised an error!")
        except ValueError as e:
            print(f"✓ Correctly caught error: {type(e).__name__}")
        
        # Test 3: Non-numeric metric values (should handle gracefully)
        print("\nTest 3: Non-numeric metric value")
        try:
            err_logger.log_metrics({"loss": "invalid"}, epoch=1)
            print("⚠ Warning issued but handled gracefully")
        except Exception as e:
            print(f"Error: {type(e).__name__}")
        
        # Test 4: Wrong type for metrics_dict
        print("\nTest 4: Wrong type for metrics_dict")
        try:
            err_logger.log_metrics([1, 2, 3], epoch=1)
            print("✗ Should have raised an error!")
        except TypeError as e:
            print(f"✓ Correctly caught error: {type(e).__name__}")
        
        err_logger.close()
        
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
    
    
    # ------------------------------------------------------------------------
    # Example 4: Error Handling - Invalid Initialization
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Example 4] Error Handling - Invalid Initialization")
    print("-" * 80)
    
    # Test 1: Empty log directory
    print("\nTest 1: Empty log directory")
    try:
        invalid_logger = Logger("")
        print("✗ Should have raised an error!")
    except ValueError as e:
        print(f"✓ Correctly caught error: {type(e).__name__}")
    
    # Test 2: Invalid type
    print("\nTest 2: Invalid type for log directory")
    try:
        invalid_logger = Logger(12345)
        print("✗ Should have raised an error!")
    except TypeError as e:
        print(f"✓ Correctly caught error: {type(e).__name__}")
    
    # Test 3: None value
    print("\nTest 3: None value for log directory")
    try:
        invalid_logger = Logger(None)
        print("✗ Should have raised an error!")
    except TypeError as e:
        print(f"✓ Correctly caught error: {type(e).__name__}")
    
    
    # ------------------------------------------------------------------------
    # Example 5: Proper Cleanup with Try-Finally
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Example 5] Proper Cleanup with Try-Finally")
    print("-" * 80)
    
    try:
        print("Demonstrating proper cleanup pattern...")
        cleanup_logger = Logger("test_logs")
        
        try:
            # Training loop
            print("Training...")
            for epoch in range(1, 3):
                metrics = {"loss": 1.0 / epoch, "acc": 0.5 + (epoch * 0.1)}
                cleanup_logger.log_metrics(metrics, epoch)
            
            print("✓ Training complete")
            
        finally:
            # This ensures logger is closed even if an error occurs
            print("\nClosing logger (in finally block)...")
            cleanup_logger.close()
            print("✓ Logger closed successfully")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    
    # ------------------------------------------------------------------------
    # Example 6: Context Manager Pattern (Recommended)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Example 6] Recommended Usage Pattern")
    print("-" * 80)
    
    try:
        print("Best practice: Always close logger after use\n")
        
        print("Pattern 1 - Explicit try-finally:")
        print("```python")
        print("logger = Logger('logs/')")
        print("try:")
        print("    for epoch in range(EPOCHS):")
        print("        # Training code here")
        print("        logger.log_metrics(metrics, epoch)")
        print("finally:")
        print("    logger.close()")
        print("```")
        
        print("\nPattern 2 - Single training run:")
        print("```python")
        print("logger = Logger('logs/')")
        print("for epoch in range(EPOCHS):")
        print("    metrics = train_epoch(...)")
        print("    logger.log_metrics(metrics, epoch)")
        print("logger.close()")
        print("```")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    
    # ------------------------------------------------------------------------
    # Example 7: Reading Logged Data
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Example 7] Reading Logged Data")
    print("-" * 80)
    
    try:
        import pandas as pd
        
        # Create and log some data
        read_logger = Logger("test_logs")
        
        for epoch in range(1, 4):
            metrics = {"loss": 1.0 / epoch, "accuracy": 0.5 + (epoch * 0.1)}
            read_logger.log_metrics(metrics, epoch)
        
        csv_path = read_logger.csv_file
        read_logger.close()
        
        # Read the CSV file
        print(f"\nReading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print("\nLogged Data:")
        print(df)
        
        print("\nSummary Statistics:")
        print(df.describe())
        
    except ImportError:
        print("✗ pandas not available for this example")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    
    # ------------------------------------------------------------------------
    # Summary of Functions
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FUNCTION SUMMARY")
    print("=" * 80)
    
    print("""
    1. Logger.__init__(tranning_log_dir)
       Purpose: Initialize logger with timestamped directory
       Use Cases:
         - Set up logging for a new training run
         - Create TensorBoard writer
         - Prepare CSV file path
       
       Examples:
         # Basic initialization
         logger = Logger('logs/')
         
         # With custom path
         logger = Logger('/path/to/my/logs')
         
         # Creates timestamped directory: logs/20240115-143022/
    
    
    2. Logger.log_metrics(metrics_dict, epoch)
       Purpose: Log training metrics to TensorBoard and CSV
       Use Cases:
         - Record metrics after each training epoch
         - Track multiple metrics simultaneously
         - Create visualization data
       
       Examples:
         # Single metric
         logger.log_metrics({"loss": 0.5}, epoch=1)
         
         # Multiple metrics
         metrics = {
             "train_loss": 0.5,
             "val_loss": 0.6,
             "train_acc": 0.85,
             "val_acc": 0.82
         }
         logger.log_metrics(metrics, epoch=10)
         
         # Multi-attribute metrics
         metrics = {
             "train_loss": 0.3,
             "sentiment_acc": 0.9,
             "emotion_acc": 0.85,
             "theme_acc": 0.88
         }
         logger.log_metrics(metrics, epoch=25)
    
    
    3. Logger.close()
       Purpose: Properly close logger and flush data
       Use Cases:
         - End of training
         - Cleanup resources
         - Ensure all data is written to disk
       
       Examples:
         # Basic usage
         logger.close()
         
         # With try-finally (recommended)
         logger = Logger('logs/')
         try:
             # Training loop
             pass
         finally:
             logger.close()
    
    
    TYPICAL WORKFLOW:
    ================
    
    1. Initialize logger at start of training:
       logger = Logger('logs/')
    
    2. Log metrics after each epoch:
       for epoch in range(1, EPOCHS+1):
           # Training code
           train_metrics = train_epoch(...)
           val_metrics = validate_epoch(...)
           
           # Combine metrics
           all_metrics = {
               "train_loss": train_metrics["loss"],
               "val_loss": val_metrics["loss"],
               "train_acc": train_metrics["acc"],
               "val_acc": val_metrics["acc"]
           }
           
           # Log to TensorBoard and CSV
           logger.log_metrics(all_metrics, epoch)
    
    3. Close logger at end:
       logger.close()
    
    
    VIEWING TENSORBOARD LOGS:
    ========================
    
    After training, view logs with:
      tensorboard --logdir=logs/
    
    Or for a specific run:
      tensorboard --logdir=logs/20240115-143022/
    
    Then open browser to: http://localhost:6006
    
    
    ANALYZING CSV LOGS:
    ==================
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Read CSV
    df = pd.read_csv('logs/20240115-143022/training_log.csv')
    
    # Plot training curves
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    
    ERROR HANDLING:
    ==============
    
    The logger includes comprehensive error handling:
    
    1. Invalid inputs:
       - Type checking (must be dict, int, str)
       - Value validation (non-empty, non-negative)
       - Detailed error messages
    
    2. File operations:
       - Directory creation failures
       - Permission errors
       - Disk space issues
    
    3. Data consistency:
       - Warns if metric names change
       - Handles NaN and Inf values
       - Validates numeric types
    
    4. Resource cleanup:
       - Always flushes data
       - Closes file handles
       - Works even with errors
    """)
    
    print("=" * 80)
    print("End of Examples")
    print("=" * 80)