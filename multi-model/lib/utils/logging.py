import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrainingLogger:
    """
    Logger for training metrics, writing to a CSV file.
    """

    def __init__(self, training_log_dir: str) -> None:
        """
        Initialize the TrainingLogger.

        Args:
            training_log_dir: Directory to store training logs.
        """
        self.training_log_dir = Path(training_log_dir)
        self.training_log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.training_log_dir / "training_log.csv"
        self.csv_file = None
        self.csv_writer = None
        self._is_open = False

    def log_metrics(self, metrics_dict: Dict[str, Any], epoch: int) -> None:
        """
        Log metrics for a given epoch.

        Args:
            metrics_dict: Dictionary of metric names to their values.
            epoch: The epoch number.
        """
        self._validate_metrics_input(metrics_dict)
        if self.csv_file is None:
            self._initialize_csv_file(metrics_dict)
        self._append_to_csv(metrics_dict, epoch)

    def _validate_metrics_input(self, metrics: Dict[str, Any]) -> None:
        """Validate that metrics input is a valid dictionary."""
        if not isinstance(metrics, dict):
            raise TypeError(f"Metrics must be a dict, not {type(metrics)}")

    def _initialize_csv_file(self, metrics_dict: Dict[str, Any]) -> None:
        """Create CSV header and open file for appending (preserves existing logs)."""
        fieldnames = ["epoch"] + sorted(metrics_dict.keys())
        file_exists = self.log_file.exists() and self.log_file.stat().st_size > 0
        self.csv_file = open(self.log_file, "a", newline="")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        if not file_exists:
            self.csv_writer.writeheader()
        self._is_open = True

    def _append_to_csv(self, metrics_dict: Dict[str, Any], epoch: int) -> None:
        """Append a row of metrics to the CSV and flush to disk."""
        row = {"epoch": epoch}
        row.update(metrics_dict)
        if self.csv_writer is not None:
            self.csv_writer.writerow(row)
            self.csv_file.flush()

    def close(self) -> None:
        """Close the log file handle."""
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self._is_open = False


def create_logger(name: str, level: int, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Create and return a named logger with the specified level and optional file handler.

    Args:
        name: The name for the logger.
        level: The logging level (e.g., logging.INFO).
        log_dir: Optional directory to create a file handler.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / f"{name}.log")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(file_handler)
    return logger
