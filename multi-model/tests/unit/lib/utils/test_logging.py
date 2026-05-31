"""
Tests for the logging module.
"""

import logging

import pytest

from lib.utils.logging import TrainingLogger, create_logger


def test_create_logger_returns_logger():
    """create_logger returns a logging.Logger instance."""
    logger = create_logger("test", logging.DEBUG)
    assert isinstance(logger, logging.Logger)


def test_training_logger_init(tmp_path):
    """TrainingLogger creates the log directory on init."""
    log_dir = tmp_path / "logs"
    tl = TrainingLogger(str(log_dir))
    assert log_dir.exists()
    tl.close()


def test_is_open_false_initially(tmp_path):
    """_is_open is False after __init__."""
    log_dir = tmp_path / "logs"
    tl = TrainingLogger(str(log_dir))
    assert tl._is_open is False
    tl.close()


def test_is_open_true_after_first_log(tmp_path):
    """_is_open becomes True after the first call to log_metrics."""
    log_dir = tmp_path / "logs"
    tl = TrainingLogger(str(log_dir))
    assert tl._is_open is False
    tl.log_metrics({"loss": 0.5}, epoch=1)
    assert tl._is_open is True
    tl.close()


def test_is_open_false_after_close(tmp_path):
    """_is_open becomes False after close is called."""
    log_dir = tmp_path / "logs"
    tl = TrainingLogger(str(log_dir))
    tl.log_metrics({"loss": 0.5}, epoch=1)
    tl.close()
    assert tl._is_open is False


def test_validate_metrics_input_raises_on_non_dict(tmp_path):
    """log_metrics raises TypeError when given a non-dict argument."""
    log_dir = tmp_path / "logs"
    tl = TrainingLogger(str(log_dir))
    with pytest.raises(TypeError, match="Metrics must be a dict"):
        tl.log_metrics([1, 2, 3], epoch=1)


def test_validate_metrics_input_accepts_dict(tmp_path):
    """log_metrics does not raise when given a dict."""
    log_dir = tmp_path / "logs"
    tl = TrainingLogger(str(log_dir))
    # Should not raise
    tl.log_metrics({"loss": 0.5, "accuracy": 0.9}, epoch=1)
    tl.close()


def test_validate_log_dir_removed(tmp_path):
    """_validate_log_dir method no longer exists on TrainingLogger."""
    log_dir = tmp_path / "logs"
    tl = TrainingLogger(str(log_dir))
    assert not hasattr(tl, "_validate_log_dir")
    tl.close()
