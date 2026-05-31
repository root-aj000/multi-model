from pathlib import Path
from lib.utils.lifecycle import setup_directories, cleanup_upload_directory


def test_setup_directories_creates_dirs(tmp_path):
    """setup_directories creates the specified directories."""
    target = tmp_path / "new_dir"
    setup_directories([target])
    assert target.exists()


def test_cleanup_upload_directory(tmp_path):
    """cleanup_upload_directory removes files from the upload dir."""
    upload = tmp_path / "uploads"
    upload.mkdir()
    (upload / "test.txt").write_text("hello")
    cleanup_upload_directory(upload)
    assert not (upload / "test.txt").exists()
