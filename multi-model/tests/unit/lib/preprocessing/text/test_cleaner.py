from lib.preprocessing.text.cleaner import clean_text


def test_clean_text_lowercase():
    """clean_text converts to lowercase."""
    assert clean_text("Hello WORLD") == "hello world"


def test_clean_text_none():
    """clean_text returns empty string for None."""
    assert clean_text(None) == ""


def test_clean_text_special_chars():
    """clean_text removes special characters."""
    result = clean_text("Hello!!! @#$ 123")
    assert "!!!" not in result
