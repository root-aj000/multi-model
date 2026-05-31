import pytest
from lib.preprocessing.text.tokenizer import tokenize_text


def test_tokenize_text_returns_dict():
    """tokenize_text returns dict with input_ids and attention_mask."""
    result = tokenize_text("hello world", max_length=16)
    assert "input_ids" in result
    assert "attention_mask" in result
