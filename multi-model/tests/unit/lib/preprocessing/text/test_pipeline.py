from lib.preprocessing.text.pipeline import (
    extract_keywords, extract_monetary_mention,
    extract_call_to_action, extract_objects_mentioned,
    build_text_pipeline,
)


def test_extract_keywords():
    """extract_keywords returns a list."""
    result = extract_keywords("big sale today")
    assert isinstance(result, list)
    assert "sale" in result


def test_extract_monetary_mention():
    """extract_monetary_mention finds dollar amounts."""
    result = extract_monetary_mention("only $9.99")
    assert result is not None


def test_build_text_pipeline():
    """build_text_pipeline returns a callable."""
    def cleaner(t): return t.lower()
    def tokenizer(t, m=512): return {"input_ids": t}
    pipeline = build_text_pipeline(cleaner, tokenizer)
    assert callable(pipeline)
