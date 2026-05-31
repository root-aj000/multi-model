from use_cases.prediction.predict_image import extract_text, predict_image


def test_extract_text_signature():
    """extract_text is importable and callable."""
    assert callable(extract_text)


def test_predict_image_signature():
    """predict_image is importable and callable."""
    assert callable(predict_image)
