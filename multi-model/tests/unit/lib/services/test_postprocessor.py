from lib.services.postprocessor import format_prediction_result


def test_format_prediction_result_removes_confidence():
    """format_prediction_result strips _confidence keys."""
    raw = {"sentiment": "positive", "sentiment_confidence": 0.95}
    result = format_prediction_result(raw)
    assert "sentiment" in result
    assert "sentiment_confidence" not in result


def test_format_prediction_result_removes_predicted_label_num():
    """format_prediction_result removes predicted_label_num."""
    raw = {"predicted_label_num": 2, "sentiment": "positive"}
    result = format_prediction_result(raw)
    assert "predicted_label_num" not in result
