from use_cases.training.evaluate import evaluate_model, compute_metrics, save_results


def test_compute_metrics_accuracy():
    """compute_metrics calculates accuracy correctly."""
    preds = [0, 1, 2, 1]
    labels = [0, 1, 2, 0]
    result = compute_metrics(preds, labels)
    assert result["accuracy"] == 0.75


def test_compute_metrics_empty():
    """compute_metrics returns 0.0 accuracy for empty inputs."""
    result = compute_metrics([], [])
    assert result["accuracy"] == 0.0
