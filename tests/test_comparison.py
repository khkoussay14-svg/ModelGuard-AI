
from ml.model_comparison import compare_models


def test_number_of_models():
    """Verify that exactly three models are evaluated."""

    # Run the model comparison
    results = compare_models()

    # Check the number of evaluated models
    assert len(results) == 3


def test_model_names():
    """Verify that all expected models are included."""

    results = compare_models()

    # Extract model names from the results
    names = {result["model"] for result in results}

    expected_names = {
        "Decision Tree",
        "Random Forest",
        "Logistic Regression",
    }

    # Compare actual and expected model names
    assert names == expected_names


def test_metrics_are_valid():
    """Verify that evaluation metrics are within valid ranges."""

    results = compare_models()

    # Check every model and its evaluation metrics
    for result in results:
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ]:
            # Classification metrics must be between 0 and 1
            assert 0.0 <= result[metric] <= 1.0
