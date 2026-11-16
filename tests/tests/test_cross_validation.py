
from ml.cross_validation import run_cross_validation


def test_cross_validation_model_count():
    """Verify that all three models are evaluated."""

    results = run_cross_validation()

    assert len(results) == 3


def test_cross_validation_fold_count():
    """Verify that each model produces five validation scores."""

    results = run_cross_validation()

    for result in results:
        assert len(result["fold_scores"]) == 5


def test_cross_validation_score_range():
    """Verify that evaluation scores are within valid bounds."""

    results = run_cross_validation()

    for result in results:
        assert 0.0 <= result["mean_f1"] <= 1.0
        assert result["std_f1"] >= 0.0

        for score in result["fold_scores"]:
            assert 0.0 <= score <= 1.0
