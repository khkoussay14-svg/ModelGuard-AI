
from ml.validation import validate_features


def test_valid_features():
    # Verify that valid input is accepted
    result = validate_features([5.1, 3.5, 1.4, 0.2])
    assert result is True


def test_negative_value():
    # Verify that negative values are rejected
    result = validate_features([5.1, -3.5, 1.4, 0.2])
    assert result is False


def test_missing_feature():
    # Verify that incomplete input is rejected
    result = validate_features([5.1, 3.5, 1.4])
    assert result is False


def test_invalid_type():
    # Verify that non-numeric values are rejected
    result = validate_features([5.1, "hello", 1.4, 0.2])
    assert result is False

def test_input_is_not_list():
    """Verify that non-list inputs are rejected."""

    # Pass a string instead of a list
    result = validate_features("invalid input")

    # Ensure the validation function rejects it
    assert result is False
