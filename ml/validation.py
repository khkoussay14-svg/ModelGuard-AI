
def validate_features(features):
    """
    Check whether the input data is valid for our ML model.
    """

    # Ensure the input is a list
    if not isinstance(features, list):
        return False

    # Ensure the input contains exactly four features
    if len(features) != 4:
        return False

    # Check each feature
    for value in features:

        # Reject non-numeric values
        if type(value) not in (int, float):
            return False

        # Reject non-positive values
        if value <= 0:
            return False

    # All checks passed
    return True
