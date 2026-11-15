
import numpy as np

from ml.train_model import load_data, train_model, evaluate_model


def test_model_accuracy():
    """Verify that the model meets the accuracy threshold."""

    # Load the dataset
    X_train, X_test, y_train, y_test, names = load_data()

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test, names)

    # Verify the minimum accuracy requirement
    assert accuracy >= 0.90


def test_prediction_count():
    """Verify that every test sample receives a prediction."""

    X_train, X_test, y_train, y_test, names = load_data()
    model = train_model(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_test)

    # Check that the prediction count matches the input count
    assert len(predictions) == len(X_test)


def test_prediction_classes():
    """Verify that predictions belong to known Iris classes."""

    X_train, X_test, y_train, y_test, names = load_data()
    model = train_model(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_test)

    # Ensure all predicted labels are valid
    assert np.isin(predictions, [0, 1, 2]).all()


def test_model_feature_count():
    """Verify that the model expects four input features."""

    X_train, X_test, y_train, y_test, names = load_data()
    model = train_model(X_train, y_train)

    # Check the number of input features
    assert model.n_features_in_ == 4
