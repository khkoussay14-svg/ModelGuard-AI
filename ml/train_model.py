
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def load_data():
    """Load and split the Iris dataset."""

    # Load the dataset
    iris = load_iris()

    # Separate input features and target labels
    X = iris.data
    y = iris.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, iris.target_names


def train_model(X_train, y_train):
    """Train a decision tree classifier."""

    # Create the classification model
    model = DecisionTreeClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Evaluate the trained model."""

    # Generate predictions
    predictions = model.predict(X_test)

    # Calculate model accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Display evaluation results
    print(f"Model accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=target_names
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return accuracy


if __name__ == "__main__":

    # Load the dataset
    X_train, X_test, y_train, y_test, names = load_data()

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test, names)
