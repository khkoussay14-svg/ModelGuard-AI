
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ml.train_model import load_data


def compare_models():
    """Train and evaluate multiple classification models."""

    # Load the same training and testing datasets
    X_train, X_test, y_train, y_test, _ = load_data()

    # Define the classification algorithms
    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),
    }

    # Store the evaluation results
    results = []

    # Train and evaluate each model
    for model_name, model in models.items():

        # Train the model on the training dataset
        model.fit(X_train, y_train)

        # Generate predictions on the test dataset
        predictions = model.predict(X_test)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="macro"
        )
        recall = recall_score(
            y_test, predictions, average="macro"
        )
        f1 = f1_score(
            y_test, predictions, average="macro"
        )

        # Save the metrics for this model
        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

    return results


if __name__ == "__main__":
    # Print the evaluation results for each model
    for result in compare_models():
        print(f"\nModel: {result['model']}")
        print(f"Accuracy: {result['accuracy']:.2%}")
        print(f"Precision: {result['precision']:.2%}")
        print(f"Recall: {result['recall']:.2%}")
        print(f"F1-score: {result['f1_score']:.2%}")
