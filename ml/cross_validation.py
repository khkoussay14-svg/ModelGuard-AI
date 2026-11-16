
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.train_model import load_data


def run_cross_validation():
    """Compare classification models using stratified cross-validation."""

    # Load the training dataset
    X_train, _, y_train, _, _ = load_data()

    # Define the classification models
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    # Create five stratified folds
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    # Evaluate each model across the five folds
    for name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cross_validation,
            scoring="f1_macro"
        )

        # Store the mean and standard deviation of the scores
        results.append({
            "model": name,
            "mean_f1": scores.mean(),
            "std_f1": scores.std(),
            "fold_scores": scores.tolist(),
        })

    return results


if __name__ == "__main__":
    # Display cross-validation results
    for result in run_cross_validation():
        print(f"\nModel: {result['model']}")
        print(f"Mean F1-score: {result['mean_f1']:.2%}")
        print(f"Std deviation: {result['std_f1']:.2%}")
        print(f"Fold scores: {result['fold_scores']}")
