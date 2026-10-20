import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model():

    df = pd.read_csv("data/heart_disease_processed.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("=" * 50)
    print("Logistic Regression Results")
    print("=" * 50)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Accuracy :", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall   :", recall_score(y_test, predictions))
    print("F1 Score :", f1_score(y_test, predictions))

    print(confusion_matrix(y_test, predictions))

    print("\n" + "=" * 50)
    print("Random Forest Results")
    print("=" * 50)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)

    print("Accuracy :", accuracy_score(y_test, rf_predictions))
    print("Precision:", precision_score(y_test, rf_predictions))
    print("Recall   :", recall_score(y_test, rf_predictions))
    print("F1 Score :", f1_score(y_test, rf_predictions))

    print(confusion_matrix(y_test, rf_predictions))


if __name__ == "__main__":
    evaluate_model()