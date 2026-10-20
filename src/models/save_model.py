import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def save_model():

    df = pd.read_csv("data/heart_disease_processed.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    Path("artifacts").mkdir(exist_ok=True)

    joblib.dump(
        model,
        "artifacts/random_forest_model.pkl"
    )

    print("Model saved successfully")


if __name__ == "__main__":
    save_model()