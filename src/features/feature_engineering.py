import pandas as pd


def create_features():

    df = pd.read_csv("data/heart_disease_processed.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    print(X.shape)
    print(y.shape)

    return X, y


if __name__ == "__main__":
    create_features()