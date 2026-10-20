import pandas as pd
from pathlib import Path


def preprocess_data():

    input_file = Path("data/heart_disease.csv")

    df = pd.read_csv(input_file)

    output_file = Path("data/heart_disease_processed.csv")

    df.to_csv(output_file, index=False)

    print(f"Processed dataset saved to: {output_file}")


if __name__ == "__main__":
    preprocess_data()