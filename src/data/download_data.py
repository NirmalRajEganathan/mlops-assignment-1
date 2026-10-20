from sklearn.datasets import fetch_openml
from pathlib import Path


def download_heart_disease_data():

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    dataset = fetch_openml(
        name="heart-disease",
        version=1,
        as_frame=True
    )

    df = dataset.data.copy()

    output_file = data_dir / "heart_disease.csv"

    df.to_csv(output_file, index=False)

    print(f"Dataset saved to: {output_file}")
    print(f"Dataset shape: {df.shape}")


if __name__ == "__main__":
    download_heart_disease_data()
