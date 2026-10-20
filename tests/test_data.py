import pandas as pd

def test_dataset_exists():
    df = pd.read_csv("data/heart_disease_processed.csv")
    assert len(df) > 0