"""Explore and prepare the UCI Wine Quality data for binary classification.

Run from the repository root:
    python wine-classification/datapreprocessing/preprocess_and_explore.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "processed_data"


def load_and_combine_data() -> pd.DataFrame:
    """Load both semicolon-delimited source files and identify each wine type."""
    red_wine = pd.read_csv(RAW_DATA_DIR / "winequality-red.csv", sep=";")
    white_wine = pd.read_csv(RAW_DATA_DIR / "winequality-white.csv", sep=";")

    red_wine["wine_type"] = "red"
    white_wine["wine_type"] = "white"
    return pd.concat([red_wine, white_wine], ignore_index=True)


def print_exploration_summary(data: pd.DataFrame) -> None:
    """Print essential exploratory information before preprocessing."""
    print("\n--- Data Exploration Summary ---")
    print(f"Dataset shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"Duplicate rows: {data.duplicated().sum()}")
    print("\nMissing values per column:")
    print(data.isna().sum().to_string())
    print("\nWine type distribution:")
    print(data["wine_type"].value_counts().to_string())
    print("\nOriginal quality-score distribution:")
    print(data["quality"].value_counts().sort_index().to_string())
    print("\nNumeric feature summary:")
    print(data.drop(columns="wine_type").describe().round(2).to_string())


def prepare_features_and_target(data: pd.DataFrame) -> pd.DataFrame:
    """Create target column and remove raw quality to avoid target leakage."""
    prepared_data = data.copy()
    prepared_data["good_quality"] = (prepared_data["quality"] >= 7).astype(int)
    prepared_data = prepared_data.drop(columns="quality")

    print("\nBinary target distribution:")
    print(prepared_data["good_quality"].value_counts().rename({0: "Not Good", 1: "Good"}).to_string())
    return prepared_data


def save_train_test_split(data: pd.DataFrame) -> None:
    """Create a reproducible 80% training / 20% test split and save both CSV files."""
    train_data, test_data = train_test_split(
        data,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data["good_quality"],
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    train_path = OUTPUT_DIR / "train_data.csv"
    test_path = OUTPUT_DIR / "test_data.csv"
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print("\n--- Saved Processed Data ---")
    print(f"Training data (80%): {train_data.shape[0]} rows -> {train_path}")
    print(f"Test data (20%): {test_data.shape[0]} rows -> {test_path}")


def main() -> None:
    raw_data = load_and_combine_data()
    print_exploration_summary(raw_data)
    prepared_data = prepare_features_and_target(raw_data)
    save_train_test_split(prepared_data)


if __name__ == "__main__":
    main()
