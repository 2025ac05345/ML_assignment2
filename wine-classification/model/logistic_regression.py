"""Train and evaluate Logistic Regression for Wine Classification.

Run from the repository root:
    python wine-classification/model/logistic_regression.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_DIR / "processed_data"
MODEL_PATH = Path(__file__).with_name("logistic_regression.joblib")
TARGET_COLUMN = "good_quality"
RANDOM_STATE = 42


def load_split_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load the previously created train/test CSV files."""
    train_data = pd.read_csv(PROCESSED_DATA_DIR / "train_data.csv")
    test_data = pd.read_csv(PROCESSED_DATA_DIR / "test_data.csv")

    x_train = train_data.drop(columns=TARGET_COLUMN)
    y_train = train_data[TARGET_COLUMN]
    x_test = test_data.drop(columns=TARGET_COLUMN)
    y_test = test_data[TARGET_COLUMN]
    return x_train, y_train, x_test, y_test


def build_pipeline(x_train: pd.DataFrame) -> Pipeline:
    """Encode the wine type, standardize numeric features, and configure Logistic Regression."""
    categorical_features = ["wine_type"]
    numeric_features = [column for column in x_train.columns if column not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculate every metric required by the assignment."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }


def main() -> None:
    x_train, y_train, x_test, y_test = load_split_data()
    model = build_pipeline(x_train)
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test)
    joblib.dump(model, MODEL_PATH)

    print("Logistic Regression model trained successfully.")
    print(f"Saved model: {MODEL_PATH}")
    print("\nTest-set evaluation metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
