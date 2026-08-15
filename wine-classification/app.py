"""Interactive evaluation app for the Wine Classification project."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "model"
DEFAULT_TEST_DATA = PROJECT_DIR / "processed_data" / "test_data.csv"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "k-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}


@st.cache_resource
def load_model(model_path: str):
    """Load one previously trained classifier or preprocessing pipeline."""
    return joblib.load(model_path)


def prepare_test_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract the binary target and leave only model input features."""
    if "good_quality" in data.columns:
        target = data["good_quality"]
        features = data.drop(columns="good_quality")
    elif "quality" in data.columns:
        target = (data["quality"] >= 7).astype(int)
        features = data.drop(columns="quality")
    else:
        raise ValueError(
            "The uploaded test CSV must contain either a 'good_quality' or 'quality' column."
        )
    return features, target.astype(int)


def calculate_auc(model, features: pd.DataFrame, target: pd.Series) -> float:
    """Return AUC using probability scores when the classifier provides them."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        return roc_auc_score(target, probabilities[:, 1])
    if hasattr(model, "decision_function"):
        return roc_auc_score(target, model.decision_function(features))
    return float("nan")


def evaluate_model(model, features: pd.DataFrame, target: pd.Series):
    predictions = model.predict(features)
    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": calculate_auc(model, features, target),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1 Score": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }
    return predictions, metrics


def show_evaluation(model, test_data: pd.DataFrame):
    features, target = prepare_test_data(test_data)
    predictions, metrics = evaluate_model(model, features, target)

    st.subheader("Evaluation Metrics")
    metric_columns = st.columns(3)
    for index, (name, value) in enumerate(metrics.items()):
        metric_columns[index % 3].metric(name, "Not available" if pd.isna(value) else f"{value:.3f}")

    st.subheader("Confusion Matrix")
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Good", "Good"],
        yticklabels=["Not Good", "Good"],
        ax=axis,
    )
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    st.pyplot(figure)
    plt.close(figure)

    st.subheader("Classification Report")
    report = pd.DataFrame(
        classification_report(
            target,
            predictions,
            target_names=["Not Good", "Good"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    st.dataframe(report.round(3), use_container_width=True)


st.set_page_config(page_title="Wine Classification", page_icon="🍷", layout="wide")
st.title("🍷 Wine Classification")
st.write("Evaluate trained classifiers using a CSV containing **test data only**.")

with st.sidebar:
    st.header("Model Selection")
    selected_name = st.selectbox("Choose a classification model", list(MODEL_FILES))
    uploaded_file = st.file_uploader("Upload test data (CSV)", type="csv")

model_path = MODEL_DIR / MODEL_FILES[selected_name]
if not model_path.exists():
    st.warning(
        f"The saved **{selected_name}** model is not available yet. "
        f"Create `{model_path.name}` in the `model/` folder during training."
    )
    st.stop()

try:
    classifier = load_model(str(model_path))
except Exception as error:
    st.error(f"The saved model could not be loaded: {error}")
    st.stop()

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    source_name = uploaded_file.name
elif DEFAULT_TEST_DATA.exists():
    test_data = pd.read_csv(DEFAULT_TEST_DATA)
    source_name = "processed_data/test_data.csv"
else:
    st.info("Upload a test CSV to evaluate the selected model.")
    st.stop()

st.success(f"Loaded {len(test_data):,} test records from `{source_name}`.")
with st.expander("Preview test data"):
    st.dataframe(test_data.head(), use_container_width=True)

try:
    show_evaluation(classifier, test_data)
except ValueError as error:
    st.error(str(error))
except Exception as error:
    st.error(f"Evaluation failed. Confirm that this test CSV matches the training features. Details: {error}")
