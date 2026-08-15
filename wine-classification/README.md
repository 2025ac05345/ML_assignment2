# Wine Classification

## a. Problem Statement

This project predicts whether a wine is of **good quality** from its physicochemical properties. A wine is labelled **Good** when its `quality` score is 7 or higher; otherwise, it is labelled **Not Good**. The project compares multiple machine-learning classifiers and presents their test-data results in a Streamlit application.

## b. Dataset Description

The project uses the [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality), which contains red and white variants of Portuguese *Vinho Verde* wine.

| Dataset file | Instances | Description |
| --- | ---: | --- |
| `winequality-red.csv` | 1,599 | Red-wine observations |
| `winequality-white.csv` | 4,898 | White-wine observations |
| **Combined dataset** | **6,497** | Dataset used for modelling |

Each raw file contains 11 physicochemical input variables, such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, sulphates, alcohol, and pH, plus the `quality` score. During preprocessing, a `wine_type` feature is added while combining the files, giving 12 input features. The raw `quality` score is converted to the binary target `good_quality`:

```text
1 (Good)      : quality >= 7
0 (Not Good)  : quality < 7
```

Raw source files are stored in `data/`. Training and test files will be created during the preprocessing stage.

## c. GitHub Repository Link

Repository URL: **To be added after the repository is pushed to GitHub.**

## d. Models Used and Evaluation Metrics

The following models will be trained on the same stratified training/test split:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN) Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)

Each model will be evaluated on the test data using Accuracy, AUC, Precision, Recall, F1 score, and Matthews Correlation Coefficient (MCC). Although the assignment text refers to six models, it names the five models listed above; this project implements all five named models.

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | To be calculated | To be calculated | To be calculated | To be calculated | To be calculated | To be calculated |
| Decision Tree | To be calculated | To be calculated | To be calculated | To be calculated | To be calculated | To be calculated |
| kNN | To be calculated | To be calculated | To be calculated | To be calculated | To be calculated | To be calculated |
| Naive Bayes | To be calculated | To be calculated | To be calculated | To be calculated | To be calculated | To be calculated |
| Random Forest (Ensemble) | To be calculated | To be calculated | To be calculated | To be calculated | To be calculated | To be calculated |

### Performance Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | To be added after evaluation. |
| Decision Tree | To be added after evaluation. |
| kNN | To be added after evaluation. |
| Naive Bayes | To be added after evaluation. |
| Random Forest (Ensemble) | To be added after evaluation. |
| **Overall Winner for this dataset** | To be determined after comparing the test metrics. |

## Streamlit Application

The Streamlit app will allow the user to upload test data, choose a trained model, review its evaluation metrics, and view a confusion matrix or classification report.
