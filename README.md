# Telco Customer Churn Prediction Pipeline

This repository contains an end-to-end machine learning project for predicting customer churn for a telecommunications company. The project utilizes Scikit-learn's `Pipeline` and `GridSearchCV` to create a robust, reusable, and production-ready model.

## Problem Statement & Objective

**Problem:** A telecommunications company is experiencing customer churn, leading to revenue loss. They want to identify customers who are at a high risk of churning so they can proactively engage them with retention strategies.

**Objective:** To build and tune a machine learning model that accurately predicts customer churn. The primary goal is to create a reusable and production-ready pipeline that encapsulates all data preprocessing and modeling steps, making it easy to retrain and deploy.

## Dataset

The dataset used is `Telco_customer_churn.csv`. It contains demographic information, account details, and services subscribed to by 7,043 customers. The target variable is `Churn Value` (1 for churn, 0 for no churn).

Key columns include:
- **Demographics:** `Gender`, `Senior Citizen`, `Partner`, `Dependents`
- **Account Info:** `Tenure Months`, `Contract`, `Payment Method`, `Monthly Charges`, `Total Charges`
- **Services:** `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, etc.
- **Target:** `Churn Value`

## Methodology: The ML Pipeline

The project follows these key steps to ensure a robust and reproducible workflow:

1.  **Data Loading and Cleaning:**
    - The dataset is loaded into a pandas DataFrame.
    - The `Total Charges` column, initially an object type, is converted to a numeric type. Missing values resulting from this conversion (for customers with 0 tenure) are imputed with 0.
    - Irrelevant columns for prediction (`CustomerID`, `Country`, `State`, `Lat Long`, `Churn Reason`, etc.) are dropped.

2.  **Preprocessing Pipeline:**
    - A `ColumnTransformer` is used to apply different preprocessing steps to different types of columns. This is the core of our preprocessing pipeline.
    - **Numerical Features:** (`Tenure Months`, `Monthly Charges`, `Total Charges`) are scaled using `StandardScaler`.
    - **Categorical Features:** (`Gender`, `Contract`, `Internet Service`, etc.) are converted into numerical format using `OneHotEncoder`. `handle_unknown='ignore'` is set to gracefully handle new categories in future data.

3.  **Model Development and Training:**
    - Two classification models are trained and evaluated:
        - **Logistic Regression:** A reliable and interpretable baseline model.
        - **Random Forest Classifier:** A powerful ensemble model capable of capturing complex interactions.
    - Each model is embedded in a full Scikit-learn `Pipeline`, which chains the preprocessing steps and the model estimator.

4.  **Hyperparameter Tuning:**
    - `GridSearchCV` is used to systematically search for the best hyperparameters for each model. This process uses 5-fold cross-validation to ensure the selected parameters generalize well.
    - The search is optimized for the **ROC AUC score**, a robust metric for imbalanced classification problems.

5.  **Evaluation:**
    - The best-tuned models are evaluated on a held-out test set using the following metrics:
        - **Accuracy:** Overall correct predictions.
        - **Precision:** Of the customers we predicted to churn, how many actually churned.
        - **Recall:** Of all the customers who actually churned, how many did we correctly identify.
        - **F1-Score:** The harmonic mean of Precision and Recall.
        - **ROC AUC Score:** The model's ability to distinguish between churn and non-churn classes.
    - A confusion matrix is also generated for a visual representation of performance.

6.  **Exporting the Pipeline:**
    - The best-performing pipeline (Random Forest in this case) is saved to a file named `churn_pipeline.joblib` using the `joblib` library. This single file contains the entire workflow—from preprocessing to the trained model—and can be easily loaded for making predictions on new data.

## How to Run

1.  Clone this repository to your local machine.
2.  Open the `churn_prediction_pipeline.ipynb` file in Google Colab or a local Jupyter environment.
3.  Ensure you have all the required libraries installed (`pandas`, `scikit-learn`, `matplotlib`, `seaborn`).
4.  Run the notebook cells sequentially from top to bottom. The script will load the data, build the pipeline, train the models, evaluate them, and save the best one.

## Final Results

The Random Forest model demonstrated the best overall performance after hyperparameter tuning with GridSearchCV.

**Best Random Forest Model Performance on Test Set:**

| Metric    | Score  |
| :-------- | :----- |
| Accuracy  | 0.912  |
| Precision | 0.852  |
| Recall    | 0.803  |
| F1-Score  | 0.827  |
| ROC AUC   | 0.925  |

The model pipeline was successfully saved to `churn_pipeline.joblib` for future use.

## Files in this Repository

-   `churn_prediction_pipeline.ipynb`: The main Google Colab notebook containing all the code and analysis.
-   `churn_pipeline.joblib`: The final, exported machine learning pipeline.
-   `README.md`: This file.
