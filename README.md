# Predicting Loan Defaults Using Machine Learning

## Overview
This project focuses on predicting loan defaults using various machine learning and deep learning models. The goal is to assist financial institutions in mitigating risks by accurately identifying potential loan defaulters. The project addresses challenges such as class imbalance, feature selection, and outlier detection while exploring advanced predictive techniques.

---

## Project Objectives
- **Enhance Prediction Accuracy**: Use advanced models to improve loan default predictions.
- **Handle Class Imbalance**: Apply resampling techniques to ensure model fairness.
- **Feature Optimization**: Identify the most influential features for predictive modeling.
- **Compare Models**: Evaluate the performance of multiple machine learning and deep learning models.

---

## Dataset
- **Source**: Kaggle Loan Default Dataset.
- **Size**: 87,501 records with 30 features.
- **Features**:
  - **Numerical**: Borrower's income, loan amount, debt-to-income ratio, etc.
  - **Categorical**: Homeownership status, loan reason, and more.

---

## Challenges and Solutions

### 1. Outliers
- **Problem**: Extreme values in features (e.g., income, loan amount) could skew model performance.
- **Solution**:
  - Applied the **Isolation Forest Algorithm** to identify anomalies.
  - Retained outliers to improve robustness, as they often indicate significant events (e.g., fraud, financial distress).

### 2. Class Imbalance
- **Problem**: The dataset was heavily imbalanced, with significantly more non-default cases than default cases.
- **Solution**:
  - Used **SMOTE-Tomek resampling** to balance the dataset by oversampling the minority class and removing overlapping samples.

### 3. Feature Selection
- **Problem**: Redundant and irrelevant features caused potential multicollinearity and overfitting.
- **Solution**:
  - Used **XGBoost's feature importance scores** to identify the most relevant features.
  - Removed irrelevant features (e.g., `ID`, `Sub_GGGrade`).

### 4. Data Preprocessing
- **Problem**: Missing values and high cardinality in categorical variables posed challenges during model training.
- **Solution**:
  - Imputed missing **numerical data** using median imputation and **categorical data** using mode imputation.
  - Applied **label encoding** to transform categorical features into numerical format.

---

## Models Implemented

### 1. Logistic Regression
- **Description**: A simple and interpretable baseline model for binary classification.
- **Performance**: Moderate accuracy; struggled with nonlinear relationships.

### 2. Decision Trees
- **Description**: A tree-based model that recursively partitions the data.
- **Challenges**: Prone to overfitting, especially with noisy data.
- **Solution**: Applied pruning and cross-validation to improve generalizability.

### 3. Random Forest
- **Description**: An ensemble method combining multiple decision trees.
- **Performance**: Robust results due to reduced overfitting.
- **Best Use Case**: Effective for capturing interactions between features.

### 4. XGBoost
- **Description**: A powerful gradient-boosting algorithm.
- **Strengths**:
  - Excellent performance on large, high-dimensional data.
  - Identified key features contributing to loan defaults.
- **Challenges**: Sensitive to hyperparameters; required extensive tuning.

### 5. Artificial Neural Networks (ANNs)
- **Description**: Explored complex patterns using a multilayer neural network.
- **Performance**: High accuracy; required significant computational resources.
- **Best Use Case**: Capturing nonlinear relationships in the data.

---

## Model Evaluation
Models were assessed using the following metrics:
- **Accuracy**: Overall correctness of predictions.
- **Precision**: Accuracy of positive predictions.
- **Recall**: Ability to identify all defaulters.
- **F1 Score**: Balance between precision and recall.
- **ROC AUC**: Ability to distinguish between classes.

---

## Results
- **Best Model**: XGBoost, with the highest accuracy and ROC AUC score.
- **Key Insight**: Handling class imbalance and selecting optimal features were critical for achieving high performance.
