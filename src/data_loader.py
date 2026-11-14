"""
Data Loader Module
Provides standard, reliable datasets for both Regression and Classification tasks.
Saves local CSV copies for reproducibility and offline execution.
"""

import os
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_regression_data(data_dir="data"):
    """
    Loads regression dataset (Diabetes progression dataset with clinical features).
    Target: Quantitative measure of disease progression one year after baseline.
    """
    os.makedirs(data_dir, exist_ok=True)
    raw = load_diabetes(as_frame=True)
    df = raw.frame
    csv_path = os.path.join(data_dir, "diabetes_regression.csv")
    df.to_csv(csv_path, index=False)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling (crucial for Linear Regression and KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, df


def load_classification_data(data_dir="data"):
    """
    Loads classification dataset (Breast Cancer diagnostic dataset).
    Target: 0 (Malignant) vs 1 (Benign).
    """
    os.makedirs(data_dir, exist_ok=True)
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame
    csv_path = os.path.join(data_dir, "cancer_classification.csv")
    df.to_csv(csv_path, index=False)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling (crucial for Logistic Regression and KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, df
