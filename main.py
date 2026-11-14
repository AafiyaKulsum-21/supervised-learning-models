"""
Main Execution Script for Week 2 Task: Supervised Machine Learning Models
Trains and evaluates:
1. Regression Models: Linear Regression, Decision Tree, Random Forest, K-Nearest Neighbors
2. Classification Models: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors
Outputs performance tables, visualization plots, and generates the final submission report (.docx).
"""

import os
import pandas as pd
from src.data_loader import load_regression_data, load_classification_data
from src.regression_models import get_regression_models, train_and_evaluate_regression
from src.classification_models import get_classification_models, train_and_evaluate_classification


def run_pipeline():
    print("=" * 80)
    print("WEEK 2 TASK: SUPERVISED MACHINE LEARNING MODELS")
    print("Internship: AI Pioneers (Machine Learning) | Yuva Intern")
    print("Author: Aafiya Kulsum")
    print("=" * 80)

    # Directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

    # =========================================================================
    # 1. REGRESSION MODELS PIPELINE
    # =========================================================================
    print("\n[STEP 1/3] Training and Evaluating Regression Models...")
    X_train_r, X_test_r, y_train_r, y_test_r, reg_features, df_reg = load_regression_data()
    print(f"Loaded Regression Dataset: {df_reg.shape[0]} rows, {df_reg.shape[1] - 1} features")

    reg_models = get_regression_models()
    reg_results, trained_reg_models, reg_preds = train_and_evaluate_regression(
        reg_models, X_train_r, X_test_r, y_train_r, y_test_r,
        feature_names=reg_features, output_dir="visualizations"
    )

    reg_results_path = os.path.join("results", "regression_metrics.csv")
    reg_results.to_csv(reg_results_path, index=False)

    print("\n--- REGRESSION MODEL PERFORMANCE COMPARISON ---")
    print(reg_results.to_string(index=False))

    # =========================================================================
    # 2. CLASSIFICATION MODELS PIPELINE
    # =========================================================================
    print("\n" + "=" * 80)
    print("[STEP 2/3] Training and Evaluating Classification Models...")
    X_train_c, X_test_c, y_train_c, y_test_c, cls_features, df_cls = load_classification_data()
    print(f"Loaded Classification Dataset: {df_cls.shape[0]} rows, {df_cls.shape[1] - 1} features")

    cls_models = get_classification_models()
    cls_results, trained_cls_models, cls_preds = train_and_evaluate_classification(
        cls_models, X_train_c, X_test_c, y_train_c, y_test_c,
        feature_names=cls_features, output_dir="visualizations"
    )

    cls_results_path = os.path.join("results", "classification_metrics.csv")
    cls_results.to_csv(cls_results_path, index=False)

    print("\n--- CLASSIFICATION MODEL PERFORMANCE COMPARISON ---")
    print(cls_results.to_string(index=False))

    print("\n" + "=" * 80)
    print("[STEP 3/3] Generating Visualizations and Saving Artifacts...")
    print("Visualizations saved to: ./visualizations/")
    print("- regression_actual_vs_predicted.png")
    print("- regression_residuals.png")
    print("- regression_model_comparison.png")
    print("- classification_confusion_matrices.png")
    print("- classification_roc_curves.png")
    print("- classification_model_comparison.png")
    print("Metrics CSVs saved to: ./results/")

    print("\nAll models trained and evaluated successfully!")
    return reg_results, cls_results


if __name__ == "__main__":
    run_pipeline()
