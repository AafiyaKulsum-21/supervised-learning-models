"""
Classification Models Module
Implements Logistic Regression, Decision Tree, Random Forest, and KNN Classifiers.
Evaluates models using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), "..", ".cache", "matplotlib")


def get_classification_models():
    """Returns a dictionary of initialized classification models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="distance")
    }


def train_and_evaluate_classification(models, X_train_scaled, X_test_scaled, y_train, y_test, feature_names=None, output_dir="visualizations"):
    """
    Trains each classification model, computes evaluation metrics, and generates comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        predictions[name] = y_test_pred

        # Predict probabilities if available
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            probabilities[name] = y_prob
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = np.nan

        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

        results.append({
            "Model": name,
            "Train Accuracy": round(train_acc, 4),
            "Test Accuracy": round(test_acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4) if not np.isnan(auc) else "N/A"
        })

    results_df = pd.DataFrame(results)

    # 1. Confusion Matrix Subplots
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for idx, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"size": 13, "weight": "bold"})
        acc_val = results_df.loc[results_df['Model'] == name, 'Test Accuracy'].values[0]
        ax.set_title(f"{name}\n(Accuracy = {acc_val * 100:.2f}%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Class (0=Malignant, 1=Benign)", fontsize=10)
        ax.set_ylabel("True Class", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_confusion_matrices.png"), dpi=300)
    plt.close()

    # 2. Combined ROC Curves
    plt.figure(figsize=(9, 7))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (name, y_prob) in enumerate(probabilities.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = results_df.loc[results_df['Model'] == name, 'ROC-AUC'].values[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val})", color=palette[idx % len(palette)], lw=2.2)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_roc_curves.png"), dpi=300)
    plt.close()

    # 3. Model Comparison Bar Chart
    metrics_melted = results_df.melt(
        id_vars=["Model"],
        value_vars=["Test Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(12, 6))
    chart = sns.barplot(data=metrics_melted, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.title("Classification Performance Metrics Comparison", fontsize=13, fontweight="bold")
    plt.ylabel("Score", fontsize=11)
    plt.ylim(0.7, 1.05)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")

    for p in chart.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            chart.annotate(f"{height:.2f}",
                           (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='bottom',
                           fontsize=8, rotation=0, xytext=(0, 2),
                           textcoords='offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_model_comparison.png"), dpi=300)
    plt.close()

    # 4. Feature Importance for Random Forest Classifier (Top 10 features)
    if feature_names is not None and "Random Forest" in models:
        rf = models["Random Forest"]
        top_importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values().tail(10)
        plt.figure(figsize=(10, 6))
        top_importances.plot(kind="barh", color="#2ca02c", edgecolor="black")
        plt.title("Random Forest Classifier - Top 10 Feature Importances", fontsize=12, fontweight="bold")
        plt.xlabel("Gini Importance Score", fontsize=10)
        plt.grid(axis="x", linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "classification_feature_importance.png"), dpi=300)
        plt.close()

    return results_df, models, predictions
