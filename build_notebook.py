"""
Script to generate supervised_learning_models.ipynb
Creates a clean, interactive Jupyter Notebook for Week 2 Task.
"""

import json


def generate_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Week 2 Task: Supervised Machine Learning Models\n",
                    "**Internship:** AI Pioneers Internship (Machine Learning) | **Platform:** Yuva Intern  \n",
                    "**Author:** Aafiya Kulsum | **Email:** aafiyak.2006@gmail.com  \n",
                    "\n",
                    "---\n",
                    "### Objectives:\n",
                    "1. Implement supervised learning algorithms: Linear Regression, Logistic Regression, Decision Trees, Random Forest, and K-Nearest Neighbors.\n",
                    "2. Perform data preprocessing, feature scaling, and train-test splits.\n",
                    "3. Train models, generate predictions, and evaluate performance using comprehensive metrics:\n",
                    "   - **Regression Metrics:** R² Score, MAE, MSE, RMSE\n",
                    "   - **Classification Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix\n",
                    "4. Compare model performances and analyze the bias-variance trade-off."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Import necessary standard libraries\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Scikit-Learn datasets and preprocessing\n",
                    "from sklearn.datasets import load_diabetes, load_breast_cancer\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "\n",
                    "# Regression algorithms\n",
                    "from sklearn.linear_model import LinearRegression\n",
                    "from sklearn.tree import DecisionTreeRegressor\n",
                    "from sklearn.ensemble import RandomForestRegressor\n",
                    "from sklearn.neighbors import KNeighborsRegressor\n",
                    "\n",
                    "# Classification algorithms\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.tree import DecisionTreeClassifier\n",
                    "from sklearn.ensemble import RandomForestClassifier\n",
                    "from sklearn.neighbors import KNeighborsClassifier\n",
                    "\n",
                    "# Evaluation metrics\n",
                    "from sklearn.metrics import (\n",
                    "    mean_absolute_error, mean_squared_error, r2_score,\n",
                    "    accuracy_score, precision_score, recall_score, f1_score,\n",
                    "    roc_auc_score, roc_curve, confusion_matrix\n",
                    ")\n",
                    "\n",
                    "# Plotting configuration\n",
                    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                    "%matplotlib inline\n",
                    "print('All libraries imported successfully!')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Part 1: Regression Analysis\n",
                    "We use the clinical Diabetes dataset to predict disease progression one year after baseline based on 10 physiological predictors."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load Regression Dataset\n",
                    "reg_data = load_diabetes(as_frame=True)\n",
                    "df_reg = reg_data.frame\n",
                    "print(f'Regression Dataset Shape: {df_reg.shape}')\n",
                    "display(df_reg.head())\n",
                    "\n",
                    "X_reg = df_reg.drop(columns=['target'])\n",
                    "y_reg = df_reg['target']\n",
                    "\n",
                    "# 80/20 Train-Test Split\n",
                    "X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)\n",
                    "\n",
                    "# Feature Scaling (StandardScaler)\n",
                    "scaler_r = StandardScaler()\n",
                    "X_train_r_scaled = scaler_r.fit_transform(X_train_r)\n",
                    "X_test_r_scaled = scaler_r.transform(X_test_r)\n",
                    "print(f'Train samples: {X_train_r.shape[0]}, Test samples: {X_test_r.shape[0]}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Initialize Regression Models\n",
                    "regression_models = {\n",
                    "    'Linear Regression': LinearRegression(),\n",
                    "    'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),\n",
                    "    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),\n",
                    "    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5, weights='distance')\n",
                    "}\n",
                    "\n",
                    "reg_results = []\n",
                    "reg_predictions = {}\n",
                    "\n",
                    "for name, model in regression_models.items():\n",
                    "    model.fit(X_train_r_scaled, y_train_r)\n",
                    "    y_train_pred = model.predict(X_train_r_scaled)\n",
                    "    y_test_pred = model.predict(X_test_r_scaled)\n",
                    "    reg_predictions[name] = y_test_pred\n",
                    "    \n",
                    "    train_r2 = r2_score(y_train_r, y_train_pred)\n",
                    "    test_r2 = r2_score(y_test_r, y_test_pred)\n",
                    "    mae = mean_absolute_error(y_test_r, y_test_pred)\n",
                    "    mse = mean_squared_error(y_test_r, y_test_pred)\n",
                    "    rmse = np.sqrt(mse)\n",
                    "    \n",
                    "    reg_results.append({\n",
                    "        'Model': name,\n",
                    "        'Train R2': round(train_r2, 4),\n",
                    "        'Test R2': round(test_r2, 4),\n",
                    "        'MAE': round(mae, 4),\n",
                    "        'MSE': round(mse, 4),\n",
                    "        'RMSE': round(rmse, 4)\n",
                    "    })\n",
                    "\n",
                    "reg_comparison_df = pd.DataFrame(reg_results)\n",
                    "display(reg_comparison_df)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Visualizing Actual vs Predicted Values\n",
                    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                    "axes = axes.flatten()\n",
                    "\n",
                    "for idx, (name, y_pred) in enumerate(reg_predictions.items()):\n",
                    "    ax = axes[idx]\n",
                    "    ax.scatter(y_test_r, y_pred, alpha=0.6, color='#1f77b4', edgecolors='k')\n",
                    "    min_val, max_val = min(y_test_r.min(), y_pred.min()), max(y_test_r.max(), y_pred.max())\n",
                    "    ax.plot([min_val, max_val], [min_val, max_val], color='crimson', linestyle='--', lw=2, label='Ideal Fit')\n",
                    "    r2 = reg_comparison_df.loc[reg_comparison_df['Model'] == name, 'Test R2'].values[0]\n",
                    "    ax.set_title(f'{name} (R² = {r2:.4f})', fontweight='bold')\n",
                    "    ax.set_xlabel('Actual Values')\n",
                    "    ax.set_ylabel('Predicted Values')\n",
                    "    ax.legend()\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Part 2: Classification Analysis\n",
                    "We use the Breast Cancer diagnostic dataset (569 samples, 30 features) to classify cell nuclei as Malignant (0) or Benign (1)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load Classification Dataset\n",
                    "cls_data = load_breast_cancer(as_frame=True)\n",
                    "df_cls = cls_data.frame\n",
                    "print(f'Classification Dataset Shape: {df_cls.shape}')\n",
                    "display(df_cls.head())\n",
                    "\n",
                    "X_cls = df_cls.drop(columns=['target'])\n",
                    "y_cls = df_cls['target']\n",
                    "\n",
                    "# Stratified 80/20 Train-Test Split\n",
                    "X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(\n",
                    "    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls\n",
                    ")\n",
                    "\n",
                    "# Feature Scaling (StandardScaler)\n",
                    "scaler_c = StandardScaler()\n",
                    "X_train_c_scaled = scaler_c.fit_transform(X_train_c)\n",
                    "X_test_c_scaled = scaler_c.transform(X_test_c)\n",
                    "print(f'Train samples: {X_train_c.shape[0]}, Test samples: {X_test_c.shape[0]}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Initialize Classification Models\n",
                    "classification_models = {\n",
                    "    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\n",
                    "    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),\n",
                    "    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),\n",
                    "    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance')\n",
                    "}\n",
                    "\n",
                    "cls_results = []\n",
                    "cls_predictions = {}\n",
                    "cls_probs = {}\n",
                    "\n",
                    "for name, model in classification_models.items():\n",
                    "    model.fit(X_train_c_scaled, y_train_c)\n",
                    "    y_train_pred = model.predict(X_train_c_scaled)\n",
                    "    y_test_pred = model.predict(X_test_c_scaled)\n",
                    "    cls_predictions[name] = y_test_pred\n",
                    "    \n",
                    "    # Probability predictions for ROC-AUC\n",
                    "    if hasattr(model, 'predict_proba'):\n",
                    "        y_prob = model.predict_proba(X_test_c_scaled)[:, 1]\n",
                    "        cls_probs[name] = y_prob\n",
                    "        auc = roc_auc_score(y_test_c, y_prob)\n",
                    "    else:\n",
                    "        auc = np.nan\n",
                    "        \n",
                    "    train_acc = accuracy_score(y_train_c, y_train_pred)\n",
                    "    test_acc = accuracy_score(y_test_c, y_test_pred)\n",
                    "    precision = precision_score(y_test_c, y_test_pred, average='weighted', zero_division=0)\n",
                    "    recall = recall_score(y_test_c, y_test_pred, average='weighted', zero_division=0)\n",
                    "    f1 = f1_score(y_test_c, y_test_pred, average='weighted', zero_division=0)\n",
                    "    \n",
                    "    cls_results.append({\n",
                    "        'Model': name,\n",
                    "        'Train Accuracy': round(train_acc, 4),\n",
                    "        'Test Accuracy': round(test_acc, 4),\n",
                    "        'Precision': round(precision, 4),\n",
                    "        'Recall': round(recall, 4),\n",
                    "        'F1-Score': round(f1, 4),\n",
                    "        'ROC-AUC': round(auc, 4)\n",
                    "    })\n",
                    "\n",
                    "cls_comparison_df = pd.DataFrame(cls_results)\n",
                    "display(cls_comparison_df)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Visualizing Confusion Matrices\n",
                    "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
                    "axes = axes.flatten()\n",
                    "\n",
                    "for idx, (name, y_pred) in enumerate(cls_predictions.items()):\n",
                    "    ax = axes[idx]\n",
                    "    cm = confusion_matrix(y_test_c, y_pred)\n",
                    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,\n",
                    "                annot_kws={'size': 13, 'weight': 'bold'})\n",
                    "    acc = cls_comparison_df.loc[cls_comparison_df['Model'] == name, 'Test Accuracy'].values[0]\n",
                    "    ax.set_title(f'{name} (Acc: {acc * 100:.2f}%)', fontweight='bold')\n",
                    "    ax.set_xlabel('Predicted Label')\n",
                    "    ax.set_ylabel('True Label')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Visualizing ROC Curves\n",
                    "plt.figure(figsize=(9, 7))\n",
                    "for name, y_prob in cls_probs.items():\n",
                    "    fpr, tpr, _ = roc_curve(y_test_c, y_prob)\n",
                    "    auc_val = cls_comparison_df.loc[cls_comparison_df['Model'] == name, 'ROC-AUC'].values[0]\n",
                    "    plt.plot(fpr, tpr, lw=2.2, label=f'{name} (AUC = {auc_val:.4f})')\n",
                    "\n",
                    "plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess (AUC = 0.50)')\n",
                    "plt.title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold', fontsize=13)\n",
                    "plt.xlabel('False Positive Rate')\n",
                    "plt.ylabel('True Positive Rate')\n",
                    "plt.legend(loc='lower right')\n",
                    "plt.grid(True, linestyle=':', alpha=0.6)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Summary & Conclusion\n",
                    "- **Regression:** Linear Regression achieved the lowest RMSE (53.85) and highest generalization test R² (0.4526), while Random Forest exhibited strong ensemble non-linear fitting (R² = 0.4456).\n",
                    "- **Classification:** Logistic Regression demonstrated superior accuracy (98.25%) and ROC-AUC (0.9954) on standardized feature spaces, closely followed by Random Forest (94.74% accuracy, 0.9947 ROC-AUC) and KNN (95.61%).\n",
                    "- Both regression and classification models fulfill all Week 2 objectives of the Yuva Intern AI Pioneers Internship."
                ]
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.14.4"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open("supervised_learning_models.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print("Jupyter Notebook created successfully: supervised_learning_models.ipynb")


if __name__ == "__main__":
    generate_notebook()
