"""
Regression Models Module
Implements Linear Regression, Decision Tree, Random Forest, and KNN Regressors.
Evaluates models using MAE, MSE, RMSE, and R-squared (R2).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), "..", ".cache", "matplotlib")


def get_regression_models():
    """Returns a dictionary of initialized regression models."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5, weights="distance")
    }


def train_and_evaluate_regression(models, X_train_scaled, X_test_scaled, y_train, y_test, feature_names=None, output_dir="visualizations"):
    """
    Trains each regression model, computes evaluation metrics, and generates comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    predictions = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        predictions[name] = y_test_pred

        # Performance metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)

        results.append({
            "Model": name,
            "Train R2 Score": round(train_r2, 4),
            "Test R2 Score": round(test_r2, 4),
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4)
        })

    results_df = pd.DataFrame(results)

    # 1. Plot Actual vs Predicted for all models
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]
        ax.scatter(y_test, y_pred, alpha=0.6, color="#1f77b4", edgecolors="k", s=35)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", lw=2, label="Ideal Fit (y=x)")
        r2_val = results_df.loc[results_df['Model'] == name, 'Test R2 Score'].values[0]
        rmse_val = results_df.loc[results_df['Model'] == name, 'RMSE'].values[0]
        ax.set_title(f"{name}\n(R² = {r2_val:.4f}, RMSE = {rmse_val:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual Disease Progression", fontsize=10)
        ax.set_ylabel("Predicted Progression", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_actual_vs_predicted.png"), dpi=300)
    plt.close()

    # 2. Plot Residuals Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, color="#2ca02c", edgecolors="k", s=35)
        ax.axhline(0, color="crimson", linestyle="--", lw=2)
        ax.set_title(f"Residual Plot: {name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Values", fontsize=10)
        ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_residuals.png"), dpi=300)
    plt.close()

    # 3. Model Comparison Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(data=results_df, x="Model", y="Test R2 Score", ax=axes[0], hue="Model", palette="Blues_d", legend=False)
    axes[0].set_title("Model Comparison - R² Score (Higher is Better)", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", linestyle=":", alpha=0.7)

    sns.barplot(data=results_df, x="Model", y="RMSE", ax=axes[1], hue="Model", palette="Reds_d", legend=False)
    axes[1].set_title("Model Comparison - RMSE (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", linestyle=":", alpha=0.7)

    sns.barplot(data=results_df, x="Model", y="MAE", ax=axes[2], hue="Model", palette="Greens_d", legend=False)
    axes[2].set_title("Model Comparison - MAE (Lower is Better)", fontsize=12, fontweight="bold")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(axis="y", linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_model_comparison.png"), dpi=300)
    plt.close()

    # 4. Feature Importance for Random Forest Regressor
    if feature_names is not None and "Random Forest" in models:
        rf = models["Random Forest"]
        importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)
        plt.figure(figsize=(9, 6))
        importances.plot(kind="barh", color="#3470a3", edgecolor="black")
        plt.title("Random Forest Regressor - Feature Importances", fontsize=12, fontweight="bold")
        plt.xlabel("Relative Importance Score", fontsize=10)
        plt.grid(axis="x", linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "regression_feature_importance.png"), dpi=300)
        plt.close()

    return results_df, models, predictions
