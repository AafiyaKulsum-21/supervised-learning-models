# Week 2: Supervised Machine Learning Models 🚀

![Python](https://img.shields.io/badge/Python-3.14+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.1.1-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

**Program:** AI Pioneers Internship (Machine Learning)  
**Organization:** Yuva Intern  
**Intern:** Aafiya Kulsum  
**Email:** [aafiyak.2006@gmail.com](mailto:aafiyak.2006@gmail.com)  

---

## 📌 Project Overview
This repository contains the complete implementation, training, prediction, and comparative performance evaluation for the **Week 2 Internship Task: Supervised Machine Learning Models**.

The project investigates both continuous estimation (**Regression**) and discrete categorization (**Classification**) using five foundational supervised learning algorithms:
1. **Linear Regression**
2. **Logistic Regression**
3. **Decision Trees** *(Regressor & Classifier)*
4. **Random Forest** *(Regressor & Classifier)*
5. **K-Nearest Neighbors (KNN)** *(Regressor & Classifier)*

---

## 🎯 Task Objectives & Deliverables
- [x] Implement supervised machine learning algorithms using Scikit-Learn.
- [x] Standardize features and perform stratified train-test splits.
- [x] Compute comprehensive evaluation metrics:
  - **Regression:** $R^2$ Score, Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE).
  - **Classification:** Accuracy, Precision, Recall, F1-Score, ROC-AUC Score, and Confusion Matrix.
- [x] Compare model performance side-by-side using tabular summaries and publication-ready charts.
- [x] Generate a formatted Word Report (`.docx`) and an interactive Jupyter Notebook (`supervised_learning_models.ipynb`).

---

## 📊 Experimental Results

### 1. Regression Models Comparison (Diabetes Progression Dataset)
Target: Quantitative disease progression one year after baseline.

| Model | Train $R^2$ | Test $R^2$ | MAE | MSE | RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.5279 | **0.4526** | 42.79 | **2900.19** | **53.85** |
| **Random Forest Regressor** | 0.8827 | **0.4456** | 44.12 | 2937.14 | 54.20 |
| **K-Nearest Neighbors (KNN)** | 1.0000 | 0.4340 | **42.16** | 2998.68 | 54.76 |
| **Decision Tree Regressor** | 0.7468 | 0.3072 | 47.48 | 3670.67 | 60.59 |

> **Key Regression Insight:** Linear Regression demonstrated superior generalization with the lowest RMSE (53.85) and highest test $R^2$ (0.4526), indicating a predominantly linear relationship among physiological predictors without overfitting.

---

### 2. Classification Models Comparison (Breast Cancer Diagnostic Dataset)
Target: Binary classification — Malignant (0) vs Benign (1).

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 98.90% | **98.25%** | **0.9825** | **0.9825** | **0.9825** | **0.9954** |
| **K-Nearest Neighbors (KNN)** | 100.00% | **95.61%** | 0.9561 | 0.9561 | 0.9560 | 0.9795 |
| **Random Forest Classifier** | 99.56% | **94.74%** | 0.9474 | 0.9474 | 0.9474 | **0.9947** |
| **Decision Tree Classifier** | 99.34% | 92.11% | 0.9234 | 0.9211 | 0.9216 | 0.9163 |

> **Key Classification Insight:** Logistic Regression attained outstanding diagnostic accuracy of **98.25%** and an ROC-AUC of **0.9954** on standardized cell nuclei features, closely followed by Random Forest (ROC-AUC = **0.9947**).

---

## 📈 Visualizations Generated

All high-resolution figures are saved under the `visualizations/` directory:
- `regression_model_comparison.png`: Side-by-side comparison of $R^2$, RMSE, and MAE across models.
- `regression_actual_vs_predicted.png`: Scatter plots comparing predicted vs ground truth with ideal $y=x$ lines.
- `regression_residuals.png`: Residual distribution plots detecting heteroscedasticity.
- `regression_feature_importance.png`: Feature importance bar plot for Random Forest Regressor.
- `classification_model_comparison.png`: Comparison bar chart across Accuracy, Precision, Recall, and F1-Score.
- `classification_confusion_matrices.png`: Confusion matrices heatmaps for all four classifiers.
- `classification_roc_curves.png`: Combined ROC curves with calculated AUC scores.
- `classification_feature_importance.png`: Top 10 diagnostic predictors identified by Random Forest.

---

## 📁 Repository Structure
```
machine-learning/
│
├── data/
│   ├── diabetes_regression.csv          # Clinical regression dataset
│   └── cancer_classification.csv        # Diagnostic classification dataset
│
├── results/
│   ├── regression_metrics.csv           # Exported regression metrics table
│   └── classification_metrics.csv       # Exported classification metrics table
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Data ingestion, train-test split & scaling
│   ├── regression_models.py             # Regression algorithms & evaluation
│   └── classification_models.py         # Classification algorithms & evaluation
│
├── visualizations/                      # High-resolution generated charts
│   ├── classification_confusion_matrices.png
│   ├── classification_model_comparison.png
│   ├── classification_roc_curves.png
│   ├── classification_feature_importance.png
│   ├── regression_actual_vs_predicted.png
│   ├── regression_model_comparison.png
│   ├── regression_residuals.png
│   └── regression_feature_importance.png
│
├── build_notebook.py                    # Script to build Jupyter Notebook
├── generate_report.py                   # Word Document (.docx) report generator
├── main.py                              # Main orchestration pipeline
├── requirements.txt                     # Python dependencies
├── supervised_learning_models.ipynb     # Interactive Jupyter Notebook
├── Week_2_Supervised_Machine_Learning_Report.docx  # Formal Word Submission Report
└── README.md                            # Comprehensive project documentation
```

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/YuvaIntern-AI-Pioneers/supervised-learning-models.git
cd supervised-learning-models
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Full Machine Learning Pipeline
```bash
python main.py
```

### 4. Generate the Formal Word Document Report (.docx)
```bash
python generate_report.py
```

### 5. Launch the Jupyter Notebook
```bash
jupyter notebook supervised_learning_models.ipynb
```

---

## 📝 License
This project is developed as part of the **AI Pioneers Internship** curriculum and is licensed under the MIT License.
