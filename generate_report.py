"""
Word Document Report Generator for Week 2 Task
Generates a professional, comprehensive report (.docx) for Yuva Intern submission.
Includes methodology, math formulas, results tables, and embedded high-resolution figures.
"""

import os
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tc_pr.append(shd)


def style_table(table, header_bg="1F497D", stripe_bg="F2F5F9"):
    """Styles a python-docx table with modern styling and zebra striping."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, row in enumerate(table.rows):
        # Prevent row split across pages
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if idx == 0:
                set_cell_background(cell, header_bg)
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)
                        run.font.size = Pt(10)
            else:
                if idx % 2 == 0:
                    set_cell_background(cell, stripe_bg)
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(9.5)


def create_report():
    doc = Document()

    # Set normal style font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(51, 51, 51)

    # Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # -------------------------------------------------------------------------
    # COVER / HEADER SECTION
    # -------------------------------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(12)
    title_p.paragraph_format.space_after = Pt(4)
    run_title = title_p.add_run("WEEK 2 INTERNSHIP REPORT: SUPERVISED MACHINE LEARNING MODELS")
    run_title.font.bold = True
    run_title.font.size = Pt(22)
    run_title.font.color.rgb = RGBColor(31, 73, 125)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(18)
    run_sub = sub_p.add_run("Comprehensive Implementation, Training, and Comparative Performance Evaluation of Regression and Classification Algorithms using Scikit-Learn")
    run_sub.font.italic = True
    run_sub.font.size = Pt(12)
    run_sub.font.color.rgb = RGBColor(89, 89, 89)

    # Metadata Box
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Candidate Name:", "Aafiya Kulsum"),
        ("Email Address:", "aafiyak.2006@gmail.com"),
        ("Program & Domain:", "AI Pioneers Internship (Machine Learning)"),
        ("Platform / Organization:", "Yuva Intern"),
        ("Submission Deliverable:", "Week 2 Task - Regression & Classification Model Comparison")
    ]
    for row_idx, (k, v) in enumerate(meta_data):
        row = meta_table.rows[row_idx]
        cell_k, cell_v = row.cells[0], row.cells[1]
        p_k = cell_k.paragraphs[0]
        run_k = p_k.add_run(k)
        run_k.font.bold = True
        run_k.font.size = Pt(10)
        run_k.font.color.rgb = RGBColor(31, 73, 125)
        cell_k.width = Inches(2.2)

        p_v = cell_v.paragraphs[0]
        run_v = p_v.add_run(v)
        run_v.font.size = Pt(10)
        cell_v.width = Inches(4.3)
        set_cell_background(cell_k, "F2F5F9")
        set_cell_background(cell_v, "FAFAFA")

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------------------
    # 1. EXECUTIVE SUMMARY
    # -------------------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    run_h1 = h1.add_run("1. Executive Summary")
    run_h1.font.color.rgb = RGBColor(31, 73, 125)

    p = doc.add_paragraph(
        "Supervised machine learning represents the cornerstone of predictive modeling, where algorithms learn the "
        "underlying mathematical mapping between input feature spaces and target variables from labeled historical data. "
        "The objective of this Week 2 task is to design, implement, evaluate, and compare fundamental supervised learning algorithms "
        "using Python and Scikit-learn across both continuous (regression) and categorical (classification) paradigms."
    )
    p = doc.add_paragraph(
        "Specifically, this project implements five key algorithms: Linear Regression, Logistic Regression, Decision Trees, "
        "Random Forest, and K-Nearest Neighbors (KNN). To ensure robust, reproducible science, all models undergo strict data "
        "preprocessing, including feature standardization and stratified train-test splitting. Rigorous quantitative evaluation is "
        "conducted using industry-standard metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error "
        "(RMSE), and R-squared (R²) for regression; and Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices for classification. "
        "Our empirical results highlight the bias-variance trade-offs of each algorithm and provide concrete guidance on model selection."
    )

    # -------------------------------------------------------------------------
    # 2. THEORETICAL OVERVIEW OF ALGORITHMS
    # -------------------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    run_h1 = h1.add_run("2. Overview of Supervised Learning Algorithms")
    run_h1.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_heading("2.1 Linear Regression", level=2)
    doc.add_paragraph(
        "Linear Regression establishes a linear relationship between input features X and a continuous target y by minimizing "
        "the Residual Sum of Squares (Ordinary Least Squares - OLS). The hypothesis is formulated as:"
    )
    doc.add_paragraph(
        "    y_hat = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ... + beta_n * x_n\n"
        "Strengths: High interpretability, analytical solution, computationally lightweight.\n"
        "Limitations: Assumes linear independence and normality; vulnerable to multicollinearity and extreme outliers."
    )

    doc.add_heading("2.2 Logistic Regression", level=2)
    doc.add_paragraph(
        "Despite its name, Logistic Regression is a foundational classification algorithm. It maps linear combinations of input features "
        "into calibrated probabilities between 0 and 1 via the standard sigmoid (logistic) function:"
    )
    doc.add_paragraph(
        "    sigma(z) = 1 / (1 + e^(-z)), where z = w^T * x + b\n"
        "The parameters are optimized using maximum likelihood estimation (minimizing binary cross-entropy loss). "
        "It excels when data is approximately linearly separable and provides probabilistically calibrated confidence scores."
    )

    doc.add_heading("2.3 Decision Trees (Classifier & Regressor)", level=2)
    doc.add_paragraph(
        "Decision Trees perform non-parametric recursive partitioning of the feature space into orthogonal axis-aligned hyperplanes. "
        "At each node, split thresholds are selected to maximize Information Gain (using Gini Impurity or Shannon Entropy for classification, "
        "and Variance Reduction / MSE for regression). While intuitive and transparent, unconstrained decision trees readily overfit training noise."
    )

    doc.add_heading("2.4 Random Forest (Classifier & Regressor)", level=2)
    doc.add_paragraph(
        "Random Forest is an ensemble meta-estimator that implements bootstrap aggregating (bagging) over randomized decision trees. "
        "By training B independent trees on bootstrap samples with random feature subsets at each split (feature bagging), it drastically reduces "
        "the variance of individual trees without inflating bias. It also generates intrinsic Gini feature importance rankings."
    )

    doc.add_heading("2.5 K-Nearest Neighbors (KNN)", level=2)
    doc.add_paragraph(
        "KNN is an instance-based, non-parametric lazy learner that makes predictions based on local proximity in feature space. "
        "Given a query instance x, it computes Euclidean distance d(x, x_i) = sqrt(sum((x_j - x_ij)^2)) to all training points, identifies the K closest neighbors, "
        "and aggregates their labels via majority voting (classification) or distance-weighted averaging (regression). Feature scaling is imperative for KNN."
    )

    # -------------------------------------------------------------------------
    # 3. EXPERIMENTAL METHODOLOGY & PREPROCESSING
    # -------------------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    run_h1 = h1.add_run("3. Experimental Methodology & Preprocessing Pipeline")
    run_h1.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "To guarantee valid empirical comparison, standardized machine learning pipelines were developed using Scikit-Learn:\n"
        "• Regression Dataset: The Diabetes progression benchmark dataset (442 instances, 10 physiological predictors including age, sex, BMI, BP, and 6 serum measurements).\n"
        "• Classification Dataset: The Diagnostic Breast Cancer benchmark dataset (569 instances, 30 morphological cell nucleus features, binary target: Malignant vs Benign).\n"
        "• Data Partitioning: 80% Training subset, 20% Holdout Testing subset, fixed random seed (seed=42) for deterministic reproducibility, with class stratification applied to classification.\n"
        "• Feature Standardization (StandardScaler): Features are centered to zero mean and scaled to unit variance (z = (x - mu) / sigma). This is strictly fit on training data and applied to test data to prevent data leakage."
    )

    # -------------------------------------------------------------------------
    # 4. REGRESSION EXPERIMENTS & PERFORMANCE COMPARISON
    # -------------------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    run_h1 = h1.add_run("4. Regression Analysis: Results & Model Comparison")
    run_h1.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "The four regression models (Linear Regression, Decision Tree Regressor, Random Forest Regressor, and K-Nearest Neighbors Regressor) "
        "were trained and evaluated across four metrics on unseen test data:\n"
        "1. R-Squared (R²): The proportion of target variance explained by model features (higher is superior).\n"
        "2. Mean Absolute Error (MAE): The average absolute residual magnitude (lower is superior).\n"
        "3. Mean Squared Error (MSE): The average squared difference between predictions and ground truth.\n"
        "4. Root Mean Squared Error (RMSE): Square root of MSE, penalizing large outlier errors in original units."
    )

    # Load results
    reg_df = pd.read_csv(os.path.join("results", "regression_metrics.csv"))

    # Table
    table_r = doc.add_table(rows=len(reg_df) + 1, cols=6)
    headers_r = ["Model", "Train R²", "Test R²", "MAE", "MSE", "RMSE"]
    for c_idx, h in enumerate(headers_r):
        table_r.rows[0].cells[c_idx].paragraphs[0].text = h

    for r_idx, row in reg_df.iterrows():
        r_cells = table_r.rows[r_idx + 1].cells
        r_cells[0].paragraphs[0].text = str(row["Model"])
        r_cells[1].paragraphs[0].text = f"{row['Train R2 Score']:.4f}"
        r_cells[2].paragraphs[0].text = f"{row['Test R2 Score']:.4f}"
        r_cells[3].paragraphs[0].text = f"{row['MAE']:.2f}"
        r_cells[4].paragraphs[0].text = f"{row['MSE']:.2f}"
        r_cells[5].paragraphs[0].text = f"{row['RMSE']:.2f}"

    style_table(table_r)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    doc.add_paragraph(
        "Key Observations (Regression):\n"
        "• Linear Regression achieved the strongest generalization performance on test data with an R² score of 0.4526 and the lowest test RMSE (53.85). "
        "Because physiological predictors exhibit near-linear associations with diabetes progression, linear regularization effectively avoids over-parameterization.\n"
        "• Random Forest Regressor achieved a close second test R² of 0.4456 and RMSE of 54.20. While its training R² reached 0.8827, bagging successfully mitigated extreme tree overfitting.\n"
        "• Single Decision Tree suffered significant overfitting (Train R² = 0.7468 vs Test R² = 0.3072, RMSE = 60.59), confirming that individual greedy partitioning creates high-variance splits.\n"
        "• KNN Regressor with distance weighting attained an effective test R² of 0.4340 and the lowest MAE of 42.16."
    )

    # Embed Regression Figures
    fig_comp_r = os.path.join("visualizations", "regression_model_comparison.png")
    if os.path.exists(fig_comp_r):
        doc.add_picture(fig_comp_r, width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 1: Comparative Evaluation of Regression Models (R², RMSE, and MAE)")
        run_cap.font.italic = True
        run_cap.font.size = Pt(9.5)

    fig_act_r = os.path.join("visualizations", "regression_actual_vs_predicted.png")
    if os.path.exists(fig_act_r):
        doc.add_picture(fig_act_r, width=Inches(6.0))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 2: Actual vs Predicted Regression Values with Ideal Reference Line (y=x)")
        run_cap.font.italic = True
        run_cap.font.size = Pt(9.5)

    # -------------------------------------------------------------------------
    # 5. CLASSIFICATION EXPERIMENTS & PERFORMANCE COMPARISON
    # -------------------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    run_h1 = h1.add_run("5. Classification Analysis: Results & Model Comparison")
    run_h1.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "Four classification models (Logistic Regression, Decision Tree Classifier, Random Forest Classifier, and K-Nearest Neighbors Classifier) "
        "were trained and evaluated across standard diagnostic metrics:\n"
        "• Accuracy: Overall proportion of correct predictions (TP + TN) / Total.\n"
        "• Precision: Proportion of predicted positives that were truly positive (TP / (TP + FP)).\n"
        "• Recall (Sensitivity): Proportion of actual positives identified (TP / (TP + FN)), crucial for medical triage.\n"
        "• F1-Score: Harmonic mean of Precision and Recall (2 * (Precision * Recall) / (Precision + Recall)).\n"
        "• ROC-AUC: Area under the Receiver Operating Characteristic curve, measuring discriminative power across all decision thresholds."
    )

    # Load results
    cls_df = pd.read_csv(os.path.join("results", "classification_metrics.csv"))

    table_c = doc.add_table(rows=len(cls_df) + 1, cols=7)
    headers_c = ["Model", "Train Acc.", "Test Acc.", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    for c_idx, h in enumerate(headers_c):
        table_c.rows[0].cells[c_idx].paragraphs[0].text = h

    for r_idx, row in cls_df.iterrows():
        c_cells = table_c.rows[r_idx + 1].cells
        c_cells[0].paragraphs[0].text = str(row["Model"])
        c_cells[1].paragraphs[0].text = f"{float(row['Train Accuracy']) * 100:.2f}%"
        c_cells[2].paragraphs[0].text = f"{float(row['Test Accuracy']) * 100:.2f}%"
        c_cells[3].paragraphs[0].text = f"{row['Precision']:.4f}"
        c_cells[4].paragraphs[0].text = f"{row['Recall']:.4f}"
        c_cells[5].paragraphs[0].text = f"{row['F1-Score']:.4f}"
        c_cells[6].paragraphs[0].text = str(row["ROC-AUC"])

    style_table(table_c)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    doc.add_paragraph(
        "Key Observations (Classification):\n"
        "• Logistic Regression delivered outstanding classification performance with 98.25% test accuracy, an F1-score of 0.9825, and an exceptional ROC-AUC of 0.9954. "
        "On high-dimensional normalized clinical data, hyperplanes separate benign and malignant tissue instances with high margin fidelity.\n"
        "• Random Forest Classifier demonstrated superior ensemble robustness with 94.74% accuracy and a near-perfect ROC-AUC of 0.9947, showing robust threshold flexibility.\n"
        "• K-Nearest Neighbors achieved 95.61% accuracy and an F1-score of 0.9560, demonstrating the effectiveness of local manifold similarity in standardized feature space.\n"
        "• Single Decision Tree achieved 92.11% accuracy, lagging due to discrete axis-aligned boundaries that are suboptimal for continuous spherical clusters."
    )

    # Embed Classification Figures
    fig_comp_c = os.path.join("visualizations", "classification_model_comparison.png")
    if os.path.exists(fig_comp_c):
        doc.add_picture(fig_comp_c, width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 3: Comparative Evaluation of Classification Models (Accuracy, Precision, Recall, F1-Score)")
        run_cap.font.italic = True
        run_cap.font.size = Pt(9.5)

    fig_cm_c = os.path.join("visualizations", "classification_confusion_matrices.png")
    if os.path.exists(fig_cm_c):
        doc.add_picture(fig_cm_c, width=Inches(6.0))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 4: Confusion Matrices Across All Four Classification Models")
        run_cap.font.italic = True
        run_cap.font.size = Pt(9.5)

    fig_roc_c = os.path.join("visualizations", "classification_roc_curves.png")
    if os.path.exists(fig_roc_c):
        doc.add_picture(fig_roc_c, width=Inches(5.5))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 5: Receiver Operating Characteristic (ROC) Curves with AUC Scores")
        run_cap.font.italic = True
        run_cap.font.size = Pt(9.5)

    # -------------------------------------------------------------------------
    # 6. COMPARATIVE DISCUSSION & PRACTICAL INSIGHTS
    # -------------------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    run_h1 = h1.add_run("6. Technical Insights & Comparative Analysis")
    run_h1.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "1. The Bias-Variance Trade-Off in Action:\n"
        "   Single Decision Trees demonstrated prominent variance: they rapidly achieve >99% training accuracy but fail to maintain that "
        "   margin on unseen test data. Random Forest directly remedies this pathology through feature subspace randomization and ensemble averaging, "
        "   compressing generalization gap while maintaining strong non-linear expressiveness.\n\n"
        "2. The Critical Necessity of Feature Standardization:\n"
        "   Both K-Nearest Neighbors and Regularized Linear / Logistic Regression depend mathematically on distance calculations (L2 norms) "
        "   and gradient descent trajectories. Without StandardScaler normalization, features with wider nominal ranges (e.g., cell area vs smoothness) "
        "   would disproportionately dominate loss gradients and distance calculations.\n\n"
        "3. Interpreting Model Selection for Production Deployments:\n"
        "   For clinical diagnostic workflows where false negatives carry critical risk (malignant cases missed), Logistic Regression and Random Forest "
        "   are the recommended models. Both offer probability calibration (predict_proba) that allows operational teams to tune decision thresholds "
        "   to enforce maximum recall."
    )

    # -------------------------------------------------------------------------
    # 7. CONCLUSION
    # -------------------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    run_h1 = h1.add_run("7. Conclusion & Deliverable Summary")
    run_h1.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "In completion of the Week 2 Task for the AI Pioneers Internship, all objectives were systematically achieved:\n"
        "1. Implemented supervised learning algorithms: Linear Regression, Logistic Regression, Decision Trees, Random Forest, and K-Nearest Neighbors.\n"
        "2. Executed end-to-end model training, holdout prediction, and multi-metric performance evaluation.\n"
        "3. Produced rigorous comparative benchmarks and visual analytics contrasting regression (R², RMSE, MAE) and classification (Accuracy, Precision, Recall, F1, ROC-AUC) models.\n"
        "All code, modular scripts, Jupyter notebooks, datasets, and generated visualizations have been version-controlled for peer review and submission."
    )

    output_path = "Week_2_Supervised_Machine_Learning_Report.docx"
    doc.save(output_path)
    print(f"Report successfully saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_report()
