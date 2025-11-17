# Dementia Risk Prediction Project (Team Adagard)

This repository contains the code and final report for a machine learning project to predict dementia risk based on non-medical, self-reported data. The project emphasizes rigorous data cleaning, feature engineering, and a comprehensive model tournament to identify the most robust predictive model.

## 🚀 How to Review This Project

This project includes a detailed report and a fully pre-run notebook. For a complete overview of our process, findings, and analysis, please start with the report:

* **📄 `Adagard_report.pdf`**: A comprehensive PDF report detailing our data exploration, feature engineering, model tournament, and final explainability insights.
* **📓 `Adagard_experiments_notebook.ipynb`**: The complete Python notebook. **This file includes all code and pre-run outputs** from our experiments, including all EDA plots and the final model leaderboard.

> **Note for Judges:** The full model tournament (Cells 13, 14, and 15) is computationally expensive and may fail on standard hardware due to RAM limitations. For your convenience, all cells have been pre-run, and their outputs (results tables, figures, and analyses) are saved directly in the notebook and summarized in the PDF report.

## 📊 Project Workflow

Our process flow was an iterative experiment designed to be robust, compliant with hackathon rules, and prevent data leakage:

1.  **Data Cleaning & Filtering:** We first defined an "allow-list" of 58 non-medical, self-reported features. A diagnostic check confirmed only 39 of these were present in the dataset. We also standardized NACC missing codes (e.g., 99, -4) to `np.nan`.
2.  **Leakage Prevention:** We critically identified and **removed the `INDEPEND` variable**. We determined this was a data leakage *symptom* (a patient's current independence level) rather than a *risk factor*.
3.  **Feature Engineering:** We hypothesized that cumulative health burden was a key predictor. We successfully created `FE_COMORBIDITY_SCORE` (sum of 18 health conditions) and `FE_SLEEP_PSYCH_SCORE` (sum of 9 health history items).
4.  **Feature Validation:** We used a `GridSearchCV`-tuned Random Forest to rank all 41 features. This validated our hypothesis, ranking our engineered `FE_COMORBIDITY_SCORE` as the **#5 most important feature**.
5.  **Preprocessing:** We built a robust `ColumnTransformer` pipeline to handle imputation (using `median` for numerics, `constant` for categoricals) and scaling to prevent data leakage.
6.  **Model Tournament:** We conducted a comprehensive 10-model tournament, including eight standard classifiers (LGBM, RF, XGB, etc.), a `StackingClassifier` meta-ensemble, and an MLP Neural Network.
7.  **Evaluation:** We used **ROC-AUC** as the primary metric due to the 70.5% vs. 29.5% class imbalance. We also dynamically tuned the **F1-Score** decision threshold for each model to provide a more practical measure of performance.

## 🏆 Final Model & Performance

Based on the 10-model tournament, the **Stacking_Ensemble** was the winning model, achieving the highest performance on the unseen test set.

* **Winning Model:** `Stacking_Ensemble` (Base: Extra Trees, Random Forest, XGBoost; Meta: LightGBM).
* **Test AUC Score:** `0.82