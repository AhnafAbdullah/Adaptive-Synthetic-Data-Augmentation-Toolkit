# 🧬 Adaptive Synthetic Data Augmentation Toolkit

A closed-loop machine learning engine that automatically detects weak model cohorts and generates targeted synthetic data to boost performance.

## 🚀 Overview
The Adaptive Synthetic Data Augmentation Toolkit is an intelligent pipeline designed to solve the problem of imbalanced datasets and structural blind-spots in machine learning models. Built natively in Python and governed by a beautiful Streamlit dashboard, the toolkit will:
1. Train a baseline classification model on your raw dataset.
2. Identify specifically which data cohorts the model struggles to predict ("False Negatives", "Minority Class Misclassifications").
3. Automatically build localized `GaussianCopulaSynthesizers` (via the Synthetic Data Vault) mapped specifically to those failing cohorts.
4. Execute an Adaptive Augmentation Loop: it generates targeted synthetic samples for those specific weak spots, injects them back into your dataset, retrains the model, and loops continuously until evaluation metrics (like AUC or F1-score) stop improving.

## ✨ Features
- **Intelligent Cohort Targeting**: Doesn't just blindly oversample the whole dataset; it finds the *exact* subpopulations your model is failing on and targets them.
- **Fluid Streamlit Interface**: Clean, dark-mode native dashboard with real-time Metric grids, progress bars, and live AUC vs Iteration line charts.
- **Multiple Architectures**: Swap easily between Random Forest, Logistic Regression, or Decision Trees on the fly.
- **Lightning Fast Synthesis**: Automatically samples and caps synthesizer parameters ensuring your SDV augmentation completes in seconds rather than hours.
- **Early Stopping**: Smart Delta-AUC thresholding ensures the loop aborts and saves your best model when synthetic data begins to degrade realism.

## ⚙️ Installation
1. Clone the repository.
2. Ensure you have Python installed.
3. Install the specific dependencies:
```bash
pip install pandas numpy scikit-learn sdv streamlit
```

## 🎮 How to Run
Trigger the frontend by running the following command in your terminal:
```bash
streamlit run app.py
```
Open `http://localhost:8501`, upload your CSV dataset (up to 1GB supported!), select your Target Column, and hit **Initialize & Train Baseline**!
