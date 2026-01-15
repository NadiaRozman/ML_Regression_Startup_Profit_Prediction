# 🤖 Machine Learning Regression: Startup Profit Prediction

**Quick Snapshot**
- Dataset: 50 startups, 3 numerical features
- Task: Profit prediction (regression)
- Best model: Decision Tree (R² = 0.9766)
- Key insight: R&D spend drives ~92% of profitability


## 📌 Project Overview

This project applies and compares multiple **machine learning regression algorithms** to predict **startup profitability** based on investment allocation across core business functions.

Inspired by my [Drug Classification ML project](https://github.com/NadiaRozman/ML_Classification_Drug_Prediction), this notebook focuses on:

- Systematic **model comparison**
- Clear **code justification**
- Translating model outputs into **business insights**

> **Objective:** Predict startup profit and identify which machine learning models perform best for continuous business forecasting tasks.

---

## 🎯 Background

1. Predict **Profit** based on: R&D Spend, Administration Spend, Marketing Spend
2. Compare multiple regression algorithms using consistent evaluation metrics
3. Identify the **best-performing model**
4. Interpret **feature importance** to support data-driven decision-making

---

## 🗂️ Dataset Overview

**Source:** Educational startup dataset (50 records)

**Features:**

* **R&D Spend** (numeric): Research and Development investment
* **Administration** (numeric): Administrative costs
* **Marketing Spend** (numeric): Marketing budget
* ~~**State** (categorical): Location of startup~~ *(Excluded from this analysis)*

**Target Variable:**

* **Profit** (numeric): Annual profit (continuous)

**Problem Type:** Multivariate regression 

> **⚠️ Disclaimer**
This dataset is synthetic and educational, intended purely for supervised learning practice and portfolio demonstration.

---

## 🛠️ Tools I Used

**Programming & Libraries:**

* **Python** – Analysis and model implementation
* **NumPy** – Numerical computations and array operations
* **Pandas** – Data manipulation and analysis
* **Matplotlib** – Data visualization and model performance plots
* **Scikit-learn** – Machine learning models, preprocessing, metrics and model evaluation

**Development Environment:**

* **Jupyter Notebook** – Interactive development and analysis
* **Google Colab** – Optional cloud-based notebook environment

---

## 🚀 Getting Started

### Prerequisites

* Python 3.11 or higher
* Jupyter Notebook or Google Colab
* Conda (recommended) or pip

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/NadiaRozman/ML_Regression_Startup_Profit_Prediction.git
   cd ML_Regression_Startup_Profit_Prediction
   ```

2. **Set up the environment**

   **Option 1: Using Conda**

   ```bash
   conda env create -f environment.yml
   conda activate ml_startup_regression
   ```

   **Option 2: Using pip**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

4. **Open and run the notebook**

   * Navigate to `notebook/Startup_Profit_Regression_ML.ipynb`
   * Ensure `startup_profit_dataset.csv` is in the correct directory
   * Run all cells to reproduce the analysis

---

## 🔬 Machine Learning Models Implemented

This project implements and compares **4 regression algorithms**:

1. **Decision Tree Regression** – Non-parametric, interpretable, captures non-linear relationships, scale-invariant  
2. **Polynomial Regression (Degree 2)** – Extends linear regression with polynomial features, captures feature interactions  
3. **Random Forest Regression** – Ensemble of decision trees, reduces overfitting, provides feature importance, scale-invariant  
4. **Support Vector Regression (SVR)** – Uses RBF kernel, effective for non-linear patterns, requires feature scaling for optimal performance  

---

## 📈 Project Workflow & Analysis

### 1. **Data Loading & Exploration**

* Load dataset using Pandas
* Explore data structure with `.head()`, `.info()`, and `.describe()`
* Dataset contains 50 startups with 5 columns, no missing values

### 2. **Feature Selection & Target Definition**

* Selected first 3 columns (R&D, Administration, Marketing) as features
* Excluded State (categorical)
* Target variable: Profit (continuous)

### 3. **Train-Test Split**

* Split ratio: 80% training, 20% testing
* Random state = 0 for reproducibility
* Training set: 40 samples; Test set: 10 samples

### 4. Feature Scaling

* **StandardScaler** applied to features (X) and target (y)  
* Critical for SVR; optional for tree-based models  
* Fitted on training data only to prevent leakage  
* Separate scalers for features and target allow inverse transformation

### 5. **Model Training & Evaluation**

* Each model trained on scaled training data
* Predictions inversely transformed to original scale
* Helper function `evaluate_model()` ensures consistent evaluation
* **Metrics**: R², RMSE, MAE

### 6. **Test Set Performance Summary**

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Decision Tree | 0.9766 | $5,470.16 | $4,480.29 |
| Random Forest | 0.9671 | $6,482.31 | $5,268.49 |
| Polynomial | 0.9202 | $10,103.37 | $8,720.93 |
| SVR | 0.8734 | $12,725.29 | $10,272.88 |

> Decision Tree slightly outperformed Random Forest on this small dataset, likely due to overfitting, which ensemble methods mitigate on larger datasets.

---

### Key Findings

- **Best-performing model:** Decision Tree (R² = 0.9766)  
- **Most stable model:** Random Forest (R² = 0.9671)  

**Feature Importance (Random Forest):**
- R&D Spend: 91.64%  
- Marketing Spend: 7.76%  
- Administration: 0.61%  

**Business Implications:**
- Prioritize R&D (~92% impact)  
- Marketing contributes (~8%)  
- Administration has minimal effect (~0.6%)  

---

## 📚 What I Learned

- Implemented tree-based, polynomial, and kernel-based regression techniques  
- Learned the critical role of feature scaling for SVR  
- Applied R², RMSE, and MAE for robust evaluation  
- Understood ensemble methods’ ability to reduce overfitting  
- Extracted feature importance for actionable business insights  
- Translated ML outputs into clear recommendations for decision-makers  

---

## 🔮 Future Enhancements

Potential improvements for this project:

1. Feature engineering: include State, interaction terms, higher-degree polynomials
2. Advanced models: XGBoost, LightGBM, Neural Networks
3. Hyperparameter tuning: GridSearchCV, RandomizedSearchCV, Bayesian optimization
4. Model interpretability: SHAP, partial dependence, residual analysis
5. Deployment: Flask/FastAPI, Streamlit, Docker
6. Extended analysis: cross-validation, learning curves, prediction intervals, time-series validation

---
### ✨ Created by Nadia Rozman | January 2026

📂 **Project Structure**
```
ML_Regression_Startup_Profit_Prediction/
│
├── data/
│   └── startup_profit_dataset.csv
│
├── notebook/
│   └── Startup_Profit_Regression_ML.ipynb
│
├── requirements.txt
├── environment.yml
├── README.md
└── .gitignore
```

**🔗 Connect with me**
- GitHub: [@NadiaRozman](https://github.com/NadiaRozman)
- LinkedIn: [Nadia Rozman](https://www.linkedin.com/in/nadia-rozman-4b4887179/)

**⭐ If you found this project helpful, please consider giving it a star!**
