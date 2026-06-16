# 📈 Gold Price Prediction using Machine Learning

Built a **regression pipeline** to predict **gold-correlated asset prices** using
**Lasso, Random Forest, and XGBoost** with hyperparameter tuning and financial
feature engineering.

---

## 📌 Problem Statement
Gold prices are influenced by multiple financial market indicators. This project
builds a machine learning pipeline to predict gold-correlated asset prices using
historical financial data, applying regression models with proper preprocessing
and hyperparameter tuning.

---

## 📂 Dataset
- **Source:** [Kaggle - Gold Price Data](https://www.kaggle.com/datasets/altruistdelhite04/gold-price-data)
- **Features:** SPX (S&P 500), GLD (Gold ETF), USO (Oil ETF), SLV (Silver ETF), EUR/USD
- **Target Variable:** EUR/USD (gold-correlated asset price)
- **Type:** Financial Time-Series Data

---

## ⚙️ Pipeline Overview

1. **Exploratory Data Analysis (EDA)**
   - Correlation heatmap to identify feature relationships
   - Distribution plots and skewness analysis
   - Rolling mean (window=20) for trend visualization

2. **Data Preprocessing**
   - Dropped low-relevance `SLV` column based on correlation
   - Applied **square root transformation** to fix skewed `USO` column
   - **Outlier capping** using 5th–95th percentile winsorization
   - **StandardScaler** normalization on train/test splits
   - **SimpleImputer** (mean strategy) for missing values from rolling window

3. **Feature Engineering**
   - Date set as index for time-series analysis
   - Polynomial features (degree=2) applied for Lasso pipeline

4. **Model Training & Hyperparameter Tuning**
   - **GridSearchCV** used for Lasso and Random Forest tuning
   - 80/20 train-test split
   - Models evaluated using **R² Score**

---

## 📊 Feature Importance
Random Forest feature importances revealed the strongest predictors of
gold-correlated asset prices among SPX, GLD, USO, and EUR/USD.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/gold-price-prediction.git
cd gold-price-prediction
pip install -r requirements.txt
jupyter notebook Gold_Price_Prediction.ipynb
```

> Download `gold_price_data.csv` from Kaggle and place it in the project
> root directory.

---
