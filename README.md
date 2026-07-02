# 🔮 Customer Churn Prediction Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-006CB4?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-powered interactive dashboard for predicting and analyzing customer churn in the telecommunications industry.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model Performance](#-model-performance) • [Tech Stack](#-tech-stack)

</div>

---

## 📋 Overview

This project implements a complete machine learning pipeline for **customer churn prediction** using the IBM Telco Customer Churn dataset. The interactive Streamlit dashboard provides end-to-end capabilities from exploratory data analysis to real-time prediction and business intelligence.

### 🎯 Problem Statement

Customer churn (customers leaving a service) is a critical business challenge in the telecom industry. Identifying at-risk customers early enables proactive retention strategies, reducing revenue loss and improving customer lifetime value.

### 📊 Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Records:** 7,043 customers
- **Features:** 21 attributes including demographics, account info, services, and charges
- **Target:** Binary classification (Churn: Yes/No)

---

## ✨ Features

### 📊 1. Exploratory Data Analysis (EDA)
- Interactive distribution plots for numeric features
- Correlation heatmap with feature-to-churn analysis
- Churn rate breakdown by categorical features
- Key business insights with actionable metrics

### 🤖 2. Model Training Pipeline
- **Data Preprocessing:** Missing value handling, label/one-hot encoding, StandardScaler
- **4 ML Models:** Logistic Regression, Random Forest, XGBoost, SVM
- **Hyperparameter Tuning:** GridSearchCV with optimized parameter grids
- **5-Fold Cross-Validation** for robust model evaluation
- **Model Comparison Table** with all key metrics
- **Automated Model Selection** based on ROC-AUC score

### 📈 3. Model Performance Analysis
- Confusion matrix with detailed breakdown
- ROC-AUC curves (multi-model overlay)
- Precision-Recall curves
- Feature importance chart (top 15 features)
- Classification report with per-class metrics
- Visual model comparison bar charts

### 🔮 4. Real-Time Prediction
- Interactive form for all customer features
- Churn probability gauge with visual indicator
- Confidence score display
- Risk level badge (Low / Medium / High / Critical)
- Key risk factor analysis
- Personalized retention recommendations

### 📁 5. Batch Prediction
- CSV file upload with format validation
- Data preview and quality check
- Bulk prediction with risk scoring
- Risk distribution visualization
- Downloadable results CSV

### 💡 6. Business Insights
- **Customer Segmentation:** Tenure-based segments (New, Growing, Mature, Loyal)
- **Risk Scoring:** Full customer base scoring with risk tier distribution
- **Revenue Impact:** Monthly/annual revenue at risk from high-risk customers
- **Churn Drivers:** Top 10 churn-driving feature combinations
- **Retention Recommendations:** Actionable strategies per risk tier

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nand999/telco-customer-churn-prediction.git
   cd telco-customer-churn-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser:** Navigate to `http://localhost:8501`

---

## 📖 Usage

### Quick Start Guide

1. **Explore Data** → Navigate to the **EDA** page to understand data distributions and churn patterns
2. **Train Models** → Go to **Model Training** to run the full ML pipeline with hyperparameter tuning
3. **Analyze Performance** → Visit **Model Performance** for detailed metrics and visualizations
4. **Make Predictions** → Use **Prediction** for single customer scoring or **Batch Prediction** for bulk CSV uploads
5. **Get Insights** → Check **Business Insights** for segmentation, risk scoring, and retention strategies

### Training Models

On the Model Training page:
- Adjust the test set size (default: 20%)
- Click **"Train All Models"**
- Wait for GridSearchCV to complete (typically 2-5 minutes)
- Review the comparison table to identify the best model
- Models are automatically saved to `models/` directory

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.80 | ~0.66 | ~0.54 | ~0.59 | ~0.84 |
| Random Forest | ~0.79 | ~0.65 | ~0.47 | ~0.55 | ~0.83 |
| **XGBoost** | **~0.81** | **~0.67** | **~0.53** | **~0.59** | **~0.85** |
| SVM | ~0.80 | ~0.66 | ~0.52 | ~0.58 | ~0.84 |

> **Note:** Actual results may vary slightly due to random state and data splitting. The table above shows approximate values.

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Frontend** | Streamlit, Plotly, HTML/CSS |
| **ML Framework** | Scikit-Learn, XGBoost |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Seaborn, Matplotlib |
| **Model Persistence** | Joblib (.pkl files) |
| **Design** | Custom CSS, Dark Theme, Glassmorphism |

---

## 📁 Project Structure

```
CUSTOMER-CHURN-PREDICTION-TELCO/
├── app.py                          # Main Streamlit entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data cleaning, encoding, scaling
│   ├── model_training.py           # Training, evaluation, comparison
│   └── utils.py                    # Helpers, CSS theme, UI components
├── pages/
│   ├── 1_📊_EDA.py                 # Exploratory Data Analysis
│   ├── 2_🤖_Model_Training.py     # Model Training Pipeline
│   ├── 3_📈_Model_Performance.py  # Performance Metrics & Curves
│   ├── 4_🔮_Prediction.py         # Real-Time Single Prediction
│   ├── 5_📁_Batch_Prediction.py   # Batch CSV Prediction
│   └── 6_💡_Business_Insights.py  # Segmentation & Recommendations
├── models/                         # Saved model artifacts (auto-generated)
│   ├── best_model.pkl
│   ├── all_models.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_names.pkl
│   └── model_comparison.pkl
└── .streamlit/
    └── config.toml                 # Dark theme configuration
```

---

## 🎨 Design

The dashboard features a **premium dark theme** with:
- 🎨 Purple-to-teal gradient accents (`#6C63FF` → `#00D4AA`)
- 🪟 Glassmorphism card effects with backdrop blur
- ✨ Micro-animations (fade-in, slide-in, hover effects)
- 📱 Responsive layout with grid-based metric cards
- 🏷️ Color-coded risk badges and insight boxes
- 🔤 Inter font family for modern typography

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

**Built using Streamlit & Scikit-Learn**

</div>
