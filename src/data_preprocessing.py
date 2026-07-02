"""
Data preprocessing pipeline for Telco Customer Churn dataset.
Handles cleaning, encoding, scaling, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from src.utils import get_models_dir


def clean_data(df):
    """
    Clean the raw dataset:
    - Convert TotalCharges to numeric (handle spaces → NaN → median fill)
    - Drop customerID
    - Return cleaned DataFrame
    """
    df = df.copy()

    # TotalCharges has some blank strings — convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with median
    median_total = df["TotalCharges"].median()
    df["TotalCharges"].fillna(median_total, inplace=True)

    # Drop customerID — not a predictive feature
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    return df


def encode_features(df, fit=True, label_encoders=None):
    """
    Encode categorical features:
    - Binary columns: LabelEncoder (Yes/No, Male/Female)
    - Multi-category columns: One-hot encoding
    Returns (encoded_df, label_encoders_dict)
    """
    df = df.copy()

    # ── Binary columns ──
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    # Filter to columns actually present in df
    binary_cols = [c for c in binary_cols if c in df.columns]

    if label_encoders is None:
        label_encoders = {}

    if fit:
        for col in binary_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    else:
        for col in binary_cols:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])

    # ── Multi-category columns — One-Hot Encode ──
    multi_cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    multi_cat_cols = [c for c in multi_cat_cols if c in df.columns]

    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # Ensure all column values are numeric
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    return df, label_encoders


def scale_features(X_train, X_test, fit=True, scaler=None):
    """
    Apply StandardScaler to numeric features.
    Returns (X_train_scaled, X_test_scaled, scaler)
    """
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    # Filter to actually present columns
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]

    if fit or scaler is None:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    else:
        X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])

    if X_test is not None and len(X_test) > 0:
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, scaler


def preprocess_pipeline(df):
    """
    Full preprocessing pipeline:
    1. Clean data
    2. Encode features
    3. Split into X, y
    Returns (X, y, label_encoders, feature_names)
    """
    df_clean = clean_data(df)
    df_encoded, label_encoders = encode_features(df_clean, fit=True)

    # Split target
    y = df_encoded["Churn"]
    X = df_encoded.drop("Churn", axis=1)

    feature_names = list(X.columns)

    return X, y, label_encoders, feature_names


def save_preprocessing_artifacts(label_encoders, scaler, feature_names):
    """Save preprocessing artifacts to models/ directory."""
    models_dir = get_models_dir()
    joblib.dump(label_encoders, os.path.join(models_dir, "label_encoders.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.pkl"))


def load_preprocessing_artifacts():
    """Load preprocessing artifacts from models/ directory."""
    models_dir = get_models_dir()
    label_encoders = joblib.load(os.path.join(models_dir, "label_encoders.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))
    return label_encoders, scaler, feature_names


def preprocess_single_customer(input_data, label_encoders, scaler, feature_names):
    """
    Preprocess a single customer input dict for prediction.
    - input_data: dict with raw feature values
    - Returns: DataFrame row ready for model.predict()
    """
    df = pd.DataFrame([input_data])

    # Encode binary columns
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in df.columns and col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])

    # One-hot encode multi-category columns
    multi_cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    multi_cat_cols = [c for c in multi_cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # Convert booleans
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    # Scale numeric
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Align columns with training feature set
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    return df


def preprocess_batch(batch_df, label_encoders, scaler, feature_names):
    """
    Preprocess a batch DataFrame for prediction.
    Returns aligned DataFrame ready for model.predict().
    """
    df = batch_df.copy()

    # Clean
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
    if "Churn" in df.columns:
        df.drop("Churn", axis=1, inplace=True)

    # Encode binary
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in df.columns and col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])

    # One-hot encode
    multi_cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    multi_cat_cols = [c for c in multi_cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    # Scale
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Align columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]
    return df
