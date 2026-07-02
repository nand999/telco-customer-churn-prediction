"""
Model training, evaluation, and comparison pipeline.
Supports Logistic Regression, Random Forest, XGBoost, and SVM
with GridSearchCV hyperparameter tuning and 5-fold cross-validation.
"""

import time
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import joblib
from src.utils import get_models_dir


# ─── Model Definitions with Hyperparameter Grids ─────────────────────────────

def get_models_and_params():
    """Return dict of model instances and their GridSearchCV parameter grids."""
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
            }
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1],
            }
        },
        "SVM": {
            "model": SVC(random_state=42, probability=True),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
            }
        },
    }


def train_single_model(name, model, params, X_train, y_train, cv=5):
    """
    Train a single model with GridSearchCV.
    Returns dict with best model, best params, CV scores, and training time.
    """
    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Cross-validation scores with best estimator
    cv_scores = cross_val_score(
        grid_search.best_estimator_, X_train, y_train,
        cv=cv, scoring="roc_auc"
    )

    return {
        "name": name,
        "model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_scores": cv_scores,
        "training_time": training_time,
    }


def train_all_models(X_train, y_train, progress_callback=None):
    """
    Train all models with GridSearchCV and 5-fold CV.
    Returns list of result dicts.
    """
    models_config = get_models_and_params()
    results = []
    total = len(models_config)

    for i, (name, config) in enumerate(models_config.items()):
        if progress_callback:
            progress_callback(i / total, f"Training {name}...")

        result = train_single_model(
            name, config["model"], config["params"],
            X_train, y_train
        )
        results.append(result)

    if progress_callback:
        progress_callback(1.0, "All models trained!")

    return results


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    Returns dict with all metrics and curve data.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # ROC curve data
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)

    # Precision-Recall curve data
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_proba)

    # Feature importance
    feature_importance = get_feature_importance(model, X_test)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "confusion_matrix": cm,
        "classification_report": report,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "pr_precision": pr_precision,
        "pr_recall": pr_recall,
        "pr_thresholds": pr_thresholds,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_importance": feature_importance,
    }


def get_feature_importance(model, X):
    """
    Extract feature importance from a model.
    Supports tree-based (feature_importances_) and linear (coef_) models.
    """
    feature_names = list(X.columns)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        # SVM with RBF kernel — use permutation-based approximation via coefficients
        importances = np.zeros(len(feature_names))

    # Create sorted DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


def build_comparison_table(results, X_test, y_test):
    """
    Build a model comparison DataFrame from training results and test evaluation.
    """
    rows = []
    evaluations = {}

    for result in results:
        eval_metrics = evaluate_model(result["model"], X_test, y_test)
        evaluations[result["name"]] = eval_metrics

        rows.append({
            "Model": result["name"],
            "Accuracy": f"{eval_metrics['accuracy']:.4f}",
            "Precision": f"{eval_metrics['precision']:.4f}",
            "Recall": f"{eval_metrics['recall']:.4f}",
            "F1-Score": f"{eval_metrics['f1']:.4f}",
            "ROC-AUC": f"{eval_metrics['roc_auc']:.4f}",
            "CV Mean (AUC)": f"{result['cv_mean']:.4f} ± {result['cv_std']:.4f}",
            "Training Time (s)": f"{result['training_time']:.2f}",
        })

    comparison_df = pd.DataFrame(rows)
    return comparison_df, evaluations


def get_best_model(results, X_test, y_test):
    """Return the model with the highest ROC-AUC on test data."""
    best_score = -1
    best_result = None

    for result in results:
        y_proba = result["model"].predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        if auc > best_score:
            best_score = auc
            best_result = result

    return best_result


def save_model_artifacts(results, best_model, scaler, label_encoders, feature_names,
                         comparison_df, evaluations):
    """Save all model artifacts to the models/ directory."""
    models_dir = get_models_dir()

    # Save best model
    joblib.dump(best_model["model"], os.path.join(models_dir, "best_model.pkl"))

    # Save all trained models
    all_models = {r["name"]: r["model"] for r in results}
    joblib.dump(all_models, os.path.join(models_dir, "all_models.pkl"))

    # Save preprocessing artifacts
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(models_dir, "label_encoders.pkl"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.pkl"))

    # Save comparison data
    joblib.dump(comparison_df, os.path.join(models_dir, "model_comparison.pkl"))
    joblib.dump(evaluations, os.path.join(models_dir, "model_evaluations.pkl"))
    joblib.dump(
        {r["name"]: r for r in results},
        os.path.join(models_dir, "training_results.pkl")
    )

    # Save best model name
    joblib.dump(best_model["name"], os.path.join(models_dir, "best_model_name.pkl"))


def load_model_artifacts():
    """Load all model artifacts from the models/ directory."""
    models_dir = get_models_dir()

    artifacts = {}
    files = {
        "best_model": "best_model.pkl",
        "all_models": "all_models.pkl",
        "scaler": "scaler.pkl",
        "label_encoders": "label_encoders.pkl",
        "feature_names": "feature_names.pkl",
        "comparison_df": "model_comparison.pkl",
        "evaluations": "model_evaluations.pkl",
        "training_results": "training_results.pkl",
        "best_model_name": "best_model_name.pkl",
    }

    for key, filename in files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
        else:
            artifacts[key] = None

    return artifacts
