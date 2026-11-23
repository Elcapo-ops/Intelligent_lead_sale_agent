"""Train and compare lead-conversion models.

Trains Logistic Regression, Random Forest, and XGBoost using a shared
preprocessing pipeline. Handles class imbalance via class weights and
evaluates on a held-out test set (never used for model selection).

Outputs:
- models/trained/preprocessor.joblib  (fitted ColumnTransformer)
- models/trained/<model_name>.joblib  (full fitted pipelines)
- models/trained/metrics.json         (comparison metrics)
- models/trained/feature_names.json   (ordered feature names for SHAP)
"""

from __future__ import annotations

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.feature_engineering import (
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_full_pipeline,
    build_preprocessing_pipeline,
    get_feature_names,
    LeadFeatureEngineer,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RANDOM_SEED = 42
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "leads_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained")
TEST_SIZE = 0.2
VAL_SIZE = 0.25  # fraction of train used for validation (model selection)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def train_and_evaluate() -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    X = df.drop(columns=[TARGET, "Domain"])  # Domain retained in feature engineer via categorical branch
    y = df[TARGET]

    # Keep Domain for categorical encoding but it's not in X if we drop it...
    # Instead keep all original raw features and let pipeline handle them.
    X = df.drop(columns=[TARGET])

    # Train / validation / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_train
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Train conversion rate: {y_train.mean():.4f}")
    print(f"Test conversion rate:  {y_test.mean():.4f}")

    # Class weight for imbalance
    neg, pos = y_train.value_counts().sort_index()
    scale_pos_weight = neg / pos

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED,
            n_jobs=-1, eval_metric="logloss"
        ),
    }

    results = {}
    best_model_name = None
    best_val_auc = -1.0

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        pipeline = build_full_pipeline(model)
        pipeline.fit(X_train, y_train)

        # Validation predictions (for model selection)
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        val_ap = average_precision_score(y_val, val_proba)

        print(f"  Val ROC-AUC: {val_auc:.4f} | Val PR-AUC: {val_ap:.4f}")

        results[name] = {
            "val_roc_auc": float(val_auc),
            "val_pr_auc": float(val_ap),
        }

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_name = name

        # Save pipeline
        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name}.joblib"))

    print(f"\nBest model (by Val ROC-AUC): {best_model_name} ({best_val_auc:.4f})")

    # Evaluate best model on held-out TEST set only
    best_pipeline = joblib.load(os.path.join(MODEL_DIR, f"{best_model_name}.joblib"))
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "pr_auc": float(average_precision_score(y_test, test_proba)),
        "precision": float(precision_score(y_test, test_pred)),
        "recall": float(recall_score(y_test, test_pred)),
        "f1": float(f1_score(y_test, test_pred)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "best_model": best_model_name,
    }

    print("\n=== Test Set Metrics ===")
    for k, v in test_metrics.items():
        if k != "confusion_matrix" and isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
    print("  Confusion Matrix:")
    print(np.array(test_metrics["confusion_matrix"]))

    # Save preprocessor + feature names for inference / SHAP
    fitted_engineer = best_pipeline.named_steps["feature_engineer"]
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]

    # Transform train to get feature names
    X_train_eng = fitted_engineer.transform(X_train)
    fitted_preprocessor.fit(X_train_eng)

    feature_names = get_feature_names(fitted_preprocessor, ENGINEERED_FEATURES)
    joblib.dump(fitted_preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nSaved artifacts to {MODEL_DIR}/")
    return test_metrics


class LeadFeatureEngineerPlaceholder:
    """Tiny placeholder so mypy / readers see intent; real engineer imported above."""
    def transform(self, X):
        from src.data.feature_engineering import LeadFeatureEngineer
        return LeadFeatureEngineer().transform(X)


if __name__ == "__main__":
    train_and_evaluate()