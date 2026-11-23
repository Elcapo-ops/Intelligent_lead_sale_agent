"""SHAP-based explainability for lead conversion predictions.

Provides top-3 contributing factors (positive and negative) for each lead
prediction using actual SHAP values from the fitted model pipeline.
"""

from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from src.data.feature_engineering import (
    LeadFeatureEngineer,
    build_preprocessing_pipeline,
    get_feature_names,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "leads_dataset.csv")



class LeadExplainer:
    """Wraps a fitted pipeline and provides SHAP explanations per lead."""

    def _safe_float(self, val) -> float:
        """Convert a potentially array-like value to a Python float."""
        arr = np.asarray(val)
        if arr.ndim == 0:
            return float(arr)
        return float(arr.flat[0])

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "random_forest.joblib")
            if not os.path.exists(model_path):
                model_path = os.path.join(MODEL_DIR, "logistic_regression.joblib")

        self.model = joblib.load(model_path)
        self.engineer = self.model.named_steps["feature_engineer"]
        self.preprocessor = self.model.named_steps["preprocessor"]
        self.classifier = self.model.named_steps["model"]

        # Build a background dataset for SHAP
        df = pd.read_csv(DATA_PATH)
        X = df.drop(columns=["Converted"])
        X_eng = self.engineer.transform(X)
        self.preprocessor.fit(X_eng)
        X_processed = self.preprocessor.transform(X_eng)

        # Use a sample of processed data as SHAP background
        self.background = shap.sample(X_processed, min(100, len(X_processed)))

        # Get ordered feature names
        self.feature_names = get_feature_names(self.preprocessor, [])

        # Initialize SHAP explainer based on model type
        if hasattr(self.classifier, "coef_"):
            self.explainer = shap.LinearExplainer(self.classifier, self.background)
        else:
            self.explainer = shap.TreeExplainer(self.classifier, self.background)

    def explain(self, lead_dict: dict[str, Any], top_n: int = 3) -> dict[str, Any]:
        """
        Generate SHAP explanation for a single lead.

        Returns:
            dict with keys:
                - probability: float
                - base_value: float (average model output)
                - top_factors: list of dicts with feature, contribution, direction
        """
        import pandas as pd

        df = pd.DataFrame([lead_dict])
        X_eng = self.engineer.transform(df)
        X_processed = self.preprocessor.transform(X_eng)

        # SHAP values for this single instance
        shap_values = self.explainer.shap_values(X_processed)

        # For binary classification shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Flatten to 1D
        shap_values = np.array(shap_values).ravel()

        # Base value
        base_value = self._safe_float(self.explainer.expected_value)

        # Pair features with contributions
        contributions = []
        for i, (name, val) in enumerate(zip(self.feature_names, shap_values)):
            contributions.append({
                "feature": name,
                "contribution": float(val),
                "direction": "positive" if val > 0 else "negative",
            })

        # Sort by absolute contribution descending
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        top_factors = contributions[:top_n]

        # Predict probability
        proba = float(self.model.predict_proba(df)[:, 1][0])

        return {
            "probability": round(proba, 4),
            "base_value": round(base_value, 4),
            "top_factors": top_factors,
        }

    def explain_batch(self, df: pd.DataFrame, top_n: int = 3) -> list[dict[str, Any]]:
        """Generate explanations for a batch of leads."""
        results = []
        for _, row in df.iterrows():
            lead_dict = row.to_dict()
            exp = self.explain(lead_dict, top_n=top_n)
            results.append(exp)
        return results


def format_factors_for_display(top_factors: list[dict]) -> list[str]:
    """Convert top factors into human-readable explanation strings."""
    lines = []
    for f in top_factors:
        direction_word = "increases" if f["direction"] == "positive" else "decreases"
        lines.append(
            f"{f['feature']}: {direction_word} conversion likelihood "
            f"(impact: {abs(f['contribution']):.3f})"
        )
    return lines


if __name__ == "__main__":
    explainer = LeadExplainer()
    sample_lead = {
        "Lead Source": "Referral",
        "Total Visits": 6,
        "Time Spent on Website": 25.5,
        "Page Views Per Visit": 4.2,
        "Last Activity": "Demo Requested",
        "City or Region": "Bangalore",
        "Occupation or Company Size": "Software Engineer",
        "Domain": "Real Estate",
    }
    result = explainer.explain(sample_lead)
    print(json.dumps(result, indent=2))
    print("\nDisplay format:")
    for line in format_factors_for_display(result["top_factors"]):
        print("-", line)