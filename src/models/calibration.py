"""Probability calibration and lead tier classification."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_DIR = "models/trained"


class _CalibratedPipeline:
    """Wrapper that combines a base pipeline with an isotonic calibrator."""

    def __init__(self, base, calibrator):
        self.base = base
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])

HOT_THRESHOLD = 0.70
WARM_THRESHOLD = 0.35


@dataclass
class LeadTier:
    tier: str
    probability: float
    action: str


def tier_and_action(probability: float) -> LeadTier:
    if probability >= HOT_THRESHOLD:
        return LeadTier(
            tier="Hot",
            probability=probability,
            action="Immediate follow-up: Call within 2 hours and assign to senior sales rep.",
        )
    if probability >= WARM_THRESHOLD:
        return LeadTier(
            tier="Warm",
            probability=probability,
            action="Follow-up within 24 hours: Send personalized email and schedule call.",
        )
    return LeadTier(
        tier="Cold",
        probability=probability,
        action="Nurture: Add to drip email campaign and re-engage in 7-14 days.",
    )


def calibrate_and_score(best_model_name: str = "logistic_regression") -> dict:
    pipeline_path = os.path.join(MODEL_DIR, f"{best_model_name}.joblib")
    base_pipeline = joblib.load(pipeline_path)

    df = pd.read_csv("data/raw/leads_dataset.csv")
    X = df.drop(columns=["Converted"])
    y = df["Converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    val_raw_proba = base_pipeline.predict_proba(X_val)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_raw_proba, y_val)

    calibrated = _CalibratedPipeline(base_pipeline, iso)

    calibrated_path = os.path.join(MODEL_DIR, f"{best_model_name}_calibrated.joblib")
    joblib.dump(calibrated, calibrated_path)

    test_proba_raw = base_pipeline.predict_proba(X_test)[:, 1]
    test_proba_cal = calibrated.predict_proba(X_test)[:, 1]

    summary = {
        "base_model": best_model_name,
        "calibrated_path": calibrated_path,
        "calibration_method": "isotonic",
        "test_roc_auc_raw": float(roc_auc_score(y_test, test_proba_raw)),
        "test_roc_auc_calibrated": float(roc_auc_score(y_test, test_proba_cal)),
        "test_brier_raw": float(brier_score_loss(y_test, test_proba_raw)),
        "test_brier_calibrated": float(brier_score_loss(y_test, test_proba_cal)),
        "thresholds": {
            "hot": HOT_THRESHOLD,
            "warm": WARM_THRESHOLD,
        },
    }

    with open(os.path.join(MODEL_DIR, "calibration_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def load_calibrated_model(model_name: str = "logistic_regression"):
    path = os.path.join(MODEL_DIR, f"{model_name}_calibrated.joblib")
    if not os.path.exists(path):
        path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    return joblib.load(path)


def predict_lead(lead_dict: dict[str, Any], model) -> dict:
    proba = float(model.predict_proba(pd.DataFrame([lead_dict]))[:, 1][0])
    tier_info = tier_and_action(proba)

    return {
        "probability": round(proba, 4),
        "probability_percent": round(proba * 100, 2),
        "tier": tier_info.tier,
        "action": tier_info.action,
    }


if __name__ == "__main__":
    calibrate_and_score()