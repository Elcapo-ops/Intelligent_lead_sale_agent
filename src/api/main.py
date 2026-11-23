"""FastAPI backend for Smart Lead Scoring & Conversion Predictor.

Endpoints:
- GET  /models              : list available models
- POST /predict             : score a single lead (model_name optional)
- POST /predict-batch       : upload CSV, score all rows (model_name optional)
- GET  /model-performance   : test-set metrics for all models
- GET  /health              : health check
"""

from __future__ import annotations

import io
import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
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
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from src.data.feature_engineering import (
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_full_pipeline,
    get_feature_names,
    LeadFeatureEngineer,
)
from src.explainability.shap_explainer import LeadExplainer, format_factors_for_display
from src.models.calibration import predict_lead

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "leads_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained")

AVAILABLE_MODELS = ["logistic_regression", "random_forest", "xgboost"]

app = FastAPI(
    title="Smart Lead Scoring & Conversion Predictor",
    version="1.0.0",
    description="Predict lead conversion probability with SHAP explanations and tiered actions.",
)

# ---------------------------------------------------------------------------
# Model cache (loaded on demand)
# ---------------------------------------------------------------------------
_model_cache: dict[str, Any] = {}
_explainer_cache: dict[str, Any] = {}


def _register_legacy_pickle_classes() -> None:
    """Make calibrated artifacts pickled from __main__ loadable.

    Older `logistic_regression_calibrated.joblib` files reference
    ``__main__._CalibratedPipeline`` because they were created by running
    calibration.py as a script. Alias the class so those pickles resolve.
    """
    import __main__
    from src.models.calibration import _CalibratedPipeline

    if not hasattr(__main__, "_CalibratedPipeline"):
        __main__._CalibratedPipeline = _CalibratedPipeline


def _load_pipeline(path: str):
    """Load a joblib pipeline, tolerating legacy __main__-pickled classes."""
    try:
        return joblib.load(path)
    except AttributeError:
        _register_legacy_pickle_classes()
        return joblib.load(path)


def _get_model(model_name: str):
    """Load and cache a model pipeline by name."""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    if model_name not in _model_cache:
        calibrated = os.path.join(MODEL_DIR, f"{model_name}_calibrated.joblib")
        path = calibrated if os.path.exists(calibrated) else os.path.join(MODEL_DIR, f"{model_name}.joblib")
        try:
            _model_cache[model_name] = _load_pipeline(path)
        except Exception:
            if path == calibrated:
                # Broken calibrated artifact: fall back to the base pipeline.
                _model_cache[model_name] = _load_pipeline(
                    os.path.join(MODEL_DIR, f"{model_name}.joblib")
                )
            else:
                raise
    return _model_cache[model_name]


def _get_explainer(model_name: str):
    """Load and cache a SHAP explainer by name."""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    if model_name not in _explainer_cache:
        path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        if not os.path.exists(path):
            path = os.path.join(MODEL_DIR, "random_forest.joblib")
        _explainer_cache[model_name] = LeadExplainer(model_path=path)
    return _explainer_cache[model_name]


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class LeadInput(BaseModel):
    Lead_Source: str = Field(..., alias="Lead Source")
    Total_Visits: int = Field(..., ge=0, alias="Total Visits")
    Time_Spent_on_Website: float = Field(..., ge=0, alias="Time Spent on Website")
    Page_Views_Per_Visit: float = Field(..., ge=0, alias="Page Views Per Visit")
    Last_Activity: str = Field(..., alias="Last Activity")
    City_or_Region: str = Field(..., alias="City or Region")
    Occupation_or_Company_Size: str = Field(..., alias="Occupation or Company Size")
    Domain: str = Field(...)

    class Config:
        populate_by_name = True

    @validator("Total_Visits")
    def _visits(cls, v):
        if v < 0:
            raise ValueError("Total Visits must be non-negative")
        return v


class PredictionResponse(BaseModel):
    model: str
    probability: float
    probability_percent: float
    tier: str
    action: str
    top_factors: list[dict[str, Any]]
    explanation: list[str]


class BatchPredictionItem(BaseModel):
    row_index: int
    probability: float
    probability_percent: float
    tier: str
    action: str
    top_factors: list[dict[str, Any]]
    explanation: list[str]


class ModelPerformance(BaseModel):
    model: str
    val_roc_auc: float | None = None
    test_roc_auc: float | None = None
    test_pr_auc: float | None = None
    test_precision: float | None = None
    test_recall: float | None = None
    test_f1: float | None = None
    confusion_matrix: list[list[int]] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _lead_dict_from_input(lead: LeadInput) -> dict[str, Any]:
    return {
        "Lead Source": lead.Lead_Source,
        "Total Visits": lead.Total_Visits,
        "Time Spent on Website": lead.Time_Spent_on_Website,
        "Page Views Per Visit": lead.Page_Views_Per_Visit,
        "Last Activity": lead.Last_Activity,
        "City or Region": lead.City_or_Region,
        "Occupation or Company Size": lead.Occupation_or_Company_Size,
        "Domain": lead.Domain,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_model_cache.keys())}


@app.get("/models")
def list_models():
    """List available models with their display names."""
    display = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    return {
        "available_models": [
            {"key": m, "name": display.get(m, m)} for m in AVAILABLE_MODELS
        ],
    }



@app.get("/model-performance", response_model=list[ModelPerformance])
def model_performance():
    """Return stored test-set metrics for all trained models."""
    comparison_path = os.path.join(MODEL_DIR, "model_comparison.json")
    if os.path.exists(comparison_path):
        with open(comparison_path) as f:
            comparison = json.load(f)
        return [
            ModelPerformance(
                model=m,
                val_roc_auc=comparison.get(m, {}).get("val_roc_auc"),
                test_roc_auc=comparison.get(m, {}).get("test_roc_auc"),
                test_pr_auc=comparison.get(m, {}).get("test_pr_auc"),
                test_precision=comparison.get(m, {}).get("test_precision"),
                test_recall=comparison.get(m, {}).get("test_recall"),
                test_f1=comparison.get(m, {}).get("test_f1"),
                confusion_matrix=comparison.get(m, {}).get("confusion_matrix"),
            )
            for m in AVAILABLE_MODELS
        ]

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    results = []
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        best = metrics.get("best_model", "random_forest")
        for m in AVAILABLE_MODELS:
            results.append(
                ModelPerformance(
                    model=m,
                    test_roc_auc=metrics.get("roc_auc") if m == best else None,
                    test_pr_auc=metrics.get("pr_auc") if m == best else None,
                    test_precision=metrics.get("precision") if m == best else None,
                    test_recall=metrics.get("recall") if m == best else None,
                    test_f1=metrics.get("f1") if m == best else None,
                    confusion_matrix=metrics.get("confusion_matrix") if m == best else None,
                )
            )
    else:
        for m in AVAILABLE_MODELS:
            results.append(ModelPerformance(model=m))
    return results


@app.post("/predict", response_model=PredictionResponse)
def predict(
    lead: LeadInput,
    model_name: str = Query(default="random_forest", description="Model to use"),
):
    model = _get_model(model_name)
    explainer = _get_explainer(model_name)

    lead_dict = _lead_dict_from_input(lead)

    # Probability + tier + action
    pred = predict_lead(lead_dict, model)

    # SHAP explanation
    try:
        exp = explainer.explain(lead_dict, top_n=3)
        top_factors = exp["top_factors"]
        explanation = format_factors_for_display(top_factors)
    except Exception as exc:
        top_factors = []
        explanation = [f"Explanation unavailable: {exc}"]

    return PredictionResponse(
        model=model_name,
        probability=pred["probability"],
        probability_percent=pred["probability_percent"],
        tier=pred["tier"],
        action=pred["action"],
        top_factors=top_factors,
        explanation=explanation,
    )


@app.post("/predict-batch")
async def predict_batch(
    file: UploadFile = File(...),
    model_name: str = Form(default="random_forest", description="Model to use"),
):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    model = _get_model(model_name)
    explainer = _get_explainer(model_name)

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}")

    required = {
        "Lead Source", "Total Visits", "Time Spent on Website",
        "Page Views Per Visit", "Last Activity", "City or Region",
        "Occupation or Company Size", "Domain",
    }
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing)}",
        )

    results = []
    for idx, row in df.iterrows():
        lead_dict = row.to_dict()
        try:
            pred = predict_lead(lead_dict, model)
            exp = explainer.explain(lead_dict, top_n=3)
            explanation = format_factors_for_display(exp["top_factors"])
            results.append(
                BatchPredictionItem(
                    row_index=int(idx),
                    probability=pred["probability"],
                    probability_percent=pred["probability_percent"],
                    tier=pred["tier"],
                    action=pred["action"],
                    top_factors=exp["top_factors"],
                    explanation=explanation,
                )
            )
        except Exception as exc:
            results.append(
                BatchPredictionItem(
                    row_index=int(idx),
                    probability=0.0,
                    probability_percent=0.0,
                    tier="Error",
                    action=str(exc),
                    top_factors=[],
                    explanation=[],
                )
            )

    # Sort by probability descending
    results.sort(key=lambda r: r.probability, reverse=True)

    # Build CSV output for export
    csv_rows = []
    for r in results:
        csv_rows.append(
            {
                "row_index": r.row_index,
                "probability": r.probability,
                "tier": r.tier,
                "action": r.action,
                "top_factor_1": r.top_factors[0]["feature"] if r.top_factors else "",
                "top_factor_2": r.top_factors[1]["feature"] if len(r.top_factors) > 1 else "",
                "top_factor_3": r.top_factors[2]["feature"] if len(r.top_factors) > 2 else "",
            }
        )
    csv_df = pd.DataFrame(csv_rows)
    csv_buffer = io.StringIO()
    csv_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return JSONResponse(
        content={
            "model": model_name,
            "total_leads": len(results),
            "predictions": [r.dict() for r in results],
        },
        headers={"X-CSV-Download": "available"},
    )


class RunTestRequest(BaseModel):
    model_name: str = "random_forest"


class RunTestResponse(BaseModel):
    model: str
    val_roc_auc: float | None = None
    val_pr_auc: float | None = None
    test_roc_auc: float | None = None
    test_pr_auc: float | None = None
    test_precision: float | None = None
    test_recall: float | None = None
    test_f1: float | None = None
    confusion_matrix: list[list[int]] | None = None


@app.post("/run-test", response_model=RunTestResponse)
def run_test(request: RunTestRequest):
    """Train and evaluate a selected model on demand, returning test-set metrics."""
    model_name = request.model_name
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    MODEL_DIR_LOCAL = MODEL_DIR
    os.makedirs(MODEL_DIR_LOCAL, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Train / validation / test split (same as training script)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    # Class weight for imbalance
    neg, pos = y_train.value_counts().sort_index()
    scale_pos_weight = neg / pos

    # Build the selected model
    if model_name == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    elif model_name == "xgboost":
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=42,
            n_jobs=-1, eval_metric="logloss"
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    # Build and fit full pipeline
    pipeline = build_full_pipeline(model)
    pipeline.fit(X_train, y_train)

    # Validation metrics
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    val_roc_auc = float(roc_auc_score(y_val, val_proba))
    val_pr_auc = float(average_precision_score(y_val, val_proba))

    # Test metrics
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    test_roc_auc = float(roc_auc_score(y_test, test_proba))
    test_pr_auc = float(average_precision_score(y_test, test_proba))
    test_precision = float(precision_score(y_test, test_pred))
    test_recall = float(recall_score(y_test, test_pred))
    test_f1 = float(f1_score(y_test, test_pred))
    cm = confusion_matrix(y_test, test_pred).tolist()

    # Save the trained pipeline for future predictions
    joblib.dump(pipeline, os.path.join(MODEL_DIR_LOCAL, f"{model_name}.joblib"))

    # Clear cache so next prediction uses the freshly trained model
    if model_name in _model_cache:
        del _model_cache[model_name]

    return RunTestResponse(
        model=model_name,
        val_roc_auc=val_roc_auc,
        val_pr_auc=val_pr_auc,
        test_roc_auc=test_roc_auc,
        test_pr_auc=test_pr_auc,
        test_precision=test_precision,
        test_recall=test_recall,
        test_f1=test_f1,
        confusion_matrix=cm,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
