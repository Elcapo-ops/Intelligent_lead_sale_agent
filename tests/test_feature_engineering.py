"""Tests for feature engineering and preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    ENGINEERED_FEATURES,
    LeadFeatureEngineer,
    build_full_pipeline,
    build_preprocessing_pipeline,
    get_feature_names,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Lead Source": ["Website", "Referral", "Google Ads", "Website"],
            "Total Visits": [1, 5, 3, 10],
            "Time Spent on Website": [2.0, 25.0, 10.0, 45.0],
            "Page Views Per Visit": [1.5, 4.0, 2.5, 5.0],
            "Last Activity": [
                "Website Visit",
                "Demo Requested",
                "Email Opened",
                "Pricing Page Visited",
            ],
            "City or Region": ["Mumbai", "Bangalore", "Delhi", "Mumbai"],
            "Occupation or Company Size": [
                "Software Engineer",
                "Entrepreneur",
                "Doctor",
                "11-50",
            ],
            "Domain": ["Real Estate", "EdTech", "Solar", "B2B Services"],
        }
    )


def test_feature_engineer_creates_columns(sample_df):
    eng = LeadFeatureEngineer()
    eng.fit(sample_df)
    out = eng.transform(sample_df)

    for feat in ENGINEERED_FEATURES:
        assert feat in out.columns, f"Missing engineered feature: {feat}"

    # Engagement Intensity Score should be log1p(visits * time * page_views)
    expected = np.log1p(
        sample_df["Total Visits"].astype(float)
        * sample_df["Time Spent on Website"].astype(float)
        * sample_df["Page Views Per Visit"].astype(float)
    )
    np.testing.assert_allclose(out["Engagement Intensity Score"].values, expected.values)

    # Recency flag should be 1 for Demo Requested / Pricing Page Visited
    assert out.loc[out["Last Activity"] == "Demo Requested", "Recency / Activity Flag"].iloc[0] == 1
    assert out.loc[out["Last Activity"] == "Website Visit", "Recency / Activity Flag"].iloc[0] == 0

    # High-intent source: Referral is high-intent
    assert out.loc[out["Lead Source"] == "Referral", "High-Intent Source Indicator"].iloc[0] == 1
    assert out.loc[out["Lead Source"] == "Website", "High-Intent Source Indicator"].iloc[0] == 0


def test_preprocessing_pipeline_runs(sample_df):
    preprocessor = build_preprocessing_pipeline()
    eng = LeadFeatureEngineer()
    X_eng = eng.transform(sample_df)
    preprocessor.fit(X_eng)
    X_out = preprocessor.transform(X_eng)

    assert X_out.shape[0] == len(sample_df)
    assert X_out.shape[1] > 0
    assert not np.isnan(X_out).any(), "Preprocessed output should not contain NaNs"


def test_full_pipeline_fit_predict(sample_df):
    from sklearn.linear_model import LogisticRegression

    y = np.array([0, 1, 0, 1])
    pipeline = build_full_pipeline(LogisticRegression())
    pipeline.fit(sample_df, y)
    proba = pipeline.predict_proba(sample_df)[:, 1]
    assert proba.shape == (4,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_get_feature_names(sample_df):
    preprocessor = build_preprocessing_pipeline()
    eng = LeadFeatureEngineer()
    X_eng = eng.transform(sample_df)
    preprocessor.fit(X_eng)
    names = get_feature_names(preprocessor, ENGINEERED_FEATURES)

    for feat in ENGINEERED_FEATURES:
        assert feat in names
    assert len(names) == preprocessor.transform(X_eng).shape[1]


def test_missing_values_handled():
    df = pd.DataFrame(
        {
            "Lead Source": ["Website", None, "Referral"],
            "Total Visits": [1, None, 5],
            "Time Spent on Website": [2.0, 25.0, None],
            "Page Views Per Visit": [1.5, 4.0, 2.5],
            "Last Activity": ["Website Visit", "Demo Requested", None],
            "City or Region": ["Mumbai", "Bangalore", None],
            "Occupation or Company Size": ["Software Engineer", "Entrepreneur", None],
            "Domain": ["Real Estate", "EdTech", "Solar"],
        }
    )
    from sklearn.linear_model import LogisticRegression

    pipeline = build_full_pipeline(LogisticRegression())
    y = np.array([0, 1, 0])
    pipeline.fit(df, y)  # should not raise
    proba = pipeline.predict_proba(df)[:, 1]
    assert proba.shape == (3,)