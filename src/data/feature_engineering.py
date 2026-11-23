"""Custom feature-engineering transformers for the lead-scoring pipeline.

Implements three domain-specific engineered features:
1. Engagement Intensity Score
2. Recency / Activity Flag
3. High-Intent Source Indicator

Designed to be used inside a scikit-learn Pipeline so that no data leakage
occurs (all statistics are computed inside fit/transform on training data only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constant feature lists (must match the dataset columns)
# ---------------------------------------------------------------------------
ENGINEERED_FEATURES = [
    "Engagement Intensity Score",
    "Recency / Activity Flag",
    "High-Intent Source Indicator",
]

NUMERIC_FEATURES = [
    "Total Visits",
    "Time Spent on Website",
    "Page Views Per Visit",
] + ENGINEERED_FEATURES

CATEGORICAL_FEATURES = [
    "Lead Source",
    "Last Activity",
    "City or Region",
    "Occupation or Company Size",
    "Domain",
]

TARGET = "Converted"

# Sources considered high-intent for the indicator feature
HIGH_INTENT_SOURCES = {"Referral", "LinkedIn", "Webinar", "Pricing Page Visited"}

# Activities considered high-intent / recent for the recency flag
HIGH_INTENT_ACTIVITIES = {
    "Demo Requested",
    "Pricing Page Visited",
    "Form Submitted",
    "Call Connected",
    "Webinar Attended",
}


class LeadFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates domain-specific engineered features from raw lead attributes.

    Features:
    1. Engagement Intensity Score = Total Visits * Time Spent * Page Views Per Visit
       (log-scaled to reduce skew)
    2. Recency / Activity Flag = 1 if last activity is high-intent, else 0
    3. High-Intent Source Indicator = 1 if lead source is high-intent, else 0

    All transformations are row-wise and require no fitting, so they are safe
    inside a scikit-learn Pipeline without data leakage.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y=None) -> "LeadFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1. Engagement Intensity Score
        visits = X["Total Visits"].astype(float)
        time_spent = X["Time Spent on Website"].astype(float)
        page_views = X["Page Views Per Visit"].astype(float)

        engagement = visits * time_spent * page_views
        # Log1p transform to compress the long right tail
        X["Engagement Intensity Score"] = np.log1p(engagement)

        # 2. Recency / Activity Flag
        X["Recency / Activity Flag"] = X["Last Activity"].isin(HIGH_INTENT_ACTIVITIES).astype(int)

        # 3. High-Intent Source Indicator
        X["High-Intent Source Indicator"] = X["Lead Source"].isin(HIGH_INTENT_SOURCES).astype(int)

        return X


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that:

    - Applies the custom feature engineer first (outside ColumnTransformer,
      handled in the full Pipeline below)
    - Scales numeric features
    - One-hot encodes categorical features with handling for unseen categories
    - Imputes missing values

    Returns a ColumnTransformer ready to be wrapped in a Pipeline.
    """
    # Numeric branch: impute median then standard scale
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical branch: impute most_frequent then one-hot encode
    # handle_unknown="ignore" prevents errors on unseen categories at inference
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_full_pipeline(model) -> Pipeline:
    """
    Assemble the complete modeling Pipeline:

    LeadFeatureEngineer -> ColumnTransformer (preprocessing) -> model

    Wrapping everything in a single Pipeline guarantees that preprocessing
    statistics are learned only from training folds, preventing leakage.
    """
    preprocessor = build_preprocessing_pipeline()

    full_pipeline = Pipeline(
        steps=[
            ("feature_engineer", LeadFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return full_pipeline


def get_feature_names(preprocessor: ColumnTransformer, engineered: list[str]) -> list[str]:
    """
    Extract human-readable feature names from a fitted ColumnTransformer.

    Returns the ordered list of feature names that the model sees after
    preprocessing (engineered + scaled numeric + one-hot categorical).
    """
    # NUMERIC_FEATURES now includes ENGINEERED_FEATURES, so use it directly
    feature_names = list(NUMERIC_FEATURES)

    # Categorical features get one-hot expanded names
    cat_pipeline = preprocessor.named_transformers_["cat"]
    onehot = cat_pipeline.named_steps["onehot"]
    cat_names = onehot.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names.extend(cat_names)

    return feature_names