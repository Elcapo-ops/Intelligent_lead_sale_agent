"""Streamlit frontend for Smart Lead Scoring & Conversion Predictor.

Pages:
1. Single Lead Simulator - predict one lead, show tier + top-3 SHAP factors + action
2. Batch Upload & Sales Queue - CSV upload, score all, rank, filter, export
3. Model Performance - compare trained models (ROC-AUC, PR-AUC, confusion matrix)
4. Business Analytics - funnel, source performance, tier distribution, model metrics
"""

from __future__ import annotations

import io
import json
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Ensure src is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# Color palette: 75% primary (slate), 15% secondary (sky), 10% accent (amber)
PRIMARY = "#1e293b"
SECONDARY = "#0ea5e9"
ACCENT = "#f59e0b"
BG_COLOR = "#f8fafc"


# ---------------------------------------------------------------------------
# Model options & API helpers
# ---------------------------------------------------------------------------
MODEL_OPTIONS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def api_get_models() -> list[dict] | None:
    """Call /models endpoint to list available models."""
    try:
        r = requests.get(f"{API_BASE}/models", timeout=10)
        if r.status_code == 200:
            return r.json().get("available_models", [])
        st.error(f"API error ({r.status_code}): {r.text}")
    except Exception as exc:
        st.error(f"Could not reach API at {API_BASE}: {exc}")
    return None


def api_get_model_performance() -> list[dict] | None:
    """Call /model-performance endpoint."""
    try:
        r = requests.get(f"{API_BASE}/model-performance", timeout=10)
        if r.status_code == 200:
            return r.json()
        st.error(f"API error ({r.status_code}): {r.text}")
    except Exception as exc:
        st.error(f"Could not reach API at {API_BASE}: {exc}")
    return None


def api_predict(lead: dict, model_key: str = "random_forest") -> dict | None:
    """Call /predict endpoint with model selection."""
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json=lead,
            params={"model_name": model_key},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"API error ({r.status_code}): {r.text}")
    except Exception as exc:
        st.error(f"Could not reach API at {API_BASE}: {exc}")
    return None


def api_batch(df: pd.DataFrame, model_key: str = "random_forest") -> dict | None:
    """Call /predict-batch endpoint with CSV upload and model selection."""
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        files = {"file": ("leads.csv", csv_buffer.getvalue(), "text/csv")}
        data = {"model_name": model_key}
        r = requests.post(f"{API_BASE}/predict-batch", files=files, data=data, timeout=60)
        if r.status_code == 200:
            return r.json()
        st.error(f"API error ({r.status_code}): {r.text}")
    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")
    return None


def api_run_test(model_key: str) -> dict | None:
    """Call /run-test endpoint to evaluate a model on demand."""
    try:
        r = requests.post(
            f"{API_BASE}/run-test",
            json={"model_name": model_key},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"API error ({r.status_code}): {r.text}")
    except Exception as exc:
        st.error(f"Test run failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Lead Scoring & Conversion Predictor",
    layout="wide",
)

st.title("Smart Lead Scoring & Conversion Predictor")
st.markdown(
    "Predict lead conversion probability, understand driving factors with SHAP, "
    "and prioritize follow-ups across Real Estate, EdTech, Solar, and B2B Services."
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def tier_color(tier: str) -> str:
    return {
        "Hot": ACCENT,
        "Warm": SECONDARY,
        "Cold": PRIMARY,
    }.get(tier, "#6b7280")


# ---------------------------------------------------------------------------
# Sidebar navigation (no model selector here)
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Single Lead Simulator",
        "Batch Upload & Sales Queue",
        "Model Performance",
        "Business Analytics",
    ],
)


st.sidebar.divider()
st.sidebar.markdown(f"**API:** `{API_BASE}`")
if st.sidebar.button("Check API Health"):
    try:
        r = requests.get(f"{API_BASE}/health", timeout=10)
        st.sidebar.json(r.json())
    except Exception as exc:
        st.sidebar.error(f"API unreachable: {exc}")


# ---------------------------------------------------------------------------
# Page 1: Single Lead Simulator
# ---------------------------------------------------------------------------
if page == "Single Lead Simulator":
    st.header("Single Lead Simulator")
    st.markdown("Enter lead details to get conversion probability, tier, and explanation.")

    # Model selection moved here - specific to this page
    col_mod, col_spacer = st.columns([2, 1])
    with col_mod:
        model_choice = st.selectbox(
            "Model to use for prediction",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda k: MODEL_OPTIONS[k],
            index=list(MODEL_OPTIONS.keys()).index("random_forest"),
            help="Select which ML model to use for scoring leads.",
        )
        st.caption(f"Using: **{MODEL_OPTIONS[model_choice]}**")

    col1, col2 = st.columns(2)

    with col1:
        lead_source = st.selectbox(
            "Lead Source",
            ["Website", "Facebook Ads", "Google Ads", "LinkedIn",
             "Referral", "Email Campaign", "Cold Call", "Webinar", "Organic Search"],
        )
        total_visits = st.number_input("Total Visits", min_value=0, max_value=100, value=3)
        time_spent = st.number_input(
            "Time Spent on Website (minutes)", min_value=0.0, max_value=300.0, value=10.0
        )
        page_views = st.number_input(
            "Page Views Per Visit", min_value=0.0, max_value=50.0, value=3.0
        )

    with col2:
        last_activity = st.selectbox(
            "Last Activity",
            ["Website Visit", "Email Opened", "Form Submitted", "Call Connected",
             "Demo Requested", "Brochure Downloaded", "Webinar Attended",
             "Pricing Page Visited", "Chat Session", "None"],
        )
        city = st.selectbox(
            "City or Region",
            ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
             "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
             "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
             "London", "Manchester", "Birmingham", "Sydney", "Melbourne"],
        )
        occupation = st.selectbox(
            "Occupation or Company Size",
            ["Software Engineer", "Doctor", "Teacher", "Accountant",
             "Marketing Manager", "Sales Executive", "Entrepreneur", "Student",
             "Retired", "Freelancer", "HR Manager", "Operations Manager",
             "1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"],
        )
        domain = st.selectbox("Domain", ["Real Estate", "EdTech", "Solar", "B2B Services"])

    if st.button("Predict Conversion", type="primary"):
        lead = {
            "Lead Source": lead_source,
            "Total Visits": int(total_visits),
            "Time Spent on Website": float(time_spent),
            "Page Views Per Visit": float(page_views),
            "Last Activity": last_activity,
            "City or Region": city,
            "Occupation or Company Size": occupation,
            "Domain": domain,
        }

        result = api_predict(lead, model_key=model_choice)
        if result:
            st.divider()
            st.subheader("Prediction Result")

            prob = result["probability"]
            tier = result["tier"]
            color = tier_color(tier)

            col_prob, col_tier = st.columns([2, 1])
            with col_prob:
                st.metric("Conversion Probability", f"{prob*100:.2f}%")
                st.progress(prob)
            with col_tier:
                st.markdown(
                    f"<div style='padding:1rem; background-color:{color}22; "
                    f"border-left:5px solid {color}; border-radius:5px;'>"
                    f"<h3 style='color:{color}; margin:0;'>{tier} Lead</h3>"
                    f"<p style='margin:0.5rem 0 0 0;'>{result['action']}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.divider()
            st.subheader("Top 3 Contributing Factors (SHAP)")
            st.markdown(
                "These are the actual SHAP contributions for this specific lead. "
                "Positive factors increase conversion likelihood; negative factors decrease it."
            )

            for i, (factor, explanation) in enumerate(zip(
                result["top_factors"], result["explanation"]
            ), start=1):
                direction = factor["direction"]
                marker = "[+]" if direction == "positive" else "[-]"
                st.write(
                    f"{i}. {marker} **{factor['feature']}** - "
                    f"contribution: {factor['contribution']:+.3f} "
                    f"({direction})"
                )
                st.caption(explanation)


# ---------------------------------------------------------------------------
# Page 2: Batch Upload & Sales Queue
# ---------------------------------------------------------------------------
elif page == "Batch Upload & Sales Queue":
    st.header("Batch Upload & Sales Queue")
    st.markdown(
        "Upload a CSV of leads to score all records, rank by probability, "
        "filter by tier, and export a prioritized call list."
    )
    st.warning(
        "The CSV must contain the following columns: Lead Source, Total Visits, "
        "Time Spent on Website, Page Views Per Visit, Last Activity, "
        "City or Region, Occupation or Company Size, Domain, Converted - "
        "otherwise the API will reject the upload."
    )

    # Model selection for batch
    col_mod, col_spacer = st.columns([2, 1])
    with col_mod:
        batch_model_choice = st.selectbox(
            "Model to use for batch scoring",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda k: MODEL_OPTIONS[k],
            index=list(MODEL_OPTIONS.keys()).index("random_forest"),
            key="batch_model_selector",
            help="Select which ML model to use for scoring all leads.",
        )
        st.caption(f"Using: **{MODEL_OPTIONS[batch_model_choice]}**")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write(f"Uploaded **{len(df)}** leads.")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Score All Leads", type="primary"):
                result = api_batch(df, model_key=batch_model_choice)
                if result:
                    preds = result["predictions"]
                    pred_df = pd.DataFrame(preds)

                    # Expand top factors into columns
                    def _top_feature(lst, idx):
                        try:
                            return lst[idx]["feature"]
                        except Exception:
                            return ""

                    pred_df["top_factor_1"] = pred_df["top_factors"].apply(
                        lambda x: _top_feature(x, 0)
                    )
                    pred_df["top_factor_2"] = pred_df["top_factors"].apply(
                        lambda x: _top_feature(x, 1)
                    )
                    pred_df["top_factor_3"] = pred_df["top_factors"].apply(
                        lambda x: _top_feature(x, 2)
                    )

                    st.session_state["batch_results"] = pred_df
                    st.session_state["batch_total"] = result["total_leads"]

        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")

    if "batch_results" in st.session_state:
        pred_df = st.session_state["batch_results"]
        total = st.session_state.get("batch_total", len(pred_df))

        st.divider()
        st.subheader("Scored Results")
        st.metric("Total Leads Scored", total)

        # Tier distribution
        col_tier1, col_tier2, col_tier3 = st.columns(3)
        tier_counts = pred_df["tier"].value_counts()
        with col_tier1:
            st.metric("Hot", int(tier_counts.get("Hot", 0)))
        with col_tier2:
            st.metric("Warm", int(tier_counts.get("Warm", 0)))
        with col_tier3:
            st.metric("Cold", int(tier_counts.get("Cold", 0)))

        # Tier filter
        st.divider()
        st.subheader("Filter & Rank")
        tier_filter = st.multiselect(
            "Filter by Tier",
            options=["Hot", "Warm", "Cold"],
            default=["Hot", "Warm", "Cold"],
        )

        filtered = pred_df[pred_df["tier"].isin(tier_filter)] if tier_filter else pred_df
        filtered = filtered.sort_values("probability", ascending=False).reset_index(drop=True)

        st.write(f"Showing **{len(filtered)}** leads (ranked by probability)")

        display_cols = [
            "row_index", "probability", "probability_percent", "tier",
            "top_factor_1", "top_factor_2", "top_factor_3", "action",
        ]
        st.dataframe(
            filtered[display_cols],
            use_container_width=True,
            column_config={
                "probability": st.column_config.ProbabilityColumn(
                    "Probability", format="%.2f"
                ),
                "probability_percent": st.column_config.NumberColumn(
                    "Probability %", format="%.2f%%"
                ),
            },
        )

        # Export CSV
        st.divider()
        st.subheader("Export Prioritized Call List")
        export_df = filtered.copy()
        export_df["rank"] = range(1, len(export_df) + 1)
        export_csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download Prioritized Call List (CSV)",
            data=export_csv,
            file_name="prioritized_call_list.csv",
            mime="text/csv",
            type="primary",
        )


# ---------------------------------------------------------------------------
# Page 3: Business Analytics
# ---------------------------------------------------------------------------
elif page == "Business Analytics":
    st.header("Business Analytics")
    st.markdown(
        "Real metrics from the trained model and dataset. No fabricated numbers."
    )

    # Load dataset for analytics
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "leads_dataset.csv"))
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        df = pd.DataFrame()

    if not df.empty:
        # Load model metrics
        metrics_path = os.path.join(PROJECT_ROOT, "models", "trained", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
        else:
            metrics = {}

        st.subheader("Model Evaluation")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        with col_m2:
            st.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")
        with col_m3:
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        with col_m4:
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        st.metric("F1-Score", f"{metrics.get('f1', 0):.4f}")
        st.metric("Best Model", metrics.get("best_model", "N/A"))

        # Confusion matrix
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Predicted 0", "Predicted 1"],
                    y=["Actual 0", "Actual 1"],
                    colorscale=[
                        [0, PRIMARY],
                        [0.5, SECONDARY],
                        [1, ACCENT],
                    ],
                    showscale=True,
                )
            )
            fig_cm.update_layout(
                title="Confusion Matrix (Test Set)",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        st.divider()
        st.subheader("Conversion Funnel")
        total_leads = len(df)
        converted = int(df["Converted"].sum())
        funnel_data = {
            "Stage": ["Total Leads", "Converted"],
            "Count": [total_leads, converted],
        }
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_data["Stage"],
            x=funnel_data["Count"],
            textinfo="value+percent initial",
            marker=dict(color=[SECONDARY, ACCENT]),
        ))
        fig_funnel.update_layout(
            title="Lead Conversion Funnel",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

        st.divider()
        st.subheader("Lead Source Performance")
        source_perf = (
            df.groupby("Lead Source")["Converted"]
            .agg(["count", "sum", "mean"])
            .reset_index()
            .rename(columns={"count": "Total Leads", "sum": "Converted", "mean": "Conversion Rate"})
            .sort_values("Conversion Rate", ascending=False)
        )
        source_perf["Conversion Rate"] = source_perf["Conversion Rate"] * 100

        fig_source = px.bar(
            source_perf,
            x="Lead Source",
            y="Conversion Rate",
            color="Conversion Rate",
            color_continuous_scale=[
                [0, PRIMARY],
                [0.5, SECONDARY],
                [1, ACCENT],
            ],
            title="Conversion Rate by Lead Source",
        )
        fig_source.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_source, use_container_width=True)
        st.dataframe(source_perf, use_container_width=True)

        st.divider()
        st.subheader("Lead Tier Distribution (Sample Predictions)")
        # Score a sample to show tier distribution
        if "batch_results" in st.session_state:
            tier_dist = st.session_state["batch_results"]["tier"].value_counts()
            fig_tier = px.pie(
                values=tier_dist.values,
                names=tier_dist.index,
                color=tier_dist.index,
                color_discrete_map={"Hot": ACCENT, "Warm": SECONDARY, "Cold": PRIMARY},
                title="Lead Tier Distribution",
            )
            fig_tier.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_tier, use_container_width=True)
        else:
            st.info("Run a batch upload to see predicted tier distribution.")

        st.divider()
        st.subheader("Domain Distribution")
        domain_dist = df["Domain"].value_counts()
        fig_domain = px.pie(
            values=domain_dist.values,
            names=domain_dist.index,
            title="Leads by Domain",
            color_discrete_sequence=[PRIMARY, SECONDARY, ACCENT],
        )
        fig_domain.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_domain, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 4: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.header("Model Performance & Testing")
    st.markdown(
        "Select a model and run a test to evaluate it on the held-out test set. "
        "Results include validation and test metrics plus a confusion matrix."
    )

    # Model selector for testing
    st.subheader("Select Model to Test")
    test_model_key = st.selectbox(
        "Choose model for testing",
        options=list(MODEL_OPTIONS.keys()),
        format_func=lambda k: MODEL_OPTIONS[k],
        key="test_model_selector",
    )

    # Run Test button - two columns for better layout
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        run_test = st.button(
            "Run Test",
            type="primary",
            use_container_width=True,
        )
    with col_btn2:
        clear_results = st.button(
            "Clear",
            use_container_width=True,
        )

    if clear_results:
        if "test_result" in st.session_state:
            del st.session_state["test_result"]
        if "test_model" in st.session_state:
            del st.session_state["test_model"]
        st.rerun()

    if run_test:
        with st.spinner(f"Training and evaluating {MODEL_OPTIONS[test_model_key]}... This may take a moment."):
            result = api_run_test(test_model_key)
            if result:
                st.session_state["test_result"] = result
                st.session_state["test_model"] = test_model_key
                st.success(f"Test completed for {MODEL_OPTIONS[test_model_key]}!")

    # Display test results if available
    if "test_result" in st.session_state:
        result = st.session_state["test_result"]
        model_name = result.get("model", "N/A").replace("_", " ").title()

        st.divider()
        st.subheader(f"Test Results - {model_name}")

        # Metrics row
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.metric("Val ROC-AUC", f"{result.get('val_roc_auc', 0):.4f}")
        with col_m2:
            st.metric("Val PR-AUC", f"{result.get('val_pr_auc', 0):.4f}")
        with col_m3:
            st.metric("Test ROC-AUC", f"{result.get('test_roc_auc', 0):.4f}")
        with col_m4:
            st.metric("Test PR-AUC", f"{result.get('test_pr_auc', 0):.4f}")
        with col_m5:
            st.metric("Test F1", f"{result.get('test_f1', 0):.4f}")

        # Additional metrics
        col_m6, col_m7, col_m8 = st.columns(3)
        with col_m6:
            st.metric("Test Precision", f"{result.get('test_precision', 0):.4f}")
        with col_m7:
            st.metric("Test Recall", f"{result.get('test_recall', 0):.4f}")
        with col_m8:
            st.metric("Confusion Matrix", "See below")

        # Metrics table
        st.divider()
        st.subheader("Full Metrics Table")
        metrics_df = pd.DataFrame(
            [
                {"Metric": "Validation ROC-AUC", "Value": f"{result.get('val_roc_auc', 0):.4f}"},
                {"Metric": "Validation PR-AUC", "Value": f"{result.get('val_pr_auc', 0):.4f}"},
                {"Metric": "Test ROC-AUC", "Value": f"{result.get('test_roc_auc', 0):.4f}"},
                {"Metric": "Test PR-AUC", "Value": f"{result.get('test_pr_auc', 0):.4f}"},
                {"Metric": "Test Precision", "Value": f"{result.get('test_precision', 0):.4f}"},
                {"Metric": "Test Recall", "Value": f"{result.get('test_recall', 0):.4f}"},
                {"Metric": "Test F1-Score", "Value": f"{result.get('test_f1', 0):.4f}"},
            ]
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Confusion Matrix
        st.divider()
        st.subheader("Confusion Matrix (Test Set)")
        cm = result.get("confusion_matrix")
        if cm:
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Predicted Negative", "Predicted Positive"],
                    y=["Actual Negative", "Actual Positive"],
                    colorscale=[
                        [0, PRIMARY],
                        [0.5, SECONDARY],
                        [1, ACCENT],
                    ],
                    showscale=True,
                    text=cm,
                    texttemplate="%{text}",
                )
            )
            fig_cm.update_layout(
                title=f"Confusion Matrix - {model_name}",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # Show matrix as text too
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"],
            )
            st.dataframe(cm_df, use_container_width=True)
        else:
            st.warning("Confusion matrix not available.")

        # Model descriptions
        st.divider()
        st.subheader("Model Description")
        model_descriptions = {
            "logistic_regression": (
                "**Logistic Regression** - A linear probabilistic model. Fast, interpretable, "
                "and works well when features are roughly linearly separable. "
                "Baseline model for lead scoring."
            ),
            "random_forest": (
                "**Random Forest** - An ensemble of decision trees. Robust to outliers, "
                "handles non-linear relationships and feature interactions well. "
                "Generally the strongest performer on this dataset."
            ),
            "xgboost": (
                "**XGBoost** - Gradient-boosted decision trees with regularization. "
                "Often state-of-the-art for tabular data. Handles class imbalance "
                "via `scale_pos_weight` and captures complex patterns."
            ),
        }
        st.markdown(model_descriptions.get(test_model_key, "No description available."))

        # Recommendation based on result
        st.divider()
        st.subheader("Interpretation")
        test_auc = result.get("test_roc_auc", 0)
        if test_auc >= 0.75:
            st.success(f"Excellent performance! Test ROC-AUC of {test_auc:.4f} indicates strong predictive power.")
        elif test_auc >= 0.65:
            st.info(f"Good performance. Test ROC-AUC of {test_auc:.4f} is acceptable for lead scoring.")
        elif test_auc >= 0.55:
            st.warning(f"Fair performance. Test ROC-AUC of {test_auc:.4f} - consider feature engineering or hyperparameter tuning.")
        else:
            st.error(f"Poor performance. Test ROC-AUC of {test_auc:.4f} - model needs significant improvement.")

    else:
        st.info(
            "Select a model from the dropdown above and click **Run Test** "
            "to evaluate it on the test set."
        )

    st.divider()
    st.info(
        "Tip: Model selection on the Single Lead Simulator and Batch Upload pages "
        "controls which model is used for predictions on those pages."
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Smart Lead Scoring & Conversion Predictor - Academic Project. "
    "Built with FastAPI, Streamlit, scikit-learn, XGBoost, and SHAP."
)