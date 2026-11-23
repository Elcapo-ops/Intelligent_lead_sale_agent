# Smart Lead Scoring & Sales Prioritization Engine

An end-to-end machine learning system that scores inbound leads by conversion
probability, explains *why* each lead scored the way it did, and turns that into a
ranked daily call list for the sales team.

Built for high-volume digital-ad lead funnels — Real Estate, EdTech, Solar and B2B
services — where hundreds of leads arrive per day and reps cannot call everyone.

**Stack:** Python · scikit-learn · XGBoost · SHAP · FastAPI · Streamlit

---

## Table of Contents

1. [Business Problem & ROI](#1-business-problem--roi)
2. [What the System Does](#2-what-the-system-does)
3. [Dataset](#3-dataset)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Training, Imbalance Handling & Evaluation](#5-model-training-imbalance-handling--evaluation)
6. [Model Comparison Table](#6-model-comparison-table)
7. [Probability Calibration & Lead Tiers](#7-probability-calibration--lead-tiers)
8. [Explainability (SHAP)](#8-explainability-shap)
9. [FastAPI Backend](#9-fastapi-backend)
10. [Streamlit Application](#10-streamlit-application)
11. [Project Structure](#11-project-structure)
12. [Setup & Run Instructions](#12-setup--run-instructions)
13. [Deployment](#13-deployment)
14. [Testing](#14-testing)
15. [Limitations & Next Steps](#15-limitations--next-steps)

---

## 1. Business Problem & ROI

**The problem.** Paid digital campaigns generate volume that human follow-up cannot
absorb. Reps work the list top-to-bottom (newest first, or whatever the CRM export
order is), so they spend their day on leads with no buying intent while genuinely
hot buyers — the ones who requested a demo an hour ago — go cold in the queue.

Two costs follow directly:

- **Wasted calling capacity.** Every call to a lead with < 5% conversion probability
  is a call not made to a lead at 70%.
- **Slow response on high-intent leads.** Conversion drops sharply with response
  time; a hot lead sitting in a queue overnight is a lost deal.

**What the system changes.** Every lead gets a calibrated conversion probability and
a Hot / Warm / Cold tier within the same day it arrives. Reps start at the top of a
ranked queue with the top three reasons the model scored the lead, plus a
next-best-action, so the call opening is already informed.

### Illustrative ROI

These are modelling assumptions, not measured client results — replace them with
your own funnel numbers before quoting an impact figure.

| Assumption | Value |
|---|---|
| Leads per day per client | 400 |
| Historical conversion rate | ~37% |
| Rep capacity | 80 dials/day/rep |
| Reps currently assigned | 5 (400 dials — full coverage) |

At 80 dials/reps-day and 400 leads/day, a 5-rep team *is* the bottleneck: coverage
only works if segments are prioritised. Concentrating the first 240 dials of the day
on Hot + Warm leads (using the tier model on this dataset's distribution) puts every
rep's effort on leads with ≥ 35% modelled probability instead of spreading it evenly.
The measurable gain to validate in a pilot is **conversion rate per dial** and
**time-to-first-touch on Hot leads** — both are already tracked by any dialer CRM.

> Deliberately no dollar figure is claimed here. The ROI number that survives
> scrutiny is the one produced by an A/B pilot on your own dialer data.

---

## 2. What the System Does

| Capability | Where it lives |
|---|---|
| Conversion probability per lead | `POST /predict` (single), `POST /predict-batch` (CSV) |
| Calibrated probability (isotonic) | `src/models/calibration.py` |
| Hot / Warm / Cold tier + Next-Best-Action | `src/models/calibration.py` |
| Top-3 SHAP reasons per lead | `src/explainability/shap_explainer.py` |
| Ranked, filterable, exportable call queue | Streamlit → *Batch Upload & Sales Queue* |
| Conversion funnel, source performance, model charts | Streamlit → *Business Analytics* |
| Live single-lead what-if testing | Streamlit → *Single Lead Simulator* |

---

## 3. Dataset

`data/raw/leads_dataset.csv`

| Property | Value |
|---|---|
| Records | 10,000 |
| Columns | 9 (8 features + target) |
| Positive class rate | 37.42% (`Converted = 1`) |
| Missing values | 431 in `Last Activity` (handled by the pipeline's imputer) |

| Column | Type | Description |
|---|---|---|
| `Lead Source` | categorical | Website, Organic Search, Google Ads, Facebook, Referral, LinkedIn, Webinar, Pricing Page Visited |
| `Total Visits` | numeric | Sessions on site |
| `Time Spent on Website` | numeric | Minutes of total session time |
| `Page Views Per Visit` | numeric | Pages / session |
| `Last Activity` | categorical | Form Submitted, Demo Requested, Call Connected, Email Opened, Webinar Attended, Page Visited, Unreachable |
| `City or Region` | categorical | Metro/region of the lead |
| `Occupation or Company Size` | categorical | Job role or SMB/Mid-Market/Enterprise segment |
| `Domain` | categorical | Real Estate, EdTech, Solar, B2B Services |
| `Converted` | binary | Target: 1 = became a customer |

The dataset is produced by `src/data/generate_dataset.py`, which models a realistic
digital-ad funnel (source-dependent intent, engagement-driven conversion, missing
activity values, class imbalance). It is synthetic by construction and shipped with
the repo so the pipeline is reproducible offline; swapping in a real CRM export with
the same columns requires no code changes.

```bash
python -m src.data.generate_dataset   # regenerates data/raw/leads_dataset.csv
```

---

## 4. Feature Engineering

All engineering happens **inside** the scikit-learn `Pipeline`
(`LeadFeatureEngineer` → `ColumnTransformer` → estimator), so every statistic —
medians, modes, scaler means and one-hot vocabularies — is fitted on training folds
only. No leakage between train, validation and test.

### Engineered features (3)

| Feature | Definition | Business meaning |
|---|---|---|
| **Engagement Intensity Score** | `log1p(Total Visits × Time Spent × Page Views Per Visit)` | How deeply the lead consumed the site; log-compressed to tame the long tail |
| **Recency / Activity Flag** | 1 if `Last Activity` ∈ {Demo Requested, Pricing Page Visited, Form Submitted, Call Connected, Webinar Attended} | Bottom-of-funnel intent, not idle browsing |
| **High-Intent Source Indicator** | 1 if `Lead Source` ∈ {Referral, LinkedIn, Webinar, Pricing Page Visited} | Channel quality: referral/social-professional vs. cold display |

### Preprocessing

```
numeric:      SimpleImputer(median) → StandardScaler
categorical:  SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
```

`handle_unknown="ignore"` means an unseen city or source at inference time scores
without crashing — important for a production lead flow.

---

## 5. Model Training, Imbalance Handling & Evaluation

Three classifiers are trained on an identical pipeline:

1. **Logistic Regression** — baseline, interpretable, `class_weight="balanced"`
2. **Random Forest** — 200 trees, max depth 10, `class_weight="balanced"`
3. **XGBoost** — 200 rounds, depth 6, `lr = 0.05`, `scale_pos_weight = neg/pos`

**Imbalance handling.** Class weighting (`class_weight="balanced"` for sklearn
models, `scale_pos_weight` for XGBoost) rather than resampling, so no synthetic rows
leak into validation.

**Split protocol** (stratified, `random_state=42`):

- 60% train · 20% validation · 20% held-out test
- Model selection uses **validation ROC-AUC only**
- The test set is scored once, after selection — it never influences the choice

**Metrics.** ROC-AUC, PR-AUC (average precision), precision, recall, F1 and a
confusion matrix. Accuracy is deliberately not used as a selection metric: with a
roughly 37/63 class split and a tuned business threshold, accuracy hides exactly the
errors that cost money.

Reproduce the full run:

```bash
python -m src.models.train_models
```

Artifacts written to `models/trained/`: `<model>.joblib`, `preprocessor.joblib`,
`feature_names.json`, `metrics.json`, `model_comparison.json`, plus
`logistic_regression_calibrated.joblib` and `calibration_summary.json` from the
calibration step.

---

## 6. Model Comparison Table

Evaluated on the same stratified split (6,000 train / 2,000 validation / 2,000 test),
threshold 0.50 for the point metrics. Stored in `models/trained/model_comparison.json`
and served by `GET /model-performance`.

| Model | Val ROC-AUC | Val PR-AUC | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** (selected) | **0.5890** | **0.4487** | **0.6269** | **0.4892** | **0.4602** | **0.5802** | **0.5133** |
| Random Forest | 0.5851 | 0.4459 | 0.6148 | 0.4733 | 0.4594 | 0.5441 | 0.4982 |
| XGBoost | 0.5751 | 0.4379 | 0.5911 | 0.4482 | 0.4396 | 0.5013 | 0.4685 |

Baselines for context: PR-AUC of a random scorer at a 37.42% positive rate is 0.3742.

**Selected model: Logistic Regression**, by validation ROC-AUC (0.5890 vs 0.5851 vs
0.5751). On the held-out test set it also leads on PR-AUC and F1. Its test confusion
matrix at threshold 0.50:

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** (1,252 leads) | 743 TN | 509 FP |
| **Actual 1** (748 leads) | 314 FN | 434 TP |

Recall 0.58 on the positive class is the number that matters commercially: the model
finds 58% of real converters while reaching only ~50% of the list, which is what makes
the ranked queue worth working. The linear model winning here is a property of the
data (largely additive, monotone engagement effects); on a real CRM export with
non-linear interactions, tree models typically overtake it — the pipeline supports
swapping by changing `best_model` selection, which is already automatic.

> The discriminating power is modest (ROC-AUC ≈ 0.63). That is an honest reflection of
> the synthetic generator's built-in noise, not a tuning shortcut. See
> [Limitations](#15-limitations--next-steps).

---

## 7. Probability Calibration & Lead Tiers

Thresholding a raw classifier output gives scores that are *ranked* but not
*probabilities*. Since the business acts on absolute values ("70% = call now"), the
selected model is calibrated with **isotonic regression** on the validation split:

| Metric (test set) | Raw | Isotonic-calibrated |
|---|---|---|
| Brier score (lower is better) | 0.2381 | **0.2246** |
| ROC-AUC | 0.6269 | 0.6233 |

Calibration improves the Brier score materially (better absolute probabilities) at a
negligible ranking cost — the right trade for tier thresholds.

### Tiers and actions

| Tier | Probability | Action |
|---|---|---|
| **Hot** | ≥ 70% | Immediate follow-up: call within 2 hours, assign to a senior rep |
| **Warm** | 35% – 69.9% | Follow-up within 24 hours: personalised email + scheduled call |
| **Cold** | < 35% | Nurture: drip email campaign, re-engage in 7–14 days |

Thresholds live in `src/models/calibration.py` (`HOT_THRESHOLD`, `WARM_THRESHOLD`)
and are exposed in `calibration_summary.json`.

```bash
python -m src.models.calibration   # refit calibrator + rewrite calibration_summary.json
```

---

## 8. Explainability (SHAP)

`src/explainability/shap_explainer.py` wraps the fitted pipeline and returns, per lead,
the three features with the largest absolute SHAP contribution plus their direction
(`increases` / `decreases` conversion likelihood).

- `LinearExplainer` for Logistic Regression, `TreeExplainer` for tree models —
  chosen automatically from the estimator type.
- SHAP background sample is drawn from the processed dataset (100 rows).
- Contributions are aggregated to readable feature names, including one-hot levels
  (e.g. `Domain_Real Estate`) and the engineered scores.

Example output for a Website / Form Submitted lead (the `/predict` response below is
this exact lead):

```json
{
  "probability": 0.3974,
  "top_factors": [
    {"feature": "Recency / Activity Flag", "contribution": 0.2425, "direction": "positive"},
    {"feature": "High-Intent Source Indicator", "contribution": -0.1229, "direction": "negative"},
    {"feature": "Domain_Real Estate", "contribution": -0.1210, "direction": "negative"}
  ]
}
```

The UI renders these as sentences ("X increases conversion likelihood, impact …")
next to the probability and the next-best-action, so a rep sees the *reason* before
dialling.

---

## 9. FastAPI Backend

Run: `uvicorn src.api.main:app --host 127.0.0.1 --port 8000`
Interactive docs: <http://localhost:8000/docs>

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + which models are loaded in cache |
| `GET` | `/models` | Available model keys and display names |
| `POST` | `/predict?model_name=logistic_regression` | Score one lead |
| `POST` | `/predict-batch` | Upload CSV (`file`) + `model_name` form field; returns ranked predictions |
| `GET` | `/model-performance` | Test-set metrics for all trained models |
| `POST` | `/run-test` | Retrain a selected model on the standard split and return fresh metrics |

### Single prediction

```bash
curl -X POST "http://localhost:8000/predict?model_name=logistic_regression" \
  -H "Content-Type: application/json" \
  -d '{
    "Lead Source": "Website",
    "Total Visits": 5,
    "Time Spent on Website": 12.5,
    "Page Views Per Visit": 4,
    "Last Activity": "Form Submitted",
    "City or Region": "Mumbai",
    "Occupation or Company Size": "Software Engineer",
    "Domain": "Real Estate"
  }'
```

```json
{
  "model": "logistic_regression",
  "probability": 0.3974,
  "probability_percent": 39.74,
  "tier": "Warm",
  "action": "Follow-up within 24 hours: Send personalised email and schedule call.",
  "top_factors": [
    {"feature": "Recency / Activity Flag", "contribution": 0.2425, "direction": "positive"},
    {"feature": "High-Intent Source Indicator", "contribution": -0.1229, "direction": "negative"},
    {"feature": "Domain_Real Estate", "contribution": -0.1210, "direction": "negative"}
  ],
  "explanation": ["Recency / Activity Flag: increases conversion likelihood (impact: 0.243)", "..."]
}
```

### Batch scoring

```bash
curl -X POST "http://localhost:8000/predict-batch" \
  -F "file=@leads.csv;type=text/csv" \
  -F "model_name=logistic_regression"
```

The CSV must contain the eight feature columns. `Converted` is optional and ignored.
A missing feature column returns `400` with the missing column names. Results are
sorted by probability descending; per-row scoring errors are returned inline as a
`tier` of `"Error"` instead of failing the whole upload.

---

## 10. Streamlit Application

Run: `streamlit run frontend/app.py` → <http://localhost:8501>

The UI uses Streamlit's native components (no custom CSS theme) and requires the API
to be running on port 8000; a sidebar **Check API Health** button verifies the link.

**1. Single Lead Simulator** — pick a model, enter the eight lead attributes, and get
the conversion probability, a Hot/Warm/Cold tier, the top-3 SHAP reasons and the
recommended next-best-action.

**2. Batch Upload & Sales Queue** — upload raw leads as CSV. The page validates the
schema (missing columns are shown in an amber warning), scores every row, shows tier
counts, sorts by probability, filters by tier with a multi-select, displays the
prioritised table and offers the filtered call list as a downloadable CSV.

**3. Model Performance** — the comparison table from `/model-performance`, with a
**Run Test** control that retrains a chosen model and reports ROC-AUC, PR-AUC,
precision/recall/F1 and the confusion matrix.

**4. Business Analytics** — model evaluation panel, conversion funnel, lead-source
performance, engagement distributions and the predicted tier mix from the last batch.

---

## 11. Project Structure

```
Intelligent_lead_sale_agent/
├── data/
│   └── raw/
│       └── leads_dataset.csv          # 10,000-row lead dataset
├── models/
│   └── trained/
│       ├── logistic_regression.joblib
│       ├── logistic_regression_calibrated.joblib
│       ├── random_forest.joblib
│       ├── xgboost.joblib
│       ├── preprocessor.joblib
│       ├── feature_names.json
│       ├── metrics.json               # selected model, test metrics
│       ├── model_comparison.json      # all three models, val + test metrics
│       └── calibration_summary.json
├── src/
│   ├── api/
│   │   └── main.py                    # FastAPI app (/predict, /predict-batch, ...)
│   ├── data/
│   │   ├── generate_dataset.py        # synthetic but realistic CRM lead generator
│   │   └── feature_engineering.py     # LeadFeatureEngineer + ColumnTransformer
│   ├── explainability/
│   │   └── shap_explainer.py          # SHAP top-3 reasons per lead
│   └── models/
│       ├── train_models.py            # trains + compares the three models
│       └── calibration.py             # isotonic calibration, tiers, actions
├── frontend/
│   └── app.py                         # Streamlit UI (4 pages)
├── tests/
│   └── test_feature_engineering.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 12. Setup & Run Instructions

Tested on Python 3.11–3.14 on Windows and Linux. All commands are run **from the
repository root** — the training and calibration scripts resolve `data/` and
`models/` relative to the working directory.

### 1. Clone and create an environment

```bash
git clone https://github.com/<your-username>/Smart-Lead-Scoring-Engine_AlchemistDigitalSolution.git
cd Smart-Lead-Scoring-Engine_AlchemistDigitalSolution

python -m venv .venv

# Windows (Git Bash / PowerShell)
source .venv/Scripts/activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .          # makes `src.*` importable from anywhere
```

### 2. Get the data and models

Pre-generated dataset and trained artifacts are committed, so this step is optional —
run it only to regenerate from scratch:

```bash
python -m src.data.generate_dataset     # writes data/raw/leads_dataset.csv
python -m src.models.train_models       # writes models/trained/*.joblib + metrics
python -m src.models.calibration        # writes the calibrated model + summary
```

### 3. Start the API (terminal 1)

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Check it: `curl http://localhost:8000/health` → `{"status":"ok","models_loaded":[]}`

### 4. Start the UI (terminal 2)

```bash
streamlit run frontend/app.py --server.port 8501
```

Open <http://localhost:8501>, click **Check API Health** in the sidebar, then use any
of the four pages.

---

## 13. Deployment

### Docker (optional, for a self-contained deployment)

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000 8501

# API and UI in one container via supervisord or two `docker run`s:
#   docker build -t lead-scoring .
#   docker run -p 8000:8000 lead-scoring uvicorn src.api.main:app --host 0.0.0.0 --port 8000
#   docker run -p 8501:8501 -e API_BASE=http://localhost:8000 lead-scoring \
#       streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

The UI reads the API location from the `API_BASE` environment variable
(`API_BASE = os.getenv("API_BASE", "http://localhost:8000")` at the top of
`frontend/app.py`), so point it at the deployed API when the two services run on
different hosts.

### Suggested managed hosting (for the submission link)

- **API → Render / Railway / Fly.io**: start command
  `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`, environment variable none
  required (the API bundles the dataset and trained artifacts).
- **UI → Streamlit Community Cloud / Hugging Face Spaces**: set the
  `API_BASE` secret to the deployed API URL. Streamlit Cloud runs the UI only; it
  cannot host the FastAPI service, so both services must be deployed.

> **Status:** no public deployment link exists yet for this repository. The commands
> above are the ones that were verified locally; add the live URLs to this section
> once the services are pushed.

---

## 14. Testing

```bash
pytest -q
pytest --cov=src --cov-report=term-missing   # with coverage
```

Current tests cover the feature-engineering contract: engineered columns are created,
the flag features respond to high-intent values, and the full pipeline fits and
predicts without leakage.

Manual verification performed against the running services:

- `/health`, `/models`, `/predict` for all three models, `/predict-batch` with a raw
  CSV, `/run-test`, `/model-performance`
- Single Lead Simulator, Batch Upload & Sales Queue (upload → rank → filter → export)
  and Business Analytics pages in the browser

---

## 15. Limitations & Next Steps

- **Modest discrimination (ROC-AUC ≈ 0.63).** The dataset is synthetic with deliberate
  noise. Ranking quality on real CRM data should be re-measured; the training script
  and comparison table regenerate automatically.
- **No notebook.** All work is in scripts (`src/data`, `src/models`) so it is
  reproducible from the CLI; a notebook walkthrough would be a presentational
  addition, not a functional one.
- **Tier thresholds are business priors**, not optimised against call outcomes.
  Once real dial outcomes exist, tune the thresholds on expected revenue per call
  rather than on F1.
- **Static scoring.** Leads are scored on request; a production version would rescore
  on activity events (form submit, demo request) and push Hot leads to the dialer.
- **Probability drift.** Isotonic calibration is fitted on one snapshot; monitor the
  predicted-versus-actual conversion rate and refit monthly.

---

## Author

Built as an end-to-end ML submission: data generation, feature engineering, model
comparison, calibration, explainability, API, and an executive-ready UI.
