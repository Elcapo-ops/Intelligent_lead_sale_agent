"""Generate a realistic synthetic lead dataset for Real Estate, EdTech, Solar, and B2B Services.

Produces at least 3,000 records with the required features:
- Lead Source
- Total Visits
- Time Spent on Website
- Page Views Per Visit
- Last Activity
- City or Region
- Occupation or Company Size
- Converted (target: 0 or 1)

Plus domain-specific engineered features added later in the pipeline.
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_RECORDS = 10000  #  above the 3,000 minimum

DOMAINS = ["Real Estate", "EdTech", "Solar", "B2B Services"]
LEAD_SOURCES = [
    "Website", "Facebook Ads", "Google Ads", "LinkedIn",
    "Referral", "Email Campaign", "Cold Call", "Webinar", "Organic Search"
]
CITIES = [
    "Kabul", "Mazar-e-Sharif", "Bangalore", "Ontario", "Madrid",
    "Berlin", "Paris", "Tokyo", "Sydney", "Toronto",
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "London", "Manchester", "Birmingham", "Sydney", "Melbourne"
]
OCCUPATIONS = [
    "Software Engineer", "Doctor", "Teacher", "Accountant",
    "Marketing Manager", "Sales Executive", "Entrepreneur", "Student",
    "Retired", "Freelancer", "HR Manager", "Operations Manager"
]
COMPANY_SIZES = ["1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"]
LAST_ACTIVITY_OPTIONS = [
    "Website Visit", "Email Opened", "Form Submitted", "Call Connected",
    "Demo Requested", "Brochure Downloaded", "Webinar Attended",
    "Pricing Page Visited", "Chat Session", "None"
]


def _generate_domain_data(rng, n_per_domain):
    """Generate records for a single domain."""
    records = []
    for _ in range(n_per_domain):
        domain = rng.choice(DOMAINS)

        # Domain-specific conversion base rates
        base_conversion = {
            "Real Estate": 0.18,
            "EdTech": 0.22,
            "Solar": 0.15,
            "B2B Services": 0.25,
        }[domain]

        lead_source = rng.choice(LEAD_SOURCES, p=[
            0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.08, 0.06, 0.06
        ])

        # Source-specific intent modifiers
        source_intent = {
            "Website": 1.0, "Facebook Ads": 0.7, "Google Ads": 1.1,
            "LinkedIn": 1.2, "Referral": 1.5, "Email Campaign": 0.9,
            "Cold Call": 0.5, "Webinar": 1.3, "Organic Search": 0.9,
        }[lead_source]

        total_visits = int(rng.poisson(3) + 1)
        total_visits = min(total_visits, 25)

        time_spent = float(rng.gamma(2.0, 5.0))  # minutes
        time_spent = round(min(time_spent, 120), 2)

        page_views_per_visit = float(rng.normal(3.5, 1.5))
        page_views_per_visit = round(max(page_views_per_visit, 0.5), 2)

        last_activity = rng.choice(LAST_ACTIVITY_OPTIONS, p=[
            0.15, 0.12, 0.18, 0.10, 0.12, 0.08, 0.08, 0.07, 0.06, 0.04
        ])

        city = rng.choice(CITIES)

        if domain == "B2B Services":
            occupation_or_size = rng.choice(COMPANY_SIZES)
        else:
            occupation_or_size = rng.choice(OCCUPATIONS)

        # Conversion probability influenced by features
        logit = (
            np.log(base_conversion / (1 - base_conversion))
            + 0.3 * np.log(source_intent)
            + 0.15 * np.log1p(total_visits)
            + 0.1 * np.log1p(time_spent / 5.0)
            + 0.1 * np.log1p(page_views_per_visit)
        )

        # High-intent sources and activities boost conversion
        if lead_source in ("Referral", "LinkedIn", "Webinar"):
            logit += 0.4
        if last_activity in ("Demo Requested", "Pricing Page Visited", "Form Submitted"):
            logit += 0.5
        if total_visits >= 5:
            logit += 0.3
        if time_spent > 20:
            logit += 0.2

        prob = 1 / (1 + np.exp(-logit))
        converted = int(rng.random() < prob)

        records.append({
            "Lead Source": lead_source,
            "Total Visits": total_visits,
            "Time Spent on Website": time_spent,
            "Page Views Per Visit": page_views_per_visit,
            "Last Activity": last_activity,
            "City or Region": city,
            "Occupation or Company Size": occupation_or_size,
            "Domain": domain,
            "Converted": converted,
        })

    return pd.DataFrame(records)


def generate_dataset(n_records: int = N_RECORDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate the full lead dataset."""
    rng = np.random.default_rng(seed)
    n_per_domain = n_records // len(DOMAINS)
    remainder = n_records % len(DOMAINS)

    dfs = []
    for i, domain in enumerate(DOMAINS):
        count = n_per_domain + (1 if i < remainder else 0)
        df_domain = _generate_domain_data(rng, count)
        dfs.append(df_domain)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, "data", "raw", "leads_dataset.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records -> {output_path}")
    print(df.head())
    print("\nConversion rate:", df["Converted"].mean())
    print("\nDomain distribution:\n", df["Domain"].value_counts())