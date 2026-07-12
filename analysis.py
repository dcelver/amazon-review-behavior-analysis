"""
Do Star Ratings Predict Future Reviews?
A Fixed Effects panel analysis of Amazon Beauty product reviews.

This script builds a monthly product-level panel from raw Amazon review
data and estimates how rating characteristics (average rating, rating
variance, review volume, and rating thresholds) relate to future review
activity.

Data source: Amazon Reviews 2023 (All Beauty category), McAuley Lab, UCSD
https://amazon-reviews-2023.github.io/
"""

import gzip

import numpy as np
import pandas as pd
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects, compare

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
# Download both files from https://amazon-reviews-2023.github.io/ and place
# them in a local `data/` folder before running this script.

REVIEW_PATH = "data/All_Beauty.jsonl"
META_PATH = "data/meta_All_Beauty.jsonl.gz"

reviews = pd.read_json(REVIEW_PATH, lines=True)

with gzip.open(META_PATH, "rt", encoding="utf-8") as f:
    meta = pd.read_json(f, lines=True)

meta = meta[
    [
        "parent_asin",
        "price",
        "store",
        "average_rating",
        "rating_number",
        "main_category",
        "categories",
        "details",
    ]
]

# ---------------------------------------------------------------------------
# 2. Prepare review-level data
# ---------------------------------------------------------------------------
reviews = reviews[["asin", "parent_asin", "rating", "timestamp"]].copy()

reviews["review_date"] = pd.to_datetime(reviews["timestamp"], unit="ms")
reviews["year_month"] = reviews["review_date"].dt.to_period("M")

# ---------------------------------------------------------------------------
# 3. Select the 100 most-reviewed products
# ---------------------------------------------------------------------------
product_review_counts = (
    reviews.groupby("asin").size().reset_index(name="total_reviews_raw")
)
top_100_asins = (
    product_review_counts.sort_values("total_reviews_raw", ascending=False)
    .head(100)["asin"]
)

reviews_top100 = reviews[reviews["asin"].isin(top_100_asins)].copy()

# ---------------------------------------------------------------------------
# 4. Sort chronologically and build cumulative rating measures
# ---------------------------------------------------------------------------
reviews_top100 = reviews_top100.sort_values(
    by=["asin", "review_date"]
).reset_index(drop=True)

reviews_top100["review_order"] = (
    reviews_top100.groupby("asin").cumcount() + 1
)

reviews_top100["cumulative_avg_rating"] = (
    reviews_top100.groupby("asin")["rating"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)

reviews_top100["cumulative_variance"] = (
    reviews_top100.groupby("asin")["rating"]
    .expanding()
    .var()
    .reset_index(level=0, drop=True)
)

reviews_top100["cumulative_review_count"] = (
    reviews_top100.groupby("asin").cumcount() + 1
)

# ---------------------------------------------------------------------------
# 5. Build the monthly product panel
# ---------------------------------------------------------------------------
panel = (
    reviews_top100.groupby(["asin", "year_month"], as_index=False).last()
)

monthly_counts = (
    reviews_top100.groupby(["asin", "year_month"])
    .size()
    .reset_index(name="monthly_review_count")
)

panel = panel.merge(monthly_counts, on=["asin", "year_month"], how="left")

panel["threshold_40"] = (panel["cumulative_avg_rating"] >= 4.0).astype(int)
panel["threshold_45"] = (panel["cumulative_avg_rating"] >= 4.5).astype(int)

# Fill in months with zero reviews so every product has a complete monthly
# timeline between its first and last observed review.
product_dates = (
    reviews_top100.groupby("asin")["year_month"].agg(["min", "max"]).reset_index()
)

complete_panel = pd.concat(
    [
        pd.DataFrame(
            {
                "asin": row["asin"],
                "year_month": pd.period_range(
                    start=row["min"], end=row["max"], freq="M"
                ),
            }
        )
        for _, row in product_dates.iterrows()
    ],
    ignore_index=True,
)

panel = complete_panel.merge(panel, on=["asin", "year_month"], how="left")

panel["monthly_review_count"] = (
    panel["monthly_review_count"].fillna(0).astype(int)
)

columns_to_fill = [
    "cumulative_avg_rating",
    "cumulative_variance",
    "cumulative_review_count",
    "threshold_40",
    "threshold_45",
]
panel[columns_to_fill] = panel.groupby("asin")[columns_to_fill].ffill()
panel["cumulative_variance"] = panel["cumulative_variance"].fillna(0)

# Dependent variable: reviews the product will receive next month
panel["next_month_review_count"] = (
    panel.groupby("asin")["monthly_review_count"].shift(-1)
)

panel = panel.drop(
    columns=["parent_asin", "rating", "timestamp", "review_date", "review_order"]
)
panel = panel.dropna(subset=["next_month_review_count"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 6. Merge in product metadata (price)
# ---------------------------------------------------------------------------
asin_parent = (
    reviews_top100[["asin", "parent_asin"]].drop_duplicates()
)
panel = panel.merge(asin_parent, on="asin", how="left")

meta_clean = meta.drop_duplicates(subset="parent_asin")
panel = panel.merge(meta_clean, on="parent_asin", how="left")

panel = panel[
    [
        "asin",
        "year_month",
        "monthly_review_count",
        "next_month_review_count",
        "cumulative_avg_rating",
        "cumulative_variance",
        "cumulative_review_count",
        "threshold_40",
        "threshold_45",
        "price",
    ]
].copy()

panel["threshold_40"] = panel["threshold_40"].astype(int)
panel["threshold_45"] = panel["threshold_45"].astype(int)

print("Final panel shape:", panel.shape)

# ---------------------------------------------------------------------------
# 7. Descriptive statistics, correlation, and multicollinearity check
# ---------------------------------------------------------------------------
analysis_data = panel[
    [
        "next_month_review_count",
        "monthly_review_count",
        "cumulative_avg_rating",
        "cumulative_variance",
        "cumulative_review_count",
        "threshold_40",
        "threshold_45",
        "price",
    ]
]

print("\nDescriptive statistics:")
print(analysis_data.describe().T.round(3))

corr_data = analysis_data.drop(columns=["next_month_review_count", "price"])
print("\nCorrelation matrix:")
print(corr_data.corr().round(3))

X_vif = add_constant(corr_data)
vif = pd.DataFrame()
vif["Variable"] = X_vif.columns
vif["VIF"] = [
    variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])
]
print("\nVariance Inflation Factors:")
print(vif.round(3))

# ---------------------------------------------------------------------------
# 8. Panel regression: Pooled OLS, Random Effects, Fixed Effects
# ---------------------------------------------------------------------------
panel["year_month"] = panel["year_month"].dt.to_timestamp()
panel_model = panel.set_index(["asin", "year_month"])

y = panel_model["next_month_review_count"]
X = panel_model[
    [
        "cumulative_avg_rating",
        "cumulative_variance",
        "monthly_review_count",
        "cumulative_review_count",
        "threshold_40",
        "threshold_45",
    ]
]
X = add_constant(X)

pooled_results = PooledOLS(y, X).fit(cov_type="clustered", cluster_entity=True)
fe_results = PanelOLS(y, X, entity_effects=True).fit(
    cov_type="clustered", cluster_entity=True
)
re_results = RandomEffects(y, X).fit(cov_type="clustered", cluster_entity=True)

print("\n=== Model comparison ===")
print(compare({"Pooled OLS": pooled_results, "Fixed Effects": fe_results,
               "Random Effects": re_results}))

# ---------------------------------------------------------------------------
# 9. Hausman test (Fixed Effects vs. Random Effects)
# ---------------------------------------------------------------------------
b_diff = fe_results.params - re_results.params
V_diff = fe_results.cov - re_results.cov

hausman_stat = np.dot(np.dot(b_diff.T, np.linalg.inv(V_diff)), b_diff)
df = len(b_diff)
from scipy.stats import chi2
p_value = 1 - chi2.cdf(hausman_stat, df)

print(f"\nHausman statistic: {hausman_stat:.4f}")
print(f"Degrees of freedom: {df}")
print(f"P-value: {p_value:.4f}")
print("-> Fixed Effects is preferred when p < 0.05")

# ---------------------------------------------------------------------------
# 10. Final Fixed Effects results table
# ---------------------------------------------------------------------------
fixed_table = pd.DataFrame(
    {
        "Coefficient": fe_results.params,
        "Std. Error": fe_results.std_errors,
        "t-statistic": fe_results.tstats,
        "p-value": fe_results.pvalues,
    }
).round(4)

print("\n=== Final Fixed Effects Results ===")
print(fixed_table)
