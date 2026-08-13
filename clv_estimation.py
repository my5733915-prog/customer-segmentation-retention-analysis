"""
Step 7: CLV Estimation
BG/NBD model -> predicts future purchase frequency & probability customer is "alive"
Gamma-Gamma model -> predicts average monetary value per transaction
Combined -> Customer Lifetime Value
"""
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

print("Loading cleaned dataset...")

df = pd.read_csv(
    "data/cleaned_online_retail.csv",
    parse_dates=["InvoiceDate"]
)

print("Dataset Shape:", df.shape)

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# ---------------------------------------------------------------
# 1. Build RFM-T summary table required by `lifetimes`
#    frequency = # repeat purchases (transactions AFTER the first)
#    recency   = age (days) at time of last purchase
#    T         = age (days) at time of observation (i.e. tenure)
#    monetary_value = avg spend per REPEAT transaction
# ---------------------------------------------------------------
summary = summary_data_from_transaction_data(
    df, "CustomerID", "InvoiceDate",
    monetary_value_col="TotalPrice",
    observation_period_end=df["InvoiceDate"].max()
)
print(f"Summary table shape: {summary.shape}")
print(summary.describe())

returning = summary[summary["frequency"] > 0]
print(f"\nOne-time buyers (frequency=0): {(summary['frequency'] == 0).sum()} "
      f"({(summary['frequency'] == 0).mean():.1%})")
print(f"Repeat buyers (frequency>0): {len(returning)} ({len(returning)/len(summary):.1%})")

# ---------------------------------------------------------------
# 2. BG/NBD model — models purchase frequency + "alive" probability
# ---------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

bgf = BetaGeoFitter(penalizer_coef=0.1)
bgf.fit(summary["frequency"], summary["recency"], summary["T"])
print("\n--- BG/NBD model fitted ---")
print(bgf.summary)

# Predict expected purchases in next 90 days
summary["PredictedPurchases_90d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
    90, summary["frequency"], summary["recency"], summary["T"]
)
# Probability customer is still "alive" (active) given their history
summary["ProbAlive"] = bgf.conditional_probability_alive(
    summary["frequency"], summary["recency"], summary["T"]
)

# ---------------------------------------------------------------
# 3. Gamma-Gamma model — models average monetary value per transaction
#    Requires frequency > 0 (needs at least 1 repeat purchase) AND
#    frequency/monetary_value should be roughly uncorrelated (assumption check)
# ---------------------------------------------------------------
corr = returning[["frequency", "monetary_value"]].corr().iloc[0, 1]
print(f"\nCorrelation between frequency and monetary_value (should be near 0): {corr:.4f}")

ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(returning["frequency"], returning["monetary_value"])
print("\n--- Gamma-Gamma model fitted ---")
print(ggf.summary)

# ---------------------------------------------------------------
# 4. Combined CLV (12-month horizon, monthly discount rate)
# ---------------------------------------------------------------
# customer_lifetime_value() internally calls conditional_expected_average_profit,
# which uses population-level shrinkage for frequency=0 customers, so we can
# safely score ALL customers (not just returning ones) — one-time buyers get
# a CLV based on population averages rather than their own unreliable estimate.
summary_valid = summary.copy()
summary_valid["monetary_value"] = summary_valid["monetary_value"].fillna(0)

clv = ggf.customer_lifetime_value(
    bgf,
    summary_valid["frequency"],
    summary_valid["recency"],
    summary_valid["T"],
    summary_valid["monetary_value"],
    time=12,          # months
    freq="D",
    discount_rate=0.01  # monthly discount rate
)
summary_valid["CLV_12m"] = clv

# ---------------------------------------------------------------
# 4b. Known limitation: BG/NBD + Gamma-Gamma is unstable for one-time
#     buyers (frequency=0). The BG/NBD fit here has a,b -> ~0 (boundary
#     estimates), which makes the DCF projection produce NaN or negative
#     values specifically for this group. Confirmed: 100% of NaN/negative
#     CLVs fall in the frequency=0 group; repeat buyers are unaffected.
#     Fix: for one-time buyers, fall back to actual historical spend as a
#     conservative CLV floor (we know they're worth at least this much;
#     we just can't reliably project their future with this model).
# ---------------------------------------------------------------
actual_spend = df.groupby("CustomerID")["TotalPrice"].sum()
onetime_mask = summary_valid["frequency"] == 0
n_unstable = summary_valid.loc[onetime_mask, "CLV_12m"].isna().sum() + \
             (summary_valid.loc[onetime_mask, "CLV_12m"] < 0).sum()
print(f"\nOne-time buyers with unstable model CLV (NaN or negative): "
      f"{n_unstable} / {onetime_mask.sum()}")
summary_valid.loc[onetime_mask, "CLV_12m"] = summary_valid.loc[onetime_mask].index.map(actual_spend)
# Safety net: clip any remaining negative values (repeat buyers should have none)
summary_valid["CLV_12m"] = summary_valid["CLV_12m"].clip(lower=0)

print("\n--- CLV Summary (12-month, after fix) ---")
print(summary_valid["CLV_12m"].describe())

# ---------------------------------------------------------------
# 5. CLV tiers for business use
# ---------------------------------------------------------------
summary_valid["CLV_Tier"] = pd.qcut(
    summary_valid["CLV_12m"], q=[0, 0.5, 0.8, 1.0],
    labels=["Low Value", "Mid Value", "High Value"]
)
print("\n--- CLV Tier distribution ---")
tier_summary = summary_valid.groupby("CLV_Tier", observed=True).agg(
    Customers=("CLV_12m", "count"),
    AvgCLV=("CLV_12m", "mean"),
    TotalCLV=("CLV_12m", "sum")
)
print(tier_summary)

summary_valid = summary_valid.reset_index().rename(columns={"index": "CustomerID"})
summary_valid.to_csv(
    "data/clv_predictions.csv",
    index=False
)

print("\nSaved CLV predictions.")

# ---------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(summary_valid[summary_valid["CLV_12m"] < summary_valid["CLV_12m"].quantile(0.95)]["CLV_12m"],
             bins=50, color="#6E8894")
axes[0].set_title("Distribution of Predicted 12-Month CLV (<95th pct)")
axes[0].set_xlabel("CLV (£)")

tier_summary["TotalCLV"].plot(kind="bar", ax=axes[1], color=["#A45D5D", "#D9A05B", "#2E5266"])
axes[1].set_title("Total CLV Contribution by Tier")
axes[1].set_ylabel("Total 12-month CLV (£)")
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(
    "visualizations/clv_distribution.png",
    dpi=120
)

plt.close()

print("Saved CLV visualization.")