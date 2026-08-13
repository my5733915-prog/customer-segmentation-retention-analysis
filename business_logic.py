"""
Step 8: Business Logic Layer
Combines: RFM/K-Means Segment + Churn Probability + CLV
Output: prioritized, actionable retention recommendations per customer.

Note on churn scoring: the Step 6 model was trained/validated using a
time-based holdout (features from day 0-283, label = inactivity in days
283-373) specifically to get an honest, leakage-free performance estimate.
For actual business scoring here, we re-fit the same Random Forest with
that same validated methodology, then apply it to features computed from
the FULL transaction history (cutoff = "today" = max date) so every
customer gets a current, real-time churn risk score. This mirrors how
you'd deploy in production: validate on a holdout, then score live data.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

sns.set_style("whitegrid")
print("Loading cleaned dataset...")

df = pd.read_csv(
    "data/cleaned_online_retail.csv",
    parse_dates=["InvoiceDate"]
)

print("Dataset Shape:", df.shape)

os.makedirs("outputs", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

FEATURE_COLS = ["Recency", "Frequency", "Monetary", "UniqueProducts",
                 "AvgQuantity", "Tenure", "AvgOrderValue", "AvgDaysBetweenPurchases"]


def build_features(obs_df, cutoff):
    feat = obs_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (cutoff - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        UniqueProducts=("StockCode", "nunique"),
        AvgQuantity=("Quantity", "mean"),
    ).reset_index()
    feat["Tenure"] = (cutoff - feat["FirstPurchase"]).dt.days
    feat["AvgOrderValue"] = feat["Monetary"] / feat["Frequency"]
    feat["AvgDaysBetweenPurchases"] = np.where(
        feat["Frequency"] > 1, feat["Tenure"] / feat["Frequency"], feat["Tenure"]
    )
    return feat.drop(columns=["FirstPurchase"])


# ---------------------------------------------------------------
# 1. Re-fit churn model using the SAME validated methodology (Step 6)
# ---------------------------------------------------------------
max_date = df["InvoiceDate"].max()
VALIDATION_CUTOFF = max_date - pd.Timedelta(days=90)
obs = df[df["InvoiceDate"] <= VALIDATION_CUTOFF]
holdout = df[df["InvoiceDate"] > VALIDATION_CUTOFF]
holdout_customers = set(holdout["CustomerID"].unique())

train_feat = build_features(obs, VALIDATION_CUTOFF)
train_feat["Churned"] = (~train_feat["CustomerID"].isin(holdout_customers)).astype(int)

rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                             random_state=42, n_jobs=-1)
rf.fit(train_feat[FEATURE_COLS], train_feat["Churned"])
print("Re-fitted validated Random Forest churn model.")

# ---------------------------------------------------------------
# 2. Score ALL current customers using full transaction history
# ---------------------------------------------------------------
live_feat = build_features(df, max_date + pd.Timedelta(days=1))
live_feat["ChurnProbability"] = rf.predict_proba(live_feat[FEATURE_COLS])[:, 1]
print(f"Scored {len(live_feat)} customers with current churn risk.")

# ---------------------------------------------------------------
# 3. Merge with Segment (RFM/K-Means) and CLV
# ---------------------------------------------------------------
rfm_clustered = pd.read_csv("data/rfm_clustered.csv")
clv = pd.read_csv("data/clv_predictions.csv")

biz = live_feat[["CustomerID", "ChurnProbability", "Recency", "Frequency", "Monetary"]].merge(
    rfm_clustered[["CustomerID", "Segment", "Cluster"]], on="CustomerID", how="left"
).merge(
    clv[["CustomerID", "CLV_12m", "CLV_Tier"]], on="CustomerID", how="left"
)
print(f"\nFinal merged business table: {biz.shape}")
print(f"Missing Segment: {biz['Segment'].isna().sum()}, Missing CLV: {biz['CLV_12m'].isna().sum()}")

# ---------------------------------------------------------------
# 4. Churn risk tiers (data-driven: tertiles)
# ---------------------------------------------------------------
biz["ChurnRiskTier"] = pd.qcut(
    biz["ChurnProbability"],
    q=3,
    labels=["Low Risk", "Medium Risk", "High Risk"],
    duplicates="drop"
)

print("\n--- Churn Risk Tier distribution ---")
print(biz["ChurnRiskTier"].value_counts())
print("\n--- CLV Tier x Churn Risk crosstab (customer count) ---")
print(pd.crosstab(biz["CLV_Tier"], biz["ChurnRiskTier"]))

# ---------------------------------------------------------------
# 5. Business logic: recommendation matrix
# ---------------------------------------------------------------
def recommend(row):
    # Handle missing values
    if pd.isna(row["CLV_Tier"]) or pd.isna(row["ChurnRiskTier"]):
        return "Unknown", 99, "Review customer manually"
    clv_tier, risk = row["CLV_Tier"], row["ChurnRiskTier"]
    if clv_tier == "High Value" and risk == "High Risk":
        return "URGENT: VIP Win-back", 1, "Personal outreach + exclusive offer; highest priority, do not lose"
    if clv_tier == "High Value" and risk == "Medium Risk":
        return "Proactive Loyalty", 2, "Early access, loyalty rewards, relationship check-in"
    if clv_tier == "High Value" and risk == "Low Risk":
        return "Nurture & Upsell", 3, "Cross-sell premium items, referral incentives"
    if clv_tier == "Mid Value" and risk == "High Risk":
        return "Automated Win-back", 4, "Targeted email campaign with moderate discount"
    if clv_tier == "Mid Value" and risk == "Medium Risk":
        return "Engagement Campaign", 5, "Personalized product recommendations"
    if clv_tier == "Mid Value" and risk == "Low Risk":
        return "Standard Loyalty", 6, "Newsletter, seasonal offers"
    if clv_tier == "Low Value" and risk == "High Risk":
        return "Low-cost Automation Only", 7, "Generic re-engagement email; retention spend not justified by CLV"
    if clv_tier == "Low Value" and risk == "Medium Risk":
        return "Monitor", 8, "No proactive action; track for tier changes"
    return "Minimal Spend", 9, "Standard mass marketing only"


biz[["Recommendation", "Priority", "Action"]] = biz.apply(
    lambda r: pd.Series(recommend(r)), axis=1
)

biz = biz.sort_values(["Priority", "CLV_12m"], ascending=[True, False])
biz.to_csv(
    "outputs/business_recommendations.csv",
    index=False
)
# ---------------------------------------------------------------
# 6. Business impact summary
# ---------------------------------------------------------------
print("\n--- Recommendation Summary ---")
summary = biz.groupby(["Priority", "Recommendation"], observed=True).agg(
    Customers=("CustomerID", "count"),
    TotalCLV=("CLV_12m", "sum"),
    AvgChurnProb=("ChurnProbability", "mean")
).round(2)
print(summary.to_string())

urgent = biz[biz["Recommendation"] == "URGENT: VIP Win-back"]
revenue_at_risk = urgent["CLV_12m"].sum()
print(f"\n*** {len(urgent)} customers flagged URGENT (High CLV + High Churn Risk) ***")
print(f"*** Revenue at risk from this group alone: £{revenue_at_risk:,.0f} ***")

low_priority = biz[biz["Recommendation"].isin(["Low-cost Automation Only", "Minimal Spend"])]
print(f"\n{len(low_priority)} customers ({len(low_priority)/len(biz):.1%}) flagged as low "
      f"priority — retention spend here would exceed their CLV.")

# ---------------------------------------------------------------
# 7. Visualization: CLV vs Churn Risk quadrant map
# ---------------------------------------------------------------
plt.figure(figsize=(10, 7))
palette = {"URGENT: VIP Win-back": "#B22222", "Proactive Loyalty": "#D2691E",
           "Nurture & Upsell": "#2E8B57", "Automated Win-back": "#DAA520",
           "Engagement Campaign": "#4682B4", "Standard Loyalty": "#5F9EA0",
           "Low-cost Automation Only": "#A9A9A9", "Monitor": "#C0C0C0",
           "Minimal Spend": "#D3D3D3"}
for rec in biz["Recommendation"].unique():
    sub = biz[biz["Recommendation"] == rec]
    plt.scatter(sub["ChurnProbability"], sub["CLV_12m"].clip(upper=biz["CLV_12m"].quantile(0.97)),
                label=rec, alpha=0.6, s=25, color=palette.get(rec, "#333333"))
plt.xlabel("Churn Probability")
plt.ylabel("Predicted 12-month CLV (£, clipped at 97th pct)")
plt.title("Retention Priority Map: Churn Risk vs Customer Lifetime Value")
plt.legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig(
    "visualizations/business_priority_map.png",
    dpi=120
)

plt.close()

print("\nSaved business priority map.")