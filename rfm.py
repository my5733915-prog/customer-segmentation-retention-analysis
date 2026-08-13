"""
Step 4: RFM Analysis
Recency, Frequency, Monetary Analysis
"""

import os
import pandas as pd
import numpy as np

# -----------------------------------
# File Paths
# -----------------------------------
DATA_PATH = "data/cleaned_online_retail.csv"
OUTPUT_PATH = "data/rfm_table.csv"

# -----------------------------------
# Load Dataset
# -----------------------------------
print("Loading cleaned dataset...")

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["InvoiceDate"]
)

print(f"Dataset Shape: {df.shape}")

# -----------------------------------
# Reference Date
# -----------------------------------
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

print(f"\nReference Date: {reference_date}")

# -----------------------------------
# Create RFM Table
# -----------------------------------
rfm = df.groupby("CustomerID").agg(

    Recency=("InvoiceDate",
              lambda x: (reference_date - x.max()).days),

    Frequency=("InvoiceNo", "nunique"),

    Monetary=("TotalPrice", "sum")

).reset_index()

print(f"\nRFM Table Shape: {rfm.shape}")

print("\nRFM Summary Statistics\n")

print(rfm.describe())

# -----------------------------------
# RFM Scoring
# -----------------------------------

# Recency (Lower is Better)
rfm["R_Score"] = pd.qcut(
    rfm["Recency"],
    5,
    labels=[5,4,3,2,1]
).astype(int)

# Frequency (Higher is Better)
rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1,2,3,4,5]
).astype(int)

# Monetary (Higher is Better)
rfm["M_Score"] = pd.qcut(
    rfm["Monetary"],
    5,
    labels=[1,2,3,4,5]
).astype(int)

# -----------------------------------
# Overall Scores
# -----------------------------------

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str)
    + rfm["F_Score"].astype(str)
    + rfm["M_Score"].astype(str)
)

rfm["RFM_Sum"] = (
    rfm["R_Score"]
    + rfm["F_Score"]
    + rfm["M_Score"]
)

# -----------------------------------
# Customer Segmentation
# -----------------------------------

def segment_customer(row):

    r = row["R_Score"]
    f = row["F_Score"]
    m = row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    elif r >= 3 and f >= 3 and m >= 3:
        return "Loyal Customers"

    elif r >= 4 and f <= 2:
        return "New Customers"

    elif r >= 3 and f <= 2 and m <= 2:
        return "Promising"

    elif r <= 2 and f >= 4 and m >= 4:
        return "Can't Lose Them"

    elif r <= 2 and f >= 3 and m >= 3:
        return "At Risk"

    elif r <= 2 and f <= 2 and m <= 2:
        return "Lost"

    elif r == 3 and f == 3:
        return "Need Attention"

    else:
        return "Others"


rfm["Segment"] = rfm.apply(segment_customer, axis=1)

# -----------------------------------
# Segment Summary
# -----------------------------------

print("\n========== CUSTOMER SEGMENTS ==========\n")

print(rfm["Segment"].value_counts())

seg_summary = (

    rfm.groupby("Segment")
    .agg(

        Customers=("CustomerID","count"),

        AvgRecency=("Recency","mean"),

        AvgFrequency=("Frequency","mean"),

        AvgMonetary=("Monetary","mean"),

        TotalRevenue=("Monetary","sum")

    )
    .sort_values("TotalRevenue",ascending=False)

)

print("\n========== SEGMENT SUMMARY ==========\n")

print(seg_summary)

# -----------------------------------
# Save RFM Table
# -----------------------------------

rfm.to_csv(OUTPUT_PATH,index=False)

print(f"\nRFM table saved successfully at:\n{OUTPUT_PATH}")