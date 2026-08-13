import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Plot Style
# -----------------------------
sns.set_style("whitegrid")

# -----------------------------
# File Paths
# -----------------------------
DATA_PATH = "data/cleaned_online_retail.csv"
OUTPUT_FOLDER = "outputs"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading cleaned dataset...")

df = pd.read_csv(DATA_PATH, parse_dates=["InvoiceDate"])

print(f"Dataset Shape: {df.shape}")

# -----------------------------
# Create Figure
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ======================================================
# 1. Monthly Revenue Trend
# ======================================================
monthly = (
    df.set_index("InvoiceDate")
      .resample("ME")["TotalPrice"]
      .sum()
)

axes[0, 0].plot(monthly.index, monthly.values, marker="o")
axes[0, 0].set_title("Monthly Revenue Trend")
axes[0, 0].set_ylabel("Revenue (£)")
axes[0, 0].tick_params(axis="x", rotation=45)

# ======================================================
# 2. Top 10 Countries by Revenue
# ======================================================
country_rev = (
    df.groupby("Country")["TotalPrice"]
      .sum()
      .sort_values(ascending=False)
)

top10 = country_rev.head(10)

axes[0, 1].barh(top10.index[::-1], top10.values[::-1])
axes[0, 1].set_title("Top 10 Countries by Revenue")
axes[0, 1].set_xlabel("Revenue (£)")

# ======================================================
# 3. Distribution of Invoice Value
# ======================================================
invoice_value = (
    df.groupby("InvoiceNo")["TotalPrice"]
      .sum()
)

axes[1, 0].hist(
    invoice_value[invoice_value < invoice_value.quantile(0.95)],
    bins=50
)

axes[1, 0].set_title("Distribution of Invoice Value (<95th Percentile)")
axes[1, 0].set_xlabel("Invoice Total (£)")
axes[1, 0].set_ylabel("Count")

# ======================================================
# 4. Purchase Frequency
# ======================================================
freq = (
    df.groupby("CustomerID")["InvoiceNo"]
      .nunique()
)

axes[1, 1].hist(
    freq[freq < freq.quantile(0.95)],
    bins=40
)

axes[1, 1].set_title("Purchase Frequency per Customer")
axes[1, 1].set_xlabel("Distinct Invoices")
axes[1, 1].set_ylabel("Customers")

# -----------------------------
# Save Figure
# -----------------------------
plt.tight_layout()

output_file = os.path.join(OUTPUT_FOLDER, "eda_overview.png")

plt.savefig(output_file, dpi=300)

plt.show()

print(f"\nEDA chart saved successfully at:")
print(output_file)

# -----------------------------
# Summary Statistics
# -----------------------------
print("\n========== KEY EDA STATISTICS ==========\n")

print(f"Average Invoice Value : £{invoice_value.mean():.2f}")
print(f"Median Invoice Value  : £{invoice_value.median():.2f}")

print(f"Average Purchases per Customer : {freq.mean():.2f}")
print(f"Median Purchases per Customer  : {freq.median():.0f}")

uk_share = (
    country_rev["United Kingdom"] /
    country_rev.sum()
) * 100

print(f"UK Revenue Share : {uk_share:.1f}%")

print("\nTop 5 Countries by Revenue\n")
print(country_rev.head())

print("\nEDA Completed Successfully!")