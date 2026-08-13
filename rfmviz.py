"""
Step 5: RFM Visualization
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

# -----------------------------
# Plot Style
# -----------------------------
sns.set_style("whitegrid")

# -----------------------------
# File Paths
# -----------------------------
DATA_PATH = "data/rfm_table.csv"
OUTPUT_FOLDER = "outputs"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Load RFM Data
# -----------------------------
print("Loading RFM table...")

rfm = pd.read_csv(DATA_PATH)

print(f"RFM Table Shape: {rfm.shape}")

# -----------------------------
# Segment Summary
# -----------------------------
seg_summary = (
    rfm.groupby("Segment")
       .agg(
           Customers=("CustomerID", "count"),
           TotalRevenue=("Monetary", "sum")
       )
       .sort_values("TotalRevenue", ascending=False)
)

# -----------------------------
# Colors
# -----------------------------
colors = sns.color_palette("viridis", len(seg_summary))

# -----------------------------
# Create Figure
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# =====================================================
# Treemap
# =====================================================
labels = [
    f"{seg}\n{cust} Customers\n£{rev/1e6:.2f}M"
    for seg, cust, rev in zip(
        seg_summary.index,
        seg_summary["Customers"],
        seg_summary["TotalRevenue"]
    )
]

squarify.plot(
    sizes=seg_summary["TotalRevenue"],
    label=labels,
    color=colors,
    alpha=0.85,
    ax=axes[0],
    text_kwargs={"fontsize": 9}
)

axes[0].set_title("Revenue Contribution by RFM Segment")
axes[0].axis("off")

# =====================================================
# Scatter Plot
# =====================================================
segments = rfm["Segment"].unique()

palette = dict(
    zip(
        segments,
        sns.color_palette("tab10", len(segments))
    )
)

for seg in segments:

    sub = rfm[rfm["Segment"] == seg]

    axes[1].scatter(
        sub["Frequency"],
        sub["Monetary"],
        label=seg,
        alpha=0.6,
        s=30,
        color=palette[seg]
    )

axes[1].set_xscale("log")
axes[1].set_yscale("log")

axes[1].set_xlabel("Frequency (Log Scale)")
axes[1].set_ylabel("Monetary (£) (Log Scale)")
axes[1].set_title("Customer Distribution by Frequency & Monetary")

axes[1].legend(
    fontsize=8,
    loc="upper left",
    ncol=2
)

# -----------------------------
# Save Figure
# -----------------------------
plt.tight_layout()

output_file = os.path.join(
    OUTPUT_FOLDER,
    "rfm_segments.png"
)

plt.savefig(
    output_file,
    dpi=300
)

plt.show()

print(f"\nRFM visualization saved successfully at:")
print(output_file)