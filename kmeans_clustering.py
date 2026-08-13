"""
Step 5: K-Means Clustering
Feature Scaling -> Elbow Method -> Silhouette Score -> Optimal K ->
PCA (2D) -> Cluster Profiling -> Business Insights
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --------------------------------------------------
# Style
# --------------------------------------------------
sns.set_style("whitegrid")

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_PATH = "data/rfm_table.csv"
OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
print("Loading RFM table...")

rfm = pd.read_csv(DATA_PATH)

print(f"Dataset Shape: {rfm.shape}")
print("Program Started...")

# --------------------------------------------------
# 1. Feature Preparation
# --------------------------------------------------
features = rfm[["Recency", "Frequency", "Monetary"]].copy()

# Log Transformation
features["Recency"] = np.log1p(features["Recency"])
features["Frequency"] = np.log1p(features["Frequency"])
features["Monetary"] = np.log1p(features["Monetary"])

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# --------------------------------------------------
# 2. Elbow Method + Silhouette Score
# --------------------------------------------------
k_range = range(2, 11)

inertias = []
sil_scores = []

print("\nCalculating optimal K...\n")

for k in k_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

    print(
        f"k={k} | "
        f"Inertia={model.inertia_:.2f} | "
        f"Silhouette={sil_scores[-1]:.4f}"
    )

# --------------------------------------------------
# Plot Elbow + Silhouette
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(k_range, inertias, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("K")
axes[0].set_ylabel("WCSS")

axes[1].plot(k_range, sil_scores, marker="o")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Score")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "kmeans_k_selection.png"
    ),
    dpi=300
)

plt.show()

print("\nElbow graph saved.")

# --------------------------------------------------
# 3. Optimal K
# --------------------------------------------------
best_k = k_range[np.argmax(sil_scores)]

print(f"\nBest K according to Silhouette Score = {best_k}")

# Business choice
OPTIMAL_K = 4

print(f"Using K = {OPTIMAL_K}")

# --------------------------------------------------
# 4. Train Final Model
# --------------------------------------------------
kmeans = KMeans(
    n_clusters=OPTIMAL_K,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(X_scaled)

print(
    f"Final Silhouette Score = "
    f"{silhouette_score(X_scaled, rfm['Cluster']):.4f}"
)

# --------------------------------------------------
# 5. PCA Visualization
# --------------------------------------------------
pca = PCA(n_components=2)

coords = pca.fit_transform(X_scaled)

rfm["PCA1"] = coords[:, 0]
rfm["PCA2"] = coords[:, 1]

print("\nExplained Variance Ratio")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(9,7))

palette = sns.color_palette("Set2", OPTIMAL_K)

for cluster in range(OPTIMAL_K):

    subset = rfm[rfm["Cluster"] == cluster]

    plt.scatter(
        subset["PCA1"],
        subset["PCA2"],
        s=30,
        alpha=0.6,
        label=f"Cluster {cluster}"
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Customer Clusters")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "kmeans_pca.png"
    ),
    dpi=300
)

plt.show()

print("PCA visualization saved.")

# --------------------------------------------------
# 6. Cluster Profiling
# --------------------------------------------------
profile = (
    rfm.groupby("Cluster")
       .agg(
            Customers=("CustomerID","count"),
            AvgRecency=("Recency","mean"),
            AvgFrequency=("Frequency","mean"),
            AvgMonetary=("Monetary","mean"),
            TotalRevenue=("Monetary","sum")
       )
       .round(2)
)

profile["PctOfCustomers"] = (
    profile["Customers"] /
    profile["Customers"].sum() * 100
).round(2)

profile["PctOfRevenue"] = (
    profile["TotalRevenue"] /
    profile["TotalRevenue"].sum() * 100
).round(2)

profile = profile.sort_values(
    "AvgMonetary",
    ascending=False
)

print("\n========== CLUSTER PROFILE ==========\n")

print(profile)

# --------------------------------------------------
# Save Files
# --------------------------------------------------
rfm.to_csv(
    "data/rfm_clustered.csv",
    index=False
)

profile.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "cluster_profile.csv"
    )
)

print("\nFiles Saved Successfully")

print("data/rfm_clustered.csv")

print("outputs/cluster_profile.csv")