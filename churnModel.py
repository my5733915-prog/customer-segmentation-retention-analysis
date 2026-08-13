"""
Step 6: Churn Definition + Prediction Model
Time-based holdout approach (avoids label leakage):
  - Observation window: features computed from this period only
  - Holdout window: churn label = did NOT purchase in this period
Models: Logistic Regression -> Random Forest -> XGBoost
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, roc_curve,
                            confusion_matrix, classification_report)

sns.set_style("whitegrid")
print("Loading cleaned dataset...")

df = pd.read_csv(
    "data/cleaned_online_retail.csv",
    parse_dates=["InvoiceDate"]
)

print("Dataset Shape:", df.shape)

# ---------------------------------------------------------------
# 1. Time-based split
# ---------------------------------------------------------------
max_date = df["InvoiceDate"].max()
CUTOFF = max_date - pd.Timedelta(days=90)

obs = df[df["InvoiceDate"] <= CUTOFF]
holdout = df[df["InvoiceDate"] > CUTOFF]

obs_customers = set(obs["CustomerID"].unique())
holdout_customers = set(holdout["CustomerID"].unique())
print(f"Customers in observation window: {len(obs_customers)}")
print(f"Customers in holdout window: {len(holdout_customers)}")

# ---------------------------------------------------------------
# 2. Feature engineering (observation window ONLY -> no leakage)
# ---------------------------------------------------------------
feat = obs.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (CUTOFF - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
    FirstPurchase=("InvoiceDate", "min"),
    UniqueProducts=("StockCode", "nunique"),
    AvgQuantity=("Quantity", "mean"),
).reset_index()

feat["Tenure"] = (CUTOFF - feat["FirstPurchase"]).dt.days
feat["AvgOrderValue"] = feat["Monetary"] / feat["Frequency"]
feat["AvgDaysBetweenPurchases"] = np.where(
    feat["Frequency"] > 1, feat["Tenure"] / feat["Frequency"], feat["Tenure"]
)
feat = feat.drop(columns=["FirstPurchase"])

# ---------------------------------------------------------------
# 3. Churn label from holdout window
# ---------------------------------------------------------------
feat["Churned"] = (~feat["CustomerID"].isin(holdout_customers)).astype(int)

print(f"\nChurn rate in target: {feat['Churned'].mean():.2%}")
print(feat["Churned"].value_counts())

feat.to_csv("data/churn_features.csv", index=False)

# ---------------------------------------------------------------
# 4. Train / test split
# ---------------------------------------------------------------
feature_cols = ["Recency", "Frequency", "Monetary", "UniqueProducts",
                 "AvgQuantity", "Tenure", "AvgOrderValue", "AvgDaysBetweenPurchases"]
X = feat[feature_cols]
y = feat["Churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 5. Models: Logistic Regression -> Random Forest -> XGBoost
# ---------------------------------------------------------------
results = {}

# --- Logistic Regression (baseline) ---
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
results["Logistic Regression"] = (y_pred_lr, y_prob_lr)

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
results["Random Forest"] = (y_pred_rf, y_prob_rf)

# --- XGBoost ---
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                     scale_pos_weight=scale_pos_weight, random_state=42,
                     eval_metric="logloss")
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
results["XGBoost"] = (y_pred_xgb, y_prob_xgb)

# ---------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------
metrics_table = []
for name, (y_pred, y_prob) in results.items():
    metrics_table.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    })
metrics_df = pd.DataFrame(metrics_table).round(4)
print("\n--- Model Comparison ---")
print(metrics_df.to_string(index=False))
os.makedirs("outputs", exist_ok=True)

metrics_df.to_csv(
    "outputs/churn_model_comparison.csv",
    index=False
)

# ROC curves
plt.figure(figsize=(7, 6))
for name, (y_pred, y_prob) in results.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Churn Prediction Models")
plt.legend()
plt.tight_layout()
os.makedirs("visualizations", exist_ok=True)

plt.savefig(
    "visualizations/churn_roc_curves.png",
    dpi=120
)
print("\nSaved ROC curve comparison.")

# Feature importance (best tree model — XGBoost)
importance = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n--- XGBoost Feature Importance ---")
print(importance)

plt.figure(figsize=(8, 5))
importance.plot(kind="barh", color="#6E8894")
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importance — Churn Prediction")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(
    "visualizations/churn_feature_importance.png",
    dpi=120
)
print("Saved feature importance chart.")

# Confusion matrix for best model (by ROC-AUC)
best_model_name = metrics_df.loc[metrics_df["ROC-AUC"].idxmax(), "Model"]
best_pred, best_prob = results[best_model_name]
cm = confusion_matrix(y_test, best_pred)
print(f"\n--- Confusion Matrix ({best_model_name}) ---")
print(cm)
print(classification_report(y_test, best_pred, target_names=["Active", "Churned"]))

# ---------------------------------------------------------------
# 7. Save churn probabilities for ALL customers (for CLV/business logic step)
# ---------------------------------------------------------------
X_all_scaled = scaler.transform(X)
final_model = xgb  # best performing, tree-based, no scaling needed but kept consistent
feat["ChurnProbability_XGB"] = xgb.predict_proba(X)[:, 1]
feat.to_csv(
    "data/churn_predictions.csv",
    index=False
)
print("\nSaved churn predictions for all customers.")