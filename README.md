# Customer Segmentation & Retention Analysis

End-to-end customer analytics pipeline on the **Online Retail (UCI)** dataset:
RFM analysis → K-Means segmentation → churn prediction → CLV estimation →
business-ready retention recommendations.

**Why this matters:** subscription and retail businesses need to know which
customers are worth fighting to keep. This project doesn't stop at
"here are some clusters" — it produces a prioritized, actionable list of
who to save and who to let go, backed by predicted revenue impact.

---

## Results at a glance

| Metric | Value |
|---|---|
| Customers analyzed | 4,334 |
| Total predicted 12-month CLV | £8.47M |
| Customers flagged URGENT (high value + high churn risk) | 18 (£213K revenue at risk) |
| Customers where retention spend isn't justified by CLV | 1,366 (31.5%) |
| Best churn model | Random Forest — ROC-AUC 0.738, Recall 0.73 |

**Headline insight:** 22% of customers ("Champions") generate 65% of
revenue. A tiny slice (0.4%) sit in the highest-value + highest-risk
quadrant — that's where retention budget should go first.

---

## Pipeline & Methodology

| Step | Script | What it does |
|---|---|---|
| 1. Environment | `requirements.txt` | Python + ML/analytics stack |
| 2. Data Cleaning | `src/01_data_cleaning.py` | Removes missing IDs, cancellations, non-product codes, invalid rows. 541,909 → 391,283 rows |
| 3. EDA | `src/02_eda.py` | Revenue trends, country mix, order value & frequency distributions |
| 4. RFM Analysis | `src/03_rfm.py`, `04_rfm_viz.py` | Quintile-based Recency/Frequency/Monetary scoring, 8 business segments |
| 5. K-Means Clustering | `src/05_kmeans_clustering.py` | StandardScaler → Elbow + Silhouette → k=4 → PCA (94% variance in 2D) → cluster profiling |
| 6. Churn Prediction | `src/06_churn_model.py` | **Time-based holdout** (not naive recency threshold — avoids label leakage). Logistic Regression → Random Forest → XGBoost, compared on Accuracy/Precision/Recall/F1/ROC-AUC |
| 7. CLV Estimation | `src/07_clv_estimation.py` | BG/NBD (purchase behavior) + Gamma-Gamma (spend behavior) via `lifetimes` |
| 8. Business Logic | `src/08_business_logic.py` | Combines Segment + live Churn Risk + CLV into a 9-cell recommendation matrix |
| 9. Dashboard | `src/09_dashboard.py` | Interactive Plotly HTML dashboard — `outputs/dashboard.html` |

### A note on rigor (things that could have gone wrong, and how they were handled)

- **Churn leakage risk:** defining churn as "Recency > N days" and also using
  Recency as a model feature would make the model circular. Fixed with a
  proper time-based train/holdout split — features from an observation
  window, label from a separate future holdout window.
- **CLV model instability:** the BG/NBD model's dropout parameters converged
  to a boundary solution (a≈0), which made Gamma-Gamma CLV projections
  unreliable (NaN/negative) — but *only* for one-time buyers (1,558
  customers, 100% of that group). Repeat buyers (2,776) were unaffected.
  Fix: one-time buyers fall back to their actual historical spend as a
  conservative CLV floor, since the model can't reliably project their
  future from a single data point.
- **k selection for K-Means:** silhouette score alone favored k=2 (too
  coarse to be actionable). Chose k=4 by balancing the elbow curve,
  silhouette score, and business interpretability — not just the
  statistical maximum.

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn (K-Means, Logistic Regression,
Random Forest) · XGBoost · `lifetimes` (BG/NBD, Gamma-Gamma) ·
Matplotlib · Seaborn · Plotly · squarify

## Setup

```bash
pip install -r requirements.txt
```

Get the raw dataset:
```bash
mkdir -p data
curl -L "https://raw.githubusercontent.com/eaintkyawthmu/UCI_Online_Retail_Dataset_Cleaned_Version/master/Online%20Retail.xlsx" -o data/online_retail.xlsx
```

## Run order

```bash
python src/01_data_cleaning.py
python src/02_eda.py
python src/03_rfm.py
python src/04_rfm_viz.py
python src/05_kmeans_clustering.py
python src/06_churn_model.py
python src/07_clv_estimation.py
python src/08_business_logic.py
python src/09_dashboard.py   # -> outputs/dashboard.html
```

## Repo structure

```
customer-segmentation/
├── src/                    # numbered pipeline scripts (run in order)
├── visualizations/         # PNG charts from every stage
├── outputs/                # business_recommendations.csv, dashboard.html, model comparison tables
├── data/                   # raw + intermediate data (gitignored — see Setup)
└── requirements.txt
```

## Business recommendation matrix

| CLV Tier \\ Churn Risk | High Risk | Medium Risk | Low Risk |
|---|---|---|---|
| **High Value** | 🔴 URGENT: VIP Win-back | Proactive Loyalty | Nurture & Upsell |
| **Mid Value** | Automated Win-back | Engagement Campaign | Standard Loyalty |
| **Low Value** | Low-cost Automation Only | Monitor | Minimal Spend |

Full per-customer output: `outputs/business_recommendations.csv`.