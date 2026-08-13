"""
Step 9: Final Interactive Dashboard
Combines KPIs, RFM segments, clusters, churn risk, CLV, and the
business priority map into a single self-contained HTML file.
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

import os

os.makedirs("outputs", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

print("Loading dashboard data...")

biz = pd.read_csv("outputs/business_recommendations.csv")
rfm = pd.read_csv("data/rfm_clustered.csv")

print("Business Table:", biz.shape)
print("RFM Table:", rfm.shape)

# ---------------------------------------------------------------
# KPI numbers
# ---------------------------------------------------------------
total_customers = len(biz)
total_clv = biz["CLV_12m"].sum()
avg_churn = biz["ChurnProbability"].mean()
urgent = biz[biz["Recommendation"] == "URGENT: VIP Win-back"]
revenue_at_risk = urgent["CLV_12m"].sum()
low_priority_pct = biz["Recommendation"].isin(
    ["Low-cost Automation Only", "Minimal Spend"]).mean()

# ---------------------------------------------------------------
# Chart 1: RFM Segment revenue treemap-style bar
# ---------------------------------------------------------------
seg_summary = rfm.groupby("Segment").agg(
    Customers=("CustomerID", "count"), Revenue=("Monetary", "sum")
).reset_index().sort_values("Revenue", ascending=True)
fig_seg = px.bar(seg_summary, x="Revenue", y="Segment", orientation="h",
                  title="Revenue by RFM Segment", text="Customers",
                  color="Revenue", color_continuous_scale="Teal")
fig_seg.update_traces(texttemplate="%{text} custs", textposition="outside")
fig_seg.update_layout(showlegend=False, coloraxis_showscale=False, height=400)

# ---------------------------------------------------------------
# Chart 2: CLV Tier distribution
# ---------------------------------------------------------------
clv_summary = biz.groupby("CLV_Tier").agg(
    Customers=("CustomerID", "count"), TotalCLV=("CLV_12m", "sum")
).reset_index()
fig_clv = px.pie(clv_summary, names="CLV_Tier", values="TotalCLV",
                  title="12-Month CLV Contribution by Tier", hole=0.45,
                  color="CLV_Tier",
                  color_discrete_map={"High Value": "#2E5266", "Mid Value": "#D9A05B",
                                       "Low Value": "#A45D5D"})
fig_clv.update_layout(height=400)

# ---------------------------------------------------------------
# Chart 3: Churn Risk distribution
# ---------------------------------------------------------------
risk_summary = biz["ChurnRiskTier"].value_counts().reset_index()
risk_summary.columns = ["ChurnRiskTier", "Customers"]
fig_risk = px.bar(risk_summary, x="ChurnRiskTier", y="Customers",
                   title="Customers by Churn Risk Tier", text="Customers",
                   color="ChurnRiskTier",
                   color_discrete_map={"Low Risk": "#2E8B57", "Medium Risk": "#DAA520",
                                        "High Risk": "#B22222"},
                   category_orders={"ChurnRiskTier": ["Low Risk", "Medium Risk", "High Risk"]})
fig_risk.update_traces(textposition="outside")
fig_risk.update_layout(showlegend=False, height=400)

# ---------------------------------------------------------------
# Chart 4: Business Priority Map (interactive scatter)
# ---------------------------------------------------------------
palette = {"URGENT: VIP Win-back": "#B22222", "Proactive Loyalty": "#D2691E",
           "Nurture & Upsell": "#2E8B57", "Automated Win-back": "#DAA520",
           "Engagement Campaign": "#4682B4", "Standard Loyalty": "#5F9EA0",
           "Low-cost Automation Only": "#A9A9A9", "Monitor": "#C0C0C0",
           "Minimal Spend": "#D3D3D3"}
biz_plot = biz.copy()
biz_plot["CLV_capped"] = biz_plot["CLV_12m"].clip(upper=biz_plot["CLV_12m"].quantile(0.97))
fig_map = px.scatter(
    biz_plot, x="ChurnProbability", y="CLV_capped", color="Recommendation",
    color_discrete_map=palette, hover_data=[
    "CustomerID",
    "Segment",
    "CLV_12m",
    "Recommendation",
    "Recency",
    "Frequency",
    "Monetary"
],
    title="Retention Priority Map — Churn Risk vs Customer Lifetime Value"
)
fig_map.update_layout(height=550, xaxis_title="Churn Probability",
                       yaxis_title="12-month CLV (£, capped at 97th pct)")

# ---------------------------------------------------------------
# Top 20 urgent customers table
# ---------------------------------------------------------------
top_urgent = (
    biz[biz["Recommendation"] == "URGENT: VIP Win-back"]
    .sort_values("CLV_12m", ascending=False)
    .head(20)
)

if top_urgent.empty:
    top_urgent = biz.sort_values("CLV_12m", ascending=False).head(20)
fig_table = go.Figure(data=[go.Table(
    header=dict(values=["Customer ID", "Segment", "Churn Prob", "12m CLV (£)", "Recency (d)"],
                fill_color="#2E5266", font=dict(color="white"), align="left"),
    cells=dict(values=[top_urgent["CustomerID"], top_urgent["Segment"],
                        top_urgent["ChurnProbability"].round(2),
                        top_urgent["CLV_12m"].round(0), top_urgent["Recency"]],
               fill_color=[["#f5f5f5", "white"] * len(top_urgent)], align="left"))
])
fig_table.update_layout(title="Top 20 Urgent Win-back Priority Customers", height=500)

# ---------------------------------------------------------------
# Assemble HTML dashboard
# ---------------------------------------------------------------
kpi_html = f"""
<div style="display:flex; gap:16px; margin-bottom:24px; flex-wrap:wrap;">
  <div style="flex:1; min-width:180px; background:#2E5266; color:white; padding:20px; border-radius:10px;">
    <div style="font-size:13px; opacity:0.85;">Total Customers</div>
    <div style="font-size:28px; font-weight:bold;">{total_customers:,}</div>
  </div>
  <div style="flex:1; min-width:180px; background:#3E7C59; color:white; padding:20px; border-radius:10px;">
    <div style="font-size:13px; opacity:0.85;">Total Predicted 12m CLV</div>
    <div style="font-size:28px; font-weight:bold;">£{total_clv:,.0f}</div>
  </div>
  <div style="flex:1; min-width:180px; background:#D9A05B; color:white; padding:20px; border-radius:10px;">
    <div style="font-size:13px; opacity:0.85;">Avg Churn Probability</div>
    <div style="font-size:28px; font-weight:bold;">{avg_churn:.1%}</div>
  </div>
  <div style="flex:1; min-width:180px; background:#B22222; color:white; padding:20px; border-radius:10px;">
    <div style="font-size:13px; opacity:0.85;">Revenue at Risk (Urgent)</div>
    <div style="font-size:28px; font-weight:bold;">£{revenue_at_risk:,.0f}</div>
  </div>
  <div style="flex:1; min-width:180px; background:#6E8894; color:white; padding:20px; border-radius:10px;">
    <div style="font-size:13px; opacity:0.85;">Low-Priority Customers</div>
    <div style="font-size:28px; font-weight:bold;">{low_priority_pct:.1%}</div>
  </div>
</div>
"""

html_parts = [
    "<html><head><title>Customer Segmentation and Retention Analysis Dashboard</title>",
    "<style>body{font-family:Arial,Helvetica,sans-serif; margin:30px; background:#fafafa;} "
    "h1{color:#2E5266;} .row{display:flex; gap:20px; flex-wrap:wrap;} "
    ".chart{flex:1; min-width:400px; background:white; border-radius:10px; padding:10px; "
    "box-shadow:0 1px 4px rgba(0,0,0,0.1); margin-bottom:20px;}</style></head><body>",
    "<h1>Customer Segmentation & Retention Analysis Dashboard</h1>",
    "<p>Online Retail Dataset (UCI) — RFM Analysis, K-Means Segmentation, "
    "Churn Prediction, and CLV-driven Retention Strategy</p>",
    kpi_html,
    '<div class="row">',
    f'<div class="chart">{pio.to_html(fig_seg, full_html=False, include_plotlyjs="cdn")}</div>',
    f'<div class="chart">{pio.to_html(fig_clv, full_html=False, include_plotlyjs=False)}</div>',
    "</div>",
    '<div class="row">',
    f'<div class="chart">{pio.to_html(fig_risk, full_html=False, include_plotlyjs=False)}</div>',
    "</div>",
    '<div class="row">',
    f'<div class="chart" style="min-width:100%;">{pio.to_html(fig_map, full_html=False, include_plotlyjs=False)}</div>',
    "</div>",
    '<div class="row">',
    f'<div class="chart" style="min-width:100%;">{pio.to_html(fig_table, full_html=False, include_plotlyjs=False)}</div>',
    "</div>",
    "</body></html>",
]

with open("outputs/dashboard.html", "w", encoding="utf-8") as f:
    f.write("\n".join(html_parts))

print("\nDashboard created successfully!")
print("Saved to: outputs/dashboard.html")
print(f"\nKPIs: {total_customers} customers | £{total_clv:,.0f} total CLV | "
      f"{avg_churn:.1%} avg churn prob | £{revenue_at_risk:,.0f} revenue at risk | "
      f"{low_priority_pct:.1%} low priority")