"""
Step 2: Data Acquisition & Cleaning
Online Retail Dataset (UCI Machine Learning Repository)
"""
import pandas as pd
import numpy as np

RAW_PATH = "data/Online Retail.xlsx"
OUT_PATH = "data/cleaned_online_retail.csv"

print("Loading raw data...")
df = pd.read_excel(RAW_PATH)
print(f"Raw shape: {df.shape}")
print(df.dtypes)
print("\nMissing values:")
print(df.isna().sum())

# ---- Cleaning steps ----
initial_rows = len(df)

# 1. Drop rows with missing CustomerID (can't do customer-level RFM/CLV without it)
df = df.dropna(subset=["CustomerID"])
print(f"\nAfter dropping missing CustomerID: {len(df)} rows "
      f"(-{initial_rows - len(df)})")

# 2. Remove cancelled invoices (InvoiceNo starting with 'C') and adjustment/bad-debt entries
before = len(df)
df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df = df[~df["InvoiceNo"].str.startswith("C")]
df = df[~df["InvoiceNo"].str.startswith("A")]
print(f"After removing cancellations/adjustments: {len(df)} rows (-{before - len(df)})")

# 3. Remove non-product stock codes (POSTAGE, DISCOUNT, MANUAL, BANK CHARGES, etc.)
before = len(df)
non_product_codes = ["POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CRUK"]
df = df[~df["StockCode"].astype(str).str.upper().isin(non_product_codes)]
print(f"After removing non-product stock codes: {len(df)} rows (-{before - len(df)})")

# 4. Remove non-positive Quantity or UnitPrice (returns, errors, free items)
before = len(df)
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
print(f"After removing non-positive quantity/price: {len(df)} rows (-{before - len(df)})")

# 5. Drop duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"After dropping duplicates: {len(df)} rows (-{before - len(df)})")

# 6. Feature engineering: TotalPrice, proper dtypes
df["CustomerID"] = df["CustomerID"].astype(int)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

df = df.reset_index(drop=True)

print(f"\nFinal cleaned shape: {df.shape}")
print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print(f"Unique customers: {df['CustomerID'].nunique()}")
print(f"Unique invoices: {df['InvoiceNo'].nunique()}")
print(f"Unique products: {df['StockCode'].nunique()}")
print(f"Countries: {df['Country'].nunique()}")
print(f"Total revenue: £{df['TotalPrice'].sum():,.2f}")

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved cleaned data to {OUT_PATH}")