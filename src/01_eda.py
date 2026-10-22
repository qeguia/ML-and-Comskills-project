# 01_eda.py
# Exploratory Data Analysis.

import os
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_and_prepare, PROJECT_ROOT

# ── 1. Load data ──────────────────────────────────────────────
df = load_and_prepare()

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Rows: {len(df)}")
print(f"Unprofitable: {df['Unprofitable'].sum()} ({df['Unprofitable'].mean():.1%})")
print(f"Profitable:   {(1 - df['Unprofitable']).sum():.0f} ({1 - df['Unprofitable'].mean():.1%})")
print()

# ── 2. Unprofitable rate by category ──────────────────────────
print("UNPROFITABLE RATE BY CATEGORY")
print("-" * 40)
for cat in ["Furniture", "Office Supplies", "Technology"]:
    subset = df[df["Category"] == cat]
    rate = subset["Unprofitable"].mean()
    print(f"  {cat:20s}: {rate:.1%}  (n={len(subset)})")
print()

# ── 3. Unprofitable rate by discount level ────────────────────
print("UNPROFITABLE RATE BY DISCOUNT LEVEL")
print("-" * 40)

# Create discount bins
bins = [0, 0.001, 0.15, 0.30, 0.50]
labels = ["0%", "(0%,15%]", "(15%,30%]", "(30%,50%]"]
df["Discount_Bin"] = pd.cut(df["Discount"], bins=bins, labels=labels,
                            include_lowest=True)

for label in labels:
    subset = df[df["Discount_Bin"] == label]
    rate = subset["Unprofitable"].mean()
    avg_profit = subset["Profit"].mean()
    print(f"  {label:15s}: {rate:5.1%} unprofitable, "
          f"avg profit = ${avg_profit:+.0f}  (n={len(subset)})")
print()

# ── 4. Worst sub-categories ──────────────────────────────────
print("TOP 5 WORST SUB-CATEGORIES")
print("-" * 40)
subcat_stats = (
    df.groupby("Sub-Category")
    .agg(unprof_rate=("Unprofitable", "mean"),
         avg_profit=("Profit", "mean"),
         count=("Unprofitable", "count"))
    .sort_values("unprof_rate", ascending=False)
    .head(5)
)
for name, row in subcat_stats.iterrows():
    print(f"  {name:15s}: {row['unprof_rate']:.1%} unprofitable, "
          f"avg profit = ${row['avg_profit']:+.0f}  (n={row['count']:.0f})")
print()

# ── 5. Regional comparison ───────────────────────────────────
print("UNPROFITABLE RATE BY REGION")
print("-" * 40)
for region in ["Central", "East", "South", "West"]:
    subset = df[df["Region"] == region]
    rate = subset["Unprofitable"].mean()
    print(f"  {region:10s}: {rate:.1%}  (n={len(subset)})")
print()

# ── 6. Save the EDA figure ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for cat in ["Furniture", "Office Supplies", "Technology"]:
    subset = df[df["Category"] == cat]
    rates = subset.groupby("Discount_Bin")["Unprofitable"].mean() * 100
    counts = subset.groupby("Discount_Bin")["Unprofitable"].count()
    ax.plot(rates.index, rates.values, marker="o", linewidth=2, label=cat)

    # Add sample sizes
    for x, y_val, n in zip(rates.index, rates.values, counts.values):
        ax.annotate(f"n={n}", (x, y_val), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)

ax.set_xlabel("Discount Level")
ax.set_ylabel("Unprofitable Transactions (%)")
ax.set_title("Unprofitable Rate by Discount Level and Category")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig_dir = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "eda_discount_by_category.png")
plt.savefig(fig_path, dpi=150)
print(f"Figure saved to {fig_path}")