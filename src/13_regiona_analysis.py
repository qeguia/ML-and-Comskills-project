# 13_region_analysis.py
# Compares regions by loss rate (left) and by dollars lost (right). Also prints why the regions differ: how often and how deeply each region discounts.

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_and_prepare, PROJECT_ROOT, apply_plot_style, CATEGORY_COLORS

df = load_and_prepare()
categories = ["Furniture", "Office Supplies", "Technology"]
regions = df.groupby("Region")["Unprofitable"].mean().sort_values(ascending=False).index.tolist()

# ── 1. Numbers behind the figure ─────────────────────────────
rate = df.groupby(["Region", "Category"])["Unprofitable"].mean().unstack().loc[regions, categories] * 100
loss = (df[df["Profit"] < 0].groupby(["Region", "Category"])["Profit"].sum()
        .unstack().reindex(index=regions, columns=categories).fillna(0) * -1 / 1000)
overall = df.groupby("Region")["Unprofitable"].mean().loc[regions] * 100

print("=" * 60)
print("REGIONAL PROFITABILITY")
print("=" * 60)
print("\nUnprofitable rate (%) by region and category:")
print(rate.round(1).to_string())
print("\nOverall unprofitable rate (%):")
print(overall.round(1).to_string())
print("\nGross loss in $ thousands (sum of negative profits) by region and category:")
print(loss.round(1).to_string())
print("\nTotal gross loss ($ thousands):")
print(loss.sum(axis=1).round(1).to_string())

print("\nWhy do regions differ?")
print("  Share of transactions with any discount (%):")
print((df.groupby("Region").apply(lambda d: (d["Discount"] > 0).mean(), include_groups=False).loc[regions] * 100).round(1).to_string())
print("  Average Furniture discount (%):")
print((df[df.Category == "Furniture"].groupby("Region")["Discount"].mean().loc[regions] * 100).round(1).to_string())
print("  Unprofitable rate (%) among transactions with exactly 20% discount:")
print((df[df["Discount"] == 0.2].groupby("Region")["Unprofitable"].mean().loc[regions] * 100).round(1).to_string())

# ── 2. Figure ────────────────────────────────────────────────
apply_plot_style(16)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
x = np.arange(len(regions))
w = 0.26

# Loss rate by region and category
for k, cat in enumerate(categories):
    bars = ax1.bar(x + (k - 1) * w, rate[cat].values, w, color=CATEGORY_COLORS[cat], label=cat)
    for b, v in zip(bars, rate[cat].values):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{r}\n(all: {overall[r]:.1f}%)" for r in regions], fontsize=14)
ax1.set_ylabel("Unprofitable transactions (%)")
ax1.set_title("How often a transaction loses money")
ax1.set_ylim(0, 62)
ax1.legend(loc="upper right")
ax1.grid(True, axis="y", alpha=0.3)
ax1.annotate("Focus here:\nCentral Furniture", xy=(0 - w, rate.loc[regions[0], "Furniture"]),
             xytext=(0.55, 52), fontsize=13, fontweight="bold", color="#c0392b",
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))

# Dollars lost by region and category (stacked)
bottom = np.zeros(len(regions))
for cat in categories:
    ax2.bar(x, loss[cat].values, 0.6, bottom=bottom, color=CATEGORY_COLORS[cat], label=cat)
    bottom += loss[cat].values
for xi, tot in zip(x, bottom):
    ax2.text(xi, tot + 0.6, f"${tot:.1f}k", ha="center", fontsize=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(regions)
ax2.set_ylabel("Money lost on unprofitable\ntransactions ($ thousands)")
ax2.set_title("How much money is lost")
ax2.set_ylim(0, bottom.max() * 1.15)
ax2.legend(loc="upper right")
ax2.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
fig_dir = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "region_profitability.png")
plt.savefig(fig_path, dpi=200)
print(f"\nFigure saved in {fig_path}")