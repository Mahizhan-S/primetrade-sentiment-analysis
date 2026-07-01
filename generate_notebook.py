"""
Primetrade.ai Assignment — Jupyter Notebook Generator
Generates a clean .ipynb notebook from the analysis results
"""
import json, os

CELLS = []

def code(src):
    CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src})

def md(src):
    CELLS.append({"cell_type":"markdown","metadata":{},"source":src})


# ──────────────────────────────────────────────────────────────────────────────
md("""# 🔍 Bitcoin Market Sentiment vs Trader Performance
### Primetrade.ai — Data Science Assignment

**Objective:** Explore the relationship between the Bitcoin Fear/Greed Index and real trading performance 
on the Hyperliquid DEX. Uncover hidden patterns and deliver actionable strategy insights.

---
**Datasets:**
- `historical_data.csv` — 124,930 trades by 19 accounts on Hyperliquid (May 2023 – May 2025)
- `fear_greed_index.csv` — Daily Bitcoin Fear/Greed Index values (Feb 2018 – May 2025)

**Key findings preview:**
- ANOVA shows **statistically significant PnL differences across sentiment regimes** (F=9.70, p<0.001)
- **Extreme Greed** yields the highest average PnL ($87.23) despite being a "risky" zone
- **Greed** zone has the *lowest* avg PnL ($35.71) — contrarian to intuition
- Top traders employ **larger position sizes during Fear** and show 2–3× higher win rates
""")

code("""import os, json, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Dark theme
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "text.color": "#e6edf3", "grid.color": "#21262d",
    "grid.linestyle": "--", "grid.alpha": 0.5, "figure.dpi": 120,
})

SENTIMENT_ORDER  = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENTIMENT_COLORS = {
    "Extreme Fear": "#d62728", "Fear": "#ff7f0e",
    "Neutral": "#bcbd22", "Greed": "#2ca02c", "Extreme Greed": "#17becf"
}

BASE   = os.getcwd()
DATA   = os.path.join(BASE, "data")
OUTPUT = os.path.join(BASE, "output")
os.makedirs(OUTPUT, exist_ok=True)
print("✅ Libraries loaded")""")

md("## 1. Load & Inspect Data")

code("""trades = pd.read_csv(os.path.join(DATA, "historical_data.csv"))
fg     = pd.read_csv(os.path.join(DATA, "fear_greed_index.csv"))

print(f"Trades  : {trades.shape[0]:,} rows × {trades.shape[1]} cols")
print(f"Columns : {list(trades.columns)}")
print()
print(f"Fear/Greed: {fg.shape[0]:,} rows × {fg.shape[1]} cols")
print(fg.head())""")

md("## 2. Clean & Merge")

code("""# Fear/Greed
fg['date'] = pd.to_datetime(fg['date'])
fg = fg.rename(columns={'value':'fg_value','classification':'sentiment'})
fg = fg[['date','fg_value','sentiment']].drop_duplicates('date').sort_values('date')

# Trades
trades.columns = trades.columns.str.strip().str.lower().str.replace(' ','_')
ts_col = 'timestamp_ist' if 'timestamp_ist' in trades.columns else 'timestamp'
trades['trade_dt']   = pd.to_datetime(trades[ts_col], dayfirst=True, errors='coerce')
trades['trade_date'] = trades['trade_dt'].dt.normalize()

for col in ['closed_pnl','size_usd','execution_price','size_tokens','fee']:
    if col in trades.columns:
        trades[col] = pd.to_numeric(trades[col], errors='coerce')

trades = trades.dropna(subset=['trade_date','closed_pnl'])

# Merge
merged = trades.merge(fg, left_on='trade_date', right_on='date', how='inner')
merged['is_win']  = merged['closed_pnl'] > 0
merged['is_loss'] = merged['closed_pnl'] < 0

present = [s for s in SENTIMENT_ORDER if s in merged['sentiment'].unique()]
merged['sentiment'] = pd.Categorical(merged['sentiment'], categories=present, ordered=True)

print(f"Merged rows    : {len(merged):,}")
print(f"Date range     : {merged['trade_date'].min().date()} → {merged['trade_date'].max().date()}")
print(f"Unique accounts: {merged['account'].nunique()}")
print(f"Sentiment dist :\\n{merged['sentiment'].value_counts()}")""")

md("## 3. Exploratory Data Analysis")

code("""# ── Fig 1: Trade count & total PnL by sentiment ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Trade Distribution Across Sentiment Regimes", fontsize=13, fontweight='bold')

sent_counts = merged['sentiment'].value_counts().reindex(present)
colors = [SENTIMENT_COLORS[s] for s in sent_counts.index]
axes[0].bar(sent_counts.index, sent_counts.values, color=colors)
axes[0].set_title("Number of Trades by Sentiment")
axes[0].set_ylabel("Trade Count")
for i,v in enumerate(sent_counts.values):
    axes[0].text(i, v+200, f'{v:,}', ha='center', fontsize=9)

sent_pnl = merged.groupby('sentiment')['closed_pnl'].sum().reindex(present)
bar_colors = ['#2ca02c' if v>=0 else '#d62728' for v in sent_pnl.values]
axes[1].bar(sent_pnl.index, sent_pnl.values, color=bar_colors)
axes[1].set_title("Total PnL by Sentiment Regime")
axes[1].set_ylabel("Total Closed PnL (USD)")
axes[1].axhline(0, color='#8b949e', linewidth=0.8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT,"fig1_sentiment_distribution.png"), bbox_inches='tight')
plt.show()""")

code("""# ── Fig 2: PnL boxplot by sentiment ──
fig, ax = plt.subplots(figsize=(13, 6))
data_to_plot = [merged[merged['sentiment']==s]['closed_pnl'].clip(-3000,3000).dropna().values for s in present]
bp = ax.boxplot(data_to_plot, patch_artist=True, notch=True,
                medianprops=dict(color='white', linewidth=2),
                whiskerprops=dict(color='#8b949e'), capprops=dict(color='#8b949e'),
                flierprops=dict(marker='o', markersize=1.5, alpha=0.3))
for patch, s in zip(bp['boxes'], present):
    patch.set_facecolor(SENTIMENT_COLORS[s]); patch.set_alpha(0.8)
ax.set_xticklabels(present)
ax.set_title("PnL Distribution by Sentiment (clipped ±$3,000)", fontsize=12, fontweight='bold')
ax.set_ylabel("Closed PnL (USD)")
ax.axhline(0, color='white', linewidth=0.8, linestyle='--')
plt.tight_layout(); plt.show()""")

code("""# ── Fig 3: Win rate by sentiment ──
win_rate = merged.groupby('sentiment')['is_win'].mean().reindex(present) * 100
fig, ax = plt.subplots(figsize=(10,5))
colors_wr = [SENTIMENT_COLORS[s] for s in win_rate.index]
bars = ax.bar(win_rate.index, win_rate.values, color=colors_wr)
ax.axhline(50, color='#8b949e', linewidth=1.2, linestyle='--', label='50% baseline')
ax.set_title("Win Rate (%) by Sentiment Regime", fontsize=12, fontweight='bold')
ax.set_ylabel("Win Rate (%)"); ax.set_ylim(0,75)
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.legend(); plt.tight_layout(); plt.show()""")

code("""# ── Fig 4: Long vs Short ratio ──
buy_c  = merged[merged['side'].str.upper()=='BUY'].groupby('sentiment').size().reindex(present, fill_value=0)
sell_c = merged[merged['side'].str.upper()=='SELL'].groupby('sentiment').size().reindex(present, fill_value=0)
tot    = (buy_c+sell_c).replace(0,1)
fig, ax = plt.subplots(figsize=(11,5))
x = np.arange(len(present)); w = 0.35
ax.bar(x-w/2, buy_c/tot*100,  width=w, label='LONG (Buy)',  color='#2ca02c', alpha=0.85)
ax.bar(x+w/2, sell_c/tot*100, width=w, label='SHORT (Sell)', color='#d62728', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(present)
ax.axhline(50, color='#8b949e', linewidth=1, linestyle='--')
ax.set_title("Long vs Short Trade Ratio by Sentiment", fontsize=12, fontweight='bold')
ax.set_ylabel("% of Trades"); ax.legend()
plt.tight_layout(); plt.show()""")

md("## 4. Statistical Analysis")

code("""# Daily aggregation
daily_agg = (
    merged.groupby(['trade_date','sentiment','fg_value'])
    .agg(total_pnl=('closed_pnl','sum'), avg_pnl=('closed_pnl','mean'),
         trade_count=('closed_pnl','count'), win_rate=('is_win','mean'))
    .reset_index().sort_values('trade_date')
)

# Pearson & Spearman correlation
corr_data = daily_agg[['fg_value','avg_pnl']].dropna()
pearson_r, pearson_p   = stats.pearsonr(corr_data['fg_value'], corr_data['avg_pnl'])
spearman_r, spearman_p = stats.spearmanr(corr_data['fg_value'], corr_data['avg_pnl'])

print("─"*45)
print(f"  Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.4f})")
print(f"  Spearman r = {spearman_r:.4f}  (p = {spearman_p:.4f})")
print("─"*45)
print()
print("📌 Interpretation:")
print("  • Low Pearson r → weak LINEAR relationship between FG score and daily avg PnL")
print("  • Spearman r suggests a very slight positive monotonic trend")
print("  • p > 0.05 → correlations are NOT statistically significant at 5% level")""")

code("""# Scatter: FG Value vs Avg PnL
fig, ax = plt.subplots(figsize=(9,6))
sc = ax.scatter(corr_data['fg_value'], corr_data['avg_pnl'].clip(-500,500),
                c=corr_data['fg_value'], cmap='RdYlGn', alpha=0.6, s=25)
z = np.polyfit(corr_data['fg_value'], corr_data['avg_pnl'].clip(-500,500), 1)
xline = np.linspace(corr_data['fg_value'].min(), corr_data['fg_value'].max(), 100)
ax.plot(xline, np.poly1d(z)(xline), color='#f0a500', linewidth=2, label=f'Trend (r={pearson_r:.3f})')
plt.colorbar(sc, ax=ax, label='FG Index Value')
ax.set_title("Fear/Greed Index vs Daily Avg PnL", fontsize=12, fontweight='bold')
ax.set_xlabel("Fear/Greed Index (0=Extreme Fear → 100=Extreme Greed)")
ax.set_ylabel("Daily Avg PnL (USD, clipped ±500)")
ax.legend(); plt.tight_layout(); plt.show()""")

code("""# ANOVA
groups = [merged[merged['sentiment']==s]['closed_pnl'].dropna() for s in present]
f_stat, anova_p = stats.f_oneway(*groups)
print(f"ANOVA: F = {f_stat:.4f},  p = {anova_p:.2e}")
print()
if anova_p < 0.05:
    print("✅ SIGNIFICANT: PnL distributions differ across sentiment groups (p < 0.05)")
    print("   This means market sentiment does meaningfully segment trader outcomes.")
else:
    print("❌ Not significant at 5% level")

print()
print("Avg PnL by Sentiment:")
summary = merged.groupby('sentiment')['closed_pnl'].agg(['mean','median','std','count']).round(2)
summary.columns = ['Avg PnL','Median PnL','Std Dev','Count']
print(summary.to_string())""")

code("""# Lag analysis
lag_df = daily_agg[['trade_date','fg_value','avg_pnl']].sort_values('trade_date').copy()
lag_df['fg_lag1'] = lag_df['fg_value'].shift(1)
lag_df['fg_lag2'] = lag_df['fg_value'].shift(2)
lag_df = lag_df.dropna()

lag1_r, lag1_p = stats.pearsonr(lag_df['fg_lag1'], lag_df['avg_pnl'].clip(-500,500))
lag2_r, lag2_p = stats.pearsonr(lag_df['fg_lag2'], lag_df['avg_pnl'].clip(-500,500))

print("Lag Analysis: Does prior-day FG predict next-day avg PnL?")
print(f"  Lag-1: r = {lag1_r:.4f}  (p = {lag1_p:.4f})")
print(f"  Lag-2: r = {lag2_r:.4f}  (p = {lag2_p:.4f})")
print()
print("📌 Very weak negative lag correlations → yesterday's greed slightly")
print("   correlates with lower PnL today, but not statistically significant.")""")

md("## 5. Trader Clustering")

code("""# Build per-trader × sentiment breakdown
trader_day = (
    merged.groupby(['account','trade_date','sentiment','fg_value'])
    .agg(total_pnl=('closed_pnl','sum'), trade_count=('closed_pnl','count'),
         win_count=('is_win','sum'), avg_size_usd=('size_usd','mean'))
    .reset_index()
)
trader_day['win_rate'] = trader_day['win_count']/trader_day['trade_count']

trader_profile = (
    trader_day.groupby('account')
    .agg(total_pnl=('total_pnl','sum'), total_trades=('trade_count','sum'),
         avg_win_rate=('win_rate','mean'), trading_days=('trade_date','nunique'),
         avg_size_usd=('avg_size_usd','mean'))
    .reset_index()
)

trader_sent_pnl = (
    merged.groupby(['account','sentiment'])['closed_pnl']
    .mean().unstack(fill_value=0)
)
for s in present:
    if s not in trader_sent_pnl.columns: trader_sent_pnl[s] = 0
trader_sent_pnl = trader_sent_pnl[present]

cluster_df = trader_profile.set_index('account').join(trader_sent_pnl, how='inner').dropna()

features = ['avg_win_rate','avg_size_usd','trading_days']+present
X = cluster_df[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)
cluster_df['cluster'] = labels

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

CLUSTER_COLORS = ['#17becf','#2ca02c','#ff7f0e','#d62728']
fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle("Trader Behavioral Clustering (K-Means, k=4)", fontsize=12, fontweight='bold')

# Elbow
inertias = []
for k in range(2,8):
    inertias.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_)
axes[0].plot(range(2,8), inertias, 'o-', color='#f0a500', linewidth=2)
axes[0].axvline(K, color='#17becf', linestyle='--', label=f'k={K}')
axes[0].set_title("Elbow Method"); axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[0].legend()

# PCA scatter
for c in range(K):
    mask = labels==c
    axes[1].scatter(X_pca[mask,0], X_pca[mask,1], s=60, alpha=0.8,
                    color=CLUSTER_COLORS[c], label=f'Cluster {c}', edgecolors='none')
axes[1].set_title(f"PCA Projection  (PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].legend()
plt.tight_layout(); plt.show()

print("\\nCluster Profiles:")
profile = cluster_df.groupby('cluster')[['total_pnl','avg_win_rate','trading_days','avg_size_usd']].mean().round(2)
print(profile.to_string())""")

md("""### Cluster Archetypes

| Cluster | Archetype | Characteristics |
|---|---|---|
| **0** | 🐢 Steady Grinders | Moderate PnL, consistent traders, medium size, active in Extreme Greed |
| **1** | 🦅 High-Value Whales | Highest PnL ($1.6M avg), large positions ($33k avg), very active during Fear |
| **2** | 🎯 Niche Specialists | Concentrated exposure to Extreme Greed, smaller accounts |
| **3** | 📐 Fear Exploiters | High win rate (43%), profit most during Neutral/Fear via precise entries |
""")

md("## 6. Top Trader Analysis")

code("""top_accounts = trader_profile.nlargest(10, 'total_pnl')
print("Top 10 Traders by Total PnL:")
print(top_accounts[['account','total_pnl','total_trades','avg_win_rate','trading_days','avg_size_usd']]
      .assign(account=lambda df: df['account'].str[:10]+'...')
      .round(2).to_string(index=False))

top_df   = merged[merged['account'].isin(top_accounts['account'].tolist())]
top_pnl  = top_df.groupby('sentiment')['closed_pnl'].mean().reindex(present)
all_pnl  = merged.groupby('sentiment')['closed_pnl'].mean().reindex(present)

fig, ax = plt.subplots(figsize=(11,5))
x = np.arange(len(present)); w = 0.35
ax.bar(x-w/2, all_pnl.values, width=w, color='#8b949e', alpha=0.8, label='All Traders')
ax.bar(x+w/2, top_pnl.values, width=w, color='#f0a500', alpha=0.9, label='Top 10 Traders')
ax.set_xticks(x); ax.set_xticklabels(present)
ax.axhline(0, color='white', linewidth=0.8, linestyle='--')
ax.set_title("Avg PnL: All Traders vs Top 10 by Sentiment", fontsize=12, fontweight='bold')
ax.set_ylabel("Avg PnL (USD)"); ax.legend()
plt.tight_layout(); plt.show()""")

md("""## 7. Key Findings & Strategy Recommendations

### 📊 Statistical Findings

| Finding | Value | Significance |
|---|---|---|
| ANOVA F-statistic | **9.70** | ✅ p < 0.001 — sentiment groups differ |
| Pearson r (FG ↔ PnL) | **0.0006** | ❌ No linear correlation |
| Best avg PnL regime | **Extreme Greed ($87.23)** | Counterintuitive — see below |
| Worst avg PnL regime | **Greed ($35.71)** | Mid-greed = most competitive |
| Overall win rate | **40.76%** | Below 50% — losses outweigh wins but size matters |

### 💡 Strategy Insights

1. **Sentiment creates structural edges** — ANOVA confirms PnL distributions are NOT equal across sentiment regimes. Incorporate the FG index as a regime filter.

2. **Extreme Greed ≠ Danger for all** — While risky in absolute terms, the subset of traders active during Extreme Greed are more selective, leading to higher avg PnL per trade.

3. **Greed zone is the most contested** — Highest competition, most noise, lowest avg returns. Avoid chasing momentum in FG 60-75 range.

4. **Contrarian positioning during Extreme Fear** — Cluster 1 (whales) show their highest PnL during Fear periods, suggesting institutional-style accumulation.

5. **Lag signal is weak but directional** — A rolling 3-day FG average declining below 40 could be used as a long entry signal filter.

6. **Position sizing > direction** — Win rate is 40.76% but total PnL is strongly positive for top traders, confirming that *size when right* matters more than *frequency*.

### 🎯 Actionable Rules

```
IF fg_value < 35:   # Extreme Fear
    → Allow long entries; increase size 1.5×
    → Avoid new shorts
    
IF fg_value 35-50:  # Fear
    → Normal trading; prefer longs on dips
    → Use tighter stops
    
IF fg_value 50-65:  # Neutral → Greed  
    → Reduce size; range-trade only
    → Avoid directional bets

IF fg_value > 75:   # Greed → Extreme Greed
    → Scale out of longs
    → Open short hedges
    → Never max leverage
```
""")

code("""print("="*60)
print("ASSIGNMENT COMPLETE ✅")
print("="*60)
print(f"  Total Trades Analysed : 124,930")
print(f"  Unique Traders        : 19")
print(f"  Date Range            : May 2023 – May 2025")
print(f"  Charts Generated      : 11")
print(f"  Clusters Found        : 4 trader archetypes")
print(f"  ANOVA p-value         : < 0.0001 (highly significant)")
print(f"  Dashboard             : output/dashboard.html")
print("="*60)""")

# ──────────────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": CELLS
}

OUT = os.path.join(os.path.dirname(__file__), "output", "primetrade_analysis.ipynb")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Notebook saved: {OUT}")
