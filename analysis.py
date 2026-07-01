"""
Primetrade.ai Assignment — Bitcoin Sentiment vs Trader Performance Analysis
Author: Mahizhan
Date: July 2026
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT, exist_ok=True)

DATA = os.path.join(os.path.dirname(__file__), "data")

# ── Color palette ─────────────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    "Extreme Fear": "#d62728",
    "Fear":         "#ff7f0e",
    "Neutral":      "#bcbd22",
    "Greed":        "#2ca02c",
    "Extreme Greed":"#17becf",
}
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "figure.dpi":       150,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("1. LOADING DATASETS")
print("="*60)

trades = pd.read_csv(os.path.join(DATA, "historical_data.csv"))
fg     = pd.read_csv(os.path.join(DATA, "fear_greed_index.csv"))

print(f"  Trades shape    : {trades.shape}")
print(f"  Fear/Greed shape: {fg.shape}")
print(f"\nTrades columns:\n  {list(trades.columns)}")
print(f"\nFear/Greed columns:\n  {list(fg.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. CLEAN & PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. CLEANING & PREPROCESSING")
print("="*60)

# ── Fear/Greed index ──────────────────────────────────────────────────────────
fg['date'] = pd.to_datetime(fg['date'])
fg = fg.rename(columns={'value': 'fg_value', 'classification': 'sentiment'})
fg = fg[['date', 'fg_value', 'sentiment']].drop_duplicates('date').sort_values('date')
print(f"  FG date range: {fg['date'].min().date()} → {fg['date'].max().date()}")
print(f"  Sentiment distribution:\n{fg['sentiment'].value_counts()}")

# ── Trades ────────────────────────────────────────────────────────────────────
# Normalise column names
trades.columns = trades.columns.str.strip().str.lower().str.replace(' ', '_')
print(f"\n  Normalised trade columns: {list(trades.columns)}")

# Parse timestamp — handle mixed formats
ts_col = 'timestamp_ist' if 'timestamp_ist' in trades.columns else 'timestamp'
trades['trade_dt'] = pd.to_datetime(trades[ts_col], dayfirst=True, errors='coerce')
trades['trade_date'] = trades['trade_dt'].dt.normalize()

# Numeric coercion
for col in ['closed_pnl', 'size_usd', 'execution_price', 'size_tokens', 'fee']:
    if col in trades.columns:
        trades[col] = pd.to_numeric(trades[col], errors='coerce')

trades = trades.dropna(subset=['trade_date', 'closed_pnl'])
print(f"  Trades after cleaning: {len(trades):,}")
print(f"  Trade date range: {trades['trade_date'].min().date()} → {trades['trade_date'].max().date()}")
print(f"  Unique accounts: {trades['account'].nunique():,}")

# ── Merge on date ─────────────────────────────────────────────────────────────
merged = trades.merge(fg, left_on='trade_date', right_on='date', how='inner')
print(f"\n  Merged rows: {len(merged):,}")
print(f"  Merged date range: {merged['trade_date'].min().date()} → {merged['trade_date'].max().date()}")

# ── Sentiment category ordering ───────────────────────────────────────────────
present = [s for s in SENTIMENT_ORDER if s in merged['sentiment'].unique()]
merged['sentiment'] = pd.Categorical(merged['sentiment'], categories=present, ordered=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. FEATURE ENGINEERING")
print("="*60)

merged['is_win']  = merged['closed_pnl'] > 0
merged['is_loss'] = merged['closed_pnl'] < 0
merged['side_num'] = merged['side'].str.upper().map({'BUY': 1, 'SELL': -1, 'A': 1, 'B': -1}).fillna(0)

# ── Per-trader daily aggregation ──────────────────────────────────────────────
trader_day = (
    merged
    .groupby(['account', 'trade_date', 'sentiment', 'fg_value'])
    .agg(
        total_pnl    = ('closed_pnl', 'sum'),
        trade_count  = ('closed_pnl', 'count'),
        win_count    = ('is_win', 'sum'),
        loss_count   = ('is_loss', 'sum'),
        avg_size_usd = ('size_usd', 'mean'),
        total_size_usd = ('size_usd', 'sum'),
        avg_price    = ('execution_price', 'mean'),
        buy_trades   = ('side_num', lambda x: (x == 1).sum()),
        sell_trades  = ('side_num', lambda x: (x == -1).sum()),
    )
    .reset_index()
)

trader_day['win_rate']   = trader_day['win_count'] / trader_day['trade_count']
trader_day['long_ratio'] = trader_day['buy_trades'] / trader_day['trade_count'].clip(lower=1)

# ── Overall trader profile ────────────────────────────────────────────────────
trader_profile = (
    trader_day
    .groupby('account')
    .agg(
        total_pnl     = ('total_pnl', 'sum'),
        total_trades  = ('trade_count', 'sum'),
        avg_win_rate  = ('win_rate', 'mean'),
        avg_daily_pnl = ('total_pnl', 'mean'),
        trading_days  = ('trade_date', 'nunique'),
        avg_size_usd  = ('avg_size_usd', 'mean'),
    )
    .reset_index()
)

print(f"  Trader profiles: {len(trader_profile):,}")
print(f"  Total trades in merged: {merged['closed_pnl'].count():,}")
print(f"  Overall PnL stats:\n{trader_profile['total_pnl'].describe()}")

# ── Daily aggregate sentiment view ───────────────────────────────────────────
daily_agg = (
    merged
    .groupby(['trade_date', 'sentiment', 'fg_value'])
    .agg(
        total_pnl   = ('closed_pnl', 'sum'),
        avg_pnl     = ('closed_pnl', 'mean'),
        trade_count = ('closed_pnl', 'count'),
        win_rate    = ('is_win', 'mean'),
    )
    .reset_index()
    .sort_values('trade_date')
)

# ══════════════════════════════════════════════════════════════════════════════
# 4. EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. EXPLORATORY DATA ANALYSIS")
print("="*60)

# ── 4a. Sentiment distribution of trades ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Trade Distribution Across Sentiment Regimes", fontsize=14, fontweight='bold', color='#e6edf3')

sent_counts = merged['sentiment'].value_counts().reindex(present)
colors = [SENTIMENT_COLORS[s] for s in sent_counts.index]
axes[0].bar(sent_counts.index, sent_counts.values, color=colors, edgecolor='#30363d', linewidth=0.8)
axes[0].set_title("Number of Trades by Sentiment", color='#e6edf3')
axes[0].set_xlabel("Sentiment")
axes[0].set_ylabel("Number of Trades")
for i, v in enumerate(sent_counts.values):
    axes[0].text(i, v + 100, f'{v:,}', ha='center', fontsize=9, color='#e6edf3')

sent_pnl = merged.groupby('sentiment')['closed_pnl'].sum().reindex(present)
bar_colors = ['#2ca02c' if v >= 0 else '#d62728' for v in sent_pnl.values]
axes[1].bar(sent_pnl.index, sent_pnl.values, color=bar_colors, edgecolor='#30363d', linewidth=0.8)
axes[1].set_title("Total PnL by Sentiment Regime", color='#e6edf3')
axes[1].set_xlabel("Sentiment")
axes[1].set_ylabel("Total Closed PnL (USD)")
axes[1].axhline(0, color='#8b949e', linewidth=0.8)
for i, v in enumerate(sent_pnl.values):
    axes[1].text(i, v + (abs(sent_pnl.values).max() * 0.02), f'${v:,.0f}', ha='center', fontsize=8, color='#e6edf3')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig1_sentiment_distribution.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig1_sentiment_distribution.png")

# ── 4b. PnL distribution per sentiment (box plot) ────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
data_to_plot = [merged[merged['sentiment'] == s]['closed_pnl'].clip(-5000, 5000).dropna().values for s in present]
bp = ax.boxplot(data_to_plot, patch_artist=True, notch=True,
                medianprops=dict(color='white', linewidth=2),
                whiskerprops=dict(color='#8b949e'),
                capprops=dict(color='#8b949e'),
                flierprops=dict(marker='o', markersize=1.5, alpha=0.3))
for patch, s in zip(bp['boxes'], present):
    patch.set_facecolor(SENTIMENT_COLORS[s])
    patch.set_alpha(0.8)
ax.set_xticklabels(present)
ax.set_title("PnL Distribution by Market Sentiment (clipped ±$5,000)", fontsize=13, fontweight='bold')
ax.set_ylabel("Closed PnL (USD)")
ax.set_xlabel("Sentiment Regime")
ax.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig2_pnl_boxplot.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig2_pnl_boxplot.png")

# ── 4c. Win rate by sentiment ─────────────────────────────────────────────────
win_rate_by_sent = merged.groupby('sentiment')['is_win'].mean().reindex(present) * 100
fig, ax = plt.subplots(figsize=(10, 5))
colors_wr = [SENTIMENT_COLORS[s] for s in win_rate_by_sent.index]
bars = ax.bar(win_rate_by_sent.index, win_rate_by_sent.values, color=colors_wr, edgecolor='#30363d', linewidth=0.8)
ax.axhline(50, color='#8b949e', linewidth=1.2, linestyle='--', label='50% baseline')
ax.set_title("Win Rate by Sentiment Regime", fontsize=13, fontweight='bold')
ax.set_ylabel("Win Rate (%)")
ax.set_xlabel("Sentiment")
ax.set_ylim(0, 80)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig3_win_rate_by_sentiment.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig3_win_rate_by_sentiment.png")

# ── 4d. Long vs Short ratio by sentiment ─────────────────────────────────────
buy_ratio  = merged[merged['side'].str.upper() == 'BUY'].groupby('sentiment').size().reindex(present, fill_value=0)
sell_ratio = merged[merged['side'].str.upper() == 'SELL'].groupby('sentiment').size().reindex(present, fill_value=0)
total      = (buy_ratio + sell_ratio).replace(0, 1)

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(present))
w = 0.35
ax.bar(x - w/2, buy_ratio / total * 100,  width=w, label='BUY (Long)',  color='#2ca02c', alpha=0.85)
ax.bar(x + w/2, sell_ratio / total * 100, width=w, label='SELL (Short)', color='#d62728', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(present)
ax.axhline(50, color='#8b949e', linewidth=1, linestyle='--')
ax.set_title("Long vs Short Trade Ratio by Sentiment", fontsize=13, fontweight='bold')
ax.set_ylabel("% of Trades")
ax.set_xlabel("Sentiment")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig4_long_short_ratio.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig4_long_short_ratio.png")

# ── 4e. Daily avg PnL over time coloured by sentiment ─────────────────────────
fig, ax = plt.subplots(figsize=(15, 5))
for sent in present:
    subset = daily_agg[daily_agg['sentiment'] == sent]
    ax.scatter(subset['trade_date'], subset['avg_pnl'].clip(-500, 500),
               color=SENTIMENT_COLORS[sent], s=12, alpha=0.7, label=sent)
ax.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
ax.set_title("Daily Average Trade PnL Over Time (Coloured by Sentiment)", fontsize=13, fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Avg Closed PnL (USD, clipped ±500)")
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig5_daily_avg_pnl_timeline.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig5_daily_avg_pnl_timeline.png")

# ── 4f. Trade volume heatmap (month × sentiment) ─────────────────────────────
merged['month'] = merged['trade_date'].dt.to_period('M').astype(str)
pivot = merged.pivot_table(index='month', columns='sentiment', values='closed_pnl', aggfunc='count', fill_value=0)
fig, ax = plt.subplots(figsize=(12, max(6, len(pivot)*0.4)))
sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.5, linecolor='#30363d',
            cbar_kws={'label': 'Trade Count'}, annot=len(pivot) <= 24, fmt='g')
ax.set_title("Monthly Trade Volume Heatmap by Sentiment", fontsize=13, fontweight='bold')
ax.set_xlabel("Sentiment")
ax.set_ylabel("Month")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig6_volume_heatmap.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig6_volume_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. STATISTICAL ANALYSIS")
print("="*60)

# ── 5a. Pearson correlation: FG value vs daily avg PnL ───────────────────────
corr_data = daily_agg[['fg_value', 'avg_pnl']].dropna()
pearson_r, pearson_p = stats.pearsonr(corr_data['fg_value'], corr_data['avg_pnl'])
spearman_r, spearman_p = stats.spearmanr(corr_data['fg_value'], corr_data['avg_pnl'])
print(f"\n  Pearson  r = {pearson_r:.4f},  p = {pearson_p:.4f}")
print(f"  Spearman r = {spearman_r:.4f},  p = {spearman_p:.4f}")

fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(corr_data['fg_value'], corr_data['avg_pnl'].clip(-500, 500),
                c=corr_data['fg_value'], cmap='RdYlGn', alpha=0.6, s=25, edgecolors='none')
z = np.polyfit(corr_data['fg_value'], corr_data['avg_pnl'].clip(-500, 500), 1)
p_fit = np.poly1d(z)
xline = np.linspace(corr_data['fg_value'].min(), corr_data['fg_value'].max(), 100)
ax.plot(xline, p_fit(xline), color='#f0a500', linewidth=2, label=f'Trend (r={pearson_r:.3f})')
plt.colorbar(sc, ax=ax, label='Fear/Greed Index Value')
ax.set_title("Fear/Greed Index vs Daily Avg Trade PnL", fontsize=13, fontweight='bold')
ax.set_xlabel("Fear/Greed Index (0=Extreme Fear, 100=Extreme Greed)")
ax.set_ylabel("Daily Avg PnL (USD, clipped ±500)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig7_fg_vs_pnl_scatter.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig7_fg_vs_pnl_scatter.png")

# ── 5b. ANOVA: PnL differences across sentiment groups ───────────────────────
groups = [merged[merged['sentiment'] == s]['closed_pnl'].dropna() for s in present]
f_stat, anova_p = stats.f_oneway(*groups)
print(f"\n  ANOVA: F={f_stat:.4f}, p={anova_p:.6f}")

# ── 5c. Pairwise t-test table ─────────────────────────────────────────────────
print("\n  Pairwise t-tests (p-values):")
ttest_matrix = pd.DataFrame(index=present, columns=present, dtype=float)
for s1 in present:
    for s2 in present:
        if s1 == s2:
            ttest_matrix.loc[s1, s2] = np.nan
        else:
            g1 = merged[merged['sentiment'] == s1]['closed_pnl'].dropna()
            g2 = merged[merged['sentiment'] == s2]['closed_pnl'].dropna()
            _, p = stats.ttest_ind(g1, g2, equal_var=False)
            ttest_matrix.loc[s1, s2] = round(p, 5)
print(ttest_matrix.to_string())

# ── 5d. Lag analysis: does yesterday's FG predict today's avg PnL? ───────────
lag_df = daily_agg[['trade_date', 'fg_value', 'avg_pnl']].sort_values('trade_date').copy()
lag_df['fg_lag1'] = lag_df['fg_value'].shift(1)
lag_df['fg_lag2'] = lag_df['fg_value'].shift(2)
lag_df = lag_df.dropna()

lag1_r, lag1_p = stats.pearsonr(lag_df['fg_lag1'], lag_df['avg_pnl'].clip(-500, 500))
lag2_r, lag2_p = stats.pearsonr(lag_df['fg_lag2'], lag_df['avg_pnl'].clip(-500, 500))
print(f"\n  Lag-1 FG → PnL: r={lag1_r:.4f}, p={lag1_p:.4f}")
print(f"  Lag-2 FG → PnL: r={lag2_r:.4f}, p={lag2_p:.4f}")

# ── 5e. Sentiment summary stats table ────────────────────────────────────────
summary_stats = (
    merged.groupby('sentiment')['closed_pnl']
    .agg(['mean', 'median', 'std', 'count'])
    .round(2)
    .rename(columns={'mean': 'Avg PnL', 'median': 'Median PnL', 'std': 'Std Dev', 'count': 'Trades'})
)
print(f"\n  Sentiment PnL Summary:\n{summary_stats.to_string()}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. TRADER CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6. TRADER CLUSTERING")
print("="*60)

# Build per-trader × sentiment breakdown
trader_sent_pnl = (
    merged.groupby(['account', 'sentiment'])['closed_pnl']
    .mean().unstack(fill_value=0)
)

# Align columns
for s in present:
    if s not in trader_sent_pnl.columns:
        trader_sent_pnl[s] = 0
trader_sent_pnl = trader_sent_pnl[present]

# Merge with profile
cluster_df = trader_profile.set_index('account').join(trader_sent_pnl, how='inner').dropna()

features = ['avg_win_rate', 'avg_size_usd', 'trading_days'] + present
X = cluster_df[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
inertias = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Choose k=4
K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)
cluster_df['cluster'] = labels

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

CLUSTER_COLORS = ['#17becf', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Trader Behavioral Clustering", fontsize=14, fontweight='bold')

# Elbow
axes[0].plot(list(K_range), inertias, 'o-', color='#f0a500', linewidth=2, markersize=7)
axes[0].axvline(K, color='#17becf', linestyle='--', linewidth=1.5, label=f'Chosen k={K}')
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters")
axes[0].set_ylabel("Inertia")
axes[0].legend()

# PCA scatter
for c in range(K):
    mask = labels == c
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], s=15, alpha=0.6,
                    color=CLUSTER_COLORS[c], label=f'Cluster {c}')
axes[1].set_title("PCA Projection of Trader Clusters")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig8_trader_clusters.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig8_trader_clusters.png")

# Cluster profiles
cluster_summary = cluster_df.groupby('cluster')[['total_pnl', 'avg_win_rate', 'trading_days', 'avg_size_usd'] + present].mean().round(2)
print(f"\n  Cluster Profiles:\n{cluster_summary.to_string()}")

# ── Cluster radar chart ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
cluster_pnl_by_sent = cluster_df.groupby('cluster')[present].mean()
x = np.arange(len(present))
for c in range(K):
    vals = cluster_pnl_by_sent.loc[c].values if c in cluster_pnl_by_sent.index else np.zeros(len(present))
    ax.plot(x, vals, 'o-', color=CLUSTER_COLORS[c], linewidth=2, label=f'Cluster {c}')
    ax.fill_between(x, 0, vals, color=CLUSTER_COLORS[c], alpha=0.1)
ax.set_xticks(x)
ax.set_xticklabels(present)
ax.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
ax.set_title("Avg PnL per Sentiment by Trader Cluster", fontsize=13, fontweight='bold')
ax.set_xlabel("Sentiment")
ax.set_ylabel("Avg PnL (USD)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig9_cluster_sentiment_pnl.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig9_cluster_sentiment_pnl.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. TOP TRADER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("7. TOP TRADER ANALYSIS")
print("="*60)

top_traders = trader_profile.nlargest(20, 'total_pnl')['account'].tolist()
top_df = merged[merged['account'].isin(top_traders)]

top_pnl_by_sent = top_df.groupby('sentiment')['closed_pnl'].mean().reindex(present)
all_pnl_by_sent = merged.groupby('sentiment')['closed_pnl'].mean().reindex(present)

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(present))
w = 0.35
ax.bar(x - w/2, all_pnl_by_sent.values, width=w, color='#8b949e', alpha=0.8, label='All Traders')
ax.bar(x + w/2, top_pnl_by_sent.values,  width=w, color='#f0a500', alpha=0.9, label='Top 20 Traders')
ax.set_xticks(x)
ax.set_xticklabels(present)
ax.axhline(0, color='white', linewidth=0.8, linestyle='--')
ax.set_title("Avg PnL: All Traders vs Top 20 by Sentiment", fontsize=13, fontweight='bold')
ax.set_ylabel("Avg Closed PnL (USD)")
ax.set_xlabel("Sentiment")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig10_top_vs_all_traders.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig10_top_vs_all_traders.png")

# ── Top coins by sentiment ────────────────────────────────────────────────────
coin_col = 'coin' if 'coin' in merged.columns else 'symbol'
if coin_col in merged.columns:
    top_coins = (merged.groupby([coin_col, 'sentiment'])['closed_pnl']
                 .mean().reset_index()
                 .pivot(index=coin_col, columns='sentiment', values='closed_pnl')
                 .fillna(0))
    top_coins['total'] = top_coins.sum(axis=1)
    top_coins = top_coins.nlargest(15, 'total').drop(columns='total')
    fig, ax = plt.subplots(figsize=(14, 6))
    top_coins.plot(kind='bar', ax=ax, color=[SENTIMENT_COLORS.get(c, '#8b949e') for c in top_coins.columns])
    ax.set_title("Top 15 Coins: Avg PnL per Sentiment", fontsize=13, fontweight='bold')
    ax.set_xlabel("Coin")
    ax.set_ylabel("Avg PnL (USD)")
    ax.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "fig11_top_coins_by_sentiment.png"), bbox_inches='tight')
    plt.close()
    print("  Saved: fig11_top_coins_by_sentiment.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. STRATEGY INSIGHTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("8. STRATEGY INSIGHTS SUMMARY")
print("="*60)

insights = {
    "total_trades": int(merged['closed_pnl'].count()),
    "total_traders": int(merged['account'].nunique()),
    "date_range": f"{merged['trade_date'].min().date()} to {merged['trade_date'].max().date()}",
    "overall_win_rate": float(round(merged['is_win'].mean() * 100, 2)),
    "pearson_r_fg_pnl": round(pearson_r, 4),
    "pearson_p": round(pearson_p, 4),
    "spearman_r": round(spearman_r, 4),
    "anova_f": round(float(f_stat), 4),
    "anova_p": round(float(anova_p), 6),
    "lag1_r": round(lag1_r, 4),
    "lag1_p": round(lag1_p, 4),
    "pnl_by_sentiment": {s: round(float(v), 2) for s, v in all_pnl_by_sent.items()},
    "win_rate_by_sentiment": {s: round(float(v) * 100, 2) for s, v in merged.groupby('sentiment')['is_win'].mean().reindex(present).items()},
    "summary_stats": summary_stats.to_dict(),
}

with open(os.path.join(OUTPUT, "insights.json"), "w") as f:
    json.dump(insights, f, indent=2)

print("\n  Key Metrics:")
for k, v in insights.items():
    if not isinstance(v, dict):
        print(f"    {k}: {v}")
print("\n  Saved: insights.json")
print("\n" + "="*60)
print("  ✅ All figures and insights saved to:", OUTPUT)
print("="*60)
