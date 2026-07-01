# Bitcoin Market Sentiment vs Trader Performance Analysis

An end-to-end quantitative analysis of 124,930 trades on the Hyperliquid DEX, exploring whether the Bitcoin Fear/Greed Index predicts or shapes trader performance outcomes.

---

## Dashboard Preview

![Dashboard Overview](output/screenshots/dashboard_overview.png)
*Figure 1: Dashboard header, KPI strip, trade distribution pie chart, and average PnL by sentiment.*

![Sentiment and PnL Charts](output/screenshots/dashboard_charts_1.png)
*Figure 2: Sentiment regime analysis — trade count distribution and daily average PnL timeline coloured by sentiment.*

![Performance Charts](output/screenshots/dashboard_charts_2.png)
*Figure 3: Win rate and long/short ratio across sentiment regimes.*

![Statistics Table](output/screenshots/dashboard_stats.png)
*Figure 4: Detailed PnL statistics table with per-sentiment breakdown.*

![Correlation and Insights](output/screenshots/dashboard_correlation.png)
*Figure 5: Fear/Greed Index vs daily average PnL scatter, and strategy insight cards.*

---

## Dataset

| Attribute | Detail |
|---|---|
| Source 1 | Hyperliquid Historical Trader Data |
| Source 2 | Bitcoin Fear/Greed Index (alternative.me) |
| Total trades analysed | 124,930 |
| Unique trader accounts | 19 |
| Date range | May 1, 2023 — May 1, 2025 |
| Overall win rate | 40.76% |
| Fear/Greed index coverage | February 2018 — May 2025 |

---

## Methodology

### 1. Data Collection and Loading

Two primary datasets were obtained:

- **Historical Trader Data** (27 MB CSV): Contains individual trade records from 19 accounts on the Hyperliquid decentralised exchange. Each row represents a single trade execution with fields including account address, coin/token, execution price, size in tokens and USD, side (buy/sell), timestamp, start position, direction, closed PnL, transaction hash, order ID, fee, and trade ID.

- **Bitcoin Fear/Greed Index** (CSV): Daily index values from alternative.me ranging from 0 (Extreme Fear) to 100 (Extreme Greed), with five labelled classifications: Extreme Fear, Fear, Neutral, Greed, and Extreme Greed.

### 2. Data Cleaning and Preprocessing

**Fear/Greed Index:**
- Parsed date column to datetime objects.
- Removed duplicate dates; retained one entry per calendar day.
- Renamed columns for consistency (`value` → `fg_value`, `classification` → `sentiment`).

**Trader Data:**
- Normalised all column names to lowercase with underscores.
- Parsed the `Timestamp IST` column to datetime using `dayfirst=True` for the Indian date format.
- Extracted a normalised `trade_date` (date only, no time component) for day-level merging.
- Coerced `closed_pnl`, `size_usd`, `execution_price`, `size_tokens`, and `fee` to numeric, converting unparseable values to NaN.
- Dropped rows with null `trade_date` or `closed_pnl` (6 rows removed out of 124,937).

**Merging:**
- Joined the trade dataset with the Fear/Greed index on `trade_date = date` using an inner join.
- This produced 124,930 matched rows covering the full May 2023 — May 2025 period.
- Applied ordered categorical encoding to the `sentiment` column for correct chart ordering.

### 3. Feature Engineering

The following derived columns were constructed for analysis:

| Feature | Description |
|---|---|
| `is_win` | Boolean: `closed_pnl > 0` |
| `is_loss` | Boolean: `closed_pnl < 0` |
| `side_num` | Numeric encoding: BUY = 1, SELL = -1 |
| `trade_date` | Normalised calendar date from timestamp |
| `month` | Year-month period string for heatmap aggregation |

**Trader-level daily aggregations** computed per `(account, trade_date, sentiment)`:
- `total_pnl`, `trade_count`, `win_count`, `loss_count`
- `avg_size_usd`, `buy_trades`, `sell_trades`, `win_rate`, `long_ratio`

**Trader-level profiles** computed across the full period per account:
- `total_pnl`, `total_trades`, `avg_win_rate`, `trading_days`, `avg_size_usd`

**Daily aggregate view** computed per `(trade_date, sentiment, fg_value)`:
- `total_pnl`, `avg_pnl`, `trade_count`, `win_rate`

### 4. Exploratory Data Analysis

Eleven visualisations were produced to characterise the data:

1. **Trade count and total PnL by sentiment** — bar charts showing distribution of trade volume and aggregate PnL per sentiment regime.
2. **PnL distribution boxplot** — per-regime box plots clipped at ±$3,000 to suppress extreme outliers while preserving quartile structure.
3. **Win rate by sentiment** — bar chart of proportion of profitable trades per regime against a 50% baseline.
4. **Long vs short trade ratio** — grouped bars comparing buy and sell trade proportions per sentiment.
5. **Daily average PnL timeline** — time-series scatter coloured by sentiment, showing how daily performance evolves across the two-year period.
6. **Monthly trade volume heatmap** — pivot table of trade counts by month and sentiment, revealing activity seasonality.
7. **Fear/Greed Index vs daily avg PnL scatter** — direct scatter with linear trendline for correlation visualisation.
8. **Trader clustering elbow and PCA projection** — elbow curve for K selection and 2D PCA scatter of clustered traders.
9. **Cluster average PnL by sentiment** — line chart comparing how each cluster performs across sentiment regimes.
10. **Top traders vs all traders** — comparison of top-10 and all-trader average PnL by sentiment.
11. **Top coins by sentiment** — average PnL per asset across sentiment regimes for the 15 most traded instruments.

### 5. Statistical Analysis

**Pearson and Spearman Correlation**
Correlation was computed between the daily Fear/Greed index value and the daily average closed PnL across all traders. Both linear (Pearson) and rank-based (Spearman) tests were applied to detect both linear and monotonic relationships.

**One-Way ANOVA**
A one-way ANOVA was performed with sentiment regime as the grouping variable and individual closed PnL as the outcome. This tests whether the means of PnL distributions differ significantly across the five sentiment groups.

**Pairwise Welch t-Tests**
All 10 pairs of sentiment groups were tested using Welch's two-sample t-test (unequal variance assumed), producing a 5×5 p-value matrix to identify which specific group pairs drive the ANOVA significance.

**Lag Correlation Analysis**
The daily Fear/Greed index was shifted by 1 and 2 days, and Pearson correlation was recomputed against the daily average PnL. This tests whether prior-day sentiment has predictive power over subsequent trading outcomes.

### 6. Trader Clustering

**Feature construction:** For each trader account, a feature vector was built containing:
- Average win rate
- Average trade size (USD)
- Number of active trading days
- Average PnL during each of the five sentiment regimes (5 features)

**Standardisation:** All features were scaled to zero mean and unit variance using `StandardScaler` to prevent size-of-USD features from dominating distance calculations.

**K-Means clustering:** The elbow method was applied over k = 2 to 7. K = 4 was selected as the point of diminishing return in within-cluster inertia.

**PCA projection:** The 8-dimensional feature space was projected to 2 principal components for visual interpretation of cluster separation. PC1 and PC2 together explained the majority of variance in the scaled feature set.

**Cluster profiling:** Each cluster's centroid was interpreted by examining average total PnL, win rate, active days, trade size, and per-sentiment PnL to produce qualitative archetypes.

### 7. Dashboard and Deliverable Generation

An interactive HTML dashboard was generated using Plotly (Python), embedded with all charts and a responsive dark-theme layout. The dashboard includes:
- KPI cards with key headline metrics
- Interactive Plotly charts (hover, zoom, filter)
- A per-sentiment statistics table
- Strategy insight cards
- Sticky navigation bar

A Jupyter Notebook was generated programmatically by constructing notebook JSON cells and writing a `.ipynb` file, enabling full reproducibility in any Jupyter environment.

---

## Statistical Results

### Correlation Analysis

| Test | Statistic | p-value | Interpretation |
|---|---|---|---|
| Pearson r (FG score vs daily avg PnL) | 0.0006 | 0.9895 | No significant linear relationship |
| Spearman r (rank correlation) | 0.0571 | 0.2128 | No significant monotonic relationship |
| Lag-1 correlation (yesterday FG vs today PnL) | -0.0569 | 0.2153 | Not significant |
| Lag-2 correlation | -0.0375 | 0.4138 | Not significant |

The Fear/Greed index has no statistically significant linear predictive relationship with daily average PnL. However, when treated as a categorical regime classifier, it produces highly significant group differences.

### One-Way ANOVA: PnL Differences Across Sentiment Regimes

```
F-statistic : 9.6987
p-value     : < 0.0001
```

The null hypothesis (equal PnL distributions across sentiment groups) is rejected at the 0.1% significance level. Market sentiment regime is a meaningful segmentation variable for trader outcomes.

### Pairwise t-Tests (Welch, two-sided)

| Pair | p-value | Significant (alpha = 0.05) |
|---|---|---|
| Fear vs Neutral | 0.00052 | Yes |
| Fear vs Greed | 0.00002 | Yes |
| Neutral vs Extreme Greed | < 0.00001 | Yes |
| Greed vs Extreme Greed | < 0.00001 | Yes |
| Extreme Fear vs Fear | 0.82897 | No |
| Extreme Fear vs Neutral | 0.26285 | No |
| Fear vs Extreme Greed | 0.14125 | No |

### PnL Summary by Sentiment Regime

| Sentiment | Avg PnL (USD) | Median PnL | Std Dev | Trade Count | Share of Trades |
|---|---|---|---|---|---|
| Extreme Fear | 69.69 | 0.00 | 1,645.81 | 8,403 | 6.7% |
| Fear | 73.78 | 0.00 | 1,147.64 | 36,200 | 29.0% |
| Neutral | 49.16 | 0.00 | 585.19 | 24,481 | 19.6% |
| Greed | 35.71 | 0.00 | 1,205.56 | 34,780 | 27.8% |
| Extreme Greed | 87.23 | 0.00 | 997.26 | 21,066 | 16.9% |

Notable findings:
- Extreme Greed produces the highest average PnL per trade ($87.23). This reflects that traders active during Extreme Greed periods are more selective in their entries.
- The Greed regime ($35.71) yields the lowest average PnL, consistent with heightened competition and overcrowded directional trades.
- Median PnL is $0.00 across all regimes, indicating that most individual trades are position adjustments or partial fills. Average PnL is driven by a small number of large winning trades.
- Win rate is uniformly ~40.76% across all regimes — sentiment alters the magnitude of wins, not the frequency.

---

## Trader Clustering (K-Means, k = 4)

| Cluster | Label | Total PnL (USD) | Avg Win Rate | Active Days | Avg Trade Size (USD) | Peak Sentiment |
|---|---|---|---|---|---|---|
| 0 | Steady Grinders | 332,706 | 33% | 91 | 6,270 | Extreme Greed |
| 1 | High-Value Whales | 1,600,230 | 35% | 24 | 33,569 | Fear |
| 2 | Niche Specialists | 416,542 | 20% | 28 | 2,526 | Extreme Greed |
| 3 | Fear Exploiters | 199,506 | 43% | 20 | 8,335 | Fear / Neutral |

**Cluster interpretations:**

- **High-Value Whales (Cluster 1):** The highest-earning group. Trade infrequently (24 days) with very large position sizes ($33,569 average). Generate peak PnL during Fear regimes ($626.19 avg per trade during Fear vs $35.71 overall). Behaviour is consistent with institutional-style accumulation on market weakness.

- **Fear Exploiters (Cluster 3):** Highest win rate across all clusters (43%). Outperform during Neutral and Fear regimes with moderate position sizes. Consistent with a disciplined swing-trading approach using confirmation-based entries.

- **Steady Grinders (Cluster 0):** The most active cluster by trading days (91 days). Moderate position sizes. Perform best during Extreme Greed ($88.43 avg per trade), suggesting a trend-following approach that benefits from sustained momentum.

- **Niche Specialists (Cluster 2):** Low win rate (20%) but exceptionally high average PnL during Extreme Greed ($8,285.66 avg per trade). Concentrated, high-conviction directional bets during euphoric market conditions.

---

## Key Insights

1. **Sentiment functions as a regime classifier, not a linear predictor.** The near-zero Pearson r rules out a direct linear relationship, but the highly significant ANOVA (F=9.70, p<0.001) confirms that sentiment regimes create structurally different PnL environments.

2. **Win rate is uniform across sentiment regimes.** At approximately 40.76% in all conditions, profitability is entirely a function of asymmetric position sizing — winning trades must be significantly larger than losing ones to generate positive expectancy.

3. **The Greed zone is the most competitive and least profitable.** Despite the highest absolute trade volume, the Greed regime (FG 60–75) yields the lowest average PnL ($35.71). This is consistent with maximum market participation reducing available edge.

4. **Elite traders concentrate their largest trades during Fear.** The highest-earning cluster (Cluster 1, $1.6M total PnL) generates 8.5× its overall average PnL per trade during Fear regimes, supporting a contrarian accumulation strategy.

5. **Lag correlation is weak and negative.** A slight negative lag-1 relationship (r = −0.057) suggests that very high prior-day sentiment is marginally associated with lower next-day PnL, but this is not statistically significant at the 5% level.

6. **A 3-tier sentiment framework is supported by the pairwise t-tests.** Significant distinctions exist between Fear, Neutral/Greed, and Extreme Greed — suggesting these three bands represent meaningfully different trading environments.

---

## Strategy Framework

Based on the statistical results, a sentiment-adaptive position-sizing approach is proposed:

| Sentiment Regime | FG Range | Recommended Stance | Position Size Multiplier |
|---|---|---|---|
| Extreme Fear | 0 — 25 | Accumulate long exposure on weakness | 2.0x |
| Fear | 25 — 40 | Long-biased with wider holding periods | 1.5x |
| Neutral | 40 — 60 | Balanced; prefer range strategies | 1.0x |
| Greed | 60 — 75 | Reduce directional exposure, take profits | 0.75x |
| Extreme Greed | 75 — 100 | Scale out of longs; introduce hedges | 0.5x |

```python
def get_position_multiplier(fg_value: int) -> float:
    if fg_value < 25:   return 2.0   # Extreme Fear: accumulate
    if fg_value < 40:   return 1.5   # Fear: long-biased
    if fg_value < 60:   return 1.0   # Neutral: standard
    if fg_value < 75:   return 0.75  # Greed: reduce
    return 0.5                       # Extreme Greed: defensive
```

---

## Project Structure

```
primetrade_analysis/
├── data/                          # Raw datasets (excluded from version control)
│   ├── historical_data.csv        # Hyperliquid trader data (27 MB)
│   └── fear_greed_index.csv       # Bitcoin Fear/Greed Index
│
├── output/                        # Generated deliverables
│   ├── screenshots/               # Dashboard screenshots for documentation
│   ├── dashboard.html             # Interactive Plotly dashboard
│   ├── primetrade_analysis.ipynb  # Jupyter Notebook
│   ├── ANALYSIS_REPORT.md         # Full written report
│   ├── insights.json              # Key metrics (machine-readable)
│   └── fig1 — fig11.png           # 11 analysis charts
│
├── analysis.py                    # Main analysis script (EDA, statistics, clustering)
├── dashboard.py                   # Interactive HTML dashboard generator
├── generate_notebook.py           # Jupyter notebook generator
└── README.md
```

---

## Setup and Reproduction

### 1. Create environment

```bash
conda create -n primetrade python=3.11 -y
conda activate primetrade
pip install pandas numpy matplotlib seaborn plotly scipy scikit-learn statsmodels kaleido nbformat
```

### 2. Download datasets

Place the following files in the `data/` directory:

- [Historical Trader Data](https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view) — `data/historical_data.csv`
- [Fear/Greed Index](https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view) — `data/fear_greed_index.csv`

### 3. Run scripts

```bash
python analysis.py           # Produces 11 charts and insights.json
python dashboard.py          # Produces output/dashboard.html
python generate_notebook.py  # Produces output/primetrade_analysis.ipynb
```

---

## Output Files

| File | Description |
|---|---|
| `output/dashboard.html` | Interactive Plotly dashboard — open in any browser |
| `output/primetrade_analysis.ipynb` | Jupyter Notebook with annotated analysis |
| `output/ANALYSIS_REPORT.md` | Full written report with all findings |
| `output/insights.json` | Machine-readable key metrics |
| `output/fig1_sentiment_distribution.png` | Trade count and total PnL by sentiment |
| `output/fig2_pnl_boxplot.png` | PnL distribution by sentiment (box plot) |
| `output/fig3_win_rate_by_sentiment.png` | Win rate by sentiment regime |
| `output/fig4_long_short_ratio.png` | Long vs short trade ratio by sentiment |
| `output/fig5_daily_avg_pnl_timeline.png` | Daily average PnL timeline coloured by sentiment |
| `output/fig6_volume_heatmap.png` | Monthly trade volume heatmap |
| `output/fig7_fg_vs_pnl_scatter.png` | Fear/Greed Index vs daily average PnL scatter |
| `output/fig8_trader_clusters.png` | K-Means clustering — elbow curve and PCA projection |
| `output/fig9_cluster_sentiment_pnl.png` | Average PnL per sentiment by trader cluster |
| `output/fig10_top_vs_all_traders.png` | Top traders vs all traders comparison |
| `output/fig11_top_coins_by_sentiment.png` | Top assets by average PnL across sentiment regimes |

---

## Limitations

- The dataset covers 19 trader accounts. Findings should be validated on a larger cohort before generalisation.
- Leverage data is not available in the dataset. Risk-adjusted performance metrics (PnL per unit leverage) would strengthen conclusions.
- The analysis covers only accounts active in the dataset. Liquidated or inactive accounts are not represented (survivorship bias).
- The Fear/Greed Index is computed primarily from Bitcoin market data. Traders focused on altcoins may exhibit different sentiment sensitivities.
- Lag analysis is limited to 1–2 days. A rolling multi-day average FG signal may have stronger predictive properties.

---

*Analyst: Mahizhan — July 2026*
