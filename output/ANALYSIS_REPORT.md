# Bitcoin Market Sentiment × Trader Performance
## Analysis Report — Primetrade.ai Hiring Assignment

**Analyst:** Mahizhan  
**Dataset Period:** May 2023 – May 2025  
**Submitted:** July 2026

---

## Executive Summary

This report investigates how the **Bitcoin Fear/Greed Index** shapes trading outcomes for 19 accounts on the **Hyperliquid DEX** across 124,930 trades. Using statistical tests, time-series analysis, and machine learning clustering, we identify four key findings:

1. **Sentiment regime significantly alters PnL distributions** (ANOVA F=9.70, p<0.001)
2. **Extreme Greed produces the highest average PnL per trade ($87.23)** — counterintuitive but explained by trader selectivity
3. **Linear FG↔PnL correlation is negligible** (r=0.0006) — the relationship is non-linear and regime-dependent
4. **Four distinct trader archetypes emerge**, each performing differently across sentiment regimes

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Total trades | **124,930** |
| Unique trader accounts | **19** |
| Date range | **May 1, 2023 → May 1, 2025** |
| Fear/Greed index coverage | Feb 2018 → May 2025 |
| Overall win rate | **40.76%** |
| Total closed PnL (all traders) | **$7,539,570** |

### Sentiment Distribution of Trades

| Sentiment | Trades | % of Total |
|---|---|---|
| Fear | 36,200 | 29.0% |
| Greed | 34,780 | 27.8% |
| Neutral | 24,481 | 19.6% |
| Extreme Greed | 21,066 | 16.9% |
| Extreme Fear | 8,403 | 6.7% |

> [!NOTE]
> Trades cluster heavily in Fear and Greed regimes — traders are most active when markets have strong directional sentiment.

---

## 2. Sentiment vs PnL Analysis

### 2.1 Average PnL by Sentiment

| Sentiment | Avg PnL | Median PnL | Std Dev | Trade Count |
|---|---|---|---|---|
| **Extreme Fear** | $69.69 | $0.00 | $1,645 | 8,403 |
| **Fear** | $73.78 | $0.00 | $1,148 | 36,200 |
| **Neutral** | $49.16 | $0.00 | $585 | 24,481 |
| **Greed** | $35.71 | $0.00 | $1,206 | 34,780 |
| **Extreme Greed** | **$87.23** | $0.00 | $998 | 21,066 |

**Key Observation:** The median PnL is $0 across all sentiment regimes — most individual trades are flat (position adjustments, partial fills). The average is driven by a small number of large winning trades.

### 2.2 Win Rate by Sentiment

Win rate is uniformly ~40-41% across all sentiment regimes.

> [!IMPORTANT]
> **Sentiment does NOT change how often you win** — it changes **how much you win when you do win**. This is the core insight: sentiment shapes *trade magnitude*, not *trade frequency*.

### 2.3 Long vs Short Behaviour

Traders show a consistent **long bias (~55-60%)** across all sentiment regimes. However:
- **Fear periods** see slightly more long entries (contrarian accumulation)
- **Extreme Greed** sees the highest short ratio (smart money hedging)

---

## 3. Statistical Analysis

### 3.1 Correlation: FG Index vs Daily Avg PnL

| Test | Coefficient | p-value | Interpretation |
|---|---|---|---|
| Pearson r | 0.0006 | 0.9895 | ❌ No linear relationship |
| Spearman r | 0.0571 | 0.2128 | ❌ Not significant (p>0.05) |

**Conclusion:** There is no statistically significant **linear** correlation between the Fear/Greed index score and daily average PnL. The relationship is **non-linear and regime-dependent**.

### 3.2 ANOVA: Do Sentiment Groups Differ?

```
F-statistic = 9.6987
p-value     = 0.0000 (< 0.001)
```

> [!IMPORTANT]
> **ANOVA is highly significant** (p < 0.001). Even though the linear correlation is weak, the sentiment categories **do create meaningfully different PnL distributions**. This confirms sentiment is a useful **regime classifier**, not a linear predictor.

### 3.3 Pairwise t-Tests (p-values)

| | Extreme Fear | Fear | Neutral | Greed | Extreme Greed |
|---|---|---|---|---|---|
| **Extreme Fear** | — | 0.829 | 0.263 | 0.075 | 0.362 |
| **Fear** | 0.829 | — | **0.001** | **0.000** | 0.141 |
| **Neutral** | 0.263 | **0.001** | — | 0.072 | **0.000** |
| **Greed** | 0.075 | **0.000** | 0.072 | — | **0.000** |
| **Extreme Greed** | 0.362 | 0.141 | **0.000** | **0.000** | — |

Significant differences exist between Fear↔Greed, Neutral↔Extreme Greed, and Greed↔Extreme Greed — supporting a 3-tier regime classification.

### 3.4 Lag Analysis

| Lag | Pearson r | p-value |
|---|---|---|
| Yesterday's FG → Today's PnL | -0.0569 | 0.2153 |
| 2-Day Lag | -0.0375 | 0.4138 |

**Finding:** Prior-day sentiment has a weak negative (not statistically significant) lag. A rolling 3-day average FG trend is more reliable than spot values.

---

## 4. Trader Clustering (K-Means, k=4)

### Cluster Profiles

| Cluster | Archetype | Total PnL | Win Rate | Trading Days | Avg Trade Size | Best Sentiment |
|---|---|---|---|---|---|---|
| **0** | 🐢 Steady Grinders | $332,706 | 33% | 91 days | $6,270 | Extreme Greed |
| **1** | 🦅 High-Value Whales | **$1,600,230** | 35% | 24 days | **$33,569** | Fear |
| **2** | 🎯 Niche Specialists | $416,542 | 20% | 28 days | $2,526 | Extreme Greed |
| **3** | 📐 Fear Exploiters | $199,506 | **43%** | 20 days | $8,335 | Fear/Neutral |

### Archetype Deep-Dives

**🦅 Cluster 1 — High-Value Whales** (Most Profitable)
- Trade sizes 5× the average ($33,569 vs $6,270 avg)
- Trade only 24 days but generate $1.6M in PnL
- Peak performance during **Fear** ($626 avg PnL/trade)
- Strategy: **Contrarian accumulation** — buy fear, size up aggressively

**📐 Cluster 3 — Fear Exploiters** (Highest Win Rate 43%)
- Best win rate across all clusters
- Peak PnL during **Neutral and Fear** regimes
- Strategy: **Swing trading with confirmation-based entries**

**🐢 Cluster 0 — Steady Grinders** (Most Active, 91 days)
- Consistent presence throughout the period
- Performs best during **Extreme Greed** ($88 avg PnL/trade)
- Strategy: **Trend following — ride euphoric market momentum**

**🎯 Cluster 2 — Niche Specialists**
- Very high Extreme Greed avg PnL ($8,285/trade)
- Low win rate (20%) but massive wins when correct
- Strategy: **High-conviction, high-risk momentum plays**

---

## 5. Top Trader vs Average Comparison

Top-10 traders outperform the average by **2–5× across all sentiment regimes**:

| Sentiment | All Traders Avg PnL | Top 10 Outperformance |
|---|---|---|
| Extreme Fear | $69.69 | ~5× |
| Fear | $73.78 | ~5.4× |
| Neutral | $49.16 | ~4× |
| Greed | $35.71 | ~4.2× |
| Extreme Greed | $87.23 | ~2.9× |

> [!TIP]
> The largest performance gap between elite and average traders appears during **Fear regimes** — elite traders use fear as opportunity; average traders hesitate or panic-sell.

---

## 6. Key Insights & Strategy Recommendations

### Insight 1: Sentiment as a Regime Filter
The FG index doesn't linearly predict PnL, but it **stratifies risk environments**:
- 🔴 **Defensive Mode** (FG > 80): Reduce size, take profits, hedge
- 🟡 **Normal Mode** (FG 35-75): Standard trading rules  
- 🟢 **Opportunity Mode** (FG < 35): Increase size, accumulate on weakness

### Insight 2: Extreme Greed ≠ Sell Everything
Data shows Extreme Greed produces the **highest avg PnL ($87.23)**. Selective traders who have pre-defined exits outperform. Be in the market but have clear exit rules.

### Insight 3: Win Rate is a Poor Metric — Optimize Expectancy
With only 40.76% win rate, profitability is driven by **asymmetric position sizing**. Winning trades are significantly larger than losing trades. Target reward:risk > 2:1.

### Insight 4: Size Up During Fear (Like the Whales Do)
Cluster 1 shows peak PnL during Fear. The FG index dropping below 35 should trigger a **1.5-2× position size increase**, not a reduction.

### Insight 5: Greed is the Danger Zone
The Greed regime (FG 60-75) produces the **lowest avg PnL ($35.71)** despite high volume. Maximum competition, minimum edge. Reduce size and wait for extreme readings.

### Insight 6: Use Rolling 3-Day FG Average
Single-day FG readings have weak predictive power. Use a rolling 3-day average:
- Declining from 60→40: Accumulation signal
- Rising from 70→80: Caution/exit signal

### Insight 7: Trade Less, Trade Better
Top traders average only 20-24 active days but generate $1.6M+ in PnL. **Selectivity and patience are the key differentiators** over high-frequency grinding.

### Insight 8: Sentiment-Adaptive Position Sizing

```python
def get_position_mode(fg_value):
    if fg_value < 25:
        return {"mode": "AGGRESSIVE_LONG", "size_multiplier": 2.0}
    elif fg_value < 40:
        return {"mode": "OPPORTUNITY",     "size_multiplier": 1.5}
    elif fg_value < 60:
        return {"mode": "NEUTRAL",         "size_multiplier": 1.0}
    elif fg_value < 75:
        return {"mode": "CAUTIOUS",        "size_multiplier": 0.75}
    else:
        return {"mode": "DEFENSIVE",       "size_multiplier": 0.5}
```

---

## 7. Deliverables

| File | Description |
|---|---|
| [analysis.py](file:///Users/mahizhan/Documents/Internshla/primetrade_analysis/analysis.py) | Full Python analysis script |
| [dashboard.py](file:///Users/mahizhan/Documents/Internshla/primetrade_analysis/dashboard.py) | Interactive Plotly dashboard generator |
| [output/dashboard.html](file:///Users/mahizhan/Documents/Internshla/primetrade_analysis/output/dashboard.html) | 🌐 **Interactive HTML dashboard** — open in browser |
| [output/primetrade_analysis.ipynb](file:///Users/mahizhan/Documents/Internshla/primetrade_analysis/output/primetrade_analysis.ipynb) | 📓 **Jupyter Notebook** with full analysis |
| [output/insights.json](file:///Users/mahizhan/Documents/Internshla/primetrade_analysis/output/insights.json) | Machine-readable key metrics |
| output/fig1–fig11.png | 11 publication-quality charts |

---

## 8. Limitations & Future Work

- **Small sample (n=19 traders):** Results may not generalize — a larger cohort would strengthen conclusions
- **Non-linear relationships:** LSTM on FG time series could reveal more complex patterns
- **Leverage data absent:** Risk-adjusted PnL (per unit leverage) would be a stronger metric
- **Survivorship bias:** Only active accounts are visible — liquidated accounts are excluded
- **Cross-asset sensitivity:** FG primarily tracks BTC; altcoin traders may respond differently

---

*Report generated by automated analysis pipeline | Primetrade.ai Assignment | July 2026*
