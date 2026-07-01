# Bitcoin Market Sentiment × Trader Performance Analysis

> **Primetrade.ai Data Science Hiring Assignment**

Exploring the relationship between the **Bitcoin Fear/Greed Index** and real trader performance on the **Hyperliquid DEX**.

---

## 📊 Key Findings

| Finding | Result |
|---|---|
| ANOVA (sentiment vs PnL) | **F=9.70, p<0.001** — highly significant |
| Linear correlation (FG ↔ PnL) | r=0.0006 — no linear relationship; FG is a **regime classifier** |
| Best avg PnL regime | **Extreme Greed ($87.23/trade)** |
| Danger zone | **Greed (FG 60-75)** — lowest avg PnL ($35.71) |
| Overall win rate | **40.76%** — profitability from asymmetric sizing, not frequency |
| Dataset | 124,930 trades · 19 traders · May 2023 – May 2025 |

---

## 🗂 Project Structure

```
primetrade_analysis/
├── data/                        # Raw datasets (not tracked — download separately)
│   ├── historical_data.csv      # Hyperliquid trader data (27 MB)
│   └── fear_greed_index.csv     # Bitcoin Fear/Greed Index
│
├── output/                      # Generated deliverables
│   ├── dashboard.html           # 🌐 Interactive Plotly dashboard
│   ├── primetrade_analysis.ipynb # 📓 Jupyter Notebook
│   ├── ANALYSIS_REPORT.md       # 📝 Full written report
│   ├── insights.json            # Key metrics (machine-readable)
│   └── fig1–fig11.png           # 11 analysis charts
│
├── analysis.py                  # Main analysis script (EDA + stats + clustering)
├── dashboard.py                 # Interactive HTML dashboard generator
├── generate_notebook.py         # Jupyter notebook generator
└── README.md
```

---

## 🚀 Setup & Run

### 1. Create conda environment
```bash
conda create -n primetrade python=3.11 -y
conda activate primetrade
pip install pandas numpy matplotlib seaborn plotly scipy scikit-learn statsmodels kaleido nbformat
```

### 2. Download datasets
Place these files in the `data/` folder:
- [Historical Trader Data](https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view)
- [Fear/Greed Index](https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view)

### 3. Run analysis
```bash
python analysis.py          # Generates charts + insights.json
python dashboard.py         # Generates interactive HTML dashboard
python generate_notebook.py # Generates Jupyter notebook
```

---

## 💡 Strategy Insights

The analysis reveals a **3-tier sentiment framework** for trading:

| Sentiment | FG Range | Recommended Action | Size Multiplier |
|---|---|---|---|
| 🔴 Extreme Fear | < 25 | Aggressive accumulation | 2.0× |
| 🟠 Fear | 25–40 | Buy dips, hold longer | 1.5× |
| 🟡 Neutral | 40–60 | Range trade, tight stops | 1.0× |
| 🟢 Greed | 60–75 | Take profits, reduce longs | 0.75× |
| 🔵 Extreme Greed | > 75 | Scale out, open hedges | 0.5× |

### Trader Archetypes (K-Means Clustering)
- 🦅 **Whales** — largest positions ($33k avg), peak during Fear, highest PnL ($1.6M)
- 🐢 **Grinders** — most active (91 days), profit during Extreme Greed trends
- 📐 **Fear Exploiters** — highest win rate (43%), swing trade during Fear/Neutral
- 🎯 **Specialists** — low win rate but massive wins during Extreme Greed

---

## 📈 Charts Generated

| Chart | Description |
|---|---|
| fig1 | Trade count & total PnL by sentiment |
| fig2 | PnL distribution boxplot by sentiment |
| fig3 | Win rate by sentiment regime |
| fig4 | Long vs Short ratio by sentiment |
| fig5 | Daily avg PnL timeline coloured by sentiment |
| fig6 | Monthly trade volume heatmap |
| fig7 | Fear/Greed Index vs daily avg PnL scatter |
| fig8 | Trader clustering (elbow + PCA projection) |
| fig9 | Cluster avg PnL by sentiment |
| fig10 | Top traders vs all traders comparison |
| fig11 | Top coins avg PnL by sentiment |

---

*Built for Primetrade.ai Hiring Assignment | Analyst: Mahizhan | July 2026*
