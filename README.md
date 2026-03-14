# Equity Research Pipeline

An end-to-end Python pipeline that mirrors the data workflow of a systematic
quantitative research team — from raw market data ingestion through quality
auditing, feature engineering, signal construction, and factor attribution.

Each module is independently runnable and produces analysis-ready outputs
consumed by the next stage.

---

## Pipeline architecture

```
Raw market data (Yahoo Finance)
        │
        ▼
┌─────────────────────┐
│  1. Data Pipeline   │  Ingestion → cleaning → feature engineering
│     pipeline.py     │  → momentum signal → performance evaluation
└────────┬────────────┘
         │  clean prices + feature matrix
         ▼
┌─────────────────────┐
│  2. Data Quality    │  Anomaly detection → audit report
│     auditor.py      │  → transformation log → verified clean dataset
└────────┬────────────┘
         │  audit-logged clean prices
         ▼
┌─────────────────────┐
│  3. Factor          │  Rolling OLS → factor exposures
│     Attribution     │  → return decomposition → interactive dashboard
│     attribution.py  │
└─────────────────────┘
```

---

## Modules

### Module 1 — Data Pipeline (`1_pipeline/pipeline.py`)
- Downloads 5 years of adjusted daily OHLCV data for 10 equity tickers
- Cleans: drops high-missing tickers, forward-fills gaps, aligns to trading calendar
- Engineers: log returns, 20/60-day rolling mean, volatility, z-score, momentum
- Constructs a cross-sectional momentum signal (long top 3 tickers by 20-day return)
- Evaluates: Sharpe ratio, annualised return, max drawdown, hit rate

**Outputs:** `output/features.csv`, `output/strategy_returns.csv`, `output/performance_metrics.csv`

---

### Module 2 — Data Quality Auditor (`2_data_quality/auditor.py`)
Detects five categories of data quality issues before they corrupt a research signal:

| Detector | What it catches |
|---|---|
| Missing days | Gaps beyond vendor tolerance |
| Price spikes | Returns with \|z-score\| > 4 — likely data errors |
| Stale prices | 3+ consecutive identical closes |
| Corporate actions | Overnight gaps > 15% — possible unadjusted splits |
| Volume anomalies | Abnormally low volume — likely feed outages |

Produces a severity-ranked audit report and a cleaned dataset with every
transformation logged for full reproducibility.

**Outputs:** `output/audit/audit_report.csv`, `output/audit/clean_prices.csv`, `output/audit/transformation_log.csv`

---

### Module 3 — Factor Return Attribution (`3_factor_attribution/attribution.py`)
Decomposes daily returns into systematic factor contributions using rolling OLS:

- **Market (beta):** Broad market exposure via SPY proxy
- **Momentum:** Cross-sectional 20-day cumulative return, demeaned
- **Volatility:** Cross-sectional realised volatility, demeaned
- **Alpha:** Return unexplained by the three factors

Rolling 60-day window regression shows how exposures evolve over time.
Interactive Plotly dashboard visualises exposures, contributions, R², and
cumulative alpha vs total return.

**Outputs:** `output/attribution/exposures_[TICKER].csv`, `output/attribution/return_decomposition.csv`, `output/attribution/attribution_dashboard.html`

---

## How to run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run modules in sequence**
```bash
# Stage 1: build feature matrix and evaluate momentum signal
python 1_pipeline/pipeline.py

# Stage 2: audit data quality and produce verified clean dataset
python 2_data_quality/auditor.py

# Stage 3: run factor attribution and generate dashboard
python 3_factor_attribution/attribution.py
```

**3. View the dashboard**

Open `output/attribution/attribution_dashboard.html` in any browser.

---

## Dependencies

```
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.0.0
```

---

## Design notes

**Why log returns throughout?**
Log returns are time-additive and more statistically tractable than simple
returns for rolling calculations. Standard convention in systematic research.

**Why rolling OLS rather than fixed-window?**
Factor exposures are not stationary — a stock's beta to the market changes
across regimes. Rolling regression surfaces these dynamics rather than
assuming constant exposures over a 5-year period.

**Why three factors specifically?**
Market, momentum, and volatility are the three most robust, academically
documented factors accessible from price data alone, requiring no fundamental
or alternative data. A natural extension would add value (P/B ratio) and
quality (ROE) factors from fundamental data sources.

**What this does not model:**
Transaction costs, slippage, position sizing, capacity constraints, or
lookahead bias in signal construction. All performance metrics are therefore
gross and optimistic relative to a live implementation.

---

## Author

Atrija Haldar
[LinkedIn](https://www.linkedin.com/in/atrija-haldar-196a3b221/)
MSc Engineering, Technology and Business Management — University of Leeds
