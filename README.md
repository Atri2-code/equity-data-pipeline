# Equity Data Pipeline & Momentum Signal Research

An end-to-end Python pipeline that ingests, cleans, and engineers features from
5 years of daily equity data, constructs a cross-sectional momentum signal, and
evaluates strategy performance using standard systematic research metrics.

Built to demonstrate data preparation workflows relevant to quantitative research
infrastructure — specifically: cleaning raw market data into analysis-ready
datasets, engineering systematic signals, and evaluating them rigorously.

---

## What this does

| Stage | Description |
|---|---|
| **Ingestion** | Downloads 5 years of adjusted daily OHLCV data for 10 tickers via `yfinance` |
| **Cleaning** | Drops tickers with >10% missing data, forward-fills gaps, aligns to trading calendar |
| **Feature engineering** | Computes log returns, 20/60-day rolling mean, volatility, z-score, and momentum |
| **Signal construction** | Cross-sectional momentum: long top 3 tickers by 20-day cumulative return each day |
| **Evaluation** | Sharpe ratio, annualised return, max drawdown, hit rate vs equal-weight benchmark |
| **Export** | Outputs analysis-ready CSVs to `/output` for downstream research consumption |

---

## Tickers covered

`AAPL` `MSFT` `GOOGL` `AMZN` `META` `TSLA` `NVDA` `JPM` `GS` `BAC`

Easily configurable — edit the `TICKERS` list in `pipeline.py`.

---

## How to run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the pipeline**
```bash
python pipeline.py
```

**3. Outputs**

All files saved to `/output`:
- `features.csv` — full feature matrix, long format, one row per ticker per day
- `strategy_returns.csv` — daily strategy vs benchmark returns, cumulative series
- `performance_metrics.csv` — Sharpe ratio, drawdown, hit rate, annualised return

---

## Sample output

```
Fetching price data for 10 tickers (2019-01-01 to 2024-01-01)...
  Raw shape: 1258 rows x 10 tickers

Cleaning price data...
  Clean shape: 1258 rows x 10 tickers
  Date range: 2019-01-02 → 2023-12-29

Engineering features...
  Feature matrix shape: 12340 rows

Constructing momentum signal...
  Long signal triggered on 3774 ticker-day observations

── Performance Summary ──────────────────
  Strategy Sharpe Ratio               0.71
  Benchmark Sharpe Ratio              0.58
  Strategy Annualised Return          9.84%
  Strategy Max Drawdown              -18.3%
  Strategy Hit Rate                   53.2%
  Total Trading Days                  1218
```

---

## Design notes

**Why log returns?** Log returns are time-additive and more statistically
well-behaved than simple returns — standard in systematic research contexts.

**Why forward-fill with a 3-day limit?** Beyond 3 consecutive missing days,
a gap likely indicates a data error or corporate event rather than a vendor
gap — forward-filling indefinitely would corrupt the signal.

**Why cross-sectional momentum?** Cross-sectional signals (rank-based) are
more robust to market regimes than time-series momentum alone. This is the
simplest version; a research extension would add transaction cost modelling,
position sizing, and turnover constraints.

**This is not a trading strategy.** It is a research workflow demonstration.
No transaction costs, slippage, or capacity constraints are modelled.

---

## Extending this pipeline

Possible research extensions:
- Add alternative signals: mean reversion, volatility targeting, carry
- Incorporate fundamental data via `pandas-datareader`
- Add proper walk-forward validation to avoid lookahead bias
- Model transaction costs and turnover constraints
- Port to a backtesting framework such as `zipline-reloaded`

---

## Dependencies

```
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
```

---

## Author

Atrija Haldar
[LinkedIn](https://www.linkedin.com/in/atrija-haldar-196a3b221/)
MSc Engineering, Technology and Business Management — University of Leeds
