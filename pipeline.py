"""
Equity Data Pipeline
====================
Ingests 5 years of daily OHLCV data for a configurable list of tickers,
cleans and aligns to trading days, engineers momentum features, and
exports analysis-ready CSV files for downstream research.

Author: Atrija Haldar
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "GS", "BAC"
]

END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Data Ingestion ──────────────────────────────────────────────────────────

def fetch_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices for all tickers.
    Returns a DataFrame with dates as index, tickers as columns.
    """
    print(f"Fetching price data for {len(tickers)} tickers ({start} to {end})...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    closes = raw["Close"]
    print(f"  Raw shape: {closes.shape[0]} rows x {closes.shape[1]} tickers")
    return closes


# ── 2. Data Cleaning ───────────────────────────────────────────────────────────

def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop tickers missing more than 10% of trading days
    - Forward-fill remaining gaps (e.g. halted stocks, data vendor gaps)
    - Drop any residual NaNs at the start of series
    - Align all series to the common trading calendar
    """
    print("\nCleaning price data...")

    # Drop tickers with >10% missing
    missing_pct = df.isnull().mean()
    dropped = missing_pct[missing_pct > 0.10].index.tolist()
    if dropped:
        print(f"  Dropping tickers with >10% missing data: {dropped}")
    df = df.drop(columns=dropped)

    # Forward-fill gaps (max 3 consecutive days — beyond that likely a data error)
    df = df.ffill(limit=3)

    # Drop rows where ALL tickers are NaN (non-trading days that slipped through)
    df = df.dropna(how="all")

    # Drop leading NaNs per ticker (IPOs mid-period)
    df = df.dropna()

    print(f"  Clean shape: {df.shape[0]} rows x {df.shape[1]} tickers")
    print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ── 3. Feature Engineering ─────────────────────────────────────────────────────

def engineer_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-ticker:
      - Daily log returns
      - 20-day rolling mean return
      - 20-day rolling volatility (std of returns)
      - 20-day z-score of return (normalised momentum signal)
      - 60-day rolling mean and volatility
      - Momentum signal: 20-day cumulative return
    
    Returns a long-format DataFrame suitable for research consumption.
    """
    print("\nEngineering features...")

    returns = np.log(prices / prices.shift(1))

    records = []

    for ticker in prices.columns:
        r = returns[ticker].dropna()
        p = prices[ticker].reindex(r.index)

        df_t = pd.DataFrame(index=r.index)
        df_t["ticker"]         = ticker
        df_t["price"]          = p
        df_t["log_return"]     = r

        # Short-window (20-day) features
        df_t["roll_mean_20"]   = r.rolling(20).mean()
        df_t["roll_vol_20"]    = r.rolling(20).std()
        df_t["zscore_20"]      = (
            (r - df_t["roll_mean_20"]) / df_t["roll_vol_20"]
        )

        # Medium-window (60-day) features
        df_t["roll_mean_60"]   = r.rolling(60).mean()
        df_t["roll_vol_60"]    = r.rolling(60).std()

        # Momentum: 20-day cumulative log return
        df_t["momentum_20"]    = r.rolling(20).sum()

        records.append(df_t)

    features = pd.concat(records).reset_index().rename(columns={"index": "date"})
    features = features.dropna()

    print(f"  Feature matrix shape: {features.shape}")
    return features


# ── 4. Signal Construction ─────────────────────────────────────────────────────

def build_signal(features: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a simple cross-sectional momentum signal.
    
    Logic: On each day, rank tickers by their 20-day momentum.
    Long signal = top 3 tickers by momentum (signal = 1).
    Flat otherwise (signal = 0).
    
    This is the simplest possible systematic signal — it exists
    to demonstrate the research workflow, not to be traded.
    """
    print("\nConstructing momentum signal...")

    features = features.copy()
    features["signal"] = 0

    for date, group in features.groupby("date"):
        if len(group) < 3:
            continue
        top3 = group.nlargest(3, "momentum_20").index
        features.loc[top3, "signal"] = 1

    long_days = features[features["signal"] == 1].shape[0]
    print(f"  Long signal triggered on {long_days} ticker-day observations")
    return features


# ── 5. Strategy Returns ────────────────────────────────────────────────────────

def compute_strategy_returns(features: pd.DataFrame) -> pd.DataFrame:
    """
    Computes next-day return of the signal.
    signal=1 on day T → captures log_return on day T+1.
    Aggregates to daily portfolio return (equal-weighted across longs).
    """
    print("\nComputing strategy returns...")

    features = features.sort_values(["ticker", "date"])
    features["next_return"] = features.groupby("ticker")["log_return"].shift(-1)

    daily = (
        features[features["signal"] == 1]
        .groupby("date")["next_return"]
        .mean()
        .rename("strategy_return")
    )

    benchmark = (
        features.groupby("date")["log_return"]
        .mean()
        .rename("benchmark_return")
    )

    returns_df = pd.concat([daily, benchmark], axis=1).dropna()
    returns_df["strategy_cum"]   = returns_df["strategy_return"].cumsum().apply(np.exp)
    returns_df["benchmark_cum"]  = returns_df["benchmark_return"].cumsum().apply(np.exp)

    print(f"  Strategy period: {returns_df.index[0].date()} → {returns_df.index[-1].date()}")
    return returns_df


# ── 6. Performance Evaluation ──────────────────────────────────────────────────

def evaluate_performance(returns_df: pd.DataFrame) -> dict:
    """
    Computes standard systematic strategy performance metrics.
    """
    print("\nEvaluating performance...")

    s = returns_df["strategy_return"]
    b = returns_df["benchmark_return"]
    trading_days = 252

    def sharpe(r):
        return (r.mean() / r.std()) * np.sqrt(trading_days) if r.std() > 0 else np.nan

    def max_drawdown(r):
        cum = r.cumsum().apply(np.exp)
        roll_max = cum.cummax()
        drawdown = (cum - roll_max) / roll_max
        return drawdown.min()

    def hit_rate(r):
        return (r > 0).mean()

    def annualised_return(r):
        return r.mean() * trading_days

    metrics = {
        "Strategy Sharpe Ratio":       round(sharpe(s), 3),
        "Benchmark Sharpe Ratio":      round(sharpe(b), 3),
        "Strategy Annualised Return":  f"{annualised_return(s)*100:.2f}%",
        "Benchmark Annualised Return": f"{annualised_return(b)*100:.2f}%",
        "Strategy Max Drawdown":       f"{max_drawdown(s)*100:.2f}%",
        "Benchmark Max Drawdown":      f"{max_drawdown(b)*100:.2f}%",
        "Strategy Hit Rate":           f"{hit_rate(s)*100:.1f}%",
        "Total Trading Days":          len(s),
    }

    print("\n  ── Performance Summary ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:<35} {v}")
    print("  ─────────────────────────────────────────")

    return metrics


# ── 7. Export ──────────────────────────────────────────────────────────────────

def export_outputs(features: pd.DataFrame, returns_df: pd.DataFrame, metrics: dict):
    """
    Saves analysis-ready CSVs and a metrics summary to /output.
    """
    print("\nExporting outputs...")

    features.to_csv(f"{OUTPUT_DIR}/features.csv", index=False)
    returns_df.to_csv(f"{OUTPUT_DIR}/strategy_returns.csv")

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["value"])
    metrics_df.to_csv(f"{OUTPUT_DIR}/performance_metrics.csv")

    print(f"  Saved: {OUTPUT_DIR}/features.csv")
    print(f"  Saved: {OUTPUT_DIR}/strategy_returns.csv")
    print(f"  Saved: {OUTPUT_DIR}/performance_metrics.csv")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prices     = fetch_prices(TICKERS, START_DATE, END_DATE)
    prices     = clean_prices(prices)
    features   = engineer_features(prices)
    features   = build_signal(features)
    returns_df = compute_strategy_returns(features)
    metrics    = evaluate_performance(returns_df)
    export_outputs(features, returns_df, metrics)
    print("\nPipeline complete.")
