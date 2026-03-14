"""
Rolling Factor Return Attribution
===================================
Decomposes daily equity returns into contributions from three systematic
factors — momentum, volatility, and market beta — using rolling OLS regression.

Shows how each stock's return can be explained by factor exposures over time,
and isolates the residual (alpha) that factors cannot explain.

Outputs:
  - Rolling factor exposures per ticker
  - Factor return contributions
  - Residual alpha series
  - Interactive HTML dashboard (Plotly)

Author: Atrija Haldar
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("  Note: plotly not installed. CSV outputs will still be produced.")
    print("  Install with: pip install plotly")

# ── Configuration ──────────────────────────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "GS", "BAC"
]

MARKET_PROXY  = "SPY"     # Broad market factor proxy
ROLLING_WINDOW = 60       # Days for rolling OLS regression
MIN_PERIODS    = 30       # Minimum observations to compute regression

END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d")

OUTPUT_DIR = "output/attribution"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Data Ingestion ──────────────────────────────────────────────────────────

def fetch_data(tickers: list, market_proxy: str,
               start: str, end: str) -> tuple:
    """
    Downloads adjusted closes for all tickers plus the market proxy.
    Returns log returns for stocks and market separately.
    """
    print(f"Fetching data for {len(tickers)} tickers + market proxy ({market_proxy})...")

    all_tickers = tickers + [market_proxy]
    raw         = yf.download(all_tickers, start=start, end=end,
                              auto_adjust=True, progress=False)
    closes      = raw["Close"].ffill().dropna()

    log_returns = np.log(closes / closes.shift(1)).dropna()

    market_returns = log_returns[market_proxy]
    stock_returns  = log_returns[tickers]

    print(f"  Shape: {stock_returns.shape[0]} days x {stock_returns.shape[1]} stocks")
    return stock_returns, market_returns


# ── 2. Factor Construction ─────────────────────────────────────────────────────

def build_factors(stock_returns: pd.DataFrame,
                  market_returns: pd.Series) -> pd.DataFrame:
    """
    Constructs three systematic factors:

    1. Market (beta): The broad market return — captures systematic
       risk exposure common to all equities.

    2. Momentum: Cross-sectional 20-day cumulative return, demeaned.
       Positive value = stock has outperformed peers recently.

    3. Volatility: Rolling 20-day realised volatility, demeaned.
       Positive value = stock is more volatile than peers.

    All factors are standardised (z-scored) to make exposures comparable.
    """
    print("\nConstructing systematic factors...")

    factors = pd.DataFrame(index=stock_returns.index)

    # Factor 1: Market
    factors["market"] = market_returns.reindex(stock_returns.index)

    # Factor 2: Cross-sectional momentum (demeaned, then take cross-sectional mean)
    momentum_raw = stock_returns.rolling(20).sum()
    factors["momentum"] = (
        momentum_raw
        .sub(momentum_raw.mean(axis=1), axis=0)   # demean cross-sectionally
        .mean(axis=1)                               # single factor series
    )

    # Factor 3: Cross-sectional volatility (demeaned)
    vol_raw = stock_returns.rolling(20).std()
    factors["volatility"] = (
        vol_raw
        .sub(vol_raw.mean(axis=1), axis=0)
        .mean(axis=1)
    )

    # Standardise all factors
    for col in factors.columns:
        mu  = factors[col].rolling(60).mean()
        std = factors[col].rolling(60).std()
        factors[col] = (factors[col] - mu) / std

    factors = factors.dropna()
    print(f"  Factors constructed: {list(factors.columns)}")
    print(f"  Factor matrix shape: {factors.shape}")
    return factors


# ── 3. Rolling OLS Regression ──────────────────────────────────────────────────

def rolling_ols(y: pd.Series, X: pd.DataFrame,
                window: int, min_periods: int) -> pd.DataFrame:
    """
    Performs rolling OLS regression of stock returns on factor matrix.
    Returns a DataFrame of rolling betas (factor exposures) over time.

    For each window ending at date t:
        R_stock(t) = β_market * F_market(t)
                   + β_momentum * F_momentum(t)
                   + β_volatility * F_volatility(t)
                   + α + ε
    """
    aligned = X.copy()
    aligned["__y__"] = y
    aligned = aligned.dropna()

    betas  = []
    dates  = []
    alphas = []
    r2s    = []

    idx = aligned.index
    for i in range(len(idx)):
        start_i = max(0, i - window + 1)
        slice_  = aligned.iloc[start_i: i + 1]

        if len(slice_) < min_periods:
            continue

        Y_slice = slice_["__y__"].values
        X_slice = np.column_stack([
            np.ones(len(slice_)),
            slice_[X.columns].values
        ])

        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(X_slice, Y_slice, rcond=None)
        except np.linalg.LinAlgError:
            continue

        alpha   = coeffs[0]
        beta    = coeffs[1:]
        y_hat   = X_slice @ coeffs
        ss_res  = np.sum((Y_slice - y_hat) ** 2)
        ss_tot  = np.sum((Y_slice - Y_slice.mean()) ** 2)
        r2      = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        betas.append(beta)
        alphas.append(alpha)
        r2s.append(r2)
        dates.append(idx[i])

    result = pd.DataFrame(
        betas,
        index=dates,
        columns=X.columns
    )
    result["alpha"] = alphas
    result["r2"]    = r2s
    return result


def run_factor_attribution(stock_returns: pd.DataFrame,
                           factors: pd.DataFrame) -> dict:
    """
    Runs rolling OLS for each ticker.
    Returns a dict of {ticker: DataFrame of rolling exposures}.
    """
    print("\nRunning rolling OLS attribution...")

    aligned_factors = factors.reindex(stock_returns.index).dropna()
    common_idx      = stock_returns.index.intersection(aligned_factors.index)

    results = {}
    for ticker in stock_returns.columns:
        y       = stock_returns.loc[common_idx, ticker]
        X       = aligned_factors.loc[common_idx]
        result  = rolling_ols(y, X, ROLLING_WINDOW, MIN_PERIODS)
        results[ticker] = result
        print(f"  {ticker}: {len(result)} observations with valid exposures")

    return results


# ── 4. Return Decomposition ────────────────────────────────────────────────────

def decompose_returns(stock_returns: pd.DataFrame,
                      factors: pd.DataFrame,
                      attribution: dict) -> pd.DataFrame:
    """
    Decomposes each day's return into:
      - Market contribution:    β_market × F_market
      - Momentum contribution:  β_momentum × F_momentum
      - Volatility contribution: β_volatility × F_volatility
      - Alpha:                  intercept
      - Residual:               unexplained return

    Returns long-format DataFrame for all tickers.
    """
    print("\nDecomposing returns into factor contributions...")

    records = []
    for ticker, exposures in attribution.items():
        r       = stock_returns[ticker].reindex(exposures.index)
        f       = factors.reindex(exposures.index)

        for factor in ["market", "momentum", "volatility"]:
            if factor in exposures.columns and factor in f.columns:
                contribution = exposures[factor] * f[factor]
            else:
                contribution = pd.Series(0, index=exposures.index)

            df_contrib = pd.DataFrame({
                "date":         exposures.index,
                "ticker":       ticker,
                "factor":       factor,
                "exposure":     exposures.get(factor, pd.Series()),
                "contribution": contribution.values,
            })
            records.append(df_contrib)

        # Alpha and residual
        explained = sum(
            exposures.get(f_name, pd.Series(0, index=exposures.index)) *
            factors.reindex(exposures.index).get(f_name,
                pd.Series(0, index=exposures.index))
            for f_name in ["market", "momentum", "volatility"]
        )
        residual = r - explained - exposures.get("alpha",
                                                  pd.Series(0, index=exposures.index))

        for label, series in [("alpha", exposures.get("alpha")),
                               ("residual", residual)]:
            if series is not None:
                records.append(pd.DataFrame({
                    "date":         exposures.index,
                    "ticker":       ticker,
                    "factor":       label,
                    "exposure":     np.nan,
                    "contribution": series.values,
                }))

    decomp = pd.concat(records, ignore_index=True)
    print(f"  Decomposition shape: {decomp.shape}")
    return decomp


# ── 5. Dashboard ───────────────────────────────────────────────────────────────

def build_dashboard(attribution: dict, decomposition: pd.DataFrame,
                    stock_returns: pd.DataFrame):
    """
    Builds an interactive Plotly HTML dashboard showing:
      - Rolling factor exposures over time per ticker
      - Factor contribution stacked area chart
      - R² (explained variance) over time
      - Cumulative alpha vs total return
    """
    if not PLOTLY_AVAILABLE:
        print("\n  Skipping dashboard — plotly not installed.")
        return

    print("\nBuilding interactive dashboard...")

    colors = {
        "market":     "#378ADD",
        "momentum":   "#1D9E75",
        "volatility": "#EF9F27",
        "alpha":      "#7F77DD",
        "residual":   "#888780",
    }

    # Use first ticker as default view
    ticker    = list(attribution.keys())[0]
    exposures = attribution[ticker]
    decomp_t  = decomposition[decomposition.ticker == ticker]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            f"Rolling Factor Exposures — {ticker}",
            "R² (Explained Variance)",
            "Factor Return Contributions",
            "Cumulative Alpha vs Total Return",
            "Cross-Sectional Average Exposures",
            "Return Decomposition Breakdown"
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    # 1. Rolling exposures
    for factor in ["market", "momentum", "volatility"]:
        if factor in exposures.columns:
            fig.add_trace(go.Scatter(
                x=exposures.index, y=exposures[factor],
                name=factor, line=dict(color=colors[factor], width=1.5),
                legendgroup=factor
            ), row=1, col=1)

    # 2. R²
    if "r2" in exposures.columns:
        fig.add_trace(go.Scatter(
            x=exposures.index, y=exposures["r2"],
            name="R²", line=dict(color="#D85A30", width=1.5),
            fill="tozeroy", fillcolor="rgba(216,90,48,0.1)",
            showlegend=False
        ), row=1, col=2)

    # 3. Factor contributions (stacked area)
    for factor in ["market", "momentum", "volatility", "alpha"]:
        contrib = decomp_t[decomp_t.factor == factor]
        if not contrib.empty:
            fig.add_trace(go.Scatter(
                x=contrib.date, y=contrib.contribution,
                name=f"{factor} contribution",
                stackgroup="one",
                line=dict(color=colors.get(factor, "#888780"), width=0.5),
                legendgroup=f"contrib_{factor}"
            ), row=2, col=1)

    # 4. Cumulative alpha vs total return
    alpha_series = decomp_t[decomp_t.factor == "alpha"].set_index("date")["contribution"]
    total_return = stock_returns[ticker].reindex(alpha_series.index)

    cum_alpha  = alpha_series.cumsum()
    cum_total  = total_return.cumsum()

    fig.add_trace(go.Scatter(
        x=cum_total.index, y=cum_total,
        name="Total return", line=dict(color="#185FA5", width=1.5),
        showlegend=True
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=cum_alpha.index, y=cum_alpha,
        name="Cumulative alpha", line=dict(color="#7F77DD", width=1.5),
        showlegend=True
    ), row=2, col=2)

    # 5. Cross-sectional mean exposures
    all_exposures = pd.concat(attribution.values(), keys=attribution.keys())
    for factor in ["market", "momentum", "volatility"]:
        mean_exp = all_exposures.groupby(level=1)[factor].mean() \
            if factor in all_exposures.columns else pd.Series()
        if not mean_exp.empty:
            fig.add_trace(go.Scatter(
                x=mean_exp.index, y=mean_exp,
                name=f"avg {factor}", line=dict(color=colors[factor],
                                                 width=1, dash="dot"),
                showlegend=False
            ), row=3, col=1)

    # 6. Decomposition bar (last 60 days)
    recent = decomp_t[decomp_t.factor.isin(
        ["market", "momentum", "volatility", "alpha"])].tail(60 * 4)
    for factor in ["market", "momentum", "volatility", "alpha"]:
        f_data = recent[recent.factor == factor]
        if not f_data.empty:
            fig.add_trace(go.Bar(
                x=f_data.date, y=f_data.contribution,
                name=factor, marker_color=colors.get(factor),
                legendgroup=f"bar_{factor}", showlegend=False
            ), row=3, col=2)

    fig.update_layout(
        title=dict(
            text="Equity Factor Return Attribution Dashboard",
            font=dict(size=16)
        ),
        height=1000,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        barmode="relative",
        font=dict(size=11)
    )

    output_path = f"{OUTPUT_DIR}/attribution_dashboard.html"
    fig.write_html(output_path)
    print(f"  Dashboard saved: {output_path}")


# ── 6. Export ──────────────────────────────────────────────────────────────────

def export_attribution_outputs(attribution: dict,
                                decomposition: pd.DataFrame):
    """Saves per-ticker exposure CSVs and full decomposition."""
    print("\nExporting attribution outputs...")

    for ticker, exposures in attribution.items():
        path = f"{OUTPUT_DIR}/exposures_{ticker}.csv"
        exposures.to_csv(path)

    decomposition.to_csv(f"{OUTPUT_DIR}/return_decomposition.csv", index=False)

    # Summary: mean exposures across all tickers
    factor_cols = ["market", "momentum", "volatility", "alpha", "r2"]
    all_exp     = pd.concat(attribution.values())
    summary     = all_exp[[c for c in factor_cols if c in all_exp.columns]].describe()
    summary.to_csv(f"{OUTPUT_DIR}/exposure_summary.csv")

    print(f"  Saved: per-ticker exposure CSVs ({len(attribution)} files)")
    print(f"  Saved: {OUTPUT_DIR}/return_decomposition.csv")
    print(f"  Saved: {OUTPUT_DIR}/exposure_summary.csv")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    stock_returns, market_returns = fetch_data(TICKERS, MARKET_PROXY,
                                               START_DATE, END_DATE)
    factors     = build_factors(stock_returns, market_returns)
    attribution = run_factor_attribution(stock_returns, factors)
    decomp      = decompose_returns(stock_returns, factors, attribution)
    build_dashboard(attribution, decomp, stock_returns)
    export_attribution_outputs(attribution, decomp)
    print("\nFactor attribution complete.")
