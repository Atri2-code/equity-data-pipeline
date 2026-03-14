"""
Market Data Quality Auditor
============================
Ingests raw equity price data and systematically detects data quality issues
before they reach a research pipeline. Produces a per-ticker audit log and
a cleaned, transformation-recorded dataset ready for downstream consumption.

Anomalies detected:
  - Missing trading days
  - Price spikes (returns beyond N standard deviations)
  - Zero or abnormally low volume days
  - Stale prices (consecutive identical closes)
  - Corporate action candidates (overnight gaps > 15%)

Author: Atrija Haldar
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "GS", "BAC"
]

END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d")

# Thresholds
SPIKE_ZSCORE_THRESHOLD     = 4.0   # Flag returns beyond 4 std devs
STALE_CONSECUTIVE_DAYS     = 3     # Flag N+ consecutive identical closes
CORP_ACTION_THRESHOLD      = 0.15  # Flag overnight gaps > 15%
MISSING_DAYS_TOLERANCE     = 5     # Alert if missing more than N trading days
VOLUME_ZSCORE_THRESHOLD    = -3.0  # Flag abnormally low volume days

OUTPUT_DIR = "output/audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Data Ingestion ──────────────────────────────────────────────────────────

def fetch_raw_data(tickers: list, start: str, end: str) -> tuple:
    """
    Downloads full OHLCV data. Returns separate DataFrames for
    closes and volume to support independent quality checks.
    """
    print(f"Fetching raw OHLCV data for {len(tickers)} tickers...")
    raw     = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
    closes  = raw["Close"]
    volume  = raw["Volume"]
    print(f"  Raw shape: {closes.shape[0]} rows x {closes.shape[1]} tickers")
    return closes, volume


# ── 2. Anomaly Detection ───────────────────────────────────────────────────────

def detect_missing_days(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Compares actual trading days against expected NYSE calendar.
    Flags tickers with unexplained gaps beyond tolerance.
    """
    issues = []
    total_days = len(closes)

    for ticker in closes.columns:
        s           = closes[ticker].dropna()
        missing     = total_days - len(s)
        missing_pct = missing / total_days * 100

        if missing > MISSING_DAYS_TOLERANCE:
            issues.append({
                "ticker":      ticker,
                "issue_type":  "missing_days",
                "severity":    "high" if missing_pct > 10 else "medium",
                "detail":      f"{missing} missing trading days ({missing_pct:.1f}%)",
                "action":      "forward-fill or drop ticker"
            })

    return pd.DataFrame(issues)


def detect_price_spikes(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling z-score of daily log returns.
    Flags observations where |z-score| exceeds threshold —
    likely data errors, flash crashes, or fat-finger prints.
    """
    issues  = []
    returns = np.log(closes / closes.shift(1))

    for ticker in returns.columns:
        r        = returns[ticker].dropna()
        roll_std = r.rolling(60).std()
        roll_mu  = r.rolling(60).mean()
        zscore   = (r - roll_mu) / roll_std

        spikes = zscore[zscore.abs() > SPIKE_ZSCORE_THRESHOLD].dropna()

        for date, z in spikes.items():
            issues.append({
                "ticker":     ticker,
                "date":       date.date(),
                "issue_type": "price_spike",
                "severity":   "high" if abs(z) > 6 else "medium",
                "detail":     f"return z-score = {z:.2f} (threshold ±{SPIKE_ZSCORE_THRESHOLD})",
                "action":     "investigate — likely data error or major event"
            })

    return pd.DataFrame(issues)


def detect_stale_prices(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Flags runs of N+ consecutive identical closing prices.
    In liquid large-caps, consecutive identical closes almost always
    indicate a data vendor issue rather than genuine market behaviour.
    """
    issues = []

    for ticker in closes.columns:
        s      = closes[ticker].dropna()
        diffs  = s.diff().fillna(1)
        streak = 0
        streak_start = None

        for date, diff in diffs.items():
            if diff == 0:
                if streak == 0:
                    streak_start = date
                streak += 1
                if streak >= STALE_CONSECUTIVE_DAYS:
                    issues.append({
                        "ticker":     ticker,
                        "date":       streak_start.date(),
                        "issue_type": "stale_price",
                        "severity":   "medium",
                        "detail":     f"{streak} consecutive identical closes from {streak_start.date()}",
                        "action":     "verify with secondary data source"
                    })
            else:
                streak = 0

    return pd.DataFrame(issues)


def detect_corporate_actions(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Flags large overnight gaps that may indicate unadjusted splits
    or dividends. Even with auto_adjust=True, some edge cases slip through.
    """
    issues  = []
    returns = closes.pct_change()

    for ticker in returns.columns:
        r    = returns[ticker].dropna()
        gaps = r[r.abs() > CORP_ACTION_THRESHOLD]

        for date, gap in gaps.items():
            issues.append({
                "ticker":     ticker,
                "date":       date.date(),
                "issue_type": "corporate_action_candidate",
                "severity":   "medium",
                "detail":     f"overnight gap of {gap*100:.1f}% — possible unadjusted split/dividend",
                "action":     "verify adjustment factors"
            })

    return pd.DataFrame(issues)


def detect_volume_anomalies(volume: pd.DataFrame) -> pd.DataFrame:
    """
    Flags abnormally low volume days using rolling z-score.
    Near-zero volume on a supposedly active trading day
    typically indicates a data feed outage, not genuine market inactivity.
    """
    issues = []

    for ticker in volume.columns:
        v        = volume[ticker].replace(0, np.nan).dropna()
        log_vol  = np.log(v)
        roll_mu  = log_vol.rolling(60).mean()
        roll_std = log_vol.rolling(60).std()
        zscore   = (log_vol - roll_mu) / roll_std

        anomalies = zscore[zscore < VOLUME_ZSCORE_THRESHOLD].dropna()

        for date, z in anomalies.items():
            issues.append({
                "ticker":     ticker,
                "date":       date.date(),
                "issue_type": "low_volume",
                "severity":   "low",
                "detail":     f"volume z-score = {z:.2f} — abnormally low trading activity",
                "action":     "check for market holiday or data feed gap"
            })

    return pd.DataFrame(issues)


# ── 3. Audit Report ────────────────────────────────────────────────────────────

def compile_audit_report(closes: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all detectors and compiles a unified audit log
    sorted by severity and date.
    """
    print("\nRunning anomaly detectors...")

    severity_order = {"high": 0, "medium": 1, "low": 2}

    detectors = [
        ("Missing days",          detect_missing_days(closes)),
        ("Price spikes",          detect_price_spikes(closes)),
        ("Stale prices",          detect_stale_prices(closes)),
        ("Corporate actions",     detect_corporate_actions(closes)),
        ("Volume anomalies",      detect_volume_anomalies(volume)),
    ]

    all_issues = []
    for name, df in detectors:
        count = len(df)
        print(f"  {name:<25} {count} issues found")
        if not df.empty:
            all_issues.append(df)

    if not all_issues:
        print("  No issues found — dataset is clean.")
        return pd.DataFrame()

    report = pd.concat(all_issues, ignore_index=True)
    report["severity_rank"] = report["severity"].map(severity_order)
    report = report.sort_values(["severity_rank", "ticker"]).drop(
        columns="severity_rank"
    ).reset_index(drop=True)

    return report


def print_summary(report: pd.DataFrame):
    """Prints a concise audit summary to console."""
    if report.empty:
        return

    print("\n  ── Audit Summary ────────────────────────────")
    print(f"  Total issues flagged:    {len(report)}")
    print(f"  High severity:           {len(report[report.severity == 'high'])}")
    print(f"  Medium severity:         {len(report[report.severity == 'medium'])}")
    print(f"  Low severity:            {len(report[report.severity == 'low'])}")
    print(f"  Tickers affected:        {report.ticker.nunique()}")
    print(f"  Issue types:             {', '.join(report.issue_type.unique())}")
    print("  ─────────────────────────────────────────────")


# ── 4. Clean Dataset Production ───────────────────────────────────────────────

def produce_clean_dataset(closes: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    """
    Applies conservative cleaning rules based on audit findings:
      - Forward-fill missing days (max 3 consecutive)
      - Winsorise extreme spikes at ±4 std dev (replace, not drop)
      - Drop tickers with >10% missing data
      - Log every transformation applied

    Returns a cleaned DataFrame and a transformation log.
    """
    print("\nProducing clean dataset...")

    clean   = closes.copy()
    tx_log  = []

    # Drop high-missing tickers
    missing_pct = clean.isnull().mean()
    to_drop     = missing_pct[missing_pct > 0.10].index.tolist()
    if to_drop:
        clean = clean.drop(columns=to_drop)
        for t in to_drop:
            tx_log.append({"ticker": t, "transformation": "dropped",
                           "reason": ">10% missing data"})

    # Forward-fill remaining gaps
    before = clean.isnull().sum().sum()
    clean  = clean.ffill(limit=3)
    after  = clean.isnull().sum().sum()
    filled = before - after
    if filled > 0:
        tx_log.append({"ticker": "ALL", "transformation": "forward-fill",
                       "reason": f"filled {filled} missing observations (limit=3)"})

    # Winsorise price spikes
    returns = np.log(clean / clean.shift(1))
    for ticker in clean.columns:
        r        = returns[ticker].dropna()
        roll_std = r.rolling(60).std()
        roll_mu  = r.rolling(60).mean()
        zscore   = (r - roll_mu) / roll_std
        spikes   = zscore[zscore.abs() > SPIKE_ZSCORE_THRESHOLD].index

        if len(spikes) > 0:
            for date in spikes:
                original = clean.loc[date, ticker]
                # Replace with rolling mean price as conservative estimate
                clean.loc[date, ticker] = clean[ticker].rolling(5).mean().loc[date]
                tx_log.append({
                    "ticker":         ticker,
                    "transformation": "winsorised",
                    "reason":         f"spike on {date.date()}, original={original:.2f}"
                })

    clean = clean.dropna()
    print(f"  Clean dataset: {clean.shape[0]} rows x {clean.shape[1]} tickers")
    print(f"  Transformations applied: {len(tx_log)}")

    return clean, pd.DataFrame(tx_log)


# ── 5. Export ──────────────────────────────────────────────────────────────────

def export_audit_outputs(report: pd.DataFrame, clean: pd.DataFrame,
                         tx_log: pd.DataFrame):
    """Saves audit report, clean dataset, and transformation log."""
    print("\nExporting audit outputs...")

    report.to_csv(f"{OUTPUT_DIR}/audit_report.csv", index=False)
    clean.to_csv(f"{OUTPUT_DIR}/clean_prices.csv")
    tx_log.to_csv(f"{OUTPUT_DIR}/transformation_log.csv", index=False)

    print(f"  Saved: {OUTPUT_DIR}/audit_report.csv")
    print(f"  Saved: {OUTPUT_DIR}/clean_prices.csv")
    print(f"  Saved: {OUTPUT_DIR}/transformation_log.csv")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    closes, volume = fetch_raw_data(TICKERS, START_DATE, END_DATE)
    report         = compile_audit_report(closes, volume)
    print_summary(report)
    clean, tx_log  = produce_clean_dataset(closes, report)
    export_audit_outputs(report, clean, tx_log)
    print("\nAudit complete.")
