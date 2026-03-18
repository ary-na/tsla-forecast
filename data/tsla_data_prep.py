"""
TSLA LSTM - Data Prep (v8 fixed)
Fixes:
- DX-Y.NYB → DX=F (yfinance ticker fix)
- NewsAPI date range made safe (20 days back instead of 29)
- NewsAPI errors handled gracefully — never crashes data prep
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pickle
from dotenv import load_dotenv
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TICKER       = "TSLA"
START_DATE   = "2010-01-01"
END_DATE     = date.today().strftime("%Y-%m-%d")
SEQUENCE_LEN = 60
FORECAST_LEN = 7
FEATURES     = [
    "Close", "Open", "High", "Low", "Volume",
    "MA_7", "MA_21", "Volatility", "Return", "RSI",
    "NASDAQ", "DXY", "TNX", "OIL",
    "Sentiment",
    "days_to_earnings",
    "is_earnings_week",
]
TARGET_COL   = "Return"
PRICE_COL    = "Close"
TRAIN_RATIO  = 0.80
OUTPUT_DIR   = "data/tsla_lstm_data"

EARNINGS_DATES = [
    "2018-02-07", "2018-05-02", "2018-08-01", "2018-10-24",
    "2019-01-30", "2019-04-24", "2019-07-24", "2019-10-23",
    "2020-01-29", "2020-04-29", "2020-07-22", "2020-10-21",
    "2021-01-27", "2021-04-26", "2021-07-26", "2021-10-20",
    "2022-01-26", "2022-04-20", "2022-07-20", "2022-10-19",
    "2023-01-25", "2023-04-19", "2023-07-19", "2023-10-18",
    "2024-01-24", "2024-04-23", "2024-07-23", "2024-10-23",
    "2025-01-29", "2025-04-22", "2025-07-22", "2025-10-21",
    "2026-01-28", "2026-04-22",
]
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Download price + macro data ────────────────────────────────────────────
def download_data():
    print("Downloading TSLA + macro data from 2010...")
    tsla   = yf.download(TICKER,  start=START_DATE, end=END_DATE, auto_adjust=True)
    nasdaq = yf.download("^IXIC", start=START_DATE, end=END_DATE, auto_adjust=True)
    dxy    = yf.download("DX=F",  start=START_DATE, end=END_DATE, auto_adjust=True)  # ← fixed
    tnx    = yf.download("^TNX",  start=START_DATE, end=END_DATE, auto_adjust=True)
    oil    = yf.download("CL=F",  start=START_DATE, end=END_DATE, auto_adjust=True)

    for d in [tsla, nasdaq, dxy, tnx, oil]:
        d.columns = d.columns.get_level_values(0)

    df           = tsla.copy()
    df["NASDAQ"] = nasdaq["Close"]
    df["DXY"]    = dxy["Close"]
    df["TNX"]    = tnx["Close"]
    df["OIL"]    = oil["Close"]
    df.dropna(inplace=True)

    print(f"  → {len(df)} trading days")
    return df


# ── 2. Technical indicators ───────────────────────────────────────────────────
def add_technicals(df):
    df = df.copy()
    df["MA_7"]       = df["Close"].rolling(7).mean()
    df["MA_21"]      = df["Close"].rolling(21).mean()
    df["Volatility"] = df["Close"].rolling(10).std()
    df["Return"]     = df["Close"].pct_change()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["NASDAQ"] = df["NASDAQ"].pct_change()
    df["DXY"]    = df["DXY"].pct_change()
    df["TNX"]    = df["TNX"].pct_change()
    df["OIL"]    = df["OIL"].pct_change()

    df.dropna(inplace=True)
    return df


# ── 3. Earnings features ──────────────────────────────────────────────────────
def add_earnings_features(df):
    df = df.copy()
    earnings     = pd.to_datetime(EARNINGS_DATES)
    trading_days = df.index.tolist()
    days_to_next = []
    is_earnings  = []

    for dt in df.index:
        future = earnings[earnings >= dt]
        if len(future) == 0:
            days_to_next.append(60)
        else:
            next_e = future[0]
            td     = sum(1 for d in trading_days if dt <= d <= next_e)
            days_to_next.append(min(td, 60))
        near = any(abs((dt - e).days) <= 3 for e in earnings)
        is_earnings.append(1 if near else 0)

    df["days_to_earnings"] = days_to_next
    df["is_earnings_week"] = is_earnings
    print(f"  → Earnings features added ({sum(is_earnings)} earnings-week days)")
    return df


# ── 4. News sentiment ─────────────────────────────────────────────────────────
def fetch_sentiment(df):
    api_key = os.getenv("NEWSAPI_KEY")

    if not api_key:
        print("  ⚠ No NEWSAPI_KEY — using neutral sentiment")
        df["Sentiment"] = 0.0
        return df

    print("  Fetching TSLA news sentiment...")
    newsapi  = NewsApiClient(api_key=api_key)
    analyzer = SentimentIntensityAnalyzer()

    # Safe window — 20 days back to avoid free tier boundary issues
    from_date = (datetime.today() - timedelta(days=20)).strftime("%Y-%m-%d")
    to_date   = date.today().strftime("%Y-%m-%d")

    try:
        response = newsapi.get_everything(
            q="TSLA OR Tesla",
            from_param=from_date,
            to=to_date,
            language="en",
            sort_by="publishedAt",
            page_size=100,
        )

        # Check for API-level errors returned in response body
        if response.get("status") != "ok":
            print(f"  ⚠ NewsAPI returned error: {response.get('message','unknown')} — using neutral sentiment")
            df["Sentiment"] = 0.0
            return df

        articles = response.get("articles", [])

        scores_by_date = {}
        for article in articles:
            pub   = article.get("publishedAt", "")[:10]
            title = article.get("title") or ""
            desc  = article.get("description") or ""
            text  = f"{title}. {desc}"
            score = analyzer.polarity_scores(text)["compound"]
            scores_by_date.setdefault(pub, []).append(score)

        daily_sentiment = {k: np.mean(v) for k, v in scores_by_date.items()}
        print(f"  → {len(articles)} articles scored across {len(daily_sentiment)} days")

    except Exception as e:
        print(f"  ⚠ NewsAPI error: {e} — using neutral sentiment")
        df["Sentiment"] = 0.0
        return df

    # Map scores to dataframe index
    sentiment_col = []
    for dt in df.index:
        key   = dt.strftime("%Y-%m-%d")
        score = daily_sentiment.get(key, None)
        sentiment_col.append(score)

    sent_series = pd.Series(sentiment_col, index=df.index, dtype=float)
    sent_series = sent_series.ffill(limit=3).fillna(0.0)
    df["Sentiment"] = sent_series.rolling(3, min_periods=1).mean()

    print(f"  → Sentiment range: {df['Sentiment'].min():.3f} → {df['Sentiment'].max():.3f}")
    return df


# ── 5. Build return targets ───────────────────────────────────────────────────
def build_return_targets(df, forecast_len):
    closes  = df["Close"].values.flatten()
    targets = []
    for i in range(len(closes) - forecast_len):
        base    = closes[i]
        fwd     = closes[i + 1 : i + 1 + forecast_len]
        returns = (fwd - base) / base
        targets.append(returns)
    return np.array(targets)


# ── 6. Scale features ─────────────────────────────────────────────────────────
def scale_data(df, feature_cols):
    scalers     = {}
    scaled_cols = []
    for col in feature_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
        scaled_cols.append(scaled)
    return np.hstack(scaled_cols), scalers


# ── 7. Build sequences ────────────────────────────────────────────────────────
def build_sequences_weighted(scaled, targets, seq_len, recent_weight=3):
    X, y         = [], []
    total        = len(scaled)
    recent_start = max(seq_len, total - 500)
    n_targets    = len(targets)

    for i in range(seq_len, min(total, n_targets + seq_len)):
        target_idx = i - seq_len
        if target_idx >= len(targets):
            break
        window = scaled[i - seq_len : i, :]
        target = targets[target_idx]
        X.append(window)
        y.append(target)
        if i >= recent_start:
            for _ in range(recent_weight - 1):
                X.append(window)
                y.append(target)

    return np.array(X), np.array(y)


def split_data(X, y, ratio):
    split   = int(len(X) * ratio)
    X_train = X[:split]
    X_test  = X[split:]
    y_train = y[:split]
    y_test  = y[split:]
    idx     = np.random.permutation(len(X_train))
    return X_train[idx], X_test, y_train[idx], y_test


# ── 8. Plot ───────────────────────────────────────────────────────────────────
def plot_raw(df, out_dir):
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle("TSLA — Data Overview (v8)", fontsize=15, fontweight="bold")

    axes[0].plot(df.index, df["Close"].values.flatten(),
                 color="#1f77b4", lw=1.2, label="TSLA Close")
    axes[0].plot(df.index, df["MA_7"].values.flatten(),
                 color="#ff7f0e", lw=1, linestyle="--", label="MA7")
    axes[0].plot(df.index, df["MA_21"].values.flatten(),
                 color="#2ca02c", lw=1, linestyle="--", label="MA21")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df.index, df["Sentiment"].values.flatten(),
                 color="#9467bd", lw=1, label="Sentiment (VADER 3d avg)")
    axes[1].axhline(0,     color="gray",  linestyle="--", lw=0.8)
    axes[1].axhline(0.05,  color="green", linestyle=":",  lw=0.8)
    axes[1].axhline(-0.05, color="red",   linestyle=":",  lw=0.8)
    axes[1].set_ylabel("Sentiment Score")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    axes[2].plot(df.index, df["days_to_earnings"].values.flatten(),
                 color="#d62728", lw=1, label="Days to Earnings")
    axes[2].set_ylabel("Trading Days")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.3)

    axes[3].plot(df.index, df["RSI"].values.flatten(),
                 color="#e377c2", lw=1)
    axes[3].axhline(70, color="red",   linestyle="--", lw=0.8)
    axes[3].axhline(30, color="green", linestyle="--", lw=0.8)
    axes[3].set_ylabel("RSI")
    axes[3].set_ylim(0, 100)
    axes[3].grid(alpha=0.3)

    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    path = os.path.join(out_dir, "tsla_raw_overview.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Chart saved: {path}")


# ── 9. Main ───────────────────────────────────────────────────────────────────
def main():
    df_raw = download_data()
    df     = add_technicals(df_raw)
    df     = add_earnings_features(df)
    df     = fetch_sentiment(df)

    print(f"\n  → Final dataframe: {len(df)} rows, {len(FEATURES)} features")

    if len(df) == 0:
        print("❌ Dataframe is empty — check download errors above")
        return

    print("\nBuilding return targets...")
    targets = build_return_targets(df, FORECAST_LEN)
    print(f"  → Targets shape: {targets.shape}")
    print(f"  → Return range : {targets.min():.4f} → {targets.max():.4f}")

    scaled, scalers = scale_data(df, FEATURES)
    print(f"  → Scaled shape : {scaled.shape}")

    scaled_trimmed = scaled[:len(targets) + SEQUENCE_LEN]
    X, y = build_sequences_weighted(scaled_trimmed, targets, SEQUENCE_LEN)
    print(f"  → X: {X.shape}  |  y: {y.shape}")

    X_train, X_test, y_train, y_test = split_data(X, y, TRAIN_RATIO)
    print(f"  → Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)

    last_close = float(df["Close"].values.flatten()[-1])

    with open(os.path.join(OUTPUT_DIR, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    df.to_csv(os.path.join(OUTPUT_DIR, "tsla_engineered.csv"))

    meta = {
        "ticker":           TICKER,
        "sequence_len":     SEQUENCE_LEN,
        "forecast_len":     FORECAST_LEN,
        "features":         FEATURES,
        "target_col":       TARGET_COL,
        "price_col":        PRICE_COL,
        "n_features":       len(FEATURES),
        "train_samples":    X_train.shape[0],
        "test_samples":     X_test.shape[0],
        "last_close":       last_close,
        "predicts_returns": True,
    }
    with open(os.path.join(OUTPUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("\n✅ All data saved to:", OUTPUT_DIR)
    plot_raw(df, OUTPUT_DIR)

    print("\n─── Data Summary ────────────────────────────────────")
    print(f"  Date range   : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Features ({len(FEATURES)}) : {FEATURES}")
    print(f"  Target       : 7-day forward returns")
    print(f"  Train/Test   : {X_train.shape[0]} / {X_test.shape[0]}")
    print("─────────────────────────────────────────────────────")
    print("\nNext → run ml/train_lstm.py")


if __name__ == "__main__":
    main()
