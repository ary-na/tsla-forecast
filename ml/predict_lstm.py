"""
TSLA LSTM - Predict (v8)
- Reconstructs prices from predicted returns
- Fetches live sentiment from NewsAPI
- Fetches live macro features
- Confidence bands from return uncertainty
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import date, datetime, timedelta
from dotenv import load_dotenv
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR = "data/tsla_lstm_data"
MODEL_DIR = "models"

EARNINGS_DATES = [
    "2018-02-07",
    "2018-05-02",
    "2018-08-01",
    "2018-10-24",
    "2019-01-30",
    "2019-04-24",
    "2019-07-24",
    "2019-10-23",
    "2020-01-29",
    "2020-04-29",
    "2020-07-22",
    "2020-10-21",
    "2021-01-27",
    "2021-04-26",
    "2021-07-26",
    "2021-10-20",
    "2022-01-26",
    "2022-04-20",
    "2022-07-20",
    "2022-10-19",
    "2023-01-25",
    "2023-04-19",
    "2023-07-19",
    "2023-10-18",
    "2024-01-24",
    "2024-04-23",
    "2024-07-23",
    "2024-10-23",
    "2025-01-29",
    "2025-04-22",
    "2025-07-22",
    "2025-10-21",
    "2026-01-28",
    "2026-04-22",
]
# ──────────────────────────────────────────────────────────────────────────────


def load_artifacts():
    with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(DATA_DIR, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    model = load_model(os.path.join(MODEL_DIR, "best_model.keras"))
    print("✅ Model and scalers loaded")
    print(f"   Features ({meta['n_features']}): {meta['features']}")
    return model, meta, scalers


def fetch_live_sentiment():
    api_key = os.getenv("NEWSAPI_KEY")
    analyzer = SentimentIntensityAnalyzer()

    if not api_key:
        print("  ⚠ No NEWSAPI_KEY — using neutral sentiment")
        return 0.0

    try:
        newsapi = NewsApiClient(api_key=api_key)
        from_date = (datetime.today() - timedelta(days=3)).strftime("%Y-%m-%d")
        articles = newsapi.get_everything(
            q="TSLA OR Tesla",
            from_param=from_date,
            to=date.today().strftime("%Y-%m-%d"),
            language="en",
            sort_by="publishedAt",
            page_size=30,
        )
        scores = []
        for a in articles.get("articles", []):
            text = f"{a.get('title','')}. {a.get('description','')}"
            score = analyzer.polarity_scores(text)["compound"]
            scores.append(score)

        sentiment = float(np.mean(scores)) if scores else 0.0
        print(
            f"  → Live sentiment: {sentiment:.4f} "
            f"({'Positive' if sentiment > 0.05 else 'Negative' if sentiment < -0.05 else 'Neutral'}) "
            f"from {len(scores)} articles"
        )
        return sentiment

    except Exception as e:
        print(f"  ⚠ Sentiment fetch failed: {e} — using neutral")
        return 0.0


def get_earnings_features(index, trading_days):
    earnings = pd.to_datetime(EARNINGS_DATES)
    days_to = []
    is_week = []

    for dt in index:
        future = earnings[earnings >= dt]
        if len(future) == 0:
            days_to.append(60)
        else:
            next_e = future[0]
            td = sum(1 for d in trading_days if dt <= d <= next_e)
            days_to.append(min(td, 60))
        near = any(abs((dt - e).days) <= 3 for e in earnings)
        is_week.append(1 if near else 0)

    return days_to, is_week


def get_recent_data(meta):
    seq_len = meta["sequence_len"]
    features = meta["features"]

    print("  Downloading TSLA + macro data...")
    tsla = yf.download("TSLA", period="180d", auto_adjust=True)
    nasdaq = yf.download("^IXIC", period="180d", auto_adjust=True)
    dxy = yf.download("DX=F", period="180d", auto_adjust=True)
    tnx = yf.download("^TNX", period="180d", auto_adjust=True)
    oil = yf.download("CL=F", period="180d", auto_adjust=True)

    for d in [tsla, nasdaq, dxy, tnx, oil]:
        d.columns = d.columns.get_level_values(0)

    df = tsla.copy()
    df["NASDAQ"] = nasdaq["Close"]
    df["DXY"] = dxy["Close"]
    df["TNX"] = tnx["Close"]
    df["OIL"] = oil["Close"]
    df.dropna(inplace=True)

    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_21"] = df["Close"].rolling(21).mean()
    df["Volatility"] = df["Close"].rolling(10).std()
    df["Return"] = df["Close"].pct_change()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["NASDAQ"] = df["NASDAQ"].pct_change()
    df["DXY"] = df["DXY"].pct_change()
    df["TNX"] = df["TNX"].pct_change()
    df["OIL"] = df["OIL"].pct_change()

    df.dropna(inplace=True)

    # Earnings features
    trading_days = df.index.tolist()
    days_to, is_week = get_earnings_features(df.index, trading_days)
    df["days_to_earnings"] = days_to
    df["is_earnings_week"] = is_week

    # Live sentiment — apply to last 3 days
    sentiment = fetch_live_sentiment()
    df["Sentiment"] = 0.0
    df.iloc[-3:, df.columns.get_loc("Sentiment")] = sentiment

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    last_date = df.index[-1]
    last_known_close = float(df["Close"].values.flatten()[-1])

    print(f"\n  Last trading day : {last_date.date()}")
    print(f"  Last known Close : ${last_known_close:.2f}")

    window = df[features].tail(seq_len)
    return window, last_date, last_known_close, df


def predict_with_uncertainty(model, window_df, scalers, meta):
    features = meta["features"]

    scaled_cols = []
    for col in features:
        scaled_cols.append(scalers[col].transform(window_df[[col]]))

    scaled_window = np.hstack(scaled_cols)
    X = scaled_window[np.newaxis, :, :]

    # Predict returns
    pred_returns = model.predict(X, verbose=0).flatten()

    last_close = meta["last_close"]
    pred_prices = last_close * (1 + pred_returns)

    # Confidence bands — ±1.5% per day, applied in return space
    forecast_len = meta["forecast_len"]
    lower, upper = [], []
    for i in range(forecast_len):
        margin = abs(pred_returns[i]) * 0.5 + 0.015 * (i + 1)
        lo = last_close * (1 + pred_returns[i] - margin)
        hi = last_close * (1 + pred_returns[i] + margin)
        lower.append(lo)
        upper.append(hi)

    print(f"\n  Raw predicted returns: " f"{[f'{r*100:+.2f}%' for r in pred_returns]}")

    return pred_prices, np.array(lower), np.array(upper), pred_returns


def next_trading_days(last_date, n):
    dates = []
    current = last_date
    while len(dates) < n:
        current += timedelta(days=1)
        if current.weekday() < 5:
            dates.append(current)
    return dates


def plot_forecast(
    df, last_date, predictions, lower, upper, returns, forecast_dates, last_close
):
    recent = df["Close"].tail(30)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        "TSLA — 7-Day Forecast (v8: Returns + Sentiment + Earnings)",
        fontsize=13,
        fontweight="bold",
    )

    # ── Price forecast ──
    ax = axes[0]
    ax.plot(
        recent.index,
        recent.values.flatten(),
        color="#1f77b4",
        lw=1.8,
        label="Actual Close",
    )

    bridge_dates = [last_date] + forecast_dates
    bridge_values = [float(recent.values.flatten()[-1])] + list(predictions)

    ax.plot(
        bridge_dates,
        bridge_values,
        color="#ff7f0e",
        lw=2,
        linestyle="--",
        marker="o",
        markersize=7,
        label="Forecast",
    )

    ax.fill_between(
        forecast_dates,
        lower,
        upper,
        color="#ff7f0e",
        alpha=0.15,
        label="Confidence band",
    )

    ax.axvspan(forecast_dates[0], forecast_dates[-1], alpha=0.05, color="#ff7f0e")

    for dt, price in zip(forecast_dates, predictions):
        ax.annotate(
            f"${price:.2f}",
            xy=(dt, price),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#d45f00",
            fontweight="bold",
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # ── Return forecast ──
    ax2 = axes[1]
    colors = ["#2ca02c" if r >= 0 else "#d62728" for r in returns]
    day_labels = [d.strftime("%a %b %d") for d in forecast_dates]
    bars = ax2.bar(day_labels, returns * 100, color=colors, alpha=0.8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Predicted Return (%)")
    ax2.set_title("Predicted Daily Returns from Today's Close")
    ax2.grid(alpha=0.3, axis="y")

    for bar, val in zip(bars, returns * 100):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.1 if val >= 0 else -0.3),
            f"{val:+.2f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("models/tsla_forecast.png", dpi=150)
    plt.show()
    print("  → Chart saved: models/tsla_forecast.png")


def main():
    model, meta, scalers = load_artifacts()

    print("\nFetching recent data...")
    window_df, last_date, last_known_close, df = get_recent_data(meta)

    print("\nRunning prediction...")
    mean_pred, lower, upper, returns = predict_with_uncertainty(
        model, window_df, scalers, meta
    )

    forecast_dates = next_trading_days(last_date, meta["forecast_len"])

    print("\n─── 7-Day Forecast ─────────────────────────────────────────────")
    print(f"  Based on data up to : {last_date.date()}")
    print(f"  Last known Close    : ${last_known_close:.2f}\n")
    print(
        f"  {'Date':<14} {'Forecast':>10} {'Low':>10} {'High':>10}"
        f" {'Return':>8}  {'Δ from today':>12}"
    )
    print(f"  {'-'*68}")
    for dt, price, lo, hi, ret in zip(forecast_dates, mean_pred, lower, upper, returns):
        change = price - last_known_close
        pct = (change / last_known_close) * 100
        arrow = "▲" if change >= 0 else "▼"
        print(
            f"  {dt.strftime('%a %b %d'):<14} ${price:>8.2f}"
            f"   ${lo:>8.2f}   ${hi:>8.2f}"
            f"  {ret*100:>+6.2f}%   {arrow} {pct:>+.1f}%"
        )
    print(f"  {'-'*68}")

    print("\nGenerating chart...")
    plot_forecast(
        df,
        last_date,
        mean_pred,
        lower,
        upper,
        returns,
        forecast_dates,
        last_known_close,
    )


if __name__ == "__main__":
    main()
