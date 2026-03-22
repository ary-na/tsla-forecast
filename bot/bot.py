"""
TSLA LSTM - Telegram Bot (v1)
Open to all users, rate limited to 1 request per 10 minutes per user.
Commands: /start, /forecast, /status, /help
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import date, datetime, timedelta
from collections import defaultdict
from time import time
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR           = "data/tsla_lstm_data"
MODEL_DIR          = "models"
TOKEN              = os.getenv("TELEGRAM_TOKEN")
RATE_LIMIT_SECONDS = 600

EARNINGS_DATES = [
    "2025-01-29", "2025-04-22", "2025-07-22", "2025-10-21",
    "2026-01-28", "2026-04-22", "2026-07-22", "2026-10-21",
]
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

user_last_request = defaultdict(float)


# ── 1. Load model artifacts ───────────────────────────────────────────────────
def load_artifacts():
    with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(DATA_DIR, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    model = load_model(os.path.join(MODEL_DIR, "best_model.keras"))
    return model, meta, scalers


# ── 2. Sentiment ──────────────────────────────────────────────────────────────
def fetch_live_sentiment():
    api_key  = os.getenv("NEWSAPI_KEY")
    analyzer = SentimentIntensityAnalyzer()

    if not api_key:
        return 0.0, 0

    try:
        newsapi   = NewsApiClient(api_key=api_key)
        from_date = (datetime.today() - timedelta(days=3)).strftime("%Y-%m-%d")
        response  = newsapi.get_everything(
            q="TSLA OR Tesla",
            from_param=from_date,
            to=date.today().strftime("%Y-%m-%d"),
            language="en",
            sort_by="publishedAt",
            page_size=30,
        )

        if response.get("status") != "ok":
            logger.warning(f"NewsAPI error: {response.get('message')}")
            return 0.0, 0

        scores = []
        for a in response.get("articles", []):
            text  = f"{a.get('title', '')}. {a.get('description', '')}"
            score = analyzer.polarity_scores(text)["compound"]
            scores.append(score)

        sentiment = float(np.mean(scores)) if scores else 0.0
        return sentiment, len(scores)

    except Exception as e:
        logger.warning(f"Sentiment fetch failed: {e}")
        return 0.0, 0


# ── 3. Earnings features ──────────────────────────────────────────────────────
def get_earnings_features(index, trading_days):
    earnings = pd.to_datetime(EARNINGS_DATES)
    days_to, is_week = [], []

    for dt in index:
        future = earnings[earnings >= dt]
        if len(future) == 0:
            days_to.append(60)
        else:
            td = min(sum(1 for d in trading_days if dt <= d <= future[0]), 60)
            days_to.append(td)
        is_week.append(
            1 if any(abs((dt - e).days) <= 3 for e in earnings) else 0
        )

    return days_to, is_week


# ── 4. Fetch + engineer live data ─────────────────────────────────────────────
def download_dxy(period="180d"):
    """Download DXY with fallback tickers."""
    dxy = yf.download("DX=F", period=period, auto_adjust=True, progress=False)
    if dxy.empty:
        logger.warning("DX=F failed, trying DX-Y.NYB...")
        dxy = yf.download("DX-Y.NYB", period=period, auto_adjust=True, progress=False)
    if dxy.empty:
        logger.warning("Both DXY tickers failed, using UUP as proxy...")
        dxy = yf.download("UUP", period=period, auto_adjust=True, progress=False)
    return dxy


def get_recent_data(meta):
    seq_len  = meta["sequence_len"]
    features = meta["features"]

    tsla   = yf.download("TSLA",  period="180d", auto_adjust=True, progress=False)
    nasdaq = yf.download("^IXIC", period="180d", auto_adjust=True, progress=False)
    dxy    = download_dxy(period="180d")
    tnx    = yf.download("^TNX",  period="180d", auto_adjust=True, progress=False)
    oil    = yf.download("CL=F",  period="180d", auto_adjust=True, progress=False)

    for d in [tsla, nasdaq, dxy, tnx, oil]:
        d.columns = d.columns.get_level_values(0)

    df           = tsla.copy()
    df["NASDAQ"] = nasdaq["Close"]
    df["DXY"]    = dxy["Close"]
    df["TNX"]    = tnx["Close"]
    df["OIL"]    = oil["Close"]
    df.dropna(inplace=True)

    if len(df) == 0:
        raise ValueError(
            "Dataframe is empty after merge — one or more tickers failed to download"
        )

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

    if len(df) == 0:
        raise ValueError(
            "Dataframe is empty after feature engineering — check ticker downloads"
        )

    trading_days           = df.index.tolist()
    days_to, is_week       = get_earnings_features(df.index, trading_days)
    df["days_to_earnings"] = days_to
    df["is_earnings_week"] = is_week

    sentiment, n_articles  = fetch_live_sentiment()
    df["Sentiment"]        = 0.0
    df.iloc[-3:, df.columns.get_loc("Sentiment")] = sentiment

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    last_date  = df.index[-1]
    last_close = float(df["Close"].values.flatten()[-1])
    window     = df[features].tail(seq_len)

    if len(window) < seq_len:
        raise ValueError(
            f"Not enough data: got {len(window)} rows, need {seq_len}"
        )

    return window, last_date, last_close, sentiment, n_articles


# ── 5. Run prediction ─────────────────────────────────────────────────────────
def run_prediction(model, window_df, scalers, meta):
    features = meta["features"]

    scaled_cols = []
    for col in features:
        scaled_cols.append(scalers[col].transform(window_df[[col]]))

    X            = np.hstack(scaled_cols)[np.newaxis, :, :]
    pred_returns = model.predict(X, verbose=0).flatten()
    last_close   = meta["last_close"]
    pred_prices  = last_close * (1 + pred_returns)

    forecast_len = meta["forecast_len"]
    lower, upper = [], []
    for i in range(forecast_len):
        margin = abs(pred_returns[i]) * 0.5 + 0.015 * (i + 1)
        lower.append(last_close * (1 + pred_returns[i] - margin))
        upper.append(last_close * (1 + pred_returns[i] + margin))

    return pred_prices, np.array(lower), np.array(upper), pred_returns


# ── 6. Next trading days ──────────────────────────────────────────────────────
def next_trading_days(last_date, n):
    dates, current = [], last_date
    while len(dates) < n:
        current += timedelta(days=1)
        if current.weekday() < 5:
            dates.append(current)
    return dates


# ── 7. Format Telegram message ────────────────────────────────────────────────
def format_message(last_date, last_close, predictions,
                   lower, upper, returns, forecast_dates,
                   sentiment, n_articles):

    if sentiment > 0.05:
        sent_label = f"🟢 Positive ({sentiment:+.3f})"
    elif sentiment < -0.05:
        sent_label = f"🔴 Negative ({sentiment:+.3f})"
    else:
        sent_label = f"⚪️ Neutral ({sentiment:+.3f})"

    lines = [
        "📈 *TSLA 7-Day Forecast*",
        f"🗓 Based on: {last_date.strftime('%a %b %d %Y')}",
        f"💵 Last close: *${last_close:.2f}*",
        f"📰 Sentiment: {sent_label} ({n_articles} articles)",
        "",
        "```",
        f"{'Date':<13} {'Price':>8} {'Low':>8} {'High':>8} {'Δ':>7}",
        "─" * 48,
    ]

    for dt, price, lo, hi, ret in zip(
        forecast_dates, predictions, lower, upper, returns
    ):
        change = price - last_close
        pct    = (change / last_close) * 100
        arrow  = "▲" if change >= 0 else "▼"
        lines.append(
            f"{dt.strftime('%a %b %d'):<13}"
            f" ${price:>7.2f}"
            f" ${lo:>7.2f}"
            f" ${hi:>7.2f}"
            f" {arrow}{abs(pct):>4.1f}%"
        )

    lines.append("─" * 48)
    lines.append("```")
    lines.append("")
    lines.append("⚠️ _Not financial advice. Model accuracy: ~$8 MAE on Day 1._")

    return "\n".join(lines)


# ── 8. Telegram handlers ──────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *TSLA Forecast Bot*\n\n"
        "Get a 7-day TSLA price forecast powered by an LSTM model.\n\n"
        "*Commands:*\n"
        "/forecast — get 7-day price forecast\n"
        "/status   — check if model is loaded\n"
        "/help     — show this message\n\n"
        "⏳ _Rate limit: 1 forecast per 10 minutes per user_",
        parse_mode="Markdown",
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        model, meta, scalers = load_artifacts()
        await update.message.reply_text(
            f"✅ *Model loaded*\n\n"
            f"Features : {meta['n_features']}\n"
            f"Sequence : {meta['sequence_len']} days\n"
            f"Forecast : {meta['forecast_len']} days\n"
            f"Predicts : {'Returns' if meta.get('predicts_returns') else 'Prices'}",
            parse_mode="Markdown",
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Model error: {e}")


async def forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    now     = time()
    elapsed = now - user_last_request[user_id]

    if elapsed < RATE_LIMIT_SECONDS:
        wait = int(RATE_LIMIT_SECONDS - elapsed)
        await update.message.reply_text(
            f"⏳ Please wait *{wait // 60}m {wait % 60}s* before requesting again.",
            parse_mode="Markdown",
        )
        return

    user_last_request[user_id] = now
    await update.message.reply_text(
        "⏳ Fetching data and running prediction...",
        parse_mode="Markdown"
    )

    try:
        model, meta, scalers = load_artifacts()

        window_df, last_date, last_close, sentiment, n_articles = \
            get_recent_data(meta)

        predictions, lower, upper, returns = run_prediction(
            model, window_df, scalers, meta
        )

        forecast_dates = next_trading_days(last_date, meta["forecast_len"])

        msg = format_message(
            last_date, last_close, predictions,
            lower, upper, returns, forecast_dates,
            sentiment, n_articles,
        )

        await update.message.reply_text(msg, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Forecast error: {e}", exc_info=True)
        await update.message.reply_text(
            f"❌ Error: {e}\n\nPlease try again in a few minutes.",
        )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)


# ── 9. Main ───────────────────────────────────────────────────────────────────
def main():
    if not TOKEN:
        print("❌ TELEGRAM_TOKEN not found in .env")
        sys.exit(1)

    print("Loading model...")
    try:
        load_artifacts()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start",    start))
    app.add_handler(CommandHandler("status",   status))
    app.add_handler(CommandHandler("forecast", forecast))
    app.add_handler(CommandHandler("help",     help_cmd))

    print("✅ Bot is running... press Ctrl+C to stop")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
