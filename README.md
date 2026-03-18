# 📈 TSLA Forecast Bot

An LSTM-based machine learning model that predicts Tesla (TSLA) stock prices 7 days into the future, delivered via a Telegram bot deployed on AWS EC2.

## Features

- **LSTM Neural Network** — Bidirectional LSTM trained on 15 years of TSLA data
- **17 input features** — OHLCV, technical indicators (RSI, MA, Volatility), macro data (NASDAQ, DXY, TNX, Oil), news sentiment, and earnings calendar
- **Live news sentiment** — Fetches recent Tesla headlines via NewsAPI and scores them with VADER
- **Earnings awareness** — Model knows how many days until next TSLA earnings
- **Macro context** — NASDAQ, Dollar Index, 10Y Treasury, and Oil price as daily returns
- **Telegram bot** — Send /forecast from your phone and get a 7-day price forecast instantly
- **Rate limited** — 1 request per 10 minutes per user

## Stack
```
Python 3.11
TensorFlow 2.16
Telegram Bot API
NewsAPI + VADER Sentiment
yfinance
AWS EC2 t2.micro
```

## Project Structure
```
tsla-bot/
├── bot/
│   └── bot.py              # Telegram bot
├── data/
│   ├── tsla_data_prep.py   # Data download + feature engineering
│   └── tsla_lstm_data/     # Processed training data
├── ml/
│   ├── train_lstm.py       # Model training
│   ├── predict_lstm.py     # Local prediction script
│   └── backtest_lstm.py    # Backtesting script
├── models/
│   └── best_model.keras    # Trained model
├── .env                    # API keys (not committed)
├── requirements.txt        # EC2 dependencies
└── requirements-mac.txt    # Mac dependencies
```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/tsla-bot.git
cd tsla-bot
```

### 2. Install dependencies
```bash
# Mac
pip install -r requirements-mac.txt

# EC2 / Linux
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Add your keys:
# TELEGRAM_TOKEN=your_token
# NEWSAPI_KEY=your_key
```

### 4. Train the model
```bash
python data/tsla_data_prep.py
python ml/train_lstm.py
```

### 5. Run the bot
```bash
python bot/bot.py
```

## Model Architecture
```
Bidirectional LSTM (64 units)
→ Dropout (0.2)
→ LSTM (32 units)
→ Dropout (0.2)
→ Dense (32, ReLU)
→ Dense (7, Linear)   # 7-day return forecast
```

- **Loss:** Huber (robust to outliers)
- **Optimizer:** Adam with gradient clipping
- **Target:** 7-day forward returns (not prices)
- **Training data:** 2010–present with 3x recency weighting

## Backtesting Results

Evaluated over 60 trading days:

| Metric | Day 1 | Day 7 |
|--------|-------|-------|
| MAE | ~$8 | ~$16 |
| MAPE | 1.9% | 3.7% |
| Direction accuracy | 51.7% | 35.0% |

## Disclaimer

This project is for educational purposes only. Not financial advice. Past model performance does not guarantee future results.
