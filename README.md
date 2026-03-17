# Tennis Prediction & Trading System

XGBoost match prediction model connected to the Kalshi prediction market API for paper and live trading.

## What it does

Trains on Jeff Sackmann's historical ATP/WTA match data and predicts match win probabilities using 83 features: Elo ratings (overall + surface-specific), ranking tiers, head-to-head records, recent form, serve/return statistics, fatigue, streaks, and clutch metrics.

When a match is coming up, you can pull live Kalshi odds, compare them against the model's probability estimate, and place paper or live trades from the terminal if there's a meaningful edge. Position sizing uses Kelly criterion.

## Files

| File | Description |
|------|-------------|
| `tennis_predictor_v2.py` | Model training, feature engineering, Kalshi API client, trading logic |
| `kalshi_tennis_bot.py` | Scans open Kalshi markets and flags edges automatically |
| `position_manager.py` | Tracks open positions, P&L, and portfolio exposure |
| `integrated_trading_system.py` | Combines prediction + position management into one workflow |
| `atp_data_loader.py` | Standalone data loading utilities |

## Setup

```bash
pip install xgboost scikit-learn numpy requests cryptography
```

Data not included due to size. Download from [github.com/JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) and [tennis_wta](https://github.com/JeffSackmann/tennis_wta), place in `./tennis_atp` and `./tennis_wta`.

```bash
# Train
python tennis_predictor_v2.py ./tennis_atp ./tennis_wta train

# Predict
python tennis_predictor_v2.py ./tennis_atp ./tennis_wta predict
```

For live trading, set Kalshi API credentials:

```bash
export KALSHI_API_KEY_ID='your_key_id'
export KALSHI_PRIVATE_KEY_PATH='path/to/key.pem'
```

## How the model works

Features are computed using only data prior to each match date — no leakage. An earlier version had a validation bug that was inflating out-of-sample numbers; that's been corrected and the pipeline now uses strict time-ordered cross-validation.

The Kalshi API client handles RSA-PSS request signing, market search, orderbook fetching, and order placement and cancellation.

## Notes

Model performance varies by surface and tournament level. This is a personal project. Use paper trading before putting real money on anything.
