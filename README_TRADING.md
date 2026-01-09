# Tennis Prediction & Kalshi Trading System

An automated tennis betting system that uses machine learning predictions to find edges on Kalshi prediction markets and execute trades with sophisticated position management.

## Features

- **XGBoost ML Model**: Trained on historical ATP/WTA match data with 83+ features
- **Elo Rating Fallback**: Surface-specific Elo ratings for prediction backup
- **Kalshi API Integration**: Full trading support (buy/sell, positions, orders)
- **Automated Edge Detection**: Scans markets continuously for betting opportunities
- **Kelly Criterion Sizing**: Optimal bet sizing based on edge magnitude
- **Dynamic Position Management**:
  - Stop-loss triggers based on match state
  - Take-profit with partial exit support
  - Edge monitoring (exit when edge disappears)
  - Momentum-based adjustments
- **Risk Controls**: Max drawdown, daily loss limits, position limits

## File Structure

```
├── integrated_trading_system.py   # Main trading bot (START HERE)
├── kalshi_tennis_bot.py           # Alternative trading bot implementation
├── position_manager.py            # Advanced position management
├── tennis_predictor_v2.py         # Original prediction system
├── atp_prediction_system.py       # ATP-specific prediction engine
├── atp_data_loader.py             # Data loading utilities
├── tennis_model_v2.pkl            # Pre-trained XGBoost model
└── tennis_atp/                    # ATP match data (clone separately)
└── tennis_wta/                    # WTA match data (clone separately)
```

## Installation

### 1. Clone Required Data
```bash
# Clone Jeff Sackmann's tennis data repositories
git clone https://github.com/JeffSackmann/tennis_atp.git
git clone https://github.com/JeffSackmann/tennis_wta.git
```

### 2. Install Dependencies
```bash
pip install xgboost scikit-learn numpy requests cryptography
```

### 3. Set Up Kalshi API Credentials
```bash
# Get your API key from kalshi.com -> Settings -> API
export KALSHI_API_KEY_ID="your_api_key_id"
export KALSHI_PRIVATE_KEY_PATH="/path/to/your/private_key.pem"
```

## Usage

### Paper Trading (Safe - No Real Money)
```bash
# Default: paper trading with $1000 bankroll, 5% min edge
python integrated_trading_system.py

# Custom settings
python integrated_trading_system.py --bankroll 500 --min-edge 8
```

### Live Trading (Real Money!)
```bash
# WARNING: This uses REAL MONEY
python integrated_trading_system.py --live --bankroll 500
```

### Command Line Options
```
--bankroll FLOAT    Starting bankroll in dollars (default: 1000)
--min-edge FLOAT    Minimum edge % to bet (default: 5)
--demo              Use Kalshi demo API
--live              Enable live trading (REAL MONEY)
--atp-dir PATH      Path to tennis_atp data (default: ./tennis_atp)
--wta-dir PATH      Path to tennis_wta data (default: ./tennis_wta)
--model PATH        Path to ML model file (default: tennis_model_v2.pkl)
```

## How It Works

### 1. Data Loading
The system loads historical ATP/WTA match data and builds:
- Player database with name lookup
- Elo ratings (overall and surface-specific)
- Head-to-head records

### 2. Market Scanning
Every 30 seconds, the bot:
- Fetches all open tennis markets from Kalshi
- Parses player matchups from market titles
- Groups markets by match

### 3. Edge Calculation
For each match:
- Looks up both players in the database
- Gets model prediction (ML or Elo-based)
- Compares to Kalshi market odds
- Calculates edge = model_prob - market_prob

### 4. Bet Sizing
Uses Kelly Criterion with safety fraction:
```
kelly = (b*p - q) / b * kelly_fraction
where:
  b = decimal odds - 1
  p = model probability
  q = 1 - p
```

### 5. Position Management
The system continuously monitors open positions:

**Stop Loss** (default 50%):
- Tightens in later sets
- Tightens when odds moving against us
- Protects existing profits

**Take Profit** (default 30%):
- Full exit at target
- Optional partial exits at intermediate levels
- Earlier exits in deciding sets

**Edge Monitoring**:
- Exits if edge drops below minimum threshold
- Exits if edge decayed significantly from entry

## Configuration

Create a `config.json` file:
```json
{
    "bankroll": 1000.0,
    "min_edge": 0.05,
    "max_edge": 0.40,
    "kelly_fraction": 0.25,
    "max_position_pct": 0.10,
    "stop_loss_pct": 0.50,
    "take_profit_pct": 0.30,
    "max_drawdown_pct": 0.20,
    "max_open_positions": 5,
    "max_daily_bets": 20,
    "max_daily_loss": 200.0,
    "scan_interval_secs": 30,
    "paper_trading": true
}
```

Load with: `python integrated_trading_system.py --config config.json`

## Risk Management

The system includes multiple safety layers:

1. **Position Limits**: Max 5 open positions, 10% max per position
2. **Daily Limits**: Max 20 bets, $200 max daily loss
3. **Drawdown Protection**: Stops trading at 20% drawdown
4. **Edge Caps**: Ignores suspiciously high edges (>40%)
5. **Paper Trading**: Default mode uses no real money

## Model Details

The XGBoost model uses features including:
- Elo ratings (overall and surface-specific)
- Recent form (wins/losses in last 60 days)
- Head-to-head record
- Ranking and ranking points
- Surface specialist metrics
- Serve statistics (aces, first serve %)
- Break point conversion/save rates
- Age and experience factors
- Tournament level adjustments

## API Rate Limits

The bot implements rate limiting:
- Max 10 requests per second
- Automatic retry on 429 errors
- Caching where appropriate

## Disclaimer

⚠️ **IMPORTANT**: This software is for educational purposes. Sports betting involves risk of financial loss. Past performance does not guarantee future results. Only bet what you can afford to lose. Gambling may be illegal in your jurisdiction.

## Data Attribution

Tennis data provided by Jeff Sackmann / Tennis Abstract under CC BY-NC-SA 4.0.
https://github.com/JeffSackmann/tennis_atp

## License

MIT License - See LICENSE file for details.
