#!/usr/bin/env python3
"""
Kalshi Tennis Trading Bot
==========================
Automated betting system that:
1. Scans Kalshi for tennis markets
2. Uses XGBoost model to find edges
3. Places bets using Kelly Criterion sizing
4. Manages positions (cash out losers, take profits on winners)
5. Tracks P&L in real-time

Requirements:
    pip install requests cryptography xgboost scikit-learn numpy

Setup:
    1. Set environment variables:
       export KALSHI_API_KEY_ID="your_key_id"
       export KALSHI_PRIVATE_KEY_PATH="/path/to/private_key.pem"
    
    2. Ensure tennis_atp and tennis_wta data directories exist
    
    3. Run: python kalshi_tennis_bot.py

Author: Trading Bot for Tennis Predictions
"""

import os
import sys
import json
import time
import uuid
import base64
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.error("Install requests: pip install requests")

try:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning("Install cryptography for authenticated trading: pip install cryptography")


# ============================================
# Configuration
# ============================================

@dataclass
class BotConfig:
    """Bot configuration parameters."""
    # Trading parameters
    bankroll: float = 1000.0
    max_position_pct: float = 0.10  # Max 10% of bankroll per position
    min_edge: float = 0.05  # Minimum 5% edge to bet
    kelly_fraction: float = 0.25  # Use 1/4 Kelly for safety
    
    # Position management
    stop_loss_pct: float = 0.50  # Cut position at 50% loss
    take_profit_pct: float = 0.30  # Take profit at 30% gain
    max_drawdown_pct: float = 0.20  # Stop trading if down 20%
    
    # Risk limits
    max_open_positions: int = 5
    max_daily_bets: int = 20
    max_loss_per_day: float = 200.0
    
    # Scanning
    scan_interval_secs: int = 30
    order_timeout_secs: int = 60
    
    # Demo mode
    demo_mode: bool = True  # Start in demo mode for safety
    paper_trading: bool = True  # Simulate trades without real money
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'BotConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'BotConfig':
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


# ============================================
# Position and Trade Tracking
# ============================================

class PositionStatus(Enum):
    OPEN = "open"
    PENDING = "pending"
    CLOSED = "closed"
    STOPPED = "stopped"
    PROFIT_TAKEN = "profit_taken"


@dataclass
class Position:
    """Tracks a single position."""
    ticker: str
    side: str  # 'yes' or 'no'
    player_name: str
    opponent_name: str
    contracts: int
    entry_price: float  # In cents
    current_price: float
    entry_time: datetime
    status: PositionStatus = PositionStatus.OPEN
    exit_price: float = 0
    exit_time: datetime = None
    pnl: float = 0
    model_prob: float = 0
    market_prob_at_entry: float = 0
    edge_at_entry: float = 0
    order_id: str = ""
    
    @property
    def cost_basis(self) -> float:
        """Total cost to enter position in dollars."""
        return (self.contracts * self.entry_price) / 100
    
    @property
    def current_value(self) -> float:
        """Current position value in dollars."""
        return (self.contracts * self.current_price) / 100
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L in dollars."""
        if self.status != PositionStatus.OPEN:
            return 0
        return self.current_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0
        return self.unrealized_pnl / self.cost_basis
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['status'] = self.status.value
        d['entry_time'] = self.entry_time.isoformat() if self.entry_time else None
        d['exit_time'] = self.exit_time.isoformat() if self.exit_time else None
        return d


@dataclass
class TradingSession:
    """Tracks a trading session."""
    start_time: datetime = field(default_factory=datetime.now)
    starting_bankroll: float = 0
    current_bankroll: float = 0
    positions: List[Position] = field(default_factory=list)
    closed_positions: List[Position] = field(default_factory=list)
    daily_pnl: float = 0
    total_bets: int = 0
    winning_bets: int = 0
    
    @property
    def open_positions(self) -> List[Position]:
        return [p for p in self.positions if p.status == PositionStatus.OPEN]
    
    @property
    def total_exposure(self) -> float:
        return sum(p.cost_basis for p in self.open_positions)
    
    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.open_positions)
    
    @property
    def realized_pnl(self) -> float:
        return sum(p.pnl for p in self.closed_positions)
    
    @property
    def win_rate(self) -> float:
        if self.total_bets == 0:
            return 0
        return self.winning_bets / self.total_bets


# ============================================
# Kalshi API Client (Enhanced)
# ============================================

class KalshiClient:
    """Enhanced Kalshi API client for automated trading."""
    
    PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"
    
    def __init__(self, api_key_id: str = None, private_key_path: str = None, demo: bool = False):
        self.session = requests.Session() if HAS_REQUESTS else None
        self.base_url = self.DEMO_URL if demo else self.PROD_URL
        self.api_key_id = api_key_id or os.environ.get('KALSHI_API_KEY_ID')
        self.private_key_path = private_key_path or os.environ.get('KALSHI_PRIVATE_KEY_PATH')
        self.private_key = None
        self.is_authenticated = False
        self._rate_limit_remaining = 100
        self._last_request_time = 0
        
        if self.private_key_path and os.path.exists(self.private_key_path) and HAS_CRYPTO:
            try:
                with open(self.private_key_path, 'rb') as f:
                    self.private_key = serialization.load_pem_private_key(f.read(), password=None)
                self.is_authenticated = True
                logger.info(f"✓ Kalshi API authenticated (key: {self.api_key_id[:8] if self.api_key_id else 'N/A'}...)")
            except Exception as e:
                logger.error(f"Failed to load private key: {e}")
    
    def _rate_limit(self):
        """Simple rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.1:  # Max 10 requests per second
            time.sleep(0.1 - elapsed)
        self._last_request_time = time.time()
    
    def _sign_request(self, method: str, path: str) -> dict:
        """Sign request with RSA-PSS."""
        if not self.is_authenticated or not self.private_key:
            return {}
        
        try:
            timestamp = str(int(datetime.now().timestamp() * 1000))
            path_for_signing = path.split('?')[0]
            message = f"{timestamp}{method}{path_for_signing}"
            
            signature = self.private_key.sign(
                message.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256()
            )
            
            return {
                'KALSHI-ACCESS-KEY': self.api_key_id,
                'KALSHI-ACCESS-SIGNATURE': base64.b64encode(signature).decode('utf-8'),
                'KALSHI-ACCESS-TIMESTAMP': timestamp,
                'Content-Type': 'application/json'
            }
        except Exception as e:
            logger.error(f"Signing error: {e}")
            return {}
    
    def _get(self, path: str, params: dict = None, auth: bool = False) -> dict:
        """Make GET request with rate limiting."""
        if not self.session:
            return {}
        
        self._rate_limit()
        url = f"{self.base_url}{path}"
        full_path = f"/trade-api/v2{path}"
        headers = self._sign_request('GET', full_path) if auth else {}
        
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                logger.warning("Rate limited, waiting...")
                time.sleep(5)
            return {}
        except Exception as e:
            logger.error(f"GET error: {e}")
            return {}
    
    def _post(self, path: str, data: dict) -> dict:
        """Make authenticated POST request."""
        if not self.session or not self.is_authenticated:
            logger.error("Not authenticated for POST requests")
            return {}
        
        self._rate_limit()
        body = json.dumps(data)
        url = f"{self.base_url}{path}"
        full_path = f"/trade-api/v2{path}"
        headers = self._sign_request('POST', full_path)
        
        try:
            response = self.session.post(url, data=body, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Order error: {e.response.text if e.response else e}")
            return {}
        except Exception as e:
            logger.error(f"POST error: {e}")
            return {}
    
    def _delete(self, path: str) -> dict:
        """Make authenticated DELETE request."""
        if not self.session or not self.is_authenticated:
            return {}
        
        self._rate_limit()
        url = f"{self.base_url}{path}"
        full_path = f"/trade-api/v2{path}"
        headers = self._sign_request('DELETE', full_path)
        
        try:
            response = self.session.delete(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"DELETE error: {e}")
            return {}
    
    # === Market Data ===
    
    def get_tennis_markets(self) -> List[dict]:
        """Get all open tennis markets."""
        markets = []
        
        for series in ['KXATPMATCH', 'KXWTAMATCH']:
            data = self._get('/markets', {'series_ticker': series, 'status': 'open', 'limit': 200})
            markets.extend(data.get('markets', []))
        
        return markets
    
    def get_market(self, ticker: str) -> dict:
        """Get specific market details."""
        data = self._get(f'/markets/{ticker}')
        return data.get('market', {})
    
    def get_orderbook(self, ticker: str) -> dict:
        """Get orderbook for market."""
        data = self._get(f'/markets/{ticker}/orderbook')
        return data.get('orderbook', {})
    
    # === Account ===
    
    def get_balance(self) -> dict:
        """Get account balance in dollars."""
        data = self._get('/portfolio/balance', auth=True)
        if not data:
            return {'balance': 0, 'available': 0}
        
        return {
            'balance': data.get('balance', 0) / 100,
            'available': data.get('portfolio_value', data.get('balance', 0)) / 100,
        }
    
    def get_positions(self) -> List[dict]:
        """Get current positions."""
        data = self._get('/portfolio/positions', auth=True)
        positions = []
        
        for pos in data.get('market_positions', []):
            positions.append({
                'ticker': pos.get('ticker'),
                'position': pos.get('position'),  # positive = yes, negative = no
                'market_exposure': pos.get('market_exposure', 0) / 100,
                'realized_pnl': pos.get('realized_pnl', 0) / 100,
                'resting_orders_count': pos.get('resting_orders_count', 0),
            })
        
        return positions
    
    def get_orders(self, status: str = 'resting') -> List[dict]:
        """Get orders by status."""
        data = self._get('/portfolio/orders', {'status': status}, auth=True)
        orders = []
        
        for order in data.get('orders', []):
            orders.append({
                'order_id': order.get('order_id'),
                'ticker': order.get('ticker'),
                'side': order.get('side'),
                'action': order.get('action'),
                'type': order.get('type'),
                'price': order.get('yes_price', order.get('no_price', 0)) / 100,
                'count': order.get('remaining_count'),
                'status': order.get('status'),
            })
        
        return orders
    
    # === Trading ===
    
    def place_order(self, ticker: str, side: str, action: str, 
                   count: int, price_cents: int = None,
                   order_type: str = 'limit') -> dict:
        """
        Place an order.
        
        Args:
            ticker: Market ticker
            side: 'yes' or 'no'
            action: 'buy' or 'sell'
            count: Number of contracts
            price_cents: Price in cents (1-99) for limit orders
            order_type: 'limit' or 'market'
        """
        order_data = {
            'ticker': ticker,
            'action': action,
            'side': side,
            'count': count,
            'type': order_type,
            'client_order_id': str(uuid.uuid4()),
        }
        
        if order_type == 'limit' and price_cents:
            if side == 'yes':
                order_data['yes_price'] = price_cents
            else:
                order_data['no_price'] = price_cents
        
        result = self._post('/portfolio/orders', order_data)
        
        if result.get('order'):
            order = result['order']
            return {
                'success': True,
                'order_id': order.get('order_id'),
                'ticker': order.get('ticker'),
                'side': order.get('side'),
                'action': order.get('action'),
                'price': order.get('yes_price', order.get('no_price', 0)) / 100,
                'count': order.get('remaining_count'),
                'status': order.get('status'),
            }
        
        return {'success': False, 'error': result}
    
    def buy_yes(self, ticker: str, count: int, price_cents: int) -> dict:
        """Buy YES contracts."""
        return self.place_order(ticker, 'yes', 'buy', count, price_cents, 'limit')
    
    def buy_no(self, ticker: str, count: int, price_cents: int) -> dict:
        """Buy NO contracts."""
        return self.place_order(ticker, 'no', 'buy', count, price_cents, 'limit')
    
    def sell_yes(self, ticker: str, count: int, price_cents: int) -> dict:
        """Sell YES contracts."""
        return self.place_order(ticker, 'yes', 'sell', count, price_cents, 'limit')
    
    def sell_no(self, ticker: str, count: int, price_cents: int) -> dict:
        """Sell NO contracts."""
        return self.place_order(ticker, 'no', 'sell', count, price_cents, 'limit')
    
    def market_buy(self, ticker: str, side: str, count: int) -> dict:
        """Market buy contracts."""
        return self.place_order(ticker, side, 'buy', count, order_type='market')
    
    def market_sell(self, ticker: str, side: str, count: int) -> dict:
        """Market sell contracts."""
        return self.place_order(ticker, side, 'sell', count, order_type='market')
    
    def cancel_order(self, order_id: str) -> dict:
        """Cancel specific order."""
        result = self._delete(f'/portfolio/orders/{order_id}')
        return {'success': bool(result), 'order': result.get('order', {})}
    
    def cancel_all_orders(self) -> int:
        """Cancel all resting orders."""
        orders = self.get_orders('resting')
        canceled = 0
        for order in orders:
            if self.cancel_order(order['order_id']).get('success'):
                canceled += 1
        return canceled


# ============================================
# Position Manager
# ============================================

class PositionManager:
    """Manages positions, stop losses, and take profits."""
    
    def __init__(self, client: KalshiClient, config: BotConfig, session: TradingSession):
        self.client = client
        self.config = config
        self.session = session
        self._position_lock = threading.Lock()
    
    def check_positions(self) -> List[dict]:
        """Check all open positions for stop loss / take profit triggers."""
        actions = []
        
        with self._position_lock:
            for pos in self.session.open_positions:
                # Get current market price
                market = self.client.get_market(pos.ticker)
                if not market:
                    continue
                
                # Update current price
                if pos.side == 'yes':
                    pos.current_price = market.get('yes_bid', pos.current_price)
                else:
                    pos.current_price = market.get('no_bid', pos.current_price)
                
                pnl_pct = pos.unrealized_pnl_pct
                
                # Check stop loss
                if pnl_pct <= -self.config.stop_loss_pct:
                    actions.append({
                        'action': 'stop_loss',
                        'position': pos,
                        'reason': f'Stop loss triggered: {pnl_pct:.1%}'
                    })
                
                # Check take profit
                elif pnl_pct >= self.config.take_profit_pct:
                    actions.append({
                        'action': 'take_profit',
                        'position': pos,
                        'reason': f'Take profit triggered: {pnl_pct:.1%}'
                    })
        
        return actions
    
    def execute_exit(self, position: Position, reason: str) -> bool:
        """Exit a position."""
        logger.info(f"Exiting position: {position.ticker} - {reason}")
        
        if self.config.paper_trading:
            # Simulate exit
            position.exit_price = position.current_price
            position.exit_time = datetime.now()
            position.pnl = position.unrealized_pnl
            position.status = PositionStatus.STOPPED if 'stop' in reason.lower() else PositionStatus.PROFIT_TAKEN
            
            with self._position_lock:
                if position in self.session.positions:
                    self.session.positions.remove(position)
                self.session.closed_positions.append(position)
            
            logger.info(f"  [PAPER] Closed at {position.exit_price}¢ | P&L: ${position.pnl:.2f}")
            return True
        
        # Real trading - market sell
        result = self.client.market_sell(position.ticker, position.side, position.contracts)
        
        if result.get('success'):
            position.exit_price = result.get('price', position.current_price) * 100
            position.exit_time = datetime.now()
            position.pnl = (position.exit_price - position.entry_price) * position.contracts / 100
            position.status = PositionStatus.STOPPED if 'stop' in reason.lower() else PositionStatus.PROFIT_TAKEN
            
            with self._position_lock:
                if position in self.session.positions:
                    self.session.positions.remove(position)
                self.session.closed_positions.append(position)
            
            logger.info(f"  Closed at {position.exit_price}¢ | P&L: ${position.pnl:.2f}")
            return True
        
        logger.error(f"  Failed to exit: {result.get('error')}")
        return False
    
    def open_position(self, ticker: str, side: str, player_name: str, opponent_name: str,
                     contracts: int, price_cents: int, model_prob: float, 
                     market_prob: float, edge: float) -> Optional[Position]:
        """Open a new position."""
        
        # Check limits
        if len(self.session.open_positions) >= self.config.max_open_positions:
            logger.warning("Max open positions reached")
            return None
        
        if self.session.total_bets >= self.config.max_daily_bets:
            logger.warning("Max daily bets reached")
            return None
        
        cost = (contracts * price_cents) / 100
        if cost > self.session.current_bankroll * self.config.max_position_pct:
            logger.warning(f"Position size ${cost:.2f} exceeds max {self.config.max_position_pct:.0%}")
            return None
        
        # Create position
        position = Position(
            ticker=ticker,
            side=side,
            player_name=player_name,
            opponent_name=opponent_name,
            contracts=contracts,
            entry_price=price_cents,
            current_price=price_cents,
            entry_time=datetime.now(),
            model_prob=model_prob,
            market_prob_at_entry=market_prob,
            edge_at_entry=edge,
        )
        
        if self.config.paper_trading:
            logger.info(f"[PAPER] Opening position: {contracts}x {side.upper()} @ {price_cents}¢")
            position.status = PositionStatus.OPEN
            
            with self._position_lock:
                self.session.positions.append(position)
                self.session.total_bets += 1
                self.session.current_bankroll -= cost
            
            return position
        
        # Real trading
        if side == 'yes':
            result = self.client.buy_yes(ticker, contracts, price_cents)
        else:
            result = self.client.buy_no(ticker, contracts, price_cents)
        
        if result.get('success'):
            position.order_id = result.get('order_id', '')
            position.status = PositionStatus.PENDING
            
            with self._position_lock:
                self.session.positions.append(position)
                self.session.total_bets += 1
            
            logger.info(f"Order placed: {result['order_id']}")
            return position
        
        logger.error(f"Failed to open position: {result.get('error')}")
        return None


# ============================================
# Edge Calculator
# ============================================

class EdgeCalculator:
    """Calculate betting edges and bet sizes."""
    
    def __init__(self, config: BotConfig):
        self.config = config
    
    def calculate_kelly(self, model_prob: float, market_prob: float) -> float:
        """
        Calculate Kelly Criterion bet size.
        
        Kelly = (bp - q) / b
        where:
            b = odds received (decimal odds - 1)
            p = probability of winning
            q = probability of losing (1 - p)
        """
        if market_prob <= 0 or market_prob >= 1:
            return 0
        
        # Convert to decimal odds
        decimal_odds = 1 / market_prob
        b = decimal_odds - 1
        p = model_prob
        q = 1 - model_prob
        
        kelly = (b * p - q) / b if b > 0 else 0
        
        # Apply fraction and cap
        kelly = max(0, kelly * self.config.kelly_fraction)
        kelly = min(kelly, self.config.max_position_pct)
        
        return kelly
    
    def calculate_edge(self, model_prob: float, market_prob: float) -> dict:
        """Calculate edge and betting recommendation."""
        edge = model_prob - market_prob
        abs_edge = abs(edge)
        
        # Determine bet quality
        if abs_edge >= 0.15:
            quality = 'EXCELLENT'
            urgency = 'HIGH'
        elif abs_edge >= 0.10:
            quality = 'GOOD'
            urgency = 'MEDIUM'
        elif abs_edge >= 0.05:
            quality = 'MARGINAL'
            urgency = 'LOW'
        else:
            quality = 'SKIP'
            urgency = 'NONE'
        
        kelly = self.calculate_kelly(model_prob, market_prob) if abs_edge >= self.config.min_edge else 0
        
        return {
            'edge': edge,
            'abs_edge': abs_edge,
            'quality': quality,
            'urgency': urgency,
            'kelly_fraction': kelly,
            'should_bet': abs_edge >= self.config.min_edge and kelly > 0,
            'model_prob': model_prob,
            'market_prob': market_prob,
        }
    
    def calculate_bet_size(self, kelly_fraction: float, bankroll: float) -> Tuple[int, int]:
        """Calculate contracts and price in cents."""
        bet_amount = bankroll * kelly_fraction
        
        # Minimum bet is $1 (1 contract at some price)
        if bet_amount < 1:
            return 0, 0
        
        # For simplicity, assume we're buying at the market ask
        # Contracts = bet_amount / (price_per_contract in dollars)
        # This will be refined with actual market data
        
        return int(bet_amount), 0  # Will be calculated with actual prices


# ============================================
# Market Scanner
# ============================================

class MarketScanner:
    """Scans Kalshi for tennis betting opportunities."""
    
    def __init__(self, client: KalshiClient, predictor=None):
        self.client = client
        self.predictor = predictor
        self._previous_odds = {}
    
    def scan(self) -> List[dict]:
        """Scan for tennis markets and parse player matchups."""
        markets = self.client.get_tennis_markets()
        
        if not markets:
            return []
        
        # Group markets by match
        matches = defaultdict(lambda: {'players': [], 'tour': ''})
        
        for m in markets:
            ticker = m.get('ticker', '')
            title = m.get('title', '')
            
            # Parse ticker to get match ID (e.g., KXATPMATCH-SINNER-ALCARAZ-SINNER -> KXATPMATCH-SINNER-ALCARAZ)
            parts = ticker.rsplit('-', 1)
            if len(parts) < 2:
                continue
            
            match_id = parts[0]
            
            if match_id not in matches:
                matches[match_id]['tour'] = 'ATP' if 'ATP' in match_id else 'WTA'
            
            # Extract player name from title "Will <player> win..."
            if 'Will ' in title and ' win ' in title:
                player_name = title.split('Will ')[1].split(' win ')[0].strip()
                
                yes_bid = m.get('yes_bid', 0)
                yes_ask = m.get('yes_ask', 0)
                no_bid = m.get('no_bid', 0)
                no_ask = m.get('no_ask', 0)
                volume = m.get('volume', 0)
                
                matches[match_id]['players'].append({
                    'name': player_name,
                    'ticker': ticker,
                    'yes_bid': yes_bid,
                    'yes_ask': yes_ask,
                    'no_bid': no_bid,
                    'no_ask': no_ask,
                    'volume': volume,
                })
        
        # Filter to matches with 2 players
        valid_matches = []
        for match_id, data in matches.items():
            if len(data['players']) >= 2:
                # Check for odds movement
                p1 = data['players'][0]
                old_odds = self._previous_odds.get(match_id, p1['yes_bid'])
                odds_change = abs(p1['yes_bid'] - old_odds) if old_odds else 0
                self._previous_odds[match_id] = p1['yes_bid']
                
                valid_matches.append({
                    'match_id': match_id,
                    'tour': data['tour'],
                    'p1': data['players'][0],
                    'p2': data['players'][1],
                    'odds_change': odds_change,
                })
        
        return valid_matches
    
    def get_model_prediction(self, p1_name: str, p2_name: str, tour: str) -> Optional[float]:
        """Get model prediction for a match."""
        if not self.predictor:
            return None
        
        try:
            # Try to find players in database
            p1_results = self.predictor.loader.find_player(p1_name, tour)
            p2_results = self.predictor.loader.find_player(p2_name, tour)
            
            if not p1_results:
                p1_results = self.predictor.loader.find_player(p1_name, None)
            if not p2_results:
                p2_results = self.predictor.loader.find_player(p2_name, None)
            
            if not p1_results or not p2_results:
                return None
            
            player1 = p1_results[0]
            player2 = p2_results[0]
            
            # Import Surface and TournamentLevel from predictor module
            from tennis_predictor_v2 import Surface, TournamentLevel
            
            result = self.predictor.predict_match(
                player1.player_id, player2.player_id,
                Surface.HARD, TournamentLevel.OTHER,
                None, None, 3
            )
            
            return result['p1_win_prob']
        
        except Exception as e:
            logger.error(f"Prediction error for {p1_name} vs {p2_name}: {e}")
            return None


# ============================================
# Main Trading Bot
# ============================================

class TennisTradingBot:
    """Main trading bot orchestrator."""
    
    def __init__(self, config: BotConfig = None, predictor=None):
        self.config = config or BotConfig()
        self.client = KalshiClient(demo=self.config.demo_mode)
        self.predictor = predictor
        
        # Initialize session
        self.session = TradingSession(
            starting_bankroll=self.config.bankroll,
            current_bankroll=self.config.bankroll,
        )
        
        # Initialize components
        self.scanner = MarketScanner(self.client, predictor)
        self.edge_calc = EdgeCalculator(self.config)
        self.position_mgr = PositionManager(self.client, self.config, self.session)
        
        self._running = False
        self._scan_thread = None
    
    def start(self):
        """Start the trading bot."""
        logger.info("=" * 60)
        logger.info("  TENNIS TRADING BOT STARTING")
        logger.info("=" * 60)
        logger.info(f"  Mode: {'DEMO' if self.config.demo_mode else 'PRODUCTION'}")
        logger.info(f"  Paper Trading: {'YES' if self.config.paper_trading else 'NO - REAL MONEY'}")
        logger.info(f"  Bankroll: ${self.config.bankroll:,.2f}")
        logger.info(f"  Min Edge: {self.config.min_edge:.1%}")
        logger.info(f"  Kelly Fraction: {self.config.kelly_fraction:.0%}")
        logger.info(f"  Stop Loss: {self.config.stop_loss_pct:.0%}")
        logger.info(f"  Take Profit: {self.config.take_profit_pct:.0%}")
        logger.info("=" * 60)
        
        # Verify account if not paper trading
        if not self.config.paper_trading:
            balance = self.client.get_balance()
            if balance['balance'] <= 0:
                logger.error("No account balance. Check API credentials.")
                return
            logger.info(f"Account balance: ${balance['balance']:,.2f}")
        
        self._running = True
        self._run_loop()
    
    def stop(self):
        """Stop the trading bot."""
        logger.info("Stopping bot...")
        self._running = False
    
    def _run_loop(self):
        """Main trading loop."""
        scan_count = 0
        
        try:
            while self._running:
                scan_count += 1
                
                # === 1. Check existing positions ===
                self._check_positions()
                
                # === 2. Check risk limits ===
                if not self._check_risk_limits():
                    logger.warning("Risk limits breached - pausing new trades")
                    time.sleep(self.config.scan_interval_secs)
                    continue
                
                # === 3. Scan for opportunities ===
                opportunities = self._scan_markets()
                
                # === 4. Execute trades ===
                for opp in opportunities:
                    self._execute_trade(opp)
                
                # === 5. Display status ===
                self._display_status(scan_count)
                
                # === 6. Wait for next scan ===
                time.sleep(self.config.scan_interval_secs)
        
        except KeyboardInterrupt:
            logger.info("\nShutdown requested...")
        finally:
            self._shutdown()
    
    def _check_positions(self):
        """Check and manage open positions."""
        actions = self.position_mgr.check_positions()
        
        for action in actions:
            pos = action['position']
            reason = action['reason']
            
            if action['action'] in ['stop_loss', 'take_profit']:
                self.position_mgr.execute_exit(pos, reason)
    
    def _check_risk_limits(self) -> bool:
        """Check if within risk limits."""
        # Check max drawdown
        drawdown = (self.session.starting_bankroll - self.session.current_bankroll) / self.session.starting_bankroll
        if drawdown >= self.config.max_drawdown_pct:
            logger.error(f"Max drawdown {drawdown:.1%} exceeded!")
            return False
        
        # Check daily loss limit
        if self.session.realized_pnl <= -self.config.max_loss_per_day:
            logger.error(f"Daily loss limit ${self.config.max_loss_per_day} exceeded!")
            return False
        
        return True
    
    def _scan_markets(self) -> List[dict]:
        """Scan markets and identify betting opportunities."""
        matches = self.scanner.scan()
        opportunities = []
        
        for match in matches:
            p1 = match['p1']
            p2 = match['p2']
            
            # Get market probabilities (convert cents to probability)
            market_p1 = p1['yes_bid'] / 100 if p1['yes_bid'] else 0.5
            market_p2 = p2['yes_bid'] / 100 if p2['yes_bid'] else 0.5
            
            # Skip if no real odds
            if market_p1 <= 0 or market_p2 <= 0:
                continue
            
            # Get model prediction
            model_p1 = self.scanner.get_model_prediction(p1['name'], p2['name'], match['tour'])
            
            if model_p1 is None:
                # No prediction available - skip or use market midpoint
                continue
            
            # Calculate edges for both players
            edge_p1 = self.edge_calc.calculate_edge(model_p1, market_p1)
            edge_p2 = self.edge_calc.calculate_edge(1 - model_p1, market_p2)
            
            # Check for betting opportunity
            if edge_p1['should_bet'] and edge_p1['edge'] > 0:
                opportunities.append({
                    'match': match,
                    'bet_player': p1['name'],
                    'bet_ticker': p1['ticker'],
                    'side': 'yes',
                    'opponent': p2['name'],
                    'edge': edge_p1,
                    'price_cents': p1['yes_ask'],  # Buy at ask
                })
            
            elif edge_p2['should_bet'] and edge_p2['edge'] > 0:
                opportunities.append({
                    'match': match,
                    'bet_player': p2['name'],
                    'bet_ticker': p2['ticker'],
                    'side': 'yes',
                    'opponent': p1['name'],
                    'edge': edge_p2,
                    'price_cents': p2['yes_ask'],
                })
        
        # Sort by edge quality
        opportunities.sort(key=lambda x: x['edge']['abs_edge'], reverse=True)
        
        return opportunities
    
    def _execute_trade(self, opportunity: dict):
        """Execute a trading opportunity."""
        edge = opportunity['edge']
        ticker = opportunity['bet_ticker']
        player = opportunity['bet_player']
        opponent = opportunity['opponent']
        side = opportunity['side']
        price_cents = opportunity['price_cents']
        
        # Skip if already have position in this match
        match_id = opportunity['match']['match_id']
        for pos in self.session.open_positions:
            if match_id in pos.ticker:
                logger.debug(f"Already have position in {match_id}")
                return
        
        # Calculate bet size
        kelly = edge['kelly_fraction']
        bet_amount = self.session.current_bankroll * kelly
        
        if bet_amount < 1:
            logger.debug(f"Bet size too small: ${bet_amount:.2f}")
            return
        
        # Calculate contracts
        if price_cents <= 0:
            price_cents = int(edge['market_prob'] * 100)
        
        contracts = int(bet_amount * 100 / price_cents) if price_cents > 0 else 0
        
        if contracts <= 0:
            return
        
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"🎾 BETTING OPPORTUNITY: {player} vs {opponent}")
        logger.info(f"   Model: {edge['model_prob']*100:.1f}% | Market: {edge['market_prob']*100:.1f}%")
        logger.info(f"   Edge: {edge['edge']*100:+.1f}% ({edge['quality']})")
        logger.info(f"   Bet: {contracts} contracts @ {price_cents}¢ = ${contracts * price_cents / 100:.2f}")
        logger.info("=" * 50)
        
        # Open position
        position = self.position_mgr.open_position(
            ticker=ticker,
            side=side,
            player_name=player,
            opponent_name=opponent,
            contracts=contracts,
            price_cents=price_cents,
            model_prob=edge['model_prob'],
            market_prob=edge['market_prob'],
            edge=edge['edge'],
        )
        
        if position:
            logger.info(f"✓ Position opened: {position.ticker}")
        else:
            logger.warning("✗ Failed to open position")
    
    def _display_status(self, scan_count: int):
        """Display current status."""
        now = datetime.now().strftime("%H:%M:%S")
        
        logger.info("")
        logger.info(f"─────────────────────────────────────────────")
        logger.info(f"SCAN #{scan_count} | {now}")
        logger.info(f"─────────────────────────────────────────────")
        logger.info(f"Bankroll: ${self.session.current_bankroll:,.2f} | "
                   f"Exposure: ${self.session.total_exposure:,.2f}")
        logger.info(f"Open: {len(self.session.open_positions)} | "
                   f"Closed: {len(self.session.closed_positions)} | "
                   f"Total Bets: {self.session.total_bets}")
        logger.info(f"Unrealized P&L: ${self.session.unrealized_pnl:+,.2f} | "
                   f"Realized P&L: ${self.session.realized_pnl:+,.2f}")
        
        # Show open positions
        if self.session.open_positions:
            logger.info("")
            logger.info("OPEN POSITIONS:")
            for pos in self.session.open_positions:
                pnl_pct = pos.unrealized_pnl_pct
                emoji = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
                logger.info(f"  {emoji} {pos.player_name}: {pos.contracts}x @ {pos.entry_price}¢ "
                           f"→ {pos.current_price}¢ ({pnl_pct:+.1%})")
    
    def _shutdown(self):
        """Graceful shutdown."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("  SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Duration: {datetime.now() - self.session.start_time}")
        logger.info(f"  Starting Bankroll: ${self.session.starting_bankroll:,.2f}")
        logger.info(f"  Ending Bankroll: ${self.session.current_bankroll:,.2f}")
        logger.info(f"  Total Bets: {self.session.total_bets}")
        logger.info(f"  Win Rate: {self.session.win_rate:.1%}")
        logger.info(f"  Realized P&L: ${self.session.realized_pnl:+,.2f}")
        logger.info(f"  Unrealized P&L: ${self.session.unrealized_pnl:+,.2f}")
        
        # Close any open positions (optional)
        if self.session.open_positions:
            logger.info("")
            logger.info(f"  {len(self.session.open_positions)} positions still open")
        
        logger.info("=" * 60)
        logger.info("  Bot stopped.")


# ============================================
# CLI Interface
# ============================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kalshi Tennis Trading Bot')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--bankroll', type=float, default=1000, help='Starting bankroll')
    parser.add_argument('--min-edge', type=float, default=5, help='Minimum edge %% to bet')
    parser.add_argument('--demo', action='store_true', help='Use demo API')
    parser.add_argument('--live', action='store_true', help='Enable live trading (not paper)')
    parser.add_argument('--atp-dir', type=str, default='./tennis_atp', help='ATP data directory')
    parser.add_argument('--wta-dir', type=str, default='./tennis_wta', help='WTA data directory')
    parser.add_argument('--model', type=str, default='tennis_model_v2.pkl', help='Path to model file')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        config = BotConfig.load(args.config)
    else:
        config = BotConfig(
            bankroll=args.bankroll,
            min_edge=args.min_edge / 100,
            demo_mode=args.demo,
            paper_trading=not args.live,
        )
    
    # Load predictor model if available
    predictor = None
    try:
        if os.path.exists(args.model):
            logger.info(f"Loading model from {args.model}...")
            with open(args.model, 'rb') as f:
                predictor = pickle.load(f)
            logger.info("✓ Model loaded")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        logger.info("Running without ML predictions (will skip betting)")
    
    # Safety confirmation for live trading
    if args.live:
        print("\n" + "!" * 60)
        print("  WARNING: LIVE TRADING MODE")
        print("  This will use REAL MONEY on Kalshi!")
        print("!" * 60)
        confirm = input("\nType 'YES' to confirm: ")
        if confirm != 'YES':
            print("Aborted.")
            return
    
    # Start bot
    bot = TennisTradingBot(config=config, predictor=predictor)
    
    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()


if __name__ == "__main__":
    main()
