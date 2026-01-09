#!/usr/bin/env python3
"""
Integrated Tennis Trading System
=================================
Complete end-to-end system for automated tennis betting on Kalshi.

Usage:
    python integrated_trading_system.py              # Paper trading (safe)
    python integrated_trading_system.py --live       # REAL MONEY

Requirements:
    pip install xgboost scikit-learn numpy requests cryptography
"""

import os
import sys
import json
import time
import pickle
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass 
class TradingConfig:
    atp_data_dir: str = "./tennis_atp"
    wta_data_dir: str = "./tennis_wta"
    model_path: str = "tennis_model_v2.pkl"
    bankroll: float = 1000.0
    min_edge: float = 0.05
    max_edge: float = 0.40
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.10
    stop_loss_pct: float = 0.50
    take_profit_pct: float = 0.30
    max_drawdown_pct: float = 0.20
    max_open_positions: int = 5
    max_daily_bets: int = 20
    max_daily_loss: float = 200.0
    scan_interval_secs: int = 30
    demo_mode: bool = True
    paper_trading: bool = True
    use_ml_model: bool = True
    use_elo_fallback: bool = True
    min_matches_for_prediction: int = 10
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TradingConfig':
        with open(filepath, 'r') as f:
            return cls(**json.load(f))


class SimpleElo:
    def __init__(self, k_factor: float = 32, initial: float = 1500):
        self.k = k_factor
        self.initial = initial
        self.ratings: Dict[str, float] = defaultdict(lambda: initial)
        self.match_counts: Dict[str, int] = defaultdict(int)
    
    def expected(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update(self, winner_id: str, loser_id: str):
        r_w = self.ratings[winner_id]
        r_l = self.ratings[loser_id]
        expected_w = self.expected(r_w, r_l)
        k_w = self.k * (1.5 if self.match_counts[winner_id] < 30 else 1.0)
        k_l = self.k * (1.5 if self.match_counts[loser_id] < 30 else 1.0)
        self.ratings[winner_id] += k_w * (1 - expected_w)
        self.ratings[loser_id] += k_l * (0 - (1 - expected_w))
        self.match_counts[winner_id] += 1
        self.match_counts[loser_id] += 1
    
    def predict(self, player1_id: str, player2_id: str) -> float:
        return self.expected(self.ratings[player1_id], self.ratings[player2_id])


class DataManager:
    def __init__(self, atp_dir: str, wta_dir: str = None):
        self.atp_dir = atp_dir
        self.wta_dir = wta_dir
        self.players: Dict[str, dict] = {}
        self.matches: List[dict] = []
        self.elo = SimpleElo()
        self._name_index: Dict[str, List[str]] = defaultdict(list)
    
    def load(self, years: range = None):
        if years is None:
            years = range(2015, 2026)
        
        if self.atp_dir and os.path.exists(self.atp_dir):
            self._load_tour(self.atp_dir, "ATP", years)
        if self.wta_dir and os.path.exists(self.wta_dir):
            self._load_tour(self.wta_dir, "WTA", years)
        
        self.matches.sort(key=lambda m: m['date'])
        for match in self.matches:
            self.elo.update(match['winner_id'], match['loser_id'])
        
        logger.info(f"Loaded {len(self.matches)} matches, {len(self.players)} players")
    
    def _load_tour(self, data_dir: str, tour: str, years: range):
        import csv
        
        players_file = os.path.join(data_dir, f"{tour.lower()}_players.csv")
        if os.path.exists(players_file):
            with open(players_file, 'r', encoding='utf-8', errors='replace') as f:
                for row in csv.DictReader(f):
                    pid = f"{tour}_{row.get('player_id', '')}"
                    # Sackmann uses name_first/name_last, not first_name/last_name
                    first = row.get('name_first', '') or row.get('first_name', '')
                    last = row.get('name_last', '') or row.get('last_name', '')
                    name = f"{first} {last}".strip()
                    self.players[pid] = {'id': pid, 'name': name, 'tour': tour}
                    for part in name.lower().split():
                        self._name_index[part].append(pid)
        
        for year in years:
            match_file = os.path.join(data_dir, f"{tour.lower()}_matches_{year}.csv")
            if os.path.exists(match_file):
                with open(match_file, 'r', encoding='utf-8', errors='replace') as f:
                    for row in csv.DictReader(f):
                        try:
                            date_str = row.get('tourney_date', '')
                            if len(date_str) >= 8:
                                date = datetime.strptime(date_str[:8], '%Y%m%d')
                                self.matches.append({
                                    'date': date,
                                    'winner_id': f"{tour}_{row.get('winner_id', '')}",
                                    'loser_id': f"{tour}_{row.get('loser_id', '')}",
                                    'tour': tour,
                                })
                        except:
                            continue
    
    def find_player(self, name: str, tour: str = None) -> List[dict]:
        name_lower = name.lower().strip()
        candidates = set()
        
        # Search by each word in the name
        for part in name_lower.split():
            candidates.update(self._name_index.get(part, []))
        
        # Also try last name only (most reliable)
        name_parts = name_lower.split()
        if len(name_parts) > 1:
            last_name = name_parts[-1]
            candidates.update(self._name_index.get(last_name, []))
        
        results = []
        for pid in candidates:
            player = self.players.get(pid)
            if not player:
                continue
            if tour and player['tour'] != tour:
                continue
            
            player_name_lower = player['name'].lower()
            
            # Match if:
            # 1. Full name matches
            # 2. All parts of search name are in player name
            # 3. Last names match
            if name_lower == player_name_lower:
                results.insert(0, player)  # Exact match first
            elif name_lower in player_name_lower or player_name_lower in name_lower:
                results.append(player)
            elif all(p in player_name_lower for p in name_lower.split()):
                results.append(player)
            elif len(name_parts) > 1 and name_parts[-1] in player_name_lower.split():
                # Last name match
                results.append(player)
        
        return results
    
    def get_prediction(self, player1_id: str, player2_id: str) -> float:
        return self.elo.predict(player1_id, player2_id)


class PredictionEngine:
    def __init__(self, config: TradingConfig, data_manager: DataManager):
        self.config = config
        self.data = data_manager
        self.ml_model = None
        if config.use_ml_model:
            self._load_ml_model()
    
    def _load_ml_model(self):
        if not os.path.exists(self.config.model_path):
            return
        try:
            with open(self.config.model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.ml_model = model_data.get('model') if isinstance(model_data, dict) else model_data
            logger.info("✓ ML model loaded")
        except Exception as e:
            logger.warning(f"ML model load failed: {e}")
    
    def predict(self, p1_name: str, p2_name: str, tour: str = None) -> Optional[dict]:
        p1_results = self.data.find_player(p1_name, tour) or self.data.find_player(p1_name)
        p2_results = self.data.find_player(p2_name, tour) or self.data.find_player(p2_name)
        
        if not p1_results or not p2_results:
            return None
        
        player1, player2 = p1_results[0], p2_results[0]
        p1_matches = self.data.elo.match_counts.get(player1['id'], 0)
        p2_matches = self.data.elo.match_counts.get(player2['id'], 0)
        
        if min(p1_matches, p2_matches) < self.config.min_matches_for_prediction:
            return None
        
        p1_prob = self.data.get_prediction(player1['id'], player2['id'])
        p1_prob = max(0.05, min(0.95, p1_prob))
        confidence = "HIGH" if min(p1_matches, p2_matches) >= 50 else "MEDIUM" if min(p1_matches, p2_matches) >= 30 else "LOW"
        
        return {
            'p1_prob': p1_prob, 'p2_prob': 1 - p1_prob,
            'confidence': confidence, 'source': 'ELO',
            'p1_name': player1['name'], 'p2_name': player2['name'],
        }


class KalshiIntegration:
    PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.base_url = self.DEMO_URL if config.demo_mode else self.PROD_URL
        try:
            import requests
            self.session = requests.Session()
        except ImportError:
            self.session = None
        
        self.api_key_id = os.environ.get('KALSHI_API_KEY_ID')
        self.private_key_path = os.environ.get('KALSHI_PRIVATE_KEY_PATH')
        self.private_key = None
        self.is_authenticated = False
        
        if self.private_key_path and os.path.exists(self.private_key_path):
            try:
                from cryptography.hazmat.primitives import serialization
                with open(self.private_key_path, 'rb') as f:
                    self.private_key = serialization.load_pem_private_key(f.read(), password=None)
                self.is_authenticated = True
                logger.info("✓ Kalshi authenticated")
            except Exception as e:
                logger.warning(f"Auth failed: {e}")
    
    def _sign(self, method: str, path: str) -> dict:
        if not self.is_authenticated:
            return {}
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64
            
            timestamp = str(int(datetime.now().timestamp() * 1000))
            message = f"{timestamp}{method}{path.split('?')[0]}"
            signature = self.private_key.sign(
                message.encode('utf-8'),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256()
            )
            return {
                'KALSHI-ACCESS-KEY': self.api_key_id,
                'KALSHI-ACCESS-SIGNATURE': base64.b64encode(signature).decode('utf-8'),
                'KALSHI-ACCESS-TIMESTAMP': timestamp,
                'Content-Type': 'application/json'
            }
        except:
            return {}
    
    def get_tennis_markets(self) -> List[dict]:
        if not self.session:
            return []
        markets = []
        for series in ['KXATPMATCH', 'KXWTAMATCH']:
            try:
                response = self.session.get(f"{self.base_url}/markets",
                    params={'series_ticker': series, 'status': 'open', 'limit': 200}, timeout=15)
                if response.status_code == 200:
                    markets.extend(response.json().get('markets', []))
            except:
                pass
        return markets
    
    def get_market(self, ticker: str) -> dict:
        if not self.session:
            return {}
        try:
            response = self.session.get(f"{self.base_url}/markets/{ticker}", timeout=10)
            return response.json().get('market', {}) if response.status_code == 200 else {}
        except:
            return {}
    
    def place_order(self, ticker: str, side: str, action: str, count: int, price_cents: int) -> dict:
        if not self.session or not self.is_authenticated:
            return {'success': False, 'error': 'Not authenticated'}
        
        import uuid
        order_data = {
            'ticker': ticker, 'action': action, 'side': side, 'count': count,
            'type': 'limit', 'client_order_id': str(uuid.uuid4()),
        }
        order_data['yes_price' if side == 'yes' else 'no_price'] = price_cents
        
        try:
            headers = self._sign('POST', '/trade-api/v2/portfolio/orders')
            response = self.session.post(f"{self.base_url}/portfolio/orders",
                json=order_data, headers=headers, timeout=15)
            if response.status_code in [200, 201]:
                order = response.json().get('order', {})
                return {'success': True, 'order_id': order.get('order_id')}
            return {'success': False, 'error': response.text}
        except Exception as e:
            return {'success': False, 'error': str(e)}


@dataclass
class ActivePosition:
    ticker: str
    side: str
    player_name: str
    opponent_name: str
    contracts: int
    entry_price: float
    current_price: float
    entry_time: datetime
    model_prob: float
    entry_market_prob: float
    pnl: float = 0
    status: str = "open"


class TradingEngine:
    def __init__(self, config: TradingConfig):
        self.config = config
        logger.info("Initializing trading system...")
        
        self.data = DataManager(config.atp_data_dir, config.wta_data_dir)
        self.data.load()
        self.predictor = PredictionEngine(config, self.data)
        self.kalshi = KalshiIntegration(config)
        
        self.positions: List[ActivePosition] = []
        self.closed_positions: List[ActivePosition] = []
        self.bankroll = config.bankroll
        self.starting_bankroll = config.bankroll
        self.daily_bets = 0
        self.daily_pnl = 0
        self._running = False
    
    def run(self):
        logger.info("=" * 60)
        logger.info("  TENNIS TRADING SYSTEM")
        logger.info("=" * 60)
        logger.info(f"  Paper Trading: {self.config.paper_trading}")
        logger.info(f"  Bankroll: ${self.config.bankroll:,.2f}")
        logger.info(f"  Min Edge: {self.config.min_edge:.1%}")
        logger.info("=" * 60)
        
        self._running = True
        scan_count = 0
        
        try:
            while self._running:
                scan_count += 1
                self._manage_positions()
                
                if not self._check_risk_limits():
                    time.sleep(self.config.scan_interval_secs)
                    continue
                
                for opp in self._scan_markets()[:3]:
                    self._execute_trade(opp)
                
                self._display_status(scan_count)
                time.sleep(self.config.scan_interval_secs)
        
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()
    
    def _manage_positions(self):
        for pos in list(self.positions):
            if pos.status != "open":
                continue
            
            market = self.kalshi.get_market(pos.ticker)
            if market:
                pos.current_price = market.get('yes_bid' if pos.side == 'yes' else 'no_bid', pos.current_price)
            
            pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
            pos.pnl = (pos.current_price - pos.entry_price) * pos.contracts / 100
            
            if pnl_pct <= -self.config.stop_loss_pct:
                self._close_position(pos, "STOP_LOSS")
            elif pnl_pct >= self.config.take_profit_pct:
                self._close_position(pos, "TAKE_PROFIT")
    
    def _close_position(self, pos: ActivePosition, reason: str):
        logger.info(f"Closing: {pos.player_name} - {reason}")
        
        if self.config.paper_trading:
            self.bankroll += pos.contracts * pos.entry_price / 100 + pos.pnl
            pos.status = reason.lower()
            self.positions.remove(pos)
            self.closed_positions.append(pos)
            self.daily_pnl += pos.pnl
            logger.info(f"  [PAPER] P&L: ${pos.pnl:.2f}")
        else:
            result = self.kalshi.place_order(pos.ticker, pos.side, 'sell', pos.contracts, int(pos.current_price))
            if result.get('success'):
                pos.status = reason.lower()
                self.positions.remove(pos)
                self.closed_positions.append(pos)
    
    def _check_risk_limits(self) -> bool:
        drawdown = (self.starting_bankroll - self.bankroll) / self.starting_bankroll
        if drawdown >= self.config.max_drawdown_pct:
            return False
        if self.daily_pnl <= -self.config.max_daily_loss:
            return False
        if len(self.positions) >= self.config.max_open_positions:
            return False
        return True
    
    def _scan_markets(self) -> List[dict]:
        markets = self.kalshi.get_tennis_markets()
        if not markets:
            logger.info("  No tennis markets found on Kalshi")
            return []
        
        logger.info(f"  Found {len(markets)} Kalshi markets")
        
        matches = defaultdict(lambda: {'players': [], 'tour': ''})
        for m in markets:
            ticker = m.get('ticker', '')
            title = m.get('title', '')
            parts = ticker.rsplit('-', 1)
            if len(parts) < 2:
                continue
            
            match_id = parts[0]
            matches[match_id]['tour'] = 'ATP' if 'ATP' in match_id else 'WTA'
            
            if 'Will ' in title and ' win ' in title:
                player_name = title.split('Will ')[1].split(' win ')[0].strip()
                matches[match_id]['players'].append({
                    'name': player_name, 'ticker': ticker,
                    'yes_bid': m.get('yes_bid', 0), 'yes_ask': m.get('yes_ask', 0),
                })
        
        logger.info(f"  Parsed {len(matches)} unique matches")
        
        opportunities = []
        for match_id, data in matches.items():
            if len(data['players']) < 2:
                continue
            
            p1, p2 = data['players'][0], data['players'][1]
            pred = self.predictor.predict(p1['name'], p2['name'], data['tour'])
            
            if not pred:
                logger.debug(f"  ✗ {p1['name']} vs {p2['name']} - Players not found in database")
                # Try to show which player failed
                p1_found = self.data.find_player(p1['name'], data['tour']) or self.data.find_player(p1['name'])
                p2_found = self.data.find_player(p2['name'], data['tour']) or self.data.find_player(p2['name'])
                if not p1_found:
                    logger.info(f"    ⚠ Player not found: '{p1['name']}'")
                if not p2_found:
                    logger.info(f"    ⚠ Player not found: '{p2['name']}'")
                continue
            
            market_p1 = p1['yes_bid'] / 100 if p1['yes_bid'] > 0 else 0.5
            market_p2 = p2['yes_bid'] / 100 if p2['yes_bid'] > 0 else 0.5
            edge_p1 = pred['p1_prob'] - market_p1
            edge_p2 = pred['p2_prob'] - market_p2
            
            # Log all matches with their edges
            best_edge = max(edge_p1, edge_p2)
            logger.info(f"  {p1['name']} vs {p2['name']}: Model {pred['p1_prob']*100:.0f}% | Market {market_p1*100:.0f}% | Edge {best_edge*100:+.1f}%")
            
            if self.config.min_edge <= edge_p1 <= self.config.max_edge:
                opportunities.append({
                    'match_id': match_id, 'bet_player': p1['name'], 'opponent': p2['name'],
                    'ticker': p1['ticker'], 'side': 'yes', 'model_prob': pred['p1_prob'],
                    'market_prob': market_p1, 'edge': edge_p1, 'price_cents': p1['yes_ask'],
                    'confidence': pred['confidence'],
                })
            
            if self.config.min_edge <= edge_p2 <= self.config.max_edge:
                opportunities.append({
                    'match_id': match_id, 'bet_player': p2['name'], 'opponent': p1['name'],
                    'ticker': p2['ticker'], 'side': 'yes', 'model_prob': pred['p2_prob'],
                    'market_prob': market_p2, 'edge': edge_p2, 'price_cents': p2['yes_ask'],
                    'confidence': pred['confidence'],
                })
        
        if not opportunities:
            logger.info(f"  No edges >= {self.config.min_edge*100:.0f}% found")
        
        opportunities.sort(key=lambda x: x['edge'], reverse=True)
        return opportunities
    
    def _execute_trade(self, opp: dict):
        for pos in self.positions:
            if opp['match_id'] in pos.ticker:
                return
        
        model_prob, market_prob = opp['model_prob'], opp['market_prob']
        decimal_odds = 1 / market_prob if market_prob > 0 else 1
        b = decimal_odds - 1
        kelly = max(0, (b * model_prob - (1 - model_prob)) / b) * self.config.kelly_fraction if b > 0 else 0
        kelly = min(kelly, self.config.max_position_pct)
        
        bet_amount = self.bankroll * kelly
        if bet_amount < 1:
            return
        
        price_cents = opp['price_cents'] or int(market_prob * 100)
        contracts = int(bet_amount * 100 / price_cents) if price_cents > 0 else 0
        if contracts <= 0:
            return
        
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"🎾 BET: {opp['bet_player']} vs {opp['opponent']}")
        logger.info(f"   Model: {model_prob*100:.1f}% | Market: {market_prob*100:.1f}% | Edge: {opp['edge']*100:+.1f}%")
        logger.info(f"   {contracts}x @ {price_cents}¢ = ${contracts * price_cents / 100:.2f}")
        logger.info("=" * 50)
        
        position = ActivePosition(
            ticker=opp['ticker'], side=opp['side'], player_name=opp['bet_player'],
            opponent_name=opp['opponent'], contracts=contracts, entry_price=price_cents,
            current_price=price_cents, entry_time=datetime.now(), model_prob=model_prob,
            entry_market_prob=market_prob,
        )
        
        if self.config.paper_trading:
            self.positions.append(position)
            self.bankroll -= contracts * price_cents / 100
            self.daily_bets += 1
            logger.info("✓ [PAPER] Position opened")
        else:
            result = self.kalshi.place_order(opp['ticker'], opp['side'], 'buy', contracts, price_cents)
            if result.get('success'):
                self.positions.append(position)
                self.daily_bets += 1
                logger.info(f"✓ Order: {result.get('order_id')}")
    
    def _display_status(self, scan_count: int):
        unrealized = sum(p.pnl for p in self.positions)
        realized = sum(p.pnl for p in self.closed_positions)
        exposure = sum(p.contracts * p.entry_price / 100 for p in self.positions)
        total_pnl = unrealized + realized
        pnl_pct = (total_pnl / self.starting_bankroll * 100) if self.starting_bankroll > 0 else 0
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  SCAN #{scan_count} | {datetime.now().strftime('%H:%M:%S')}")
        logger.info("=" * 60)
        logger.info(f"  💰 PORTFOLIO")
        logger.info(f"     Starting:    ${self.starting_bankroll:,.2f}")
        logger.info(f"     Current:     ${self.bankroll:,.2f}")
        logger.info(f"     Exposure:    ${exposure:,.2f}")
        logger.info(f"     Unrealized:  ${unrealized:+,.2f}")
        logger.info(f"     Realized:    ${realized:+,.2f}")
        logger.info(f"     Total P&L:   ${total_pnl:+,.2f} ({pnl_pct:+.1f}%)")
        logger.info("")
        logger.info(f"  📊 STATS")
        logger.info(f"     Open Positions:   {len(self.positions)}")
        logger.info(f"     Closed Positions: {len(self.closed_positions)}")
        logger.info(f"     Total Bets Today: {self.daily_bets}")
        logger.info(f"     Win Rate:         {self._calc_win_rate():.1f}%")
        
        if self.positions:
            logger.info("")
            logger.info(f"  📈 OPEN POSITIONS")
            for pos in self.positions:
                pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
                emoji = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
                logger.info(f"     {emoji} {pos.player_name}")
                logger.info(f"        {pos.contracts}x @ {pos.entry_price}¢ → {pos.current_price}¢ ({pnl_pct:+.1f}%)")
                logger.info(f"        P&L: ${pos.pnl:+.2f} | Model: {pos.model_prob*100:.0f}%")
        else:
            logger.info("")
            logger.info(f"  📈 NO OPEN POSITIONS")
        
        if self.closed_positions:
            logger.info("")
            logger.info(f"  📉 RECENT CLOSED")
            for pos in self.closed_positions[-3:]:  # Show last 3
                emoji = "✅" if pos.pnl > 0 else "❌"
                logger.info(f"     {emoji} {pos.player_name}: ${pos.pnl:+.2f} ({pos.status})")
        
        logger.info("")
        logger.info(f"  ⏱️  Next scan in {self.config.scan_interval_secs}s...")
        logger.info("=" * 60)
    
    def _calc_win_rate(self) -> float:
        if not self.closed_positions:
            return 0.0
        wins = sum(1 for p in self.closed_positions if p.pnl > 0)
        return (wins / len(self.closed_positions)) * 100
    
    def _shutdown(self):
        total_pnl = sum(p.pnl for p in self.positions + self.closed_positions)
        logger.info("")
        logger.info("=" * 60)
        logger.info("  SESSION SUMMARY")
        logger.info(f"  P&L: ${total_pnl:+,.2f} | Bets: {self.daily_bets}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Tennis Trading System')
    parser.add_argument('--bankroll', type=float, default=1000)
    parser.add_argument('--min-edge', type=float, default=5, help='Min edge %%')
    parser.add_argument('--scan-interval', type=int, default=30, help='Seconds between scans')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--live', action='store_true', help='REAL MONEY!')
    parser.add_argument('--atp-dir', default='./tennis_atp')
    parser.add_argument('--wta-dir', default='./tennis_wta')
    parser.add_argument('--model', default='tennis_model_v2.pkl')
    
    args = parser.parse_args()
    
    config = TradingConfig(
        bankroll=args.bankroll, min_edge=args.min_edge / 100,
        demo_mode=args.demo, paper_trading=not args.live,
        atp_data_dir=args.atp_dir, wta_data_dir=args.wta_dir, model_path=args.model,
        scan_interval_secs=args.scan_interval,
    )
    
    if args.live:
        print("\n⚠️  LIVE TRADING - REAL MONEY!")
        if input("Type 'I UNDERSTAND': ") != 'I UNDERSTAND':
            return
    
    TradingEngine(config).run()


if __name__ == "__main__":
    main()
