#!/usr/bin/env python3
"""
ATP Tennis Prediction System - Enhanced Edition
================================================
- XGBoost ML model with 83 features
- Enhanced ranking weight for big gaps (#7 vs #57)
- Kalshi API integration for live odds
- Edge realization analysis (accounts for odds movement)
- Match profiles for quick tracking
- Kelly Criterion bet sizing

Requirements:
    pip install xgboost scikit-learn numpy requests

Usage:
    python tennis_predictor_v2.py ./tennis_atp ./tennis_wta train
    python tennis_predictor_v2.py ./tennis_atp ./tennis_wta predict
"""

import os
import math
import pickle
import random
import json
import glob
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import numpy as np

# ML imports
try:
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("Install ML packages: pip install xgboost scikit-learn numpy")

# For Kalshi API
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Install requests: pip install requests")


# ============================================
# Kalshi API Integration (with Trading)
# ============================================

class KalshiAPI:
    """
    Full Kalshi API client with authentication and trading.
    
    Setup:
        1. Create account at kalshi.com
        2. Go to Settings -> API -> Create API Key
        3. Save your API Key ID and Private Key
        4. Set environment variables or pass to constructor:
           - KALSHI_API_KEY_ID
           - KALSHI_PRIVATE_KEY_PATH (path to .pem file)
    """
    
    PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"
    
    def __init__(self, api_key_id: str = None, private_key_path: str = None, demo: bool = False):
        self.session = requests.Session() if HAS_REQUESTS else None
        self.base_url = self.DEMO_URL if demo else self.PROD_URL
        self.api_key_id = api_key_id or os.environ.get('KALSHI_API_KEY_ID')
        self.private_key_path = private_key_path or os.environ.get('KALSHI_PRIVATE_KEY_PATH')
        self.private_key = None
        self.is_authenticated = False
        self._markets_cache = {}
        
        # Load private key if available
        if self.private_key_path and os.path.exists(self.private_key_path):
            try:
                from cryptography.hazmat.primitives import serialization
                with open(self.private_key_path, 'rb') as f:
                    self.private_key = serialization.load_pem_private_key(f.read(), password=None)
                self.is_authenticated = True
                print(f"  Kalshi API authenticated (key: {self.api_key_id[:8]}...)")
            except ImportError:
                print("  Install cryptography for Kalshi trading: pip install cryptography")
            except Exception as e:
                print(f"  Failed to load Kalshi private key: {e}")
    
    def _sign_request(self, method: str, path: str) -> dict:
        """Create signed headers for authenticated requests using RSA-PSS."""
        if not self.is_authenticated or not self.private_key:
            return {}
        
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64
            
            # Timestamp in milliseconds
            timestamp = str(int(datetime.now().timestamp() * 1000))
            
            # Message to sign: timestamp + method + path
            # Path must include /trade-api/v2 prefix and strip query params
            path_for_signing = path.split('?')[0]
            message = f"{timestamp}{method}{path_for_signing}"
            
            # Sign with RSA-PSS (Kalshi's required format)
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
            print(f"  Signing error: {e}")
            return {}
    
    def _get(self, path: str, params: dict = None, auth: bool = False) -> dict:
        """Make GET request."""
        if not self.session:
            return {}
        
        # Full URL for request
        url = f"{self.base_url}{path}"
        
        # For signing, use the full path including /trade-api/v2
        full_path = f"/trade-api/v2{path}"
        headers = {}
        
        if auth:
            headers = self._sign_request('GET', full_path)
            if not headers:
                # Signing failed, return empty
                return {}
        
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Don't print error for every failed request - just return empty
            return {}
        except requests.exceptions.Timeout:
            return {}
        except Exception:
            return {}
    
    def _post(self, path: str, data: dict) -> dict:
        """Make authenticated POST request."""
        if not self.session or not self.is_authenticated:
            print("  Not authenticated. Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")
            return {}
        
        import json
        body = json.dumps(data)
        url = f"{self.base_url}{path}"
        full_path = f"/trade-api/v2{path}"
        headers = self._sign_request('POST', full_path)
        
        try:
            response = self.session.post(url, data=body, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"  Order error: {e.response.text if e.response else e}")
            return {}
        except Exception as e:
            print(f"  API error: {e}")
            return {}
    
    def _delete(self, path: str) -> dict:
        """Make authenticated DELETE request."""
        if not self.session or not self.is_authenticated:
            return {}
        
        url = f"{self.base_url}{path}"
        full_path = f"/trade-api/v2{path}"
        headers = self._sign_request('DELETE', full_path)
        
        try:
            response = self.session.delete(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  API error: {e}")
            return {}
    
    # ========== Market Data (Public) ==========
    
    def search_tennis_markets(self, player1: str = None, player2: str = None) -> List[Dict]:
        """Search for tennis markets on Kalshi."""
        data = self._get('/markets', {'status': 'open', 'limit': 200})
        
        tennis_markets = []
        for market in data.get('markets', []):
            title = market.get('title', '').lower()
            
            is_tennis = any(term in title for term in [
                'tennis', 'atp', 'wta', 'grand slam', 'australian open',
                'french open', 'roland garros', 'wimbledon', 'us open'
            ])
            
            if player1 and player2:
                if player1.lower() in title or player2.lower() in title:
                    is_tennis = True
            
            if is_tennis:
                tennis_markets.append({
                    'ticker': market.get('ticker'),
                    'title': market.get('title'),
                    'yes_bid': market.get('yes_bid', 0) / 100,
                    'yes_ask': market.get('yes_ask', 0) / 100,
                    'no_bid': market.get('no_bid', 0) / 100,
                    'no_ask': market.get('no_ask', 0) / 100,
                    'last_price': market.get('last_price', 0) / 100,
                    'volume': market.get('volume', 0),
                })
        
        return tennis_markets
    
    def get_market(self, ticker: str) -> dict:
        """Get specific market details."""
        data = self._get(f'/markets/{ticker}')
        market = data.get('market', {})
        
        return {
            'ticker': ticker,
            'title': market.get('title'),
            'status': market.get('status'),
            'yes_bid': market.get('yes_bid', 0) / 100,
            'yes_ask': market.get('yes_ask', 0) / 100,
            'no_bid': market.get('no_bid', 0) / 100,
            'no_ask': market.get('no_ask', 0) / 100,
            'last_price': market.get('last_price', 0) / 100,
            'volume': market.get('volume', 0),
        }
    
    def get_orderbook(self, ticker: str) -> dict:
        """Get orderbook depth for a market."""
        data = self._get(f'/markets/{ticker}/orderbook')
        return data.get('orderbook', {})
    
    # ========== Account (Authenticated) ==========
    
    def get_balance(self) -> dict:
        """Get account balance."""
        try:
            data = self._get('/portfolio/balance', auth=True)
            if not data:
                return {'balance': 0, 'available': 0}
            
            # API returns balance in cents
            balance_cents = data.get('balance', 0)
            # portfolio_value might also be present
            portfolio_cents = data.get('portfolio_value', 0)
            
            return {
                'balance': balance_cents / 100,  # Convert cents to dollars
                'available': balance_cents / 100,  # Available is same as balance for basic accounts
                'portfolio_value': portfolio_cents / 100,
            }
        except Exception as e:
            # Don't crash, just return zeros
            return {'balance': 0, 'available': 0, 'portfolio_value': 0}
    
    def get_positions(self) -> List[dict]:
        """Get current positions."""
        data = self._get('/portfolio/positions', auth=True)
        
        positions = []
        for pos in data.get('market_positions', []):
            positions.append({
                'ticker': pos.get('ticker'),
                'position': pos.get('position'),
                'market_exposure': pos.get('market_exposure', 0) / 100,
                'realized_pnl': pos.get('realized_pnl', 0) / 100,
            })
        
        return positions
    
    def get_orders(self, status: str = 'resting') -> List[dict]:
        """Get orders. Status: resting, pending, executed, canceled."""
        data = self._get('/portfolio/orders', {'status': status}, auth=True)
        
        orders = []
        for order in data.get('orders', []):
            orders.append({
                'order_id': order.get('order_id'),
                'ticker': order.get('ticker'),
                'side': order.get('side'),
                'action': order.get('action'),
                'type': order.get('type'),
                'price': order.get('yes_price', 0) / 100,
                'count': order.get('remaining_count'),
                'status': order.get('status'),
            })
        
        return orders
    
    # ========== Trading (Authenticated) ==========
    
    def place_order(self, ticker: str, side: str, action: str, 
                   count: int, price_cents: int = None, 
                   order_type: str = 'limit') -> dict:
        """
        Place an order on Kalshi.
        
        Args:
            ticker: Market ticker (e.g., 'TENNIS-AO-SINNER')
            side: 'yes' or 'no'
            action: 'buy' or 'sell'
            count: Number of contracts
            price_cents: Price in cents (1-99) for limit orders
            order_type: 'limit' or 'market'
        
        Returns:
            Order details or error
        """
        import uuid
        
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
        else:
            return {'success': False, 'error': result}
    
    def buy_yes(self, ticker: str, count: int, price_cents: int) -> dict:
        """Buy YES contracts at limit price."""
        return self.place_order(ticker, 'yes', 'buy', count, price_cents, 'limit')
    
    def buy_no(self, ticker: str, count: int, price_cents: int) -> dict:
        """Buy NO contracts at limit price."""
        return self.place_order(ticker, 'no', 'buy', count, price_cents, 'limit')
    
    def sell_yes(self, ticker: str, count: int, price_cents: int) -> dict:
        """Sell YES contracts at limit price."""
        return self.place_order(ticker, 'yes', 'sell', count, price_cents, 'limit')
    
    def sell_no(self, ticker: str, count: int, price_cents: int) -> dict:
        """Sell NO contracts at limit price."""
        return self.place_order(ticker, 'no', 'sell', count, price_cents, 'limit')
    
    def market_buy_yes(self, ticker: str, count: int) -> dict:
        """Market buy YES contracts."""
        return self.place_order(ticker, 'yes', 'buy', count, order_type='market')
    
    def market_buy_no(self, ticker: str, count: int) -> dict:
        """Market buy NO contracts."""
        return self.place_order(ticker, 'no', 'buy', count, order_type='market')
    
    def cancel_order(self, order_id: str) -> dict:
        """Cancel an order."""
        result = self._delete(f'/portfolio/orders/{order_id}')
        return {'success': True, 'order': result.get('order', {})} if result else {'success': False}
    
    def cancel_all_orders(self) -> dict:
        """Cancel all resting orders."""
        orders = self.get_orders('resting')
        canceled = 0
        for order in orders:
            if self.cancel_order(order['order_id']).get('success'):
                canceled += 1
        return {'canceled': canceled}


def setup_kalshi_credentials():
    """Interactive setup for Kalshi API credentials."""
    print("\n" + "=" * 60)
    print("  KALSHI API SETUP")
    print("=" * 60)
    print("\n  To trade on Kalshi, you need API credentials:")
    print("  1. Log into kalshi.com")
    print("  2. Go to Settings → API")
    print("  3. Create new API key")
    print("  4. Download the private key (.pem file)")
    
    print("\n  Enter credentials (or set environment variables):")
    print("    KALSHI_API_KEY_ID - Your API Key ID")
    print("    KALSHI_PRIVATE_KEY_PATH - Path to private key .pem file")
    
    key_id = input("\n  API Key ID [skip]: ").strip()
    key_path = input("  Private Key Path [skip]: ").strip()
    
    if key_id:
        os.environ['KALSHI_API_KEY_ID'] = key_id
    if key_path:
        os.environ['KALSHI_PRIVATE_KEY_PATH'] = key_path
    
    if key_id and key_path:
        print("\n  Credentials set for this session.")
        print("  To persist, add to your shell profile:")
        print(f"    export KALSHI_API_KEY_ID='{key_id}'")
        print(f"    export KALSHI_PRIVATE_KEY_PATH='{key_path}'")
        return True
    
    return False


# ============================================
# Data Structures
# ============================================

class Surface(Enum):
    HARD = "Hard"
    CLAY = "Clay"
    GRASS = "Grass"
    CARPET = "Carpet"
    
    @classmethod
    def from_string(cls, s: str) -> 'Surface':
        if not s:
            return cls.HARD
        s = s.strip().lower()
        if 'clay' in s:
            return cls.CLAY
        if 'grass' in s:
            return cls.GRASS
        if 'carpet' in s:
            return cls.CARPET
        return cls.HARD


class TournamentLevel(Enum):
    GRAND_SLAM = "G"
    MASTERS = "M"
    ATP500 = "A"
    ATP250 = "B"
    OTHER = ""
    
    @classmethod
    def from_string(cls, s: str) -> 'TournamentLevel':
        if not s:
            return cls.OTHER
        s = s.strip().upper()
        if s == 'G':
            return cls.GRAND_SLAM
        if s == 'M':
            return cls.MASTERS
        return cls.OTHER


@dataclass
class Player:
    player_id: str
    name: str = ""
    hand: str = "R"
    height_cm: int = 185
    country: str = ""
    birth_date: datetime = None
    
    def age_at(self, date: datetime) -> float:
        if not self.birth_date:
            return 27.0
        return (date - self.birth_date).days / 365.25


@dataclass
class Match:
    tourney_date: datetime
    surface: Surface
    tourney_level: TournamentLevel
    winner_id: str
    winner_name: str
    loser_id: str
    loser_name: str
    winner_rank: int = None
    loser_rank: int = None
    winner_rank_points: int = None
    loser_rank_points: int = None
    best_of: int = 3
    round_name: str = ""
    minutes: int = None
    score: str = ""
    w_ace: int = None
    w_df: int = None
    w_svpt: int = None
    w_1stIn: int = None
    w_1stWon: int = None
    w_2ndWon: int = None
    w_bpSaved: int = None
    w_bpFaced: int = None
    l_ace: int = None
    l_df: int = None
    l_svpt: int = None
    l_1stIn: int = None
    l_1stWon: int = None
    l_2ndWon: int = None
    l_bpSaved: int = None
    l_bpFaced: int = None


# ============================================
# Data Loader
# ============================================

class DataLoader:
    """Load Jeff Sackmann's ATP and WTA data."""
    
    def __init__(self, data_dir: str, wta_dir: str = None):
        self.data_dir = data_dir
        self.wta_dir = wta_dir
        self.matches: List[Match] = []
        self.players: Dict[str, Player] = {}
        self._player_matches: Dict[str, List[Match]] = defaultdict(list)
    
    def load(self, years: range = None, include_wta: bool = True):
        """Load matches and players."""
        if years is None:
            years = range(2000, 2026)
        
        # Load ATP
        players_file = os.path.join(self.data_dir, "atp_players.csv")
        if os.path.exists(players_file):
            self._load_players(players_file, "ATP")
        
        for year in years:
            match_file = os.path.join(self.data_dir, f"atp_matches_{year}.csv")
            if os.path.exists(match_file):
                self._load_matches(match_file, "ATP")
        
        atp_count = len(self.matches)
        print(f"Loaded {atp_count} ATP matches")
        
        # Load WTA
        if include_wta and self.wta_dir and os.path.exists(self.wta_dir):
            wta_players_file = os.path.join(self.wta_dir, "wta_players.csv")
            if os.path.exists(wta_players_file):
                self._load_players(wta_players_file, "WTA")
            
            for year in years:
                match_file = os.path.join(self.wta_dir, f"wta_matches_{year}.csv")
                if os.path.exists(match_file):
                    self._load_matches(match_file, "WTA")
            
            print(f"Loaded {len(self.matches) - atp_count} WTA matches")
        
        self.matches.sort(key=lambda m: m.tourney_date)
        
        for match in self.matches:
            self._player_matches[match.winner_id].append(match)
            self._player_matches[match.loser_id].append(match)
        
        print(f"Total: {len(self.matches)} matches, {len(self.players)} players")
    
    def _load_players(self, filepath: str, tour: str):
        import csv
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = str(row.get('player_id', ''))
                if not pid:
                    continue
                pid = f"{tour}_{pid}"
                
                birth_date = None
                bd = row.get('dob', '') or row.get('birth_date', '')
                if bd and len(str(bd)) >= 8:
                    try:
                        birth_date = datetime.strptime(str(int(float(bd))), "%Y%m%d")
                    except:
                        pass
                
                height = 185 if tour == "ATP" else 170
                h = row.get('height', '')
                if h:
                    try:
                        height = int(float(h))
                    except:
                        pass
                
                self.players[pid] = Player(
                    player_id=pid,
                    name=f"{row.get('name_first', '')} {row.get('name_last', '')}".strip(),
                    hand=row.get('hand', 'R'),
                    height_cm=height,
                    country=row.get('ioc', ''),
                    birth_date=birth_date
                )
    
    def _load_matches(self, filepath: str, tour: str):
        import csv
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    date_str = row.get('tourney_date', '')
                    if not date_str:
                        continue
                    tourney_date = datetime.strptime(str(int(float(date_str))), "%Y%m%d")
                    
                    def safe_int(v):
                        if not v:
                            return None
                        try:
                            return int(float(v))
                        except:
                            return None
                    
                    match = Match(
                        tourney_date=tourney_date,
                        surface=Surface.from_string(row.get('surface', '')),
                        tourney_level=TournamentLevel.from_string(row.get('tourney_level', '')),
                        winner_id=f"{tour}_{row.get('winner_id', '')}",
                        winner_name=row.get('winner_name', ''),
                        loser_id=f"{tour}_{row.get('loser_id', '')}",
                        loser_name=row.get('loser_name', ''),
                        winner_rank=safe_int(row.get('winner_rank')),
                        loser_rank=safe_int(row.get('loser_rank')),
                        winner_rank_points=safe_int(row.get('winner_rank_points')),
                        loser_rank_points=safe_int(row.get('loser_rank_points')),
                        best_of=safe_int(row.get('best_of')) or 3,
                        round_name=row.get('round', ''),
                        minutes=safe_int(row.get('minutes')),
                        score=row.get('score', ''),
                        w_ace=safe_int(row.get('w_ace')),
                        w_df=safe_int(row.get('w_df')),
                        w_svpt=safe_int(row.get('w_svpt')),
                        w_1stIn=safe_int(row.get('w_1stIn')),
                        w_1stWon=safe_int(row.get('w_1stWon')),
                        w_2ndWon=safe_int(row.get('w_2ndWon')),
                        w_bpSaved=safe_int(row.get('w_bpSaved')),
                        w_bpFaced=safe_int(row.get('w_bpFaced')),
                        l_ace=safe_int(row.get('l_ace')),
                        l_df=safe_int(row.get('l_df')),
                        l_svpt=safe_int(row.get('l_svpt')),
                        l_1stIn=safe_int(row.get('l_1stIn')),
                        l_1stWon=safe_int(row.get('l_1stWon')),
                        l_2ndWon=safe_int(row.get('l_2ndWon')),
                        l_bpSaved=safe_int(row.get('l_bpSaved')),
                        l_bpFaced=safe_int(row.get('l_bpFaced')),
                    )
                    self.matches.append(match)
                except:
                    continue
    
    def find_player(self, name: str, tour: str = None) -> List[Player]:
        """Search for player by name with flexible matching."""
        name_lower = name.lower().strip()
        results = []
        
        # First pass: exact substring match
        for p in self.players.values():
            if name_lower in p.name.lower():
                if tour is None or p.player_id.startswith(tour):
                    results.append(p)
        
        # If no results, try matching last name only
        if not results:
            for p in self.players.values():
                # Get last name from player
                player_last = p.name.split()[-1].lower() if p.name else ""
                if name_lower == player_last or name_lower in player_last:
                    if tour is None or p.player_id.startswith(tour):
                        results.append(p)
        
        # If still no results, try matching first name
        if not results:
            for p in self.players.values():
                player_first = p.name.split()[0].lower() if p.name else ""
                if name_lower == player_first or name_lower in player_first:
                    if tour is None or p.player_id.startswith(tour):
                        results.append(p)
        
        # If still no results, try partial match on any part
        if not results:
            for p in self.players.values():
                for part in p.name.lower().split():
                    if name_lower in part or part in name_lower:
                        if tour is None or p.player_id.startswith(tour):
                            if p not in results:
                                results.append(p)
                                break
        
        # Sort by relevance (exact matches first, then by name length)
        def sort_key(p):
            name_l = p.name.lower()
            # Exact match on last name
            if name_lower == name_l.split()[-1]:
                return (0, len(p.name))
            # Exact match anywhere
            if name_lower in name_l.split():
                return (1, len(p.name))
            # Substring match
            return (2, len(p.name))
        
        results.sort(key=sort_key)
        
        return results[:10]  # Return top 10 matches
    
    def get_h2h(self, p1_id: str, p2_id: str, before: datetime = None) -> Dict:
        result = {'p1_wins': 0, 'p2_wins': 0}
        for m in self._player_matches.get(p1_id, []):
            if before and m.tourney_date >= before:
                continue
            if m.winner_id == p2_id or m.loser_id == p2_id:
                if m.winner_id == p1_id:
                    result['p1_wins'] += 1
                else:
                    result['p2_wins'] += 1
        return result
    
    def get_recent_form(self, player_id: str, before: datetime, days: int = 60) -> Tuple[float, int]:
        cutoff = before - timedelta(days=days)
        wins, total = 0, 0
        for m in self._player_matches.get(player_id, []):
            if cutoff <= m.tourney_date < before:
                total += 1
                if m.winner_id == player_id:
                    wins += 1
        return (wins / total if total > 0 else 0.5, total)
    
    def get_surface_record(self, player_id: str, surface: Surface, before: datetime = None) -> Tuple[int, int]:
        wins, losses = 0, 0
        for m in self._player_matches.get(player_id, []):
            if m.surface != surface:
                continue
            if before and m.tourney_date >= before:
                continue
            if m.winner_id == player_id:
                wins += 1
            else:
                losses += 1
        return (wins, losses)


# ============================================
# Feature Engineering (Enhanced Rankings)
# ============================================

class FeatureEngine:
    """Extract features for ML model - Enhanced with better ranking weights."""
    
    FEATURE_NAMES = [
        # Elo (4)
        'elo_diff', 'surface_elo_diff', 'elo_p1', 'elo_p2',
        # Ranking (8) - ENHANCED
        'rank_diff_log', 'rank_pts_ratio', 'rank_p1_log', 'rank_p2_log',
        'elite_factor', 'rank_gap_norm', 'rank_tier_diff', 'underdog_factor',
        # H2H (6)
        'h2h_rate', 'h2h_count', 'h2h_recent_rate', 'h2h_recent_count',
        'h2h_surface_rate', 'h2h_surface_count',
        # Form (8)
        'form_p1', 'form_p2', 'form_diff',
        'form_30d_p1', 'form_30d_p2',
        'surface_wr_p1', 'surface_wr_p2', 'surface_wr_diff',
        # Fatigue (6)
        'matches_7d_p1', 'matches_7d_p2',
        'matches_14d_p1', 'matches_14d_p2',
        'days_rest_p1', 'days_rest_p2',
        # Serve stats (8)
        'ace_rate_diff', 'ace_rate_p1', 'ace_rate_p2',
        'first_serve_won_diff', 'bp_save_diff',
        'second_serve_won_diff',
        'serve_rating_p1', 'serve_rating_p2',
        # Return stats (6)
        'return_won_diff', 'return_won_p1', 'return_won_p2',
        'bp_convert_diff', 'bp_convert_p1', 'bp_convert_p2',
        # Physical (6)
        'height_diff', 'height_p1', 'height_p2',
        'age_diff', 'age_p1', 'age_p2',
        # Context (6)
        'is_grand_slam', 'is_masters', 'is_best_of_5',
        'round_number', 'is_outdoor', 'draw_size_log',
        # Handedness (3)
        'is_lefty_p1', 'is_lefty_p2', 'lefty_matchup',
        # Big match (6)
        'vs_top10_wr_p1', 'vs_top10_wr_p2', 'vs_top10_diff',
        'vs_top50_wr_p1', 'vs_top50_wr_p2', 'vs_top50_diff',
        # Clutch (6)
        'tiebreak_wr_p1', 'tiebreak_wr_p2', 'tiebreak_diff',
        'deciding_set_wr_p1', 'deciding_set_wr_p2', 'deciding_set_diff',
        # Momentum (4)
        'win_streak_p1', 'win_streak_p2',
        'loss_streak_p1', 'loss_streak_p2',
        # Experience (4)
        'career_matches_p1', 'career_matches_p2',
        'surface_matches_p1', 'surface_matches_p2',
        # Composite (4)
        'serve_rating_diff', 'return_rating_diff',
        'overall_rating_p1', 'overall_rating_p2',
    ]
    
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.elo = defaultdict(lambda: 1500.0)
        self.surface_elo = {s: defaultdict(lambda: 1500.0) for s in Surface}
        self._stats_cache = {}
    
    def update_elo(self, winner_id: str, loser_id: str, surface: Surface, k: float = 32):
        rw, rl = self.elo[winner_id], self.elo[loser_id]
        exp = 1 / (1 + 10**((rl - rw)/400))
        self.elo[winner_id] += k * (1 - exp)
        self.elo[loser_id] -= k * (1 - exp)
        
        srw, srl = self.surface_elo[surface][winner_id], self.surface_elo[surface][loser_id]
        sexp = 1 / (1 + 10**((srl - srw)/400))
        self.surface_elo[surface][winner_id] += k * (1 - sexp)
        self.surface_elo[surface][loser_id] -= k * (1 - sexp)
    
    def _get_rank_tier(self, rank: int) -> int:
        """Categorize rank into tiers."""
        if rank <= 5:
            return 6  # Elite
        elif rank <= 10:
            return 5  # Top 10
        elif rank <= 20:
            return 4  # Top 20
        elif rank <= 50:
            return 3  # Top 50
        elif rank <= 100:
            return 2  # Top 100
        else:
            return 1  # Outside top 100
    
    def _get_extended_stats(self, player_id: str, surface: Surface, 
                           match_date: datetime) -> Dict:
        """Get extended player statistics."""
        cache_key = (player_id, surface, match_date.year, match_date.month)
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]
        
        stats = {
            'matches': 0, 'wins': 0,
            'aces': 0, 'svpt': 0,
            'first_in': 0, 'first_won': 0, 'second_won': 0,
            'bp_saved': 0, 'bp_faced': 0,
            'return_won': 0, 'return_total': 0,
            'bp_converted': 0, 'bp_opportunities': 0,
            'vs_top10_wins': 0, 'vs_top10_matches': 0,
            'vs_top50_wins': 0, 'vs_top50_matches': 0,
            'tiebreaks_won': 0, 'tiebreaks_played': 0,
            'deciding_sets_won': 0, 'deciding_sets_played': 0,
            'surface_matches': 0, 'surface_wins': 0,
        }
        
        for m in self.loader._player_matches.get(player_id, []):
            if m.tourney_date >= match_date:
                continue
            
            is_winner = m.winner_id == player_id
            opp_rank = m.loser_rank if is_winner else m.winner_rank
            
            stats['matches'] += 1
            if is_winner:
                stats['wins'] += 1
            
            if m.surface == surface:
                stats['surface_matches'] += 1
                if is_winner:
                    stats['surface_wins'] += 1
            
            if opp_rank:
                if opp_rank <= 10:
                    stats['vs_top10_matches'] += 1
                    if is_winner:
                        stats['vs_top10_wins'] += 1
                if opp_rank <= 50:
                    stats['vs_top50_matches'] += 1
                    if is_winner:
                        stats['vs_top50_wins'] += 1
            
            # Serve/return stats
            if is_winner:
                if m.w_ace: stats['aces'] += m.w_ace
                if m.w_svpt: stats['svpt'] += m.w_svpt
                if m.w_1stIn: stats['first_in'] += m.w_1stIn
                if m.w_1stWon: stats['first_won'] += m.w_1stWon
                if m.w_2ndWon: stats['second_won'] += m.w_2ndWon
                if m.w_bpSaved: stats['bp_saved'] += m.w_bpSaved
                if m.w_bpFaced: stats['bp_faced'] += m.w_bpFaced
                if m.l_svpt and m.l_1stWon and m.l_2ndWon:
                    stats['return_won'] += m.l_svpt - m.l_1stWon - m.l_2ndWon
                    stats['return_total'] += m.l_svpt
                if m.l_bpFaced and m.l_bpSaved:
                    stats['bp_converted'] += m.l_bpFaced - m.l_bpSaved
                    stats['bp_opportunities'] += m.l_bpFaced
            else:
                if m.l_ace: stats['aces'] += m.l_ace
                if m.l_svpt: stats['svpt'] += m.l_svpt
                if m.l_1stIn: stats['first_in'] += m.l_1stIn
                if m.l_1stWon: stats['first_won'] += m.l_1stWon
                if m.l_2ndWon: stats['second_won'] += m.l_2ndWon
                if m.l_bpSaved: stats['bp_saved'] += m.l_bpSaved
                if m.l_bpFaced: stats['bp_faced'] += m.l_bpFaced
                if m.w_svpt and m.w_1stWon and m.w_2ndWon:
                    stats['return_won'] += m.w_svpt - m.w_1stWon - m.w_2ndWon
                    stats['return_total'] += m.w_svpt
                if m.w_bpFaced and m.w_bpSaved:
                    stats['bp_converted'] += m.w_bpFaced - m.w_bpSaved
                    stats['bp_opportunities'] += m.w_bpFaced
        
        # Calculate percentages
        stats['ace_rate'] = stats['aces'] / stats['svpt'] if stats['svpt'] > 0 else 0.05
        stats['first_serve_pct'] = stats['first_in'] / stats['svpt'] if stats['svpt'] > 0 else 0.6
        stats['first_serve_won'] = stats['first_won'] / stats['first_in'] if stats['first_in'] > 0 else 0.7
        second_total = stats['svpt'] - stats['first_in']
        stats['second_serve_won'] = stats['second_won'] / second_total if second_total > 0 else 0.5
        stats['bp_save_pct'] = stats['bp_saved'] / stats['bp_faced'] if stats['bp_faced'] > 0 else 0.6
        stats['return_won_pct'] = stats['return_won'] / stats['return_total'] if stats['return_total'] > 0 else 0.35
        stats['bp_convert_pct'] = stats['bp_converted'] / stats['bp_opportunities'] if stats['bp_opportunities'] > 0 else 0.4
        stats['vs_top10_wr'] = stats['vs_top10_wins'] / stats['vs_top10_matches'] if stats['vs_top10_matches'] > 0 else 0.3
        stats['vs_top50_wr'] = stats['vs_top50_wins'] / stats['vs_top50_matches'] if stats['vs_top50_matches'] > 0 else 0.4
        stats['tiebreak_wr'] = stats['tiebreaks_won'] / stats['tiebreaks_played'] if stats['tiebreaks_played'] > 0 else 0.5
        stats['deciding_set_wr'] = stats['deciding_sets_won'] / stats['deciding_sets_played'] if stats['deciding_sets_played'] > 0 else 0.5
        stats['surface_wr'] = stats['surface_wins'] / stats['surface_matches'] if stats['surface_matches'] > 0 else 0.5
        stats['win_pct'] = stats['wins'] / stats['matches'] if stats['matches'] > 0 else 0.5
        
        self._stats_cache[cache_key] = stats
        return stats
    
    def _get_fatigue(self, player_id: str, match_date: datetime) -> Dict:
        matches_7d, matches_14d = 0, 0
        days_since_last = 30
        
        for m in self.loader._player_matches.get(player_id, []):
            if m.tourney_date >= match_date:
                continue
            days_ago = (match_date - m.tourney_date).days
            if days_ago < days_since_last:
                days_since_last = days_ago
            if days_ago <= 7:
                matches_7d += 1
            if days_ago <= 14:
                matches_14d += 1
            if days_ago > 30:
                break
        
        return {'matches_7d': matches_7d, 'matches_14d': matches_14d, 'days_rest': min(days_since_last, 30)}
    
    def _get_streaks(self, player_id: str, match_date: datetime) -> Dict:
        win_streak, loss_streak = 0, 0
        matches = sorted(
            [m for m in self.loader._player_matches.get(player_id, []) if m.tourney_date < match_date],
            key=lambda m: m.tourney_date, reverse=True
        )[:20]
        
        for m in matches:
            if m.winner_id == player_id:
                if loss_streak == 0:
                    win_streak += 1
                else:
                    break
            else:
                if win_streak == 0:
                    loss_streak += 1
                else:
                    break
        
        return {'win_streak': min(win_streak, 15), 'loss_streak': min(loss_streak, 10)}
    
    def _get_h2h_extended(self, p1_id: str, p2_id: str, surface: Surface, match_date: datetime) -> Dict:
        result = {'p1_wins': 0, 'p2_wins': 0, 'p1_wins_recent': 0, 'p2_wins_recent': 0,
                  'p1_wins_surface': 0, 'p2_wins_surface': 0}
        two_years_ago = match_date - timedelta(days=730)
        
        for m in self.loader._player_matches.get(p1_id, []):
            if m.tourney_date >= match_date:
                continue
            if m.winner_id != p2_id and m.loser_id != p2_id:
                continue
            p1_won = m.winner_id == p1_id
            if p1_won:
                result['p1_wins'] += 1
            else:
                result['p2_wins'] += 1
            if m.tourney_date >= two_years_ago:
                if p1_won:
                    result['p1_wins_recent'] += 1
                else:
                    result['p2_wins_recent'] += 1
            if m.surface == surface:
                if p1_won:
                    result['p1_wins_surface'] += 1
                else:
                    result['p2_wins_surface'] += 1
        return result
    
    def _round_to_number(self, round_name: str) -> float:
        round_map = {'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3, 'R64': 2, 'R128': 1, 'RR': 5, 'BR': 6}
        for key, val in round_map.items():
            if key in round_name:
                return val / 7.0
        return 0.5
    
    def extract(self, p1_id: str, p2_id: str, surface: Surface, 
                tourney_level: TournamentLevel, match_date: datetime,
                p1_rank: int = None, p2_rank: int = None,
                p1_pts: int = None, p2_pts: int = None,
                best_of: int = 3, round_name: str = "", draw_size: int = 32) -> np.ndarray:
        """Extract feature vector with ENHANCED ranking features."""
        features = []
        
        p1 = self.loader.players.get(p1_id, Player(p1_id))
        p2 = self.loader.players.get(p2_id, Player(p2_id))
        stats1 = self._get_extended_stats(p1_id, surface, match_date)
        stats2 = self._get_extended_stats(p2_id, surface, match_date)
        
        # ===== ELO (4) =====
        elo1, elo2 = self.elo[p1_id], self.elo[p2_id]
        selo1, selo2 = self.surface_elo[surface][p1_id], self.surface_elo[surface][p2_id]
        features.extend([(elo1 - elo2) / 400, (selo1 - selo2) / 400, (elo1 - 1500) / 400, (elo2 - 1500) / 400])
        
        # ===== RANKING (8) - ENHANCED =====
        r1 = p1_rank or 200
        r2 = p2_rank or 200
        pts1 = p1_pts or 500
        pts2 = p2_pts or 500
        
        # Basic log difference
        log_diff = math.log10(r2) - math.log10(r1)
        
        # ENHANCED: Amplify for big gaps
        rank_gap = abs(r1 - r2)
        rank_gap_factor = 0
        if rank_gap > 40:
            rank_gap_factor = math.log10(rank_gap) * 0.4  # Increased from 0.3
        elif rank_gap > 20:
            rank_gap_factor = math.log10(rank_gap) * 0.25
        elif rank_gap > 10:
            rank_gap_factor = math.log10(rank_gap) * 0.15
        
        if r1 > r2:  # p1 is lower ranked (higher number)
            rank_gap_factor = -rank_gap_factor
        
        # ENHANCED: Elite vs non-elite factor
        elite_factor = 0
        if r1 <= 10 and r2 > 50:
            elite_factor = 0.25  # Increased from 0.15
        elif r1 <= 10 and r2 > 30:
            elite_factor = 0.15
        elif r1 <= 20 and r2 > 50:
            elite_factor = 0.12
        elif r1 <= 30 and r2 > 70:
            elite_factor = 0.08
        elif r2 <= 10 and r1 > 50:
            elite_factor = -0.25
        elif r2 <= 10 and r1 > 30:
            elite_factor = -0.15
        elif r2 <= 20 and r1 > 50:
            elite_factor = -0.12
        elif r2 <= 30 and r1 > 70:
            elite_factor = -0.08
        
        # Tier difference
        tier1 = self._get_rank_tier(r1)
        tier2 = self._get_rank_tier(r2)
        tier_diff = (tier1 - tier2) / 6.0
        
        # Underdog factor - lower ranked player sometimes has less to lose
        underdog_factor = 0
        if r1 > 50 and r2 <= 10:
            underdog_factor = 0.03  # Slight boost for big underdog
        elif r2 > 50 and r1 <= 10:
            underdog_factor = -0.03
        
        features.extend([
            log_diff + rank_gap_factor,  # Enhanced log difference
            math.log(pts1 / max(pts2, 1)) / 5,
            math.log10(r1) / 3,
            math.log10(r2) / 3,
            elite_factor,
            rank_gap / 100,
            tier_diff,
            underdog_factor,
        ])
        
        # ===== H2H (6) =====
        h2h = self._get_h2h_extended(p1_id, p2_id, surface, match_date)
        total = h2h['p1_wins'] + h2h['p2_wins']
        recent_total = h2h['p1_wins_recent'] + h2h['p2_wins_recent']
        surface_total = h2h['p1_wins_surface'] + h2h['p2_wins_surface']
        
        features.extend([
            (h2h['p1_wins'] / total - 0.5) if total > 0 else 0,
            min(total, 20) / 20,
            (h2h['p1_wins_recent'] / recent_total - 0.5) if recent_total > 0 else 0,
            min(recent_total, 10) / 10,
            (h2h['p1_wins_surface'] / surface_total - 0.5) if surface_total > 0 else 0,
            min(surface_total, 10) / 10,
        ])
        
        # ===== FORM (8) =====
        form1_60, _ = self.loader.get_recent_form(p1_id, match_date, 60)
        form2_60, _ = self.loader.get_recent_form(p2_id, match_date, 60)
        form1_30, _ = self.loader.get_recent_form(p1_id, match_date, 30)
        form2_30, _ = self.loader.get_recent_form(p2_id, match_date, 30)
        
        features.extend([
            form1_60 - 0.5, form2_60 - 0.5, form1_60 - form2_60,
            form1_30 - 0.5, form2_30 - 0.5,
            stats1['surface_wr'] - 0.5, stats2['surface_wr'] - 0.5, stats1['surface_wr'] - stats2['surface_wr'],
        ])
        
        # ===== FATIGUE (6) =====
        fat1 = self._get_fatigue(p1_id, match_date)
        fat2 = self._get_fatigue(p2_id, match_date)
        features.extend([
            fat1['matches_7d'] / 5, fat2['matches_7d'] / 5,
            fat1['matches_14d'] / 8, fat2['matches_14d'] / 8,
            fat1['days_rest'] / 14, fat2['days_rest'] / 14,
        ])
        
        # ===== SERVE STATS (8) =====
        ace1, ace2 = stats1['ace_rate'], stats2['ace_rate']
        fsw1, fsw2 = stats1['first_serve_won'], stats2['first_serve_won']
        ssw1, ssw2 = stats1['second_serve_won'], stats2['second_serve_won']
        bp1, bp2 = stats1['bp_save_pct'], stats2['bp_save_pct']
        serve_rating1 = ace1 * 10 + fsw1 * 0.5 + ssw1 * 0.3 + bp1 * 0.2
        serve_rating2 = ace2 * 10 + fsw2 * 0.5 + ssw2 * 0.3 + bp2 * 0.2
        
        features.extend([
            (ace1 - ace2) * 10, ace1 * 10, ace2 * 10,
            fsw1 - fsw2, bp1 - bp2, ssw1 - ssw2,
            serve_rating1, serve_rating2,
        ])
        
        # ===== RETURN STATS (6) =====
        ret1, ret2 = stats1['return_won_pct'], stats2['return_won_pct']
        bpc1, bpc2 = stats1['bp_convert_pct'], stats2['bp_convert_pct']
        features.extend([ret1 - ret2, ret1, ret2, bpc1 - bpc2, bpc1, bpc2])
        
        # ===== PHYSICAL (6) =====
        h1 = p1.height_cm or 185
        h2 = p2.height_cm or 185
        age1 = p1.age_at(match_date)
        age2 = p2.age_at(match_date)
        features.extend([(h1 - h2) / 20, h1 / 200, h2 / 200, (age1 - age2) / 10, (age1 - 27) / 10, (age2 - 27) / 10])
        
        # ===== CONTEXT (6) =====
        is_gs = tourney_level == TournamentLevel.GRAND_SLAM
        is_masters = tourney_level == TournamentLevel.MASTERS
        features.extend([
            1.0 if is_gs else 0.0, 1.0 if is_masters else 0.0, 1.0 if best_of == 5 else 0.0,
            self._round_to_number(round_name), 1.0, math.log2(max(draw_size, 2)) / 7,
        ])
        
        # ===== HANDEDNESS (3) =====
        is_lefty1 = 1.0 if p1.hand == 'L' else 0.0
        is_lefty2 = 1.0 if p2.hand == 'L' else 0.0
        features.extend([is_lefty1, is_lefty2, is_lefty1 - is_lefty2])
        
        # ===== BIG MATCH (6) =====
        features.extend([
            stats1['vs_top10_wr'], stats2['vs_top10_wr'], stats1['vs_top10_wr'] - stats2['vs_top10_wr'],
            stats1['vs_top50_wr'], stats2['vs_top50_wr'], stats1['vs_top50_wr'] - stats2['vs_top50_wr'],
        ])
        
        # ===== CLUTCH (6) =====
        features.extend([
            stats1['tiebreak_wr'], stats2['tiebreak_wr'], stats1['tiebreak_wr'] - stats2['tiebreak_wr'],
            stats1['deciding_set_wr'], stats2['deciding_set_wr'], stats1['deciding_set_wr'] - stats2['deciding_set_wr'],
        ])
        
        # ===== MOMENTUM (4) =====
        streaks1 = self._get_streaks(p1_id, match_date)
        streaks2 = self._get_streaks(p2_id, match_date)
        features.extend([
            streaks1['win_streak'] / 10, streaks2['win_streak'] / 10,
            streaks1['loss_streak'] / 5, streaks2['loss_streak'] / 5,
        ])
        
        # ===== EXPERIENCE (4) =====
        features.extend([
            min(stats1['matches'], 500) / 500, min(stats2['matches'], 500) / 500,
            min(stats1['surface_matches'], 200) / 200, min(stats2['surface_matches'], 200) / 200,
        ])
        
        # ===== COMPOSITE (4) =====
        return_rating1 = ret1 * 0.6 + bpc1 * 0.4
        return_rating2 = ret2 * 0.6 + bpc2 * 0.4
        overall1 = serve_rating1 * 0.5 + return_rating1 * 0.5 + stats1['win_pct'] * 0.3
        overall2 = serve_rating2 * 0.5 + return_rating2 * 0.5 + stats2['win_pct'] * 0.3
        features.extend([serve_rating1 - serve_rating2, return_rating1 - return_rating2, overall1, overall2])
        
        return np.array(features, dtype=np.float32)


# ============================================
# Live Match Probability
# ============================================

class LiveMatchPredictor:
    """Calculate win probability during a live match."""
    
    def __init__(self, p1_serve_pct: float = 0.65, p2_serve_pct: float = 0.65, best_of: int = 3):
        self.p1_serve = p1_serve_pct
        self.p2_serve = p2_serve_pct
        self.best_of = best_of
        self.sets_to_win = (best_of + 1) // 2
        self._cache = {}
    
    def prob_win_game(self, p1_serving: bool) -> float:
        p = self.p1_serve if p1_serving else (1 - self.p2_serve)
        p_deuce_win = (p * p) / (p * p + (1-p) * (1-p)) if (p * p + (1-p) * (1-p)) > 0 else 0.5
        from math import comb
        p_win_4_0 = p ** 4
        p_win_4_1 = comb(4, 1) * (p ** 4) * ((1-p) ** 1)
        p_win_4_2 = comb(5, 2) * (p ** 4) * ((1-p) ** 2)
        p_deuce = comb(6, 3) * (p ** 3) * ((1-p) ** 3)
        return p_win_4_0 + p_win_4_1 + p_win_4_2 + p_deuce * p_deuce_win
    
    def prob_win_tiebreak(self) -> float:
        p_avg = (self.p1_serve + (1 - self.p2_serve)) / 2
        return self._race_to_n(p_avg, 7)
    
    def _race_to_n(self, p: float, n: int) -> float:
        from math import comb
        total = 0
        for i in range(n):
            total += comb(n - 1 + i, i) * (p ** n) * ((1-p) ** i)
        p_deuce = comb(2*(n-1), n-1) * (p ** (n-1)) * ((1-p) ** (n-1))
        p_deuce_win = (p * p) / (p * p + (1-p) * (1-p)) if (p * p + (1-p) * (1-p)) > 0 else 0.5
        total += p_deuce * p_deuce_win
        return min(total, 1.0)
    
    def prob_win_set(self, p1_games: int, p2_games: int, p1_serving: bool, is_tiebreak_set: bool = True) -> float:
        cache_key = ('set', p1_games, p2_games, p1_serving, is_tiebreak_set)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if p1_games >= 6 and p1_games - p2_games >= 2:
            return 1.0
        if p2_games >= 6 and p2_games - p1_games >= 2:
            return 0.0
        if p1_games == 7:
            return 1.0
        if p2_games == 7:
            return 0.0
        if p1_games == 6 and p2_games == 6 and is_tiebreak_set:
            return self.prob_win_tiebreak()
        
        p_win_game = self.prob_win_game(p1_serving)
        prob_if_win = self.prob_win_set(p1_games + 1, p2_games, not p1_serving, is_tiebreak_set)
        prob_if_lose = self.prob_win_set(p1_games, p2_games + 1, not p1_serving, is_tiebreak_set)
        
        result = p_win_game * prob_if_win + (1 - p_win_game) * prob_if_lose
        self._cache[cache_key] = result
        return result
    
    def prob_win_match(self, p1_sets: int, p2_sets: int, p1_games: int, p2_games: int, p1_serving: bool) -> float:
        cache_key = ('match', p1_sets, p2_sets, p1_games, p2_games, p1_serving)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if p1_sets >= self.sets_to_win:
            return 1.0
        if p2_sets >= self.sets_to_win:
            return 0.0
        
        p_win_set = self.prob_win_set(p1_games, p2_games, p1_serving)
        prob_if_win_set = self.prob_win_match(p1_sets + 1, p2_sets, 0, 0, p1_serving)
        prob_if_lose_set = self.prob_win_match(p1_sets, p2_sets + 1, 0, 0, p1_serving)
        
        result = p_win_set * prob_if_win_set + (1 - p_win_set) * prob_if_lose_set
        self._cache[cache_key] = result
        return result


# ============================================
# Edge Realization Analysis (Enhanced)
# ============================================

class EdgeAnalyzer:
    """
    Analyze if an edge can be realized given odds movement.
    
    Key insights:
    1. If underdog (20% market) loses first game, odds drop to ~15%
    2. If favorite (80% market) wins first game, odds rise to ~85%
    3. Small edges on underdogs are DANGEROUS - odds move against you fast
    4. Best bets: Favorites the market undervalues, or big edges on underdogs
    
    This analyzer predicts odds movement and filters out bets that
    are likely to become worse before you can act.
    """
    
    # Typical odds movement per game (based on market behavior)
    ODDS_MOVEMENT_PER_GAME = {
        # (current_prob_range): (move_if_win, move_if_lose)
        (0.0, 0.15): (0.04, -0.03),   # Big underdog: gains 4% on win, loses 3% on loss
        (0.15, 0.25): (0.05, -0.04),  # Underdog: gains 5%, loses 4%
        (0.25, 0.35): (0.05, -0.05),  # Slight underdog: symmetric
        (0.35, 0.50): (0.04, -0.04),  # Coin flip: symmetric
        (0.50, 0.65): (0.04, -0.04),  # Slight favorite
        (0.65, 0.75): (0.04, -0.05),  # Favorite: gains less, loses more
        (0.75, 0.85): (0.03, -0.05),  # Big favorite
        (0.85, 1.0): (0.02, -0.04),   # Huge favorite: little to gain
    }
    
    def __init__(self, live_pred: LiveMatchPredictor):
        self.live_pred = live_pred
    
    def _get_odds_movement(self, prob: float) -> tuple:
        """Get expected odds movement for a probability level."""
        for (low, high), movement in self.ODDS_MOVEMENT_PER_GAME.items():
            if low <= prob < high:
                return movement
        return (0.04, -0.04)  # Default
    
    def predict_odds_after_game(self, market_prob: float, model_prob: float, 
                                p1_serving: bool) -> dict:
        """
        Predict where odds will be after the next game.
        
        Returns expected market odds after P1 wins or loses next game.
        """
        # Get probability of P1 winning next game (from our model)
        p_game = self.live_pred.prob_win_game(p1_serving)
        
        # Get typical market movement
        move_win, move_lose = self._get_odds_movement(market_prob)
        
        # Predict market odds after next game
        market_if_win = min(0.95, market_prob + move_win)
        market_if_lose = max(0.05, market_prob + move_lose)
        
        # Our model's probability also changes
        # Use a simplified model: +3% if win, -3% if lose (roughly)
        model_move = 0.03
        model_if_win = min(0.95, model_prob + model_move)
        model_if_lose = max(0.05, model_prob - model_move)
        
        return {
            'p_game_win': p_game,
            'market_if_win': market_if_win,
            'market_if_lose': market_if_lose,
            'model_if_win': model_if_win,
            'model_if_lose': model_if_lose,
            'edge_if_win': model_if_win - market_if_win,
            'edge_if_lose': model_if_lose - market_if_lose,
        }
    
    def analyze_edge_realization(self, model_prob: float, market_prob: float,
                                 p1_serving: bool, bankroll: float) -> dict:
        """
        Comprehensive edge analysis with odds movement prediction.
        
        Key decision factors:
        1. Current edge (model - market)
        2. Expected edge after next game
        3. Probability edge survives (doesn't go negative)
        4. Risk/reward ratio
        """
        
        # Current edge
        edge = model_prob - market_prob
        
        # Predict odds after next game
        prediction = self.predict_odds_after_game(market_prob, model_prob, p1_serving)
        
        p_game = prediction['p_game_win']
        edge_if_win = prediction['edge_if_win']
        edge_if_lose = prediction['edge_if_lose']
        
        # Expected edge after next game
        expected_edge_after = p_game * edge_if_win + (1 - p_game) * edge_if_lose
        
        # Probability that edge remains positive after next game
        prob_edge_survives = 0
        if edge_if_win > 0 and edge_if_lose > 0:
            prob_edge_survives = 1.0
        elif edge_if_win > 0:
            prob_edge_survives = p_game
        elif edge_if_lose > 0:
            prob_edge_survives = 1 - p_game
        
        # ============================================
        # SMART FILTERING LOGIC
        # ============================================
        
        skip_reasons = []
        bet_quality = "GOOD"
        
        # Rule 1: Edge too small to begin with
        MIN_EDGE = 0.05  # 5% minimum edge
        if abs(edge) < MIN_EDGE:
            skip_reasons.append(f"Edge too small ({edge*100:+.1f}% < 5%)")
            bet_quality = "SKIP"
        
        # Rule 2: Edge likely to disappear
        # If betting on underdog and they're likely to lose, edge will shrink
        if edge > 0 and model_prob < 0.5:
            # We think underdog is undervalued
            # But if they lose next game (likely), our edge shrinks
            if edge_if_lose < MIN_EDGE * 0.5:
                skip_reasons.append(f"Underdog edge vanishes on game loss ({edge_if_lose*100:+.1f}%)")
                if edge < 0.10:  # Only skip if edge isn't huge
                    bet_quality = "RISKY"
        
        # Rule 3: Expected edge after game is negative
        if expected_edge_after < 0:
            skip_reasons.append(f"Expected edge goes negative ({expected_edge_after*100:+.1f}%)")
            bet_quality = "SKIP"
        
        # Rule 4: Low probability of edge surviving
        if prob_edge_survives < 0.4 and edge < 0.12:
            skip_reasons.append(f"Only {prob_edge_survives*100:.0f}% chance edge survives")
            bet_quality = "RISKY"
        
        # Rule 5: Odds moving in wrong direction
        # If market is already moving against us, don't chase
        if edge > 0 and expected_edge_after < edge * 0.5:
            skip_reasons.append(f"Edge decaying fast ({edge*100:.1f}% → {expected_edge_after*100:.1f}%)")
            if bet_quality == "GOOD":
                bet_quality = "MARGINAL"
        
        # Rule 6: Great bet - edge increases or stable
        if expected_edge_after >= edge * 0.8 and edge >= MIN_EDGE:
            bet_quality = "EXCELLENT"
        
        # ============================================
        # CALCULATE BET SIZE
        # ============================================
        
        # Kelly calculation
        decimal_odds = 1 / market_prob if market_prob > 0 else 10
        b = decimal_odds - 1
        kelly_raw = (b * model_prob - (1 - model_prob)) / b if b > 0 else 0
        kelly_raw = max(0, kelly_raw)
        
        # Adjust Kelly based on bet quality
        kelly_multiplier = {
            "EXCELLENT": 0.6,   # 60% Kelly for great bets
            "GOOD": 0.5,       # 50% Kelly (half Kelly)
            "MARGINAL": 0.25,  # 25% Kelly for marginal
            "RISKY": 0.1,      # 10% Kelly for risky
            "SKIP": 0,         # No bet
        }
        
        kelly_adjusted = kelly_raw * kelly_multiplier.get(bet_quality, 0)
        bet_amount = bankroll * kelly_adjusted
        
        # ============================================
        # URGENCY & RECOMMENDATION
        # ============================================
        
        if bet_quality == "SKIP":
            urgency = "NONE"
            urgency_note = "Don't bet - " + "; ".join(skip_reasons[:2])
            recommendation = "NO BET"
        elif bet_quality == "RISKY":
            urgency = "LOW"
            urgency_note = "Risky - " + skip_reasons[0] if skip_reasons else "Consider skipping"
            recommendation = f"RISKY - ${bet_amount:.2f} (reduced size)"
        elif bet_quality == "MARGINAL":
            urgency = "LOW"
            urgency_note = "Marginal edge - wait for better opportunity"
            recommendation = f"MARGINAL - ${bet_amount:.2f}"
        elif bet_quality == "EXCELLENT":
            urgency = "HIGH"
            urgency_note = "Strong edge, stable - act now!"
            recommendation = f"BET NOW - ${bet_amount:.2f}"
        else:  # GOOD
            # Check if edge is time-sensitive
            if expected_edge_after < edge * 0.7:
                urgency = "HIGH"
                urgency_note = "Edge decaying - act quickly"
            else:
                urgency = "MEDIUM"
                urgency_note = "Solid edge - can wait 1-2 games"
            recommendation = f"BET - ${bet_amount:.2f}"
        
        # ============================================
        # ALTERNATIVE: Should we bet the other side?
        # ============================================
        
        # Check if betting against our model is actually better
        # (i.e., if market overvalues someone we think is only slightly better)
        opposite_edge = -edge
        opposite_analysis = None
        
        if bet_quality in ["SKIP", "RISKY"] and abs(opposite_edge) > MIN_EDGE:
            # Market might be wrong in the other direction
            opposite_market = 1 - market_prob
            opposite_model = 1 - model_prob
            
            opp_pred = self.predict_odds_after_game(opposite_market, opposite_model, not p1_serving)
            opp_expected_edge = opp_pred['p_game_win'] * opp_pred['edge_if_win'] + \
                               (1 - opp_pred['p_game_win']) * opp_pred['edge_if_lose']
            
            if opp_expected_edge > expected_edge_after and opp_expected_edge > MIN_EDGE * 0.5:
                opposite_analysis = {
                    'edge': opposite_edge,
                    'expected_edge': opp_expected_edge,
                    'note': "Consider betting OTHER player instead"
                }
        
        return {
            'edge': edge,
            'expected_edge_after': expected_edge_after,
            'prob_edge_survives': prob_edge_survives,
            'edge_if_win': edge_if_win,
            'edge_if_lose': edge_if_lose,
            'bet_quality': bet_quality,
            'skip_reasons': skip_reasons,
            'urgency': urgency,
            'urgency_note': urgency_note,
            'kelly_raw': kelly_raw,
            'kelly_adjusted': kelly_adjusted,
            'bet_amount': bet_amount,
            'recommendation': recommendation,
            'model_prob': model_prob,
            'market_prob': market_prob,
            'opposite_analysis': opposite_analysis,
            'prediction': prediction,
        }


# ============================================
# Main Prediction Model
# ============================================

class TennisPredictor:
    """Complete tennis prediction system."""
    
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.feature_engine = FeatureEngine(loader)
        self.model = None
        self.calibrated_model = None
        self.scaler = None
        self.is_trained = False
        self.kalshi = KalshiAPI()
    
    def train(self, test_year: int = None) -> Dict:
        """Train the model."""
        if not HAS_ML:
            raise ImportError("Install: pip install xgboost scikit-learn numpy")
        
        print("Preparing training data...")
        
        self.feature_engine.elo = defaultdict(lambda: 1500.0)
        self.feature_engine.surface_elo = {s: defaultdict(lambda: 1500.0) for s in Surface}
        self.feature_engine._stats_cache = {}
        
        X_list, y_list = [], []
        matches = sorted(self.loader.matches, key=lambda m: m.tourney_date)
        
        if test_year:
            train_matches = [m for m in matches if m.tourney_date.year < test_year]
            test_matches = [m for m in matches if m.tourney_date.year == test_year]
        else:
            split_idx = int(len(matches) * 0.8)
            train_matches = matches[:split_idx]
            test_matches = matches[split_idx:]
        
        print(f"Training on {len(train_matches)} matches...")
        
        for i, match in enumerate(train_matches):
            if not match.winner_rank or not match.loser_rank:
                continue
            if match.winner_rank > 300 and match.loser_rank > 300:
                continue
            
            if random.random() < 0.5:
                p1_id, p2_id = match.winner_id, match.loser_id
                p1_rank, p2_rank = match.winner_rank, match.loser_rank
                p1_pts, p2_pts = match.winner_rank_points, match.loser_rank_points
                label = 1
            else:
                p1_id, p2_id = match.loser_id, match.winner_id
                p1_rank, p2_rank = match.loser_rank, match.winner_rank
                p1_pts, p2_pts = match.loser_rank_points, match.winner_rank_points
                label = 0
            
            try:
                features = self.feature_engine.extract(
                    p1_id, p2_id, match.surface, match.tourney_level,
                    match.tourney_date, p1_rank, p2_rank, p1_pts, p2_pts,
                    match.best_of, match.round_name, 32
                )
                X_list.append(features)
                y_list.append(label)
            except:
                pass
            
            self.feature_engine.update_elo(match.winner_id, match.loser_id, match.surface)
            
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1} matches...")
        
        X = np.nan_to_num(np.array(X_list), nan=0.0, posinf=1.0, neginf=-1.0)
        y = np.array(y_list)
        
        print(f"  Training samples: {len(X)}, Features: {X.shape[1]}")
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print("Training XGBoost...")
        
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=7,
            learning_rate=0.03,
            n_estimators=800,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            gamma=0.1,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
            eval_metric='logloss',
        )
        
        split = int(len(X_scaled) * 0.85)
        self.model.fit(X_scaled[:split], y[:split], eval_set=[(X_scaled[split:], y[split:])], verbose=False)
        
        print("Calibrating...")
        self.calibrated_model = CalibratedClassifierCV(self.model, method='isotonic', cv='prefit')
        self.calibrated_model.fit(X_scaled[split:], y[split:])
        
        self.is_trained = True
        
        print("\nTesting...")
        return self._test(test_matches)
    
    def _test(self, test_matches: List[Match]) -> Dict:
        X_test, y_test = [], []
        
        for match in test_matches:
            if not match.winner_rank or not match.loser_rank:
                continue
            if match.winner_rank > 300 and match.loser_rank > 300:
                continue
            
            if random.random() < 0.5:
                p1_id, p2_id = match.winner_id, match.loser_id
                p1_rank, p2_rank = match.winner_rank, match.loser_rank
                p1_pts, p2_pts = match.winner_rank_points, match.loser_rank_points
                label = 1
            else:
                p1_id, p2_id = match.loser_id, match.winner_id
                p1_rank, p2_rank = match.loser_rank, match.winner_rank
                p1_pts, p2_pts = match.loser_rank_points, match.winner_rank_points
                label = 0
            
            try:
                features = self.feature_engine.extract(
                    p1_id, p2_id, match.surface, match.tourney_level,
                    match.tourney_date, p1_rank, p2_rank, p1_pts, p2_pts,
                    match.best_of, match.round_name, 32
                )
                X_test.append(features)
                y_test.append(label)
            except:
                continue
            
            self.feature_engine.update_elo(match.winner_id, match.loser_id, match.surface)
        
        if not X_test:
            return {'error': 'No test data'}
        
        X_test = np.nan_to_num(np.array(X_test), nan=0.0, posinf=1.0, neginf=-1.0)
        X_test_scaled = self.scaler.transform(X_test)
        y_test = np.array(y_test)
        
        y_proba = self.calibrated_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        brier = np.mean((y_proba - y_test) ** 2)
        eps = 1e-15
        y_proba_clipped = np.clip(y_proba, eps, 1 - eps)
        logloss = -np.mean(y_test * np.log(y_proba_clipped) + (1 - y_test) * np.log(1 - y_proba_clipped))
        
        results = {'matches': len(y_test), 'accuracy': accuracy, 'brier_score': brier, 'log_loss': logloss}
        
        print(f"\nResults on {results['matches']} test matches:")
        print(f"  Accuracy:    {results['accuracy']:.1%}")
        print(f"  Brier Score: {results['brier_score']:.4f}")
        print(f"  Log Loss:    {results['log_loss']:.4f}")
        
        return results
    
    def predict_match(self, p1_id: str, p2_id: str, surface: Surface,
                     tourney_level: TournamentLevel = TournamentLevel.OTHER,
                     p1_rank: int = None, p2_rank: int = None,
                     best_of: int = 3, round_name: str = "R32") -> Dict:
        """Predict match outcome."""
        if not self.is_trained:
            raise RuntimeError("Model not trained!")
        
        features = self.feature_engine.extract(
            p1_id, p2_id, surface, tourney_level,
            datetime.now(), p1_rank, p2_rank, None, None, 
            best_of, round_name, 32
        )
        
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prob = self.calibrated_model.predict_proba(features_scaled)[0, 1]
        
        h2h = self.loader.get_h2h(p1_id, p2_id)
        
        # Estimate serve percentages
        stats1 = self.feature_engine._get_extended_stats(p1_id, surface, datetime.now())
        stats2 = self.feature_engine._get_extended_stats(p2_id, surface, datetime.now())
        
        p1_serve = stats1.get('first_serve_won', 0.7) * stats1.get('first_serve_pct', 0.6) + \
                   stats1.get('second_serve_won', 0.5) * (1 - stats1.get('first_serve_pct', 0.6))
        p2_serve = stats2.get('first_serve_won', 0.7) * stats2.get('first_serve_pct', 0.6) + \
                   stats2.get('second_serve_won', 0.5) * (1 - stats2.get('first_serve_pct', 0.6))
        
        p1_serve = min(0.75, max(0.55, p1_serve - stats2.get('return_won_pct', 0.35) + 0.35))
        p2_serve = min(0.75, max(0.55, p2_serve - stats1.get('return_won_pct', 0.35) + 0.35))
        
        return {
            'p1_win_prob': float(prob),
            'p2_win_prob': float(1 - prob),
            'h2h': f"{h2h['p1_wins']}-{h2h['p2_wins']}",
            'p1_serve_pct': p1_serve,
            'p2_serve_pct': p2_serve,
            'best_of': best_of,
        }
    
    def get_feature_importance(self) -> List[Tuple[str, float]]:
        if not self.is_trained:
            return []
        importances = self.model.feature_importances_
        result = list(zip(FeatureEngine.FEATURE_NAMES, importances))
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'calibrated_model': self.calibrated_model,
                'scaler': self.scaler,
                'elo': dict(self.feature_engine.elo),
                'surface_elo': {s: dict(v) for s, v in self.feature_engine.surface_elo.items()},
            }, f)
        print(f"Saved to {filepath}")
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.calibrated_model = data['calibrated_model']
        self.scaler = data['scaler']
        self.feature_engine.elo = defaultdict(lambda: 1500.0, data['elo'])
        self.feature_engine.surface_elo = {s: defaultdict(lambda: 1500.0, v) for s, v in data['surface_elo'].items()}
        self.is_trained = True
        print(f"Loaded from {filepath}")


# ============================================
# Interactive CLI
# ============================================

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Tennis Predictor v2 - Enhanced Edition")
        print("=" * 50)
        print("\nUsage:")
        print("  python tennis_predictor_v2.py <atp_dir> [wta_dir] [command]")
        print("\nCommands:")
        print("  train    - Train the model")
        print("  predict  - Interactive predictions")
        print("  features - Show feature importance")
        print("\nExample:")
        print("  python tennis_predictor_v2.py ./tennis_atp ./tennis_wta train")
        return
    
    atp_dir = sys.argv[1]
    wta_dir = None
    command = "train"
    
    if len(sys.argv) > 2:
        if sys.argv[2] in ["train", "predict", "features"]:
            command = sys.argv[2]
        else:
            wta_dir = sys.argv[2]
            if len(sys.argv) > 3:
                command = sys.argv[3]
    
    print(f"\nLoading data...")
    loader = DataLoader(atp_dir, wta_dir)
    loader.load(years=range(2005, 2026), include_wta=(wta_dir is not None))
    
    predictor = TennisPredictor(loader)
    model_file = "tennis_model_v2.pkl"
    
    if command == "train":
        dates = [m.tourney_date for m in loader.matches]
        latest_year = max(dates).year
        results = predictor.train(test_year=latest_year)
        
        print("\n" + "=" * 60)
        print("Top Features:")
        for name, imp in predictor.get_feature_importance()[:15]:
            bar = "█" * int(imp * 50)
            print(f"  {name:25s} {imp:.3f} {bar}")
        
        save = input("\nSave model? (y/n): ").strip().lower()
        if save == 'y':
            predictor.save(model_file)
    
    elif command == "predict":
        if os.path.exists(model_file):
            predictor.load(model_file)
        else:
            print("Training model...")
            predictor.train()
            predictor.save(model_file)
        
        interactive_session(predictor)
    
    elif command == "features":
        if os.path.exists(model_file):
            predictor.load(model_file)
        else:
            predictor.train()
        
        print("\nFeature Importance:")
        for name, imp in predictor.get_feature_importance():
            bar = "█" * int(imp * 50)
            print(f"  {name:25s} {imp:.4f} {bar}")


def interactive_session(predictor: TennisPredictor):
    """Main interactive session."""
    loader = predictor.loader
    
    while True:
        print("\n" + "=" * 60)
        print("  TENNIS PREDICTOR v2")
        print("=" * 60)
        print("\n  1. Create new match profile")
        print("  2. Load existing profile")
        print("  3. Quick prediction")
        print("  4. Check Kalshi markets")
        print("  5. Setup Kalshi API (for trading)")
        print("  6. Today's matches (bulk create profiles)")
        print("  7. Test API connection")
        print("  8. LIVE MONITOR (auto-scan for edges)")
        print("  q. Quit")
        
        # Show balance if authenticated (with error handling)
        try:
            if predictor.kalshi.is_authenticated:
                bal = predictor.kalshi.get_balance()
                if bal.get('balance', 0) > 0:
                    print(f"\n  [Kalshi: ${bal.get('balance', 0):,.2f} available]")
                else:
                    print(f"\n  [Kalshi: Connected]")
        except Exception:
            pass  # Don't crash if balance check fails
        
        choice = input("\n  Select: ").strip().lower()
        
        if choice == 'q' or choice == 'quit' or choice == 'exit':
            print("\n  Goodbye!")
            break
        
        try:
            if choice == "1":
                create_profile(predictor)
            elif choice == "2":
                load_profile(predictor)
            elif choice == "3":
                quick_predict(predictor)
            elif choice == "4":
                check_kalshi(predictor)
            elif choice == "5":
                if setup_kalshi_credentials():
                    predictor.kalshi = KalshiAPI()
            elif choice == "6":
                todays_matches(predictor)
            elif choice == "7":
                test_api_connection(predictor)
            elif choice == "8":
                live_monitor(predictor)
            elif choice == "":
                continue  # Empty input, just show menu again
            else:
                print(f"  Unknown option: {choice}")
        except KeyboardInterrupt:
            print("\n  Cancelled.")
        except Exception as e:
            print(f"\n  Error: {e}")
            print("  Try again or select a different option.")
    else:
        quick_predict(predictor)


def test_api_connection(predictor: TennisPredictor):
    """Test and diagnose API connections."""
    print("\n" + "=" * 60)
    print("  API CONNECTION TEST")
    print("=" * 60)
    
    # Test 1: Check if requests is installed
    print("\n  1. Checking dependencies...")
    try:
        import requests
        print("     ✓ requests library installed")
    except ImportError:
        print("     ✗ requests NOT installed")
        print("       Run: pip3 install requests")
        return
    
    # Test 2: Check if cryptography is installed (for trading)
    try:
        from cryptography.hazmat.primitives import serialization
        print("     ✓ cryptography library installed (trading enabled)")
    except ImportError:
        print("     ⚠ cryptography NOT installed (trading disabled)")
        print("       Run: pip3 install cryptography")
    
    # Test 3: Test Kalshi API connection
    print("\n  2. Testing Kalshi API connection...")
    
    kalshi = predictor.kalshi
    
    try:
        url = f"{kalshi.base_url}/exchange/status"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"     ✓ Kalshi API reachable")
            print(f"       URL: {kalshi.base_url}")
            print(f"       Exchange status: {data.get('exchange_active', 'unknown')}")
            print(f"       Trading active: {data.get('trading_active', 'unknown')}")
        else:
            print(f"     ✗ Kalshi API returned status {response.status_code}")
            print(f"       Response: {response.text[:200]}")
    except requests.exceptions.Timeout:
        print("     ✗ Kalshi API timed out")
        print("       Check your internet connection")
    except requests.exceptions.ConnectionError as e:
        print("     ✗ Could not connect to Kalshi API")
        print(f"       Error: {e}")
    except Exception as e:
        print(f"     ✗ Error: {e}")
    
    # Test 4: Fetch markets
    print("\n  3. Fetching market data...")
    
    try:
        url = f"{kalshi.base_url}/markets"
        params = {'status': 'open', 'limit': 10}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            markets = data.get('markets', [])
            print(f"     ✓ Successfully fetched markets")
            print(f"       Sample markets returned: {len(markets)}")
            
            if markets:
                print(f"\n     Sample market:")
                m = markets[0]
                print(f"       Ticker: {m.get('ticker')}")
                print(f"       Title: {m.get('title', '')[:50]}...")
                print(f"       Status: {m.get('status')}")
                print(f"       Yes bid: {m.get('yes_bid')}¢")
        else:
            print(f"     ✗ Failed to fetch markets: {response.status_code}")
    except Exception as e:
        print(f"     ✗ Error fetching markets: {e}")
    
    # Test 5: Search for tennis markets
    print("\n  4. Searching for tennis markets...")
    
    try:
        url = f"{kalshi.base_url}/markets"
        params = {'status': 'open', 'limit': 500}
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            all_markets = data.get('markets', [])
            
            # Search for tennis
            tennis_keywords = ['tennis', 'australian open', 'french open', 'wimbledon', 
                             'us open', 'atp', 'wta', 'sinner', 'alcaraz', 'djokovic',
                             'swiatek', 'sabalenka', 'gauff']
            
            tennis_markets = []
            for m in all_markets:
                title = m.get('title', '').lower()
                ticker = m.get('ticker', '').lower()
                if any(kw in title or kw in ticker for kw in tennis_keywords):
                    tennis_markets.append(m)
            
            print(f"     Total open markets: {len(all_markets)}")
            print(f"     Tennis markets found: {len(tennis_markets)}")
            
            if tennis_markets:
                print(f"\n     Tennis markets:")
                for m in tennis_markets[:10]:
                    yes = m.get('yes_bid', 0)
                    print(f"       • {m.get('title', '')[:45]}...")
                    print(f"         Ticker: {m.get('ticker')} | Yes: {yes}¢")
            else:
                print("\n     No tennis markets currently available.")
                print("     This is normal if:")
                print("       - No matches are scheduled today")
                print("       - Markets haven't been posted yet")
                print("       - Tournament is between sessions")
                
                # Show other sports for reference
                sports_keywords = {'basketball': 0, 'football': 0, 'baseball': 0, 
                                  'soccer': 0, 'hockey': 0, 'politics': 0, 'crypto': 0}
                for m in all_markets:
                    title = m.get('title', '').lower()
                    for sport in sports_keywords:
                        if sport in title:
                            sports_keywords[sport] += 1
                
                active_cats = {k: v for k, v in sports_keywords.items() if v > 0}
                if active_cats:
                    print(f"\n     Other active categories:")
                    for cat, count in sorted(active_cats.items(), key=lambda x: -x[1])[:5]:
                        print(f"       • {cat}: {count} markets")
        else:
            print(f"     ✗ Failed: {response.status_code}")
    except Exception as e:
        print(f"     ✗ Error: {e}")
    
    # Test 6: Check authentication
    print("\n  5. Checking authentication...")
    
    if kalshi.is_authenticated:
        print("     ✓ API credentials configured")
        print(f"       Key ID: {kalshi.api_key_id[:12]}...")
        
        # Try to get balance
        try:
            balance = kalshi.get_balance()
            if balance.get('balance', 0) > 0 or balance.get('available', 0) >= 0:
                print(f"     ✓ Authentication working")
                print(f"       Balance: ${balance.get('balance', 0):,.2f}")
                print(f"       Available: ${balance.get('available', 0):,.2f}")
            else:
                print("     ⚠ Could not fetch balance (may need to re-authenticate)")
        except Exception as e:
            print(f"     ⚠ Balance check failed: {e}")
    else:
        print("     ⚠ Not authenticated (trading disabled)")
        print("       To enable trading:")
        print("       1. Go to kalshi.com → Settings → API")
        print("       2. Create API key and download private key")
        print("       3. Set environment variables:")
        print("          export KALSHI_API_KEY_ID='your-key-id'")
        print("          export KALSHI_PRIVATE_KEY_PATH='/path/to/key.pem'")
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    issues = []
    
    if not HAS_REQUESTS:
        issues.append("Install requests: pip3 install requests")
    
    try:
        response = requests.get(f"{kalshi.base_url}/exchange/status", timeout=5)
        if response.status_code != 200:
            issues.append("Kalshi API not responding properly")
    except:
        issues.append("Cannot connect to Kalshi API")
    
    if not kalshi.is_authenticated:
        issues.append("Trading disabled (no API credentials)")
    
    if issues:
        print("\n  Issues found:")
        for issue in issues:
            print(f"    ⚠ {issue}")
    else:
        print("\n  ✓ All systems operational!")
        print("    - API connection working")
        print("    - Market data accessible")
        if kalshi.is_authenticated:
            print("    - Trading enabled")
    
    print("\n" + "=" * 60)
    input("\n  Press Enter to continue...")


def todays_matches(predictor: TennisPredictor):
    """Create profiles for today's upcoming matches."""
    loader = predictor.loader
    
    print("\n" + "=" * 60)
    print("  TODAY'S MATCHES")
    print("=" * 60)
    
    print("\n  Options:")
    print("    1. Fetch from Kalshi (tennis markets)")
    print("    2. Enter matches manually (quick entry)")
    print("    3. Load from file (matches.txt)")
    
    choice = input("\n  Select: ").strip()
    
    if choice == "1":
        fetch_kalshi_matches(predictor)
    elif choice == "2":
        manual_match_entry(predictor)
    elif choice == "3":
        load_matches_file(predictor)
    else:
        manual_match_entry(predictor)


def fetch_kalshi_matches(predictor: TennisPredictor):
    """Automatically fetch tennis matches from Kalshi and create profiles."""
    print("\n  Fetching Kalshi tennis markets...")
    
    if not predictor.kalshi.session:
        print("  ✗ Kalshi API not available. Install: pip3 install requests")
        return
    
    try:
        # Tennis series tickers on Kalshi
        tennis_series = [
            'KXATPMATCH',     # ATP Tennis Match
            'KXWTAMATCH',     # WTA Tennis Match
            'KXATPGAME',      # ATP Tennis Winner (game by game)
            'KXWTAGAME',      # WTA Tennis Winner (game by game)
        ]
        
        all_markets = []
        
        print("  Fetching from tennis series...")
        for series in tennis_series:
            url = f"{predictor.kalshi.base_url}/markets"
            params = {'series_ticker': series, 'status': 'open', 'limit': 100}
            
            response = predictor.kalshi.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                markets = data.get('markets', [])
                if markets:
                    print(f"    {series}: {len(markets)} markets")
                    all_markets.extend(markets)
        
        if not all_markets:
            print("\n  ✗ No tennis markets found.")
            print("  Try option 2 for manual entry.")
            return
        
        # Group markets by match (each match has 2 markets - one per player)
        # Extract unique matches from tickers like KXATPMATCH-26JAN08KOVMPE-MPE
        matches = {}
        for m in all_markets:
            ticker = m.get('ticker', '')
            title = m.get('title', '')
            
            # Extract match identifier (everything before the last dash)
            parts = ticker.rsplit('-', 1)
            if len(parts) == 2:
                match_id = parts[0]
                player_code = parts[1]
                
                if match_id not in matches:
                    matches[match_id] = {
                        'markets': [],
                        'players': [],
                        'match_title': '',
                    }
                
                matches[match_id]['markets'].append(m)
                
                # Extract player name from title: "Will X win the Y vs Z match?"
                if 'Will ' in title and ' win ' in title:
                    player_name = title.split('Will ')[1].split(' win ')[0].strip()
                    yes_bid = m.get('yes_bid', 0)
                    matches[match_id]['players'].append({
                        'name': player_name,
                        'yes_bid': yes_bid,
                        'ticker': ticker,
                    })
                
                # Extract match title
                if ' : ' in title:
                    match_title = title.split(' the ')[1].split(' match')[0] if ' the ' in title else ''
                    matches[match_id]['match_title'] = match_title
        
        print(f"\n  ✓ Found {len(matches)} tennis matches!\n")
        
        # Display matches
        match_list = list(matches.items())
        for i, (match_id, match_data) in enumerate(match_list[:25]):
            players = match_data['players']
            if len(players) >= 2:
                p1 = players[0]
                p2 = players[1]
                print(f"  {i+1}. {p1['name']} ({p1['yes_bid']}¢) vs {p2['name']} ({p2['yes_bid']}¢)")
            elif len(players) == 1:
                print(f"  {i+1}. {players[0]['name']} ({players[0]['yes_bid']}¢) vs ???")
            else:
                print(f"  {i+1}. {match_data['match_title']}")
        
        if len(match_list) > 25:
            print(f"\n  ... and {len(match_list) - 25} more")
        
        proceed = input(f"\n  Create profiles for these {len(matches)} matches? (y/n): ").strip().lower()
        if proceed != 'y':
            print("  Cancelled.")
            return
        
        # Get settings
        print("\n  Settings for all matches:")
        surface = Surface.from_string(input("    Surface (Hard/Clay/Grass) [Hard]: ").strip() or "Hard")
        level_str = input("    Level (G=Grand Slam, M=Masters, O=Other) [O]: ").strip().upper() or "O"
        level = TournamentLevel.GRAND_SLAM if level_str == 'G' else TournamentLevel.MASTERS if level_str == 'M' else TournamentLevel.OTHER
        bankroll = float(input("    Bankroll [$] [1000]: ").strip() or "1000")
        
        print(f"\n  Creating profiles...")
        print("  " + "-" * 50)
        
        created = 0
        failed = []
        
        for match_id, match_data in match_list:
            players = match_data['players']
            if len(players) < 2:
                continue
            
            p1_name = players[0]['name']
            p2_name = players[1]['name']
            p1_ticker = players[0]['ticker']
            
            # Detect tour from ticker
            tour = "ATP" if "ATP" in match_id.upper() else "WTA"
            
            # Create profile with Kalshi ticker
            profile = create_profile_for_match(
                predictor, p1_name, p2_name,
                surface, level, tour, bankroll, p1_ticker
            )
            
            if profile:
                # Store both player tickers
                profile['kalshi_p1_ticker'] = players[0]['ticker']
                profile['kalshi_p2_ticker'] = players[1]['ticker']
                profile['kalshi_p1_odds'] = players[0]['yes_bid'] / 100
                profile['kalshi_p2_odds'] = players[1]['yes_bid'] / 100
                created += 1
            else:
                failed.append({'p1': p1_name, 'p2': p2_name})
        
        print("  " + "-" * 50)
        print(f"\n  ✓ Created {created} profiles")
        
        if failed:
            print(f"  ✗ Failed: {len(failed)}")
            for f in failed[:5]:
                print(f"      {f['p1']} vs {f['p2']}")
        
        if created > 0:
            print("\n  Use option 2 to load and track matches.")
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n  Try option 2 (manual entry) instead.")


def fetch_kalshi_matches_fallback(predictor: TennisPredictor):
    """Fallback method: scan all markets for tennis keywords."""
    print("  Scanning all markets for tennis...")
    
    try:
        url = f"{predictor.kalshi.base_url}/markets"
        params = {'status': 'open', 'limit': 1000}
        
        response = predictor.kalshi.session.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        all_markets = data.get('markets', [])
        print(f"  Scanning {len(all_markets)} open markets...")
        
        # Debug: show first few tickers
        print("  Sample tickers:")
        for m in all_markets[:5]:
            print(f"    {m.get('ticker')}: {m.get('title', '')[:40]}...")
        
        tennis_markets = []
        
        # Tennis keywords to search for
        tennis_keywords = [
            'tennis', 'atp', 'wta', 'auckland', 'brisbane', 'australian open',
            'french open', 'wimbledon', 'us open', 'roland garros', 'united cup'
        ]
        
        tennis_players = [
            'marcinko', 'eala', 'norrie', 'kovacevic', 'sabalenka', 'cirstea',
            'linette', 'cocciaretto', 'rybakina', 'badosa', 'sinner', 'alcaraz',
            'djokovic', 'zverev', 'medvedev', 'rune', 'fritz', 'tsitsipas',
            'swiatek', 'gauff', 'pegula', 'zheng', 'keys', 'navarro', 'muchova'
        ]
        
        for market in all_markets:
            title = market.get('title', '').lower()
            ticker = market.get('ticker', '').lower()
            event_ticker = market.get('event_ticker', '').lower()
            subtitle = market.get('subtitle', '').lower()
            
            combined = f"{title} {ticker} {event_ticker} {subtitle}"
            
            is_tennis = False
            
            # Check keywords
            if any(kw in combined for kw in tennis_keywords):
                is_tennis = True
            
            # Check player names
            if any(player in combined for player in tennis_players):
                is_tennis = True
            
            if is_tennis:
                tennis_markets.append(market)
        
        if not tennis_markets:
            print("\n  ✗ No tennis markets found.")
            print("  Try option 2 for manual entry.")
            return
        
        print(f"\n  ✓ Found {len(tennis_markets)} tennis markets!\n")
        process_tennis_markets(predictor, tennis_markets)
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        print("  Try option 2 (manual entry) instead.")


def process_tennis_markets(predictor: TennisPredictor, tennis_markets: list):
    """Process and create profiles for tennis markets."""
    
    # Show what we found
    for i, m in enumerate(tennis_markets[:20]):
        yes_bid = m.get('yes_bid', 0)
        if yes_bid:
            yes_pct = yes_bid / 100 * 100  # cents to percent
        else:
            yes_pct = m.get('last_price', 0)
        print(f"  {i+1}. {m.get('title', 'Unknown')}")
        print(f"      Yes: {yes_pct:.0f}% | Ticker: {m.get('ticker')}")
    
    if len(tennis_markets) > 20:
        print(f"\n  ... and {len(tennis_markets) - 20} more")
    
    proceed = input(f"\n  Create profiles for these matches? (y/n): ").strip().lower()
    if proceed != 'y':
        print("  Cancelled.")
        return
    
    # Get settings
    print("\n  Settings for all matches:")
    surface = Surface.from_string(input("    Surface (Hard/Clay/Grass) [Hard]: ").strip() or "Hard")
    level_str = input("    Level (G=Grand Slam, M=Masters, O=Other) [O]: ").strip().upper() or "O"
    level = TournamentLevel.GRAND_SLAM if level_str == 'G' else TournamentLevel.MASTERS if level_str == 'M' else TournamentLevel.OTHER
    bankroll = float(input("    Bankroll [$] [1000]: ").strip() or "1000")
    
    print(f"\n  Creating profiles...")
    print("  " + "-" * 50)
    
    created = 0
    failed = []
    
    for m in tennis_markets:
        title = m.get('title', '')
        ticker = m.get('ticker', '')
        
        # Parse players from title
        players = parse_players_from_title(title)
        if not players:
            subtitle = m.get('subtitle', '')
            if subtitle:
                players = parse_players_from_title(subtitle)
        
        if not players:
            print(f"  ? Could not parse: {title[:50]}...")
            failed.append(m)
            continue
        
        p1_name, p2_name = players
        
        # Auto-detect tour
        p1_results = predictor.loader.find_player(p1_name, "ATP")
        tour = "ATP" if p1_results else "WTA"
        
        # Create profile
        profile = create_profile_for_match(
            predictor, p1_name, p2_name,
            surface, level, tour, bankroll, ticker
        )
        
        if profile:
            created += 1
        else:
            failed.append(m)
    
    print("  " + "-" * 50)
    print(f"\n  ✓ Created {created} profiles")
    
    if failed:
        print(f"  ✗ Failed: {len(failed)}")
        for m in failed[:3]:
            print(f"      {m.get('title', '')[:50]}...")
    
    if created > 0:
        print("\n  Use option 2 to load and track matches.")


def parse_players_from_title(title: str) -> tuple:
    """Try to extract player names from market title."""
    import re
    
    original_title = title
    title_lower = title.lower()
    
    # Common Kalshi title patterns:
    # "Jannik Sinner to win vs Carlos Alcaraz"
    # "Will Sinner beat Alcaraz?"
    # "Sinner vs Alcaraz - Australian Open"
    # "Australian Open: Sinner vs Alcaraz"
    # "Sinner v Alcaraz"
    # "Sinner to defeat Alcaraz"
    
    patterns = [
        # "Player1 to win vs Player2" or "Player1 to win against Player2"
        r'(.+?)\s+to\s+win\s+(?:vs\.?|v\.?|against)\s+(.+?)(?:\s*[-–—]|\s*\?|$)',
        
        # "Player1 vs Player2" with optional suffix
        r'(.+?)\s+(?:vs\.?|v\.?)\s+(.+?)(?:\s*[-–—]|\s*\?|\s+match|\s+winner|$)',
        
        # "Will Player1 beat/defeat Player2"
        r'will\s+(.+?)\s+(?:beat|defeat)\s+(.+?)(?:\s*\?|$)',
        
        # "Player1 to defeat/beat Player2"
        r'(.+?)\s+to\s+(?:beat|defeat)\s+(.+?)(?:\s*[-–—]|\s*\?|$)',
        
        # "Tournament: Player1 vs Player2"
        r'(?:australian open|french open|wimbledon|us open|roland garros)[:\s]+(.+?)\s+(?:vs\.?|v\.?)\s+(.+?)(?:\s*$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, title_lower, re.IGNORECASE)
        if match:
            p1 = match.group(1).strip()
            p2 = match.group(2).strip()
            
            # Clean up names
            for remove in ['?', '-', '–', '—', 'match', 'winner', 'australian open', 
                          'french open', 'wimbledon', 'us open', 'roland garros',
                          'grand slam', 'atp', 'wta', ':']:
                p1 = p1.replace(remove, '').strip()
                p2 = p2.replace(remove, '').strip()
            
            # Title case
            p1 = p1.title()
            p2 = p2.title()
            
            # Validate - names should be reasonable length
            if 2 < len(p1) < 30 and 2 < len(p2) < 30:
                return (p1, p2)
    
    # Fallback: try to find "vs" or "v" anywhere
    if ' vs ' in title_lower or ' v ' in title_lower:
        # Split on vs/v
        parts = re.split(r'\s+(?:vs\.?|v\.?)\s+', title_lower)
        if len(parts) >= 2:
            p1 = parts[0].strip()
            p2 = parts[1].strip()
            
            # Remove common prefixes/suffixes
            for remove in ['will ', 'can ', '?', '-', ':', 'australian open', 
                          'french open', 'wimbledon', 'us open']:
                p1 = p1.replace(remove, '').strip()
                p2 = p2.replace(remove, '').strip()
            
            # Get last few words (likely the name)
            p1_words = p1.split()[-3:] if p1 else []
            p2_words = p2.split()[:3] if p2 else []
            
            p1 = ' '.join(p1_words).title()
            p2 = ' '.join(p2_words).title()
            
            if 2 < len(p1) < 30 and 2 < len(p2) < 30:
                return (p1, p2)
    
    return None


def manual_match_entry(predictor: TennisPredictor):
    """Quick manual entry for today's matches."""
    loader = predictor.loader
    
    print("\n  QUICK MATCH ENTRY")
    print("  " + "-" * 40)
    print("  Enter matches one per line:")
    print("  Format: Player1 vs Player2")
    print("  Example: Sinner vs Alcaraz")
    print("  (empty line or 'q' to finish)")
    
    matches = []
    while True:
        line = input("  > ").strip()
        if not line or line.lower() == 'q':
            break
        
        # Parse "Player1 vs Player2"
        if ' vs ' in line.lower():
            parts = line.lower().split(' vs ')
            if len(parts) == 2:
                p1 = parts[0].strip().title()
                p2 = parts[1].strip().title()
                matches.append((p1, p2))
                print(f"    Added: {p1} vs {p2}")
        else:
            print("    Invalid format. Use: Player1 vs Player2")
    
    if not matches:
        print("  No matches entered.")
        return
    
    print(f"\n  {len(matches)} matches to create:")
    for p1, p2 in matches:
        print(f"    • {p1} vs {p2}")
    
    proceed = input("\n  Continue? (y/n): ").strip().lower()
    if proceed != 'y':
        print("  Cancelled.")
        return
    
    # Get common settings
    surface = Surface.from_string(input("\n  Surface (Hard/Clay/Grass) [Hard]: ").strip() or "Hard")
    level_str = input("  Level (G/M/other) [G]: ").strip().upper() or "G"
    level = TournamentLevel.GRAND_SLAM if level_str == 'G' else TournamentLevel.MASTERS if level_str == 'M' else TournamentLevel.OTHER
    tour = input("  Tour (ATP/WTA) [ATP]: ").strip().upper() or "ATP"
    bankroll = float(input("  Bankroll [$] [1000]: ").strip() or "1000")
    
    created = 0
    failed = []
    for p1_name, p2_name in matches:
        profile = create_profile_for_match(
            predictor, p1_name, p2_name,
            surface, level, tour, bankroll
        )
        if profile:
            created += 1
        else:
            failed.append((p1_name, p2_name))
    
    print(f"\n  ✓ Created {created} profiles!")
    
    if failed:
        print(f"\n  ✗ Failed to create {len(failed)} profiles:")
        for p1, p2 in failed:
            print(f"    • {p1} vs {p2}")
        print("\n  Tips:")
        print("    - Try last name only: 'Rune' instead of 'Holger Rune'")
        print("    - Check spelling")
        print("    - Make sure tour is correct (ATP vs WTA)")
    
    if created > 0:
        print("\n  Use option 2 to load and track matches.")


def retry_match(predictor: TennisPredictor, p1_orig: str, p2_orig: str,
                surface: Surface, level: TournamentLevel, tour: str, bankroll: float):
    """Retry creating a match with interactive player search."""
    loader = predictor.loader
    
    print(f"\n  Retrying: {p1_orig} vs {p2_orig}")
    
    # Search for player 1
    print(f"\n  Searching for '{p1_orig}'...")
    p1_results = loader.find_player(p1_orig, tour)
    
    if not p1_results:
        # Try without tour filter
        p1_results = loader.find_player(p1_orig, None)
    
    if not p1_results:
        new_name = input(f"  Not found. Enter different name for Player 1 (or skip): ").strip()
        if not new_name:
            return
        p1_results = loader.find_player(new_name, None)
    
    if not p1_results:
        print(f"    Still not found. Skipping.")
        return
    
    print("  Found:")
    for i, p in enumerate(p1_results[:5]):
        t = "ATP" if p.player_id.startswith("ATP") else "WTA"
        print(f"    {i+1}. [{t}] {p.name}")
    
    sel = input("  Select (1-5) or skip: ").strip()
    if not sel.isdigit():
        return
    player1 = p1_results[int(sel) - 1]
    actual_tour = "ATP" if player1.player_id.startswith("ATP") else "WTA"
    
    # Search for player 2
    print(f"\n  Searching for '{p2_orig}'...")
    p2_results = loader.find_player(p2_orig, actual_tour)
    
    if not p2_results:
        new_name = input(f"  Not found. Enter different name for Player 2 (or skip): ").strip()
        if not new_name:
            return
        p2_results = loader.find_player(new_name, actual_tour)
    
    if not p2_results:
        print(f"    Still not found. Skipping.")
        return
    
    print("  Found:")
    for i, p in enumerate(p2_results[:5]):
        t = "ATP" if p.player_id.startswith("ATP") else "WTA"
        print(f"    {i+1}. [{t}] {p.name}")
    
    sel = input("  Select (1-5) or skip: ").strip()
    if not sel.isdigit():
        return
    player2 = p2_results[int(sel) - 1]
    
    # Create profile
    best_of = 5 if level == TournamentLevel.GRAND_SLAM and actual_tour == "ATP" else 3
    
    try:
        result = predictor.predict_match(
            player1.player_id, player2.player_id,
            surface, level, None, None, best_of
        )
        
        profile = {
            'p1_id': player1.player_id,
            'p2_id': player2.player_id,
            'p1_name': player1.name,
            'p2_name': player2.name,
            'tour': actual_tour,
            'surface': surface.value,
            'level': level.value,
            'best_of': best_of,
            'r1': None,
            'r2': None,
            'bankroll': bankroll,
            'pre_match_prob': result['p1_win_prob'],
            'p1_serve_pct': result['p1_serve_pct'],
            'p2_serve_pct': result['p2_serve_pct'],
            'h2h': result['h2h'],
            'kalshi_ticker': None,
            'bets': [],
            'created': datetime.now().isoformat(),
        }
        
        filename = f"match_{player1.name.split()[-1]}_{player2.name.split()[-1]}.json".lower()
        with open(filename, 'w') as f:
            json.dump(profile, f, indent=2)
        
        prob = result['p1_win_prob'] * 100
        print(f"\n    ✓ {player1.name} ({prob:.0f}%) vs {player2.name} ({100-prob:.0f}%)")
        print(f"      Saved: {filename}")
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")


def load_matches_file(predictor: TennisPredictor):
    """Load matches from a text file."""
    
    filepath = input("  File path [matches.txt]: ").strip() or "matches.txt"
    
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        print("\n  Create a file with matches, one per line:")
        print("    Sinner vs Alcaraz")
        print("    Djokovic vs Medvedev")
        print("    Swiatek vs Sabalenka")
        return
    
    matches = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ' vs ' in line.lower():
                parts = line.lower().split(' vs ')
                if len(parts) == 2:
                    matches.append((parts[0].strip().title(), parts[1].strip().title()))
    
    if not matches:
        print("  No valid matches in file.")
        return
    
    print(f"\n  Found {len(matches)} matches:")
    for p1, p2 in matches:
        print(f"    • {p1} vs {p2}")
    
    # Get common settings
    surface = Surface.from_string(input("\n  Surface (Hard/Clay/Grass) [Hard]: ").strip() or "Hard")
    level_str = input("  Level (G/M/other) [other]: ").strip().upper()
    level = TournamentLevel.GRAND_SLAM if level_str == 'G' else TournamentLevel.MASTERS if level_str == 'M' else TournamentLevel.OTHER
    tour = input("  Tour (ATP/WTA) [ATP]: ").strip().upper() or "ATP"
    bankroll = float(input("  Bankroll [$]: ").strip() or "1000")
    
    created = 0
    for p1_name, p2_name in matches:
        profile = create_profile_for_match(
            predictor, p1_name, p2_name,
            surface, level, tour, bankroll
        )
        if profile:
            created += 1
    
    print(f"\n  ✓ Created {created} profiles!")


def create_profile_for_match(predictor: TennisPredictor, p1_name: str, p2_name: str,
                             surface: Surface, level: TournamentLevel, tour: str,
                             bankroll: float, kalshi_ticker: str = None) -> dict:
    """Create a profile for a single match."""
    loader = predictor.loader
    
    # Find players - try with tour first, then without
    p1_results = loader.find_player(p1_name, tour)
    if not p1_results:
        p1_results = loader.find_player(p1_name, None)  # Try any tour
    
    p2_results = loader.find_player(p2_name, tour)
    if not p2_results:
        p2_results = loader.find_player(p2_name, None)
    
    if not p1_results:
        print(f"    ✗ Player not found: '{p1_name}' - try last name only or check spelling")
        return None
    if not p2_results:
        print(f"    ✗ Player not found: '{p2_name}' - try last name only or check spelling")
        return None
    
    player1 = p1_results[0]  # Take first match
    player2 = p2_results[0]
    
    # Make sure both players are from same tour
    p1_tour = "ATP" if player1.player_id.startswith("ATP") else "WTA"
    p2_tour = "ATP" if player2.player_id.startswith("ATP") else "WTA"
    
    if p1_tour != p2_tour:
        print(f"    ✗ Tour mismatch: {player1.name} ({p1_tour}) vs {player2.name} ({p2_tour})")
        return None
    
    actual_tour = p1_tour
    
    # Get best_of
    best_of = 5 if level == TournamentLevel.GRAND_SLAM and actual_tour == "ATP" else 3
    
    # Get prediction
    try:
        result = predictor.predict_match(
            player1.player_id, player2.player_id,
            surface, level, None, None, best_of
        )
    except Exception as e:
        print(f"    ✗ Prediction failed for {player1.name} vs {player2.name}: {e}")
        return None
    
    # Create profile
    profile = {
        'p1_id': player1.player_id,
        'p2_id': player2.player_id,
        'p1_name': player1.name,
        'p2_name': player2.name,
        'tour': actual_tour,
        'surface': surface.value,
        'level': level.value,
        'best_of': best_of,
        'r1': None,
        'r2': None,
        'bankroll': bankroll,
        'pre_match_prob': result['p1_win_prob'],
        'p1_serve_pct': result['p1_serve_pct'],
        'p2_serve_pct': result['p2_serve_pct'],
        'h2h': result['h2h'],
        'kalshi_ticker': kalshi_ticker,
        'bets': [],
        'created': datetime.now().isoformat(),
    }
    
    # Save
    filename = f"match_{player1.name.split()[-1]}_{player2.name.split()[-1]}.json".lower()
    with open(filename, 'w') as f:
        json.dump(profile, f, indent=2)
    
    prob = result['p1_win_prob'] * 100
    print(f"    ✓ {player1.name} ({prob:.0f}%) vs {player2.name} ({100-prob:.0f}%) → {filename}")
    
    return profile


def create_profile(predictor: TennisPredictor):
    """Create a new match profile."""
    loader = predictor.loader
    
    print("\n  CREATE MATCH PROFILE")
    print("  (Enter 'q' at any prompt to cancel)")
    print("-" * 40)
    
    tour = input("  Tour (ATP/WTA) [ATP]: ").strip().upper() or "ATP"
    if tour.lower() == 'q':
        return
    
    p1_name = input("  Player 1: ").strip()
    if p1_name.lower() == 'q' or not p1_name:
        return
    
    p1_results = loader.find_player(p1_name, tour)
    if not p1_results:
        print(f"  Not found: {p1_name}")
        return
    
    for i, p in enumerate(p1_results[:5]):
        print(f"    {i+1}. {p.name}")
    sel = input("  Select (or q to cancel): ").strip()
    if sel.lower() == 'q':
        return
    player1 = p1_results[int(sel or "1") - 1]
    
    p2_name = input("  Player 2: ").strip()
    if p2_name.lower() == 'q' or not p2_name:
        return
    
    p2_results = loader.find_player(p2_name, tour)
    if not p2_results:
        print(f"  Not found: {p2_name}")
        return
    
    for i, p in enumerate(p2_results[:5]):
        print(f"    {i+1}. {p.name}")
    sel = input("  Select (or q to cancel): ").strip()
    if sel.lower() == 'q':
        return
    player2 = p2_results[int(sel or "1") - 1]
    
    surface = Surface.from_string(input("  Surface (Hard/Clay/Grass) [Hard]: ").strip() or "Hard")
    level_str = input("  Level (G/M/other) [other]: ").strip().upper()
    level = TournamentLevel.GRAND_SLAM if level_str == 'G' else TournamentLevel.MASTERS if level_str == 'M' else TournamentLevel.OTHER
    
    best_of = 5 if level == TournamentLevel.GRAND_SLAM and tour == "ATP" else 3
    
    r1 = input(f"  {player1.name.split()[0]} ranking: ").strip()
    if r1.lower() == 'q':
        return
    r1 = int(r1) if r1 else None
    
    r2 = input(f"  {player2.name.split()[0]} ranking: ").strip()
    if r2.lower() == 'q':
        return
    r2 = int(r2) if r2 else None
    
    bankroll = input("  Bankroll [$]: ").strip() or "1000"
    if bankroll.lower() == 'q':
        return
    bankroll = float(bankroll)
    
    result = predictor.predict_match(player1.player_id, player2.player_id, surface, level, r1, r2, best_of)
    
    profile = {
        'p1_id': player1.player_id, 'p2_id': player2.player_id,
        'p1_name': player1.name, 'p2_name': player2.name,
        'tour': tour, 'surface': surface.value, 'level': level.value,
        'best_of': best_of, 'r1': r1, 'r2': r2, 'bankroll': bankroll,
        'pre_match_prob': result['p1_win_prob'],
        'p1_serve_pct': result['p1_serve_pct'],
        'p2_serve_pct': result['p2_serve_pct'],
        'h2h': result['h2h'], 'bets': []
    }
    
    filename = f"match_{player1.name.split()[-1]}_{player2.name.split()[-1]}.json".lower()
    with open(filename, 'w') as f:
        json.dump(profile, f, indent=2)
    
    print(f"\n  Saved: {filename}")
    print(f"\n  {player1.name}: {result['p1_win_prob']*100:.1f}%")
    print(f"  {player2.name}: {result['p2_win_prob']*100:.1f}%")
    print(f"  H2H: {result['h2h']}")
    
    if input("\n  Start tracking? (y/n): ").strip().lower() == 'y':
        live_tracker(predictor, profile, filename)


def load_profile(predictor: TennisPredictor):
    """Load existing profile."""
    profiles = glob.glob("match_*.json")
    if not profiles:
        print("  No profiles found.")
        return
    
    print("\n  SAVED PROFILES:")
    profile_data = []
    for i, pf in enumerate(profiles):
        with open(pf, 'r') as f:
            data = json.load(f)
            profile_data.append((pf, data))
            print(f"    {i+1}. {data['p1_name']} vs {data['p2_name']}")
    
    idx = int(input("  Select: ") or "1") - 1
    filename, profile = profile_data[idx]
    
    live_tracker(predictor, profile, filename)


def live_tracker(predictor: TennisPredictor, profile: dict, filename: str):
    """Live match tracking with edge analysis and trading."""
    
    p1_name = profile['p1_name']
    p2_name = profile['p2_name']
    pre_match = profile['pre_match_prob']
    bankroll = profile['bankroll']
    
    live_pred = LiveMatchPredictor(
        profile['p1_serve_pct'], profile['p2_serve_pct'], profile['best_of']
    )
    edge_analyzer = EdgeAnalyzer(live_pred)
    
    # Selected Kalshi market ticker
    selected_ticker = profile.get('kalshi_ticker')
    
    print(f"\n  LIVE: {p1_name} vs {p2_name}")
    print(f"  Pre-match: {p1_name.split()[0]} {pre_match*100:.0f}%")
    if selected_ticker:
        print(f"  Kalshi: {selected_ticker}")
    print("-" * 50)
    print("  Score format: 1-0 3-2 1 (sets games server)")
    print("  Commands:")
    print("    h=history, b=bankroll, k=kalshi, q=quit")
    print("    t=trade, o=orders, p=positions, $=balance")
    
    while True:
        cmd = input("\n> ").strip()
        
        if cmd.lower() == 'q':
            break
        elif cmd.lower() == 'h':
            if profile.get('bets'):
                print("\n  BET HISTORY:")
                for bet in profile['bets']:
                    print(f"    {bet['player']} @ {bet['score']} - ${bet['amount']:.2f}")
            continue
        elif cmd.lower().startswith('b '):
            bankroll = float(cmd[2:].replace('$', ''))
            profile['bankroll'] = bankroll
            with open(filename, 'w') as f:
                json.dump(profile, f, indent=2)
            print(f"  Bankroll: ${bankroll:,.0f}")
            continue
        elif cmd.lower() == '$':
            # Check Kalshi balance
            bal = predictor.kalshi.get_balance()
            print(f"\n  KALSHI BALANCE:")
            print(f"    Total:     ${bal['balance']:,.2f}")
            print(f"    Available: ${bal['available']:,.2f}")
            continue
        elif cmd.lower() == 'o':
            # Show orders
            orders = predictor.kalshi.get_orders()
            if orders:
                print("\n  OPEN ORDERS:")
                for o in orders:
                    print(f"    {o['ticker']} {o['action']} {o['side']} x{o['count']} @ {o['price']*100:.0f}¢")
            else:
                print("  No open orders")
            continue
        elif cmd.lower() == 'p':
            # Show positions
            positions = predictor.kalshi.get_positions()
            if positions:
                print("\n  POSITIONS:")
                for p in positions:
                    print(f"    {p['ticker']}: {p['position']} contracts, PnL: ${p['realized_pnl']:.2f}")
            else:
                print("  No positions")
            continue
        elif cmd.lower() == 'k':
            # Search/select Kalshi market
            markets = predictor.kalshi.search_tennis_markets(
                p1_name.split()[-1], p2_name.split()[-1]
            )
            if markets:
                print("\n  KALSHI MARKETS:")
                for i, m in enumerate(markets[:10]):
                    print(f"    {i+1}. {m['title']}")
                    print(f"       Yes: {m['yes_bid']*100:.0f}¢/{m['yes_ask']*100:.0f}¢ | Vol: {m['volume']}")
                
                sel = input("\n  Select market # (or skip): ").strip()
                if sel.isdigit() and 1 <= int(sel) <= len(markets):
                    selected_ticker = markets[int(sel)-1]['ticker']
                    profile['kalshi_ticker'] = selected_ticker
                    with open(filename, 'w') as f:
                        json.dump(profile, f, indent=2)
                    print(f"  Selected: {selected_ticker}")
            else:
                print("  No matching markets found")
            continue
        elif cmd.lower() == 't':
            # Trading menu
            if not selected_ticker:
                print("  No market selected. Use 'k' to select a market first.")
                continue
            
            trade_menu(predictor.kalshi, selected_ticker, bankroll)
            continue
        
        # Parse score
        try:
            parts = cmd.split()
            sets = parts[0].split('-')
            games = parts[1].split('-')
            server = int(parts[2]) if len(parts) > 2 else 1
            
            p1_sets, p2_sets = int(sets[0]), int(sets[1])
            p1_games, p2_games = int(games[0]), int(games[1])
            p1_serving = server == 1
        except:
            print("  Format: 1-0 3-2 1")
            continue
        
        # Calculate probability
        current_prob = live_pred.prob_win_match(p1_sets, p2_sets, p1_games, p2_games, p1_serving)
        
        score_str = f"{p1_sets}-{p2_sets}, {p1_games}-{p2_games}"
        change = (current_prob - pre_match) * 100
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        
        print(f"\n  {score_str} ({p1_name.split()[0] if p1_serving else p2_name.split()[0]} serving)")
        print(f"  {p1_name.split()[0]}: {current_prob*100:.1f}% ({arrow}{abs(change):.1f}%)")
        
        # Get market odds (from Kalshi if available, else manual)
        market_prob = None
        if selected_ticker:
            market = predictor.kalshi.get_market(selected_ticker)
            if market.get('yes_bid'):
                market_prob = market['yes_bid']
                print(f"  Kalshi ({selected_ticker}): Yes {market_prob*100:.0f}¢")
        
        if market_prob is None:
            market_input = input(f"  Market % for {p1_name.split()[0]} [skip]: ").strip()
            if not market_input:
                continue
            try:
                val = float(market_input)
                market_prob = val / 100 if val > 1 else val
            except:
                continue
        
        # Edge analysis
        analysis = edge_analyzer.analyze_edge_realization(
            current_prob, market_prob, p1_serving, bankroll
        )
        
        print(f"\n  " + "=" * 50)
        print(f"  EDGE ANALYSIS")
        print(f"  " + "-" * 50)
        print(f"  Model: {current_prob*100:.1f}% | Market: {market_prob*100:.1f}%")
        print(f"  Current Edge: {analysis['edge']*100:+.1f}%")
        
        # Show prediction
        pred = analysis['prediction']
        print(f"\n  After next game:")
        print(f"    If {p1_name.split()[0]} wins:  edge → {analysis['edge_if_win']*100:+.1f}%")
        print(f"    If {p1_name.split()[0]} loses: edge → {analysis['edge_if_lose']*100:+.1f}%")
        print(f"    Expected edge: {analysis['expected_edge_after']*100:+.1f}%")
        print(f"    Prob edge survives: {analysis['prob_edge_survives']*100:.0f}%")
        
        print(f"\n  Bet Quality: {analysis['bet_quality']}")
        if analysis['skip_reasons']:
            for reason in analysis['skip_reasons'][:2]:
                print(f"    ⚠ {reason}")
        
        print(f"  Urgency: {analysis['urgency']}")
        print(f"  {analysis['urgency_note']}")
        print(f"  " + "-" * 50)
        
        # Check if we should bet the other player instead
        if analysis['opposite_analysis']:
            opp = analysis['opposite_analysis']
            other_player = p2_name if current_prob > market_prob else p1_name
            print(f"\n  💡 TIP: {opp['note']}")
            print(f"     {other_player}: {opp['edge']*100:+.1f}% edge, expected {opp['expected_edge']*100:+.1f}%")
        
        if analysis['bet_amount'] > 0 and analysis['bet_quality'] not in ['SKIP']:
            bet_on_p1 = current_prob > market_prob
            bet_player = p1_name if bet_on_p1 else p2_name
            bet_side = 'yes' if bet_on_p1 else 'no'
            
            print(f"\n  ✓ {analysis['recommendation']}")
            print(f"    Bet on: {bet_player} ({bet_side.upper()})")
            print(f"    Kelly: {analysis['kelly_adjusted']*100:.1f}% of bankroll")
            
            # Calculate contracts
            price_cents = int(market_prob * 100) if bet_on_p1 else int((1-market_prob) * 100)
            contracts = int(analysis['bet_amount'] / (price_cents / 100)) if price_cents > 0 else 0
            
            if selected_ticker and predictor.kalshi.is_authenticated:
                print(f"\n    Kalshi: {contracts} contracts @ {price_cents}¢")
                execute = input("    Execute trade? (y/n): ").strip().lower()
                
                if execute == 'y':
                    if bet_side == 'yes':
                        result = predictor.kalshi.buy_yes(selected_ticker, contracts, price_cents)
                    else:
                        result = predictor.kalshi.buy_no(selected_ticker, contracts, price_cents)
                    
                    if result.get('success'):
                        print(f"    ✓ Order placed: {result['order_id']}")
                        profile['bets'].append({
                            'score': score_str,
                            'player': bet_player,
                            'edge': analysis['edge'] * 100,
                            'amount': analysis['bet_amount'],
                            'kalshi_order': result['order_id'],
                            'contracts': contracts,
                            'price': price_cents,
                        })
                        with open(filename, 'w') as f:
                            json.dump(profile, f, indent=2)
                    else:
                        print(f"    ✗ Order failed: {result.get('error')}")
            else:
                if input("  Record bet? (y/n): ").strip().lower() == 'y':
                    profile['bets'].append({
                        'score': score_str,
                        'player': bet_player,
                        'edge': analysis['edge'] * 100,
                        'amount': analysis['bet_amount']
                    })
                    with open(filename, 'w') as f:
                        json.dump(profile, f, indent=2)
                    print("  Recorded!")
        else:
            print(f"  ✗ {analysis['recommendation']}")
        
        print(f"  " + "=" * 45)


def trade_menu(kalshi: KalshiAPI, ticker: str, bankroll: float):
    """Interactive trading menu."""
    
    if not kalshi.is_authenticated:
        print("\n  Not authenticated. Run setup first.")
        setup_kalshi_credentials()
        return
    
    market = kalshi.get_market(ticker)
    
    print(f"\n  TRADE: {ticker}")
    print(f"  {market.get('title', 'Unknown')}")
    print(f"  Yes: {market['yes_bid']*100:.0f}¢ bid / {market['yes_ask']*100:.0f}¢ ask")
    print(f"  No:  {market['no_bid']*100:.0f}¢ bid / {market['no_ask']*100:.0f}¢ ask")
    
    print("\n  Actions:")
    print("    1. Buy YES")
    print("    2. Buy NO")
    print("    3. Sell YES")
    print("    4. Sell NO")
    print("    5. Cancel all orders")
    print("    6. Back")
    
    action = input("\n  Select: ").strip()
    
    if action == '6':
        return
    
    if action == '5':
        result = kalshi.cancel_all_orders()
        print(f"  Canceled {result['canceled']} orders")
        return
    
    if action not in ['1', '2', '3', '4']:
        return
    
    contracts = input("  Contracts: ").strip()
    if not contracts.isdigit():
        return
    contracts = int(contracts)
    
    price = input("  Price (cents): ").strip()
    if not price.isdigit():
        return
    price = int(price)
    
    if action == '1':
        result = kalshi.buy_yes(ticker, contracts, price)
    elif action == '2':
        result = kalshi.buy_no(ticker, contracts, price)
    elif action == '3':
        result = kalshi.sell_yes(ticker, contracts, price)
    elif action == '4':
        result = kalshi.sell_no(ticker, contracts, price)
    
    if result.get('success'):
        print(f"\n  ✓ Order placed!")
        print(f"    ID: {result['order_id']}")
        print(f"    {result['action']} {result['side']} x{result['count']} @ {result['price']*100:.0f}¢")
    else:
        print(f"\n  ✗ Failed: {result.get('error')}")


def quick_predict(predictor: TennisPredictor):
    """Quick prediction without saving."""
    loader = predictor.loader
    
    print("\n  QUICK PREDICTION")
    print("  (Enter 'q' at any prompt to cancel)")
    
    tour = input("\n  Tour (ATP/WTA) [ATP]: ").strip().upper() or "ATP"
    if tour.lower() == 'q':
        return
    
    p1_name = input("  Player 1: ").strip()
    if p1_name.lower() == 'q' or not p1_name:
        return
    
    p1 = loader.find_player(p1_name, tour)
    if not p1:
        print(f"  Not found: {p1_name}")
        return
    
    for i, p in enumerate(p1[:5]):
        print(f"    {i+1}. {p.name}")
    sel = input("  Select (or q to cancel): ").strip()
    if sel.lower() == 'q':
        return
    player1 = p1[int(sel or "1") - 1]
    
    p2_name = input("  Player 2: ").strip()
    if p2_name.lower() == 'q' or not p2_name:
        return
    
    p2 = loader.find_player(p2_name, tour)
    if not p2:
        print(f"  Not found: {p2_name}")
        return
    
    for i, p in enumerate(p2[:5]):
        print(f"    {i+1}. {p.name}")
    sel = input("  Select (or q to cancel): ").strip()
    if sel.lower() == 'q':
        return
    player2 = p2[int(sel or "1") - 1]
    
    surface = Surface.from_string(input("  Surface [Hard]: ").strip() or "Hard")
    
    r1 = input(f"  {player1.name.split()[0]} rank: ").strip()
    if r1.lower() == 'q':
        return
    r1 = int(r1) if r1 else None
    
    r2 = input(f"  {player2.name.split()[0]} rank: ").strip()
    if r2.lower() == 'q':
        return
    r2 = int(r2) if r2 else None
    
    result = predictor.predict_match(player1.player_id, player2.player_id, surface, 
                                     TournamentLevel.OTHER, r1, r2, 3)
    
    print(f"\n  {player1.name}: {result['p1_win_prob']*100:.1f}%")
    print(f"  {player2.name}: {result['p2_win_prob']*100:.1f}%")
    print(f"  H2H: {result['h2h']}")


def check_kalshi(predictor: TennisPredictor):
    """Check Kalshi for tennis markets."""
    print("\n  Searching Kalshi for tennis markets...")
    markets = predictor.kalshi.search_tennis_markets()
    
    if not markets:
        print("  No tennis markets found on Kalshi")
        return
    
    print(f"\n  Found {len(markets)} potential tennis markets:")
    for m in markets[:10]:
        print(f"\n  {m['title']}")
        print(f"    Yes: {m['yes_bid']*100:.0f}¢ bid / {m['yes_ask']*100:.0f}¢ ask")
        print(f"    Volume: {m['volume']}")


def live_monitor(predictor: TennisPredictor):
    """Live monitor - continuously scan matches for betting edges with odds tracking."""
    import time
    
    print("\n" + "=" * 70)
    print("  LIVE MONITOR - Scanning for betting edges")
    print("=" * 70)
    print("  Press Ctrl+C to stop\n")
    
    # Get settings
    bankroll_input = input("  Your bankroll [$]: ").strip() or "1000"
    if bankroll_input.lower() == 'q':
        return
    bankroll = float(bankroll_input)
    
    min_edge_input = input("  Minimum edge % to alert [5]: ").strip() or "5"
    if min_edge_input.lower() == 'q':
        return
    min_edge = float(min_edge_input) / 100
    
    refresh_input = input("  Refresh interval (seconds) [30]: ").strip() or "30"
    if refresh_input.lower() == 'q':
        return
    refresh_secs = int(refresh_input)
    
    print(f"\n  Monitoring with:")
    print(f"    Bankroll: ${bankroll:,.2f}")
    print(f"    Min edge: {min_edge*100:.1f}%")
    print(f"    Refresh: every {refresh_secs}s")
    print("\n  Starting monitor...\n")
    
    # Simple edge analysis function (no LiveMatchPredictor needed)
    def simple_edge_analysis(model_prob: float, market_prob: float) -> dict:
        """Simple edge analysis."""
        edge = model_prob - market_prob
        abs_edge = abs(edge)
        
        if abs_edge >= 0.15:
            quality = 'EXCELLENT'
            kelly = min(0.15, 0.6 * abs_edge)
            urgency = 'HIGH'
        elif abs_edge >= 0.10:
            quality = 'GOOD'
            kelly = min(0.12, 0.5 * abs_edge)
            urgency = 'MEDIUM'
        elif abs_edge >= 0.05:
            quality = 'MARGINAL'
            kelly = min(0.08, 0.25 * abs_edge)
            urgency = 'LOW'
        else:
            quality = 'RISKY'
            kelly = min(0.05, 0.1 * abs_edge)
            urgency = 'LOW'
        
        return {
            'bet_quality': quality,
            'kelly_fraction': kelly,
            'urgency': urgency,
            'recommendation': 'BET' if abs_edge >= 0.03 else 'SKIP'
        }
    
    scan_count = 0
    opportunities_found = []
    
    # Track previous odds to detect swings
    previous_odds = {}
    
    try:
        while True:
            scan_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n{'='*70}")
            print(f"  SCAN #{scan_count} - {current_time}")
            print(f"{'='*70}")
            
            # Fetch all tennis markets
            tennis_series = ['KXATPMATCH', 'KXWTAMATCH']
            all_markets = []
            
            for series in tennis_series:
                try:
                    url = f"{predictor.kalshi.base_url}/markets"
                    params = {'series_ticker': series, 'status': 'open', 'limit': 100}
                    response = predictor.kalshi.session.get(url, params=params, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        all_markets.extend(data.get('markets', []))
                except Exception as e:
                    print(f"  Error fetching {series}: {e}")
            
            if not all_markets:
                print("  No tennis markets found.")
                print(f"  Next scan in {refresh_secs}s...")
                time.sleep(refresh_secs)
                continue
            
            # Group by match
            matches = {}
            for m in all_markets:
                ticker = m.get('ticker', '')
                title = m.get('title', '')
                
                parts = ticker.rsplit('-', 1)
                if len(parts) == 2:
                    match_id = parts[0]
                    
                    if match_id not in matches:
                        matches[match_id] = {'players': [], 'tour': 'ATP' if 'ATP' in match_id else 'WTA'}
                    
                    # Extract player name
                    if 'Will ' in title and ' win ' in title:
                        player_name = title.split('Will ')[1].split(' win ')[0].strip()
                        yes_bid = m.get('yes_bid', 0)
                        matches[match_id]['players'].append({
                            'name': player_name,
                            'yes_bid': yes_bid,
                            'ticker': ticker,
                        })
            
            print(f"  Found {len(matches)} active matches\n")
            
            # Analyze each match
            edges_found = []
            odds_swings = []
            
            for match_id, match_data in matches.items():
                players = match_data['players']
                if len(players) < 2:
                    continue
                
                p1 = players[0]
                p2 = players[1]
                tour = match_data['tour']
                
                # Get market odds (convert cents to probability)
                market_p1 = p1['yes_bid'] / 100 if p1['yes_bid'] else 0.5
                market_p2 = p2['yes_bid'] / 100 if p2['yes_bid'] else 0.5
                
                # Skip if no real odds
                if market_p1 == 0 or market_p2 == 0:
                    continue
                
                # Check for odds swing from previous scan
                if match_id in previous_odds:
                    prev_p1 = previous_odds[match_id]
                    swing = abs(market_p1 - prev_p1)
                    
                    if swing >= 0.05:  # 5% swing
                        direction = "↑" if market_p1 > prev_p1 else "↓"
                        swing_info = {
                            'match': f"{p1['name']} vs {p2['name']}",
                            'player': p1['name'],
                            'prev': prev_p1,
                            'now': market_p1,
                            'swing': swing,
                            'direction': direction,
                        }
                        odds_swings.append(swing_info)
                
                # Update previous odds
                previous_odds[match_id] = market_p1
                
                # Find players in database
                p1_results = predictor.loader.find_player(p1['name'], tour)
                p2_results = predictor.loader.find_player(p2['name'], tour)
                
                if not p1_results or not p2_results:
                    if not p1_results:
                        p1_results = predictor.loader.find_player(p1['name'], None)
                    if not p2_results:
                        p2_results = predictor.loader.find_player(p2['name'], None)
                
                if not p1_results or not p2_results:
                    continue
                
                player1 = p1_results[0]
                player2 = p2_results[0]
                
                # Get model prediction (pre-match baseline)
                try:
                    result = predictor.predict_match(
                        player1.player_id, player2.player_id,
                        Surface.HARD, TournamentLevel.OTHER,
                        None, None, 3
                    )
                    model_p1 = result['p1_win_prob']
                except Exception:
                    continue
                
                # Calculate edge
                edge_p1 = model_p1 - market_p1
                edge_p2 = (1 - model_p1) - market_p2
                
                # Infer match status from odds vs pre-match model
                # If market differs significantly from model, match is likely in progress
                odds_diff = abs(market_p1 - model_p1)
                match_status = "PRE-MATCH" if odds_diff < 0.10 else "LIVE" if odds_diff < 0.25 else "LATE STAGE"
                
                # Check if either player has edge above threshold
                if abs(edge_p1) >= min_edge or abs(edge_p2) >= min_edge:
                    if edge_p1 > 0 and edge_p1 >= min_edge:
                        bet_player = p1['name']
                        bet_ticker = p1['ticker']
                        edge = edge_p1
                        model_prob = model_p1
                        market_prob = market_p1
                    elif edge_p2 > 0 and edge_p2 >= min_edge:
                        bet_player = p2['name']
                        bet_ticker = p2['ticker']
                        edge = edge_p2
                        model_prob = 1 - model_p1
                        market_prob = market_p2
                    else:
                        continue
                    
                    # Analyze edge quality
                    analysis = simple_edge_analysis(model_prob, market_prob)
                    
                    if analysis['recommendation'] != 'SKIP':
                        edges_found.append({
                            'p1_name': p1['name'],
                            'p2_name': p2['name'],
                            'bet_player': bet_player,
                            'bet_ticker': bet_ticker,
                            'model_prob': model_prob,
                            'market_prob': market_prob,
                            'edge': edge,
                            'quality': analysis['bet_quality'],
                            'kelly': analysis['kelly_fraction'],
                            'bet_amount': analysis['kelly_fraction'] * bankroll,
                            'urgency': analysis['urgency'],
                            'match_status': match_status,
                        })
            
            # Display odds swings (indicates live match action)
            if odds_swings:
                print("  " + "-" * 66)
                print("  📊 ODDS MOVEMENTS DETECTED (Live action!)")
                print("  " + "-" * 66)
                for swing in odds_swings:
                    print(f"  {swing['direction']} {swing['match']}")
                    print(f"     {swing['player']}: {swing['prev']*100:.0f}% → {swing['now']*100:.0f}% ({swing['swing']*100:+.0f}%)")
                print()
            
            # Display betting opportunities
            if edges_found:
                edges_found.sort(key=lambda x: x['edge'], reverse=True)
                
                print("  " + "=" * 66)
                print("  🎾 BETTING OPPORTUNITIES!")
                print("  " + "=" * 66)
                
                for opp in edges_found:
                    quality_emoji = {
                        'EXCELLENT': '🔥',
                        'GOOD': '✅',
                        'MARGINAL': '⚠️',
                        'RISKY': '❓',
                    }.get(opp['quality'], '❓')
                    
                    status_tag = f"[{opp['match_status']}]"
                    
                    print(f"\n  {quality_emoji} {opp['p1_name']} vs {opp['p2_name']} {status_tag}")
                    print(f"     BET ON: {opp['bet_player']}")
                    print(f"     Model: {opp['model_prob']*100:.1f}% | Market: {opp['market_prob']*100:.1f}% | Edge: {opp['edge']*100:+.1f}%")
                    print(f"     Quality: {opp['quality']} | Urgency: {opp['urgency']}")
                    print(f"     Suggested bet: ${opp['bet_amount']:.2f} ({opp['kelly']*100:.1f}% Kelly)")
                    print(f"     Ticker: {opp['bet_ticker']}")
                
                print(f"\n  " + "=" * 66)
                print("\a")  # Terminal bell
                
                opportunities_found.extend(edges_found)
            else:
                print("  No edges above threshold found.")
            
            # Show all matches summary
            print(f"\n  MATCH SUMMARY:")
            print(f"  " + "-" * 50)
            for match_id, match_data in list(matches.items())[:10]:
                players = match_data['players']
                if len(players) >= 2:
                    p1, p2 = players[0], players[1]
                    p1_pct = p1['yes_bid'] if p1['yes_bid'] else 0
                    p2_pct = p2['yes_bid'] if p2['yes_bid'] else 0
                    
                    # Show who's favored
                    if p1_pct > p2_pct:
                        print(f"  • {p1['name']} ({p1_pct}¢) vs {p2['name']} ({p2_pct}¢)")
                    else:
                        print(f"  • {p2['name']} ({p2_pct}¢) vs {p1['name']} ({p1_pct}¢)")
            
            if len(matches) > 10:
                print(f"  ... and {len(matches) - 10} more matches")
            
            print(f"\n  Total opportunities this session: {len(opportunities_found)}")
            print(f"  Next scan in {refresh_secs}s... (Ctrl+C to stop)")
            
            time.sleep(refresh_secs)
            
    except KeyboardInterrupt:
        print("\n\n  Monitor stopped.")
        
        if opportunities_found:
            print(f"\n  SESSION SUMMARY")
            print(f"  " + "-" * 40)
            print(f"  Total opportunities found: {len(opportunities_found)}")
            
            best = sorted(opportunities_found, key=lambda x: x['edge'], reverse=True)[:5]
            if best:
                print(f"\n  Top opportunities:")
                for opp in best:
                    print(f"    • {opp['bet_player']}: {opp['edge']*100:+.1f}% edge ({opp['quality']})")
        
        print("\n  Returning to menu...")


if __name__ == "__main__":
    main()
