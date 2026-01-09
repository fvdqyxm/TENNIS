#!/usr/bin/env python3
"""
Advanced Position Management for Kalshi Tennis Trading
=======================================================
Sophisticated position management including:
- Dynamic stop-loss based on match state
- Partial profit taking
- Hedging strategies
- Live odds monitoring
- Position sizing optimization

This module extends the basic position management with smarter
cash-out logic based on tennis match dynamics.
"""

import math
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class MatchState(Enum):
    """Tennis match state for position management."""
    PRE_MATCH = "pre_match"
    FIRST_SET = "first_set"
    SECOND_SET = "second_set"
    THIRD_SET = "third_set"  # Deciding set for BO3
    FOURTH_SET = "fourth_set"  # BO5 only
    FIFTH_SET = "fifth_set"  # Deciding set for BO5
    COMPLETED = "completed"
    UNKNOWN = "unknown"


class CashOutReason(Enum):
    """Reasons for cashing out a position."""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    PARTIAL_PROFIT = "partial_profit"
    HEDGE = "hedge"
    EDGE_DISAPPEARED = "edge_disappeared"
    MOMENTUM_SHIFT = "momentum_shift"
    TIME_DECAY = "time_decay"
    MANUAL = "manual"
    MATCH_COMPLETED = "match_completed"


@dataclass
class OddsSnapshot:
    """Point-in-time odds snapshot."""
    timestamp: datetime
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume: int = 0
    
    @property
    def mid_price(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2
    
    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid


@dataclass
class PositionAnalysis:
    """Analysis of a position's current state."""
    current_pnl: float
    current_pnl_pct: float
    current_edge: float
    implied_prob: float
    odds_momentum: float  # Positive = moving our way
    time_in_position: timedelta
    match_state: MatchState
    recommendation: str
    reason: CashOutReason = None
    partial_size: float = 0  # For partial exits


class LiveOddsTracker:
    """Track odds movement for a specific market."""
    
    def __init__(self, ticker: str, window_size: int = 20):
        self.ticker = ticker
        self.history: deque = deque(maxlen=window_size)
        self.significant_moves: List[OddsSnapshot] = []
    
    def add_snapshot(self, snapshot: OddsSnapshot):
        """Add new odds snapshot."""
        self.history.append(snapshot)
        
        # Track significant moves (>3% change)
        if len(self.history) >= 2:
            prev = self.history[-2]
            change = abs(snapshot.mid_price - prev.mid_price)
            if change >= 3:  # 3 cents = 3%
                self.significant_moves.append(snapshot)
    
    @property
    def current_odds(self) -> Optional[OddsSnapshot]:
        return self.history[-1] if self.history else None
    
    @property
    def momentum(self) -> float:
        """
        Calculate odds momentum over recent history.
        Returns: positive if odds increasing, negative if decreasing.
        """
        if len(self.history) < 3:
            return 0
        
        # Weight recent changes more heavily
        recent = list(self.history)[-5:]
        if len(recent) < 2:
            return 0
        
        changes = []
        for i in range(1, len(recent)):
            changes.append(recent[i].mid_price - recent[i-1].mid_price)
        
        # Weighted average with more weight on recent
        weights = [1, 2, 3, 4, 5][:len(changes)]
        weighted_sum = sum(c * w for c, w in zip(changes, weights))
        return weighted_sum / sum(weights)
    
    @property
    def volatility(self) -> float:
        """Calculate recent odds volatility."""
        if len(self.history) < 5:
            return 0
        
        recent = [s.mid_price for s in list(self.history)[-10:]]
        if len(recent) < 2:
            return 0
        
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        return math.sqrt(variance)
    
    def get_trend(self, minutes: int = 5) -> str:
        """Get odds trend over specified minutes."""
        if not self.history:
            return "UNKNOWN"
        
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent = [s for s in self.history if s.timestamp > cutoff]
        
        if len(recent) < 2:
            return "STABLE"
        
        change = recent[-1].mid_price - recent[0].mid_price
        
        if change > 5:
            return "STRONG_UP"
        elif change > 2:
            return "UP"
        elif change < -5:
            return "STRONG_DOWN"
        elif change < -2:
            return "DOWN"
        return "STABLE"


class DynamicStopLoss:
    """
    Dynamic stop-loss calculator that adjusts based on:
    - Current P&L
    - Match state
    - Odds momentum
    - Position age
    """
    
    def __init__(self, base_stop: float = 0.50):
        self.base_stop = base_stop
    
    def calculate_stop(self, position, odds_tracker: LiveOddsTracker, 
                       match_state: MatchState) -> Tuple[float, str]:
        """
        Calculate dynamic stop-loss level.
        
        Returns:
            (stop_level, reason)
        """
        stop = self.base_stop
        reasons = []
        
        # Adjust for match state
        state_adjustments = {
            MatchState.PRE_MATCH: 0,
            MatchState.FIRST_SET: -0.05,  # Tighter stop early
            MatchState.SECOND_SET: -0.10,
            MatchState.THIRD_SET: -0.15,  # Even tighter in deciding set
            MatchState.FIFTH_SET: -0.15,
        }
        
        if match_state in state_adjustments:
            adj = state_adjustments[match_state]
            stop += adj
            if adj != 0:
                reasons.append(f"match_state:{match_state.value}")
        
        # Adjust for position profitability (protect profits)
        pnl_pct = position.unrealized_pnl_pct
        if pnl_pct > 0.20:  # Up 20%+
            stop = min(stop, 0.10)  # Don't let winner become big loser
            reasons.append("protect_profit")
        elif pnl_pct > 0.10:  # Up 10%+
            stop = min(stop, 0.25)
            reasons.append("protect_gain")
        
        # Adjust for momentum (if odds moving against us, tighter stop)
        momentum = odds_tracker.momentum
        if (position.side == 'yes' and momentum < -2) or \
           (position.side == 'no' and momentum > 2):
            stop = min(stop, stop * 0.7)  # 30% tighter
            reasons.append("adverse_momentum")
        
        # Adjust for volatility (higher vol = tighter stop)
        volatility = odds_tracker.volatility
        if volatility > 5:
            stop = min(stop, stop * 0.8)
            reasons.append("high_volatility")
        
        # Minimum stop of 10%
        stop = max(stop, 0.10)
        
        return stop, "+".join(reasons) if reasons else "base"


class TakeProfit:
    """
    Take profit calculator with partial exit support.
    """
    
    def __init__(self, base_target: float = 0.30):
        self.base_target = base_target
        self.partial_levels = [
            (0.20, 0.25),  # At 20% profit, take 25% off
            (0.40, 0.25),  # At 40% profit, take another 25%
            (0.60, 0.25),  # At 60% profit, take another 25%
        ]
    
    def check_take_profit(self, position, odds_tracker: LiveOddsTracker,
                          match_state: MatchState) -> Optional[Tuple[float, float, str]]:
        """
        Check if should take profit.
        
        Returns:
            (exit_fraction, price, reason) or None
        """
        pnl_pct = position.unrealized_pnl_pct
        
        # Full exit at base target
        if pnl_pct >= self.base_target:
            return (1.0, position.current_price, f"full_profit_{pnl_pct:.0%}")
        
        # Check partial exits
        for level_pct, exit_fraction in self.partial_levels:
            if pnl_pct >= level_pct:
                # Check if we've already taken this level
                # (Would need position tracking for this)
                return (exit_fraction, position.current_price, f"partial_{level_pct:.0%}")
        
        # Special case: match state indicates high confidence
        if match_state in [MatchState.THIRD_SET, MatchState.FIFTH_SET]:
            # In deciding set, consider taking profit earlier
            if pnl_pct >= 0.15:
                return (0.50, position.current_price, "deciding_set_profit")
        
        return None


class EdgeMonitor:
    """
    Monitor whether our edge still exists.
    """
    
    def __init__(self, min_edge: float = 0.03):
        self.min_edge = min_edge
    
    def check_edge(self, position, model_prob: float, 
                   current_market_prob: float) -> Tuple[bool, float, str]:
        """
        Check if edge still exists.
        
        Returns:
            (edge_exists, current_edge, reason)
        """
        # Calculate current edge
        if position.side == 'yes':
            current_edge = model_prob - current_market_prob
        else:
            current_edge = (1 - model_prob) - (1 - current_market_prob)
        
        # Compare to entry edge
        edge_decay = position.edge_at_entry - current_edge
        
        if current_edge < self.min_edge:
            return (False, current_edge, f"edge_below_min:{current_edge:.1%}")
        
        if edge_decay > 0.10:  # Edge shrunk by 10%+
            return (False, current_edge, f"edge_decayed:{edge_decay:.1%}")
        
        return (True, current_edge, "edge_maintained")


class AdvancedPositionManager:
    """
    Advanced position manager with dynamic cash-out strategies.
    """
    
    def __init__(self, client, config, session, predictor=None):
        self.client = client
        self.config = config
        self.session = session
        self.predictor = predictor
        
        # Components
        self.stop_loss = DynamicStopLoss(config.stop_loss_pct)
        self.take_profit = TakeProfit(config.take_profit_pct)
        self.edge_monitor = EdgeMonitor(config.min_edge)
        
        # Odds trackers per position
        self.odds_trackers: Dict[str, LiveOddsTracker] = {}
    
    def update_odds(self, ticker: str, market_data: dict):
        """Update odds tracker for a ticker."""
        if ticker not in self.odds_trackers:
            self.odds_trackers[ticker] = LiveOddsTracker(ticker)
        
        snapshot = OddsSnapshot(
            timestamp=datetime.now(),
            yes_bid=market_data.get('yes_bid', 0),
            yes_ask=market_data.get('yes_ask', 0),
            no_bid=market_data.get('no_bid', 0),
            no_ask=market_data.get('no_ask', 0),
            volume=market_data.get('volume', 0),
        )
        self.odds_trackers[ticker].add_snapshot(snapshot)
    
    def infer_match_state(self, position, current_price: float) -> MatchState:
        """
        Infer match state from price movement.
        
        Big price swings indicate set/match changes.
        """
        entry_price = position.entry_price
        time_in_position = datetime.now() - position.entry_time
        
        # Get odds history
        tracker = self.odds_trackers.get(position.ticker)
        if not tracker or len(tracker.significant_moves) == 0:
            # No big moves - likely pre-match or early first set
            if time_in_position < timedelta(hours=1):
                return MatchState.PRE_MATCH
            return MatchState.FIRST_SET
        
        # Count significant price moves (set changes typically cause big swings)
        set_changes = len(tracker.significant_moves)
        
        if set_changes == 0:
            return MatchState.FIRST_SET
        elif set_changes == 1:
            return MatchState.SECOND_SET
        elif set_changes >= 2:
            return MatchState.THIRD_SET
        
        return MatchState.UNKNOWN
    
    def analyze_position(self, position) -> PositionAnalysis:
        """Comprehensive position analysis."""
        # Get current market data
        market = self.client.get_market(position.ticker)
        if not market:
            return None
        
        # Update position price
        if position.side == 'yes':
            position.current_price = market.get('yes_bid', position.current_price)
        else:
            position.current_price = market.get('no_bid', position.current_price)
        
        # Update odds tracker
        self.update_odds(position.ticker, market)
        
        tracker = self.odds_trackers.get(position.ticker)
        match_state = self.infer_match_state(position, position.current_price)
        
        # Calculate metrics
        pnl_pct = position.unrealized_pnl_pct
        time_in = datetime.now() - position.entry_time
        
        # Get current implied probability
        implied_prob = position.current_price / 100
        
        # Check edge if we have predictor
        current_edge = position.edge_at_entry
        edge_exists = True
        
        if self.predictor:
            model_prob = self._get_model_prob(position)
            if model_prob:
                edge_exists, current_edge, _ = self.edge_monitor.check_edge(
                    position, model_prob, implied_prob
                )
        
        # Check stop loss
        stop_level, stop_reason = self.stop_loss.calculate_stop(
            position, tracker, match_state
        )
        
        # Determine recommendation
        recommendation = "HOLD"
        reason = None
        partial_size = 0
        
        # Priority 1: Stop loss
        if pnl_pct <= -stop_level:
            recommendation = "EXIT"
            reason = CashOutReason.STOP_LOSS
        
        # Priority 2: Take profit
        elif tp_result := self.take_profit.check_take_profit(position, tracker, match_state):
            exit_frac, price, tp_reason = tp_result
            recommendation = "PARTIAL_EXIT" if exit_frac < 1.0 else "EXIT"
            reason = CashOutReason.TAKE_PROFIT if exit_frac >= 1.0 else CashOutReason.PARTIAL_PROFIT
            partial_size = exit_frac
        
        # Priority 3: Edge disappeared
        elif not edge_exists:
            recommendation = "EXIT"
            reason = CashOutReason.EDGE_DISAPPEARED
        
        # Priority 4: Momentum shift
        elif tracker and tracker.momentum:
            momentum = tracker.momentum
            # If significant momentum against us
            if (position.side == 'yes' and momentum < -3) or \
               (position.side == 'no' and momentum > 3):
                if pnl_pct > 0:
                    recommendation = "PARTIAL_EXIT"
                    reason = CashOutReason.MOMENTUM_SHIFT
                    partial_size = 0.50
        
        return PositionAnalysis(
            current_pnl=position.unrealized_pnl,
            current_pnl_pct=pnl_pct,
            current_edge=current_edge,
            implied_prob=implied_prob,
            odds_momentum=tracker.momentum if tracker else 0,
            time_in_position=time_in,
            match_state=match_state,
            recommendation=recommendation,
            reason=reason,
            partial_size=partial_size,
        )
    
    def _get_model_prob(self, position) -> Optional[float]:
        """Get current model probability for position."""
        if not self.predictor:
            return None
        
        try:
            # This would need to be implemented based on your predictor
            # For now, return the original model probability
            return position.model_prob
        except:
            return None
    
    def manage_positions(self) -> List[dict]:
        """
        Check all positions and return recommended actions.
        """
        actions = []
        
        for position in self.session.open_positions:
            analysis = self.analyze_position(position)
            
            if not analysis:
                continue
            
            if analysis.recommendation in ["EXIT", "PARTIAL_EXIT"]:
                actions.append({
                    'position': position,
                    'action': analysis.recommendation.lower(),
                    'reason': analysis.reason,
                    'partial_size': analysis.partial_size,
                    'analysis': analysis,
                })
                
                logger.info(f"📊 Position Analysis: {position.player_name}")
                logger.info(f"   P&L: {analysis.current_pnl_pct:+.1%} | "
                           f"Edge: {analysis.current_edge:+.1%} | "
                           f"State: {analysis.match_state.value}")
                logger.info(f"   → {analysis.recommendation}: {analysis.reason.value if analysis.reason else 'N/A'}")
        
        return actions
    
    def execute_action(self, action: dict) -> bool:
        """Execute a recommended action."""
        position = action['position']
        action_type = action['action']
        reason = action['reason']
        partial_size = action.get('partial_size', 1.0)
        
        logger.info(f"Executing {action_type}: {position.ticker}")
        logger.info(f"  Reason: {reason.value if reason else 'N/A'}")
        
        if self.config.paper_trading:
            # Simulate execution
            contracts_to_sell = int(position.contracts * partial_size)
            
            if action_type == 'partial_exit':
                # Update position
                position.contracts -= contracts_to_sell
                realized = (position.current_price - position.entry_price) * contracts_to_sell / 100
                self.session.current_bankroll += realized + (contracts_to_sell * position.entry_price / 100)
                logger.info(f"  [PAPER] Partial exit: {contracts_to_sell} contracts, "
                           f"realized ${realized:.2f}")
            else:
                # Full exit
                position.exit_price = position.current_price
                position.exit_time = datetime.now()
                position.pnl = position.unrealized_pnl
                position.status = self._get_status_from_reason(reason)
                
                self.session.positions.remove(position)
                self.session.closed_positions.append(position)
                self.session.current_bankroll += position.cost_basis + position.pnl
                
                logger.info(f"  [PAPER] Full exit @ {position.exit_price}¢, "
                           f"P&L: ${position.pnl:.2f}")
            
            return True
        
        # Real trading
        contracts_to_sell = int(position.contracts * partial_size)
        result = self.client.market_sell(position.ticker, position.side, contracts_to_sell)
        
        if result.get('success'):
            if action_type == 'partial_exit':
                position.contracts -= contracts_to_sell
            else:
                position.exit_price = result.get('price', position.current_price) * 100
                position.exit_time = datetime.now()
                position.pnl = (position.exit_price - position.entry_price) * position.contracts / 100
                position.status = self._get_status_from_reason(reason)
                
                self.session.positions.remove(position)
                self.session.closed_positions.append(position)
            
            logger.info(f"  Executed @ {result.get('price', 0) * 100}¢")
            return True
        
        logger.error(f"  Failed: {result.get('error')}")
        return False
    
    def _get_status_from_reason(self, reason: CashOutReason):
        """Convert cash-out reason to position status."""
        from kalshi_tennis_bot import PositionStatus
        
        mapping = {
            CashOutReason.STOP_LOSS: PositionStatus.STOPPED,
            CashOutReason.TAKE_PROFIT: PositionStatus.PROFIT_TAKEN,
            CashOutReason.PARTIAL_PROFIT: PositionStatus.OPEN,
            CashOutReason.EDGE_DISAPPEARED: PositionStatus.CLOSED,
            CashOutReason.MOMENTUM_SHIFT: PositionStatus.CLOSED,
            CashOutReason.MANUAL: PositionStatus.CLOSED,
        }
        return mapping.get(reason, PositionStatus.CLOSED)


# ============================================
# Hedging Strategies
# ============================================

class HedgeManager:
    """
    Manage hedging positions to lock in profits or limit losses.
    """
    
    def __init__(self, client, config):
        self.client = client
        self.config = config
    
    def calculate_hedge(self, position, target_pnl: float = 0) -> Optional[dict]:
        """
        Calculate hedge trade to lock in target P&L.
        
        For a YES position, hedge by buying NO at current price.
        For a NO position, hedge by buying YES at current price.
        
        Args:
            position: Current position
            target_pnl: Target P&L to lock in (0 = break even)
        
        Returns:
            Hedge trade details or None
        """
        market = self.client.get_market(position.ticker)
        if not market:
            return None
        
        if position.side == 'yes':
            # Currently long YES, hedge with NO
            hedge_side = 'no'
            hedge_price = market.get('no_ask', 0)
        else:
            # Currently long NO, hedge with YES
            hedge_side = 'yes'
            hedge_price = market.get('yes_ask', 0)
        
        if hedge_price <= 0:
            return None
        
        # Calculate hedge size
        # With hedge: guaranteed payout = hedge_price * contracts
        # Cost = entry_price + hedge_price
        # P&L = payout - cost = 100 - entry_price - hedge_price
        
        potential_pnl = (100 - position.entry_price - hedge_price) * position.contracts / 100
        
        if potential_pnl < target_pnl:
            # Can't achieve target P&L with hedge
            return None
        
        return {
            'side': hedge_side,
            'contracts': position.contracts,
            'price_cents': hedge_price,
            'locked_pnl': potential_pnl,
            'cost': hedge_price * position.contracts / 100,
        }
    
    def should_hedge(self, position, analysis: PositionAnalysis) -> bool:
        """
        Determine if hedging is recommended.
        """
        # Hedge if:
        # 1. Match is in late stages
        # 2. We have significant unrealized profit
        # 3. Odds are volatile
        
        if analysis.match_state in [MatchState.THIRD_SET, MatchState.FIFTH_SET]:
            if analysis.current_pnl_pct >= 0.15:  # 15%+ profit
                return True
        
        # Hedge if very profitable and momentum shifting
        if analysis.current_pnl_pct >= 0.25 and analysis.odds_momentum < -2:
            return True
        
        return False


# ============================================
# Module Test
# ============================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Advanced Position Management Module")
    print("=" * 50)
    print("This module provides:")
    print("  - Dynamic stop-loss based on match state")
    print("  - Partial profit taking")
    print("  - Edge monitoring")
    print("  - Hedging strategies")
    print("  - Live odds tracking")
    print()
    print("Import and use with TennisTradingBot:")
    print("  from position_manager import AdvancedPositionManager")
