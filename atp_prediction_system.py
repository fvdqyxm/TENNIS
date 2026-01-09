#!/usr/bin/env python3
"""
ATP Tennis Prediction System
=============================
Complete system for predicting ATP tennis matches using Jeff Sackmann's data.

Features:
- Loads and processes Sackmann ATP data
- Elo rating system with surface adjustments
- Head-to-head analysis
- Form and fatigue modeling
- Backtesting with calibration metrics
- Parameter optimization

Data Attribution:
    Tennis databases by Jeff Sackmann / Tennis Abstract
    Licensed under CC BY-NC-SA 4.0
    https://github.com/JeffSackmann/tennis_atp

Usage:
    python atp_prediction_system.py /path/to/tennis_atp backtest
    python atp_prediction_system.py /path/to/tennis_atp calibrate
    python atp_prediction_system.py /path/to/tennis_atp predict
"""

import math
import json
import statistics
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from atp_data_loader import (
    ATPDataLoader, ATPMatch, ATPPlayer, 
    Surface, TournamentLevel
)


# ============================================
# Model Parameters
# ============================================

@dataclass
class ModelParameters:
    """Tunable parameters for the prediction model."""
    
    # ===== ELO SETTINGS =====
    elo_k_factor: float = 32.0
    elo_k_factor_new_player: float = 48.0  # Higher K for players with <30 matches
    elo_k_factor_established: float = 24.0  # Lower K for players with >100 matches
    elo_initial: float = 1500.0
    elo_scale: float = 400.0
    
    # Surface Elo weight (blend between overall and surface-specific)
    surface_elo_weight: float = 0.3
    
    # ===== RANKING FACTORS =====
    ranking_weight: float = 0.15
    ranking_scale: float = 0.08
    ranking_points_weight: float = 0.05  # Use ranking points difference
    
    # ===== HEAD-TO-HEAD =====
    h2h_weight: float = 0.08
    h2h_min_matches: int = 3
    h2h_surface_weight: float = 0.05
    h2h_recent_weight: float = 0.06  # Last 2 years
    h2h_very_recent_weight: float = 0.04  # Last 6 months
    h2h_decay_years: float = 3.0  # How fast old H2H results decay
    
    # ===== RECENT FORM =====
    form_weight: float = 0.10
    form_lookback_days: int = 60
    form_surface_weight: float = 0.06  # Form on specific surface
    form_surface_lookback_days: int = 365
    form_weighted: bool = True  # Weight recent matches more
    
    # ===== FATIGUE =====
    fatigue_weight: float = 0.04
    fatigue_matches_7d_factor: float = 0.02
    fatigue_matches_14d_factor: float = 0.01
    fatigue_minutes_factor: float = 0.00005  # Per minute played in last 14 days
    fatigue_rest_bonus: float = 0.01  # Bonus for 5+ days rest
    
    # ===== TOURNAMENT CONTEXT =====
    grand_slam_factor: float = 0.02  # Favorites more likely to win
    masters_factor: float = 0.01
    home_advantage: float = 0.02  # Playing in home country
    
    # ===== SURFACE SPECIALISTS =====
    surface_specialist_weight: float = 0.04
    surface_specialist_min_matches: int = 20
    surface_win_rate_threshold: float = 0.65  # Win rate to be considered specialist
    
    # ===== SERVE STATISTICS =====
    serve_stats_weight: float = 0.05
    ace_rate_weight: float = 0.02
    first_serve_pct_weight: float = 0.02
    bp_save_weight: float = 0.03
    
    # ===== RETURN STATISTICS =====
    return_stats_weight: float = 0.04
    bp_convert_weight: float = 0.03
    
    # ===== PHYSICAL FACTORS =====
    height_weight: float = 0.01  # Taller players advantage on serve
    height_grass_bonus: float = 0.015  # Extra height advantage on grass
    age_peak_start: int = 24
    age_peak_end: int = 29
    age_factor: float = 0.005
    age_experience_bonus: float = 0.01  # Older players in big matches
    
    # ===== MATCH CONTEXT =====
    round_factor_weight: float = 0.02  # Later rounds favor higher ranked
    best_of_5_factor: float = 0.03  # Favorites do better in best of 5
    
    # ===== MOMENTUM =====
    tournament_momentum_weight: float = 0.03  # Winning streak in current tournament
    season_momentum_weight: float = 0.02  # Overall season trajectory
    
    # ===== HANDEDNESS =====
    lefty_advantage: float = 0.01  # Small advantage for lefties (less common)
    lefty_vs_lefty_neutral: bool = True  # Neutralize when both lefty
    
    # ===== CONSISTENCY FACTORS =====
    upset_history_weight: float = 0.02  # Players who often cause/suffer upsets
    tiebreak_skill_weight: float = 0.02
    deciding_set_weight: float = 0.02  # Performance in deciding sets
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelParameters':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelParameters':
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


# ============================================
# Elo Rating System
# ============================================

class EloRatingSystem:
    """
    Elo rating system with surface-specific ratings.
    """
    
    def __init__(self, k_factor: float = 32.0, 
                 initial: float = 1500.0,
                 scale: float = 400.0):
        self.k_factor = k_factor
        self.initial = initial
        self.scale = scale
        
        # Overall Elo
        self.ratings: Dict[str, float] = defaultdict(lambda: initial)
        
        # Surface-specific Elo
        self.surface_ratings: Dict[Surface, Dict[str, float]] = {
            s: defaultdict(lambda: initial) for s in Surface
        }
        
        # Match count for K-factor adjustment
        self.match_counts: Dict[str, int] = defaultdict(int)
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / self.scale))
    
    def get_k_factor(self, player_id: str) -> float:
        """Adjusted K-factor based on match count."""
        matches = self.match_counts.get(player_id, 0)
        if matches < 30:
            return self.k_factor * 1.5  # New players move faster
        elif matches < 100:
            return self.k_factor
        else:
            return self.k_factor * 0.8  # Established players more stable
    
    def update(self, winner_id: str, loser_id: str, surface: Surface):
        """Update ratings after a match."""
        # Get current ratings
        winner_rating = self.ratings[winner_id]
        loser_rating = self.ratings[loser_id]
        
        winner_surface = self.surface_ratings[surface][winner_id]
        loser_surface = self.surface_ratings[surface][loser_id]
        
        # Expected scores
        expected_winner = self.expected_score(winner_rating, loser_rating)
        expected_loser = 1 - expected_winner
        
        expected_winner_surf = self.expected_score(winner_surface, loser_surface)
        expected_loser_surf = 1 - expected_winner_surf
        
        # K-factors
        k_winner = self.get_k_factor(winner_id)
        k_loser = self.get_k_factor(loser_id)
        
        # Update overall ratings
        self.ratings[winner_id] += k_winner * (1 - expected_winner)
        self.ratings[loser_id] += k_loser * (0 - expected_loser)
        
        # Update surface ratings
        self.surface_ratings[surface][winner_id] += k_winner * (1 - expected_winner_surf)
        self.surface_ratings[surface][loser_id] += k_loser * (0 - expected_loser_surf)
        
        # Update match counts
        self.match_counts[winner_id] += 1
        self.match_counts[loser_id] += 1
    
    def get_rating(self, player_id: str, surface: Surface = None,
                  surface_weight: float = 0.3) -> float:
        """Get blended rating for player."""
        overall = self.ratings[player_id]
        
        if surface is None:
            return overall
        
        surface_rating = self.surface_ratings[surface][player_id]
        
        # Blend overall and surface-specific
        return overall * (1 - surface_weight) + surface_rating * surface_weight
    
    def predict(self, player1_id: str, player2_id: str,
               surface: Surface = None,
               surface_weight: float = 0.3) -> float:
        """Predict probability player1 wins."""
        r1 = self.get_rating(player1_id, surface, surface_weight)
        r2 = self.get_rating(player2_id, surface, surface_weight)
        return self.expected_score(r1, r2)


# ============================================
# Main Prediction Engine
# ============================================

class ATPPredictionEngine:
    """
    Main prediction engine combining multiple factors.
    """
    
    def __init__(self, loader: ATPDataLoader, 
                 params: ModelParameters = None):
        self.loader = loader
        self.params = params or ModelParameters()
        
        # Initialize Elo system
        self.elo = EloRatingSystem(
            k_factor=self.params.elo_k_factor,
            initial=self.params.elo_initial,
            scale=self.params.elo_scale
        )
        
        # Track if Elo is initialized
        self._elo_initialized = False
    
    def initialize_elo(self, up_to_date: datetime = None):
        """
        Initialize Elo ratings by processing historical matches.
        """
        print("Initializing Elo ratings...")
        
        matches = self.loader.matches
        if up_to_date:
            matches = [m for m in matches if m.tourney_date < up_to_date]
        
        for match in matches:
            self.elo.update(match.winner_id, match.loser_id, match.surface)
        
        self._elo_initialized = True
        print(f"  Processed {len(matches)} matches")
        print(f"  Rated {len(self.elo.ratings)} players")
    
    def predict_match(self, 
                     player1_id: str,
                     player2_id: str,
                     surface: Surface,
                     tourney_level: TournamentLevel,
                     match_date: datetime,
                     player1_rank: int = None,
                     player2_rank: int = None,
                     player1_rank_points: int = None,
                     player2_rank_points: int = None,
                     round_name: str = "",
                     best_of: int = 3,
                     tourney_name: str = "") -> Dict:
        """
        Predict match outcome with comprehensive analysis.
        
        Returns:
            Dict with 'probability', 'factors', 'confidence', etc.
        """
        result = {
            'player1_id': player1_id,
            'player2_id': player2_id,
            'probability': 0.5,
            'factors': {},
            'confidence': 'medium'
        }
        
        p = self.params  # Shorthand
        
        # Get player objects
        player1 = self.loader.players.get(str(player1_id))
        player2 = self.loader.players.get(str(player2_id))
        
        # ===== 1. ELO-BASED PROBABILITY (BASE) =====
        if self._elo_initialized:
            elo_prob = self.elo.predict(
                player1_id, player2_id, surface,
                p.surface_elo_weight
            )
            result['factors']['elo'] = elo_prob
        else:
            elo_prob = 0.5
        
        adjustments = 0.0
        
        # ===== 2. RANKING ADJUSTMENT =====
        rank_adj = 0.0
        if player1_rank and player2_rank:
            if player1_rank > 0 and player2_rank > 0:
                log_diff = math.log10(player2_rank) - math.log10(player1_rank)
                rank_adj = log_diff * p.ranking_scale * p.ranking_weight
                rank_adj = max(-0.15, min(0.15, rank_adj))
        
        # Ranking points adjustment
        if player1_rank_points and player2_rank_points:
            points_ratio = player1_rank_points / max(player2_rank_points, 1)
            points_adj = (math.log(points_ratio) / 5) * p.ranking_points_weight
            rank_adj += max(-0.05, min(0.05, points_adj))
        
        result['factors']['ranking'] = rank_adj
        adjustments += rank_adj
        
        # ===== 3. HEAD-TO-HEAD =====
        h2h_adj = 0.0
        h2h = self.loader.get_h2h(player1_id, player2_id, match_date)
        total_h2h = h2h['p1_wins'] + h2h['p2_wins']
        
        if total_h2h >= p.h2h_min_matches:
            # Overall H2H with time decay
            h2h_rate = h2h['p1_wins'] / total_h2h
            h2h_adj += (h2h_rate - 0.5) * p.h2h_weight
            
            # Surface-specific H2H
            surf_h2h = h2h['by_surface'].get(surface, {'p1': 0, 'p2': 0})
            surf_total = surf_h2h['p1'] + surf_h2h['p2']
            if surf_total >= 2:
                surf_rate = surf_h2h['p1'] / surf_total
                h2h_adj += (surf_rate - 0.5) * p.h2h_surface_weight
            
            # Recent H2H (last 2 years)
            recent_total = h2h['recent']['p1'] + h2h['recent']['p2']
            if recent_total >= 2:
                recent_rate = h2h['recent']['p1'] / recent_total
                h2h_adj += (recent_rate - 0.5) * p.h2h_recent_weight
        
        result['factors']['h2h'] = h2h_adj
        result['h2h_record'] = f"{h2h['p1_wins']}-{h2h['p2_wins']}"
        adjustments += h2h_adj
        
        # ===== 4. RECENT FORM =====
        form_adj = 0.0
        form1, matches1 = self.loader.get_player_form(
            player1_id, match_date, p.form_lookback_days
        )
        form2, matches2 = self.loader.get_player_form(
            player2_id, match_date, p.form_lookback_days
        )
        
        if matches1 >= 3 and matches2 >= 3:
            form_adj = (form1 - form2) * p.form_weight
        
        # Surface-specific form
        surf_form1, surf_m1 = self.loader.get_player_form(
            player1_id, match_date, p.form_surface_lookback_days
        )
        surf_form2, surf_m2 = self.loader.get_player_form(
            player2_id, match_date, p.form_surface_lookback_days
        )
        # Filter to surface matches only
        surf_record1 = self.loader.get_surface_record(player1_id, surface, match_date)
        surf_record2 = self.loader.get_surface_record(player2_id, surface, match_date)
        
        if surf_record1[0] + surf_record1[1] >= 10 and surf_record2[0] + surf_record2[1] >= 10:
            sr1 = surf_record1[0] / (surf_record1[0] + surf_record1[1])
            sr2 = surf_record2[0] / (surf_record2[0] + surf_record2[1])
            form_adj += (sr1 - sr2) * p.form_surface_weight
        
        result['factors']['form'] = form_adj
        adjustments += form_adj
        
        # ===== 5. MOMENTUM/TREND =====
        momentum_adj = 0.0
        trend1 = self.loader.get_recent_performance_trend(player1_id, match_date)
        trend2 = self.loader.get_recent_performance_trend(player2_id, match_date)
        
        if len(trend1) >= 2 and len(trend2) >= 2:
            # Check if improving or declining
            slope1 = trend1[-1] - trend1[0]
            slope2 = trend2[-1] - trend2[0]
            momentum_adj = (slope1 - slope2) * p.season_momentum_weight
        
        result['factors']['momentum'] = momentum_adj
        adjustments += momentum_adj
        
        # ===== 6. FATIGUE =====
        fatigue_adj = 0.0
        fatigue1 = self.loader.get_fatigue_metrics(player1_id, match_date)
        fatigue2 = self.loader.get_fatigue_metrics(player2_id, match_date)
        
        def calc_fatigue(f: Dict) -> float:
            score = 0.0
            score += f.get('matches_7d', 0) * p.fatigue_matches_7d_factor
            score += f.get('matches_14d', 0) * p.fatigue_matches_14d_factor
            score += f.get('minutes_14d', 0) * p.fatigue_minutes_factor
            if f.get('days_since_last') and f['days_since_last'] >= 5:
                score -= p.fatigue_rest_bonus
            return score
        
        f1_score = calc_fatigue(fatigue1)
        f2_score = calc_fatigue(fatigue2)
        fatigue_adj = (f2_score - f1_score) * p.fatigue_weight
        
        result['factors']['fatigue'] = fatigue_adj
        adjustments += fatigue_adj
        
        # ===== 7. SERVE STATISTICS =====
        serve_adj = 0.0
        stats1 = self.loader.get_player_stats_summary(player1_id, surface)
        stats2 = self.loader.get_player_stats_summary(player2_id, surface)
        
        if stats1.get('first_serve_total', 0) > 500 and stats2.get('first_serve_total', 0) > 500:
            # Ace rate comparison
            ace1 = stats1.get('ace_rate', 0)
            ace2 = stats2.get('ace_rate', 0)
            serve_adj += (ace1 - ace2) * p.ace_rate_weight * 10
            
            # First serve won percentage
            fsw1 = stats1.get('first_serve_won_pct', 0.65)
            fsw2 = stats2.get('first_serve_won_pct', 0.65)
            serve_adj += (fsw1 - fsw2) * p.first_serve_pct_weight
            
            # Break point saving
            bp1 = stats1.get('bp_save_pct', 0.6)
            bp2 = stats2.get('bp_save_pct', 0.6)
            serve_adj += (bp1 - bp2) * p.bp_save_weight
        
        result['factors']['serve'] = serve_adj
        adjustments += serve_adj
        
        # ===== 8. RETURN STATISTICS =====
        return_adj = 0.0
        if stats1.get('return_points_total', 0) > 500 and stats2.get('return_points_total', 0) > 500:
            ret1 = stats1.get('return_points_won_pct', 0.35)
            ret2 = stats2.get('return_points_won_pct', 0.35)
            return_adj += (ret1 - ret2) * p.return_stats_weight
            
            # Break point conversion
            bpc1 = stats1.get('bp_convert_pct', 0.4)
            bpc2 = stats2.get('bp_convert_pct', 0.4)
            return_adj += (bpc1 - bpc2) * p.bp_convert_weight
        
        result['factors']['return'] = return_adj
        adjustments += return_adj
        
        # ===== 9. SURFACE SPECIALIST =====
        specialist_adj = 0.0
        if surf_record1[0] + surf_record1[1] >= p.surface_specialist_min_matches:
            sr1 = surf_record1[0] / (surf_record1[0] + surf_record1[1])
            if sr1 >= p.surface_win_rate_threshold:
                specialist_adj += p.surface_specialist_weight
        
        if surf_record2[0] + surf_record2[1] >= p.surface_specialist_min_matches:
            sr2 = surf_record2[0] / (surf_record2[0] + surf_record2[1])
            if sr2 >= p.surface_win_rate_threshold:
                specialist_adj -= p.surface_specialist_weight
        
        result['factors']['specialist'] = specialist_adj
        adjustments += specialist_adj
        
        # ===== 10. TOURNAMENT LEVEL =====
        level_adj = 0.0
        if tourney_level == TournamentLevel.GRAND_SLAM:
            if elo_prob > 0.5:
                level_adj = p.grand_slam_factor
            else:
                level_adj = -p.grand_slam_factor
            
            # Best of 5 favors favorite
            if best_of == 5 and elo_prob > 0.5:
                level_adj += p.best_of_5_factor
        elif tourney_level == TournamentLevel.MASTERS_1000:
            if elo_prob > 0.5:
                level_adj = p.masters_factor
        
        result['factors']['level'] = level_adj
        adjustments += level_adj
        
        # ===== 11. PHYSICAL FACTORS =====
        physical_adj = 0.0
        
        # Height advantage (especially on grass for serve)
        if player1 and player2 and player1.height_cm and player2.height_cm:
            height_diff = (player1.height_cm - player2.height_cm) / 100  # Normalize
            physical_adj += height_diff * p.height_weight
            if surface == Surface.GRASS:
                physical_adj += height_diff * p.height_grass_bonus
        
        # Age factors
        if player1 and player2:
            age1 = player1.age_at(match_date) if player1.birth_date else None
            age2 = player2.age_at(match_date) if player2.birth_date else None
            
            if age1 and age2:
                def age_score(age):
                    if p.age_peak_start <= age <= p.age_peak_end:
                        return 0
                    elif age < p.age_peak_start:
                        return -(p.age_peak_start - age) * p.age_factor * 0.5
                    else:
                        return -(age - p.age_peak_end) * p.age_factor
                
                physical_adj += age_score(age1) - age_score(age2)
                
                # Experience bonus in big matches
                if tourney_level in [TournamentLevel.GRAND_SLAM, TournamentLevel.MASTERS_1000]:
                    if age1 > age2 and age1 <= 34:
                        physical_adj += p.age_experience_bonus
                    elif age2 > age1 and age2 <= 34:
                        physical_adj -= p.age_experience_bonus
        
        result['factors']['physical'] = physical_adj
        adjustments += physical_adj
        
        # ===== 12. HANDEDNESS =====
        hand_adj = 0.0
        if player1 and player2:
            p1_lefty = player1.hand == 'L'
            p2_lefty = player2.hand == 'L'
            
            if p1_lefty and not p2_lefty:
                hand_adj = p.lefty_advantage
            elif p2_lefty and not p1_lefty:
                hand_adj = -p.lefty_advantage
            # Lefty vs lefty - neutralize if enabled
        
        result['factors']['handedness'] = hand_adj
        adjustments += hand_adj
        
        # ===== 13. CLUTCH/MENTAL FACTORS =====
        clutch_adj = 0.0
        
        # Tiebreak skill
        tb1 = stats1.get('tiebreak_win_pct', 0.5)
        tb2 = stats2.get('tiebreak_win_pct', 0.5)
        if stats1.get('tiebreaks_played', 0) >= 10 and stats2.get('tiebreaks_played', 0) >= 10:
            clutch_adj += (tb1 - tb2) * p.tiebreak_skill_weight
        
        # Deciding set performance
        ds1 = stats1.get('deciding_set_win_pct', 0.5)
        ds2 = stats2.get('deciding_set_win_pct', 0.5)
        if stats1.get('deciding_sets_played', 0) >= 10 and stats2.get('deciding_sets_played', 0) >= 10:
            clutch_adj += (ds1 - ds2) * p.deciding_set_weight
        
        result['factors']['clutch'] = clutch_adj
        adjustments += clutch_adj
        
        # ===== 14. UPSET TENDENCY =====
        upset_adj = 0.0
        if stats1.get('matches', 0) >= 30 and stats2.get('matches', 0) >= 30:
            # Player 1 tendency to cause upsets
            upset_rate1 = stats1.get('upsets_caused', 0) / max(stats1['matches'], 1)
            upset_rate2 = stats2.get('upsets_caused', 0) / max(stats2['matches'], 1)
            
            # If player1 is underdog, boost if they cause upsets often
            if elo_prob < 0.5:
                upset_adj += upset_rate1 * p.upset_history_weight
            if elo_prob > 0.5:
                upset_adj -= upset_rate2 * p.upset_history_weight
        
        result['factors']['upset_tendency'] = upset_adj
        adjustments += upset_adj
        
        # ===== COMBINE ALL FACTORS =====
        prob = elo_prob + adjustments
        
        # Clamp to valid probability
        prob = max(0.02, min(0.98, prob))
        
        result['probability'] = prob
        result['adjustments'] = adjustments
        
        # Set confidence level
        if abs(prob - 0.5) > 0.25:
            result['confidence'] = 'high'
        elif abs(prob - 0.5) > 0.1:
            result['confidence'] = 'medium'
        else:
            result['confidence'] = 'low'
        
        return result
    
    def update_elo_with_result(self, winner_id: str, loser_id: str, 
                               surface: Surface):
        """Update Elo after a match result."""
        self.elo.update(winner_id, loser_id, surface)


# ============================================
# Backtesting System
# ============================================

@dataclass
class BacktestResult:
    """Results from backtesting."""
    total_matches: int = 0
    correct: int = 0
    brier_score: float = 0.0
    log_loss: float = 0.0
    
    by_confidence: Dict[str, Dict] = field(default_factory=dict)
    by_surface: Dict[str, Dict] = field(default_factory=dict)
    by_level: Dict[str, Dict] = field(default_factory=dict)
    by_ranking_diff: Dict[str, Dict] = field(default_factory=dict)
    
    predictions: List[Dict] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        return self.correct / self.total_matches if self.total_matches > 0 else 0.0
    
    def summary(self) -> str:
        lines = [
            "=" * 65,
            "BACKTEST RESULTS",
            "=" * 65,
            f"Total Matches:      {self.total_matches}",
            f"Correct:            {self.correct}",
            f"Accuracy:           {self.accuracy:.1%}",
            f"Brier Score:        {self.brier_score:.4f}",
            f"Log Loss:           {self.log_loss:.4f}",
            "",
            "By Confidence:",
        ]
        
        for bucket, data in sorted(self.by_confidence.items()):
            acc = data['correct'] / data['total'] if data['total'] > 0 else 0
            lines.append(f"  {bucket}: {data['correct']}/{data['total']} = {acc:.1%}")
        
        lines.append("\nBy Surface:")
        for surface, data in self.by_surface.items():
            acc = data['correct'] / data['total'] if data['total'] > 0 else 0
            lines.append(f"  {surface}: {data['correct']}/{data['total']} = {acc:.1%}")
        
        lines.append("\nBy Tournament Level:")
        for level, data in self.by_level.items():
            acc = data['correct'] / data['total'] if data['total'] > 0 else 0
            lines.append(f"  {level}: {data['correct']}/{data['total']} = {acc:.1%}")
        
        if self.by_ranking_diff:
            lines.append("\nBy Ranking Difference:")
            for diff, data in sorted(self.by_ranking_diff.items()):
                acc = data['correct'] / data['total'] if data['total'] > 0 else 0
                lines.append(f"  {diff}: {data['correct']}/{data['total']} = {acc:.1%}")
        
        return "\n".join(lines)


class Backtester:
    """Backtest the prediction model."""
    
    def __init__(self, engine: ATPPredictionEngine):
        self.engine = engine
    
    def get_data_date_range(self) -> Tuple[datetime, datetime]:
        """Get the actual date range of loaded matches."""
        if not self.engine.loader.matches:
            return None, None
        dates = [m.tourney_date for m in self.engine.loader.matches]
        return min(dates), max(dates)
    
    def run(self, 
            start_date: datetime = None,
            end_date: datetime = None,
            min_ranking: int = None,
            surfaces: List[Surface] = None,
            levels: List[TournamentLevel] = None,
            update_elo: bool = True,
            train_ratio: float = 0.7) -> BacktestResult:
        """
        Run backtest on historical matches.
        
        Args:
            start_date: Start of test period (auto-detected if None)
            end_date: End of test period (auto-detected if None)
            min_ranking: Only include matches where both players ranked better
            surfaces: Filter to specific surfaces
            levels: Filter to specific tournament levels
            update_elo: Whether to update Elo after each prediction
            train_ratio: If dates not specified, use this ratio for train/test split
        """
        result = BacktestResult()
        
        brier_scores = []
        log_losses = []
        
        # Auto-detect date range if not specified
        data_start, data_end = self.get_data_date_range()
        
        if data_start is None:
            print("ERROR: No matches loaded!")
            return result
        
        print(f"Data range: {data_start.date()} to {data_end.date()}")
        
        if start_date is None or end_date is None:
            # Auto-split: use train_ratio for training, rest for testing
            total_days = (data_end - data_start).days
            train_days = int(total_days * train_ratio)
            
            if start_date is None:
                start_date = data_start + timedelta(days=train_days)
            if end_date is None:
                end_date = data_end + timedelta(days=1)  # Include last day
        
        # Validate dates are within data range
        if start_date > data_end:
            print(f"WARNING: start_date ({start_date.date()}) is after data ends ({data_end.date()})")
            print(f"Adjusting to use last 30% of data for testing...")
            total_days = (data_end - data_start).days
            start_date = data_start + timedelta(days=int(total_days * 0.7))
            end_date = data_end + timedelta(days=1)
        
        print(f"Test period: {start_date.date()} to {end_date.date()}")
        
        # Initialize Elo up to start date
        self.engine.initialize_elo(start_date)
        
        # Get matches in test period
        matches = [m for m in self.engine.loader.matches 
                  if start_date <= m.tourney_date <= end_date]
        
        print(f"Backtesting {len(matches)} matches...")
        
        for match in matches:
            # Apply filters
            if surfaces and match.surface not in surfaces:
                continue
            if levels and match.tourney_level not in levels:
                continue
            if min_ranking:
                if not match.winner_rank or not match.loser_rank:
                    continue
                if match.winner_rank > min_ranking and match.loser_rank > min_ranking:
                    continue
            
            # Make prediction (always predict as if player1 = winner)
            pred = self.engine.predict_match(
                player1_id=match.winner_id,
                player2_id=match.loser_id,
                surface=match.surface,
                tourney_level=match.tourney_level,
                match_date=match.tourney_date,
                player1_rank=match.winner_rank,
                player2_rank=match.loser_rank,
                player1_rank_points=match.winner_rank_points,
                player2_rank_points=match.loser_rank_points,
                round_name=match.round_name,
                best_of=match.best_of,
                tourney_name=match.tourney_name
            )
            
            prob = pred['probability']
            
            # Winner always wins (by definition), so actual = 1
            actual = 1.0
            correct = prob > 0.5
            
            result.total_matches += 1
            if correct:
                result.correct += 1
            
            # Brier score: (pred - actual)^2
            brier = (prob - actual) ** 2
            brier_scores.append(brier)
            
            # Log loss
            eps = 1e-10
            ll = -(actual * math.log(prob + eps) + (1 - actual) * math.log(1 - prob + eps))
            log_losses.append(ll)
            
            # Categorize by confidence
            confidence = prob  # Since we always predict winner as p1
            if confidence >= 0.8:
                bucket = "80-100%"
            elif confidence >= 0.7:
                bucket = "70-80%"
            elif confidence >= 0.6:
                bucket = "60-70%"
            else:
                bucket = "50-60%"
            
            if bucket not in result.by_confidence:
                result.by_confidence[bucket] = {'correct': 0, 'total': 0}
            result.by_confidence[bucket]['total'] += 1
            if correct:
                result.by_confidence[bucket]['correct'] += 1
            
            # By surface
            surf = match.surface.value
            if surf not in result.by_surface:
                result.by_surface[surf] = {'correct': 0, 'total': 0}
            result.by_surface[surf]['total'] += 1
            if correct:
                result.by_surface[surf]['correct'] += 1
            
            # By level
            lev = match.tourney_level.name
            if lev not in result.by_level:
                result.by_level[lev] = {'correct': 0, 'total': 0}
            result.by_level[lev]['total'] += 1
            if correct:
                result.by_level[lev]['correct'] += 1
            
            # By ranking difference
            if match.winner_rank and match.loser_rank:
                diff = abs(match.winner_rank - match.loser_rank)
                if diff <= 5:
                    diff_bucket = "1-5"
                elif diff <= 20:
                    diff_bucket = "6-20"
                elif diff <= 50:
                    diff_bucket = "21-50"
                else:
                    diff_bucket = "50+"
                
                if diff_bucket not in result.by_ranking_diff:
                    result.by_ranking_diff[diff_bucket] = {'correct': 0, 'total': 0}
                result.by_ranking_diff[diff_bucket]['total'] += 1
                if correct:
                    result.by_ranking_diff[diff_bucket]['correct'] += 1
            
            # Store prediction
            result.predictions.append({
                'date': match.tourney_date.isoformat(),
                'winner': match.winner_name,
                'loser': match.loser_name,
                'prob': prob,
                'correct': correct,
                'surface': surf,
                'level': lev
            })
            
            # Update Elo
            if update_elo:
                self.engine.update_elo_with_result(
                    match.winner_id, match.loser_id, match.surface
                )
        
        # Calculate aggregate metrics
        if brier_scores:
            result.brier_score = statistics.mean(brier_scores)
        if log_losses:
            result.log_loss = statistics.mean(log_losses)
        
        return result


# ============================================
# Parameter Calibration
# ============================================

class Calibrator:
    """Calibrate model parameters."""
    
    def __init__(self, loader: ATPDataLoader):
        self.loader = loader
    
    def get_data_date_range(self) -> Tuple[datetime, datetime]:
        """Get the actual date range of loaded matches."""
        if not self.loader.matches:
            return None, None
        dates = [m.tourney_date for m in self.loader.matches]
        return min(dates), max(dates)
    
    def calibrate(self,
                 train_start: datetime = None,
                 train_end: datetime = None,
                 param_grid: Dict[str, List] = None,
                 metric: str = 'brier',
                 train_ratio: float = 0.7,
                 preset: str = 'standard') -> Tuple[ModelParameters, BacktestResult]:
        """
        Grid search for optimal parameters.
        
        Args:
            preset: 'quick', 'standard', 'full', or 'custom'
        """
        if param_grid is None:
            if preset == 'quick':
                param_grid = {
                    'elo_k_factor': [24, 32, 40],
                    'h2h_weight': [0.05, 0.08, 0.12],
                }
            elif preset == 'standard':
                param_grid = {
                    'elo_k_factor': [24, 32, 40],
                    'surface_elo_weight': [0.2, 0.3, 0.4],
                    'h2h_weight': [0.05, 0.08, 0.12],
                    'form_weight': [0.08, 0.12, 0.15],
                    'serve_stats_weight': [0.03, 0.05, 0.07],
                }
            elif preset == 'full':
                param_grid = {
                    'elo_k_factor': [24, 32, 40],
                    'surface_elo_weight': [0.25, 0.35],
                    'ranking_weight': [0.10, 0.15, 0.20],
                    'h2h_weight': [0.06, 0.10],
                    'h2h_surface_weight': [0.03, 0.06],
                    'form_weight': [0.08, 0.12],
                    'form_surface_weight': [0.04, 0.08],
                    'fatigue_weight': [0.03, 0.05],
                    'serve_stats_weight': [0.04, 0.06],
                    'return_stats_weight': [0.03, 0.05],
                    'surface_specialist_weight': [0.02, 0.05],
                    'grand_slam_factor': [0.01, 0.03],
                    'best_of_5_factor': [0.02, 0.04],
                }
            else:
                param_grid = {
                    'elo_k_factor': [24, 32, 40],
                    'h2h_weight': [0.05, 0.08, 0.12],
                    'form_weight': [0.08, 0.12, 0.15],
                }
        
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        total = 1
        for v in param_values:
            total *= len(v)
        
        print(f"Testing {total} parameter combinations...")
        
        best_params = ModelParameters()
        best_score = float('inf') if metric != 'accuracy' else 0.0
        best_result = None
        
        for i, combo in enumerate(product(*param_values)):
            # Create params
            params = ModelParameters()
            for name, value in zip(param_names, combo):
                setattr(params, name, value)
            
            # Run backtest
            engine = ATPPredictionEngine(self.loader, params)
            backtester = Backtester(engine)
            result = backtester.run(train_start, train_end, min_ranking=200)
            
            # Get score
            if metric == 'brier':
                score = result.brier_score
                is_better = score < best_score
            elif metric == 'accuracy':
                score = result.accuracy
                is_better = score > best_score
            else:
                score = result.log_loss
                is_better = score < best_score
            
            if is_better:
                best_score = score
                best_params = params
                best_result = result
            
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{total} - Best {metric}: {best_score:.4f}")
        
        print(f"\nBest {metric}: {best_score:.4f}")
        
        return best_params, best_result


# ============================================
# Interactive Prediction
# ============================================

def interactive_predict(engine: ATPPredictionEngine):
    """Interactive match prediction."""
    print("\n" + "=" * 60)
    print("  MATCH PREDICTION")
    print("=" * 60)
    
    # Search for player 1
    p1_search = input("\nPlayer 1 name: ").strip()
    p1_results = engine.loader.find_player_by_name(p1_search)
    
    if not p1_results:
        print(f"No players found matching '{p1_search}'")
        return
    
    print("Found players:")
    for i, p in enumerate(p1_results[:5]):
        print(f"  {i + 1}. {p.name} ({p.country_code}) - ID: {p.player_id}")
    
    p1_idx = int(input("Select player (1-5): ") or "1") - 1
    player1 = p1_results[p1_idx]
    
    # Search for player 2
    p2_search = input("\nPlayer 2 name: ").strip()
    p2_results = engine.loader.find_player_by_name(p2_search)
    
    if not p2_results:
        print(f"No players found matching '{p2_search}'")
        return
    
    print("Found players:")
    for i, p in enumerate(p2_results[:5]):
        print(f"  {i + 1}. {p.name} ({p.country_code}) - ID: {p.player_id}")
    
    p2_idx = int(input("Select player (1-5): ") or "1") - 1
    player2 = p2_results[p2_idx]
    
    # Surface
    surface_input = input("\nSurface (Hard/Clay/Grass) [Hard]: ").strip() or "Hard"
    surface = Surface.from_string(surface_input)
    
    # Level
    level_input = input("Level (G=Grand Slam, M=Masters, A=ATP) [A]: ").strip().upper() or "A"
    level = TournamentLevel.from_string(level_input)
    
    # Get rankings
    p1_rank = engine.loader.get_player_ranking(player1.player_id, datetime.now())
    p2_rank = engine.loader.get_player_ranking(player2.player_id, datetime.now())
    
    # Predict
    pred = engine.predict_match(
        player1.player_id,
        player2.player_id,
        surface,
        level,
        datetime.now(),
        p1_rank,
        p2_rank
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("  PREDICTION")
    print("=" * 60)
    print(f"\n  {player1.name} vs {player2.name}")
    print(f"  Surface: {surface.value} | Level: {level.name}")
    
    if p1_rank and p2_rank:
        print(f"  Rankings: #{p1_rank} vs #{p2_rank}")
    
    print(f"\n  H2H: {pred.get('h2h_record', 'N/A')}")
    
    print(f"\n  WIN PROBABILITY:")
    print(f"    {player1.name}: {pred['probability']:.1%}")
    print(f"    {player2.name}: {1 - pred['probability']:.1%}")
    
    print(f"\n  Factors:")
    for factor, value in pred['factors'].items():
        if isinstance(value, float):
            print(f"    {factor}: {value:+.3f}")
    
    print("=" * 60)


# ============================================
# Main Entry Point
# ============================================

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("ATP Tennis Prediction System")
        print("=" * 50)
        print("\nUsage:")
        print("  python atp_prediction_system.py <data_dir> [command]")
        print("\nCommands:")
        print("  backtest  - Run backtesting on historical data")
        print("  calibrate - Calibrate model parameters")
        print("  predict   - Interactive match prediction")
        print("  stats     - Show player statistics")
        print("\nExample:")
        print("  python atp_prediction_system.py ./tennis_atp backtest")
        print("\nFirst, clone the data repository:")
        print("  git clone https://github.com/JeffSackmann/tennis_atp.git")
        return
    
    data_dir = sys.argv[1]
    command = sys.argv[2] if len(sys.argv) > 2 else "predict"
    
    # Load data
    print(f"\nLoading ATP data from: {data_dir}")
    print("=" * 60)
    
    loader = ATPDataLoader(data_dir)
    loader.load_matches(years=range(2010, 2026))
    loader.load_players()
    
    if command == "backtest":
        print("\n" + "=" * 60)
        print("  BACKTESTING")
        print("=" * 60)
        
        params = ModelParameters()
        engine = ATPPredictionEngine(loader, params)
        backtester = Backtester(engine)
        
        # Auto-detect dates - test on last 30% of data
        result = backtester.run(min_ranking=200, train_ratio=0.7)
        print("\n" + result.summary())
        
    elif command == "calibrate":
        print("\n" + "=" * 60)
        print("  CALIBRATING")
        print("=" * 60)
        
        calibrator = Calibrator(loader)
        
        # Auto-detect dates from data
        data_start, data_end = calibrator.get_data_date_range()
        if data_start:
            print(f"Data range: {data_start.date()} to {data_end.date()}")
            # Use last 2 years of data for calibration
            total_days = (data_end - data_start).days
            train_start = data_start + timedelta(days=int(total_days * 0.7))
            train_end = data_end + timedelta(days=1)
            print(f"Calibrating on: {train_start.date()} to {train_end.date()}")
        else:
            print("ERROR: No matches loaded!")
            return
        
        best_params, best_result = calibrator.calibrate(train_start, train_end)
        
        print("\nBest Parameters:")
        for k, v in best_params.to_dict().items():
            print(f"  {k}: {v}")
        
        print("\n" + best_result.summary())
        
        # Save
        save = input("\nSave parameters? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"params_{datetime.now().strftime('%Y%m%d')}.json"
            best_params.save(filename)
            print(f"Saved to {filename}")
        
    elif command == "predict":
        params = ModelParameters()
        engine = ATPPredictionEngine(loader, params)
        engine.initialize_elo()
        
        while True:
            interactive_predict(engine)
            again = input("\nAnother prediction? (y/n): ").strip().lower()
            if again != 'y':
                break
        
    elif command == "stats":
        player_name = input("Player name: ").strip()
        results = loader.find_player_by_name(player_name)
        
        if results:
            player = results[0]
            print(f"\n{player.name} ({player.country_code})")
            print(f"ID: {player.player_id}")
            print(f"Hand: {player.hand}")
            if player.height_cm:
                print(f"Height: {player.height_cm} cm")
            
            stats = loader.get_player_stats_summary(player.player_id)
            print(f"\nCareer Stats:")
            print(f"  Matches: {stats['matches']}")
            print(f"  Wins: {stats['wins']}")
            if 'win_pct' in stats:
                print(f"  Win %: {stats['win_pct']:.1%}")
            if 'first_serve_pct' in stats:
                print(f"  1st Serve %: {stats['first_serve_pct']:.1%}")
            if 'bp_save_pct' in stats:
                print(f"  BP Save %: {stats['bp_save_pct']:.1%}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
