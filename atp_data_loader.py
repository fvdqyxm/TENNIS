#!/usr/bin/env python3
"""
ATP Tennis Data Loader for Jeff Sackmann's tennis_atp Repository
=================================================================
https://github.com/JeffSackmann/tennis_atp

This module provides optimized loading and parsing of the Sackmann ATP dataset,
which is the gold standard for historical tennis data.

Data Attribution:
    Tennis databases, files, and algorithms by Jeff Sackmann / Tennis Abstract
    Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
    https://github.com/JeffSackmann/tennis_atp

Usage:
    # Download data files first:
    # git clone https://github.com/JeffSackmann/tennis_atp.git
    
    loader = ATPDataLoader("path/to/tennis_atp")
    loader.load_matches(years=range(2015, 2025))
    loader.load_players()
    loader.load_rankings()
"""

import os
import csv
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


# ============================================
# Enums and Constants
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
        s = s.strip().title()
        for surface in cls:
            if surface.value == s:
                return surface
        return cls.HARD


class TournamentLevel(Enum):
    GRAND_SLAM = "G"
    MASTERS_1000 = "M"
    ATP_500 = "A"
    ATP_250 = "A"  # Same code as 500 in Sackmann data
    DAVIS_CUP = "D"
    TOUR_FINALS = "F"
    OLYMPICS = "O"
    CHALLENGER = "C"
    FUTURES = "S"  # Satellite
    OTHER = ""
    
    @classmethod
    def from_string(cls, s: str) -> 'TournamentLevel':
        if not s:
            return cls.OTHER
        s = s.strip().upper()
        for level in cls:
            if level.value == s:
                return level
        return cls.OTHER
    
    @property
    def best_of(self) -> int:
        """Default best-of for this tournament level."""
        if self == TournamentLevel.GRAND_SLAM:
            return 5
        return 3


# ============================================
# Data Classes
# ============================================

@dataclass
class ATPPlayer:
    """Player from ATP database."""
    player_id: str
    first_name: str = ""
    last_name: str = ""
    hand: str = ""  # R, L, U (unknown)
    birth_date: Optional[datetime] = None
    country_code: str = ""
    height_cm: Optional[int] = None
    
    @property
    def name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()
    
    @property
    def age_at(self) -> callable:
        """Returns function to calculate age at a given date."""
        def calc_age(date: datetime) -> Optional[float]:
            if not self.birth_date:
                return None
            delta = date - self.birth_date
            return delta.days / 365.25
        return calc_age
    
    def __str__(self):
        return f"{self.name} ({self.country_code})"


@dataclass
class ATPMatch:
    """Single match from ATP database."""
    # Required fields (no defaults) - must come first
    tourney_id: str
    tourney_name: str
    tourney_date: datetime
    surface: Surface
    draw_size: int
    tourney_level: TournamentLevel
    match_num: int
    round_name: str
    best_of: int
    score: str
    winner_id: str
    winner_name: str
    loser_id: str
    loser_name: str
    
    # Optional fields (with defaults)
    minutes: Optional[int] = None
    winner_hand: str = ""
    winner_ht: Optional[int] = None
    winner_ioc: str = ""
    winner_age: Optional[float] = None
    winner_rank: Optional[int] = None
    winner_rank_points: Optional[int] = None
    loser_hand: str = ""
    loser_ht: Optional[int] = None
    loser_ioc: str = ""
    loser_age: Optional[float] = None
    loser_rank: Optional[int] = None
    loser_rank_points: Optional[int] = None
    
    # Winner stats
    w_ace: Optional[int] = None
    w_df: Optional[int] = None
    w_svpt: Optional[int] = None
    w_1stIn: Optional[int] = None
    w_1stWon: Optional[int] = None
    w_2ndWon: Optional[int] = None
    w_SvGms: Optional[int] = None
    w_bpSaved: Optional[int] = None
    w_bpFaced: Optional[int] = None
    
    # Loser stats  
    l_ace: Optional[int] = None
    l_df: Optional[int] = None
    l_svpt: Optional[int] = None
    l_1stIn: Optional[int] = None
    l_1stWon: Optional[int] = None
    l_2ndWon: Optional[int] = None
    l_SvGms: Optional[int] = None
    l_bpSaved: Optional[int] = None
    l_bpFaced: Optional[int] = None
    
    @property
    def has_stats(self) -> bool:
        """Check if match has detailed statistics."""
        return self.w_svpt is not None and self.w_svpt > 0
    
    def get_winner_serve_pct(self) -> Optional[float]:
        """Winner's 1st serve percentage."""
        if self.w_svpt and self.w_1stIn:
            return self.w_1stIn / self.w_svpt
        return None
    
    def get_loser_serve_pct(self) -> Optional[float]:
        """Loser's 1st serve percentage."""
        if self.l_svpt and self.l_1stIn:
            return self.l_1stIn / self.l_svpt
        return None


@dataclass
class ATPRanking:
    """Weekly ranking entry."""
    ranking_date: datetime
    rank: int
    player_id: str
    points: Optional[int] = None


# ============================================
# Main Data Loader
# ============================================

class ATPDataLoader:
    """
    Loads and manages Jeff Sackmann's ATP tennis data.
    
    Usage:
        loader = ATPDataLoader("/path/to/tennis_atp")
        loader.load_matches(years=range(2015, 2025))
        loader.load_players()
        
        # Access data
        matches = loader.matches
        players = loader.players
        
        # Get head-to-head
        h2h = loader.get_h2h("104745", "104925")  # Djokovic vs Nadal
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize loader with path to tennis_atp repository.
        
        Args:
            data_dir: Path to cloned tennis_atp repository
        """
        self.data_dir = data_dir
        self.matches: List[ATPMatch] = []
        self.players: Dict[str, ATPPlayer] = {}
        self.rankings: Dict[str, List[ATPRanking]] = defaultdict(list)
        
        # Indexes for fast lookup
        self._player_matches: Dict[str, List[ATPMatch]] = defaultdict(list)
        self._h2h_cache: Dict[Tuple[str, str], Dict] = {}
        self._surface_matches: Dict[Surface, List[ATPMatch]] = defaultdict(list)
        
    def load_matches(self, 
                    years: range = None,
                    include_qual_chall: bool = False,
                    include_futures: bool = False) -> int:
        """
        Load match data for specified years.
        
        Args:
            years: Range of years to load (default: 2000-2024)
            include_qual_chall: Include qualifying and challenger matches
            include_futures: Include futures/ITF matches
            
        Returns:
            Number of matches loaded
        """
        if years is None:
            years = range(2000, 2025)
        
        files_to_load = []
        
        for year in years:
            # Main tour matches
            main_file = os.path.join(self.data_dir, f"atp_matches_{year}.csv")
            if os.path.exists(main_file):
                files_to_load.append(main_file)
            
            # Qualifying and challenger
            if include_qual_chall:
                qual_file = os.path.join(self.data_dir, f"atp_matches_qual_chall_{year}.csv")
                if os.path.exists(qual_file):
                    files_to_load.append(qual_file)
            
            # Futures
            if include_futures:
                futures_file = os.path.join(self.data_dir, f"atp_matches_futures_{year}.csv")
                if os.path.exists(futures_file):
                    files_to_load.append(futures_file)
        
        count = 0
        for filepath in files_to_load:
            count += self._load_match_file(filepath)
        
        # Sort by date
        self.matches.sort(key=lambda m: m.tourney_date)
        
        # Build indexes
        self._build_indexes()
        
        print(f"Loaded {count} matches from {len(files_to_load)} files")
        return count
    
    def _load_match_file(self, filepath: str) -> int:
        """Load single match CSV file."""
        count = 0
        
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    match = self._parse_match_row(row)
                    if match:
                        self.matches.append(match)
                        count += 1
                except Exception as e:
                    continue
        
        return count
    
    def _parse_match_row(self, row: Dict) -> Optional[ATPMatch]:
        """Parse single CSV row into ATPMatch."""
        # Parse date (format: YYYYMMDD)
        date_str = row.get('tourney_date', '')
        if not date_str:
            return None
        
        try:
            tourney_date = datetime.strptime(str(int(float(date_str))), "%Y%m%d")
        except:
            return None
        
        # Get required fields
        winner_id = row.get('winner_id', '')
        loser_id = row.get('loser_id', '')
        if not winner_id or not loser_id:
            return None
        
        # Parse optional integers
        def safe_int(val) -> Optional[int]:
            if not val or val == '':
                return None
            try:
                return int(float(val))
            except:
                return None
        
        def safe_float(val) -> Optional[float]:
            if not val or val == '':
                return None
            try:
                return float(val)
            except:
                return None
        
        # Determine best_of from tourney_level if not provided
        level = TournamentLevel.from_string(row.get('tourney_level', ''))
        best_of = safe_int(row.get('best_of')) or level.best_of
        
        return ATPMatch(
            # Tournament
            tourney_id=row.get('tourney_id', ''),
            tourney_name=row.get('tourney_name', ''),
            tourney_date=tourney_date,
            surface=Surface.from_string(row.get('surface', '')),
            draw_size=safe_int(row.get('draw_size')) or 32,
            tourney_level=level,
            
            # Match
            match_num=safe_int(row.get('match_num')) or 0,
            round_name=row.get('round', ''),
            best_of=best_of,
            score=row.get('score', ''),
            minutes=safe_int(row.get('minutes')),
            
            # Winner
            winner_id=str(winner_id),
            winner_name=row.get('winner_name', ''),
            winner_hand=row.get('winner_hand', ''),
            winner_ht=safe_int(row.get('winner_ht')),
            winner_ioc=row.get('winner_ioc', ''),
            winner_age=safe_float(row.get('winner_age')),
            winner_rank=safe_int(row.get('winner_rank')),
            winner_rank_points=safe_int(row.get('winner_rank_points')),
            
            # Loser
            loser_id=str(loser_id),
            loser_name=row.get('loser_name', ''),
            loser_hand=row.get('loser_hand', ''),
            loser_ht=safe_int(row.get('loser_ht')),
            loser_ioc=row.get('loser_ioc', ''),
            loser_age=safe_float(row.get('loser_age')),
            loser_rank=safe_int(row.get('loser_rank')),
            loser_rank_points=safe_int(row.get('loser_rank_points')),
            
            # Winner stats
            w_ace=safe_int(row.get('w_ace')),
            w_df=safe_int(row.get('w_df')),
            w_svpt=safe_int(row.get('w_svpt')),
            w_1stIn=safe_int(row.get('w_1stIn')),
            w_1stWon=safe_int(row.get('w_1stWon')),
            w_2ndWon=safe_int(row.get('w_2ndWon')),
            w_SvGms=safe_int(row.get('w_SvGms')),
            w_bpSaved=safe_int(row.get('w_bpSaved')),
            w_bpFaced=safe_int(row.get('w_bpFaced')),
            
            # Loser stats
            l_ace=safe_int(row.get('l_ace')),
            l_df=safe_int(row.get('l_df')),
            l_svpt=safe_int(row.get('l_svpt')),
            l_1stIn=safe_int(row.get('l_1stIn')),
            l_1stWon=safe_int(row.get('l_1stWon')),
            l_2ndWon=safe_int(row.get('l_2ndWon')),
            l_SvGms=safe_int(row.get('l_SvGms')),
            l_bpSaved=safe_int(row.get('l_bpSaved')),
            l_bpFaced=safe_int(row.get('l_bpFaced')),
        )
    
    def load_players(self) -> int:
        """
        Load player biographical data from atp_players.csv.
        
        Returns:
            Number of players loaded
        """
        filepath = os.path.join(self.data_dir, "atp_players.csv")
        if not os.path.exists(filepath):
            print(f"Warning: Players file not found at {filepath}")
            return 0
        
        count = 0
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    player_id = row.get('player_id', '')
                    if not player_id:
                        continue
                    
                    # Parse birth date
                    birth_date = None
                    bd_str = row.get('dob', '') or row.get('birth_date', '')
                    if bd_str and len(str(bd_str)) >= 8:
                        try:
                            birth_date = datetime.strptime(str(int(float(bd_str))), "%Y%m%d")
                        except:
                            pass
                    
                    # Parse height
                    height = None
                    ht_str = row.get('height', '')
                    if ht_str:
                        try:
                            height = int(float(ht_str))
                        except:
                            pass
                    
                    player = ATPPlayer(
                        player_id=str(player_id),
                        first_name=row.get('name_first', '') or row.get('first_name', ''),
                        last_name=row.get('name_last', '') or row.get('last_name', ''),
                        hand=row.get('hand', ''),
                        birth_date=birth_date,
                        country_code=row.get('ioc', '') or row.get('country_code', ''),
                        height_cm=height
                    )
                    
                    self.players[str(player_id)] = player
                    count += 1
                    
                except Exception as e:
                    continue
        
        print(f"Loaded {count} players")
        return count
    
    def load_rankings(self, years: range = None) -> int:
        """
        Load historical rankings.
        
        Args:
            years: Range of years (default: all available)
            
        Returns:
            Number of ranking entries loaded
        """
        # Find all ranking files
        ranking_files = []
        for filename in os.listdir(self.data_dir):
            if filename.startswith('atp_rankings_') and filename.endswith('.csv'):
                ranking_files.append(os.path.join(self.data_dir, filename))
        
        count = 0
        for filepath in sorted(ranking_files):
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        date_str = row.get('ranking_date', '')
                        if not date_str:
                            continue
                        
                        ranking_date = datetime.strptime(str(int(float(date_str))), "%Y%m%d")
                        
                        if years and ranking_date.year not in years:
                            continue
                        
                        player_id = str(row.get('player', '') or row.get('player_id', ''))
                        rank = int(row.get('rank', '') or row.get('ranking', ''))
                        
                        points = None
                        pts_str = row.get('points', '') or row.get('ranking_points', '')
                        if pts_str:
                            try:
                                points = int(float(pts_str))
                            except:
                                pass
                        
                        ranking = ATPRanking(
                            ranking_date=ranking_date,
                            rank=rank,
                            player_id=player_id,
                            points=points
                        )
                        
                        self.rankings[player_id].append(ranking)
                        count += 1
                        
                    except:
                        continue
        
        # Sort rankings by date for each player
        for player_id in self.rankings:
            self.rankings[player_id].sort(key=lambda r: r.ranking_date)
        
        print(f"Loaded {count} ranking entries")
        return count
    
    def _build_indexes(self):
        """Build lookup indexes after loading matches."""
        self._player_matches.clear()
        self._surface_matches.clear()
        self._h2h_cache.clear()
        
        for match in self.matches:
            self._player_matches[match.winner_id].append(match)
            self._player_matches[match.loser_id].append(match)
            self._surface_matches[match.surface].append(match)
    
    def get_player_matches(self, player_id: str, 
                          before_date: datetime = None) -> List[ATPMatch]:
        """Get all matches for a player."""
        matches = self._player_matches.get(str(player_id), [])
        if before_date:
            matches = [m for m in matches if m.tourney_date < before_date]
        return matches
    
    def get_h2h(self, player1_id: str, player2_id: str,
               before_date: datetime = None) -> Dict:
        """
        Get head-to-head record between two players.
        
        Returns:
            Dict with keys: p1_wins, p2_wins, matches, by_surface
        """
        p1, p2 = str(player1_id), str(player2_id)
        
        # Normalize order for caching
        if p1 > p2:
            p1, p2 = p2, p1
            swapped = True
        else:
            swapped = False
        
        result = {
            'p1_wins': 0,
            'p2_wins': 0,
            'matches': [],
            'by_surface': defaultdict(lambda: {'p1': 0, 'p2': 0}),
            'recent': {'p1': 0, 'p2': 0}  # Last 2 years
        }
        
        two_years_ago = (before_date or datetime.now()) - timedelta(days=730)
        
        # Find all H2H matches
        p1_matches = set(id(m) for m in self._player_matches.get(p1, []))
        
        for match in self._player_matches.get(p2, []):
            if id(match) not in p1_matches:
                continue
            
            if before_date and match.tourney_date >= before_date:
                continue
            
            result['matches'].append(match)
            
            p1_won = match.winner_id == p1
            
            if p1_won:
                result['p1_wins'] += 1
                result['by_surface'][match.surface]['p1'] += 1
            else:
                result['p2_wins'] += 1
                result['by_surface'][match.surface]['p2'] += 1
            
            if match.tourney_date >= two_years_ago:
                if p1_won:
                    result['recent']['p1'] += 1
                else:
                    result['recent']['p2'] += 1
        
        result['matches'].sort(key=lambda m: m.tourney_date, reverse=True)
        
        # Swap back if needed
        if swapped:
            result['p1_wins'], result['p2_wins'] = result['p2_wins'], result['p1_wins']
            result['recent']['p1'], result['recent']['p2'] = result['recent']['p2'], result['recent']['p1']
            for surf in result['by_surface']:
                s = result['by_surface'][surf]
                s['p1'], s['p2'] = s['p2'], s['p1']
        
        return result
    
    def get_player_form(self, player_id: str, before_date: datetime,
                       lookback_days: int = 90) -> Tuple[float, int]:
        """
        Get player's recent form (win rate).
        
        Returns:
            Tuple of (win_rate, num_matches)
        """
        cutoff = before_date - timedelta(days=lookback_days)
        
        wins = 0
        total = 0
        
        for match in self._player_matches.get(str(player_id), []):
            if cutoff <= match.tourney_date < before_date:
                total += 1
                if match.winner_id == str(player_id):
                    wins += 1
        
        if total == 0:
            return (0.5, 0)
        
        return (wins / total, total)
    
    def get_player_ranking(self, player_id: str, 
                          at_date: datetime) -> Optional[int]:
        """Get player's ranking at or before a specific date."""
        rankings = self.rankings.get(str(player_id), [])
        
        # Find most recent ranking before date
        best = None
        for r in rankings:
            if r.ranking_date <= at_date:
                best = r.rank
            else:
                break
        
        return best
    
    def get_surface_record(self, player_id: str, surface: Surface,
                          before_date: datetime = None) -> Tuple[int, int]:
        """
        Get player's win-loss record on a surface.
        
        Returns:
            Tuple of (wins, losses)
        """
        wins = 0
        losses = 0
        
        for match in self._player_matches.get(str(player_id), []):
            if match.surface != surface:
                continue
            if before_date and match.tourney_date >= before_date:
                continue
            
            if match.winner_id == str(player_id):
                wins += 1
            else:
                losses += 1
        
        return (wins, losses)
    
    def get_fatigue_metrics(self, player_id: str, 
                           match_date: datetime) -> Dict:
        """
        Calculate fatigue metrics for a player before a match.
        
        Returns:
            Dict with matches_7d, matches_14d, minutes_14d, etc.
        """
        result = {
            'matches_7d': 0,
            'matches_14d': 0,
            'matches_30d': 0,
            'minutes_14d': 0,
            'days_since_last': None
        }
        
        for match in self._player_matches.get(str(player_id), []):
            if match.tourney_date >= match_date:
                continue
            
            days_ago = (match_date - match.tourney_date).days
            
            if result['days_since_last'] is None or days_ago < result['days_since_last']:
                result['days_since_last'] = days_ago
            
            if days_ago <= 7:
                result['matches_7d'] += 1
            if days_ago <= 14:
                result['matches_14d'] += 1
                if match.minutes:
                    result['minutes_14d'] += match.minutes
            if days_ago <= 30:
                result['matches_30d'] += 1
        
        return result
    
    def find_player_by_name(self, name: str) -> List[ATPPlayer]:
        """Search for players by name (partial match)."""
        name_lower = name.lower()
        results = []
        
        for player in self.players.values():
            if name_lower in player.name.lower():
                results.append(player)
        
        return results
    
    def get_player_stats_summary(self, player_id: str,
                                surface: Surface = None,
                                years: range = None) -> Dict:
        """
        Get aggregate statistics for a player.
        
        Returns:
            Dict with serve stats, return stats, etc.
        """
        stats = {
            'matches': 0,
            'wins': 0,
            'aces': 0,
            'double_faults': 0,
            'first_serve_in': 0,
            'first_serve_total': 0,
            'first_serve_won': 0,
            'second_serve_won': 0,
            'second_serve_total': 0,
            'bp_saved': 0,
            'bp_faced': 0,
            'total_minutes': 0,
            'return_points_won': 0,
            'return_points_total': 0,
            'bp_converted': 0,
            'bp_opportunities': 0,
            'tiebreaks_won': 0,
            'tiebreaks_played': 0,
            'deciding_sets_won': 0,
            'deciding_sets_played': 0,
            'upsets_caused': 0,  # Beat higher ranked
            'upsets_suffered': 0,  # Lost to lower ranked
            'matches_vs_top10': 0,
            'wins_vs_top10': 0,
            'matches_vs_top50': 0,
            'wins_vs_top50': 0,
        }
        
        for match in self._player_matches.get(str(player_id), []):
            if surface and match.surface != surface:
                continue
            if years and match.tourney_date.year not in years:
                continue
            
            is_winner = match.winner_id == str(player_id)
            stats['matches'] += 1
            
            # Determine opponent rank
            if is_winner:
                opp_rank = match.loser_rank
                my_rank = match.winner_rank
            else:
                opp_rank = match.winner_rank
                my_rank = match.loser_rank
            
            if is_winner:
                stats['wins'] += 1
                if match.w_ace:
                    stats['aces'] += match.w_ace
                if match.w_df:
                    stats['double_faults'] += match.w_df
                if match.w_1stIn:
                    stats['first_serve_in'] += match.w_1stIn
                if match.w_svpt:
                    stats['first_serve_total'] += match.w_svpt
                if match.w_1stWon:
                    stats['first_serve_won'] += match.w_1stWon
                if match.w_2ndWon:
                    stats['second_serve_won'] += match.w_2ndWon
                    # Calculate 2nd serve attempts
                    if match.w_svpt and match.w_1stIn:
                        stats['second_serve_total'] += (match.w_svpt - match.w_1stIn)
                if match.w_bpSaved:
                    stats['bp_saved'] += match.w_bpSaved
                if match.w_bpFaced:
                    stats['bp_faced'] += match.w_bpFaced
                
                # Return stats (from loser's serve data)
                if match.l_svpt and match.l_1stWon and match.l_2ndWon:
                    total_opp_serve = match.l_svpt
                    opp_won = match.l_1stWon + match.l_2ndWon
                    stats['return_points_won'] += (total_opp_serve - opp_won)
                    stats['return_points_total'] += total_opp_serve
                if match.l_bpFaced and match.l_bpSaved:
                    stats['bp_converted'] += (match.l_bpFaced - match.l_bpSaved)
                    stats['bp_opportunities'] += match.l_bpFaced
                
                # Upsets
                if my_rank and opp_rank and my_rank > opp_rank:
                    stats['upsets_caused'] += 1
            else:
                if match.l_ace:
                    stats['aces'] += match.l_ace
                if match.l_df:
                    stats['double_faults'] += match.l_df
                if match.l_1stIn:
                    stats['first_serve_in'] += match.l_1stIn
                if match.l_svpt:
                    stats['first_serve_total'] += match.l_svpt
                if match.l_1stWon:
                    stats['first_serve_won'] += match.l_1stWon
                if match.l_2ndWon:
                    stats['second_serve_won'] += match.l_2ndWon
                    if match.l_svpt and match.l_1stIn:
                        stats['second_serve_total'] += (match.l_svpt - match.l_1stIn)
                if match.l_bpSaved:
                    stats['bp_saved'] += match.l_bpSaved
                if match.l_bpFaced:
                    stats['bp_faced'] += match.l_bpFaced
                
                # Return stats
                if match.w_svpt and match.w_1stWon and match.w_2ndWon:
                    total_opp_serve = match.w_svpt
                    opp_won = match.w_1stWon + match.w_2ndWon
                    stats['return_points_won'] += (total_opp_serve - opp_won)
                    stats['return_points_total'] += total_opp_serve
                if match.w_bpFaced and match.w_bpSaved:
                    stats['bp_converted'] += (match.w_bpFaced - match.w_bpSaved)
                    stats['bp_opportunities'] += match.w_bpFaced
                
                # Upsets
                if my_rank and opp_rank and my_rank < opp_rank:
                    stats['upsets_suffered'] += 1
            
            # Top opponent stats
            if opp_rank:
                if opp_rank <= 10:
                    stats['matches_vs_top10'] += 1
                    if is_winner:
                        stats['wins_vs_top10'] += 1
                if opp_rank <= 50:
                    stats['matches_vs_top50'] += 1
                    if is_winner:
                        stats['wins_vs_top50'] += 1
            
            # Parse score for tiebreaks and deciding sets
            if match.score:
                self._parse_score_stats(match.score, match.best_of, is_winner, stats)
            
            if match.minutes:
                stats['total_minutes'] += match.minutes
        
        # Calculate percentages
        if stats['first_serve_total'] > 0:
            stats['first_serve_pct'] = stats['first_serve_in'] / stats['first_serve_total']
        if stats['first_serve_in'] > 0:
            stats['first_serve_won_pct'] = stats['first_serve_won'] / stats['first_serve_in']
        if stats['second_serve_total'] > 0:
            stats['second_serve_won_pct'] = stats['second_serve_won'] / stats['second_serve_total']
        if stats['bp_faced'] > 0:
            stats['bp_save_pct'] = stats['bp_saved'] / stats['bp_faced']
        if stats['matches'] > 0:
            stats['win_pct'] = stats['wins'] / stats['matches']
        if stats['return_points_total'] > 0:
            stats['return_points_won_pct'] = stats['return_points_won'] / stats['return_points_total']
        if stats['bp_opportunities'] > 0:
            stats['bp_convert_pct'] = stats['bp_converted'] / stats['bp_opportunities']
        if stats['first_serve_total'] > 0:
            stats['ace_rate'] = stats['aces'] / stats['first_serve_total']
            stats['df_rate'] = stats['double_faults'] / stats['first_serve_total']
        if stats['tiebreaks_played'] > 0:
            stats['tiebreak_win_pct'] = stats['tiebreaks_won'] / stats['tiebreaks_played']
        if stats['deciding_sets_played'] > 0:
            stats['deciding_set_win_pct'] = stats['deciding_sets_won'] / stats['deciding_sets_played']
        if stats['matches_vs_top10'] > 0:
            stats['win_pct_vs_top10'] = stats['wins_vs_top10'] / stats['matches_vs_top10']
        if stats['matches_vs_top50'] > 0:
            stats['win_pct_vs_top50'] = stats['wins_vs_top50'] / stats['matches_vs_top50']
        
        return stats
    
    def _parse_score_stats(self, score: str, best_of: int, is_winner: bool, stats: Dict):
        """Parse score string for tiebreaks and deciding sets."""
        try:
            sets = score.replace('RET', '').replace('W/O', '').strip().split()
            sets_won = 0
            sets_lost = 0
            
            for s in sets:
                if not s or not s[0].isdigit():
                    continue
                
                # Check for tiebreak (indicated by parentheses)
                if '(' in s:
                    stats['tiebreaks_played'] += 1
                    # Determine if won - winner's games listed first
                    parts = s.split('-')
                    if len(parts) == 2:
                        try:
                            g1 = int(parts[0].split('(')[0])
                            g2 = int(parts[1].split('(')[0])
                            if g1 > g2:
                                stats['tiebreaks_won'] += 1
                        except:
                            pass
                
                # Count sets
                parts = s.replace('(', '-').split('-')
                if len(parts) >= 2:
                    try:
                        g1 = int(parts[0])
                        g2 = int(parts[1])
                        if g1 > g2:
                            sets_won += 1
                        else:
                            sets_lost += 1
                    except:
                        pass
            
            # Check for deciding set
            deciding_set_threshold = (best_of + 1) // 2 - 1
            if sets_won >= deciding_set_threshold and sets_lost >= deciding_set_threshold:
                stats['deciding_sets_played'] += 1
                if is_winner:
                    stats['deciding_sets_won'] += 1
                    
        except Exception:
            pass
    
    def get_tournament_history(self, player_id: str, tourney_name: str,
                              before_date: datetime = None) -> Dict:
        """Get player's history at a specific tournament."""
        result = {
            'matches': 0,
            'wins': 0,
            'titles': 0,
            'finals': 0,
            'best_round': '',
        }
        
        round_order = ['F', 'SF', 'QF', 'R16', 'R32', 'R64', 'R128', 'RR', 'Q3', 'Q2', 'Q1']
        best_round_idx = len(round_order)
        
        for match in self._player_matches.get(str(player_id), []):
            if before_date and match.tourney_date >= before_date:
                continue
            if tourney_name.lower() not in match.tourney_name.lower():
                continue
            
            result['matches'] += 1
            is_winner = match.winner_id == str(player_id)
            
            if is_winner:
                result['wins'] += 1
                if match.round_name == 'F':
                    result['titles'] += 1
            else:
                if match.round_name == 'F':
                    result['finals'] += 1
            
            # Track best round
            for i, r in enumerate(round_order):
                if r in match.round_name:
                    if i < best_round_idx:
                        best_round_idx = i
                        result['best_round'] = match.round_name
                    break
        
        return result
    
    def get_recent_performance_trend(self, player_id: str, 
                                     before_date: datetime,
                                     periods: int = 4,
                                     period_days: int = 30) -> List[float]:
        """
        Get win rates over multiple recent periods to detect trends.
        Returns list of win rates from oldest to most recent.
        """
        trends = []
        
        for i in range(periods - 1, -1, -1):
            period_end = before_date - timedelta(days=i * period_days)
            period_start = period_end - timedelta(days=period_days)
            
            wins = 0
            total = 0
            
            for match in self._player_matches.get(str(player_id), []):
                if period_start <= match.tourney_date < period_end:
                    total += 1
                    if match.winner_id == str(player_id):
                        wins += 1
            
            if total > 0:
                trends.append(wins / total)
            else:
                trends.append(0.5)
        
        return trends


# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python atp_data_loader.py /path/to/tennis_atp")
        print("\nThis module loads Jeff Sackmann's ATP tennis data.")
        print("Clone the repo first: git clone https://github.com/JeffSackmann/tennis_atp.git")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    print(f"\nLoading ATP data from: {data_dir}")
    print("=" * 60)
    
    loader = ATPDataLoader(data_dir)
    loader.load_matches(years=range(2020, 2025))
    loader.load_players()
    
    print(f"\nTotal matches: {len(loader.matches)}")
    print(f"Total players: {len(loader.players)}")
    
    # Example: Find top players
    print("\n" + "=" * 60)
    print("Example: Search for 'Djokovic'")
    print("=" * 60)
    
    results = loader.find_player_by_name("Djokovic")
    for p in results[:3]:
        print(f"  {p.player_id}: {p.name} ({p.country_code})")
        
        # Get H2H vs Nadal
        nadal_results = loader.find_player_by_name("Nadal")
        if nadal_results:
            nadal = nadal_results[0]
            h2h = loader.get_h2h(p.player_id, nadal.player_id)
            print(f"    vs {nadal.name}: {h2h['p1_wins']}-{h2h['p2_wins']}")
    
    print("\n" + "=" * 60)
    print("Data loaded successfully!")
    print("=" * 60)
