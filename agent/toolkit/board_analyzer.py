"""
Advanced Board Texture Analysis for Project PokerMind.

This module implements state-of-the-art board analysis including texture analysis,
draw detection, range advantage calculation, and equity distribution analysis.

Based on professional solver methodology and modern poker theory.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter
from itertools import combinations

from .helpers import parse_card, RANKS, SUITS


class BoardAnalyzer:
    """
    State-of-the-art board texture and equity analysis system.
    
    Provides comprehensive analysis of board textures, drawing opportunities,
    range advantages, and strategic implications for multi-player scenarios.
    """
    
    def __init__(self):
        """Initialize the board analyzer."""
        self.logger = logging.getLogger(__name__)
        
    def analyze_board_texture(self, board: List[str]) -> Dict[str, Any]:
        """
        Comprehensive board texture analysis.
        
        Args:
            board: Community cards (e.g., ['Qh', 'Jd', '9s'])
            
        Returns:
            Dict containing complete texture analysis
        """
        if not board or len(board) < 3:
            return self._get_default_texture_analysis()
            
        try:
            # Parse board cards
            ranks, suits = self._parse_board(board)
            
            # Analyze various texture components
            texture_analysis = {
                "wetness_score": self._calculate_wetness_score(ranks, suits),
                "connectivity": self._analyze_connectivity(ranks),
                "flush_potential": self._analyze_flush_potential(suits),
                "straight_potential": self._analyze_straight_potential(ranks),
                "pair_potential": self._analyze_pair_potential(ranks),
                "high_card_strength": self._analyze_high_cards(ranks),
                "texture_category": "",
                "draw_heavy": False,
                "favorability": {}
            }
            
            # Determine overall texture category
            texture_analysis["texture_category"] = self._classify_texture(texture_analysis)
            texture_analysis["draw_heavy"] = self._is_draw_heavy(texture_analysis)
            
            # Analyze range favorability
            texture_analysis["favorability"] = self._analyze_range_favorability(
                ranks, suits, texture_analysis
            )
            
            return texture_analysis
            
        except Exception as e:
            self.logger.error(f"Board texture analysis error: {e}")
            return self._get_default_texture_analysis()
            
    def _parse_board(self, board: List[str]) -> Tuple[List[int], List[str]]:
        """Parse board into ranks and suits."""
        ranks = []
        suits = []
        
        for card in board:
            rank, suit = parse_card(card)
            ranks.append(RANKS.index(rank))
            suits.append(suit)
            
        return ranks, suits
        
    def _calculate_wetness_score(self, ranks: List[int], suits: List[str]) -> float:
        """
        Calculate how "wet" (draw-heavy) the board is.
        
        Returns score from 0.0 (very dry) to 1.0 (very wet).
        """
        wetness = 0.0
        
        # Connectivity component (0-0.4)
        connectivity_score = self._get_connectivity_score(ranks)
        wetness += connectivity_score * 0.4
        
        # Flush draw component (0-0.3)
        suit_counts = Counter(suits)
        max_suit = max(suit_counts.values()) if suit_counts else 0
        
        if max_suit >= 3:
            flush_component = min(0.3, (max_suit - 2) * 0.15)
        else:
            flush_component = 0.0
            
        wetness += flush_component
        
        # Straight draw component (0-0.3)
        straight_draws = self._count_straight_draws(ranks)
        straight_component = min(0.3, straight_draws * 0.1)
        wetness += straight_component
        
        return min(1.0, wetness)
        
    def _get_connectivity_score(self, ranks: List[int]) -> float:
        """Calculate connectivity score based on rank gaps."""
        if len(ranks) < 2:
            return 0.0
            
        sorted_ranks = sorted(set(ranks), reverse=True)
        total_gaps = 0
        gap_count = 0
        
        for i in range(len(sorted_ranks) - 1):
            gap = sorted_ranks[i] - sorted_ranks[i + 1]
            total_gaps += gap
            gap_count += 1
            
        if gap_count == 0:
            return 0.0
            
        avg_gap = total_gaps / gap_count
        
        # Lower average gap = higher connectivity
        # Gap of 1 = maximum connectivity (1.0)
        # Gap of 6+ = minimum connectivity (0.0)
        connectivity = max(0.0, (6 - avg_gap) / 5)
        
        return connectivity
        
    def _analyze_connectivity(self, ranks: List[int]) -> Dict[str, Any]:
        """Analyze straight connectivity of the board."""
        sorted_ranks = sorted(set(ranks), reverse=True)
        
        # Count gaps between cards
        gaps = []
        for i in range(len(sorted_ranks) - 1):
            gap = sorted_ranks[i] - sorted_ranks[i + 1] - 1
            gaps.append(gap)
            
        connectivity_type = "disconnected"
        if not gaps:
            connectivity_type = "paired"
        elif max(gaps) <= 1:
            connectivity_type = "highly_connected"
        elif max(gaps) <= 3:
            connectivity_type = "connected" 
        
        return {
            "type": connectivity_type,
            "gaps": gaps,
            "max_gap": max(gaps) if gaps else 0,
            "straight_draws_possible": self._count_straight_draws(ranks)
        }
        
    def _analyze_flush_potential(self, suits: List[str]) -> Dict[str, Any]:
        """Analyze flush drawing potential."""
        suit_counts = Counter(suits)
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        
        flush_type = "no_flush_draw"
        if max_suit_count >= 4:
            flush_type = "flush_possible"
        elif max_suit_count == 3:
            flush_type = "flush_draw"
        elif max_suit_count == 2:
            flush_type = "backdoor_flush"
            
        return {
            "type": flush_type,
            "max_suited": max_suit_count,
            "suit_distribution": dict(suit_counts),
            "flush_cards_needed": max(0, 5 - max_suit_count)
        }
        
    def _analyze_straight_potential(self, ranks: List[int]) -> Dict[str, Any]:
        """Analyze straight drawing potential."""
        straight_draws = self._count_straight_draws(ranks)
        gutshot_draws = self._count_gutshot_draws(ranks)
        
        if any(self._check_straight_made(ranks, i) for i in range(len(RANKS) - 4)):
            straight_type = "straight_made"
        elif straight_draws > 0:
            straight_type = "open_ended_draw"
        elif gutshot_draws > 0:
            straight_type = "gutshot_draw"
        else:
            straight_type = "no_straight_draw"
            
        return {
            "type": straight_type,
            "open_ended_draws": straight_draws,
            "gutshot_draws": gutshot_draws,
            "total_outs": straight_draws * 8 + gutshot_draws * 4
        }
        
    def _count_straight_draws(self, ranks: List[int]) -> int:
        """Count open-ended straight draw possibilities."""
        if len(ranks) < 2:
            return 0
            
        unique_ranks = sorted(set(ranks))
        draw_count = 0
        
        # Check each possible straight
        for start in range(len(RANKS) - 4):
            straight_ranks = list(range(start, start + 5))
            board_contribution = len([r for r in unique_ranks if r in straight_ranks])
            
            # Open-ended draw requires exactly 3 cards from the straight
            if board_contribution == 3:
                # Check if it's actually open-ended (not inside)
                if self._is_open_ended_draw(unique_ranks, straight_ranks):
                    draw_count += 1
                    
        return draw_count
        
    def _count_gutshot_draws(self, ranks: List[int]) -> int:
        """Count gutshot (inside) straight draw possibilities."""
        if len(ranks) < 2:
            return 0
            
        unique_ranks = sorted(set(ranks))
        gutshot_count = 0
        
        # Check each possible straight
        for start in range(len(RANKS) - 4):
            straight_ranks = list(range(start, start + 5))
            board_contribution = len([r for r in unique_ranks if r in straight_ranks])
            
            # Gutshot requires exactly 3 cards from straight
            if board_contribution == 3:
                if not self._is_open_ended_draw(unique_ranks, straight_ranks):
                    gutshot_count += 1
                    
        return gutshot_count
        
    def _is_open_ended_draw(self, board_ranks: List[int], straight_ranks: List[int]) -> bool:
        """Check if a 3-card straight contribution is open-ended."""
        board_in_straight = [r for r in board_ranks if r in straight_ranks]
        
        if len(board_in_straight) != 3:
            return False
            
        board_in_straight.sort()
        
        # Open-ended if the 3 cards are consecutive and at ends of straight
        consecutive = all(board_in_straight[i+1] - board_in_straight[i] == 1 
                         for i in range(2))
        
        if not consecutive:
            return False
            
        # Check if at ends of the potential straight
        straight_start = min(straight_ranks)
        straight_end = max(straight_ranks)
        
        board_start = min(board_in_straight)
        board_end = max(board_in_straight)
        
        # Open-ended if board cards are at either end
        at_low_end = board_start == straight_start
        at_high_end = board_end == straight_end
        
        return at_low_end or at_high_end
        
    def _check_straight_made(self, ranks: List[int], start_rank: int) -> bool:
        """Check if a straight is made starting at given rank."""
        straight_ranks = set(range(start_rank, start_rank + 5))
        board_ranks = set(ranks)
        
        return straight_ranks.issubset(board_ranks)
        
    def _analyze_pair_potential(self, ranks: List[int]) -> Dict[str, Any]:
        """Analyze pairing potential of the board."""
        rank_counts = Counter(ranks)
        
        pairs = [rank for rank, count in rank_counts.items() if count >= 2]
        trips = [rank for rank, count in rank_counts.items() if count >= 3]
        
        if trips:
            pair_type = "trips_on_board"
        elif len(pairs) >= 2:
            pair_type = "two_pair_on_board"
        elif pairs:
            pair_type = "pair_on_board"
        else:
            pair_type = "rainbow_board"
            
        return {
            "type": pair_type,
            "pairs": pairs,
            "trips": trips,
            "rank_distribution": dict(rank_counts)
        }
        
    def _analyze_high_cards(self, ranks: List[int]) -> Dict[str, Any]:
        """Analyze high card strength of board."""
        high_cards = [r for r in ranks if r >= RANKS.index('T')]  # T, J, Q, K, A
        broadway_cards = [r for r in ranks if r >= RANKS.index('T')]
        
        avg_rank = sum(ranks) / len(ranks) if ranks else 0
        high_card_percentage = len(high_cards) / len(ranks) if ranks else 0
        
        if high_card_percentage >= 0.67:
            board_strength = "high"
        elif high_card_percentage >= 0.33:
            board_strength = "medium"
        else:
            board_strength = "low"
            
        return {
            "strength": board_strength,
            "high_cards": len(high_cards),
            "broadway_cards": len(broadway_cards),
            "average_rank": avg_rank,
            "high_card_percentage": high_card_percentage
        }
        
    def _classify_texture(self, analysis: Dict[str, Any]) -> str:
        """Classify overall board texture."""
        wetness = analysis["wetness_score"]
        connectivity = analysis["connectivity"]["type"]
        flush_potential = analysis["flush_potential"]["type"]
        
        if wetness > 0.7:
            return "very_wet"
        elif wetness > 0.5:
            return "wet"
        elif wetness > 0.3:
            return "semi_wet"
        elif wetness > 0.15:
            return "dry"
        else:
            return "very_dry"
            
    def _is_draw_heavy(self, analysis: Dict[str, Any]) -> bool:
        """Determine if board is draw-heavy."""
        flush_draws = analysis["flush_potential"]["type"] in ["flush_draw", "flush_possible"]
        straight_draws = analysis["straight_potential"]["open_ended_draws"] > 0
        
        return flush_draws or straight_draws
        
    def _analyze_range_favorability(
        self, 
        ranks: List[int], 
        suits: List[str], 
        texture_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze which types of ranges are favored by this board.
        
        This is crucial for determining who has "range advantage" in multi-player scenarios.
        """
        favorability = {
            "preflop_aggressor": 0.5,  # Baseline
            "preflop_caller": 0.5,
            "tight_ranges": 0.5,
            "loose_ranges": 0.5,
            "position_advantage": 0.0
        }
        
        # High boards favor preflop aggressor (more high pairs and broadways)
        high_card_strength = texture_analysis["high_card_strength"]["strength"]
        if high_card_strength == "high":
            favorability["preflop_aggressor"] += 0.2
            favorability["preflop_caller"] -= 0.2
            favorability["tight_ranges"] += 0.15
            favorability["loose_ranges"] -= 0.15
            
        # Connected boards favor calling ranges (more speculative hands)
        if texture_analysis["connectivity"]["type"] in ["connected", "highly_connected"]:
            favorability["preflop_caller"] += 0.15
            favorability["preflop_aggressor"] -= 0.15
            favorability["loose_ranges"] += 0.1
            favorability["tight_ranges"] -= 0.1
            
        # Draw-heavy boards favor position
        if texture_analysis["draw_heavy"]:
            favorability["position_advantage"] += 0.2
            
        # Clamp values to reasonable ranges
        for key in favorability:
            favorability[key] = max(0.0, min(1.0, favorability[key]))
            
        return favorability
        
    def who_has_range_advantage(
        self,
        board: List[str],
        preflop_aggressor_position: str,
        caller_position: str
    ) -> Dict[str, Any]:
        """
        Determine who has range advantage on this board.
        
        Critical for multi-player strategy - determines who should be
        more aggressive and who should play more passively.
        """
        texture_analysis = self.analyze_board_texture(board)
        favorability = texture_analysis["favorability"]
        
        aggressor_advantage = favorability["preflop_aggressor"]
        caller_advantage = favorability["preflop_caller"]
        position_factor = favorability["position_advantage"]
        
        # Adjust for position
        if preflop_aggressor_position in ["BTN", "CO"]:  # Good position
            aggressor_advantage += position_factor * 0.5
        if caller_position in ["BTN", "CO"]:  # Good position
            caller_advantage += position_factor * 0.5
            
        # Determine winner
        if aggressor_advantage > caller_advantage + 0.1:
            advantage_holder = "preflop_aggressor"
            advantage_strength = aggressor_advantage - caller_advantage
        elif caller_advantage > aggressor_advantage + 0.1:
            advantage_holder = "preflop_caller" 
            advantage_strength = caller_advantage - aggressor_advantage
        else:
            advantage_holder = "neutral"
            advantage_strength = 0.0
            
        return {
            "advantage_holder": advantage_holder,
            "advantage_strength": advantage_strength,
            "aggressor_score": aggressor_advantage,
            "caller_score": caller_advantage,
            "texture_category": texture_analysis["texture_category"],
            "recommendation": self._get_range_advantage_recommendation(
                advantage_holder, advantage_strength, texture_analysis
            )
        }
        
    def _get_range_advantage_recommendation(
        self,
        advantage_holder: str,
        advantage_strength: float,
        texture_analysis: Dict[str, Any]
    ) -> str:
        """Get strategic recommendation based on range advantage."""
        if advantage_holder == "preflop_aggressor":
            if advantage_strength > 0.3:
                return "Strong range advantage - bet aggressively for value and as bluffs"
            elif advantage_strength > 0.15:
                return "Moderate range advantage - continue betting with reasonable frequency"
            else:
                return "Slight range advantage - can bet but be selective"
        elif advantage_holder == "preflop_caller":
            if advantage_strength > 0.3:
                return "Caller has strong advantage - check/call and look for check-raises"
            elif advantage_strength > 0.15:
                return "Caller has moderate advantage - check/call and play passively"
            else:
                return "Caller has slight advantage - play cautiously"
        else:
            return "Neutral board - play standard strategy based on hand strength"
            
    def _get_default_texture_analysis(self) -> Dict[str, Any]:
        """Return default analysis for error cases."""
        return {
            "wetness_score": 0.5,
            "connectivity": {"type": "unknown", "gaps": [], "max_gap": 0},
            "flush_potential": {"type": "no_flush_draw", "max_suited": 0},
            "straight_potential": {"type": "no_straight_draw", "open_ended_draws": 0},
            "pair_potential": {"type": "rainbow_board", "pairs": []},
            "high_card_strength": {"strength": "medium", "high_cards": 0},
            "texture_category": "unknown",
            "draw_heavy": False,
            "favorability": {
                "preflop_aggressor": 0.5,
                "preflop_caller": 0.5,
                "tight_ranges": 0.5,
                "loose_ranges": 0.5,
                "position_advantage": 0.0
            },
            "error": "Invalid or insufficient board data"
        }