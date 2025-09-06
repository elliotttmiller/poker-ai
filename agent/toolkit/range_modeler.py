"""
Professional Position-Aware Range Modeler for Project PokerMind.

This module implements professional-grade range modeling with position awareness,
based on GTO solver methodology and modern poker theory.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import re

from .helpers import RANKS, SUITS


@dataclass
class HandRange:
    """Represents a poker hand range with associated probabilities."""
    hands: Dict[str, float]  # Hand -> probability mapping
    total_combos: int
    position: str
    action_context: str
    
    
class RangeModeler:
    """
    Professional position-aware range modeling system.
    
    Implements range parsing, construction, and analysis based on
    modern poker theory and GTO principles.
    """
    
    def __init__(self):
        """Initialize the range modeler with professional preflop ranges."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize standard position-based preflop ranges
        self.gto_preflop_ranges = self._initialize_gto_ranges()
        self.position_order = ['UTG', 'UTG+1', 'MP', 'CO', 'BTN', 'SB', 'BB']
        
    def parse_range_notation(self, range_string: str) -> Set[str]:
        """
        Parse standard range notation into specific hands.
        
        Supports formats like:
        - "AA,KK,QQ" (specific pairs)
        - "AKs,AQs" (suited hands)
        - "AKo" (offsuit hands) 
        - "A5s+" (suited connectors and better)
        - "22+" (pocket pairs and better)
        - "A2s-A5s" (suited ranges)
        
        Args:
            range_string: Standard poker range notation
            
        Returns:
            Set of specific hand combinations
        """
        if not range_string or not isinstance(range_string, str):
            return set()
            
        hands = set()
        
        # Split by commas and process each part
        parts = [part.strip() for part in range_string.split(',')]
        
        for part in parts:
            try:
                hands.update(self._parse_range_part(part))
            except Exception as e:
                self.logger.warning(f"Could not parse range part '{part}': {e}")
                
        return hands
        
    def _parse_range_part(self, part: str) -> Set[str]:
        """Parse a single part of range notation."""
        hands = set()
        
        # Handle ranges like "A2s-A5s"
        if '-' in part:
            return self._parse_range_sequence(part)
            
        # Handle plus notation like "22+" or "A5s+"
        if part.endswith('+'):
            return self._parse_plus_notation(part[:-1])
            
        # Handle specific hands like "AA", "AKs", "AKo"
        return self._parse_specific_hand(part)
        
    def _parse_range_sequence(self, sequence: str) -> Set[str]:
        """Parse range sequences like 'A2s-A5s'."""
        if '-' not in sequence:
            return set()
            
        start_hand, end_hand = sequence.split('-', 1)
        start_hand = start_hand.strip()
        end_hand = end_hand.strip()
        
        # TODO: Implement full sequence parsing
        # For now, return both endpoints
        hands = set()
        hands.update(self._parse_specific_hand(start_hand))
        hands.update(self._parse_specific_hand(end_hand))
        
        return hands
        
    def _parse_plus_notation(self, base_hand: str) -> Set[str]:
        """Parse plus notation like '22+' or 'A5s+'."""
        hands = set()
        
        # Handle pocket pairs like "22+"
        if len(base_hand) == 2 and base_hand[0] == base_hand[1]:
            pair_rank = base_hand[0]
            pair_index = RANKS.index(pair_rank)
            
            # Add all pairs from this rank up to AA
            for rank_idx in range(pair_index, len(RANKS)):
                rank = RANKS[rank_idx]
                hands.update(self._get_pair_combinations(rank))
                
        # Handle suited hands like "A5s+"
        elif base_hand.endswith('s') and len(base_hand) == 3:
            high_rank = base_hand[0]
            low_rank = base_hand[1]
            
            high_idx = RANKS.index(high_rank)
            low_idx = RANKS.index(low_rank)
            
            # Add all suited combinations with this high card and better kickers
            for kicker_idx in range(low_idx, high_idx):
                kicker_rank = RANKS[kicker_idx]
                hands.update(self._get_suited_combinations(high_rank, kicker_rank))
                
        # Handle offsuit hands like "A5o+"  
        elif base_hand.endswith('o') and len(base_hand) == 3:
            high_rank = base_hand[0]
            low_rank = base_hand[1]
            
            high_idx = RANKS.index(high_rank)
            low_idx = RANKS.index(low_rank)
            
            # Add all offsuit combinations with this high card and better kickers
            for kicker_idx in range(low_idx, high_idx):
                kicker_rank = RANKS[kicker_idx]
                hands.update(self._get_offsuit_combinations(high_rank, kicker_rank))
                
        return hands
        
    def _parse_specific_hand(self, hand: str) -> Set[str]:
        """Parse specific hand like 'AA', 'AKs', 'AKo'."""
        hands = set()
        
        if len(hand) == 2:
            # Pocket pair like "AA"
            if hand[0] == hand[1]:
                hands.update(self._get_pair_combinations(hand[0]))
            # Two different ranks - add both suited and offsuit
            else:
                hands.update(self._get_suited_combinations(hand[0], hand[1]))
                hands.update(self._get_offsuit_combinations(hand[0], hand[1]))
                
        elif len(hand) == 3:
            rank1, rank2, suit_indicator = hand[0], hand[1], hand[2]
            
            if suit_indicator == 's':
                hands.update(self._get_suited_combinations(rank1, rank2))
            elif suit_indicator == 'o':
                hands.update(self._get_offsuit_combinations(rank1, rank2))
                
        return hands
        
    def _get_pair_combinations(self, rank: str) -> Set[str]:
        """Get all combinations for a pocket pair."""
        if rank not in RANKS:
            return set()
            
        hands = set()
        for i, suit1 in enumerate(SUITS):
            for j, suit2 in enumerate(SUITS[i+1:], i+1):
                hands.add(f"{rank}{suit1}{rank}{suit2}")
                
        return hands
        
    def _get_suited_combinations(self, rank1: str, rank2: str) -> Set[str]:
        """Get all suited combinations for two ranks."""
        if rank1 not in RANKS or rank2 not in RANKS:
            return set()
            
        hands = set()
        for suit in SUITS:
            hands.add(f"{rank1}{suit}{rank2}{suit}")
            
        return hands
        
    def _get_offsuit_combinations(self, rank1: str, rank2: str) -> Set[str]:
        """Get all offsuit combinations for two ranks."""
        if rank1 not in RANKS or rank2 not in RANKS:
            return set()
            
        hands = set()
        for suit1 in SUITS:
            for suit2 in SUITS:
                if suit1 != suit2:
                    hands.add(f"{rank1}{suit1}{rank2}{suit2}")
                    
        return hands
        
    def get_position_range(
        self, 
        position: str, 
        action: str = "open", 
        stack_depth: str = "100bb"
    ) -> HandRange:
        """
        Get GTO-based range for position and action.
        
        Args:
            position: Position name ('UTG', 'MP', 'CO', 'BTN', 'SB', 'BB')
            action: Action type ('open', 'call', '3bet', 'fold')
            stack_depth: Stack depth category ('short', '100bb', 'deep')
            
        Returns:
            HandRange object with position-appropriate range
        """
        range_key = f"{position}_{action}_{stack_depth}"
        
        if range_key in self.gto_preflop_ranges:
            range_data = self.gto_preflop_ranges[range_key]
            return HandRange(
                hands=range_data,
                total_combos=sum(range_data.values()),
                position=position,
                action_context=f"{action}_{stack_depth}"
            )
        
        # Fallback to basic positional range
        return self._get_basic_positional_range(position, action)
        
    def _get_basic_positional_range(self, position: str, action: str) -> HandRange:
        """Get basic positional range as fallback."""
        # Simplified ranges based on position
        basic_ranges = {
            'UTG': "22+,A2s+,K9s+,Q9s+,J9s+,T9s,98s,ATo+,KTo+",
            'MP': "22+,A2s+,K8s+,Q8s+,J8s+,T8s+,97s+,87s,ATo+,KTo+,QTo+",
            'CO': "22+,A2s+,K6s+,Q6s+,J7s+,T7s+,96s+,86s+,75s+,65s,A8o+,K9o+,Q9o+,J9o+",
            'BTN': "22+,A2s+,K2s+,Q4s+,J6s+,T6s+,95s+,85s+,74s+,64s+,54s,A5o+,K8o+,Q8o+,J8o+,T8o+",
            'SB': "22+,A2s+,K5s+,Q6s+,J7s+,T7s+,96s+,85s+,75s+,64s+,A7o+,K9o+,Q9o+,J9o+",
            'BB': "22+,A2s+,K2s+,Q2s+,J4s+,T6s+,95s+,84s+,74s+,63s+,53s+,43s,A2o+,K5o+,Q7o+,J8o+"
        }
        
        range_string = basic_ranges.get(position, "22+,A5s+,KTs+,ATo+")
        parsed_hands = self.parse_range_notation(range_string)
        
        # Convert to probability mapping (all hands equally likely for simplicity)
        hand_probs = {hand: 1.0 for hand in parsed_hands}
        
        return HandRange(
            hands=hand_probs,
            total_combos=len(parsed_hands),
            position=position,
            action_context=action
        )
        
    def _initialize_gto_ranges(self) -> Dict[str, Dict[str, float]]:
        """Initialize GTO-based preflop ranges."""
        # TODO: Implement full GTO range database
        # For now, return empty dict to use basic ranges
        return {}
        
    def estimate_opponent_range(
        self, 
        opponent_stats: Dict[str, Any],
        position: str,
        action_sequence: List[str],
        board: List[str] = None
    ) -> HandRange:
        """
        Estimate opponent's range based on stats and actions.
        
        Args:
            opponent_stats: Player statistics (VPIP, PFR, etc.)
            position: Opponent's position
            action_sequence: Sequence of actions taken
            board: Community cards if postflop
            
        Returns:
            Estimated HandRange for opponent
        """
        # Get baseline range for position
        baseline_range = self.get_position_range(position)
        
        # Adjust range based on player stats
        adjusted_range = self._adjust_range_for_player_type(
            baseline_range, opponent_stats
        )
        
        # Further adjust based on action sequence
        final_range = self._adjust_range_for_actions(
            adjusted_range, action_sequence, board
        )
        
        return final_range
        
    def _adjust_range_for_player_type(
        self, 
        baseline_range: HandRange, 
        stats: Dict[str, Any]
    ) -> HandRange:
        """Adjust range based on player type."""
        vpip = stats.get('vpip', 0.25)
        pfr = stats.get('pfr', 0.15)
        
        # Create adjusted hand probabilities
        adjusted_hands = {}
        
        for hand, prob in baseline_range.hands.items():
            # Adjust probability based on player stats
            if vpip > 0.35:  # Loose player
                adjusted_prob = prob * 1.3
            elif vpip < 0.15:  # Tight player  
                adjusted_prob = prob * 0.7
            else:
                adjusted_prob = prob
                
            # Cap probability at 1.0
            adjusted_hands[hand] = min(1.0, adjusted_prob)
            
        return HandRange(
            hands=adjusted_hands,
            total_combos=sum(adjusted_hands.values()),
            position=baseline_range.position,
            action_context=f"{baseline_range.action_context}_adjusted"
        )
        
    def _adjust_range_for_actions(
        self,
        range_estimate: HandRange,
        action_sequence: List[str], 
        board: List[str] = None
    ) -> HandRange:
        """Adjust range based on action sequence."""
        # TODO: Implement action-based range narrowing
        # For now, return the input range unchanged
        return range_estimate
        
    def calculate_range_advantage(
        self,
        our_range: HandRange,
        opponent_range: HandRange, 
        board: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate range advantage on given board.
        
        This analysis determines who has the stronger range overall
        and identifies betting/bluffing opportunities.
        """
        # TODO: Implement proper range vs range analysis
        # For now, return simplified analysis
        
        our_strength = sum(our_range.hands.values()) / len(our_range.hands)
        opp_strength = sum(opponent_range.hands.values()) / len(opponent_range.hands)
        
        advantage = our_strength - opp_strength
        
        return {
            "advantage": advantage,
            "our_strength": our_strength,
            "opponent_strength": opp_strength,
            "recommendation": "bet" if advantage > 0.1 else "check" if advantage > -0.1 else "fold",
            "confidence": min(0.9, abs(advantage) * 2)
        }