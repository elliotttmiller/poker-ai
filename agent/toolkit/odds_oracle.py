"""
Pre-computed Odds Oracle for PokerMind.

This module provides instant access to pre-calculated poker probabilities,
eliminating expensive real-time calculations during critical decision making.
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache


class OddsOracle:
    """
    Lightning-fast pre-computed odds lookup system.

    The Oracle provides instant access to:
    - Pre-flop equity matchups (169x169 matrix)
    - Draw completion probabilities
    - Board texture danger levels
    - Position-based adjustments

    All data is pre-computed and cached for maximum performance.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the Odds Oracle.

        Args:
            data_dir: Directory containing pre-computed data files.
                     If None, uses default oracle/data directory.
        """
        self.logger = logging.getLogger(__name__)

        # Set data directory
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.data_dir = data_dir

        # Initialize data containers
        self.preflop_equity = {}
        self.draw_odds = {}
        self.board_texture_odds = {}
        self.position_adjustments = {}
        self.metadata = {}

        # Load all data
        self._load_all_data()

        self.logger.info("OddsOracle initialized with pre-computed data")

    def _load_all_data(self) -> None:
        """Load all pre-computed data from JSON files."""
        try:
            # Load pre-flop equity table
            self.preflop_equity = self._load_json_file("preflop_equity.json")

            # Load draw completion odds
            self.draw_odds = self._load_json_file("draw_completion_odds.json")

            # Load board texture odds
            self.board_texture_odds = self._load_json_file("board_texture_odds.json")

            # Load position adjustments
            self.position_adjustments = self._load_json_file("position_adjustments.json")

            # Load metadata
            self.metadata = self._load_json_file("metadata.json")

            self.logger.info(
                f"Loaded oracle data: "
                f"{len(self.preflop_equity)} pre-flop hands, "
                f"{len(self.draw_odds)} draw types, "
                f"{len(self.board_texture_odds)} board textures"
            )

        except Exception as e:
            self.logger.error(f"Failed to load oracle data: {e}")
            self._create_fallback_data()

    def _load_json_file(self, filename: str) -> Dict[str, Any]:
        """Load data from a JSON file."""
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            self.logger.warning(f"Oracle data file not found: {filepath}")
            return {}

        with open(filepath, "r") as f:
            return json.load(f)

    def _create_fallback_data(self) -> None:
        """Create minimal fallback data if files are missing."""
        self.logger.warning("Using fallback oracle data")

        # Minimal pre-flop data
        self.preflop_equity = {
            "AA": {"KK": 0.82, "QQ": 0.85, "AK": 0.74},
            "KK": {"AA": 0.18, "QQ": 0.73, "AK": 0.65},
            "AK": {"AA": 0.26, "KK": 0.35, "QQ": 0.43},
        }

        # Basic draw odds
        self.draw_odds = {
            "flush_draw": {"turn_and_river": 0.35},
            "straight_draw": {"turn_and_river": 0.32},
        }

        # Basic board textures
        self.board_texture_odds = {"rainbow_low": 0.15, "monotone": 0.65}

    @lru_cache(maxsize=1000)
    def get_preflop_equity(self, our_hand: str, opponent_hand: str) -> float:
        """
        Get pre-flop equity against opponent hand.

        Args:
            our_hand: Our hand in standard notation (e.g., 'AKs', 'QQ')
            opponent_hand: Opponent's hand in same notation

        Returns:
            Our equity as a decimal (0.0 to 1.0)

        Example:
            >>> oracle.get_preflop_equity('AA', 'KK')
            0.82
        """
        our_hand = self._normalize_hand(our_hand)
        opponent_hand = self._normalize_hand(opponent_hand)

        if our_hand in self.preflop_equity:
            if opponent_hand in self.preflop_equity[our_hand]:
                return self.preflop_equity[our_hand][opponent_hand]

        # Fallback estimation if exact matchup not found
        return self._estimate_preflop_equity(our_hand, opponent_hand)

    def _normalize_hand(self, hand: str) -> str:
        """Normalize hand notation for consistent lookup."""
        if len(hand) == 2 and hand[0] == hand[1]:
            # Pair - ensure descending order (AA, KK, etc.)
            return hand
        elif len(hand) == 3 and hand.endswith("s"):
            # Suited - ensure high card first
            rank1, rank2 = hand[0], hand[1]
            if self._rank_value(rank1) < self._rank_value(rank2):
                return rank2 + rank1 + "s"
            return hand
        elif len(hand) == 2:
            # Offsuit - ensure high card first
            rank1, rank2 = hand[0], hand[1]
            if self._rank_value(rank1) < self._rank_value(rank2):
                return rank2 + rank1
            return hand

        return hand

    def _rank_value(self, rank: str) -> int:
        """Get numeric value of a card rank."""
        rank_values = {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "T": 10,
            "J": 11,
            "Q": 12,
            "K": 13,
            "A": 14,
        }
        return rank_values.get(rank, 0)

    def _estimate_preflop_equity(self, our_hand: str, opponent_hand: str) -> float:
        """Estimate pre-flop equity if exact data unavailable."""
        # Very simplified estimation - in production, use lookup tables
        if our_hand == opponent_hand:
            return 0.50

        # Rough estimation based on hand types
        our_strength = self._estimate_hand_strength(our_hand)
        opp_strength = self._estimate_hand_strength(opponent_hand)

        if our_strength > opp_strength:
            return 0.55 + min(0.25, (our_strength - opp_strength) * 0.5)
        else:
            return 0.45 - min(0.25, (opp_strength - our_strength) * 0.5)

    def _estimate_hand_strength(self, hand: str) -> float:
        """Rough hand strength estimation."""
        if len(hand) == 2 and hand[0] == hand[1]:
            # Pairs
            rank_val = self._rank_value(hand[0])
            return 0.3 + (rank_val / 14.0) * 0.6

        # High cards
        rank1_val = self._rank_value(hand[0])
        rank2_val = self._rank_value(hand[1]) if len(hand) > 1 else 0
        base_strength = (rank1_val + rank2_val) / 28.0

        # Suited bonus
        if hand.endswith("s"):
            base_strength *= 1.15

        return base_strength

    def get_draw_completion_odds(
        self, draw_type: str, streets_remaining: str = "turn_and_river"
    ) -> float:
        """
        Get odds of completing a draw.

        Args:
            draw_type: Type of draw ('flush_draw', 'straight_draw', etc.)
            streets_remaining: 'turn', 'river', or 'turn_and_river'

        Returns:
            Completion probability as decimal

        Example:
            >>> oracle.get_draw_completion_odds('flush_draw', 'turn_and_river')
            0.348
        """
        if draw_type in self.draw_odds:
            return self.draw_odds[draw_type].get(streets_remaining, 0.0)

        return 0.0

    def get_board_danger_level(self, board_texture: str) -> float:
        """
        Get danger level of board texture.

        Args:
            board_texture: Board classification

        Returns:
            Danger level from 0.0 (safe) to 1.0 (very dangerous)
        """
        return self.board_texture_odds.get(board_texture, 0.3)  # Default moderate danger

    def get_position_multiplier(self, position: str) -> float:
        """
        Get position-based equity multiplier.

        Args:
            position: Position name ('button', 'early_position', etc.)

        Returns:
            Multiplier to apply to hand strength/equity
        """
        return self.position_adjustments.get(position, 1.0)

    def analyze_matchup(self, our_hand: str, opponent_range: List[str]) -> Dict[str, Any]:
        """
        Analyze our equity against an opponent range.

        Args:
            our_hand: Our hand notation
            opponent_range: List of opponent possible hands

        Returns:
            Detailed analysis including average equity, best/worst cases
        """
        if not opponent_range:
            return {"average_equity": 0.5, "confidence": 0.0}

        equities = []
        for opp_hand in opponent_range:
            equity = self.get_preflop_equity(our_hand, opp_hand)
            equities.append(equity)

        avg_equity = sum(equities) / len(equities)

        return {
            "average_equity": round(avg_equity, 3),
            "min_equity": min(equities),
            "max_equity": max(equities),
            "range_size": len(opponent_range),
            "confidence": 0.9 if len(opponent_range) > 5 else 0.6,
        }

    def get_quick_recommendation(
        self, our_hand: str, position: str, board_texture: str = None
    ) -> Dict[str, Any]:
        """
        Get quick play recommendation based on oracle data.

        Args:
            our_hand: Our hand notation
            position: Our position
            board_texture: Optional board texture classification

        Returns:
            Quick recommendation with confidence
        """
        # Get base hand strength
        hand_strength = self._estimate_hand_strength(our_hand)

        # Apply position adjustment
        position_mult = self.get_position_multiplier(position)
        adjusted_strength = hand_strength * position_mult

        # Apply board danger if provided
        if board_texture:
            danger = self.get_board_danger_level(board_texture)
            adjusted_strength *= 1.0 - danger * 0.3  # Reduce strength on dangerous boards

        # Make recommendation
        if adjusted_strength > 0.70:
            action = "aggressive"
            confidence = 0.85
        elif adjusted_strength > 0.50:
            action = "moderate"
            confidence = 0.70
        elif adjusted_strength > 0.30:
            action = "cautious"
            confidence = 0.60
        else:
            action = "fold"
            confidence = 0.80

        return {
            "recommendation": action,
            "confidence": confidence,
            "hand_strength": hand_strength,
            "adjusted_strength": adjusted_strength,
            "factors": {
                "position_multiplier": position_mult,
                "board_danger": (
                    self.get_board_danger_level(board_texture) if board_texture else None
                ),
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get oracle statistics and metadata."""
        return {
            "preflop_hands": len(self.preflop_equity),
            "draw_types": len(self.draw_odds),
            "board_textures": len(self.board_texture_odds),
            "positions": len(self.position_adjustments),
            "metadata": self.metadata,
            "data_loaded": bool(self.preflop_equity),
        }
