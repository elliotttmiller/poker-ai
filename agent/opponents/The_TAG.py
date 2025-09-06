"""
The TAG (Tight-Aggressive) Player

A professional-grade tight-aggressive opponent that:
- Plays a tight range of starting hands
- Plays aggressively when involved in a pot
- Adjusts strategy based on position and tournament stage
- Uses mathematical concepts for decision making
"""

import random
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class HandRange:
    """Represents a range of starting hands."""

    premium: List[str]  # AA, KK, QQ, etc.
    strong: List[str]  # AK, AQ, JJ, etc.
    playable: List[str]  # Medium pairs, suited connectors, etc.


class TightAggressivePlayer:
    """
    Tight-Aggressive (TAG) Player Implementation

    Key characteristics:
    - VPIP: ~18-22% (tight starting hand selection)
    - PFR: ~15-18% (aggressive when playing)
    - Aggression Factor: ~2.5-3.5
    - Position-aware play
    - Tournament stage awareness
    """

    def __init__(self, name: str = "TAG_Player"):
        self.name = name
        self.logger = logging.getLogger(f"opponent.{name}")

        # Player statistics tracking
        self.hands_played = 0
        self.vpip_hands = 0  # Voluntarily put money in pot
        self.pfr_hands = 0  # Pre-flop raise
        self.total_bets = 0
        self.total_calls = 0

        # Tournament state awareness
        self.tournament_stage = "early"  # early, middle, late
        self.stack_size_category = "deep"  # deep, medium, short

        # Hand ranges by position
        self.hand_ranges = {
            "early": HandRange(
                premium=["AA", "KK", "QQ", "AKs", "AK"],
                strong=["JJ", "TT", "AQs", "AQ", "KQs"],
                playable=["99", "88", "AJs", "KJs", "QJs", "JTs"],
            ),
            "middle": HandRange(
                premium=["AA", "KK", "QQ", "JJ", "AKs", "AK"],
                strong=["TT", "99", "AQs", "AQ", "KQs", "KQ", "AJs", "KJs"],
                playable=["88", "77", "A9s", "KTs", "QJs", "JTs", "T9s", "98s"],
            ),
            "late": HandRange(
                premium=["AA", "KK", "QQ", "JJ", "TT", "AKs", "AK"],
                strong=["99", "88", "77", "AQs", "AQ", "KQs", "KQ", "AJs", "AJ", "KJs", "KJ"],
                playable=[
                    "66",
                    "55",
                    "44",
                    "33",
                    "22",
                    "A9s",
                    "A8s",
                    "A7s",
                    "KTs",
                    "K9s",
                    "QJs",
                    "QTs",
                    "Q9s",
                    "JTs",
                    "J9s",
                    "T9s",
                    "98s",
                    "87s",
                    "76s",
                ],
            ),
        }

    def update_tournament_stage(self, round_state: Dict[str, Any]) -> None:
        """Update tournament stage awareness based on blinds and stack sizes."""
        try:
            # Get current stack and blind info
            our_stack = self._get_our_stack(round_state)
            small_blind = round_state.get("small_blind_amount", 10)
            big_blind = small_blind * 2

            # Calculate M-ratio (stack / (blinds + antes))
            m_ratio = our_stack / (small_blind + big_blind)

            if m_ratio > 20:
                self.tournament_stage = "early"
                self.stack_size_category = "deep"
            elif m_ratio > 10:
                self.tournament_stage = "middle"
                self.stack_size_category = "medium"
            else:
                self.tournament_stage = "late"
                self.stack_size_category = "short"

        except Exception as e:
            self.logger.warning(f"Error updating tournament stage: {e}")

    def _get_our_stack(self, round_state: Dict[str, Any]) -> int:
        """Get our current stack size from round state."""
        try:
            # This would need to be adapted based on the actual round_state structure
            seats = round_state.get("seats", [])
            for seat in seats:
                if hasattr(self, "uuid") and seat.get("uuid") == self.uuid:
                    return seat.get("stack", 1000)
            return 1000  # Default fallback
        except Exception:
            return 1000

    def _get_position_category(self, round_state: Dict[str, Any]) -> str:
        """Determine our position category (early/middle/late)."""
        try:
            # This is a simplified position calculation
            # In a real implementation, this would consider dealer button position
            num_players = len([s for s in round_state.get("seats", []) if s.get("stack", 0) > 0])
            our_seat = self._get_our_seat_id(round_state)

            if num_players <= 3:
                return "late"  # Short-handed play
            elif our_seat <= num_players // 3:
                return "early"
            elif our_seat <= 2 * num_players // 3:
                return "middle"
            else:
                return "late"
        except Exception:
            return "middle"

    def _get_our_seat_id(self, round_state: Dict[str, Any]) -> int:
        """Get our seat ID from round state."""
        try:
            # Simplified - would need proper implementation
            return 1
        except Exception:
            return 1

    def _evaluate_hand_strength(self, hole_cards: List[str]) -> Tuple[str, float]:
        """
        Evaluate starting hand strength and return category and score.

        Returns:
            Tuple of (category, strength_score) where category is premium/strong/playable/fold
            and strength_score is 0.0-1.0
        """
        if len(hole_cards) != 2:
            return "fold", 0.0

        try:
            # Convert hole cards to simplified format
            card1, card2 = hole_cards[0], hole_cards[1]

            # Extract ranks and suits
            rank1, suit1 = card1[0], card1[1]
            rank2, suit2 = card2[0], card2[1]

            # Convert ranks to numeric values for comparison
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

            val1 = rank_values.get(rank1, 2)
            val2 = rank_values.get(rank2, 2)

            # Determine if suited
            is_suited = suit1 == suit2
            is_pair = rank1 == rank2

            # Order by high card first
            high_card = max(val1, val2)
            low_card = min(val1, val2)

            # Check for premium hands
            if is_pair and high_card >= 10:  # TT+
                if high_card >= 12:  # QQ+
                    return "premium", 0.95
                else:
                    return "premium", 0.85

            if high_card == 14:  # Ace hands
                if low_card >= 10:  # AK, AQ, AJ, AT
                    strength = 0.9 if is_suited else 0.8
                    return "premium" if low_card >= 12 else "strong", strength
                elif low_card >= 7:  # A9-A7
                    strength = 0.6 if is_suited else 0.4
                    return "playable" if is_suited else "fold", strength

            if high_card == 13:  # King hands
                if low_card >= 10:  # KQ, KJ, KT
                    strength = 0.75 if is_suited else 0.65
                    return "strong", strength
                elif low_card >= 8 and is_suited:  # K9s, K8s
                    return "playable", 0.5

            # Pairs
            if is_pair:
                if high_card >= 7:  # 77+
                    return "playable", 0.6 + (high_card - 7) * 0.05
                elif high_card >= 5:  # 55, 66
                    return "playable", 0.4

            # Suited connectors
            if is_suited and abs(val1 - val2) <= 1 and min(val1, val2) >= 6:
                return "playable", 0.45

            # Everything else is fold
            return "fold", 0.1

        except Exception as e:
            self.logger.error(f"Error evaluating hand strength: {e}")
            return "fold", 0.0

    def _should_play_hand(self, hole_cards: List[str], position: str) -> bool:
        """Determine if we should play this hand based on position and strength."""
        category, strength = self._evaluate_hand_strength(hole_cards)

        # Get position-specific ranges
        pos_range = self.hand_ranges.get(position, self.hand_ranges["middle"])

        # Always play premium hands
        if category == "premium":
            return True

        # Play strong hands in any position
        if category == "strong":
            return True

        # Play playable hands in middle/late position
        if category == "playable" and position in ["middle", "late"]:
            return True

        # In tournament late stage, open up ranges slightly
        if self.tournament_stage == "late" and category == "playable":
            return True

        return False

    def _get_bet_sizing(
        self, action_type: str, pot_size: int, our_stack: int, valid_actions: List[Dict[str, Any]]
    ) -> int:
        """Determine appropriate bet sizing for TAG strategy."""
        try:
            if action_type == "fold":
                return 0

            if action_type == "call":
                # Find call amount
                for action in valid_actions:
                    if action["action"] == "call":
                        return action.get("amount", 0)
                return 0

            if action_type == "raise":
                # TAG prefers standard raise sizes
                big_blind = pot_size // 3 if pot_size > 0 else 20  # Estimate BB

                # Preflop: 2.5-3.5x BB
                # Postflop: 60-75% pot
                if self.tournament_stage == "early":
                    return min(int(big_blind * 3), our_stack)
                elif self.tournament_stage == "middle":
                    return min(int(pot_size * 0.7), our_stack)
                else:  # Late stage - more aggressive
                    return min(int(pot_size * 0.8), our_stack)

        except Exception as e:
            self.logger.error(f"Error calculating bet sizing: {e}")

        return 0

    def declare_action(
        self,
        valid_actions: List[Dict[str, Any]],
        hole_cards: List[str],
        round_state: Dict[str, Any],
    ) -> Tuple[str, int]:
        """
        Main decision-making method for TAG player.

        Returns:
            Tuple of (action, amount)
        """
        try:
            self.hands_played += 1
            self.update_tournament_stage(round_state)

            position = self._get_position_category(round_state)
            street = round_state.get("street", "preflop")
            pot_size = round_state.get("pot", {}).get("main", {}).get("amount", 0)
            our_stack = self._get_our_stack(round_state)

            self.logger.debug(
                f"TAG decision: {street} street, position: {position}, "
                f"cards: {hole_cards}, pot: {pot_size}"
            )

            # Preflop decision logic
            if street == "preflop":
                return self._decide_preflop(
                    hole_cards, position, valid_actions, pot_size, our_stack, round_state
                )
            else:
                return self._decide_postflop(
                    hole_cards,
                    round_state.get("community_card", []),
                    valid_actions,
                    pot_size,
                    our_stack,
                    round_state,
                )

        except Exception as e:
            self.logger.error(f"Error in declare_action: {e}")
            # Safe fallback
            return self._safe_action(valid_actions)

    def _decide_preflop(
        self,
        hole_cards: List[str],
        position: str,
        valid_actions: List[Dict[str, Any]],
        pot_size: int,
        our_stack: int,
        round_state: Dict[str, Any],
    ) -> Tuple[str, int]:
        """Make preflop decisions based on TAG strategy."""

        # Check if we should play this hand
        if not self._should_play_hand(hole_cards, position):
            return "fold", 0

        self.vpip_hands += 1

        # Determine if we should raise or call
        category, strength = self._evaluate_hand_strength(hole_cards)

        # Check for previous action
        has_raise = any(
            "raise" in str(action)
            for action in round_state.get("action_histories", {}).get("preflop", [])
        )

        if not has_raise:
            # First to act or everyone folded - raise with strong hands
            if category in ["premium", "strong"] or (category == "playable" and position == "late"):
                self.pfr_hands += 1
                self.total_bets += 1
                amount = self._get_bet_sizing("raise", pot_size, our_stack, valid_actions)
                return "raise", amount
            else:
                # Limp with marginal hands in early position
                self.total_calls += 1
                amount = self._get_bet_sizing("call", pot_size, our_stack, valid_actions)
                return "call", amount
        else:
            # Facing a raise - TAG strategy: fold marginal hands, call/3-bet strong hands
            if category == "premium":
                # 3-bet premium hands
                self.total_bets += 1
                amount = self._get_bet_sizing("raise", pot_size, our_stack, valid_actions)
                return "raise", amount
            elif category == "strong":
                # Call with strong hands
                self.total_calls += 1
                amount = self._get_bet_sizing("call", pot_size, our_stack, valid_actions)
                return "call", amount
            else:
                # Fold playable hands facing a raise (tight)
                return "fold", 0

    def _decide_postflop(
        self,
        hole_cards: List[str],
        community_cards: List[str],
        valid_actions: List[Dict[str, Any]],
        pot_size: int,
        our_stack: int,
        round_state: Dict[str, Any],
    ) -> Tuple[str, int]:
        """Make postflop decisions based on TAG strategy."""

        # Simplified postflop logic for TAG
        # In a full implementation, this would include hand equity calculation,
        # board texture analysis, opponent modeling, etc.

        try:
            # Estimate hand strength (simplified)
            hand_strength = self._estimate_postflop_strength(hole_cards, community_cards)

            # Check for previous betting action
            street = round_state.get("street", "flop")
            action_history = round_state.get("action_histories", {}).get(street, [])
            facing_bet = any(
                "raise" in str(action) or "bet" in str(action) for action in action_history
            )

            if hand_strength > 0.7:  # Strong hand
                if facing_bet:
                    # Call or raise with strong hands
                    if random.random() < 0.6:  # 60% raise, 40% call
                        self.total_bets += 1
                        amount = self._get_bet_sizing("raise", pot_size, our_stack, valid_actions)
                        return "raise", amount
                    else:
                        self.total_calls += 1
                        amount = self._get_bet_sizing("call", pot_size, our_stack, valid_actions)
                        return "call", amount
                else:
                    # Bet for value
                    self.total_bets += 1
                    amount = self._get_bet_sizing("raise", pot_size, our_stack, valid_actions)
                    return "raise", amount

            elif hand_strength > 0.4:  # Medium hand
                if facing_bet:
                    # Call with drawing hands, fold weak hands
                    if hand_strength > 0.5:
                        self.total_calls += 1
                        amount = self._get_bet_sizing("call", pot_size, our_stack, valid_actions)
                        return "call", amount
                    else:
                        return "fold", 0
                else:
                    # Check with medium hands
                    amount = self._get_bet_sizing("call", pot_size, our_stack, valid_actions)
                    return "call", amount
            else:  # Weak hand
                if facing_bet:
                    return "fold", 0
                else:
                    # Check with weak hands
                    amount = self._get_bet_sizing("call", pot_size, our_stack, valid_actions)
                    return "call", amount

        except Exception as e:
            self.logger.error(f"Error in postflop decision: {e}")
            return self._safe_action(valid_actions)

    def _estimate_postflop_strength(
        self, hole_cards: List[str], community_cards: List[str]
    ) -> float:
        """Estimate hand strength postflop (simplified)."""
        # This is a very simplified hand strength estimation
        # A full implementation would use proper poker hand evaluation

        try:
            if not hole_cards or len(hole_cards) != 2:
                return 0.1

            # Get card ranks
            hole_ranks = [card[0] for card in hole_cards]
            community_ranks = [card[0] for card in community_cards] if community_cards else []

            all_ranks = hole_ranks + community_ranks
            rank_counts = {}
            for rank in all_ranks:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1

            # Check for pairs, trips, etc.
            max_count = max(rank_counts.values()) if rank_counts else 1

            if max_count >= 4:
                return 0.95  # Four of a kind
            elif max_count >= 3:
                return 0.85  # Three of a kind
            elif max_count >= 2:
                # Check for multiple pairs
                pair_count = sum(1 for count in rank_counts.values() if count >= 2)
                if pair_count >= 2:
                    return 0.75  # Two pair
                else:
                    return 0.6  # One pair
            else:
                # High card - evaluate based on high card
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

                hole_values = [rank_values.get(rank, 2) for rank in hole_ranks]
                max_hole = max(hole_values)

                if max_hole >= 13:
                    return 0.45  # Ace or King high
                elif max_hole >= 11:
                    return 0.35  # Queen or Jack high
                else:
                    return 0.25  # Low cards

        except Exception as e:
            self.logger.error(f"Error estimating hand strength: {e}")
            return 0.3  # Default medium-low strength

    def _safe_action(self, valid_actions: List[Dict[str, Any]]) -> Tuple[str, int]:
        """Return a safe fallback action."""
        # Try to call if possible, otherwise fold
        for action in valid_actions:
            if action["action"] == "call":
                return "call", action.get("amount", 0)
        return "fold", 0

    def get_statistics(self) -> Dict[str, float]:
        """Get player statistics for analysis."""
        if self.hands_played == 0:
            return {}

        vpip = (self.vpip_hands / self.hands_played) * 100
        pfr = (self.pfr_hands / self.hands_played) * 100

        total_aggressive_actions = self.total_bets + self.total_calls
        aggression_factor = (
            (self.total_bets / max(self.total_calls, 1)) if total_aggressive_actions > 0 else 0
        )

        return {
            "hands_played": self.hands_played,
            "vpip": vpip,  # Should be ~18-22% for TAG
            "pfr": pfr,  # Should be ~15-18% for TAG
            "aggression_factor": aggression_factor,  # Should be ~2.5-3.5 for TAG
            "tournament_stage": self.tournament_stage,
            "stack_category": self.stack_size_category,
        }

    # PyPokerEngine compatibility methods
    def receive_game_start_message(self, game_info: Dict[str, Any]):
        """Called when game starts."""
        self.logger.info(f"TAG player {self.name} starting new game")

    def receive_round_start_message(
        self, round_count: int, hole_card: List[str], seats: List[Dict[str, Any]]
    ):
        """Called when round starts."""
        self.logger.debug(f"TAG player {self.name} round {round_count} started")

    def receive_street_start_message(self, street: str, round_state: Dict[str, Any]):
        """Called when street starts."""
        pass

    def receive_game_update_message(self, action: Dict[str, Any], round_state: Dict[str, Any]):
        """Called when any action happens."""
        pass

    def receive_round_result_message(
        self,
        winners: List[Dict[str, Any]],
        hand_info: List[Dict[str, Any]],
        round_state: Dict[str, Any],
    ):
        """Called when round ends."""
        pass
