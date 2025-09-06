"""
The LAG (Loose-Aggressive) Player

A professional-grade loose-aggressive opponent that:
- Plays a wide range of starting hands
- Plays very aggressively with betting and raising
- Uses position and aggression to win pots
- Applies pressure throughout the tournament
"""

import random
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class LAGHandRange:
    """Represents a loose range of starting hands for LAG play."""

    premium: List[str]  # Top hands for maximum aggression
    strong: List[str]  # Strong hands to play aggressively
    playable: List[str]  # Wide range of playable hands
    bluff: List[str]  # Hands to use for bluffs and semi-bluffs


class LooseAggressivePlayer:
    """
    Loose-Aggressive (LAG) Player Implementation

    Key characteristics:
    - VPIP: ~28-35% (loose starting hand selection)
    - PFR: ~22-28% (very aggressive pre-flop)
    - Aggression Factor: ~3.5-5.0
    - High 3-bet percentage
    - Continuation betting frequency
    - Uses position and aggression as weapons
    """

    def __init__(self, name: str = "LAG_Player"):
        self.name = name
        self.logger = logging.getLogger(f"opponent.{name}")

        # Player statistics tracking
        self.hands_played = 0
        self.vpip_hands = 0  # Voluntarily put money in pot
        self.pfr_hands = 0  # Pre-flop raise
        self.three_bet_hands = 0  # 3-bet frequency
        self.total_bets = 0
        self.total_calls = 0
        self.continuation_bets = 0
        self.continuation_bet_opportunities = 0

        # LAG-specific tracking
        self.bluffs_attempted = 0
        self.value_bets = 0

        # Tournament state awareness
        self.tournament_stage = "early"
        self.stack_size_category = "deep"
        self.last_action_was_raise = False

        # LAG hand ranges - much wider than TAG
        self.hand_ranges = {
            "early": LAGHandRange(
                premium=["AA", "KK", "QQ", "JJ", "AKs", "AK", "AQs", "AQ"],
                strong=["TT", "99", "88", "AJs", "AJ", "KQs", "KQ", "KJs", "KJ", "QJs", "QJ"],
                playable=[
                    "77",
                    "66",
                    "55",
                    "44",
                    "33",
                    "22",
                    "A9s",
                    "A8s",
                    "A7s",
                    "A6s",
                    "A5s",
                    "A4s",
                    "A3s",
                    "A2s",
                    "KTs",
                    "K9s",
                    "K8s",
                    "QTs",
                    "Q9s",
                    "JTs",
                    "J9s",
                    "T9s",
                    "98s",
                    "87s",
                    "76s",
                    "65s",
                ],
                bluff=["K9", "K8", "K7", "Q9", "Q8", "J9", "J8", "T8", "97", "86", "75", "64"],
            ),
            "middle": LAGHandRange(
                premium=["AA", "KK", "QQ", "JJ", "TT", "AKs", "AK", "AQs", "AQ"],
                strong=[
                    "99",
                    "88",
                    "77",
                    "AJs",
                    "AJ",
                    "KQs",
                    "KQ",
                    "KJs",
                    "KJ",
                    "QJs",
                    "QJ",
                    "JTs",
                ],
                playable=[
                    "66",
                    "55",
                    "44",
                    "33",
                    "22",
                    "A9s",
                    "A8s",
                    "A7s",
                    "A6s",
                    "A5s",
                    "A4s",
                    "A3s",
                    "A2s",
                    "KTs",
                    "K9s",
                    "K8s",
                    "K7s",
                    "QTs",
                    "Q9s",
                    "Q8s",
                    "JTs",
                    "J9s",
                    "J8s",
                    "T9s",
                    "T8s",
                    "98s",
                    "87s",
                    "76s",
                    "65s",
                    "54s",
                ],
                bluff=[
                    "K9",
                    "K8",
                    "K7",
                    "K6",
                    "Q9",
                    "Q8",
                    "Q7",
                    "J9",
                    "J8",
                    "J7",
                    "T8",
                    "T7",
                    "97",
                    "96",
                    "86",
                    "85",
                    "75",
                    "74",
                    "64",
                    "63",
                ],
            ),
            "late": LAGHandRange(
                premium=["AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AK", "AQs", "AQ"],
                strong=[
                    "88",
                    "77",
                    "66",
                    "55",
                    "AJs",
                    "AJ",
                    "A9s",
                    "A9",
                    "KQs",
                    "KQ",
                    "KJs",
                    "KJ",
                    "KTs",
                    "QJs",
                    "QJ",
                    "JTs",
                ],
                playable=[
                    "44",
                    "33",
                    "22",
                    "A8s",
                    "A7s",
                    "A6s",
                    "A5s",
                    "A4s",
                    "A3s",
                    "A2s",
                    "K9s",
                    "K8s",
                    "K7s",
                    "K6s",
                    "K5s",
                    "QTs",
                    "Q9s",
                    "Q8s",
                    "Q7s",
                    "JTs",
                    "J9s",
                    "J8s",
                    "J7s",
                    "T9s",
                    "T8s",
                    "T7s",
                    "98s",
                    "97s",
                    "87s",
                    "86s",
                    "76s",
                    "75s",
                    "65s",
                    "64s",
                    "54s",
                    "53s",
                    "43s",
                ],
                bluff=["Any two cards"],  # LAG will bluff with any two in late position
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
            return 1  # Simplified
        except Exception:
            return 1

    def _evaluate_hand_for_lag_play(
        self, hole_cards: List[str], position: str
    ) -> Tuple[str, float]:
        """
        Evaluate starting hand strength for LAG strategy.
        LAG players play many more hands and look for aggression opportunities.

        Returns:
            Tuple of (category, aggression_score) where category is premium/strong/playable/bluff
            and aggression_score represents how aggressively to play (0.0-1.0)
        """
        if len(hole_cards) != 2:
            return "fold", 0.0

        try:
            # Convert hole cards to simplified format
            card1, card2 = hole_cards[0], hole_cards[1]

            # Extract ranks and suits
            rank1, suit1 = card1[0], card1[1]
            rank2, suit2 = card2[0], card2[1]

            # Convert ranks to numeric values
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

            # Determine properties
            is_suited = suit1 == suit2
            is_pair = rank1 == rank2
            is_connected = abs(val1 - val2) <= 1
            gap = abs(val1 - val2) - 1

            # Order by high card first
            high_card = max(val1, val2)
            low_card = min(val1, val2)

            # Premium hands - maximum aggression
            if is_pair and high_card >= 10:  # TT+
                return "premium", 0.95

            if high_card == 14:  # Ace hands
                if low_card >= 10:  # AK, AQ, AJ, AT
                    aggression = 0.9 if is_suited else 0.85
                    return "premium", aggression
                elif low_card >= 5:  # A9-A5 (wheel cards)
                    aggression = 0.7 if is_suited else 0.5
                    return "strong" if is_suited else "playable", aggression
                else:  # A4-A2
                    aggression = 0.6 if is_suited else 0.3
                    return "playable" if is_suited else "bluff", aggression

            # King hands - LAG plays these aggressively
            if high_card == 13:  # King hands
                if low_card >= 9:  # KQ, KJ, KT, K9
                    aggression = 0.8 if is_suited else 0.7
                    return "strong", aggression
                elif is_suited:  # Any suited king
                    return "playable", 0.6
                elif low_card >= 7:  # K8, K7 offsuit
                    return "bluff", 0.4

            # Queen hands
            if high_card == 12:  # Queen hands
                if low_card >= 9:  # QJ, QT, Q9
                    aggression = 0.75 if is_suited else 0.65
                    return "strong", aggression
                elif is_suited:  # Any suited queen
                    return "playable", 0.55

            # Pairs - LAG loves to play pairs aggressively
            if is_pair:
                if high_card >= 7:  # 77+
                    return "playable", 0.7 + (high_card - 7) * 0.05
                elif high_card >= 3:  # 33-66
                    return "playable", 0.5 + (high_card - 3) * 0.05
                else:  # 22
                    return "playable", 0.45

            # Suited connectors and one-gappers - LAG speciality
            if is_suited:
                if is_connected and low_card >= 5:  # 65s+
                    return "playable", 0.65
                elif gap <= 1 and low_card >= 4:  # One gap, 54s+
                    return "playable", 0.6
                elif gap <= 2 and low_card >= 4:  # Two gap, 64s+
                    return "playable", 0.5

            # Broadway cards
            if low_card >= 10:  # JT, J9, T9, etc.
                if is_connected:
                    aggression = 0.65 if is_suited else 0.5
                    return "playable", aggression
                elif gap <= 1:
                    aggression = 0.6 if is_suited else 0.4
                    return "playable" if is_suited else "bluff", aggression

            # Late position - LAG opens up significantly
            if position == "late":
                # Almost any two cards can be played in late position
                if high_card >= 8 or is_suited or gap <= 2:
                    return "bluff", 0.4

            # Any suited cards in position
            if is_suited and position != "early":
                return "playable", 0.4

            # Connector potential
            if is_connected and low_card >= 3:
                aggression = 0.4 if is_suited else 0.25
                return "playable" if position != "early" else "fold", aggression

            # Everything else might be bluff material in position
            if position == "late" and (high_card >= 9 or random.random() < 0.15):
                return "bluff", 0.3

            return "fold", 0.1

        except Exception as e:
            self.logger.error(f"Error evaluating hand for LAG play: {e}")
            return "fold", 0.0

    def _should_play_hand(self, hole_cards: List[str], position: str) -> Tuple[bool, str, float]:
        """
        Determine if LAG should play this hand.

        Returns:
            Tuple of (should_play, category, aggression_score)
        """
        category, aggression = self._evaluate_hand_for_lag_play(hole_cards, position)

        # LAG plays much wider ranges than TAG
        if category in ["premium", "strong"]:
            return True, category, aggression

        if category == "playable":
            # Always play playable hands
            return True, category, aggression

        if category == "bluff":
            # LAG likes to bluff, especially in position
            if position == "late":
                return True, category, aggression
            elif position == "middle" and random.random() < 0.4:
                return True, category, aggression
            elif position == "early" and random.random() < 0.1:  # Occasional early bluff
                return True, category, aggression

        return False, category, aggression

    def _get_lag_bet_sizing(
        self,
        action_type: str,
        pot_size: int,
        our_stack: int,
        valid_actions: List[Dict[str, Any]],
        aggression_score: float,
    ) -> int:
        """Determine bet sizing for LAG strategy - typically larger and more aggressive."""
        try:
            if action_type == "fold":
                return 0

            if action_type == "call":
                for action in valid_actions:
                    if action["action"] == "call":
                        return action.get("amount", 0)
                return 0

            if action_type == "raise":
                big_blind = max(pot_size // 6, 20) if pot_size > 0 else 20

                # LAG uses bigger sizing and varies more
                size_multiplier = 1.0 + (aggression_score * 0.5)  # 1.0x to 1.5x based on aggression

                if self.tournament_stage == "early":
                    # Larger preflop raises
                    base_size = int(big_blind * 3.5 * size_multiplier)
                elif self.tournament_stage == "middle":
                    # Pot-sized or slightly larger bets
                    base_size = int(pot_size * (0.8 + aggression_score * 0.4))
                else:  # Late stage
                    # Very aggressive sizing in late stage
                    base_size = int(pot_size * (1.0 + aggression_score * 0.5))

                # Add some randomization to be less predictable
                variance = int(base_size * 0.2)
                final_size = base_size + random.randint(-variance, variance)

                return min(max(final_size, big_blind), our_stack)

        except Exception as e:
            self.logger.error(f"Error calculating LAG bet sizing: {e}")

        return min(pot_size // 2, our_stack) if pot_size > 0 else min(40, our_stack)

    def declare_action(
        self,
        valid_actions: List[Dict[str, Any]],
        hole_cards: List[str],
        round_state: Dict[str, Any],
    ) -> Tuple[str, int]:
        """
        Main decision-making method for LAG player.

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
                f"LAG decision: {street} street, position: {position}, "
                f"cards: {hole_cards}, pot: {pot_size}"
            )

            # Preflop decision logic
            if street == "preflop":
                return self._decide_preflop_lag(
                    hole_cards, position, valid_actions, pot_size, our_stack, round_state
                )
            else:
                return self._decide_postflop_lag(
                    hole_cards,
                    round_state.get("community_card", []),
                    valid_actions,
                    pot_size,
                    our_stack,
                    round_state,
                )

        except Exception as e:
            self.logger.error(f"Error in LAG declare_action: {e}")
            return self._safe_action(valid_actions)

    def _decide_preflop_lag(
        self,
        hole_cards: List[str],
        position: str,
        valid_actions: List[Dict[str, Any]],
        pot_size: int,
        our_stack: int,
        round_state: Dict[str, Any],
    ) -> Tuple[str, int]:
        """Make preflop decisions based on LAG strategy."""

        should_play, category, aggression_score = self._should_play_hand(hole_cards, position)

        if not should_play:
            return "fold", 0

        self.vpip_hands += 1

        # Check action history
        action_history = round_state.get("action_histories", {}).get("preflop", [])
        has_raise = any("raise" in str(action) for action in action_history)
        num_raises = sum(1 for action in action_history if "raise" in str(action))

        if not has_raise:
            # First to act or everyone folded - LAG almost always raises
            if category in ["premium", "strong", "playable"]:
                self.pfr_hands += 1
                self.total_bets += 1
                amount = self._get_lag_bet_sizing(
                    "raise", pot_size, our_stack, valid_actions, aggression_score
                )
                return "raise", amount
            else:  # Bluff hands
                # LAG raises bluffs frequently, especially in position
                if position == "late" or random.random() < 0.6:
                    self.pfr_hands += 1
                    self.total_bets += 1
                    amount = self._get_lag_bet_sizing(
                        "raise", pot_size, our_stack, valid_actions, aggression_score
                    )
                    return "raise", amount
                else:
                    # Occasionally limp with bluff hands
                    self.total_calls += 1
                    amount = self._get_lag_bet_sizing(
                        "call", pot_size, our_stack, valid_actions, aggression_score
                    )
                    return "call", amount
        else:
            # Facing a raise - LAG loves to 3-bet
            if category == "premium":
                # Always 3-bet premium hands
                self.three_bet_hands += 1
                self.total_bets += 1
                amount = self._get_lag_bet_sizing(
                    "raise", pot_size, our_stack, valid_actions, aggression_score
                )
                return "raise", amount
            elif category == "strong":
                # 3-bet strong hands frequently
                if random.random() < 0.7:  # 70% 3-bet frequency
                    self.three_bet_hands += 1
                    self.total_bets += 1
                    amount = self._get_lag_bet_sizing(
                        "raise", pot_size, our_stack, valid_actions, aggression_score
                    )
                    return "raise", amount
                else:
                    self.total_calls += 1
                    amount = self._get_lag_bet_sizing(
                        "call", pot_size, our_stack, valid_actions, aggression_score
                    )
                    return "call", amount
            elif category == "playable":
                # 3-bet playable hands sometimes, call sometimes
                three_bet_frequency = 0.4 if position == "late" else 0.2
                if num_raises == 1 and random.random() < three_bet_frequency:
                    self.three_bet_hands += 1
                    self.total_bets += 1
                    amount = self._get_lag_bet_sizing(
                        "raise", pot_size, our_stack, valid_actions, aggression_score
                    )
                    return "raise", amount
                elif num_raises == 1:  # Only call facing single raise
                    self.total_calls += 1
                    amount = self._get_lag_bet_sizing(
                        "call", pot_size, our_stack, valid_actions, aggression_score
                    )
                    return "call", amount
                else:
                    return "fold", 0
            else:  # Bluff category
                # LAG 3-bets bluffs occasionally for balance
                if num_raises == 1 and position == "late" and random.random() < 0.15:
                    self.three_bet_hands += 1
                    self.bluffs_attempted += 1
                    self.total_bets += 1
                    amount = self._get_lag_bet_sizing(
                        "raise", pot_size, our_stack, valid_actions, aggression_score
                    )
                    return "raise", amount
                else:
                    return "fold", 0

    def _decide_postflop_lag(
        self,
        hole_cards: List[str],
        community_cards: List[str],
        valid_actions: List[Dict[str, Any]],
        pot_size: int,
        our_stack: int,
        round_state: Dict[str, Any],
    ) -> Tuple[str, int]:
        """Make postflop decisions based on LAG strategy."""

        try:
            # Estimate hand strength and draw potential
            hand_strength = self._estimate_postflop_strength(hole_cards, community_cards)
            draw_potential = self._estimate_draw_potential(hole_cards, community_cards)

            # Check for previous betting action
            street = round_state.get("street", "flop")
            action_history = round_state.get("action_histories", {}).get(street, [])
            facing_bet = any(
                "raise" in str(action) or "bet" in str(action) for action in action_history
            )

            # LAG continuation betting
            if self.last_action_was_raise and not facing_bet:
                self.continuation_bet_opportunities += 1

                # LAG has high c-bet frequency
                c_bet_frequency = 0.8  # 80% c-bet frequency
                if random.random() < c_bet_frequency or hand_strength > 0.5:
                    self.continuation_bets += 1
                    self.total_bets += 1
                    amount = self._get_lag_bet_sizing(
                        "raise", pot_size, our_stack, valid_actions, 0.7
                    )
                    return "raise", amount

            # Strong hands - bet for value or raise
            if hand_strength > 0.7:
                if facing_bet:
                    # Raise for value or call
                    if random.random() < 0.7:  # 70% raise frequency with strong hands
                        self.value_bets += 1
                        self.total_bets += 1
                        amount = self._get_lag_bet_sizing(
                            "raise", pot_size, our_stack, valid_actions, 0.8
                        )
                        return "raise", amount
                    else:
                        self.total_calls += 1
                        amount = self._get_lag_bet_sizing(
                            "call", pot_size, our_stack, valid_actions, 0.7
                        )
                        return "call", amount
                else:
                    # Bet for value
                    self.value_bets += 1
                    self.total_bets += 1
                    amount = self._get_lag_bet_sizing(
                        "raise", pot_size, our_stack, valid_actions, 0.8
                    )
                    return "raise", amount

            # Medium strength hands with draws
            elif hand_strength > 0.4 or draw_potential > 0.6:
                if facing_bet:
                    # Semi-bluff raise or call with draws
                    if draw_potential > 0.7 and random.random() < 0.5:
                        self.total_bets += 1
                        amount = self._get_lag_bet_sizing(
                            "raise", pot_size, our_stack, valid_actions, 0.6
                        )
                        return "raise", amount
                    elif hand_strength > 0.4 or draw_potential > 0.5:
                        self.total_calls += 1
                        amount = self._get_lag_bet_sizing(
                            "call", pot_size, our_stack, valid_actions, 0.5
                        )
                        return "call", amount
                    else:
                        return "fold", 0
                else:
                    # Bet with medium hands and draws
                    if hand_strength > 0.5 or (draw_potential > 0.6 and random.random() < 0.6):
                        self.total_bets += 1
                        amount = self._get_lag_bet_sizing(
                            "raise", pot_size, our_stack, valid_actions, 0.6
                        )
                        return "raise", amount
                    else:
                        # Check
                        amount = self._get_lag_bet_sizing(
                            "call", pot_size, our_stack, valid_actions, 0.3
                        )
                        return "call", amount

            # Weak hands - bluff or fold
            else:
                if facing_bet:
                    # LAG occasionally bluff-raises weak hands
                    if random.random() < 0.2:  # 20% bluff frequency
                        self.bluffs_attempted += 1
                        self.total_bets += 1
                        amount = self._get_lag_bet_sizing(
                            "raise", pot_size, our_stack, valid_actions, 0.4
                        )
                        return "raise", amount
                    else:
                        return "fold", 0
                else:
                    # Bluff with weak hands sometimes
                    if random.random() < 0.4:  # 40% bluff frequency when checked to
                        self.bluffs_attempted += 1
                        self.total_bets += 1
                        amount = self._get_lag_bet_sizing(
                            "raise", pot_size, our_stack, valid_actions, 0.4
                        )
                        return "raise", amount
                    else:
                        # Check
                        amount = self._get_lag_bet_sizing(
                            "call", pot_size, our_stack, valid_actions, 0.2
                        )
                        return "call", amount

        except Exception as e:
            self.logger.error(f"Error in LAG postflop decision: {e}")
            return self._safe_action(valid_actions)

    def _estimate_postflop_strength(
        self, hole_cards: List[str], community_cards: List[str]
    ) -> float:
        """Estimate hand strength postflop (simplified but more aggressive than TAG)."""
        try:
            if not hole_cards or len(hole_cards) != 2:
                return 0.1

            # Get all cards
            all_cards = hole_cards + (community_cards or [])

            # Count ranks
            ranks = [card[0] for card in all_cards]
            rank_counts = {}
            for rank in ranks:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1

            # Check for made hands
            max_count = max(rank_counts.values()) if rank_counts else 1

            if max_count >= 4:
                return 0.95  # Four of a kind
            elif max_count >= 3:
                # Check for full house
                pair_count = sum(1 for count in rank_counts.values() if count >= 2)
                if pair_count >= 2:
                    return 0.9  # Full house
                else:
                    return 0.8  # Three of a kind
            elif max_count >= 2:
                # Check for two pair
                pair_count = sum(1 for count in rank_counts.values() if count >= 2)
                if pair_count >= 2:
                    return 0.65  # Two pair
                else:
                    # One pair - evaluate pair strength
                    pair_rank = [rank for rank, count in rank_counts.items() if count >= 2][0]
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
                    pair_value = rank_values.get(pair_rank, 7)
                    return 0.4 + (pair_value - 2) * 0.02  # 0.4 to 0.64 based on pair strength
            else:
                # High card - LAG values high cards more aggressively
                hole_ranks = [card[0] for card in hole_cards]
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
                    return 0.35  # Ace or King high - LAG plays these more aggressively
                elif max_hole >= 11:
                    return 0.25  # Queen or Jack high
                else:
                    return 0.15  # Low cards

        except Exception as e:
            self.logger.error(f"Error estimating hand strength: {e}")
            return 0.2

    def _estimate_draw_potential(self, hole_cards: List[str], community_cards: List[str]) -> float:
        """Estimate drawing potential for semi-bluffs and aggressive play."""
        try:
            if not hole_cards or len(hole_cards) != 2 or not community_cards:
                return 0.0

            # Simplified draw evaluation
            all_cards = hole_cards + community_cards
            suits = [card[1] for card in all_cards]
            ranks = [card[0] for card in all_cards]

            # Count suits for flush draws
            suit_counts = {}
            for suit in suits:
                suit_counts[suit] = suit_counts.get(suit, 0) + 1

            max_suit_count = max(suit_counts.values()) if suit_counts else 0

            # Flush draws
            if max_suit_count == 4:
                return 0.8  # Strong flush draw
            elif max_suit_count == 3:
                return 0.6  # Weak flush draw

            # Straight draws (simplified)
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

            numeric_ranks = sorted([rank_values.get(rank, 2) for rank in ranks])

            # Check for straight potential
            consecutive_count = 1
            max_consecutive = 1

            for i in range(1, len(numeric_ranks)):
                if numeric_ranks[i] == numeric_ranks[i - 1] + 1:
                    consecutive_count += 1
                    max_consecutive = max(max_consecutive, consecutive_count)
                else:
                    consecutive_count = 1

            if max_consecutive >= 4:
                return 0.7  # Open-ended straight draw
            elif max_consecutive >= 3:
                return 0.5  # Gutshot draw

            return 0.2  # Weak draw potential

        except Exception as e:
            self.logger.error(f"Error estimating draw potential: {e}")
            return 0.1

    def _safe_action(self, valid_actions: List[Dict[str, Any]]) -> Tuple[str, int]:
        """Return a safe fallback action."""
        # LAG prefers calling to folding when possible
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
        three_bet_pct = (self.three_bet_hands / max(self.hands_played, 1)) * 100

        total_aggressive_actions = self.total_bets + self.total_calls
        aggression_factor = (
            (self.total_bets / max(self.total_calls, 1)) if total_aggressive_actions > 0 else 0
        )

        c_bet_pct = (self.continuation_bets / max(self.continuation_bet_opportunities, 1)) * 100

        return {
            "hands_played": self.hands_played,
            "vpip": vpip,  # Should be ~28-35% for LAG
            "pfr": pfr,  # Should be ~22-28% for LAG
            "three_bet_pct": three_bet_pct,  # Should be high for LAG
            "aggression_factor": aggression_factor,  # Should be ~3.5-5.0 for LAG
            "c_bet_pct": c_bet_pct,  # Should be ~80%+ for LAG
            "bluffs_attempted": self.bluffs_attempted,
            "value_bets": self.value_bets,
            "tournament_stage": self.tournament_stage,
            "stack_category": self.stack_size_category,
        }

    # Track last action for continuation betting
    def receive_round_start_message(
        self, round_count: int, hole_card: List[str], seats: List[Dict[str, Any]]
    ):
        """Called when round starts."""
        self.last_action_was_raise = False
        self.logger.debug(f"LAG player {self.name} round {round_count} started")

    # PyPokerEngine compatibility methods
    def receive_game_start_message(self, game_info: Dict[str, Any]):
        """Called when game starts."""
        self.logger.info(f"LAG player {self.name} starting new game")

    def receive_street_start_message(self, street: str, round_state: Dict[str, Any]):
        """Called when street starts."""
        pass

    def receive_game_update_message(self, action: Dict[str, Any], round_state: Dict[str, Any]):
        """Called when any action happens."""
        # Track if our last action was a raise for c-betting
        if action.get("player_uuid") == getattr(self, "uuid", None):
            if action.get("action") == "raise":
                self.last_action_was_raise = True
            else:
                self.last_action_was_raise = False

    def receive_round_result_message(
        self,
        winners: List[Dict[str, Any]],
        hand_info: List[Dict[str, Any]],
        round_state: Dict[str, Any],
    ):
        """Called when round ends."""
        pass
