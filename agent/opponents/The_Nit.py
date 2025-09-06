"""
The Nit (The Rock) Player

A professional-grade extremely tight opponent that:
- Plays an extremely narrow range of starting hands
- Only plays premium hands aggressively
- Folds to aggression unless holding very strong hands
- Plays very predictably but is hard to extract value from
"""

import random
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass 
class NitHandRange:
    """Represents an extremely tight range of starting hands."""
    premium: List[str]  # Only the absolute best hands
    playable: List[str] # Very few additional hands in late position


class NitPlayer:
    """
    Nit (The Rock) Player Implementation
    
    Key characteristics:
    - VPIP: ~8-12% (extremely tight starting hand selection)
    - PFR: ~6-10% (only raises with premium hands)
    - Aggression Factor: ~1.5-2.5 (passive)
    - Very low 3-bet percentage
    - Folds to most aggression
    - Only continues with strong hands
    - Extremely predictable but hard to bluff
    """
    
    def __init__(self, name: str = "Nit_Player"):
        self.name = name
        self.logger = logging.getLogger(f"opponent.{name}")
        
        # Player statistics tracking
        self.hands_played = 0
        self.vpip_hands = 0  # Voluntarily put money in pot
        self.pfr_hands = 0   # Pre-flop raise
        self.three_bet_hands = 0  # Should be very low
        self.total_bets = 0
        self.total_calls = 0
        self.folds_to_aggression = 0
        self.aggression_faced = 0
        
        # Nit-specific tracking
        self.premium_hands_played = 0
        self.times_folded_to_single_bet = 0
        
        # Tournament state awareness
        self.tournament_stage = "early"
        self.stack_size_category = "deep"
        
        # Extremely tight hand ranges
        self.hand_ranges = {
            "early": NitHandRange(
                premium=["AA", "KK", "QQ", "AKs", "AK"],  # Only the absolute nuts
                playable=[]  # Nothing else
            ),
            "middle": NitHandRange(
                premium=["AA", "KK", "QQ", "JJ", "AKs", "AK"],
                playable=["TT", "AQs"]  # Slightly wider but still very tight
            ),
            "late": NitHandRange(
                premium=["AA", "KK", "QQ", "JJ", "TT", "AKs", "AK", "AQs", "AQ"],
                playable=["99", "88", "AJs", "KQs"]  # Only in late position
            )
        }
    
    def update_tournament_stage(self, round_state: Dict[str, Any]) -> None:
        """Update tournament stage awareness - nits become slightly less tight when short-stacked."""
        try:
            our_stack = self._get_our_stack(round_state)
            small_blind = round_state.get("small_blind_amount", 10)
            big_blind = small_blind * 2
            
            # Calculate M-ratio
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
            return 1000
        except Exception:
            return 1000
            
    def _get_position_category(self, round_state: Dict[str, Any]) -> str:
        """Determine our position category."""
        try:
            num_players = len([s for s in round_state.get("seats", []) if s.get("stack", 0) > 0])
            our_seat = self._get_our_seat_id(round_state)
            
            if num_players <= 3:
                return "late"
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
    
    def _evaluate_hand_for_nit_play(self, hole_cards: List[str]) -> Tuple[str, float]:
        """
        Evaluate starting hand strength for extremely tight nit strategy.
        
        Returns:
            Tuple of (category, strength_score) where category is premium/playable/fold
            and strength_score represents hand strength (0.0-1.0)
        """
        if len(hole_cards) != 2:
            return "fold", 0.0
            
        try:
            card1, card2 = hole_cards[0], hole_cards[1]
            rank1, suit1 = card1[0], card1[1]
            rank2, suit2 = card2[0], card2[1]
            
            rank_values = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, 
                          "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
            
            val1 = rank_values.get(rank1, 2)
            val2 = rank_values.get(rank2, 2)
            
            is_suited = suit1 == suit2
            is_pair = rank1 == rank2
            
            high_card = max(val1, val2)
            low_card = min(val1, val2)
            
            # Pocket pairs - only premium pairs for nits
            if is_pair:
                if high_card >= 12:  # QQ+
                    return "premium", 0.95
                elif high_card >= 10:  # JJ, TT
                    return "premium", 0.85
                elif high_card >= 8 and self.tournament_stage == "late":  # 88, 99 when short
                    return "playable", 0.7
                else:
                    return "fold", 0.1
                    
            # Ace hands - very selective
            if high_card == 14:
                if low_card >= 13:  # AK
                    strength = 0.9 if is_suited else 0.85
                    return "premium", strength
                elif low_card >= 12:  # AQ
                    if is_suited or self.tournament_stage == "late":
                        return "premium" if is_suited else "playable", 0.8 if is_suited else 0.7
                    else:
                        return "fold", 0.3
                elif low_card >= 11 and is_suited:  # AJs
                    return "playable", 0.65
                else:
                    return "fold", 0.2
                    
            # King hands - only KQ suited and only sometimes
            if high_card == 13 and low_card >= 12 and is_suited:
                if self.tournament_stage == "late":
                    return "playable", 0.6
                else:
                    return "fold", 0.3
                    
            # Everything else is an automatic fold for a nit
            return "fold", 0.1
            
        except Exception as e:
            self.logger.error(f"Error evaluating hand for nit play: {e}")
            return "fold", 0.0
    
    def _should_play_hand(self, hole_cards: List[str], position: str) -> Tuple[bool, str, float]:
        """
        Determine if nit should play this hand.
        
        Returns:
            Tuple of (should_play, category, strength_score)
        """
        category, strength = self._evaluate_hand_for_nit_play(hole_cards)
        
        # Get position-specific ranges
        pos_range = self.hand_ranges.get(position, self.hand_ranges["early"])
        
        # Always play premium hands
        if category == "premium":
            return True, category, strength
            
        # Only play playable hands in late position or when short-stacked
        if category == "playable":
            if position == "late" or self.tournament_stage == "late":
                return True, category, strength
            else:
                return False, category, strength
                
        return False, category, strength
    
    def _get_nit_bet_sizing(self, action_type: str, pot_size: int, our_stack: int,
                           valid_actions: List[Dict[str, Any]]) -> int:
        """Determine bet sizing for nit strategy - typically smaller and more conservative."""
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
                
                # Nits use smaller, more conservative sizing
                if self.tournament_stage == "early":
                    # Standard 2.5x preflop raise
                    return min(int(big_blind * 2.5), our_stack)
                elif self.tournament_stage == "middle":
                    # Small to medium postflop bets
                    return min(int(pot_size * 0.5), our_stack)
                else:  # Late stage - only bet when very strong
                    return min(int(pot_size * 0.6), our_stack)
                    
        except Exception as e:
            self.logger.error(f"Error calculating nit bet sizing: {e}")
            
        return min(pot_size // 3, our_stack) if pot_size > 0 else min(30, our_stack)
    
    def declare_action(self, valid_actions: List[Dict[str, Any]], hole_cards: List[str], 
                      round_state: Dict[str, Any]) -> Tuple[str, int]:
        """
        Main decision-making method for Nit player.
        
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
            
            self.logger.debug(f"Nit decision: {street} street, position: {position}, "
                            f"cards: {hole_cards}, pot: {pot_size}")
            
            # Preflop decision logic
            if street == "preflop":
                return self._decide_preflop_nit(hole_cards, position, valid_actions, 
                                              pot_size, our_stack, round_state)
            else:
                return self._decide_postflop_nit(hole_cards, round_state.get("community_card", []),
                                               valid_actions, pot_size, our_stack, round_state)
                                           
        except Exception as e:
            self.logger.error(f"Error in Nit declare_action: {e}")
            return self._safe_fold(valid_actions)
    
    def _decide_preflop_nit(self, hole_cards: List[str], position: str, 
                           valid_actions: List[Dict[str, Any]], pot_size: int,
                           our_stack: int, round_state: Dict[str, Any]) -> Tuple[str, int]:
        """Make preflop decisions based on nit strategy."""
        
        should_play, category, strength = self._should_play_hand(hole_cards, position)
        
        if not should_play:
            return "fold", 0
            
        self.vpip_hands += 1
        
        # Check for previous raises
        action_history = round_state.get("action_histories", {}).get("preflop", [])
        has_raise = any("raise" in str(action) for action in action_history)
        num_raises = sum(1 for action in action_history if "raise" in str(action))
        
        if not has_raise:
            # First to act - nits usually just limp with most hands
            if category == "premium" and strength >= 0.9:
                # Only raise with the absolute best hands
                self.pfr_hands += 1
                self.total_bets += 1
                self.premium_hands_played += 1
                amount = self._get_nit_bet_sizing("raise", pot_size, our_stack, valid_actions)
                return "raise", amount
            else:
                # Limp with other playable hands
                self.total_calls += 1
                amount = self._get_nit_bet_sizing("call", pot_size, our_stack, valid_actions)
                return "call", amount
        else:
            # Facing a raise - nits fold almost everything
            self.aggression_faced += 1
            
            if category == "premium" and strength >= 0.9:
                # Only continue with the absolute nuts
                if num_raises == 1:
                    # Call with premium hands facing single raise
                    self.total_calls += 1
                    amount = self._get_nit_bet_sizing("call", pot_size, our_stack, valid_actions)
                    return "call", amount
                else:
                    # Fold to multiple raises unless we have AA or KK
                    if strength >= 0.95:
                        self.total_calls += 1
                        amount = self._get_nit_bet_sizing("call", pot_size, our_stack, valid_actions)
                        return "call", amount
                    else:
                        self.folds_to_aggression += 1
                        return "fold", 0
            elif category == "premium" and strength >= 0.85:
                # Marginal premium hands - only call single raise
                if num_raises == 1:
                    self.total_calls += 1
                    amount = self._get_nit_bet_sizing("call", pot_size, our_stack, valid_actions)
                    return "call", amount
                else:
                    self.folds_to_aggression += 1
                    return "fold", 0
            else:
                # Everything else is a fold
                self.folds_to_aggression += 1
                return "fold", 0
    
    def _decide_postflop_nit(self, hole_cards: List[str], community_cards: List[str],
                            valid_actions: List[Dict[str, Any]], pot_size: int, 
                            our_stack: int, round_state: Dict[str, Any]) -> Tuple[str, int]:
        """Make postflop decisions based on nit strategy."""
        
        try:
            # Estimate hand strength
            hand_strength = self._estimate_postflop_strength(hole_cards, community_cards)
            
            # Check for previous betting action
            street = round_state.get("street", "flop")
            action_history = round_state.get("action_histories", {}).get(street, [])
            facing_bet = any("raise" in str(action) or "bet" in str(action) for action in action_history)
            
            if facing_bet:
                self.aggression_faced += 1
                
                # Nits only continue with strong hands
                if hand_strength > 0.8:
                    # Very strong hands - call or occasionally raise
                    if hand_strength > 0.9 and random.random() < 0.3:
                        # Rarely raise with the nuts
                        self.total_bets += 1
                        amount = self._get_nit_bet_sizing("raise", pot_size, our_stack, valid_actions)
                        return "raise", amount
                    else:
                        # Usually just call
                        self.total_calls += 1
                        amount = self._get_nit_bet_sizing("call", pot_size, our_stack, valid_actions)
                        return "call", amount
                elif hand_strength > 0.6:
                    # Medium hands - call small bets only
                    call_amount = 0
                    for action in valid_actions:
                        if action["action"] == "call":
                            call_amount = action.get("amount", 0)
                            break
                    
                    # Only call if it's a small bet relative to pot
                    if call_amount <= pot_size * 0.3:
                        self.total_calls += 1
                        return "call", call_amount
                    else:
                        self.folds_to_aggression += 1
                        self.times_folded_to_single_bet += 1
                        return "fold", 0
                else:
                    # Weak hands - always fold to any bet
                    self.folds_to_aggression += 1
                    self.times_folded_to_single_bet += 1
                    return "fold", 0
            else:
                # No one has bet - nits rarely bet unless very strong
                if hand_strength > 0.8:
                    # Strong hands - bet for value
                    self.total_bets += 1
                    amount = self._get_nit_bet_sizing("raise", pot_size, our_stack, valid_actions)
                    return "raise", amount
                elif hand_strength > 0.6:
                    # Medium hands - check
                    amount = self._get_nit_bet_sizing("call", pot_size, our_stack, valid_actions)
                    return "call", amount
                else:
                    # Weak hands - check and hope to see cheap card
                    amount = self._get_nit_bet_sizing("call", pot_size, our_stack, valid_actions)
                    return "call", amount
                    
        except Exception as e:
            self.logger.error(f"Error in nit postflop decision: {e}")
            return self._safe_fold(valid_actions)
    
    def _estimate_postflop_strength(self, hole_cards: List[str], community_cards: List[str]) -> float:
        """Estimate hand strength postflop - nits are very conservative in evaluation."""
        try:
            if not hole_cards or len(hole_cards) != 2:
                return 0.0
                
            all_cards = hole_cards + (community_cards or [])
            ranks = [card[0] for card in all_cards]
            rank_counts = {}
            for rank in ranks:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            
            max_count = max(rank_counts.values()) if rank_counts else 1
            
            if max_count >= 4:
                return 0.98  # Four of a kind - nuts
            elif max_count >= 3:
                # Check for full house
                pair_count = sum(1 for count in rank_counts.values() if count >= 2)
                if pair_count >= 2:
                    return 0.95  # Full house
                else:
                    return 0.85  # Three of a kind
            elif max_count >= 2:
                # Pairs
                pair_count = sum(1 for count in rank_counts.values() if count >= 2)
                if pair_count >= 2:
                    return 0.75  # Two pair - good for nit
                else:
                    # One pair - evaluate conservatively
                    pair_rank = [rank for rank, count in rank_counts.items() if count >= 2][0]
                    rank_values = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, 
                                  "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
                    pair_value = rank_values.get(pair_rank, 7)
                    
                    # Nits are conservative about pair strength
                    if pair_value >= 11:  # JJ+
                        return 0.65
                    elif pair_value >= 9:  # 99, TT
                        return 0.55
                    else:
                        return 0.35  # Lower pairs
            else:
                # High card - nits don't value this much
                hole_ranks = [card[0] for card in hole_cards]
                rank_values = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, 
                              "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
                
                hole_values = [rank_values.get(rank, 2) for rank in hole_ranks]
                max_hole = max(hole_values)
                
                if max_hole >= 14:
                    return 0.25  # Ace high - still not great for nit
                elif max_hole >= 13:
                    return 0.20  # King high
                else:
                    return 0.10  # Everything else is weak
                    
        except Exception as e:
            self.logger.error(f"Error estimating hand strength: {e}")
            return 0.1
    
    def _safe_fold(self, valid_actions: List[Dict[str, Any]]) -> Tuple[str, int]:
        """Return fold action - nits default to folding."""
        return "fold", 0
    
    def get_statistics(self) -> Dict[str, float]:
        """Get player statistics for analysis."""
        if self.hands_played == 0:
            return {}
            
        vpip = (self.vpip_hands / self.hands_played) * 100
        pfr = (self.pfr_hands / self.hands_played) * 100
        three_bet_pct = (self.three_bet_hands / max(self.hands_played, 1)) * 100
        
        total_aggressive_actions = self.total_bets + self.total_calls
        aggression_factor = (self.total_bets / max(self.total_calls, 1)) if total_aggressive_actions > 0 else 0
        
        fold_to_aggression_pct = (self.folds_to_aggression / max(self.aggression_faced, 1)) * 100
        
        return {
            "hands_played": self.hands_played,
            "vpip": vpip,  # Should be ~8-12% for Nit
            "pfr": pfr,    # Should be ~6-10% for Nit
            "three_bet_pct": three_bet_pct,  # Should be very low for Nit
            "aggression_factor": aggression_factor,  # Should be ~1.5-2.5 for Nit
            "fold_to_aggression_pct": fold_to_aggression_pct,  # Should be high for Nit
            "premium_hands_played": self.premium_hands_played,
            "times_folded_to_single_bet": self.times_folded_to_single_bet,
            "tournament_stage": self.tournament_stage,
            "stack_category": self.stack_size_category
        }
    
    # PyPokerEngine compatibility methods
    def receive_game_start_message(self, game_info: Dict[str, Any]):
        """Called when game starts."""
        self.logger.info(f"Nit player {self.name} starting new game")
        
    def receive_round_start_message(self, round_count: int, hole_card: List[str], seats: List[Dict[str, Any]]):
        """Called when round starts."""
        self.logger.debug(f"Nit player {self.name} round {round_count} started")
        
    def receive_street_start_message(self, street: str, round_state: Dict[str, Any]):
        """Called when street starts."""
        pass
        
    def receive_game_update_message(self, action: Dict[str, Any], round_state: Dict[str, Any]):
        """Called when any action happens."""
        pass
        
    def receive_round_result_message(self, winners: List[Dict[str, Any]], 
                                   hand_info: List[Dict[str, Any]], round_state: Dict[str, Any]):
        """Called when round ends."""
        pass