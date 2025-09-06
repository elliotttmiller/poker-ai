"""
Heuristics Engine Module for Project PokerMind.

This module provides fast, rule-based decisions for trivial situations
where complex analysis is unnecessary.
"""

import logging
from typing import Dict, Any, Optional, List


class HeuristicsEngine:
    """
    Fast heuristics engine for obvious poker decisions.
    
    Identifies and handles trivial situations with high-confidence
    rule-based logic, allowing the Synthesizer to focus on complex decisions.
    """

    def __init__(self):
        """Initialize the heuristics engine."""
        self.logger = logging.getLogger(__name__)
        
        # Configuration thresholds
        self.auto_fold_threshold = 0.05  # Fold if win probability < 5%
        self.auto_call_threshold = 0.95  # Call if win probability > 95%
        self.stack_preservation_threshold = 0.1  # Preserve last 10% of stack
        
        self.logger.info("Heuristics Engine initialized")

    def check_trivial_decisions(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for trivial decisions that don't require complex analysis.
        
        Args:
            game_state: Current game state
            
        Returns:
            Dict containing heuristic recommendation or None if no trivial decision
        """
        try:
            # Check for obvious decisions in order of priority
            
            # 1. Stack preservation
            stack_decision = self._check_stack_preservation(game_state)
            if stack_decision:
                return stack_decision
            
            # 2. Nuts or near-nuts
            nuts_decision = self._check_nuts_situations(game_state)
            if nuts_decision:
                return nuts_decision
            
            # 3. Obvious bluff spots
            bluff_decision = self._check_obvious_bluff_spots(game_state)
            if bluff_decision:
                return bluff_decision
            
            # 4. Trivial preflop decisions
            preflop_decision = self._check_preflop_trivial(game_state)
            if preflop_decision:
                return preflop_decision
            
            # 5. Pot odds slam dunks
            pot_odds_decision = self._check_pot_odds_trivial(game_state)
            if pot_odds_decision:
                return pot_odds_decision
            
            # No trivial decision found
            return {
                'recommendation': None,
                'confidence': 0.0,
                'reasoning': 'No trivial decision detected',
                'source': 'heuristics'
            }
            
        except Exception as e:
            self.logger.warning(f"Heuristics engine error: {e}")
            return {'recommendation': None, 'confidence': 0.0}

    def _check_stack_preservation(self, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if we should preserve our remaining stack."""
        our_stack = game_state.get('our_stack', 1000)
        pot_size = game_state.get('pot_size', 0)
        
        # Get the cost to continue
        call_cost = 0
        valid_actions = game_state.get('valid_actions', [])
        
        for action in valid_actions:
            if action['action'] == 'call':
                call_cost = action.get('amount', 0)
                break
        
        # If call cost is significant portion of our stack and we're short
        if our_stack < 200 and call_cost > our_stack * 0.3:  # Short stack protection
            # Only continue with very strong hands
            hole_cards = game_state.get('hole_cards', [])
            if not self._is_premium_hand(hole_cards):
                return {
                    'recommendation': {'action': 'fold', 'amount': 0},
                    'confidence': 0.8,
                    'reasoning': 'Stack preservation - folding marginal hand when short',
                    'source': 'heuristics_stack_preservation'
                }
        
        return None

    def _check_nuts_situations(self, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for nuts or near-nuts situations."""
        hole_cards = game_state.get('hole_cards', [])
        community_cards = game_state.get('community_cards', [])
        
        if len(community_cards) >= 3:  # Post-flop
            hand_strength = self._estimate_hand_strength(hole_cards, community_cards)
            
            # Very strong hands - bet/raise aggressively
            if hand_strength > 0.95:
                pot_size = game_state.get('pot_size', 0)
                our_stack = game_state.get('our_stack', 1000)
                
                # Calculate aggressive bet size
                bet_size = min(int(pot_size * 0.8), our_stack)
                
                return {
                    'recommendation': {'action': 'raise', 'amount': bet_size},
                    'confidence': 0.95,
                    'reasoning': 'Near-nuts hand - betting for value',
                    'source': 'heuristics_nuts'
                }
        
        return None

    def _check_obvious_bluff_spots(self, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for obvious bluffing opportunities."""
        street = game_state.get('street', 'preflop')
        community_cards = game_state.get('community_cards', [])
        
        # Dry board bluffing opportunities
        if street == 'flop' and len(community_cards) == 3:
            if self._is_dry_board(community_cards):
                pot_size = game_state.get('pot_size', 0)
                our_stack = game_state.get('our_stack', 1000)
                
                # Check if we were the preflop aggressor
                # TODO: Track preflop aggressor status
                
                # Small continuation bet on dry board
                bet_size = min(int(pot_size * 0.5), our_stack)
                
                return {
                    'recommendation': {'action': 'bet', 'amount': bet_size},
                    'confidence': 0.7,
                    'reasoning': 'Continuation bet on dry board',
                    'source': 'heuristics_cbet'
                }
        
        return None

    def _check_preflop_trivial(self, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for trivial preflop decisions."""
        if game_state.get('street') != 'preflop':
            return None
        
        hole_cards = game_state.get('hole_cards', [])
        if len(hole_cards) != 2:
            return None
        
        # Premium hands - always play aggressively
        if self._is_premium_hand(hole_cards):
            pot_size = game_state.get('pot_size', 0)
            our_stack = game_state.get('our_stack', 1000)
            
            # Raise with premium hands
            raise_size = max(int(pot_size * 3), 30)  # 3x pot or 3BB minimum
            raise_size = min(raise_size, our_stack)
            
            return {
                'recommendation': {'action': 'raise', 'amount': raise_size},
                'confidence': 0.9,
                'reasoning': f'Premium hand: {hole_cards}',
                'source': 'heuristics_premium'
            }
        
        # Trash hands - fold to aggression
        if self._is_trash_hand(hole_cards):
            # Check if there's been aggression
            action_histories = game_state.get('action_histories', {})
            preflop_actions = action_histories.get('preflop', [])
            
            has_aggression = any(
                action.get('action') in ['raise', 'bet'] 
                for action in preflop_actions
            )
            
            if has_aggression:
                return {
                    'recommendation': {'action': 'fold', 'amount': 0},
                    'confidence': 0.85,
                    'reasoning': f'Trash hand facing aggression: {hole_cards}',
                    'source': 'heuristics_trash'
                }
        
        return None

    def _check_pot_odds_trivial(self, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for obvious pot odds situations."""
        valid_actions = game_state.get('valid_actions', [])
        pot_size = game_state.get('pot_size', 0)
        
        # Find call cost
        call_cost = 0
        for action in valid_actions:
            if action['action'] == 'call':
                call_cost = action.get('amount', 0)
                break
        
        if call_cost == 0:
            return None  # No cost to call
        
        # Calculate pot odds
        pot_odds = call_cost / (pot_size + call_cost) if (pot_size + call_cost) > 0 else 1.0
        
        # Estimate our equity (simplified)
        hole_cards = game_state.get('hole_cards', [])
        community_cards = game_state.get('community_cards', [])
        
        if not hole_cards:
            return None
        
        equity = self._estimate_equity(hole_cards, community_cards)
        
        # Clear mathematical decisions
        if equity > pot_odds + 0.1:  # 10% buffer for clear calls
            return {
                'recommendation': {'action': 'call', 'amount': call_cost},
                'confidence': min(0.9, (equity - pot_odds) * 5),
                'reasoning': f'Clear pot odds call: {equity:.2f} equity vs {pot_odds:.2f} pot odds',
                'source': 'heuristics_pot_odds'
            }
        elif equity < pot_odds - 0.1:  # 10% buffer for clear folds
            return {
                'recommendation': {'action': 'fold', 'amount': 0},
                'confidence': min(0.9, (pot_odds - equity) * 5),
                'reasoning': f'Clear pot odds fold: {equity:.2f} equity vs {pot_odds:.2f} pot odds',
                'source': 'heuristics_pot_odds'
            }
        
        return None

    # Helper methods
    def _is_premium_hand(self, hole_cards: List[str]) -> bool:
        """Check if hole cards represent a premium hand."""
        if len(hole_cards) != 2:
            return False
        
        # Parse cards
        ranks = [card[0] for card in hole_cards]
        suits = [card[1] for card in hole_cards]
        
        # Convert face cards to numbers for comparison
        rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
        for i, rank in enumerate(ranks):
            if rank in rank_values:
                ranks[i] = rank_values[rank]
            else:
                ranks[i] = int(rank)
        
        # Premium hands: AA, KK, QQ, JJ, AK
        if ranks[0] == ranks[1] and ranks[0] >= 11:  # Pocket pairs JJ+
            return True
        if set(ranks) == {14, 13}:  # AK
            return True
        
        return False

    def _is_trash_hand(self, hole_cards: List[str]) -> bool:
        """Check if hole cards represent a trash hand."""
        if len(hole_cards) != 2:
            return False
        
        # Parse cards
        ranks = [card[0] for card in hole_cards]
        suits = [card[1] for card in hole_cards]
        
        # Convert face cards to numbers
        rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
        for i, rank in enumerate(ranks):
            if rank in rank_values:
                ranks[i] = rank_values[rank]
            else:
                ranks[i] = int(rank)
        
        # Trash hands: low unsuited cards, big gaps
        max_rank = max(ranks)
        min_rank = min(ranks)
        is_suited = suits[0] == suits[1]
        gap = abs(ranks[0] - ranks[1])
        
        # Very low cards
        if max_rank <= 7 and not is_suited:
            return True
        
        # Big gap with low cards
        if gap >= 4 and max_rank <= 9 and not is_suited:
            return True
        
        return False

    def _is_dry_board(self, community_cards: List[str]) -> bool:
        """Check if the board is dry (few draws available)."""
        if len(community_cards) < 3:
            return False
        
        # Parse ranks and suits
        ranks = [card[0] for card in community_cards]
        suits = [card[1] for card in community_cards]
        
        # Check for flush draws
        suit_counts = {}
        for suit in suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        max_suit_count = max(suit_counts.values())
        if max_suit_count >= 3:  # Flush draw possible
            return False
        
        # Check for straight draws
        rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
        numeric_ranks = []
        for rank in ranks:
            if rank in rank_values:
                numeric_ranks.append(rank_values[rank])
            else:
                numeric_ranks.append(int(rank))
        
        numeric_ranks.sort()
        
        # Check for potential straights
        for i in range(len(numeric_ranks) - 1):
            if numeric_ranks[i+1] - numeric_ranks[i] <= 2:  # Close ranks = straight draws
                return False
        
        return True  # Board is dry

    def _estimate_hand_strength(self, hole_cards: List[str], community_cards: List[str]) -> float:
        """Rough estimation of hand strength (0.0 to 1.0)."""
        # TODO: Implement proper hand evaluation using deuces library
        # For now, return a placeholder value
        
        if not hole_cards or not community_cards:
            return 0.3  # Weak by default
        
        # Very simplified: just check for pairs
        all_cards = hole_cards + community_cards
        ranks = [card[0] for card in all_cards]
        
        # Count rank occurrences
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        max_count = max(rank_counts.values())
        
        if max_count >= 4:  # Quads
            return 0.98
        elif max_count >= 3:  # Trips
            return 0.80
        elif max_count >= 2:  # Pair
            return 0.60
        else:
            return 0.35  # High card

    def _estimate_equity(self, hole_cards: List[str], community_cards: List[str]) -> float:
        """Estimate our equity in the hand."""
        # TODO: Use proper Monte Carlo simulation or lookup table
        # For now, use simplified hand strength estimation
        
        if not community_cards:  # Preflop
            if self._is_premium_hand(hole_cards):
                return 0.85
            elif self._is_trash_hand(hole_cards):
                return 0.15
            else:
                return 0.45
        else:  # Postflop
            return self._estimate_hand_strength(hole_cards, community_cards)

    def get_heuristic_adjustments(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommended heuristic adjustments based on game state."""
        adjustments = {}
        
        # Stack size adjustments
        our_stack = game_state.get('our_stack', 1000)
        if our_stack < 300:  # Short stack
            adjustments['play_style'] = 'tight_aggressive'
            adjustments['push_fold_mode'] = True
        elif our_stack > 2000:  # Deep stack
            adjustments['play_style'] = 'loose_aggressive'
            adjustments['implied_odds_focus'] = True
        
        # Table position adjustments
        # TODO: Implement position-based heuristics
        
        return adjustments