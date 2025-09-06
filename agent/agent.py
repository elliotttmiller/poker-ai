"""
Main PokerMind Agent Implementation.

This module contains the PokerMindAgent class, which inherits from 
PyPokerEngine's BasePokerPlayer and integrates the Unified Cognitive Core.
"""

import logging
import time
from typing import Dict, Any, List

from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

from .cognitive_core import CognitiveCore


class PokerMindAgent(BasePokerPlayer):
    """
    The main PokerMind AI agent.
    
    Inherits from PyPokerEngine's BasePokerPlayer and implements the
    dual-process cognitive architecture (System 1 & System 2).
    """

    def __init__(self):
        """Initialize the PokerMind agent."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Initialize the Unified Cognitive Core
        self.cognitive_core = CognitiveCore()
        
        # Performance tracking
        self.decision_times = []
        
        self.logger.info("PokerMind agent initialized successfully")

    def declare_action(
        self, 
        valid_actions: List[Dict[str, Any]], 
        hole_card: List[str], 
        round_state: Dict[str, Any]
    ) -> tuple:
        """
        Main decision-making method called by PyPokerEngine.
        
        This is where the Unified Cognitive Core processes the game state
        and returns the final action.
        
        Args:
            valid_actions: List of valid actions (fold, call, raise)
            hole_card: Player's hole cards
            round_state: Current game state
            
        Returns:
            Tuple of (action_type, amount)
        """
        start_time = time.time()
        
        try:
            # Convert PyPokerEngine state to internal representation
            game_state = self._parse_game_state(
                valid_actions, hole_card, round_state
            )
            
            # Process through the Unified Cognitive Core
            action, decision_packet = self.cognitive_core.make_decision(game_state)
            
            # Track performance
            decision_time = time.time() - start_time
            self.decision_times.append(decision_time)
            
            self.logger.info(
                f"Decision made in {decision_time:.3f}s: {action['action']}"
            )
            
            return action['action'], action.get('amount', 0)
            
        except Exception as e:
            self.logger.error(f"Error in declare_action: {e}")
            # Fallback to safe action (fold or call)
            return self._get_fallback_action(valid_actions)

    def receive_game_start_message(self, game_info: Dict[str, Any]):
        """Called when a new game starts."""
        self.logger.info("New game started")
        self.cognitive_core.reset_for_new_game()

    def receive_round_start_message(
        self, 
        round_count: int, 
        hole_card: List[str], 
        seats: List[Dict[str, Any]]
    ):
        """Called when a new round (hand) starts."""
        self.logger.info(f"Round {round_count} started with cards: {hole_card}")
        self.cognitive_core.reset_for_new_round(round_count, hole_card, seats)

    def receive_street_start_message(
        self, 
        street: str, 
        round_state: Dict[str, Any]
    ):
        """Called when a new street starts (preflop, flop, turn, river)."""
        self.logger.debug(f"Street started: {street}")
        self.cognitive_core.update_street(street, round_state)

    def receive_game_update_message(
        self, 
        action: Dict[str, Any], 
        round_state: Dict[str, Any]
    ):
        """Called when any player makes an action."""
        self.cognitive_core.process_opponent_action(action, round_state)

    def receive_round_result_message(
        self, 
        winners: List[Dict[str, Any]], 
        hand_info: List[Dict[str, Any]], 
        round_state: Dict[str, Any]
    ):
        """Called when a round ends."""
        self.logger.info(f"Round ended. Winners: {[w['name'] for w in winners]}")
        self.cognitive_core.process_round_result(winners, hand_info, round_state)

    def _parse_game_state(
        self, 
        valid_actions: List[Dict[str, Any]], 
        hole_card: List[str], 
        round_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert PyPokerEngine state to internal game state representation.
        
        Args:
            valid_actions: Valid actions from PyPokerEngine
            hole_card: Player's hole cards
            round_state: Game state from PyPokerEngine
            
        Returns:
            Standardized game state dict
        """
        # Extract key information from round_state
        pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)
        street = round_state.get('street', 'preflop')
        community_cards = round_state.get('community_card', [])
        
        # Get our seat information
        our_uuid = self.uuid
        our_seat = None
        for seat in round_state.get('seats', []):
            if seat.get('uuid') == our_uuid:
                our_seat = seat
                break
        
        if not our_seat:
            raise ValueError("Could not find our seat in the game state")
        
        game_state = {
            'hole_cards': hole_card,
            'community_cards': community_cards,
            'pot_size': pot_size,
            'street': street,
            'valid_actions': valid_actions,
            'our_stack': our_seat.get('stack', 0),
            'our_seat_id': our_seat.get('seat_id'),
            'seats': round_state.get('seats', []),
            'round_count': round_state.get('round_count', 0),
            'small_blind': round_state.get('small_blind_amount', 10),
            'action_histories': round_state.get('action_histories', {}),
        }
        
        return game_state

    def _get_fallback_action(self, valid_actions: List[Dict[str, Any]]) -> tuple:
        """
        Get a safe fallback action in case of errors.
        
        Args:
            valid_actions: List of valid actions
            
        Returns:
            Tuple of (action_type, amount) - defaults to fold or call
        """
        self.logger.warning("Using fallback action due to error")
        
        # Try to call if possible, otherwise fold
        for action in valid_actions:
            if action['action'] == 'call':
                return 'call', action.get('amount', 0)
        
        # If can't call, fold
        return 'fold', 0

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for this agent."""
        if not self.decision_times:
            return {}
        
        return {
            'avg_decision_time': sum(self.decision_times) / len(self.decision_times),
            'max_decision_time': max(self.decision_times),
            'min_decision_time': min(self.decision_times),
            'total_decisions': len(self.decision_times),
        }