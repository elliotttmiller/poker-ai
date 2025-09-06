"""
Synthesizer Module for Project PokerMind.

This module implements System 2 (deliberate analysis) by synthesizing
inputs from all System 1 modules to make final decisions.
"""

import logging
import math
from typing import Dict, Any, Tuple, List, Optional
import random

from ..utils import calculate_pot_odds


class Synthesizer:
    """
    The Synthesizer implements System 2 of the dual-process architecture.
    
    Enhanced in Phase 5 to use confidence-weighted blending instead of simple rules.
    Takes inputs from all System 1 modules and synthesizes them using confidence scores.
    """

    def __init__(self):
        """Initialize the Synthesizer with enhanced Phase 5 capabilities."""
        self.logger = logging.getLogger(__name__)
        
        # Phase 5: Confidence-based weighting parameters
        self.min_confidence_threshold = 0.3  # Ignore recommendations below this
        self.high_confidence_threshold = 0.8  # Trust recommendations above this
        
        # Default module weights (can be dynamically adjusted)
        self.module_weights = {
            'gto': 0.4,         # GTO gets high base weight
            'heuristics': 0.3,  # Heuristics for obvious situations
            'hand_strength': 0.2, # Hand strength for equity calculations
            'opponents': 0.1    # Opponent modeling for exploits
        }
        
        # Player style parameters (dynamic)
        self.tightness = 0.5  # 0.0 = very loose, 1.0 = very tight
        self.aggression = 0.5  # 0.0 = very passive, 1.0 = very aggressive
        
        # Risk management
        self.risk_tolerance = 0.5
        
        self.logger.info("Enhanced Synthesizer (Phase 5) initialized")

    def synthesize_decision(
        self, 
        game_state: Dict[str, Any], 
        system1_outputs: Dict[str, Any],
        opponent_profile: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Enhanced Phase 5 synthesizer using confidence-weighted blending.
        
        This is the core method of the Synthesizer that implements the confidence-based
        decision synthesis introduced in Phase 5. Instead of simple if/then rules,
        it uses mathematical confidence scoring to weight and blend recommendations
        from all System 1 modules.
        
        The process follows these steps:
        1. Check for high-confidence heuristic overrides (emergency situations)
        2. Extract confidence scores from all System 1 modules
        3. Perform confidence-weighted blending using voting system
        4. Apply opponent-specific adjustments (if confident about opponent model)
        5. Apply final meta-cognitive adjustments for style and risk management
        6. Generate comprehensive analysis with confidence breakdown
        
        Args:
            game_state: Current game state containing hole cards, community cards,
                       pot size, valid actions, and other game context
            system1_outputs: Dict containing outputs from all System 1 modules:
                           - 'gto': GTO Core recommendation with confidence
                           - 'hand_strength': Hand strength analysis with confidence
                           - 'heuristics': Heuristic recommendations with confidence
                           - 'opponents': Opponent analysis with confidence
            opponent_profile: Optional primary opponent profile for targeted
                            adjustments (used when opponent confidence is high)
            
        Returns:
            Tuple containing:
            - final_action: Dict with 'action', 'amount', 'confidence' keys
            - analysis: Comprehensive analysis dict with confidence breakdown,
                       module contributions, reasoning, and decision path
            
        Raises:
            Exception: Catches all exceptions and returns safe fallback action
            
        Example:
            >>> system1_outputs = {
            ...     'gto': {'action': 'call', 'amount': 50, 'confidence': 0.8},
            ...     'hand_strength': {'overall_strength': 0.7, 'confidence': 0.9},
            ...     'heuristics': {'recommendation': None, 'confidence': 0.0},
            ...     'opponents': {'exploit_opportunities': [], 'confidence': 0.6}
            ... }
            >>> action, analysis = synthesizer.synthesize_decision(game_state, system1_outputs)
            >>> print(f"Action: {action['action']}, Confidence: {action['confidence']:.2f}")
            Action: call, Confidence: 0.85
        """
        try:
            # Phase 1: Check for high-confidence heuristic overrides (unchanged)
            heuristic_decision = self._check_heuristic_override(
                system1_outputs, game_state
            )
            if heuristic_decision:
                return heuristic_decision
            
            # Phase 2: Extract confidence scores from all modules
            confidence_scores = self._extract_confidence_scores(system1_outputs)
            
            # Phase 3: Phase 5 Enhancement - Confidence-weighted recommendation blending
            blended_recommendation = self._blend_recommendations_by_confidence(
                system1_outputs, confidence_scores, game_state
            )
            
            # Phase 4: Apply opponent-specific adjustments using confidence
            adjusted_recommendation = self._apply_confident_opponent_adjustments(
                blended_recommendation, opponent_profile, system1_outputs, confidence_scores
            )
            
            # Phase 5: Final validation and meta-adjustments
            final_action = self._apply_meta_adjustments(
                adjusted_recommendation, game_state, system1_outputs
            )
            
            # Phase 6: Generate comprehensive analysis with confidence breakdown
            analysis = self._generate_confidence_analysis(
                game_state, system1_outputs, confidence_scores, 
                blended_recommendation, final_action
            )
            
            return final_action, analysis
            
        except Exception as e:
            self.logger.error(f"Enhanced Synthesizer error: {e}")
            fallback_action = {'action': 'fold', 'amount': 0}
            fallback_analysis = {
                'reasoning': f'Enhanced Synthesizer error: {e}',
                'confidence': 0.1,
                'source': 'synthesizer_error'
            }
            return fallback_action, fallback_analysis

    def make_final_decision(
        self,
        game_state: Dict[str, Any],
        system1_outputs: Dict[str, Any],
        opponent_profile: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Make final decision with opponent profile integration (as requested in directive).
        
        This is the method requested in Sub-Task 3.2 that accepts opponent_profile.
        """
        return self.synthesize_decision(game_state, system1_outputs, opponent_profile)

    def synthesize_decision(
        self, 
        game_state: Dict[str, Any], 
        system1_outputs: Dict[str, Any],
        opponent_profile: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Synthesize System 1 outputs into a final decision.
        
        Args:
            game_state: Current game state
            system1_outputs: Outputs from all System 1 modules
            
        Returns:
            Tuple of (final_action, analysis_dict)
        """
        try:
            # Phase 1: Check for high-confidence heuristic overrides
            heuristic_decision = self._check_heuristic_override(
                system1_outputs, game_state
            )
            if heuristic_decision:
                return heuristic_decision
            
            # Phase 2: Calculate equity from hand strength estimator
            hand_strength_output = system1_outputs.get('hand_strength', {})
            our_equity = self._calculate_equity_from_hand_strength(hand_strength_output)
            
            # Phase 3: Calculate pot odds and required equity
            equity_analysis = self._analyze_equity_and_odds(game_state)
            required_equity = equity_analysis.get('required_equity', 0.5)
            
            # Phase 4: Apply exploitative adjustments to required equity (Sub-Task 3.2)
            adjusted_required_equity = self._apply_opponent_adjustments(
                required_equity, opponent_profile, system1_outputs
            )
            
            # Phase 4: Make core decision based on equity vs adjusted pot odds
            core_decision = self._make_equity_based_decision(
                our_equity, adjusted_required_equity, game_state, equity_analysis
            )
            
            # Phase 5: Apply GTO and exploitative adjustments including loose player value betting
            adjusted_decision = self._apply_strategic_adjustments(
                core_decision, system1_outputs, game_state, opponent_profile
            )
            
            # Phase 6: Apply meta-cognitive adjustments
            final_action = self._apply_meta_adjustments(
                adjusted_decision, game_state, system1_outputs
            )
            
            # Phase 7: Generate comprehensive analysis
            analysis = self._generate_analysis(
                game_state, system1_outputs, our_equity, required_equity,
                equity_analysis, core_decision, final_action
            )
            
            return final_action, analysis
            
        except Exception as e:
            self.logger.error(f"Synthesizer error: {e}")
            fallback_action = {'action': 'fold', 'amount': 0}
            fallback_analysis = {
                'reasoning': f'Synthesizer error: {e}',
                'confidence': 0.1,
                'source': 'synthesizer_error'
            }
            return fallback_action, fallback_analysis

    def _apply_opponent_adjustments(
        self, 
        required_equity: float, 
        opponent_profile: Dict[str, Any],
        system1_outputs: Dict[str, Any]
    ) -> float:
        """
        Apply exploitative adjustments to required equity based on opponent tendencies.
        
        This implements the specific logic requested in Sub-Task 3.2:
        - vs. Tight Player: required_equity = pot_odds_equity * 1.15
        - vs. Loose Player: adjust for value betting opportunities
        """
        # Start with base required equity
        adjusted_equity = required_equity
        
        # Get primary opponent profile from direct input or system1 outputs
        primary_opponent = opponent_profile
        if not primary_opponent:
            # Try to get from opponent analysis
            opponent_output = system1_outputs.get('opponents', {})
            opponents = opponent_output.get('opponents', {})
            if opponents:
                # Get the first opponent profile for now
                primary_opponent = list(opponents.values())[0]
        
        if not primary_opponent:
            return adjusted_equity
        
        classification = primary_opponent.get('classification', 'unknown')
        vpip = primary_opponent.get('vpip', 0.25)
        pfr = primary_opponent.get('pfr', 0.15)
        
        # Example 1: vs. Tight Player - be more cautious (require better odds)
        if 'tight' in classification.lower() or vpip < 0.2:
            adjusted_equity = required_equity * 1.15  # Exact formula from directive
            self.logger.debug(f"Tight opponent adjustment: {required_equity:.3f} -> {adjusted_equity:.3f}")
        
        # Example 2: vs. Loose Player - can call with worse odds
        elif 'loose' in classification.lower() or vpip > 0.4:
            adjusted_equity = required_equity * 0.9  # Slightly better odds against loose players
            self.logger.debug(f"Loose opponent adjustment: {required_equity:.3f} -> {adjusted_equity:.3f}")
        
        # vs. Passive Player - can bluff more (need less equity)
        if 'passive' in classification.lower() or pfr < 0.1:
            adjusted_equity *= 0.95  # Slightly less equity needed vs passive
            self.logger.debug(f"Passive opponent adjustment applied")
        
        # vs. Aggressive Player - need more equity to call
        elif 'aggressive' in classification.lower() or pfr > 0.25:
            adjusted_equity *= 1.1  # More equity needed vs aggressive
            self.logger.debug(f"Aggressive opponent adjustment applied")
        
        # Ensure reasonable bounds
        return max(0.1, min(0.9, adjusted_equity))

    def _check_heuristic_override(
        self, 
        system1_outputs: Dict[str, Any], 
        game_state: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Check if heuristics provide a high-confidence override."""
        heuristics_output = system1_outputs.get('heuristics', {})
        recommendation = heuristics_output.get('recommendation')
        confidence = heuristics_output.get('confidence', 0.0)
        
        if recommendation and confidence >= 0.8:
            self.logger.info(f"Heuristic override: {recommendation} (confidence: {confidence})")
            
            analysis = {
                'reasoning': heuristics_output.get('reasoning', 'Heuristic override'),
                'confidence': confidence,
                'source': 'heuristics_override',
                'meta_adjustments': 'none_applied'
            }
            
            return recommendation, analysis
        
        return None

    def _analyze_equity_and_odds(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pot odds and estimate our equity."""
        pot_size = game_state.get('pot_size', 0)
        our_stack = game_state.get('our_stack', 1000)
        valid_actions = game_state.get('valid_actions', [])
        
        # Find call cost
        call_cost = 0
        min_raise = 0
        max_raise = our_stack
        
        for action in valid_actions:
            if action['action'] == 'call':
                call_cost = action.get('amount', 0)
            elif action['action'] == 'raise':
                amount_info = action.get('amount', {})
                if isinstance(amount_info, dict):
                    min_raise = amount_info.get('min', 0)
                    max_raise = amount_info.get('max', our_stack)
                else:
                    min_raise = amount_info
        
        # Calculate pot odds using the utility function
        required_equity = calculate_pot_odds(pot_size, call_cost)
        
        return {
            'pot_size': pot_size,
            'call_cost': call_cost,
            'required_equity': required_equity,
            'min_raise': min_raise,
            'max_raise': max_raise,
        }

    def _calculate_equity_from_hand_strength(self, hand_strength_output: Dict[str, Any]) -> float:
        """
        Convert hand strength probabilities to equity (win probability).
        
        Args:
            hand_strength_output: Output from HandStrengthEstimator
            
        Returns:
            float: Estimated equity (win probability)
        """
        if not hand_strength_output:
            return 0.3  # Default low equity
        
        # Get the overall strength from hand strength estimator
        overall_strength = hand_strength_output.get('overall_strength', 0.3)
        probabilities = hand_strength_output.get('probabilities', [])
        
        if probabilities and len(probabilities) >= 9:
            # Convert hand strength categories to approximate win probabilities
            # Two Pair or better has good equity
            two_pair_plus_prob = sum(probabilities[2:])  # Two Pair through Straight Flush
            
            # Calculate weighted equity
            base_equity = overall_strength * 0.6  # Base from overall strength
            strong_hand_bonus = two_pair_plus_prob * 0.3  # Bonus for strong hands
            
            equity = base_equity + strong_hand_bonus
        else:
            # Fallback to overall strength
            equity = overall_strength
        
        # Ensure reasonable bounds
        return max(0.05, min(0.95, equity))

    def _make_equity_based_decision(
        self, 
        our_equity: float, 
        required_equity: float, 
        game_state: Dict[str, Any], 
        equity_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make the core decision based on equity vs pot odds.
        
        Args:
            our_equity: Our estimated win probability
            required_equity: Required equity from pot odds
            game_state: Current game state
            equity_analysis: Pot odds analysis
            
        Returns:
            Dict containing the core decision
        """
        call_cost = equity_analysis.get('call_cost', 0)
        pot_size = equity_analysis.get('pot_size', 0)
        valid_actions = game_state.get('valid_actions', [])
        
        # Basic equity-based decision
        if our_equity > required_equity:
            # We have profitable equity - decide between call and raise
            if our_equity > required_equity * 1.5:  # Strong equity advantage
                # Consider raising for value
                min_raise = equity_analysis.get('min_raise', 0)
                if min_raise > 0 and self._has_raise_action(valid_actions):
                    bet_size = self._calculate_value_bet_size(pot_size, our_equity)
                    return {
                        'action': 'raise',
                        'amount': max(min_raise, bet_size),
                        'reasoning': f'Value betting with {our_equity:.2f} equity vs {required_equity:.2f} required',
                        'confidence': min(0.8, our_equity),
                        'equity': our_equity,
                        'required_equity': required_equity
                    }
            
            # Call - we have profitable equity but not strong enough to raise
            return {
                'action': 'call',
                'amount': call_cost,
                'reasoning': f'Calling with profitable equity: {our_equity:.2f} vs {required_equity:.2f} required',
                'confidence': 0.6 + (our_equity - required_equity),
                'equity': our_equity,
                'required_equity': required_equity
            }
        else:
            # We don't have profitable equity - fold
            return {
                'action': 'fold',
                'amount': 0,
                'reasoning': f'Folding with insufficient equity: {our_equity:.2f} vs {required_equity:.2f} required',
                'confidence': 0.7 + (required_equity - our_equity),
                'equity': our_equity,
                'required_equity': required_equity
            }

    def _apply_strategic_adjustments(
        self, 
        core_decision: Dict[str, Any], 
        system1_outputs: Dict[str, Any], 
        game_state: Dict[str, Any],
        opponent_profile: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Apply GTO and exploitative adjustments to the core decision.
        Enhanced with opponent profile integration from Sub-Task 3.2.
        
        Args:
            core_decision: Core equity-based decision
            system1_outputs: System 1 module outputs
            game_state: Current game state
            opponent_profile: Primary opponent profile for adjustments
            
        Returns:
            Dict containing the adjusted decision
        """
        adjusted_decision = core_decision.copy()
        
        # Apply opponent-specific adjustments as requested in directive
        if opponent_profile:
            adjusted_decision = self._apply_opponent_specific_adjustments(
                adjusted_decision, opponent_profile, game_state, system1_outputs
            )
        
        # Get GTO and opponent analysis
        gto_output = system1_outputs.get('gto', {})
        opponent_output = system1_outputs.get('opponents', {})
        
        # Check for exploitative opportunities
        exploit_opportunities = opponent_output.get('exploit_opportunities', [])
        
        if exploit_opportunities and core_decision['action'] != 'fold':
            # Consider exploitative adjustments
            best_exploit = max(exploit_opportunities, key=lambda x: x.get('confidence', 0))
            exploit_confidence = best_exploit.get('confidence', 0)
            
            if exploit_confidence > 0.7:
                exploit_type = best_exploit.get('type', '')
                
                if exploit_type == 'bluff_opportunity' and core_decision['action'] == 'call':
                    # Convert call to bluff raise
                    pot_size = game_state.get('pot_size', 0)
                    bluff_size = int(pot_size * 0.6)
                    adjusted_decision.update({
                        'action': 'raise',
                        'amount': bluff_size,
                        'reasoning': core_decision['reasoning'] + ' + bluffing opportunity',
                        'confidence': min(adjusted_decision['confidence'], exploit_confidence)
                    })
                    
                elif exploit_type == 'value_bet_opportunity' and core_decision['action'] == 'call':
                    # Convert call to value raise against calling station
                    pot_size = game_state.get('pot_size', 0)
                    value_size = int(pot_size * 0.8)
                    adjusted_decision.update({
                        'action': 'raise',
                        'amount': value_size,
                        'reasoning': core_decision['reasoning'] + ' + value betting vs calling station',
                        'confidence': min(adjusted_decision['confidence'], exploit_confidence)
                    })
        
        # Apply GTO adjustments (balance frequency)
        gto_action = gto_output.get('action', '')
        gto_confidence = gto_output.get('confidence', 0)
        
        if gto_confidence > 0.6 and gto_action != adjusted_decision['action']:
            # Consider mixing strategies based on GTO weight
            if random.random() < self.gto_weight:
                self.logger.debug(f"Applying GTO adjustment: {gto_action} over {adjusted_decision['action']}")
                # For now, keep core decision but log the GTO consideration
                adjusted_decision['reasoning'] += f' (GTO suggests: {gto_action})'
        
        return adjusted_decision

    def _apply_opponent_specific_adjustments(
        self,
        decision: Dict[str, Any],
        opponent_profile: Dict[str, Any],
        game_state: Dict[str, Any],
        system1_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply opponent-specific adjustments as requested in Sub-Task 3.2.
        
        Example 2 from directive: vs. Loose Player AND my_hand_is_strong THEN increase_raise_amount
        """
        classification = opponent_profile.get('classification', 'unknown')
        
        # Check if we have a strong hand
        hand_strength_output = system1_outputs.get('hand_strength', {})
        our_equity = decision.get('equity', 0.5)
        my_hand_is_strong = our_equity > 0.7 or hand_strength_output.get('overall_strength', 0) > 0.7
        
        # Example 2: vs. Loose Player AND strong hand -> increase raise amount
        if 'loose' in classification.lower() and my_hand_is_strong and decision['action'] == 'raise':
            pot_size = game_state.get('pot_size', 0)
            current_amount = decision.get('amount', 0)
            
            # Increase raise amount for value betting against loose players
            increased_amount = min(int(current_amount * 1.3), int(pot_size * 1.0))  # Cap at pot size
            
            decision['amount'] = increased_amount
            decision['reasoning'] += ' + increased vs loose player with strong hand'
            self.logger.debug(f"Loose player value bet adjustment: {current_amount} -> {increased_amount}")
        
        return decision

    def _has_raise_action(self, valid_actions: List[Dict]) -> bool:
        """Check if raising is a valid action."""
        return any(action['action'] == 'raise' for action in valid_actions)

    def _calculate_value_bet_size(self, pot_size: int, equity: float) -> int:
        """Calculate an appropriate value bet size."""
        if equity > 0.8:
            return int(pot_size * 0.8)  # Large bet with very strong hands
        elif equity > 0.65:
            return int(pot_size * 0.6)  # Medium bet with strong hands
        else:
            return int(pot_size * 0.4)  # Small bet with marginal value hands
        """Estimate our equity in the current situation."""
        # TODO: Use more sophisticated equity calculation
        # For now, use simplified estimation based on hand strength
        
        hole_cards = game_state.get('hole_cards', [])
        community_cards = game_state.get('community_cards', [])
        street = game_state.get('street', 'preflop')
        
        if street == 'preflop':
            return self._estimate_preflop_equity(hole_cards)
        else:
            return self._estimate_postflop_equity(hole_cards, community_cards)

    def _estimate_preflop_equity(self, hole_cards: List[str]) -> float:
        """Estimate preflop equity based on hole cards."""
        if len(hole_cards) != 2:
            return 0.3
        
        # Simple preflop equity estimation
        ranks = [card[0] for card in hole_cards]
        suits = [card[1] for card in hole_cards]
        is_suited = suits[0] == suits[1]
        is_pair = ranks[0] == ranks[1]
        
        # Convert face cards
        rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
        numeric_ranks = []
        for rank in ranks:
            if rank in rank_values:
                numeric_ranks.append(rank_values[rank])
            else:
                numeric_ranks.append(int(rank))
        
        max_rank = max(numeric_ranks)
        min_rank = min(numeric_ranks)
        
        # Basic equity calculation
        base_equity = 0.3  # Minimum equity
        
        # Pocket pairs
        if is_pair:
            base_equity = 0.5 + (max_rank - 2) * 0.03  # Higher pairs = better equity
        else:
            # High cards
            base_equity += (max_rank - 7) * 0.025
            base_equity += (min_rank - 2) * 0.015
            
            # Suited bonus
            if is_suited:
                base_equity += 0.05
            
            # Connected cards bonus
            gap = abs(numeric_ranks[0] - numeric_ranks[1])
            if gap <= 1:
                base_equity += 0.03
            elif gap == 2:
                base_equity += 0.01
        
        return min(max(base_equity, 0.1), 0.9)

    def _estimate_postflop_equity(self, hole_cards: List[str], community_cards: List[str]) -> float:
        """Estimate postflop equity."""
        # TODO: Implement proper hand evaluation
        # For now, return simplified estimate
        if len(community_cards) == 3:  # Flop
            return 0.4 + random.uniform(-0.1, 0.3)
        elif len(community_cards) == 4:  # Turn
            return 0.35 + random.uniform(-0.1, 0.4)
        else:  # River
            return 0.5 + random.uniform(-0.2, 0.2)

    def _weight_recommendations(
        self, 
        system1_outputs: Dict[str, Any], 
        equity_analysis: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Weight GTO and exploitative recommendations."""
        gto_output = system1_outputs.get('gto', {})
        opponent_output = system1_outputs.get('opponents', {})
        
        # Get base GTO recommendation
        gto_action = gto_output.get('action', 'fold')
        gto_confidence = gto_output.get('confidence', 0.5)
        
        # Check for exploitative opportunities
        exploit_opportunities = opponent_output.get('exploit_opportunities', [])
        
        # Start with GTO recommendation
        base_action = gto_action
        base_confidence = gto_confidence * self.gto_weight
        
        # Adjust based on exploitative opportunities
        if exploit_opportunities:
            exploitation_adjustment = self._calculate_exploitation_adjustment(
                exploit_opportunities, game_state, equity_analysis
            )
            
            if exploitation_adjustment:
                # Blend GTO and exploitative recommendations
                exploit_action = exploitation_adjustment['action']
                exploit_confidence = exploitation_adjustment['confidence']
                
                if exploit_confidence > gto_confidence * 0.8:  # Significant exploit opportunity
                    base_action = exploit_action
                    base_confidence = (gto_confidence * self.gto_weight + 
                                    exploit_confidence * self.exploit_weight)
        
        # Ensure we have a valid action
        valid_actions = game_state.get('valid_actions', [])
        if not self._is_valid_action(base_action, valid_actions):
            base_action = self._get_fallback_action(valid_actions)
            base_confidence *= 0.5  # Reduce confidence for fallback
        
        return {
            'action': base_action,
            'confidence': min(base_confidence, 1.0),
            'gto_component': gto_confidence * self.gto_weight,
            'exploit_component': base_confidence - gto_confidence * self.gto_weight
        }

    def _calculate_exploitation_adjustment(
        self, 
        exploit_opportunities: List[Dict], 
        game_state: Dict[str, Any],
        equity_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate adjustments based on exploitative opportunities."""
        if not exploit_opportunities:
            return None
        
        # Find the highest confidence exploit
        best_exploit = max(exploit_opportunities, key=lambda x: x.get('confidence', 0))
        exploit_type = best_exploit.get('type')
        confidence = best_exploit.get('confidence', 0)
        
        # Convert exploit type to action recommendation
        if exploit_type == 'bluff_opportunity' and confidence > 0.7:
            # Increase bluffing frequency
            pot_size = game_state.get('pot_size', 0)
            bet_size = int(pot_size * 0.6)
            
            return {
                'action': 'raise',
                'amount': bet_size,
                'confidence': confidence,
                'reasoning': f"Bluffing against {best_exploit.get('player', 'opponent')}"
            }
        
        elif exploit_type == 'value_bet_opportunity' and confidence > 0.6:
            # Increase value betting with marginal hands
            if equity_analysis.get('equity', 0) > 0.6:
                pot_size = game_state.get('pot_size', 0)
                bet_size = int(pot_size * 0.7)
                
                return {
                    'action': 'raise',
                    'amount': bet_size,
                    'confidence': confidence,
                    'reasoning': f"Value betting against calling station"
                }
        
        elif exploit_type == 'steal_opportunity' and confidence > 0.75:
            # Increase steal attempts
            if game_state.get('street') == 'preflop':
                pot_size = game_state.get('pot_size', 0)
                steal_size = max(int(pot_size * 2.5), 30)
                
                return {
                    'action': 'raise',
                    'amount': steal_size,
                    'confidence': confidence,
                    'reasoning': "Stealing against tight opponent"
                }
        
        return None

    # Phase 5 Enhancement Methods - Confidence-Based Blending
    
    def _extract_confidence_scores(self, system1_outputs: Dict[str, Any]) -> Dict[str, float]:
        """Extract confidence scores from all System 1 modules."""
        confidence_scores = {}
        
        # GTO Core confidence
        gto_output = system1_outputs.get('gto', {})
        confidence_scores['gto'] = gto_output.get('confidence', 0.0)
        
        # Hand Strength Estimator confidence
        hand_strength_output = system1_outputs.get('hand_strength', {})
        confidence_scores['hand_strength'] = hand_strength_output.get('confidence', 0.0)
        
        # Heuristics Engine confidence
        heuristics_output = system1_outputs.get('heuristics', {})
        confidence_scores['heuristics'] = heuristics_output.get('confidence', 0.0)
        
        # Opponent Modeler confidence
        opponent_output = system1_outputs.get('opponents', {})
        confidence_scores['opponents'] = opponent_output.get('confidence', 0.0)
        
        return confidence_scores

    def _blend_recommendations_by_confidence(
        self, 
        system1_outputs: Dict[str, Any], 
        confidence_scores: Dict[str, float],
        game_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Phase 5 core method: Blend recommendations using confidence-weighted averaging.
        """
        valid_actions = game_state.get('valid_actions', [])
        
        # Collect actionable recommendations with confidence above threshold
        actionable_recommendations = []
        
        # Process each module's recommendation
        for module_name, base_weight in self.module_weights.items():
            module_output = system1_outputs.get(module_name, {})
            confidence = confidence_scores.get(module_name, 0.0)
            
            if confidence < self.min_confidence_threshold:
                continue  # Skip low-confidence recommendations
            
            # Extract recommendation
            recommendation = self._extract_module_recommendation(module_output, module_name, game_state)
            if recommendation:
                # Calculate effective weight = base_weight * confidence
                effective_weight = base_weight * confidence
                
                actionable_recommendations.append({
                    'module': module_name,
                    'action': recommendation['action'],
                    'amount': recommendation.get('amount', 0),
                    'confidence': confidence,
                    'weight': effective_weight,
                    'reasoning': recommendation.get('reasoning', f'{module_name} recommendation')
                })
        
        # If no actionable recommendations, fall back to conservative play
        if not actionable_recommendations:
            return self._get_conservative_fallback(valid_actions)
        
        # Weighted voting for action selection
        action_votes = {}
        for rec in actionable_recommendations:
            action = rec['action']
            weight = rec['weight']
            
            if action not in action_votes:
                action_votes[action] = {'total_weight': 0, 'recommendations': []}
            
            action_votes[action]['total_weight'] += weight
            action_votes[action]['recommendations'].append(rec)
        
        # Select action with highest weighted vote
        best_action = max(action_votes.keys(), key=lambda a: action_votes[a]['total_weight'])
        best_recommendations = action_votes[best_action]['recommendations']
        
        # Calculate blended amount (weighted average)
        total_weight = sum(rec['weight'] for rec in best_recommendations)
        blended_amount = sum(rec['amount'] * rec['weight'] for rec in best_recommendations) / total_weight
        
        # Calculate overall confidence
        overall_confidence = action_votes[best_action]['total_weight'] / sum(self.module_weights.values())
        
        return {
            'action': best_action,
            'amount': int(blended_amount),
            'confidence': min(1.0, overall_confidence),
            'contributing_modules': [rec['module'] for rec in best_recommendations],
            'reasoning': self._blend_reasoning(best_recommendations),
            'source': 'confidence_weighted_blend'
        }

    def _extract_module_recommendation(
        self, 
        module_output: Dict[str, Any], 
        module_name: str, 
        game_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract actionable recommendation from a module's output."""
        
        if module_name == 'gto':
            action = module_output.get('action')
            amount = module_output.get('amount', 0)
            if action:
                return {'action': action, 'amount': amount, 'reasoning': 'GTO recommendation'}
        
        elif module_name == 'heuristics':
            recommendation = module_output.get('recommendation')
            if recommendation and isinstance(recommendation, dict):
                return recommendation
        
        elif module_name == 'hand_strength':
            # Convert hand strength to action recommendation
            overall_strength = module_output.get('overall_strength', 0.5)
            return self._hand_strength_to_action(overall_strength, game_state)
        
        elif module_name == 'opponents':
            # Extract best exploit opportunity
            exploits = module_output.get('exploit_opportunities', [])
            if exploits:
                best_exploit = max(exploits, key=lambda x: x.get('confidence', 0))
                return self._exploit_to_action(best_exploit, game_state)
        
        return None

    def _hand_strength_to_action(self, overall_strength: float, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert hand strength to action recommendation."""
        valid_actions = game_state.get('valid_actions', [])
        pot_size = game_state.get('pot_size', 0)
        
        if overall_strength > 0.7:  # Strong hand
            # Look for raise opportunity
            for action in valid_actions:
                if action['action'] == 'raise':
                    amount = int(pot_size * 0.6)  # 60% pot bet
                    return {'action': 'raise', 'amount': amount, 'reasoning': 'Strong hand value bet'}
            # Fall back to call if can't raise
            for action in valid_actions:
                if action['action'] == 'call':
                    return {'action': 'call', 'amount': action.get('amount', 0), 'reasoning': 'Strong hand call'}
        
        elif overall_strength > 0.4:  # Marginal hand
            # Call if odds are good
            for action in valid_actions:
                if action['action'] == 'call':
                    return {'action': 'call', 'amount': action.get('amount', 0), 'reasoning': 'Marginal hand call'}
        
        # Weak hand - fold
        return {'action': 'fold', 'amount': 0, 'reasoning': 'Weak hand fold'}

    def _exploit_to_action(self, exploit: Dict[str, Any], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert exploit opportunity to action recommendation."""
        exploit_type = exploit.get('type', '')
        pot_size = game_state.get('pot_size', 0)
        
        if exploit_type == 'bluff_opportunity':
            return {
                'action': 'raise', 
                'amount': int(pot_size * 0.5), 
                'reasoning': 'Bluff opportunity'
            }
        elif exploit_type == 'value_bet_opportunity':
            return {
                'action': 'raise', 
                'amount': int(pot_size * 0.7), 
                'reasoning': 'Value bet opportunity'
            }
        
        return {'action': 'call', 'amount': 0, 'reasoning': 'Generic exploit'}

    def _get_conservative_fallback(self, valid_actions: List[Dict]) -> Dict[str, Any]:
        """Get conservative fallback when no confident recommendations."""
        # Try to call if cheap, otherwise fold
        for action in valid_actions:
            if action['action'] == 'call' and action.get('amount', 0) == 0:
                return {
                    'action': 'call', 
                    'amount': 0, 
                    'confidence': 0.3,
                    'reasoning': 'Conservative fallback - free call'
                }
        
        return {
            'action': 'fold', 
            'amount': 0, 
            'confidence': 0.5,
            'reasoning': 'Conservative fallback - fold'
        }

    def _blend_reasoning(self, recommendations: List[Dict]) -> str:
        """Blend reasoning from multiple recommendations."""
        reasons = [rec['reasoning'] for rec in recommendations if rec.get('reasoning')]
        modules = [rec['module'] for rec in recommendations]
        
        return f"Consensus from {', '.join(modules)}: {'; '.join(reasons[:2])}"

    def _apply_confident_opponent_adjustments(
        self, 
        blended_recommendation: Dict[str, Any], 
        opponent_profile: Dict[str, Any],
        system1_outputs: Dict[str, Any],
        confidence_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Apply opponent adjustments only when we're confident about opponent model."""
        opponent_confidence = confidence_scores.get('opponents', 0.0)
        
        # Only apply opponent adjustments if we have high confidence
        if opponent_confidence < self.high_confidence_threshold or not opponent_profile:
            return blended_recommendation
        
        # Apply the existing opponent adjustment logic with confidence weighting
        adjusted = self._apply_opponent_adjustments_legacy(
            blended_recommendation, opponent_profile, system1_outputs
        )
        
        # Blend original and adjusted based on opponent confidence
        confidence_weight = opponent_confidence
        
        # If amounts differ, blend them
        if adjusted.get('amount') != blended_recommendation.get('amount'):
            original_amount = blended_recommendation.get('amount', 0)
            adjusted_amount = adjusted.get('amount', 0)
            blended_amount = int(
                original_amount * (1 - confidence_weight) + 
                adjusted_amount * confidence_weight
            )
            adjusted['amount'] = blended_amount
        
        adjusted['reasoning'] += f" (opponent confidence: {opponent_confidence:.2f})"
        return adjusted

    def _apply_opponent_adjustments_legacy(
        self, 
        recommendation: Dict[str, Any], 
        opponent_profile: Dict[str, Any],
        system1_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Legacy opponent adjustment logic (from original synthesizer)."""
        # This preserves the existing opponent adjustment logic
        return self._apply_opponent_specific_adjustments(
            recommendation.copy(), opponent_profile, {}, system1_outputs
        )

    def _generate_confidence_analysis(
        self, 
        game_state: Dict[str, Any],
        system1_outputs: Dict[str, Any],
        confidence_scores: Dict[str, float],
        blended_recommendation: Dict[str, Any],
        final_action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis with confidence breakdown."""
        
        # Module confidence summary
        active_modules = [
            module for module, conf in confidence_scores.items() 
            if conf >= self.min_confidence_threshold
        ]
        
        # Overall decision confidence
        overall_confidence = final_action.get('confidence', 0.5)
        
        # Reasoning summary
        reasoning_parts = []
        reasoning_parts.append(f"Confidence-weighted decision from {len(active_modules)} modules")
        reasoning_parts.append(f"Contributing: {', '.join(active_modules)}")
        reasoning_parts.append(blended_recommendation.get('reasoning', ''))
        
        analysis = {
            'reasoning': '; '.join(reasoning_parts),
            'confidence': overall_confidence,
            'source': 'enhanced_synthesizer_phase5',
            
            # Phase 5 specific analysis
            'confidence_breakdown': confidence_scores,
            'active_modules': active_modules,
            'blending_method': 'confidence_weighted_voting',
            'decision_path': {
                'initial_blend': blended_recommendation.get('action'),
                'final_action': final_action.get('action'),
                'confidence_threshold_used': self.min_confidence_threshold
            },
            
            # Detailed module analysis
            'module_analysis': {
                module: {
                    'confidence': confidence_scores.get(module, 0.0),
                    'active': confidence_scores.get(module, 0.0) >= self.min_confidence_threshold,
                    'output': system1_outputs.get(module, {})
                }
                for module in self.module_weights.keys()
            }
        }
        
        return analysis

    def _apply_style_adjustments(
        self, 
        action: str, 
        game_state: Dict[str, Any], 
        recommendation: Dict[str, Any]
    ) -> Tuple[str, int]:
        """Apply player style adjustments (tightness/aggression)."""
        pot_size = game_state.get('pot_size', 0)
        our_stack = game_state.get('our_stack', 1000)
        
        # Calculate base amounts
        if action == 'raise':
            base_amount = int(pot_size * 0.6)  # Base bet size
            
            # Adjust for aggression
            aggression_multiplier = 0.5 + (self.aggression * 1.0)
            amount = int(base_amount * aggression_multiplier)
            
            # Apply tightness filter
            if self.tightness > 0.7:
                # Very tight - only bet with strong hands
                if recommendation.get('confidence', 0) < 0.7:
                    action = 'call'
                    amount = self._get_call_amount(game_state)
            
            amount = min(amount, our_stack)
            
        elif action == 'call':
            amount = self._get_call_amount(game_state)
            
            # Loose players call more often
            if self.tightness < 0.3 and recommendation.get('confidence', 0) > 0.4:
                pass  # Keep the call
            elif self.tightness > 0.7 and recommendation.get('confidence', 0) < 0.6:
                action = 'fold'
                amount = 0
        else:
            amount = 0
        
        return action, amount

    def _apply_risk_management(
        self, 
        action: str, 
        amount: int, 
        game_state: Dict[str, Any], 
        confidence: float
    ) -> Tuple[str, int]:
        """Apply risk management constraints."""
        our_stack = game_state.get('our_stack', 1000)
        pot_size = game_state.get('pot_size', 0)
        
        # Stack preservation
        if action in ['raise', 'bet'] and amount > our_stack * 0.3:
            if confidence < 0.8:  # Only risk big with high confidence
                amount = int(our_stack * 0.2)  # Smaller bet
        
        # Don't risk more than 20% of stack without high confidence
        if amount > our_stack * 0.2 and confidence < 0.75:
            amount = int(our_stack * 0.15)
        
        # Emergency fold for very low confidence
        if confidence < 0.3 and amount > pot_size * 0.5:
            action = 'fold'
            amount = 0
        
        return action, min(amount, our_stack)

    def _apply_variance_reduction(
        self, 
        action: str, 
        amount: int, 
        game_state: Dict[str, Any]
    ) -> Tuple[str, int]:
        """Apply variance reduction techniques."""
        street = game_state.get('street', 'preflop')
        
        # Reduce bet sizes on later streets to control variance
        if street in ['turn', 'river'] and action in ['raise', 'bet']:
            amount = int(amount * 0.8)  # Smaller bets on later streets
        
        # Round bet sizes to avoid giving away information
        if amount > 0:
            # Round to nearest 5 chips for smaller bets, 10 for larger
            if amount < 100:
                amount = round(amount / 5) * 5
            else:
                amount = round(amount / 10) * 10
        
        return action, amount

    def _generate_analysis(
        self, 
        game_state: Dict[str, Any],
        system1_outputs: Dict[str, Any],
        our_equity: float,
        required_equity: float,
        equity_analysis: Dict[str, Any],
        core_decision: Dict[str, Any],
        final_action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis of the decision."""
        reasoning_parts = []
        
        # Core equity analysis
        reasoning_parts.append(f"Equity: {our_equity:.2f}, Required: {required_equity:.2f}")
        
        if our_equity > required_equity:
            reasoning_parts.append("Mathematically profitable")
        else:
            reasoning_parts.append("Mathematically unprofitable")
        
        # Hand strength analysis
        hand_strength_output = system1_outputs.get('hand_strength', {})
        if hand_strength_output:
            most_likely_hand = hand_strength_output.get('most_likely_hand', 'Unknown')
            hand_confidence = hand_strength_output.get('confidence', 0)
            reasoning_parts.append(f"Most likely: {most_likely_hand} (conf: {hand_confidence:.2f})")
        
        # GTO component
        gto_output = system1_outputs.get('gto', {})
        if gto_output:
            gto_action = gto_output.get('action', 'unknown')
            gto_conf = gto_output.get('confidence', 0)
            reasoning_parts.append(f"GTO suggests: {gto_action} (conf: {gto_conf:.2f})")
        
        # Exploitative component
        opponent_output = system1_outputs.get('opponents', {})
        exploits = opponent_output.get('exploit_opportunities', [])
        if exploits:
            reasoning_parts.append(f"Found {len(exploits)} exploit opportunities")
        
        # Final synthesis
        final_reasoning = "; ".join(reasoning_parts)
        
        return {
            'reasoning': final_reasoning,
            'confidence': final_action.get('confidence', 0.5),
            'our_equity': our_equity,
            'required_equity': required_equity,
            'pot_odds_analysis': equity_analysis,
            'core_decision': core_decision['action'],
            'final_decision': final_action['action'],
            'hand_strength_analysis': hand_strength_output,
            'meta_adjustments': 'style and risk management applied',
            'source': 'synthesizer'
        }

    # Helper methods
    def _is_valid_action(self, action: str, valid_actions: List[Dict]) -> bool:
        """Check if an action is valid."""
        return any(a['action'] == action for a in valid_actions)

    def _get_fallback_action(self, valid_actions: List[Dict]) -> str:
        """Get a safe fallback action."""
        for action in valid_actions:
            if action['action'] == 'call':
                return 'call'
        return 'fold'

    def _get_call_amount(self, game_state: Dict[str, Any]) -> int:
        """Get the amount needed to call."""
        valid_actions = game_state.get('valid_actions', [])
        for action in valid_actions:
            if action['action'] == 'call':
                return action.get('amount', 0)
        return 0
        """Check if an action is valid."""
        return any(a['action'] == action for a in valid_actions)

    def _get_fallback_action(self, valid_actions: List[Dict]) -> str:
        """Get a safe fallback action."""
        for action in valid_actions:
            if action['action'] == 'call':
                return 'call'
        return 'fold'

    def _get_call_amount(self, game_state: Dict[str, Any]) -> int:
        """Get the amount needed to call."""
        valid_actions = game_state.get('valid_actions', [])
        for action in valid_actions:
            if action['action'] == 'call':
                return action.get('amount', 0)
        return 0

    # Dynamic parameter adjustment methods
    def adjust_style(self, tightness: float, aggression: float):
        """Dynamically adjust playing style."""
        self.tightness = max(0.0, min(1.0, tightness))
        self.aggression = max(0.0, min(1.0, aggression))
        self.logger.info(f"Style adjusted: tightness={self.tightness:.2f}, aggression={self.aggression:.2f}")

    def adjust_gto_exploit_balance(self, gto_weight: float):
        """Adjust the balance between GTO and exploitative play."""
        self.gto_weight = max(0.0, min(1.0, gto_weight))
        self.exploit_weight = 1.0 - self.gto_weight
        self.logger.info(f"GTO/Exploit balance adjusted: {self.gto_weight:.2f}/{self.exploit_weight:.2f}")