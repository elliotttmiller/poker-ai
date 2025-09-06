"""
Synthesizer Module for Project PokerMind.

This module implements System 2 (deliberate analysis) by synthesizing
inputs from all System 1 modules to make final decisions.
"""

import logging
import math
from typing import Dict, Any, Tuple, List
import random


class Synthesizer:
    """
    The Synthesizer implements System 2 of the dual-process architecture.
    
    It takes inputs from all System 1 modules (GTO Core, Opponent Modeler,
    Heuristics Engine) and synthesizes them into a final, reasoned decision.
    """

    def __init__(self):
        """Initialize the Synthesizer."""
        self.logger = logging.getLogger(__name__)
        
        # Default parameters (can be dynamically adjusted)
        self.gto_weight = 0.7
        self.exploit_weight = 0.3
        self.uncertainty_threshold = 0.6
        
        # Player style parameters (dynamic)
        self.tightness = 0.5  # 0.0 = very loose, 1.0 = very tight
        self.aggression = 0.5  # 0.0 = very passive, 1.0 = very aggressive
        
        # Risk management
        self.risk_tolerance = 0.5
        self.bluff_frequency_target = 0.25
        
        self.logger.info("Synthesizer initialized")

    def synthesize_decision(
        self, 
        game_state: Dict[str, Any], 
        system1_outputs: Dict[str, Any]
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
            
            # Phase 2: Calculate base equity and pot odds
            equity_analysis = self._analyze_equity_and_odds(game_state)
            
            # Phase 3: Weight GTO vs Exploitative recommendations
            weighted_recommendation = self._weight_recommendations(
                system1_outputs, equity_analysis, game_state
            )
            
            # Phase 4: Apply meta-cognitive adjustments
            final_action = self._apply_meta_adjustments(
                weighted_recommendation, game_state, system1_outputs
            )
            
            # Phase 5: Generate analysis
            analysis = self._generate_analysis(
                game_state, system1_outputs, equity_analysis, 
                weighted_recommendation, final_action
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
        
        # Calculate pot odds
        total_pot_after_call = pot_size + call_cost
        pot_odds = call_cost / total_pot_after_call if total_pot_after_call > 0 else 0
        
        # Estimate our equity (simplified)
        equity = self._estimate_equity(game_state)
        
        # Calculate expected value for calling
        ev_call = equity * total_pot_after_call - call_cost
        
        return {
            'pot_size': pot_size,
            'call_cost': call_cost,
            'pot_odds': pot_odds,
            'equity': equity,
            'ev_call': ev_call,
            'min_raise': min_raise,
            'max_raise': max_raise,
            'profitable_call': equity > pot_odds
        }

    def _estimate_equity(self, game_state: Dict[str, Any]) -> float:
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

    def _apply_meta_adjustments(
        self, 
        weighted_recommendation: Dict[str, Any], 
        game_state: Dict[str, Any],
        system1_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply meta-cognitive adjustments based on game context."""
        action = weighted_recommendation['action']
        confidence = weighted_recommendation['confidence']
        
        # Apply player style adjustments
        action, amount = self._apply_style_adjustments(
            action, game_state, weighted_recommendation
        )
        
        # Apply risk management
        action, amount = self._apply_risk_management(
            action, amount, game_state, confidence
        )
        
        # Apply variance reduction
        action, amount = self._apply_variance_reduction(
            action, amount, game_state
        )
        
        return {
            'action': action,
            'amount': amount,
            'confidence': confidence,
            'meta_adjustments_applied': True
        }

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
        equity_analysis: Dict[str, Any],
        weighted_recommendation: Dict[str, Any],
        final_action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis of the decision."""
        reasoning_parts = []
        
        # Equity analysis
        equity = equity_analysis.get('equity', 0)
        pot_odds = equity_analysis.get('pot_odds', 0)
        
        reasoning_parts.append(f"Equity: {equity:.2f}, Pot odds: {pot_odds:.2f}")
        
        if equity_analysis.get('profitable_call'):
            reasoning_parts.append("Mathematically profitable call")
        
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
            'equity_analysis': equity_analysis,
            'gto_component': weighted_recommendation.get('gto_component', 0),
            'exploit_component': weighted_recommendation.get('exploit_component', 0),
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