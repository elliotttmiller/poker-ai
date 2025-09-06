"""
GTO Core Module for Project PokerMind.

This module interfaces with PokerRL-trained models to provide
Game Theory Optimal (GTO) recommendations.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np


class GTOCore:
    """
    GTO Core module providing game theory optimal recommendations.
    
    This module will interface with PokerRL-trained models and provide
    GTO-sound baseline decisions for the Synthesizer.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the GTO Core module.
        
        Args:
            model_path: Path to the trained GTO model (ONNX format)
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
        # Load the model if path is provided
        if model_path:
            self._load_model(model_path)

    def get_recommendation(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get GTO recommendation for the current game state.
        
        Enhanced in Phase 5 to include confidence scoring for weighted blending.
        
        Args:
            game_state: Current game state dict
            
        Returns:
            Dict containing GTO recommendation with confidence score (0.0-1.0)
        """
        try:
            if not self.is_loaded:
                return self._get_fallback_recommendation(game_state)
            
            # Convert game state to model input format
            model_input = self._encode_game_state(game_state)
            
            # Get model prediction
            action_probs = self._run_inference(model_input)
            
            # Convert to recommendation format with enhanced confidence
            recommendation = self._decode_model_output(
                action_probs, game_state
            )
            
            self.logger.debug(f"GTO recommendation: {recommendation}")
            return recommendation
            
        except Exception as e:
            self.logger.warning(f"GTO Core error: {e}")
            return self._get_fallback_recommendation(game_state)

    def _load_model(self, model_path: str):
        """Load the GTO model from file."""
        try:
            # TODO: Implement ONNX model loading
            # import onnxruntime as ort
            # self.model = ort.InferenceSession(model_path)
            # self.is_loaded = True
            self.logger.info(f"GTO model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load GTO model: {e}")
            self.is_loaded = False

    def _encode_game_state(self, game_state: Dict[str, Any]) -> np.ndarray:
        """
        Encode game state into format expected by the GTO model.
        
        Args:
            game_state: Current game state
            
        Returns:
            Numpy array suitable for model input
        """
        # TODO: Implement proper state encoding for PokerRL models
        # This should encode:
        # - Hole cards (one-hot or integer encoding)
        # - Community cards
        # - Betting history
        # - Stack sizes
        # - Position information
        
        # Placeholder implementation
        features = []
        
        # Encode hole cards (simplified)
        hole_cards = game_state.get('hole_cards', [])
        features.extend([len(hole_cards), 0])  # Placeholder
        
        # Encode pot size and stack (normalized)
        pot_size = game_state.get('pot_size', 0)
        our_stack = game_state.get('our_stack', 1000)
        features.extend([pot_size / 1000.0, our_stack / 1000.0])
        
        # Encode street
        street_encoding = {
            'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3
        }
        street = game_state.get('street', 'preflop')
        features.append(street_encoding.get(street, 0))
        
        return np.array(features, dtype=np.float32)

    def _run_inference(self, model_input: np.ndarray) -> np.ndarray:
        """Run inference on the loaded model."""
        # TODO: Implement actual model inference
        # return self.model.run(None, {'input': model_input})[0]
        
        # Placeholder: return uniform random distribution
        return np.random.dirichlet([1, 1, 1])  # fold, call, raise

    def _decode_model_output(
        self, 
        action_probs: np.ndarray, 
        game_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decode model output into action recommendation with enhanced confidence scoring.
        
        Enhanced in Phase 5 to provide better confidence metrics for weighted blending.
        
        Args:
            action_probs: Model output probabilities
            game_state: Current game state for valid actions
            
        Returns:
            Action recommendation dict with enhanced confidence scoring
        """
        valid_actions = game_state.get('valid_actions', [])
        
        # Map probabilities to valid actions
        action_names = ['fold', 'call', 'raise']
        
        # Find the highest probability valid action
        best_action = None
        best_prob = 0.0
        second_best_prob = 0.0
        
        for i, prob in enumerate(action_probs):
            action_name = action_names[i] if i < len(action_names) else 'fold'
            
            # Check if this action is valid
            if self._is_action_valid(action_name, valid_actions):
                if prob > best_prob:
                    second_best_prob = best_prob
                    best_action = action_name
                    best_prob = prob
                elif prob > second_best_prob:
                    second_best_prob = prob
        
        # Enhanced confidence calculation
        # Consider the gap between best and second-best options
        probability_gap = best_prob - second_best_prob if second_best_prob > 0 else best_prob
        
        # Entropy-based confidence (lower entropy = higher confidence)
        entropy = -np.sum(action_probs * np.log2(action_probs + 1e-10))
        max_entropy = np.log2(len(action_probs))
        entropy_confidence = 1.0 - (entropy / max_entropy)
        
        # Combined confidence score (weighted average)
        confidence = (0.6 * best_prob + 0.3 * probability_gap + 0.1 * entropy_confidence)
        confidence = min(1.0, max(0.0, confidence))
        
        # Get amount if it's a raise/bet
        amount = 0
        if best_action == 'raise':
            amount = self._calculate_bet_size(game_state, action_probs)
        elif best_action == 'call':
            for action in valid_actions:
                if action['action'] == 'call':
                    amount = action.get('amount', 0)
                    break
        
        return {
            'action': best_action or 'fold',
            'amount': amount,
            'confidence': float(confidence),
            'raw_probability': float(best_prob),
            'probability_gap': float(probability_gap),
            'entropy_confidence': float(entropy_confidence),
            'action_probs': action_probs.tolist(),
            'source': 'gto_core'
        }

    def _is_action_valid(self, action_name: str, valid_actions: list) -> bool:
        """Check if an action is valid in the current context."""
        for action in valid_actions:
            if action['action'] == action_name:
                return True
        return False

    def _calculate_bet_size(
        self, 
        game_state: Dict[str, Any], 
        action_probs: np.ndarray
    ) -> int:
        """Calculate appropriate bet/raise size based on GTO principles."""
        pot_size = game_state.get('pot_size', 0)
        our_stack = game_state.get('our_stack', 1000)
        
        # Simple bet sizing: 50-75% of pot
        bet_fraction = 0.5 + (action_probs[2] * 0.25)  # Use raise probability
        bet_size = int(pot_size * bet_fraction)
        
        # Ensure bet is within valid range
        valid_actions = game_state.get('valid_actions', [])
        for action in valid_actions:
            if action['action'] == 'raise':
                min_raise = action.get('amount', {}).get('min', bet_size)
                max_raise = action.get('amount', {}).get('max', our_stack)
                bet_size = max(min_raise, min(bet_size, max_raise))
                break
        
        return bet_size

    def _get_fallback_recommendation(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback recommendation when model is unavailable."""
        # Conservative fallback: fold or call
        valid_actions = game_state.get('valid_actions', [])
        
        for action in valid_actions:
            if action['action'] == 'call':
                return {
                    'action': 'call',
                    'amount': action.get('amount', 0),
                    'confidence': 0.3,
                    'source': 'gto_fallback'
                }
        
        return {
            'action': 'fold',
            'amount': 0,
            'confidence': 0.5,
            'source': 'gto_fallback'
        }

    def update_model(self, model_path: str):
        """Update the GTO model with a new trained version."""
        self._load_model(model_path)
        self.logger.info("GTO model updated successfully")