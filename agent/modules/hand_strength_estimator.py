"""
Hand Strength Estimator Module for Project PokerMind.

This module implements a neural network-based hand strength estimator
that predicts the probability distribution over different hand strengths.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
import os

# Try to import ONNX runtime, fallback gracefully if not available
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from ..utils import vectorize_cards, estimate_preflop_hand_strength


class HandStrengthEstimator:
    """
    Neural network-based hand strength estimator.
    
    Predicts the probability distribution over 9 hand strength categories:
    0: High Card
    1: One Pair
    2: Two Pair
    3: Three of a Kind
    4: Straight
    5: Flush
    6: Full House
    7: Four of a Kind
    8: Straight Flush / Royal Flush
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the Hand Strength Estimator.
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.session = None
        self.use_fallback = True
        
        # Hand strength categories
        self.categories = [
            "High Card",
            "One Pair", 
            "Two Pair",
            "Three of a Kind",
            "Straight",
            "Flush", 
            "Full House",
            "Four of a Kind",
            "Straight Flush"
        ]
        
        # Try to load the ONNX model
        if model_path and os.path.exists(model_path) and ONNX_AVAILABLE:
            self._load_onnx_model(model_path)
        else:
            self.logger.info("Using fallback heuristic-based hand strength estimation")
            self.use_fallback = True

    def _load_onnx_model(self, model_path: str):
        """Load the ONNX model for inference."""
        try:
            # Configure ONNX providers (prefer CUDA if available)
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.use_fallback = False
            self.logger.info(f"Loaded ONNX model from {model_path}")
            
            # Validate input/output shapes
            input_shape = self.session.get_inputs()[0].shape
            output_shape = self.session.get_outputs()[0].shape
            
            self.logger.info(f"Model input shape: {input_shape}")
            self.logger.info(f"Model output shape: {output_shape}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load ONNX model: {e}")
            self.logger.info("Falling back to heuristic-based estimation")
            self.use_fallback = True

    def estimate(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate hand strength probabilities for the current game state.
        
        Args:
            game_state: Current game state containing hole cards and community cards
            
        Returns:
            Dict containing hand strength analysis
        """
        try:
            hole_cards = game_state.get('hole_cards', [])
            community_cards = game_state.get('community_cards', [])
            street = game_state.get('street', 'preflop')
            
            if self.use_fallback:
                return self._fallback_estimation(hole_cards, community_cards, street)
            else:
                return self._onnx_estimation(hole_cards, community_cards, street)
                
        except Exception as e:
            self.logger.error(f"Hand strength estimation error: {e}")
            return self._get_default_estimate()

    def _onnx_estimation(self, hole_cards: List[str], community_cards: List[str], street: str) -> Dict[str, Any]:
        """Use ONNX model for hand strength estimation."""
        # Vectorize cards for neural network input
        card_vector = vectorize_cards(hole_cards, community_cards)
        
        # Prepare input for ONNX model (batch size = 1)
        input_data = card_vector.reshape(1, -1).astype(np.float32)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        result = self.session.run([output_name], {input_name: input_data})
        probabilities = result[0][0]  # Remove batch dimension
        
        # Ensure probabilities sum to 1.0
        probabilities = probabilities / np.sum(probabilities)
        
        return self._format_estimation_result(probabilities, street, 'onnx_model')

    def _fallback_estimation(self, hole_cards: List[str], community_cards: List[str], street: str) -> Dict[str, Any]:
        """Use heuristic-based hand strength estimation as fallback."""
        
        if street == 'preflop':
            return self._estimate_preflop_strength(hole_cards)
        else:
            return self._estimate_postflop_strength(hole_cards, community_cards, street)

    def _estimate_preflop_strength(self, hole_cards: List[str]) -> Dict[str, Any]:
        """Estimate preflop hand strength using heuristics."""
        base_strength = estimate_preflop_hand_strength(hole_cards)
        
        # Convert single strength value to probability distribution
        # Most preflop hands will make one pair or better
        probabilities = np.zeros(9)
        
        if base_strength > 0.8:  # Premium hands (AA, KK, etc.)
            probabilities[0] = 0.1   # High Card
            probabilities[1] = 0.4   # One Pair
            probabilities[2] = 0.25  # Two Pair
            probabilities[3] = 0.15  # Three of a Kind
            probabilities[4] = 0.05  # Straight
            probabilities[5] = 0.03  # Flush
            probabilities[6] = 0.015 # Full House
            probabilities[7] = 0.003 # Four of a Kind
            probabilities[8] = 0.002 # Straight Flush
        elif base_strength > 0.6:  # Good hands
            probabilities[0] = 0.15  # High Card
            probabilities[1] = 0.5   # One Pair
            probabilities[2] = 0.2   # Two Pair
            probabilities[3] = 0.1   # Three of a Kind
            probabilities[4] = 0.03  # Straight
            probabilities[5] = 0.02  # Flush
            probabilities[6] = 0.0   # Full House
            probabilities[7] = 0.0   # Four of a Kind
            probabilities[8] = 0.0   # Straight Flush
        elif base_strength > 0.4:  # Marginal hands
            probabilities[0] = 0.25  # High Card
            probabilities[1] = 0.6   # One Pair
            probabilities[2] = 0.12  # Two Pair
            probabilities[3] = 0.03  # Three of a Kind
            probabilities[4] = 0.0   # Straight
            probabilities[5] = 0.0   # Flush
            probabilities[6] = 0.0   # Full House
            probabilities[7] = 0.0   # Four of a Kind
            probabilities[8] = 0.0   # Straight Flush
        else:  # Weak hands
            probabilities[0] = 0.4   # High Card
            probabilities[1] = 0.55  # One Pair
            probabilities[2] = 0.05  # Two Pair
            probabilities[3] = 0.0   # Three of a Kind
            probabilities[4] = 0.0   # Straight
            probabilities[5] = 0.0   # Flush
            probabilities[6] = 0.0   # Full House
            probabilities[7] = 0.0   # Four of a Kind
            probabilities[8] = 0.0   # Straight Flush
        
        return self._format_estimation_result(probabilities, 'preflop', 'heuristic')

    def _estimate_postflop_strength(self, hole_cards: List[str], community_cards: List[str], street: str) -> Dict[str, Any]:
        """Estimate postflop hand strength using basic heuristics."""
        # This is a simplified implementation
        # In a real scenario, this would evaluate the actual hand
        
        # For now, use a rough estimation based on street
        probabilities = np.zeros(9)
        
        if street == 'flop':
            # Most flops will be high card or pair
            probabilities[0] = 0.3   # High Card
            probabilities[1] = 0.45  # One Pair
            probabilities[2] = 0.15  # Two Pair
            probabilities[3] = 0.07  # Three of a Kind
            probabilities[4] = 0.02  # Straight
            probabilities[5] = 0.01  # Flush
            probabilities[6] = 0.0   # Full House
            probabilities[7] = 0.0   # Four of a Kind
            probabilities[8] = 0.0   # Straight Flush
        elif street == 'turn':
            # Turn gives more opportunities for strong hands
            probabilities[0] = 0.25  # High Card
            probabilities[1] = 0.4   # One Pair
            probabilities[2] = 0.2   # Two Pair
            probabilities[3] = 0.1   # Three of a Kind
            probabilities[4] = 0.03  # Straight
            probabilities[5] = 0.015 # Flush
            probabilities[6] = 0.005 # Full House
            probabilities[7] = 0.0   # Four of a Kind
            probabilities[8] = 0.0   # Straight Flush
        else:  # river
            # River - final hand strength
            probabilities[0] = 0.2   # High Card
            probabilities[1] = 0.35  # One Pair
            probabilities[2] = 0.25  # Two Pair
            probabilities[3] = 0.12  # Three of a Kind
            probabilities[4] = 0.05  # Straight
            probabilities[5] = 0.025 # Flush
            probabilities[6] = 0.003 # Full House
            probabilities[7] = 0.001 # Four of a Kind
            probabilities[8] = 0.001 # Straight Flush
        
        return self._format_estimation_result(probabilities, street, 'heuristic')

    def _format_estimation_result(self, probabilities: np.ndarray, street: str, method: str) -> Dict[str, Any]:
        """Format the estimation result into a standardized dictionary."""
        
        # Ensure probabilities sum to 1.0
        probabilities = probabilities / np.sum(probabilities)
        
        # Calculate overall hand strength (weighted sum)
        weights = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.95, 0.98, 1.0])
        overall_strength = np.sum(probabilities * weights)
        
        # Find most likely hand type
        most_likely_index = np.argmax(probabilities)
        most_likely_hand = self.categories[most_likely_index]
        
        # Calculate confidence (entropy-based)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        max_entropy = np.log2(len(probabilities))
        confidence = 1.0 - (entropy / max_entropy)
        
        return {
            'probabilities': probabilities.tolist(),
            'categories': self.categories,
            'overall_strength': float(overall_strength),
            'most_likely_hand': most_likely_hand,
            'most_likely_probability': float(probabilities[most_likely_index]),
            'confidence': float(confidence),
            'street': street,
            'method': method,
            'raw_strength': float(overall_strength)  # For compatibility
        }

    def _get_default_estimate(self) -> Dict[str, Any]:
        """Return a safe default estimate in case of errors."""
        # Default to moderate strength
        probabilities = np.array([0.2, 0.5, 0.2, 0.08, 0.02, 0.0, 0.0, 0.0, 0.0])
        return self._format_estimation_result(probabilities, 'unknown', 'default')

    def get_win_probability(self, hand_strength_result: Dict[str, Any]) -> float:
        """
        Convert hand strength probabilities to win probability estimate.
        
        Args:
            hand_strength_result: Result from estimate() method
            
        Returns:
            float: Estimated win probability (0.0 to 1.0)
        """
        overall_strength = hand_strength_result.get('overall_strength', 0.5)
        
        # Simple conversion from hand strength to win probability
        # This is a rough heuristic - in practice, this would consider:
        # - Number of opponents
        # - Board texture
        # - Position
        # - Betting action
        
        # Baseline win probability based on hand strength
        if overall_strength < 0.2:
            return 0.1 + overall_strength * 0.5  # 0.1 to 0.2
        elif overall_strength < 0.5:
            return 0.2 + (overall_strength - 0.2) * 1.0  # 0.2 to 0.5
        elif overall_strength < 0.8:
            return 0.5 + (overall_strength - 0.5) * 1.0  # 0.5 to 0.8
        else:
            return 0.8 + (overall_strength - 0.8) * 1.0  # 0.8 to 1.0

    def update_from_result(self, game_result: Dict[str, Any]):
        """
        Update the estimator based on game results.
        
        This would be used for online learning in a full implementation.
        
        Args:
            game_result: Result information from a completed hand
        """
        # TODO: Implement online learning updates
        # For now, just log the result for potential future training
        self.logger.debug(f"Hand result logged for potential learning: {game_result}")
        pass