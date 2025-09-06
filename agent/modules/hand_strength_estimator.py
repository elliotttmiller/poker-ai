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
        
        Enhanced in Phase 5 to provide confidence scoring for weighted blending.
        
        Args:
            game_state: Current game state containing hole cards and community cards
            
        Returns:
            Dict containing hand strength analysis with enhanced confidence scoring
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
        """Estimate postflop hand strength using basic heuristics that analyze actual cards."""
        # Analyze the actual hand combination
        all_cards = hole_cards + community_cards
        
        if len(all_cards) < 5:
            # Not enough cards for full analysis, fall back to basic heuristic
            return self._estimate_incomplete_hand(hole_cards, community_cards, street)
        
        # Analyze the actual made hand
        hand_analysis = self._analyze_made_hand(all_cards)
        probabilities = np.zeros(9)
        
        # Set probabilities based on actual hand strength
        made_hand_type = hand_analysis['hand_type']
        
        if made_hand_type == 'straight_flush':
            probabilities[8] = 1.0
        elif made_hand_type == 'four_of_kind':
            probabilities[7] = 1.0
        elif made_hand_type == 'full_house':
            probabilities[6] = 1.0
        elif made_hand_type == 'flush':
            probabilities[5] = 1.0
        elif made_hand_type == 'straight':
            probabilities[4] = 1.0
        elif made_hand_type == 'three_of_kind':
            probabilities[3] = 1.0
        elif made_hand_type == 'two_pair':
            probabilities[2] = 1.0
        elif made_hand_type == 'one_pair':
            probabilities[1] = 1.0
        else:  # high card
            probabilities[0] = 1.0
        
        return self._format_estimation_result(probabilities, street, 'heuristic')
    
    def _estimate_incomplete_hand(self, hole_cards: List[str], community_cards: List[str], street: str) -> Dict[str, Any]:
        """Estimate hand strength when we don't have all 5 cards yet."""
        all_cards = hole_cards + community_cards
        probabilities = np.zeros(9)
        
        if len(all_cards) < 2:
            # Not enough cards
            probabilities[0] = 1.0  # High card
            return self._format_estimation_result(probabilities, street, 'heuristic')
        
        # Check for pairs and potential hands
        ranks = {}
        suits = {}
        
        for card in all_cards:
            if len(card) >= 2:
                rank = card[0]
                suit = card[1]
                ranks[rank] = ranks.get(rank, 0) + 1
                suits[suit] = suits.get(suit, 0) + 1
        
        max_rank_count = max(ranks.values()) if ranks else 1
        max_suit_count = max(suits.values()) if suits else 1
        
        # Assign probabilities based on what we can see
        if max_rank_count >= 3:  # Already have trips
            probabilities[3] = 0.8  # Three of a kind
            probabilities[6] = 0.15  # Potential full house
            probabilities[7] = 0.05  # Potential four of a kind
        elif max_rank_count >= 2:  # Have a pair
            pair_count = sum(1 for count in ranks.values() if count >= 2)
            if pair_count >= 2:  # Two pair
                probabilities[2] = 0.7
                probabilities[6] = 0.2  # Potential full house
                probabilities[1] = 0.1
            else:  # One pair
                probabilities[1] = 0.6
                probabilities[3] = 0.2  # Potential trips
                probabilities[2] = 0.15  # Potential two pair
                probabilities[0] = 0.05
        else:  # No pair yet
            # Check flush/straight potential
            if max_suit_count >= 3:  # Flush draw
                probabilities[0] = 0.4
                probabilities[1] = 0.3
                probabilities[5] = 0.2  # Potential flush
                probabilities[8] = 0.1  # Potential straight flush
            else:  # High card most likely
                probabilities[0] = 0.5
                probabilities[1] = 0.4
                probabilities[2] = 0.1
        
        return self._format_estimation_result(probabilities, street, 'heuristic')
    
    def _analyze_made_hand(self, cards: List[str]) -> Dict[str, Any]:
        """Analyze what hand is made with the given cards."""
        if len(cards) < 5:
            return {'hand_type': 'high_card', 'strength': 0.1}
        
        # Parse cards
        ranks = {}
        suits = {}
        card_values = []
        
        for card in cards:
            if len(card) >= 2:
                rank = card[0]
                suit = card[1]
                
                ranks[rank] = ranks.get(rank, 0) + 1
                suits[suit] = suits.get(suit, 0) + 1
                
                # Convert rank to numeric value for straight detection
                if rank == 'A':
                    card_values.append(14)
                elif rank == 'K':
                    card_values.append(13)
                elif rank == 'Q':
                    card_values.append(12)
                elif rank == 'J':
                    card_values.append(11)
                else:
                    try:
                        card_values.append(int(rank))
                    except:
                        card_values.append(2)
        
        # Check for various hands
        rank_counts = sorted(ranks.values(), reverse=True)
        max_suit_count = max(suits.values()) if suits else 0
        
        # Check for straight
        unique_values = sorted(set(card_values))
        is_straight = False
        if len(unique_values) >= 5:
            for i in range(len(unique_values) - 4):
                if unique_values[i+4] - unique_values[i] == 4:
                    is_straight = True
                    break
        
        # Determine hand type
        if is_straight and max_suit_count >= 5:
            return {'hand_type': 'straight_flush', 'strength': 0.95}
        elif rank_counts[0] >= 4:
            return {'hand_type': 'four_of_kind', 'strength': 0.9}
        elif rank_counts[0] >= 3 and rank_counts[1] >= 2:
            return {'hand_type': 'full_house', 'strength': 0.85}
        elif max_suit_count >= 5:
            return {'hand_type': 'flush', 'strength': 0.75}
        elif is_straight:
            return {'hand_type': 'straight', 'strength': 0.65}
        elif rank_counts[0] >= 3:
            return {'hand_type': 'three_of_kind', 'strength': 0.55}
        elif rank_counts[0] >= 2 and rank_counts[1] >= 2:
            return {'hand_type': 'two_pair', 'strength': 0.45}
        elif rank_counts[0] >= 2:
            return {'hand_type': 'one_pair', 'strength': 0.25}
        else:
            # High card strength based on best card
            high_card_strength = max(card_values) / 14.0 if card_values else 0.1
            return {'hand_type': 'high_card', 'strength': high_card_strength * 0.2}

    def _format_estimation_result(self, probabilities: np.ndarray, street: str, method: str) -> Dict[str, Any]:
        """
        Format the estimation result into a standardized dictionary.
        
        Enhanced in Phase 5 to provide multiple confidence metrics for weighted blending.
        """
        
        # Ensure probabilities sum to 1.0
        probabilities = probabilities / np.sum(probabilities)
        
        # Calculate overall hand strength (weighted sum)
        weights = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.95, 0.98, 1.0])
        overall_strength = np.sum(probabilities * weights)
        
        # Find most likely hand type
        most_likely_index = np.argmax(probabilities)
        most_likely_hand = self.categories[most_likely_index]
        most_likely_prob = probabilities[most_likely_index]
        
        # Enhanced confidence scoring for Phase 5
        
        # 1. Entropy-based confidence (lower entropy = higher confidence)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        max_entropy = np.log2(len(probabilities))
        entropy_confidence = 1.0 - (entropy / max_entropy)
        
        # 2. Dominance confidence (how much the top prediction dominates)
        dominance_confidence = most_likely_prob
        
        # 3. Gap confidence (gap between first and second most likely)
        sorted_probs = np.sort(probabilities)[::-1]  # Descending
        gap_confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        
        # 4. Street-based confidence (more certain as more cards are revealed)
        street_confidence_map = {
            'preflop': 0.3,  # Low certainty preflop
            'flop': 0.6,     # Moderate certainty on flop
            'turn': 0.8,     # High certainty on turn
            'river': 1.0     # Full certainty on river
        }
        street_confidence = street_confidence_map.get(street, 0.5)
        
        # 5. Method-based confidence (model vs heuristic)
        method_confidence = 0.9 if method == 'onnx_model' else 0.7
        
        # Combined confidence score (weighted average of all factors)
        confidence = (
            0.3 * entropy_confidence +
            0.2 * dominance_confidence +
            0.2 * gap_confidence +
            0.2 * street_confidence +
            0.1 * method_confidence
        )
        confidence = min(1.0, max(0.0, confidence))
        
        return {
            'probabilities': probabilities.tolist(),
            'categories': self.categories,
            'overall_strength': float(overall_strength),
            'strength': float(overall_strength),  # Add for compatibility with tests
            'most_likely_hand': most_likely_hand,
            'most_likely_probability': float(most_likely_prob),
            'confidence': float(confidence),
            
            # Enhanced confidence breakdown for Phase 5
            'confidence_breakdown': {
                'entropy_confidence': float(entropy_confidence),
                'dominance_confidence': float(dominance_confidence),
                'gap_confidence': float(gap_confidence),
                'street_confidence': float(street_confidence),
                'method_confidence': float(method_confidence)
            },
            
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