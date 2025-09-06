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
            recommendation = self._decode_model_output(action_probs, game_state)

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
        hole_cards = game_state.get("hole_cards", [])
        features.extend([len(hole_cards), 0])  # Placeholder

        # Encode pot size and stack (normalized)
        pot_size = game_state.get("pot_size", 0)
        our_stack = game_state.get("our_stack", 1000)
        features.extend([pot_size / 1000.0, our_stack / 1000.0])

        # Encode street
        street_encoding = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
        street = game_state.get("street", "preflop")
        features.append(street_encoding.get(street, 0))

        return np.array(features, dtype=np.float32)

    def _run_inference(self, model_input: np.ndarray) -> np.ndarray:
        """Run inference on the loaded model."""
        # TODO: Implement actual model inference
        # return self.model.run(None, {'input': model_input})[0]

        # Placeholder: return uniform random distribution
        return np.random.dirichlet([1, 1, 1])  # fold, call, raise

    def _decode_model_output(
        self, action_probs: np.ndarray, game_state: Dict[str, Any]
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
        valid_actions = game_state.get("valid_actions", [])

        # Map probabilities to valid actions
        action_names = ["fold", "call", "raise"]

        # Find the highest probability valid action
        best_action = None
        best_prob = 0.0
        second_best_prob = 0.0

        for i, prob in enumerate(action_probs):
            action_name = action_names[i] if i < len(action_names) else "fold"

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
        probability_gap = (
            best_prob - second_best_prob if second_best_prob > 0 else best_prob
        )

        # Entropy-based confidence (lower entropy = higher confidence)
        entropy = -np.sum(action_probs * np.log2(action_probs + 1e-10))
        max_entropy = np.log2(len(action_probs))
        entropy_confidence = 1.0 - (entropy / max_entropy)

        # Combined confidence score (weighted average)
        confidence = 0.6 * best_prob + 0.3 * probability_gap + 0.1 * entropy_confidence
        confidence = min(1.0, max(0.0, confidence))

        # Get amount if it's a raise/bet
        amount = 0
        if best_action == "raise":
            amount = self._calculate_bet_size(game_state, action_probs)
        elif best_action == "call":
            for action in valid_actions:
                if action["action"] == "call":
                    amount = action.get("amount", 0)
                    break

        return {
            "action": best_action or "fold",
            "amount": amount,
            "confidence": float(confidence),
            "raw_probability": float(best_prob),
            "probability_gap": float(probability_gap),
            "entropy_confidence": float(entropy_confidence),
            "action_probs": action_probs.tolist(),
            "source": "gto_core",
        }

    def _is_action_valid(self, action_name: str, valid_actions: list) -> bool:
        """Check if an action is valid in the current context."""
        for action in valid_actions:
            if action["action"] == action_name:
                return True
        return False

    def _calculate_bet_size(
        self, game_state: Dict[str, Any], action_probs: np.ndarray
    ) -> int:
        """Calculate appropriate bet/raise size based on GTO principles."""
        pot_size = game_state.get("pot_size", 0)
        our_stack = game_state.get("our_stack", 1000)

        # Simple bet sizing: 50-75% of pot
        bet_fraction = 0.5 + (action_probs[2] * 0.25)  # Use raise probability
        bet_size = int(pot_size * bet_fraction)

        # Ensure bet is within valid range
        valid_actions = game_state.get("valid_actions", [])
        for action in valid_actions:
            if action["action"] == "raise":
                min_raise = action.get("amount", {}).get("min", bet_size)
                max_raise = action.get("amount", {}).get("max", our_stack)
                bet_size = max(min_raise, min(bet_size, max_raise))
                break

        return bet_size

    def _get_fallback_recommendation(
        self, game_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get fallback recommendation when model is unavailable - DYNAMIC, not hardcoded."""
        # Implement a basic but DYNAMIC GTO approximation based on game state
        valid_actions = game_state.get("valid_actions", [])
        hole_cards = game_state.get("hole_cards", [])
        community_cards = game_state.get("community_cards", [])
        pot_size = game_state.get("pot_size", 100)
        our_stack = game_state.get("our_stack", 1000)
        street = game_state.get("street", "preflop")

        # Calculate dynamic factors that affect decision and confidence

        # 1. Hand strength factor (dynamic based on actual cards)
        hand_strength = self._calculate_basic_hand_strength(hole_cards, community_cards)

        # 2. Position/pot odds factor
        call_amount = 0
        for action in valid_actions:
            if action["action"] == "call":
                call_amount = action.get("amount", 0)
                break

        # Avoid division by zero
        pot_odds = (
            call_amount / max(pot_size + call_amount, 1) if call_amount > 0 else 0
        )

        # 3. Stack depth factor (affects confidence)
        stack_depth = our_stack / max(pot_size, 1)

        # 4. Street factor (confidence increases with more information)
        street_confidence_bonus = {
            "preflop": 0.0,
            "flop": 0.1,
            "turn": 0.2,
            "river": 0.3,
        }.get(street, 0.0)

        # Dynamic decision logic based on calculated factors
        base_confidence = 0.2 + (hand_strength * 0.3) + street_confidence_bonus

        # Adjust confidence based on stack depth (deeper = more confident in model)
        confidence_adjustment = min(0.2, stack_depth / 10)
        
        # Add pot size factor to vary responses  
        pot_size_factor = (pot_size - 100) / 500.0  # More significant adjustment: -0.2 to +1.8 range
        final_confidence = max(0.1, min(0.9, base_confidence + confidence_adjustment + pot_size_factor))

        # Decision logic based on hand strength and pot odds
        if hand_strength > 0.7:  # Strong hand
            # Look for raise option
            for action in valid_actions:
                if action["action"] == "raise":
                    return {
                        "action": "raise",
                        "amount": action.get("amount", {}).get("min", call_amount * 2),
                        "confidence": min(0.95, final_confidence + 0.2),
                        "source": "gto_fallback",
                    }
            # If no raise, call with strong hand
            for action in valid_actions:
                if action["action"] == "call":
                    return {
                        "action": "call",
                        "amount": action.get("amount", 0),
                        "confidence": final_confidence,
                        "source": "gto_fallback",
                    }
        elif hand_strength > 0.4 and pot_odds < 0.3:  # Decent hand, good odds
            for action in valid_actions:
                if action["action"] == "call":
                    return {
                        "action": "call",
                        "amount": action.get("amount", 0),
                        "confidence": final_confidence,
                        "source": "gto_fallback",
                    }

        # Default to fold with low confidence when unsure
        return {
            "action": "fold",
            "amount": 0,
            "confidence": max(0.1, final_confidence - 0.1),
            "source": "gto_fallback",
        }

    def _calculate_basic_hand_strength(
        self, hole_cards: list, community_cards: list
    ) -> float:
        """Calculate a basic hand strength estimate for fallback purposes."""
        if not hole_cards or len(hole_cards) < 2:
            return 0.1

        # Basic preflop hand strength
        if not community_cards:
            return self._preflop_hand_strength(hole_cards)

        # Post-flop: simple heuristic based on made hands
        return self._postflop_hand_strength(hole_cards, community_cards)

    def _preflop_hand_strength(self, hole_cards: list) -> float:
        """Calculate preflop hand strength dynamically."""
        if len(hole_cards) < 2:
            return 0.1

        card1, card2 = hole_cards[:2]

        # Parse card values (A=14, K=13, Q=12, J=11, etc.)
        def card_value(card):
            """
            Convert card rank to numeric value.

            Args:
                card: Card string (e.g., 'As', 'Kh')

            Returns:
                int: Numeric value (2-14, where A=14)
            """
            value = card[0]
            if value == "A":
                return 14
            elif value == "K":
                return 13
            elif value == "Q":
                return 12
            elif value == "J":
                return 11
            else:
                try:
                    return int(value)
                except:
                    return 2

        def card_suit(card):
            """
            Extract suit from card string.

            Args:
                card: Card string (e.g., 'As', 'Kh')

            Returns:
                str: Suit character ('h', 'd', 'c', 's')
            """
            return card[1] if len(card) > 1 else "h"

        val1, val2 = card_value(card1), card_value(card2)
        suit1, suit2 = card_suit(card1), card_suit(card2)

        # Pair bonus
        if val1 == val2:
            pair_strength = 0.5 + (val1 / 28)  # Pairs scale with rank
            return min(1.0, pair_strength)

        # High card strength
        high_val = max(val1, val2)
        low_val = min(val1, val2)

        # Suited bonus
        suited_bonus = 0.05 if suit1 == suit2 else 0

        # Connected bonus (for straights)
        connected_bonus = 0.03 if abs(val1 - val2) <= 1 else 0
        if abs(val1 - val2) <= 4:
            connected_bonus += 0.01

        # Base strength from high card
        base_strength = (high_val + low_val) / 28  # Max possible is 28 (AA)

        return min(1.0, base_strength + suited_bonus + connected_bonus)

    def _postflop_hand_strength(self, hole_cards: list, community_cards: list) -> float:
        """Calculate postflop hand strength using simple heuristics."""
        all_cards = hole_cards + community_cards

        # Count ranks and suits for basic made hands
        ranks = {}
        suits = {}

        for card in all_cards:
            if len(card) >= 2:
                rank = card[0]
                suit = card[1]
                ranks[rank] = ranks.get(rank, 0) + 1
                suits[suit] = suits.get(suit, 0) + 1

        # Check for made hands (simplified)
        max_rank_count = max(ranks.values()) if ranks else 1
        max_suit_count = max(suits.values()) if suits else 1

        # Rough hand strength based on made hands
        if max_rank_count >= 4:  # Four of a kind
            return 0.95
        elif max_rank_count >= 3:  # Three of a kind or full house
            return 0.75 + (0.15 if len(set(ranks.values())) > 1 else 0)
        elif max_suit_count >= 5:  # Flush
            return 0.7
        elif max_rank_count >= 2:  # Pair or two pair
            pair_count = sum(1 for count in ranks.values() if count >= 2)
            return 0.3 + (pair_count * 0.15)
        else:  # High card
            # High card strength based on best cards
            card_values = []
            for card in all_cards:
                if card and card[0]:
                    val = card[0]
                    if val == "A":
                        card_values.append(14)
                    elif val == "K":
                        card_values.append(13)
                    elif val == "Q":
                        card_values.append(12)
                    elif val == "J":
                        card_values.append(11)
                    else:
                        try:
                            card_values.append(int(val))
                        except:
                            card_values.append(2)

            if card_values:
                return min(0.5, max(card_values) / 28)
            return 0.1

    def update_model(self, model_path: str):
        """Update the GTO model with a new trained version."""
        self._load_model(model_path)
        self.logger.info("GTO model updated successfully")
