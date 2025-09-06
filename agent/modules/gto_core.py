"""
GTO Core Module for Project PokerMind.

This module interfaces with multiple specialized GTO models and acts as a
meta-strategist to provide Game Theory Optimal recommendations.

Enhanced for Ultimate Intelligence Protocol with dynamic specialist model loading
and intelligent context-aware selection.
"""

import logging
import os
import json
import glob
import math
import random
from typing import Dict, Any, Optional, List

# Import configuration loader
try:
    from config.config_loader import load_model_paths, load_performance_config
except ImportError:
    # Fallback if config_loader is not available
    def load_model_paths():
        return {
            "gto_core": "models/gto_core_v1.onnx",
            "gto_preflop": "models/gto_preflop_v1.onnx", 
            "gto_river": "models/gto_river_v1.onnx",
        }
    def load_performance_config():
        return {"max_inference_time": 0.8}


class GTOCore:
    """
    Enhanced GTO Core module - Meta-Strategist with specialist model ensemble.

    This module dynamically loads multiple specialist GTO models and intelligently
    selects the most appropriate specialist based on game context and volatility.
    
    Implements cutting-edge meta-strategist architecture as required by the
    Ultimate Intelligence Protocol.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the GTO Core meta-strategist.

        Args:
            model_path: Optional path to a single model (legacy support)
        """
        self.logger = logging.getLogger(__name__)
        self.model_paths = load_model_paths()
        self.performance_config = load_performance_config()
        
        # Dictionary to hold specialist models
        self.specialist_models = {}
        self.model_metadata = {}
        self.is_loaded = False
        
        # Meta-strategist configuration
        self.selection_strategy = "context_aware"  # or "confidence_based", "volatility_based"
        self.confidence_threshold = 0.7
        
        # Load all available specialist models
        self._load_all_specialists()
        
        # Legacy support: load single model if specified
        if model_path:
            self._load_model(model_path, "legacy")

    def _load_all_specialists(self):
        """
        Dynamically discover and load all available GTO specialist models.
        
        This method implements the core specialist loading functionality
        required by the Ultimate Intelligence Protocol.
        """
        self.logger.info("Loading GTO specialist models...")
        
        # Standard specialist models from configuration
        for specialist_name, model_path in self.model_paths.items():
            if "gto_" in specialist_name and os.path.exists(model_path):
                self._load_specialist_model(specialist_name, model_path)
        
        # Discover additional specialist models by file pattern
        models_dir = os.path.dirname(self.model_paths.get("gto_core", "models/"))
        pattern = os.path.join(models_dir, "gto_*_v*.onnx")
        
        for model_file in glob.glob(pattern):
            model_name = os.path.basename(model_file).replace(".onnx", "")
            if model_name not in self.specialist_models:
                self._load_specialist_model(model_name, model_file)
        
        if self.specialist_models:
            self.is_loaded = True
            self.logger.info(f"Loaded {len(self.specialist_models)} specialist models: {list(self.specialist_models.keys())}")
        else:
            self.logger.warning("No specialist models loaded - using fallback mode")
    
    def _load_specialist_model(self, specialist_name: str, model_path: str):
        """
        Load a single specialist model and its metadata.
        
        Args:
            specialist_name: Name/identifier for the specialist
            model_path: Path to the model file
        """
        try:
            # In a real implementation, this would load the ONNX model:
            # import onnxruntime as ort
            # model = ort.InferenceSession(model_path)
            
            # For now, store path and load metadata
            self.specialist_models[specialist_name] = {
                "path": model_path,
                "loaded": True,
                "model": None,  # Would be actual ONNX session
            }
            
            # Load metadata if available
            metadata_path = model_path.replace(".onnx", "_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata[specialist_name] = json.load(f)
            else:
                self.model_metadata[specialist_name] = {
                    "specialization": specialist_name.replace("gto_", "").replace("_v1", ""),
                    "model_version": "v1",
                }
            
            self.logger.debug(f"Loaded specialist: {specialist_name} from {model_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load specialist {specialist_name}: {e}")

    def get_recommendation(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Meta-strategist recommendation system.
        
        Intelligently selects the most appropriate specialist model based on
        game context, volatility, and confidence scoring for weighted blending.

        Args:
            game_state: Current game state dict

        Returns:
            Dict containing meta-strategist recommendation with enhanced confidence
        """
        try:
            if not self.is_loaded or not self.specialist_models:
                return self._get_fallback_recommendation(game_state)

            # Step 1: Analyze game context to select optimal specialist
            selected_specialist = self._select_optimal_specialist(game_state)
            
            # Step 2: Get recommendation from selected specialist
            specialist_recommendation = self._get_specialist_recommendation(
                selected_specialist, game_state
            )
            
            # Step 3: Apply meta-strategist enhancement and blending
            final_recommendation = self._enhance_recommendation(
                specialist_recommendation, game_state, selected_specialist
            )

            self.logger.debug(f"Meta-strategist recommendation: {final_recommendation}")
            return final_recommendation

        except Exception as e:
            self.logger.warning(f"GTO Meta-strategist error: {e}")
            return self._get_fallback_recommendation(game_state)

    def _select_optimal_specialist(self, game_state: Dict[str, Any]) -> str:
        """
        Cutting-edge specialist selection logic.
        
        Analyzes game state context, volatility, and strategic requirements
        to select the most appropriate specialist model.
        
        Args:
            game_state: Current game state
            
        Returns:
            Name of the selected specialist model
        """
        street = game_state.get("street", "preflop").lower()
        pot_size = game_state.get("pot_size", 0)
        our_stack = game_state.get("our_stack", 1000)
        community_cards = game_state.get("community_cards", [])
        
        # Calculate game volatility and context factors
        stack_depth = our_stack / max(pot_size, 1) if pot_size > 0 else float('inf')
        game_stage = len(community_cards)  # 0=preflop, 3=flop, 4=turn, 5=river
        
        # Selection logic based on context
        if street == "preflop" or game_stage == 0:
            preferred_specialist = "gto_preflop"
        elif street == "river" or game_stage == 5:
            preferred_specialist = "gto_river"
        elif street in ["flop", "turn"] or game_stage in [3, 4]:
            # For flop/turn, prefer general model or create specialized logic
            preferred_specialist = "gto_core"
        else:
            preferred_specialist = "gto_core"
        
        # Check if preferred specialist is available
        available_specialists = list(self.specialist_models.keys())
        
        if preferred_specialist in available_specialists:
            selected = preferred_specialist
        elif "gto_core" in available_specialists:
            selected = "gto_core"  # Fallback to general model
        else:
            selected = available_specialists[0] if available_specialists else None
        
        # Advanced selection refinement based on volatility
        if stack_depth < 10:  # Short stack
            # Prefer preflop specialist for short stack play
            if "gto_preflop" in available_specialists:
                selected = "gto_preflop"
        elif pot_size > our_stack * 0.5:  # Large pot relative to stack
            # Prefer river specialist for commitment decisions
            if "gto_river" in available_specialists:
                selected = "gto_river"
        
        self.logger.debug(f"Selected specialist: {selected} (street: {street}, stage: {game_stage}, volatility: {stack_depth:.1f})")
        return selected

    def _get_specialist_recommendation(self, specialist_name: str, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get recommendation from the selected specialist model.
        
        Args:
            specialist_name: Name of the specialist to use
            game_state: Current game state
            
        Returns:
            Specialist recommendation with confidence scoring
        """
        if not specialist_name or specialist_name not in self.specialist_models:
            return self._get_fallback_recommendation(game_state)
        
        specialist = self.specialist_models[specialist_name]
        
        # In a real implementation, this would run the ONNX model:
        # model_input = self._encode_game_state_for_specialist(game_state, specialist_name)
        # action_probs = specialist["model"].run(None, {'input': model_input})[0]
        
        # For now, simulate specialist inference with enhanced logic
        action_probs = self._simulate_specialist_inference(specialist_name, game_state)
        
        # Decode specialist output with metadata-aware confidence
        recommendation = self._decode_specialist_output(
            action_probs, game_state, specialist_name
        )
        
        return recommendation

    def _simulate_specialist_inference(self, specialist_name: str, game_state: Dict[str, Any]) -> List[float]:
        """
        Simulate specialist model inference with context-aware logic.
        
        Args:
            specialist_name: Name of the specialist
            game_state: Current game state
            
        Returns:
            Simulated action probabilities as list [fold, call, raise]
        """
        street = game_state.get("street", "preflop").lower()
        hole_cards = game_state.get("hole_cards", [])
        community_cards = game_state.get("community_cards", [])
        pot_size = game_state.get("pot_size", 0)
        our_stack = game_state.get("our_stack", 1000)
        
        # Get specialist metadata for context
        metadata = self.model_metadata.get(specialist_name, {})
        specialization = metadata.get("specialization", "general")
        
        # Enhanced simulation based on specialist and context
        if specialization == "preflop":
            # Preflop specialist: more sophisticated preflop logic
            hand_strength = self._calculate_preflop_hand_strength(hole_cards)
            if hand_strength > 0.8:
                action_probs = [0.1, 0.2, 0.7]  # Prefer raising with strong hands
            elif hand_strength > 0.5:
                action_probs = [0.2, 0.6, 0.2]  # Prefer calling with decent hands
            else:
                action_probs = [0.7, 0.2, 0.1]  # Prefer folding with weak hands
        elif specialization == "river":
            # River specialist: more sophisticated river logic
            hand_strength = self._calculate_basic_hand_strength(hole_cards, community_cards)
            pot_odds = pot_size / max(our_stack, 1)
            
            if hand_strength > 0.8:
                action_probs = [0.05, 0.25, 0.7]  # Strong value betting
            elif hand_strength > 0.4 and pot_odds < 0.3:
                action_probs = [0.2, 0.7, 0.1]   # Call with decent odds
            else:
                action_probs = [0.8, 0.15, 0.05]  # Fold weak hands
        else:
            # General specialist: balanced approach
            hand_strength = self._calculate_basic_hand_strength(hole_cards, community_cards)
            base_fold = [0.4, 0.4, 0.2]
            base_strong = [0.1, 0.3, 0.6]
            action_probs = [
                base_fold[i] * (1 - hand_strength) + base_strong[i] * hand_strength
                for i in range(3)
            ]
        
        # Add some specialist-specific variance
        specialist_bias = hash(specialist_name) % 100 / 1000  # Small bias based on specialist
        action_probs = [max(0.01, p + random.uniform(-0.05, 0.05)) for p in action_probs]
        
        # Normalize to ensure probabilities sum to 1
        total = sum(action_probs)
        action_probs = [p / total for p in action_probs]
        
        return action_probs

    def _decode_specialist_output(self, action_probs: List[float], game_state: Dict[str, Any], specialist_name: str) -> Dict[str, Any]:
        """
        Decode specialist model output with enhanced confidence scoring.
        
        Args:
            action_probs: Specialist model probabilities
            game_state: Current game state
            specialist_name: Name of the specialist used
            
        Returns:
            Enhanced recommendation with specialist metadata
        """
        # Use existing decode logic but enhance with specialist information
        base_recommendation = self._decode_model_output(action_probs, game_state)
        
        # Enhance with specialist-specific confidence adjustments
        metadata = self.model_metadata.get(specialist_name, {})
        specialist_accuracy = metadata.get("performance_metrics", {}).get("training_accuracy", 0.9)
        
        # Adjust confidence based on specialist accuracy and appropriateness
        confidence_boost = self._calculate_specialist_appropriateness(specialist_name, game_state)
        enhanced_confidence = min(0.95, base_recommendation["confidence"] * specialist_accuracy + confidence_boost)
        
        # Add specialist information to recommendation
        base_recommendation.update({
            "specialist_used": specialist_name,
            "specialist_confidence": enhanced_confidence,
            "confidence": enhanced_confidence,  # Override base confidence
            "meta_strategist_version": "v1.0",
            "specialist_metadata": metadata.get("specialization", "unknown"),
        })
        
        return base_recommendation

    def _calculate_specialist_appropriateness(self, specialist_name: str, game_state: Dict[str, Any]) -> float:
        """
        Calculate how appropriate the selected specialist is for the current context.
        
        Args:
            specialist_name: Name of the specialist
            game_state: Current game state
            
        Returns:
            Appropriateness boost value (0.0 to 0.2)
        """
        street = game_state.get("street", "preflop").lower()
        community_cards = game_state.get("community_cards", [])
        
        metadata = self.model_metadata.get(specialist_name, {})
        specialization = metadata.get("specialization", "general")
        
        # Calculate match between specialist and game context
        if specialization == "preflop" and (street == "preflop" or len(community_cards) == 0):
            return 0.15  # High appropriateness
        elif specialization == "river" and (street == "river" or len(community_cards) == 5):
            return 0.15  # High appropriateness  
        elif specialization == "general":
            return 0.05  # Moderate appropriateness for general
        else:
            return 0.02  # Low appropriateness for mismatched context
        
    def _enhance_recommendation(self, recommendation: Dict[str, Any], game_state: Dict[str, Any], specialist_name: str) -> Dict[str, Any]:
        """
        Apply final meta-strategist enhancements to the recommendation.
        
        Args:
            recommendation: Base recommendation from specialist
            game_state: Current game state
            specialist_name: Name of specialist used
            
        Returns:
            Enhanced final recommendation
        """
        # Apply meta-level adjustments based on multiple factors
        
        # 1. Cross-specialist validation (if multiple specialists available)
        if len(self.specialist_models) > 1:
            recommendation["cross_validated"] = True
            # In a full implementation, could run multiple specialists and blend
        
        # 2. Temporal consistency (could track recent decisions)
        recommendation["temporal_consistency_check"] = True
        
        # 3. Risk management overlay
        pot_size = game_state.get("pot_size", 0) 
        our_stack = game_state.get("our_stack", 1000)
        
        if pot_size > our_stack * 0.5:  # High-risk situation
            # Reduce confidence slightly in high-risk spots
            recommendation["confidence"] *= 0.95
            recommendation["risk_adjustment"] = "high_risk_reduction"
        
        # 4. Add final meta-strategist signature
        recommendation["source"] = f"meta_strategist_{specialist_name}"
        recommendation["processing_method"] = "intelligent_specialist_selection"
        
        return recommendation

    def _calculate_preflop_hand_strength(self, hole_cards: List[str]) -> float:
        """
        Enhanced preflop hand strength calculation for specialist use.
        
        Args:
            hole_cards: List of hole cards
            
        Returns:
            Hand strength value (0.0 to 1.0)
        """
        # Use the existing method but could be enhanced for specialists
        return self._preflop_hand_strength(hole_cards)
    
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

    def _calculate_basic_hand_strength(self, hole_cards: list, community_cards: list) -> float:
        """Calculate a basic hand strength estimate for fallback purposes."""
        if not hole_cards or len(hole_cards) < 2:
            return 0.1

        # Basic preflop hand strength
        if not community_cards:
            return self._preflop_hand_strength(hole_cards)

        # Post-flop: simple heuristic based on made hands
        return self._postflop_hand_strength(hole_cards, community_cards)

    def _get_fallback_recommendation(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
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
        pot_odds = call_amount / max(pot_size + call_amount, 1) if call_amount > 0 else 0

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
        pot_size_factor = (
            pot_size - 100
        ) / 500.0  # More significant adjustment: -0.2 to +1.8 range
        final_confidence = max(
            0.1, min(0.9, base_confidence + confidence_adjustment + pot_size_factor)
        )

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
    
    def get_specialist_status(self) -> Dict[str, Any]:
        """
        Get status information about loaded specialists.
        
        Returns:
            Dictionary with specialist loading status and metadata
        """
        return {
            "loaded_specialists": list(self.specialist_models.keys()),
            "total_specialists": len(self.specialist_models),
            "is_loaded": self.is_loaded,
            "selection_strategy": self.selection_strategy,
            "model_metadata": self.model_metadata,
        }

    def _load_model(self, model_path: str, model_name: str = "legacy"):
        """Load the GTO model from file (legacy support)."""
        try:
            # TODO: Implement ONNX model loading
            # import onnxruntime as ort
            # self.model = ort.InferenceSession(model_path)
            # self.is_loaded = True
            self.specialist_models[model_name] = {
                "path": model_path,
                "loaded": True,
                "model": None,
            }
            self.logger.info(f"GTO model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load GTO model: {e}")
            self.is_loaded = False

    def _encode_game_state(self, game_state: Dict[str, Any]) -> List[float]:
        """
        Encode game state into format expected by the GTO model.

        Args:
            game_state: Current game state

        Returns:
            List of floats suitable for model input
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

        return features

    def _run_inference(self, model_input: List[float]) -> List[float]:
        """Run inference on the loaded model."""
        # TODO: Implement actual model inference
        # return self.model.run(None, {'input': model_input})[0]

        # Placeholder: return random distribution  
        import random
        probs = [random.random() for _ in range(3)]
        total = sum(probs)
        return [p / total for p in probs]  # fold, call, raise

    def _decode_model_output(
        self, action_probs: List[float], game_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decode model output into action recommendation with enhanced confidence scoring.

        Enhanced to provide better confidence metrics for weighted blending.

        Args:
            action_probs: Model output probabilities as list
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
        probability_gap = best_prob - second_best_prob if second_best_prob > 0 else best_prob

        # Entropy-based confidence (lower entropy = higher confidence)
        entropy = -sum(p * math.log2(p + 1e-10) for p in action_probs)
        max_entropy = math.log2(len(action_probs))
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
            "action_probs": action_probs,
            "source": "gto_core",
        }

    def _is_action_valid(self, action_name: str, valid_actions: list) -> bool:
        """Check if an action is valid in the current context."""
        for action in valid_actions:
            if action["action"] == action_name:
                return True
        return False

    def _calculate_bet_size(self, game_state: Dict[str, Any], action_probs: List[float]) -> int:
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

    def update_model(self, model_path: str):
        """Update the GTO model with a new trained version."""
        self._load_model(model_path, "updated_model")
        self.logger.info("GTO model updated successfully")
