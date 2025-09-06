"""
Logic Verification Tests for Project PokerMind - Phase 5 Protocol Enforcement

These tests enforce the Principle of Verifiable Logic by proving that all core modules
produce dynamic, input-dependent outputs rather than hardcoded/mock values.

Tests verify that different inputs produce different outputs, proving dynamic calculation.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.modules.gto_core import GTOCore
from agent.modules.hand_strength_estimator import HandStrengthEstimator
from agent.modules.heuristics import HeuristicsEngine
from agent.modules.synthesizer import Synthesizer
from agent.modules.opponent_modeler import OpponentModeler


class TestDynamicLogicVerification:
    """Test that all modules produce dynamic, input-dependent outputs."""

    def setup_method(self):
        """Set up test fixtures."""
        self.gto_core = GTOCore()
        self.hand_strength = HandStrengthEstimator()
        self.heuristics = HeuristicsEngine()
        self.synthesizer = Synthesizer()
        self.opponent_modeler = OpponentModeler()

    def test_gto_core_dynamic_responses(self):
        """Test that GTOCore produces different outputs for different inputs."""
        # Create two distinctly different game states
        state1 = {
            "hole_cards": ["As", "Ks"],  # Premium hand
            "community_cards": [],
            "pot_size": 100,
            "street": "preflop",
            "valid_actions": [
                {"action": "fold", "amount": 0},
                {"action": "call", "amount": 50},
                {"action": "raise", "amount": {"min": 100, "max": 300}},
            ],
            "our_stack": 1000,
            "position": "button",
        }

        state2 = {
            "hole_cards": ["2c", "7h"],  # Weak hand
            "community_cards": ["Ks", "Qs", "Jd"],  # Board doesn't help
            "pot_size": 500,
            "street": "flop",
            "valid_actions": [
                {"action": "fold", "amount": 0},
                {"action": "call", "amount": 200},
                {"action": "raise", "amount": {"min": 400, "max": 1000}},
            ],
            "our_stack": 800,
            "position": "early",
        }

        # Get recommendations from both states
        rec1 = self.gto_core.get_recommendation(state1)
        rec2 = self.gto_core.get_recommendation(state2)

        # Verify we got valid responses
        assert rec1 is not None, "GTO Core should return a recommendation"
        assert rec2 is not None, "GTO Core should return a recommendation"
        assert "action" in rec1, "GTO recommendation should have 'action'"
        assert "confidence" in rec1, "GTO recommendation should have 'confidence'"
        assert "action" in rec2, "GTO recommendation should have 'action'"
        assert "confidence" in rec2, "GTO recommendation should have 'confidence'"

        # Verify dynamic behavior - different inputs should produce different outputs
        # At minimum, confidence or action should differ for these vastly different scenarios
        dynamic_behavior = (
            rec1["action"] != rec2["action"] or abs(rec1["confidence"] - rec2["confidence"]) > 0.1
        )
        assert dynamic_behavior, (
            f"GTO Core shows hardcoded behavior: "
            f"State1({state1['hole_cards']}, {state1['street']}) -> {rec1}, "
            f"State2({state2['hole_cards']}, {state2['street']}) -> {rec2}. "
            f"Different inputs should produce meaningfully different outputs."
        )

    def test_hand_strength_estimator_dynamic_responses(self):
        """Test that HandStrengthEstimator produces different outputs for different inputs."""
        # Strong hand scenario
        state1 = {
            "hole_cards": ["As", "Ad"],  # Pocket aces
            "community_cards": ["Ac", "2d", "7h"],  # Set of aces
            "street": "flop",
        }

        # Weak hand scenario
        state2 = {
            "hole_cards": ["2c", "3d"],  # Weak starting hand
            "community_cards": ["Ks", "Qs", "Jh"],  # No help
            "street": "flop",
        }

        # Get strength estimates
        est1 = self.hand_strength.estimate(state1)
        est2 = self.hand_strength.estimate(state2)

        # Verify responses
        assert est1 is not None, "Hand strength estimator should return a result"
        assert est2 is not None, "Hand strength estimator should return a result"
        assert "strength" in est1, "Should have 'strength' key"
        assert "confidence" in est1, "Should have 'confidence' key"
        assert "strength" in est2, "Should have 'strength' key"
        assert "confidence" in est2, "Should have 'confidence' key"

        # Verify dynamic behavior - set of aces should be much stronger than 2-3 offsuit
        strength_diff = est1["strength"] - est2["strength"]
        assert strength_diff > 0.3, (
            f"Hand Strength Estimator shows hardcoded behavior: "
            f"Set of Aces -> {est1['strength']:.3f}, "
            f"Weak hand -> {est2['strength']:.3f}. "
            f"Difference of {strength_diff:.3f} is too small. Expected >0.3"
        )

    def test_heuristics_engine_dynamic_responses(self):
        """Test that HeuristicsEngine produces different outputs for different inputs."""
        # Obvious fold scenario (very short stack, facing big bet)
        state1 = {
            "hole_cards": ["2c", "7h"],
            "community_cards": ["Ks", "Qs", "Jd", "9h", "8c"],
            "pot_size": 1000,
            "street": "river",
            "valid_actions": [
                {"action": "fold", "amount": 0},
                {"action": "call", "amount": 800},
            ],
            "our_stack": 100,  # Very short
        }

        # Obvious call scenario (nuts with small bet)
        state2 = {
            "hole_cards": ["As", "Ks"],
            "community_cards": ["Ad", "Kd", "Ah", "Kh", "Ac"],  # Full house aces full
            "pot_size": 200,
            "street": "river",
            "valid_actions": [
                {"action": "fold", "amount": 0},
                {"action": "call", "amount": 50},
            ],
            "our_stack": 1000,
        }

        # Get heuristic recommendations
        heur1 = self.heuristics.get_recommendation(state1)
        heur2 = self.heuristics.get_recommendation(state2)

        # For obvious situations, heuristics should have high confidence recommendations
        assert heur1 is not None, "Heuristics should return a recommendation"
        assert heur2 is not None, "Heuristics should return a recommendation"

        # At least one should have a clear recommendation with high confidence
        has_strong_recommendation = (heur1.get("confidence", 0) > 0.7) or (
            heur2.get("confidence", 0) > 0.7
        )
        assert has_strong_recommendation, (
            f"Heuristics Engine shows hardcoded behavior: "
            f"Obvious situations should produce high confidence recommendations. "
            f"Got: state1 confidence={heur1.get('confidence', 0):.3f}, "
            f"state2 confidence={heur2.get('confidence', 0):.3f}"
        )

    def test_synthesizer_dynamic_decision_making(self):
        """Test that Synthesizer produces different outputs based on different inputs."""
        # Create mock System 1 inputs with different confidence patterns

        # Scenario 1: High GTO confidence, low exploit confidence
        inputs1 = {
            "gto": {
                "action": "call",
                "amount": 100,
                "confidence": 0.9,
                "source": "gto",
            },
            "heuristics": {
                "recommendation": None,
                "confidence": 0.0,
                "source": "heuristics",
            },
            "hand_strength": {
                "strength": 0.6,
                "confidence": 0.7,
                "source": "hand_strength",
            },
            "opponents": {
                "recommendation": "fold",
                "confidence": 0.3,
                "source": "opponents",
            },
        }

        # Scenario 2: Low GTO confidence, high exploit confidence
        inputs2 = {
            "gto": {
                "action": "call",
                "amount": 100,
                "confidence": 0.2,
                "source": "gto",
            },
            "heuristics": {
                "recommendation": "raise",
                "amount": 200,
                "confidence": 0.9,
                "source": "heuristics",
            },
            "hand_strength": {
                "strength": 0.8,
                "confidence": 0.9,
                "source": "hand_strength",
            },
            "opponents": {
                "recommendation": "raise",
                "confidence": 0.8,
                "source": "opponents",
            },
        }

        game_state = {
            "pot_size": 200,
            "valid_actions": [
                {"action": "fold", "amount": 0},
                {"action": "call", "amount": 100},
                {"action": "raise", "amount": {"min": 200, "max": 500}},
            ],
        }

        # Get synthesized decisions - fix parameter order and handle tuple return
        result1 = self.synthesizer.synthesize_decision(game_state, inputs1)
        result2 = self.synthesizer.synthesize_decision(game_state, inputs2)

        # Extract decisions from results (they might be tuples)
        if isinstance(result1, tuple):
            decision1 = result1[0]
        else:
            decision1 = result1

        if isinstance(result2, tuple):
            decision2 = result2[0]
        else:
            decision2 = result2

        # Verify responses
        assert decision1 is not None, "Synthesizer should return a decision"
        assert decision2 is not None, "Synthesizer should return a decision"
        assert "action" in decision1, "Decision should have 'action'"
        assert "confidence" in decision1, "Decision should have 'confidence'"
        assert "action" in decision2, "Decision should have 'action'"
        assert "confidence" in decision2, "Decision should have 'confidence'"

        # Verify dynamic behavior - different confidence patterns should lead to different decisions
        # The synthesizer should weight high-confidence inputs more heavily
        dynamic_behavior = (
            decision1["action"] != decision2["action"]
            or abs(decision1["confidence"] - decision2["confidence"]) > 0.1
        )

        assert dynamic_behavior, (
            f"Synthesizer shows hardcoded behavior: "
            f"High GTO confidence -> {decision1}, "
            f"High exploit confidence -> {decision2}. "
            f"Different confidence patterns should produce different decisions."
        )

    def test_opponent_modeler_dynamic_profiles(self):
        """Test that OpponentModeler produces different profiles for different players."""
        # Create two different player profiles through different action patterns

        # Tight player - mostly folds
        tight_actions = [
            ("fold", 0, "preflop"),
            ("fold", 0, "preflop"),
            ("fold", 0, "preflop"),
            ("fold", 0, "preflop"),
            ("call", 50, "preflop"),  # Only plays 1 in 5 hands
        ]

        # Loose player - plays most hands
        loose_actions = [
            ("call", 50, "preflop"),
            ("raise", 100, "preflop"),
            ("call", 50, "preflop"),
            ("call", 50, "preflop"),
            ("raise", 150, "preflop"),  # Plays 5 out of 5 hands
        ]

        # Update opponent models
        for action, amount, street in tight_actions:
            self.opponent_modeler.update("TightPlayer", action, amount, street, 100)

        for action, amount, street in loose_actions:
            self.opponent_modeler.update("LoosePlayer", action, amount, street, 100)

        # Get profiles (may be None if insufficient data, that's OK)
        tight_profile = self.opponent_modeler.get_profile("TightPlayer")
        loose_profile = self.opponent_modeler.get_profile("LoosePlayer")

        # Check for dynamic behavior in the statistics
        tight_stats = self.opponent_modeler.get_basic_stats("TightPlayer")
        loose_stats = self.opponent_modeler.get_basic_stats("LoosePlayer")

        assert tight_stats is not None, "Should have basic stats for tight player"
        assert loose_stats is not None, "Should have basic stats for loose player"

        # VPIP should be different
        tight_vpip = tight_stats.get("vpip", 0.5)  # Default if not found
        loose_vpip = loose_stats.get("vpip", 0.5)

        # Verify dynamic behavior
        vpip_difference = abs(loose_vpip - tight_vpip)
        assert vpip_difference > 0.3, (
            f"Opponent Modeler shows hardcoded behavior: "
            f"Tight player VPIP: {tight_vpip:.3f}, "
            f"Loose player VPIP: {loose_vpip:.3f}. "
            f"Difference of {vpip_difference:.3f} is too small. Expected >0.3"
        )

    def test_no_hardcoded_constants_in_responses(self):
        """Test that modules don't return the same hardcoded values repeatedly."""
        # Run the same input through modules multiple times with slight variations
        base_state = {
            "hole_cards": ["As", "Ks"],
            "community_cards": [],
            "pot_size": 100,
            "street": "preflop",
            "valid_actions": [
                {"action": "fold", "amount": 0},
                {"action": "call", "amount": 50},
                {"action": "raise", "amount": {"min": 100, "max": 300}},
            ],
            "our_stack": 1000,
        }

        responses = []

        # Test with slight variations
        for pot_variation in [90, 100, 110, 120, 130]:
            state = base_state.copy()
            state["pot_size"] = pot_variation

            # Get responses from different modules
            gto_rec = self.gto_core.get_recommendation(state)
            hand_est = self.hand_strength.estimate(state)

            responses.append(
                {
                    "pot_size": pot_variation,
                    "gto_confidence": gto_rec.get("confidence", 0),
                    "hand_strength": hand_est.get("strength", 0),
                }
            )

        # Check for variation in responses
        gto_confidences = [r["gto_confidence"] for r in responses]
        hand_strengths = [r["hand_strength"] for r in responses]

        # At least some variation should occur (not all identical)
        gto_has_variation = len(set(gto_confidences)) > 1
        hand_has_variation = len(set(hand_strengths)) > 1

        # We should see some dynamic behavior
        has_dynamic_behavior = gto_has_variation or hand_has_variation

        assert has_dynamic_behavior, (
            f"Modules show hardcoded behavior across pot size variations: "
            f"GTO confidences: {gto_confidences}, "
            f"Hand strengths: {hand_strengths}. "
            f"Expected some variation in responses to different inputs."
        )
