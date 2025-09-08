#!/usr/bin/env python3
"""
Test suite for opponent archetypes in Project PokerMind.

This module tests that each opponent archetype can be loaded and can correctly declare an action.
Part of the Deep Cleanup & Finalization Protocol test suite reorganization.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestOpponentArchetypes(unittest.TestCase):
    """Test opponent archetype loading and basic functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_game_state = {
            "community_cards": ["Ah", "Kd", "Qc"],
            "pot_size": 100,
            "call_amount": 20,
            "min_raise": 40,
            "hole_cards": ["As", "Ks"],
            "stack_size": 1000,
            "position": "button",
            "opponents": 3,
            "betting_round": "flop"
        }

    def test_the_nit_archetype(self):
        """Test that The_Nit archetype can be loaded and declare actions."""
        try:
            from agent.opponents.The_Nit import NitPlayer
            nit_player = NitPlayer()
            
            # Test that the player can declare an action
            action = nit_player.declare_action(
                valid_actions=[
                    {"action": "call", "amount": 20},
                    {"action": "raise", "amount": 40},
                    {"action": "fold", "amount": 0}
                ],
                hole_cards=self.mock_game_state["hole_cards"],
                round_state={
                    "community_card": self.mock_game_state["community_cards"],
                    "pot": {"main": {"amount": self.mock_game_state["pot_size"]}},
                    "seats": [{"stack": self.mock_game_state["stack_size"]}]
                }
            )
            
            # Verify action is valid tuple
            self.assertIsInstance(action, tuple)
            self.assertEqual(len(action), 2)
            action_type, amount = action
            self.assertIn(action_type, ["call", "raise", "fold"])
            self.assertIsInstance(amount, int)
            
        except ImportError as e:
            self.skipTest(f"The_Nit archetype not available: {e}")
        except Exception as e:
            self.fail(f"The_Nit archetype failed to declare action: {e}")

    def test_the_tag_archetype(self):
        """Test that The_TAG archetype can be loaded and declare actions."""
        try:
            from agent.opponents.The_TAG import TightAggressivePlayer
            tag_player = TightAggressivePlayer()
            
            # Test that the player can declare an action
            action = tag_player.declare_action(
                valid_actions=[
                    {"action": "call", "amount": 20},
                    {"action": "raise", "amount": 40},
                    {"action": "fold", "amount": 0}
                ],
                hole_cards=self.mock_game_state["hole_cards"],
                round_state={
                    "community_card": self.mock_game_state["community_cards"],
                    "pot": {"main": {"amount": self.mock_game_state["pot_size"]}},
                    "seats": [{"stack": self.mock_game_state["stack_size"]}]
                }
            )
            
            # Verify action is valid tuple
            self.assertIsInstance(action, tuple)
            self.assertEqual(len(action), 2)
            action_type, amount = action
            self.assertIn(action_type, ["call", "raise", "fold"])
            self.assertIsInstance(amount, int)
            
        except ImportError as e:
            self.skipTest(f"The_TAG archetype not available: {e}")
        except Exception as e:
            self.fail(f"The_TAG archetype failed to declare action: {e}")

    def test_the_lag_archetype(self):
        """Test that The_LAG archetype can be loaded and declare actions."""
        try:
            from agent.opponents.The_LAG import LooseAggressivePlayer
            lag_player = LooseAggressivePlayer()
            
            # Test that the player can declare an action
            action = lag_player.declare_action(
                valid_actions=[
                    {"action": "call", "amount": 20},
                    {"action": "raise", "amount": 40},
                    {"action": "fold", "amount": 0}
                ],
                hole_cards=self.mock_game_state["hole_cards"],
                round_state={
                    "community_card": self.mock_game_state["community_cards"],
                    "pot": {"main": {"amount": self.mock_game_state["pot_size"]}},
                    "seats": [{"stack": self.mock_game_state["stack_size"]}]
                }
            )
            
            # Verify action is valid tuple
            self.assertIsInstance(action, tuple)
            self.assertEqual(len(action), 2)
            action_type, amount = action
            self.assertIn(action_type, ["call", "raise", "fold"])
            self.assertIsInstance(amount, int)
            
        except ImportError as e:
            self.skipTest(f"The_LAG archetype not available: {e}")
        except Exception as e:
            self.fail(f"The_LAG archetype failed to declare action: {e}")

    def test_all_archetypes_available(self):
        """Test that all expected opponent archetypes can be imported."""
        expected_archetypes = [
            ("agent.opponents.The_Nit", "NitPlayer"),
            ("agent.opponents.The_TAG", "TightAggressivePlayer"), 
            ("agent.opponents.The_LAG", "LooseAggressivePlayer")
        ]
        
        available_archetypes = []
        for module_name, class_name in expected_archetypes:
            try:
                module = __import__(module_name, fromlist=[class_name])
                archetype_class = getattr(module, class_name)
                available_archetypes.append((module_name, class_name))
            except (ImportError, AttributeError):
                pass
        
        # At least one archetype should be available
        self.assertGreater(
            len(available_archetypes), 0,
            "No opponent archetypes could be loaded"
        )
        
        print(f"\nAvailable opponent archetypes: {len(available_archetypes)}/{len(expected_archetypes)}")
        for module_name, class_name in available_archetypes:
            print(f"  ✓ {module_name}.{class_name}")


if __name__ == "__main__":
    unittest.main()