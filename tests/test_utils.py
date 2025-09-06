"""
Unit tests for the Project PokerMind utilities module.

Tests cover card vectorization, pot odds calculation, and other utility functions.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the agent module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.utils import (
    vectorize_cards,
    calculate_pot_odds,
    parse_card,
    validate_card_format,
    get_card_strength_value,
    estimate_preflop_hand_strength,
    cards_to_readable_string,
    RANKS,
    SUITS,
)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for card processing and calculations."""

    def test_vectorize_cards_basic(self):
        """Test basic card vectorization functionality."""
        hole_cards = ["Ah", "Kd"]
        community_cards = ["Qc", "Js", "Th"]

        vector = vectorize_cards(hole_cards, community_cards)

        # Should be 104-element vector
        self.assertEqual(len(vector), 104)

        # Should be numpy array of floats
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.dtype, np.float32)

        # Should contain only 0s and 1s
        unique_values = np.unique(vector)
        self.assertTrue(all(val in [0.0, 1.0] for val in unique_values))

        # Should have exactly 5 ones (2 hole + 3 community)
        self.assertEqual(np.sum(vector), 5.0)

    def test_vectorize_cards_ace_of_hearts(self):
        """Test that Ace of Hearts is vectorized correctly."""
        hole_cards = ["Ah"]
        vector = vectorize_cards(hole_cards, [])

        # Ah should be at index: A(12) * 4 + h(0) = 48
        ace_hearts_index = 12 * 4 + 0  # A is index 12 in RANKS, h is index 0 in SUITS
        self.assertEqual(vector[ace_hearts_index], 1.0)

        # Should have exactly 1 one
        self.assertEqual(np.sum(vector), 1.0)

    def test_vectorize_cards_community_section(self):
        """Test that community cards go in the second half of the vector."""
        community_cards = ["2c"]  # 2 of clubs
        vector = vectorize_cards([], community_cards)

        # 2c should be at community index: 52 + (0 * 4 + 2) = 54
        two_clubs_community_index = 52 + (0 * 4 + 2)
        self.assertEqual(vector[two_clubs_community_index], 1.0)

        # Should have exactly 1 one
        self.assertEqual(np.sum(vector), 1.0)

    def test_vectorize_cards_empty_input(self):
        """Test vectorization with empty inputs."""
        vector = vectorize_cards([], [])

        self.assertEqual(len(vector), 104)
        self.assertEqual(np.sum(vector), 0.0)

    def test_vectorize_cards_invalid_format(self):
        """Test vectorization with invalid card formats."""
        # This should not crash, but should warn and skip invalid cards
        hole_cards = ["Ah", "Zx", ""]  # Zx and empty string are invalid
        vector = vectorize_cards(hole_cards, [])

        # Should only have 1 one (for Ah)
        self.assertEqual(np.sum(vector), 1.0)

    def test_calculate_pot_odds_basic(self):
        """Test basic pot odds calculation."""
        # Call $50 into $100 pot -> need 50/(100+50) = 1/3 = 0.333... equity
        required_equity = calculate_pot_odds(100, 50)
        self.assertAlmostEqual(required_equity, 1 / 3, places=5)

    def test_calculate_pot_odds_edge_cases(self):
        """Test pot odds calculation edge cases."""
        # No cost to call
        self.assertEqual(calculate_pot_odds(100, 0), 0.0)

        # Zero pot size - need 100% equity to call (all money goes to call)
        self.assertEqual(calculate_pot_odds(0, 50), 1.0)

        # Negative call amount (shouldn't happen but handle gracefully)
        self.assertEqual(calculate_pot_odds(100, -10), 0.0)

    def test_calculate_pot_odds_common_scenarios(self):
        """Test pot odds for common poker scenarios."""
        # Half pot bet: call 50 into 100 -> 50/150 = 1/3
        self.assertAlmostEqual(calculate_pot_odds(100, 50), 1 / 3, places=5)

        # Pot-sized bet: call 100 into 100 -> 100/200 = 1/2
        self.assertAlmostEqual(calculate_pot_odds(100, 100), 0.5, places=5)

        # Quarter pot bet: call 25 into 100 -> 25/125 = 1/5
        self.assertAlmostEqual(calculate_pot_odds(100, 25), 0.2, places=5)

    def test_parse_card_valid(self):
        """Test parsing valid card strings."""
        self.assertEqual(parse_card("Ah"), ("A", "h"))
        self.assertEqual(parse_card("Kd"), ("K", "d"))
        self.assertEqual(parse_card("Ts"), ("T", "s"))
        self.assertEqual(parse_card("2c"), ("2", "c"))

    def test_parse_card_invalid(self):
        """Test parsing invalid card strings."""
        self.assertEqual(parse_card(""), ("", ""))
        self.assertEqual(parse_card("A"), ("", ""))
        self.assertEqual(parse_card("X"), ("", ""))

    def test_validate_card_format(self):
        """Test card format validation."""
        # Valid cards
        self.assertTrue(validate_card_format("Ah"))
        self.assertTrue(validate_card_format("Kd"))
        self.assertTrue(validate_card_format("2c"))
        self.assertTrue(validate_card_format("Ts"))

        # Invalid cards
        self.assertFalse(validate_card_format("Zx"))  # Invalid rank
        self.assertFalse(validate_card_format("Ax"))  # Invalid suit
        self.assertFalse(validate_card_format("A"))  # Too short
        self.assertFalse(validate_card_format(""))  # Empty

    def test_get_card_strength_value(self):
        """Test card strength value calculation."""
        self.assertEqual(get_card_strength_value("2h"), 2)
        self.assertEqual(get_card_strength_value("3d"), 3)
        self.assertEqual(get_card_strength_value("Tc"), 10)
        self.assertEqual(get_card_strength_value("Jh"), 11)
        self.assertEqual(get_card_strength_value("Qd"), 12)
        self.assertEqual(get_card_strength_value("Ks"), 13)
        self.assertEqual(get_card_strength_value("Ah"), 14)

        # Invalid cards
        self.assertEqual(get_card_strength_value("Zx"), 0)
        self.assertEqual(get_card_strength_value(""), 0)

    def test_estimate_preflop_hand_strength(self):
        """Test preflop hand strength estimation."""
        # Pocket Aces should be very strong
        aa_strength = estimate_preflop_hand_strength(["Ah", "Ad"])
        self.assertGreater(aa_strength, 0.9)

        # Pocket Kings should be very strong but less than Aces
        kk_strength = estimate_preflop_hand_strength(["Kh", "Kd"])
        self.assertGreater(kk_strength, 0.85)
        self.assertLess(kk_strength, aa_strength)

        # AK suited should be strong
        ak_suited_strength = estimate_preflop_hand_strength(["Ah", "Kh"])
        self.assertGreater(ak_suited_strength, 0.7)

        # AK offsuit should be strong but less than suited
        ak_offsuit_strength = estimate_preflop_hand_strength(["Ah", "Kd"])
        self.assertGreater(ak_offsuit_strength, 0.65)
        self.assertLess(ak_offsuit_strength, ak_suited_strength)

        # 72 offsuit should be very weak
        weak_strength = estimate_preflop_hand_strength(["7h", "2d"])
        self.assertLess(weak_strength, 0.3)

        # Invalid input
        invalid_strength = estimate_preflop_hand_strength(["Ah"])
        self.assertEqual(invalid_strength, 0.1)

    def test_cards_to_readable_string(self):
        """Test conversion of cards to readable string."""
        cards = ["Ah", "Kd", "Qc", "Js"]
        readable = cards_to_readable_string(cards)

        # Should contain suit symbols
        self.assertIn("♥", readable)  # Hearts
        self.assertIn("♦", readable)  # Diamonds
        self.assertIn("♣", readable)  # Clubs
        self.assertIn("♠", readable)  # Spades

        # Should contain all ranks
        self.assertIn("A", readable)
        self.assertIn("K", readable)
        self.assertIn("Q", readable)
        self.assertIn("J", readable)

    def test_constants(self):
        """Test that constants are properly defined."""
        self.assertEqual(len(RANKS), 13)
        self.assertEqual(len(SUITS), 4)

        self.assertIn("A", RANKS)
        self.assertIn("K", RANKS)
        self.assertIn("2", RANKS)

        self.assertIn("h", SUITS)
        self.assertIn("d", SUITS)
        self.assertIn("c", SUITS)
        self.assertIn("s", SUITS)


if __name__ == "__main__":
    # Set up basic logging to avoid warnings during testing
    import logging

    logging.basicConfig(level=logging.WARNING)

    unittest.main()
