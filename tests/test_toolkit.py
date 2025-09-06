"""
Test cases for the professional strategic toolkit.

These tests verify the functionality of the new toolkit modules
created for multi-player poker analysis.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add the parent directory to the path so we can import the toolkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.toolkit.equity_calculator import EquityCalculator
from agent.toolkit.range_modeler import RangeModeler  
from agent.toolkit.board_analyzer import BoardAnalyzer
from agent.toolkit.post_game_analyzer import PostGameAnalyzer
from agent.toolkit.gto_tools import (
    count_combos, 
    calculate_mdf,
    calculate_pot_equity_needed,
    calculate_spr
)


class TestEquityCalculator(unittest.TestCase):
    """Test the professional equity calculator."""
    
    def setUp(self):
        self.calc = EquityCalculator(simulation_iterations=100)  # Small for testing
        
    def test_initialization(self):
        """Test equity calculator initialization."""
        self.assertEqual(self.calc.simulation_iterations, 100)
        self.assertEqual(len(self.calc.full_deck), 52)
        
    def test_multi_way_equity_calculation(self):
        """Test basic multi-way equity calculation."""
        our_cards = ["As", "Ks"]
        community_cards = ["Qh", "Jd", "Tc"]
        opponent_ranges = [
            {"type": "tight"}, 
            {"type": "loose"}
        ]  # 2 opponents
        
        result = self.calc.calculate_multi_way_equity(
            our_cards, community_cards, opponent_ranges, "flop"
        )
        
        self.assertIn("equity", result)
        self.assertIn("win_percentage", result)
        self.assertIn("opponents", result)
        self.assertEqual(result["opponents"], 2)
        self.assertGreaterEqual(result["equity"], 0.0)
        self.assertLessEqual(result["equity"], 1.0)
        
    def test_pot_equity_needed(self):
        """Test pot equity needed calculation."""
        equity_needed = self.calc.calculate_pot_equity_needed(100, 50, 2)
        
        # With 2 opponents, should need more than basic pot odds
        basic_pot_odds = 50 / 150  # 0.333
        self.assertGreater(equity_needed, basic_pot_odds)


class TestRangeModeler(unittest.TestCase):
    """Test the professional range modeler."""
    
    def setUp(self):
        self.modeler = RangeModeler()
        
    def test_range_parsing(self):
        """Test standard range notation parsing."""
        # Test pocket pairs
        aa_combos = self.modeler.parse_range_notation("AA")
        self.assertEqual(len(aa_combos), 6)  # 6 combinations of AA
        
        # Test suited hands
        aks_combos = self.modeler.parse_range_notation("AKs")
        self.assertEqual(len(aks_combos), 4)  # 4 suited combinations
        
        # Test offsuit hands
        ako_combos = self.modeler.parse_range_notation("AKo")
        self.assertEqual(len(ako_combos), 12)  # 12 offsuit combinations
        
    def test_position_ranges(self):
        """Test position-based range generation."""
        utg_range = self.modeler.get_position_range("UTG", "open")
        btn_range = self.modeler.get_position_range("BTN", "open")
        
        self.assertIsNotNone(utg_range)
        self.assertIsNotNone(btn_range)
        self.assertEqual(utg_range.position, "UTG")
        self.assertEqual(btn_range.position, "BTN")
        
        # Button should have more hands than UTG
        self.assertGreater(btn_range.total_combos, utg_range.total_combos)


class TestBoardAnalyzer(unittest.TestCase):
    """Test the advanced board analyzer."""
    
    def setUp(self):
        self.analyzer = BoardAnalyzer()
        
    def test_board_texture_analysis(self):
        """Test board texture analysis."""
        # Dry board
        dry_board = ["As", "7h", "2c"]
        dry_analysis = self.analyzer.analyze_board_texture(dry_board)
        
        self.assertIn("texture_category", dry_analysis)
        self.assertIn("wetness_score", dry_analysis)
        self.assertIn("connectivity", dry_analysis)
        
        # Wet board
        wet_board = ["9s", "8s", "7h"]
        wet_analysis = self.analyzer.analyze_board_texture(wet_board)
        
        # Wet board should have higher wetness score
        self.assertGreater(
            wet_analysis["wetness_score"], 
            dry_analysis["wetness_score"]
        )
        
    def test_range_advantage_analysis(self):
        """Test range advantage calculation."""
        board = ["Ah", "Kc", "Qd"]
        
        advantage = self.analyzer.who_has_range_advantage(
            board, "BTN", "BB"
        )
        
        self.assertIn("advantage_holder", advantage)
        self.assertIn("advantage_strength", advantage)
        self.assertIn("recommendation", advantage)


class TestGTOTools(unittest.TestCase):
    """Test the professional GTO tools."""
    
    def test_calculate_mdf(self):
        """Test Minimum Defense Frequency calculation."""
        # Half pot bet should require 33% defense frequency
        mdf = calculate_mdf(50, 100)
        self.assertAlmostEqual(mdf, 0.667, places=2)
        
        # Pot sized bet should require 50% defense frequency
        mdf = calculate_mdf(100, 100)
        self.assertAlmostEqual(mdf, 0.5, places=2)
        
    def test_calculate_spr(self):
        """Test Stack-to-Pot Ratio calculation."""
        # 1000 stack with 100 pot = 10 SPR
        spr = calculate_spr(1000, 100)
        self.assertEqual(spr, 10.0)
        
        # Zero pot should return infinity
        spr = calculate_spr(1000, 0)
        self.assertEqual(spr, float('inf'))
        
    def test_pot_equity_needed(self):
        """Test pot equity calculation."""
        # Half pot bet = 33.33% equity needed
        equity = calculate_pot_equity_needed(100, 50)
        self.assertAlmostEqual(equity, 0.333, places=2)
        
        # Full pot bet = 50% equity needed
        equity = calculate_pot_equity_needed(100, 100)
        self.assertAlmostEqual(equity, 0.5, places=2)


class TestPostGameAnalyzer(unittest.TestCase):
    """Test the post-game analyzer."""
    
    def setUp(self):
        self.analyzer = PostGameAnalyzer()
        
    def test_leak_analysis(self):
        """Test leak detection analysis."""
        # Mock session logs
        session_logs = [
            {
                "final_action": {"action": "fold"},
                "confidence_score": 0.8,
                "street": "preflop",
                "seats": [
                    {"seat_id": 1, "name": "us"},
                    {"seat_id": 2, "name": "opponent1"},
                    {"seat_id": 3, "name": "opponent2"}
                ],
                "our_seat_id": 1,
                "pot_size": 100,
                "community_cards": []
            },
            {
                "final_action": {"action": "call", "amount": 50},
                "confidence_score": 0.6,
                "street": "flop",
                "seats": [
                    {"seat_id": 1, "name": "us"},
                    {"seat_id": 2, "name": "opponent1"}
                ],
                "our_seat_id": 1,
                "pot_size": 150,
                "community_cards": ["Ah", "Kc", "Qd"]
            }
        ]
        
        leak_analysis = self.analyzer.find_my_leaks(session_logs)
        
        self.assertIn("session_overview", leak_analysis)
        self.assertIn("priority_improvements", leak_analysis)
        self.assertIn("confidence", leak_analysis)
        
    def test_improvement_report_generation(self):
        """Test improvement report generation."""
        # Mock leak analysis
        leak_analysis = {
            "session_overview": {"total_hands": 100, "average_confidence": 0.7},
            "priority_improvements": [
                {
                    "category": "positional",
                    "description": "Over-folding in late position",
                    "priority_score": 15
                }
            ],
            "confidence": 0.8
        }
        
        report = self.analyzer.generate_improvement_report(leak_analysis)
        
        self.assertIsInstance(report, str)
        self.assertIn("POKERMIND SELF-IMPROVEMENT ANALYSIS", report)
        self.assertIn("PRIORITY IMPROVEMENTS", report)


if __name__ == "__main__":
    unittest.main()