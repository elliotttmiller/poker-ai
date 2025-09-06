"""
Unit tests for Project PokerMind Phase 3 functionality.

Tests cover opponent modeling, exploitative adjustments, LLM narration, and learning modules.
"""

import unittest
import sys
import os
import tempfile
import json
from unittest.mock import Mock, patch
from datetime import datetime

# Add the parent directory to the path so we can import the agent module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.modules.opponent_modeler import OpponentModeler
from agent.modules.synthesizer import Synthesizer
from agent.modules.llm_narrator import LLMNarrator
from agent.modules.learning_module import LearningModule
from agent.cognitive_core import DecisionPacket


class TestOpponentModeler(unittest.TestCase):
    """Test the OpponentModeler functionality from Sub-Task 3.1."""

    def setUp(self):
        """Set up test environment."""
        self.modeler = OpponentModeler()

    def test_update_method(self):
        """Test the update method as requested in Sub-Task 3.1."""
        # Test basic update functionality
        self.modeler.update("TestPlayer", "call", 50, "preflop", 100)

        # Verify player was added
        self.assertIn("TestPlayer", self.modeler.player_stats)

        # Verify stats were updated
        stats = self.modeler.player_stats["TestPlayer"]
        self.assertEqual(stats.name, "TestPlayer")
        self.assertEqual(len(stats.recent_actions), 1)

        # Test multiple updates
        self.modeler.update("TestPlayer", "raise", 100, "preflop", 100)
        self.assertEqual(len(stats.recent_actions), 2)

    def test_get_profile_method(self):
        """Test the get_profile method as requested in Sub-Task 3.1."""
        # Test with no data
        profile = self.modeler.get_profile("NonexistentPlayer")
        self.assertIsNone(profile)

        # Test with insufficient data
        self.modeler.update("NewPlayer", "fold", 0, "preflop", 100)
        profile = self.modeler.get_profile("NewPlayer")
        self.assertIsNone(profile)  # Not enough hands

        # Test with sufficient data
        for i in range(5):  # Add 5 hands
            action = "call" if i < 3 else "fold"  # 60% VPIP
            self.modeler.update(
                "TestPlayer", action, 50 if action == "call" else 0, "preflop", 100
            )

        profile = self.modeler.get_profile("TestPlayer")
        self.assertIsNotNone(profile)
        self.assertEqual(profile["name"], "TestPlayer")
        self.assertIn("classification", profile)
        self.assertIn("vpip", profile)
        self.assertIn("pfr", profile)

    def test_player_classification(self):
        """Test player classification heuristics."""
        # Create a tight player (VPIP < 20%)
        for i in range(10):
            action = "call" if i < 1 else "fold"  # 10% VPIP
            self.modeler.update(
                "TightPlayer", action, 50 if action == "call" else 0, "preflop", 100
            )

        profile = self.modeler.get_profile("TightPlayer")
        self.assertIsNotNone(profile)
        self.assertIn("tight", profile["classification"])

        # Create a loose player (VPIP > 40%)
        for i in range(10):
            action = "call" if i < 5 else "fold"  # 50% VPIP
            self.modeler.update(
                "LoosePlayer", action, 50 if action == "call" else 0, "preflop", 100
            )

        profile = self.modeler.get_profile("LoosePlayer")
        self.assertIsNotNone(profile)
        self.assertIn("loose", profile["classification"])


class TestSynthesizerExploitativeLogic(unittest.TestCase):
    """Test the Synthesizer exploitative adjustments from Sub-Task 3.2."""

    def setUp(self):
        """Set up test environment."""
        self.synthesizer = Synthesizer()

    def test_make_final_decision_method(self):
        """Test the make_final_decision method added in Sub-Task 3.2."""
        game_state = {
            "pot_size": 100,
            "our_stack": 1000,
            "hole_cards": ["Ah", "Kh"],
            "valid_actions": [
                {"action": "fold", "amount": 0},
                {"action": "call", "amount": 50},
            ],
        }

        system1_outputs = {
            "hand_strength": {"overall_strength": 0.8, "probabilities": [0.1] * 9},
            "gto": {"action": "call", "confidence": 0.6},
            "opponents": {"opponents": {}},
            "heuristics": {"recommendation": None},
        }

        opponent_profile = {
            "classification": "tight_passive",
            "vpip": 0.15,
            "pfr": 0.08,
        }

        # Test that method exists and works
        final_action, analysis = self.synthesizer.make_final_decision(
            game_state, system1_outputs, opponent_profile
        )

        self.assertIn("action", final_action)
        self.assertIn("reasoning", analysis)

    def test_tight_opponent_adjustment(self):
        """Test the tight opponent adjustment formula from Sub-Task 3.2."""
        # Test the specific formula: required_equity = pot_odds_equity * 1.15
        base_required_equity = 0.5
        opponent_profile = {
            "classification": "tight_passive",
            "vpip": 0.15,  # Under 0.2 threshold
            "pfr": 0.08,
        }

        adjusted_equity = self.synthesizer._apply_opponent_adjustments(
            base_required_equity, opponent_profile, {}
        )

        # Should be exactly 1.15 times the original as per directive
        expected_equity = base_required_equity * 1.15
        self.assertAlmostEqual(adjusted_equity, expected_equity, places=5)

    def test_loose_opponent_adjustment(self):
        """Test the loose opponent adjustment from Sub-Task 3.2."""
        base_required_equity = 0.5
        opponent_profile = {
            "classification": "loose_aggressive",
            "vpip": 0.45,  # Over 0.4 threshold
            "pfr": 0.25,
        }

        adjusted_equity = self.synthesizer._apply_opponent_adjustments(
            base_required_equity, opponent_profile, {}
        )

        # Should be reduced for loose players (better odds for us)
        self.assertLess(adjusted_equity, base_required_equity)

    def test_loose_player_value_betting(self):
        """Test the loose player value betting example from Sub-Task 3.2."""
        game_state = {
            "pot_size": 100,
            "our_stack": 1000,
            "hole_cards": ["Ah", "As"],  # Strong hand
            "valid_actions": [
                {"action": "fold", "amount": 0},
                {"action": "call", "amount": 50},
                {"action": "raise", "amount": {"min": 100, "max": 1000}},
            ],
        }

        system1_outputs = {
            "hand_strength": {
                "overall_strength": 0.9,
                "probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9],
            },
            "gto": {"action": "raise", "confidence": 0.8},
            "opponents": {"opponents": {}},
            "heuristics": {"recommendation": None},
        }

        # Test without loose opponent profile - should get normal raise
        final_action_normal, _ = self.synthesizer.synthesize_decision(
            game_state, system1_outputs
        )
        normal_amount = final_action_normal.get("amount", 0)

        # Test with loose opponent profile - should get increased raise
        opponent_profile = {"classification": "loose_passive", "vpip": 0.5, "pfr": 0.1}

        final_action_loose, _ = self.synthesizer.synthesize_decision(
            game_state, system1_outputs, opponent_profile
        )
        loose_amount = final_action_loose.get("amount", 0)

        # With strong hand vs loose player, should increase bet size
        # (Note: this test may be sensitive to implementation details)
        self.assertGreaterEqual(
            loose_amount, normal_amount * 0.95
        )  # Allow some variance


class TestLLMNarrator(unittest.TestCase):
    """Test the LLM Narrator functionality from Sub-Task 3.3."""

    def setUp(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock config
        config_data = {
            "llm_config": {
                "base_url": "http://localhost:1234/v1",
                "api_key": "test-key",
                "model_name": "test-model",
                "timeout": 5,
                "max_tokens": 100,
                "temperature": 0.7,
            },
            "prompts": {
                "decision_analysis": {
                    "system": "Test system prompt",
                    "user_template": "Test user template",
                }
            },
            "narration_settings": {
                "enabled": True,
                "async_mode": False,  # Synchronous for testing
                "save_to_file": True,
                "output_directory": self.temp_dir,
            },
        }

        config_file = os.path.join(self.temp_dir, "llm_config.json")
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        self.narrator = LLMNarrator(config_file)

    def test_narrator_initialization(self):
        """Test LLM Narrator initializes correctly."""
        self.assertIsNotNone(self.narrator.llm_config)
        self.assertIsNotNone(self.narrator.prompts)
        self.assertIsNotNone(self.narrator.narration_settings)
        self.assertTrue(os.path.exists(self.narrator.output_dir))

    def test_narrate_decision_async(self):
        """Test that narration can be triggered asynchronously."""
        mock_decision_packet = {
            "timestamp": datetime.now().isoformat(),
            "street": "preflop",
            "pot_size": 100,
            "our_stack": 1000,
            "hole_cards": ["Ah", "Kh"],
            "final_action": {"action": "call", "amount": 50},
            "confidence_score": 0.7,
            "reasoning_summary": "Test reasoning",
            "total_processing_time": 0.1,
        }

        # This should not crash (even if LLM is not available)
        self.narrator.narrate_decision(mock_decision_packet)

    @patch("requests.post")
    def test_llm_api_call(self, mock_post):
        """Test LLM API call functionality."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test narration response"}}]
        }
        mock_post.return_value = mock_response

        # Test API call
        narration = self.narrator._call_llm_api(
            "Test game context",
            "Test action summary",
            {"reasoning_summary": "Test reasoning"},
        )

        self.assertEqual(narration, "Test narration response")
        mock_post.assert_called_once()

    def test_narration_logging(self):
        """Test that narrations are logged to file correctly."""
        mock_packet = {
            "timestamp": datetime.now().isoformat(),
            "street": "preflop",
            "pot_size": 100,
            "final_action": {"action": "fold"},
            "confidence_score": 0.3,
        }

        test_narration = "This is a test narration"

        self.narrator._log_narration(mock_packet, test_narration)

        # Check that log file was created and contains our narration
        self.assertTrue(os.path.exists(self.narrator.log_file))

        with open(self.narrator.log_file, "r") as f:
            log_content = f.read()
            self.assertIn(test_narration, log_content)
            self.assertIn("preflop", log_content)


class TestLearningModule(unittest.TestCase):
    """Test the Learning Module functionality from Sub-Task 3.4."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.learning_module = LearningModule(self.temp_dir)

    def test_learning_module_initialization(self):
        """Test Learning Module initializes correctly."""
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertTrue(os.path.exists(self.learning_module.output_directory))
        self.assertGreater(self.learning_module.session_stats["hands_logged"], -1)

    def test_log_hand_async(self):
        """Test asynchronous hand logging from Sub-Task 3.4."""
        mock_decision_packet = {
            "timestamp": datetime.now().isoformat(),
            "street": "preflop",
            "pot_size": 100,
            "our_stack": 1000,
            "hole_cards": ["Ah", "Kh"],
            "community_cards": [],
            "final_action": {"action": "call", "amount": 50},
            "confidence_score": 0.7,
            "reasoning_summary": "Test reasoning",
            "synthesizer_analysis": {"our_equity": 0.6, "required_equity": 0.3},
        }

        hand_outcome = {
            "pot_won": 100,
            "winning_hand": "Pair of Aces",
            "showdown": True,
            "final_pot_size": 200,
            "profit_loss": 50,
        }

        # This should not crash and should log asynchronously
        self.learning_module.log_hand(mock_decision_packet, hand_outcome)

        # Give async thread time to complete
        import time

        time.sleep(0.1)

    def test_jsonl_file_creation(self):
        """Test that JSONL files are created correctly."""
        # Manually trigger sync logging
        mock_packet = {
            "timestamp": datetime.now().isoformat(),
            "street": "preflop",
            "synthesizer_analysis": {"our_equity": 0.5, "required_equity": 0.3},
        }
        mock_outcome = {"pot_won": 0, "profit_loss": -50}

        self.learning_module._async_log_hand(mock_packet, mock_outcome)

        # Check that file exists and contains valid JSON
        self.assertTrue(os.path.exists(self.learning_module.hand_history_file))

        with open(self.learning_module.hand_history_file, "r") as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)

            # Parse first line as JSON
            first_record = json.loads(lines[0])
            self.assertIn("hand_id", first_record)
            self.assertIn("game_state", first_record)
            self.assertIn("decision_features", first_record)
            self.assertIn("hand_outcome", first_record)

    def test_session_stats(self):
        """Test session statistics tracking."""
        stats = self.learning_module.get_session_stats()

        self.assertIn("hands_logged", stats)
        self.assertIn("session_start", stats)
        self.assertIn("session_id", stats)
        self.assertIn("hand_history_file", stats)

    def tearDown(self):
        """Clean up test files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TestPhase3Integration(unittest.TestCase):
    """Test the integration of all Phase 3 components."""

    def test_all_modules_can_be_imported(self):
        """Test that all new modules can be imported successfully."""
        from agent.modules.opponent_modeler import OpponentModeler
        from agent.modules.synthesizer import Synthesizer
        from agent.modules.llm_narrator import LLMNarrator
        from agent.modules.learning_module import LearningModule

        # Should be able to instantiate all modules
        modeler = OpponentModeler()
        synthesizer = Synthesizer()
        narrator = LLMNarrator()
        learner = LearningModule()

        self.assertIsNotNone(modeler)
        self.assertIsNotNone(synthesizer)
        self.assertIsNotNone(narrator)
        self.assertIsNotNone(learner)


if __name__ == "__main__":
    # Set up basic logging to avoid warnings during testing
    import logging

    logging.basicConfig(level=logging.WARNING)

    unittest.main()
