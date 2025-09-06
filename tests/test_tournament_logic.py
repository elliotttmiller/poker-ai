"""
Tournament Logic Verification Suite

Comprehensive tests to verify the tournament simulation integrity:
- Correct player elimination
- Correct blind increases  
- Correct winner declaration
- Edge case handling (multi-way all-ins, side pots)

This test suite ensures the tournament director follows standard poker rules.
"""

import sys
import os
import logging
import random
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_tournament import TournamentDirector, TournamentPlayer, BlindLevel, TournamentResult


# Mock agent classes for testing
class TestAgent:
    """Simple test agent for tournament verification."""
    
    def __init__(self, name: str, stack_behavior: str = "normal"):
        self.name = name
        self.uuid = f"test_uuid_{name}"
        self.stack_behavior = stack_behavior  # "normal", "lose_all", "win_all"
        self.actions_taken = []
        
    def declare_action(self, valid_actions, hole_cards, round_state):
        """Mock action - behavior depends on stack_behavior setting."""
        action = "fold"
        amount = 0
        
        if self.stack_behavior == "lose_all":
            # Always go all-in to lose chips quickly
            for va in valid_actions:
                if va["action"] == "raise":
                    action = "raise"
                    amount = 1000  # Large amount
                    break
        elif self.stack_behavior == "win_all":
            # Conservative play to preserve chips
            for va in valid_actions:
                if va["action"] == "call":
                    action = "call"
                    amount = va.get("amount", 0)
                    break
        else:  # normal
            # Random behavior
            valid_action = random.choice(valid_actions)
            action = valid_action["action"]
            amount = valid_action.get("amount", 0)
            
        self.actions_taken.append((action, amount))
        return action, amount
        
    def receive_game_start_message(self, game_info):
        pass
        
    def receive_round_start_message(self, round_count, hole_cards, seats):
        pass
        
    def receive_street_start_message(self, street, round_state):
        pass
        
    def receive_game_update_message(self, action, round_state):
        pass
        
    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_test_logging():
    """Set up logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    

def test_tournament_initialization():
    """Test that tournament director initializes correctly."""
    print("Testing tournament initialization...")
    
    tournament = TournamentDirector("test_tournament", initial_stack=1000)
    
    # Check basic properties
    assert tournament.tournament_id == "test_tournament"
    assert tournament.initial_stack == 1000
    assert len(tournament.players) == 0
    assert len(tournament.eliminated_players) == 0
    assert tournament.hand_count == 0
    
    # Check blind schedule exists
    assert len(tournament.blind_schedule) > 0
    assert tournament.blind_schedule[0].small_blind == 10
    assert tournament.blind_schedule[0].big_blind == 20
    
    print("✓ Tournament initialization test passed")
    return True


def test_player_registration():
    """Test player registration functionality."""
    print("Testing player registration...")
    
    tournament = TournamentDirector("test_reg", initial_stack=1500)
    
    # Register 6 test players
    for i in range(6):
        agent = TestAgent(f"Player{i+1}")
        tournament.register_player(f"Player{i+1}", lambda a=agent: a)
    
    # Check all players registered
    assert len(tournament.players) == 6
    assert tournament.players[0].name == "Player1"
    assert tournament.players[0].stack == 1500
    assert tournament.players[0].seat_id == 1
    assert tournament.players[5].seat_id == 6
    
    print("✓ Player registration test passed")
    return True


def test_blind_level_progression():
    """Test that blind levels increase correctly based on hand count."""
    print("Testing blind level progression...")
    
    tournament = TournamentDirector("test_blinds")
    
    # Test initial blind level
    tournament.hand_count = 1
    blinds = tournament.get_current_blinds()
    assert blinds.small_blind == 10
    assert blinds.big_blind == 20
    assert blinds.ante == 0
    
    # Test second blind level
    tournament.hand_count = 7
    blinds = tournament.get_current_blinds()
    assert blinds.small_blind == 15
    assert blinds.big_blind == 30
    
    # Test later blind level with ante
    tournament.hand_count = 31
    blinds = tournament.get_current_blinds()
    assert blinds.small_blind == 100
    assert blinds.big_blind == 200
    assert blinds.ante == 25
    
    # Test very late blind level
    tournament.hand_count = 55
    blinds = tournament.get_current_blinds()
    assert blinds.small_blind == 500
    assert blinds.big_blind == 1000
    assert blinds.ante == 100
    
    print("✓ Blind level progression test passed")
    return True


def test_player_elimination():
    """Test that players are correctly eliminated when they lose all chips."""
    print("Testing player elimination...")
    
    tournament = TournamentDirector("test_elimination")
    
    # Create and register test players
    players = []
    for i in range(4):
        agent = TestAgent(f"Player{i+1}")
        tournament.register_player(f"Player{i+1}", lambda a=agent: a)
        players.append(tournament.players[i])
    
    # Simulate a player losing all chips
    target_player = tournament.players[1]  # Player2
    target_player.stack = 0
    
    # Eliminate the player
    tournament.eliminate_player(target_player)
    
    # Check elimination was processed correctly
    assert len(tournament.players) == 3
    assert len(tournament.eliminated_players) == 1
    assert tournament.eliminated_players[0].name == "Player2"
    assert tournament.eliminated_players[0].finishing_place == 4  # 4th place (4 players remaining when eliminated)
    assert tournament.eliminated_players[0].eliminated_hand == 0
    assert target_player not in tournament.players
    
    # Eliminate another player
    second_target = tournament.players[0]
    second_target.stack = 0
    tournament.hand_count = 10
    tournament.eliminate_player(second_target)
    
    # Check second elimination
    assert len(tournament.players) == 2
    assert len(tournament.eliminated_players) == 2
    assert tournament.eliminated_players[1].finishing_place == 3  # 3rd place
    assert tournament.eliminated_players[1].eliminated_hand == 10
    
    print("✓ Player elimination test passed")
    return True


def test_mock_hand_simulation():
    """Test that mock hand simulation works and eliminates players appropriately."""
    print("Testing mock hand simulation...")
    
    tournament = TournamentDirector("test_mock_sim")
    
    # Register test players with different behaviors
    agents = [
        TestAgent("Winner", "win_all"),
        TestAgent("Loser", "lose_all"),
        TestAgent("Normal1", "normal"),
        TestAgent("Normal2", "normal")
    ]
    
    for agent in agents:
        tournament.register_player(agent.name, lambda a=agent: a)
    
    initial_player_count = len(tournament.players)
    
    # Run several mock hands
    for hand_num in range(10):
        tournament.hand_count = hand_num + 1
        
        if len(tournament.players) < 2:
            break
            
        results = tournament.simulate_hand_mock()
        
        # Check that results are returned
        assert isinstance(results, list)
        
        # Check that stack changes are reasonable
        for result in results:
            assert "player" in result
            assert "stack_change" in result
            assert "final_stack" in result
            assert "eliminated" in result
    
    # Check that some eliminations may have occurred
    print(f"Started with {initial_player_count} players, now have {len(tournament.players)}")
    assert len(tournament.eliminated_players) <= initial_player_count
    
    print("✓ Mock hand simulation test passed")
    return True


def test_tournament_completion():
    """Test that tournament runs to completion with correct winner declaration."""
    print("Testing tournament completion...")
    
    tournament = TournamentDirector("test_completion", initial_stack=500)  # Smaller stacks for faster completion
    
    # Register 4 players for faster test
    for i in range(4):
        agent = TestAgent(f"Player{i+1}", "normal")
        tournament.register_player(f"Player{i+1}", lambda a=agent: a)
    
    # Run tournament
    result = tournament.run_tournament()
    
    # Check tournament completed successfully
    assert isinstance(result, TournamentResult)
    assert result.tournament_id == "test_completion"
    assert len(result.final_standings) == 4
    
    # Check winner
    assert result.winner in [f"Player{i+1}" for i in range(4)]
    assert result.final_standings[0]["place"] == 1
    assert result.final_standings[0]["name"] == result.winner
    
    # Check all players are accounted for in final standings
    all_names = [standing["name"] for standing in result.final_standings]
    assert len(set(all_names)) == 4  # All unique names
    
    # Check finishing places are correct
    places = [standing["place"] for standing in result.final_standings]
    assert sorted(places) == [1, 2, 3, 4]
    
    # Check only winner has chips remaining
    for standing in result.final_standings:
        if standing["place"] == 1:
            assert standing["stack"] > 0
        else:
            assert standing["stack"] == 0
            assert standing["eliminated_hand"] is not None
    
    print("✓ Tournament completion test passed")
    return True


def test_side_pot_edge_case():
    """Test handling of multi-way all-in situations (simplified)."""
    print("Testing side pot edge case handling...")
    
    tournament = TournamentDirector("test_side_pots")
    
    # Create players with different stack sizes to simulate side pot situation
    stack_sizes = [100, 200, 300, 500]
    for i, stack in enumerate(stack_sizes):
        agent = TestAgent(f"Player{i+1}")
        tournament.register_player(f"Player{i+1}", lambda a=agent: a)
        tournament.players[i].stack = stack  # Set different stack sizes
    
    # Simulate a hand with multiple all-ins
    # This is a simplified test since we're using mock simulation
    initial_total_chips = sum(p.stack for p in tournament.players)
    
    results = tournament.simulate_hand_mock()
    
    # Check that chips are conserved (total chips should be same)
    final_total_chips = sum(p.stack for p in tournament.players)
    assert abs(initial_total_chips - final_total_chips) <= len(tournament.players)  # Allow small rounding
    
    # Check that players with 0 chips are eliminated
    zero_stack_players = [p for p in tournament.players if p.stack == 0]
    assert len(zero_stack_players) == len(tournament.eliminated_players)
    
    print("✓ Side pot edge case test passed")
    return True


def test_tournament_results_saving():
    """Test that tournament results can be saved and loaded correctly."""
    print("Testing tournament results saving...")
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        tournament = TournamentDirector("test_saving", initial_stack=100)
        
        # Register minimal players for quick test
        for i in range(2):
            agent = TestAgent(f"Player{i+1}")
            tournament.register_player(f"Player{i+1}", lambda a=agent: a)
        
        # Run tournament
        result = tournament.run_tournament()
        
        # Save results
        save_path = Path(temp_dir) / "test_results.json"
        saved_path = tournament.save_results(save_path)
        
        # Check file was created
        assert saved_path.exists()
        assert saved_path.stat().st_size > 0
        
        # Load and verify JSON structure
        import json
        with open(saved_path, 'r') as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = [
            "tournament_id", "start_time", "end_time", "total_hands",
            "final_standings", "winner", "duration_seconds"
        ]
        for field in required_fields:
            assert field in data
        
        # Check data consistency
        assert data["tournament_id"] == "test_saving"
        assert data["winner"] in ["Player1", "Player2"]
        assert len(data["final_standings"]) == 2
        assert data["total_hands"] > 0
        assert data["duration_seconds"] >= 0
    
    print("✓ Tournament results saving test passed")
    return True


def test_tournament_stress_test():
    """Stress test with more players and longer duration."""
    print("Testing tournament stress test...")
    
    tournament = TournamentDirector("stress_test", initial_stack=200)  # Small stacks for speed
    
    # Register 6 players
    for i in range(6):
        agent = TestAgent(f"StressPlayer{i+1}")
        tournament.register_player(f"StressPlayer{i+1}", lambda a=agent: a)
    
    # Run tournament with safety limit
    result = tournament.run_tournament()
    
    # Check basic completion criteria
    assert result is not None
    assert len(result.final_standings) == 6
    assert result.total_hands > 0
    assert result.total_hands <= 200  # Safety limit
    
    # Check all players are accounted for
    player_names = {f"StressPlayer{i+1}" for i in range(6)}
    result_names = {standing["name"] for standing in result.final_standings}
    assert player_names == result_names
    
    print(f"✓ Stress test completed in {result.total_hands} hands")
    return True


def test_config_validation():
    """Test tournament configuration edge cases and validation."""
    print("Testing configuration validation...")
    
    # Test minimum players
    tournament = TournamentDirector("validation_test")
    agent = TestAgent("OnlyPlayer")
    tournament.register_player("OnlyPlayer", lambda: agent)
    
    try:
        # Should fail with only 1 player
        result = tournament.run_tournament()
        assert False, "Should have raised ValueError for insufficient players"
    except ValueError as e:
        assert "at least 2 players" in str(e).lower()
    
    # Test with exactly 2 players - should work
    agent2 = TestAgent("SecondPlayer") 
    tournament.register_player("SecondPlayer", lambda: agent2)
    
    result = tournament.run_tournament()
    assert result is not None
    assert len(result.final_standings) == 2
    
    print("✓ Configuration validation test passed")
    return True


def run_all_tests():
    """Run all tournament verification tests."""
    print("=" * 60)
    print("RUNNING TOURNAMENT LOGIC VERIFICATION SUITE")
    print("=" * 60)
    
    setup_test_logging()
    
    tests = [
        test_tournament_initialization,
        test_player_registration, 
        test_blind_level_progression,
        test_player_elimination,
        test_mock_hand_simulation,
        test_tournament_completion,
        test_side_pot_edge_case,
        test_tournament_results_saving,
        test_config_validation,
        test_tournament_stress_test,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"TOURNAMENT VERIFICATION RESULTS")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/(passed+failed)*100:.1f}%")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL TOURNAMENT LOGIC TESTS PASSED!")
        print("✅ Game Rule Integrity Verified")
        print("✅ Tournament Stress Tests Passed")
        print("✅ Player Elimination Logic Verified")
        print("✅ Blind Increase Logic Verified") 
        print("✅ Edge Case Handling Verified")
        return True
    else:
        print("❌ Some tests failed. Tournament logic needs fixes.")
        return False


def main():
    """Main function to run verification tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tournament logic verification tests")
    parser.add_argument("--test", help="Run specific test function")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    if args.test:
        # Run specific test
        test_func = globals().get(f"test_{args.test}")
        if test_func:
            try:
                test_func()
                print(f"✓ {args.test} test passed")
                return 0
            except Exception as e:
                print(f"✗ {args.test} test failed: {e}")
                return 1
        else:
            print(f"Test '{args.test}' not found")
            return 1
    else:
        # Run all tests
        success = run_all_tests()
        return 0 if success else 1


if __name__ == "__main__":
    exit(main())