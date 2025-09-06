#!/usr/bin/env python3
"""
Performance profiler for PokerMind decision making.
"""

import sys
import time
import cProfile
import pstats
from pathlib import Path
from io import StringIO

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent.agent import PokerMindAgent


def run_decision_performance_test(num_iterations=100):
    """Run focused performance test on decision making."""
    
    agent = PokerMindAgent()
    
    # Create a representative game state
    game_state = {
        'hole_cards': ['As', 'Kd'],
        'community_cards': ['Qh', 'Jc', '9d'],
        'pot_size': 150,
        'street': 'flop',
        'valid_actions': [
            {'action': 'fold', 'amount': 0},
            {'action': 'call', 'amount': 50},
            {'action': 'raise', 'amount': {'min': 100, 'max': 300}}
        ],
        'our_stack': 1000,
        'our_seat_id': 1,
        'seats': [
            {'seat_id': 1, 'name': 'PokerMind', 'stack': 1000},
            {'seat_id': 2, 'name': 'Opponent1', 'stack': 800}
        ],
        'round_count': 50,
        'small_blind': 10
    }
    
    print(f"Running {num_iterations} decision iterations...")
    
    start_time = time.time()
    
    for i in range(num_iterations):
        # Vary the game state slightly to prevent caching effects
        game_state['pot_size'] = 150 + (i % 50)
        game_state['round_count'] = 50 + i
        
        try:
            action, decision_packet = agent.cognitive_core.make_decision(game_state)
        except Exception as e:
            print(f"Error on iteration {i}: {e}")
            break
        
        if (i + 1) % 25 == 0:
            print(f"Completed {i + 1}/{num_iterations} decisions")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_decision = total_time / num_iterations
    
    print(f"\nPerformance Results:")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per decision: {avg_time_per_decision*1000:.3f} ms")
    print(f"Decisions per second: {num_iterations/total_time:.1f}")
    print(f"Target met (<10ms): {'✅ Yes' if avg_time_per_decision < 0.01 else '❌ No'}")
    
    return {
        'total_time': total_time,
        'avg_time_ms': avg_time_per_decision * 1000,
        'decisions_per_second': num_iterations / total_time
    }


def main():
    """Main profiler function."""
    print("🔍 PokerMind Performance Profiler")
    print("=" * 50)
    
    # Run the focused performance test with cProfile
    print("\n📊 Running focused decision performance test...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    results = run_decision_performance_test(200)
    
    profiler.disable()
    
    # Save profile
    profiler.dump_stats('decision_profile.stats')
    
    # Analyze profile
    print(f"\n📈 Analyzing performance profile...")
    stats = pstats.Stats(profiler)
    
    # Capture output
    s = StringIO()
    stats.sort_stats('cumulative')
    stats.print_stats(20, file=s)
    profile_output = s.getvalue()
    
    print("\nTop 20 functions by cumulative time:")
    print(profile_output)
    
    return results, profile_output


if __name__ == "__main__":
    main()