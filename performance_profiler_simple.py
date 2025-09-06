#!/usr/bin/env python3
"""
Performance profiler for PokerMind - works without PyPokerEngine.
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

from agent.cognitive_core import CognitiveCore
from agent.modules.gto_core import GTOCore
from agent.modules.hand_strength_estimator import HandStrengthEstimator
from agent.modules.heuristics import HeuristicsEngine
from agent.modules.synthesizer import Synthesizer
from agent.modules.opponent_modeler import OpponentModeler


def run_decision_performance_test(num_iterations=100):
    """Run focused performance test on decision making."""
    
    # Initialize the cognitive core
    cognitive_core = CognitiveCore()
    
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
    decision_times = []
    
    for i in range(num_iterations):
        # Vary the game state slightly to prevent caching effects
        game_state_variant = game_state.copy()
        game_state_variant['pot_size'] = 150 + (i % 50)
        game_state_variant['round_count'] = 50 + i
        
        # Vary hole cards occasionally for realistic testing
        if i % 10 == 0:
            hole_cards_variants = [
                ['As', 'Kd'], ['Qh', 'Qc'], ['7c', '2d'], 
                ['Ah', 'Kh'], ['9s', '8s'], ['2c', '2h']
            ]
            game_state_variant['hole_cards'] = hole_cards_variants[i % len(hole_cards_variants)]
        
        iteration_start = time.time()
        
        try:
            action, decision_packet = cognitive_core.make_decision(game_state_variant)
            iteration_end = time.time()
            decision_times.append((iteration_end - iteration_start) * 1000)  # Convert to ms
            
        except Exception as e:
            print(f"Error on iteration {i}: {e}")
            iteration_end = time.time()
            decision_times.append((iteration_end - iteration_start) * 1000)
        
        if (i + 1) % 25 == 0:
            print(f"Completed {i + 1}/{num_iterations} decisions")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_decision = total_time / num_iterations
    
    # Calculate statistics
    min_time = min(decision_times)
    max_time = max(decision_times)
    avg_time = sum(decision_times) / len(decision_times)
    
    # Sort decision times for percentiles
    sorted_times = sorted(decision_times)
    p50 = sorted_times[len(sorted_times) // 2]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    
    print(f"\nPerformance Results:")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per decision: {avg_time:.3f} ms")
    print(f"Median time per decision: {p50:.3f} ms")
    print(f"95th percentile: {p95:.3f} ms")
    print(f"Min/Max time: {min_time:.3f}ms / {max_time:.3f}ms")
    print(f"Decisions per second: {num_iterations/total_time:.1f}")
    print(f"Target met (<10ms): {'✅ Yes' if avg_time < 10 else '❌ No'}")
    
    return {
        'total_time': total_time,
        'avg_time_ms': avg_time,
        'median_time_ms': p50,
        'p95_time_ms': p95,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'decisions_per_second': num_iterations / total_time,
        'target_met': avg_time < 10
    }


def profile_individual_modules():
    """Profile individual modules to identify bottlenecks."""
    print("\n🔍 Profiling Individual Modules:")
    
    # Test game state
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
    }
    
    modules = {
        'GTO Core': GTOCore(),
        'Hand Strength': HandStrengthEstimator(),
        'Heuristics': HeuristicsEngine(),
        'Opponent Modeler': OpponentModeler(),
        'Synthesizer': Synthesizer()
    }
    
    module_times = {}
    
    for name, module in modules.items():
        times = []
        
        for i in range(50):
            start_time = time.time()
            
            try:
                if name == 'GTO Core':
                    result = module.get_recommendation(game_state)
                elif name == 'Hand Strength':
                    result = module.estimate(game_state)
                elif name == 'Heuristics':
                    result = module.get_recommendation(game_state)
                elif name == 'Opponent Modeler':
                    result = module.get_opponent_analysis(game_state)
                elif name == 'Synthesizer':
                    # Need mock system1 inputs for synthesizer
                    mock_inputs = {
                        'gto': {'action': 'call', 'confidence': 0.7},
                        'hand_strength': {'strength': 0.6, 'confidence': 0.8},
                        'heuristics': {'recommendation': None, 'confidence': 0.0},
                        'opponents': {'recommendation': 'fold', 'confidence': 0.3}
                    }
                    result = module.synthesize_decision(game_state, mock_inputs)
                
            except Exception as e:
                print(f"Error testing {name}: {e}")
                result = None
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = sum(times) / len(times)
        module_times[name] = avg_time
        print(f"  {name:15}: {avg_time:6.3f}ms avg")
    
    return module_times


def main():
    """Main profiler function."""
    print("🔍 PokerMind Performance Profiler")
    print("=" * 50)
    
    # Profile individual modules first
    module_times = profile_individual_modules()
    
    # Run the focused performance test with cProfile
    print(f"\n📊 Running integrated decision performance test...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    results = run_decision_performance_test(200)
    
    profiler.disable()
    
    # Save and analyze profile
    profiler.dump_stats('decision_profile.stats')
    
    print(f"\n📈 Analyzing performance profile...")
    stats = pstats.Stats(profiler)
    
    # Get top functions by cumulative time
    s = StringIO()
    stats.sort_stats('cumulative')
    stats.print_stats(15)  # Print to stdout
    
    # Get top functions by total time
    print(f"\n📈 Top functions by total time:")
    stats.sort_stats('tottime')
    stats.print_stats(10)
    
    return results, module_times


if __name__ == "__main__":
    main()