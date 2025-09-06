#!/usr/bin/env python3
"""
Main entry point for Project PokerMind - Enhanced CLI (Sub-Task 5.4)

This script provides a comprehensive command-line interface for running 
poker simulations with the PokerMind AI agent.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_logging(no_log: bool = False, log_level: str = "INFO"):
    """Configure logging for the application."""
    if no_log:
        logging.disable(logging.CRITICAL)
        return
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pokermind.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_mock_game_config(args):
    """Create a mock game configuration when PyPokerEngine is not available."""
    print("⚠️ PyPokerEngine not available, running mock simulation")
    
    from agent.agent import PokerMindAgent
    
    # Create mock game simulation
    agent = PokerMindAgent()
    
    # Simulate multiple hands
    results = []
    for hand_num in range(args.max_rounds):
        # Create mock game state
        game_state = {
            'hole_cards': ['As', 'Ks'],
            'community_cards': ['Qh', '7c', '3d'] if hand_num % 3 == 0 else [],
            'pot_size': 150,
            'street': 'flop' if hand_num % 3 == 0 else 'preflop',
            'valid_actions': [
                {'action': 'fold', 'amount': 0},
                {'action': 'call', 'amount': 50},
                {'action': 'raise', 'amount': {'min': 100, 'max': 500}}
            ],
            'our_stack': args.initial_stack,
            'our_seat_id': 1,
            'seats': [
                {'seat_id': 1, 'name': 'PokerMind', 'stack': args.initial_stack},
                {'seat_id': 2, 'name': 'Opponent1', 'stack': 800}
            ],
            'round_count': hand_num,
            'small_blind': 10
        }
        
        try:
            action, decision_packet = agent.cognitive_core.make_decision(game_state)
            results.append({
                'hand': hand_num,
                'action': action,
                'confidence': decision_packet.confidence if hasattr(decision_packet, 'confidence') else 0.7,
                'reasoning': decision_packet.reasoning_summary if hasattr(decision_packet, 'reasoning_summary') else 'Mock simulation'
            })
            
            if (hand_num + 1) % 10 == 0:
                print(f"Completed {hand_num + 1}/{args.max_rounds} hands")
                
        except Exception as e:
            print(f"Error in hand {hand_num}: {e}")
            break
    
    return {
        'hands_played': len(results),
        'avg_confidence': sum(r['confidence'] for r in results) / len(results) if results else 0,
        'actions_taken': [r['action'] for r in results]
    }


def create_game_config(args):
    """Create the poker game configuration."""
    try:
        from pypokerengine.api.game import setup_config, start_poker
        from pypokerengine.players import FishPlayer
        
        config = setup_config(
            max_round=args.max_rounds,
            initial_stack=args.initial_stack,
            small_blind_amount=10
        )
        
        # Register PokerMind agent
        from agent.agent import PokerMindAgent
        pokermind_agent = PokerMindAgent()
        
        # Configure agent style if specified
        if hasattr(pokermind_agent, 'cognitive_core') and hasattr(pokermind_agent.cognitive_core, 'synthesizer'):
            synthesizer = pokermind_agent.cognitive_core.synthesizer
            
            if args.agent_style == 'aggressive':
                synthesizer.aggression = 0.8
                synthesizer.tightness = 0.4
            elif args.agent_style == 'tight':
                synthesizer.aggression = 0.3
                synthesizer.tightness = 0.8
            elif args.agent_style == 'loose':
                synthesizer.aggression = 0.6
                synthesizer.tightness = 0.2
            # 'normal' is default (0.5, 0.5)
        
        config.register_player(name="PokerMind", algorithm=pokermind_agent)
        
        # Register opponents based on num_players
        opponent_types = [FishPlayer] * (args.num_players - 1)
        
        for i, opponent_class in enumerate(opponent_types):
            config.register_player(name=f"Opponent{i+1}", algorithm=opponent_class())
        
        return config, start_poker
        
    except ImportError:
        return None, None


def run_simulation(args):
    """Run the poker simulation with the given arguments."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting PokerMind simulation with {args.num_players} players...")
    logger.info(f"Configuration: {args.max_rounds} rounds, {args.initial_stack} initial stack, {args.agent_style} style")
    
    try:
        config, start_poker = create_game_config(args)
        
        if config is None:
            # Fall back to mock simulation
            result = create_mock_game_config(args)
            print(f"\nMock Simulation Results:")
            print(f"Hands Played: {result['hands_played']}")
            print(f"Average Confidence: {result['avg_confidence']:.2f}")
            return result
        
        # Run the real simulation
        game_result = start_poker(config, verbose=0 if args.no_log else 1)
        
        logger.info("Game completed successfully!")
        logger.info(f"Game result: {game_result}")
        
        return game_result
        
    except Exception as e:
        logger.error(f"Error running simulation: {e}")
        if args.no_log:
            print(f"Error: {e}")
        raise


def run_evaluation(args):
    """Run evaluation mode"""
    print(f"🎯 Running evaluation mode...")
    
    try:
        from evaluation.run_evaluation import PokerEvaluator
        
        evaluator = PokerEvaluator()
        stats = evaluator.run_evaluation(
            num_hands=args.max_rounds, 
            opponent_style=args.eval_opponent or "calling_station"
        )
        
        print(f"\n🏆 Evaluation Results:")
        print(f"   Win Rate: {stats.win_rate:.1%}")
        print(f"   BB/100: {stats.bb_per_100:+.2f}")
        print(f"   Hands: {stats.hands_played:,}")
        
        return stats
        
    except ImportError:
        print("⚠️ Evaluation module not available, running basic simulation instead")
        return run_simulation(args)


def run_performance_profile(args):
    """Run performance profiling mode"""
    print(f"📊 Running performance profiling...")
    
    try:
        from performance_profiler import PerformanceProfiler
        
        profiler = PerformanceProfiler()
        profiler.setup_cognitive_core()
        
        results = profiler.run_performance_analysis(num_iterations=args.max_rounds)
        
        print(f"\n📈 Performance Results:")
        avg_times = [scenario['avg_time'] for scenario in results.values()]
        overall_avg = sum(avg_times) / len(avg_times)
        print(f"   Average Decision Time: {overall_avg*1000:.3f}ms")
        print(f"   Target Met: {'✅ Yes' if overall_avg < 0.01 else '❌ No'}")
        
        return results
        
    except ImportError:
        print("⚠️ Performance profiler not available, running basic simulation instead")
        return run_simulation(args)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PokerMind: Advanced AI Poker Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run default simulation
  %(prog)s --num_players 6 --max_rounds 100  # 6-player, 100 hands
  %(prog)s --agent_style aggressive --no-log # Aggressive style, no logging
  %(prog)s --mode evaluation --max_rounds 1000 # Evaluation mode
  %(prog)s --mode profile --max_rounds 100   # Performance profiling
        """
    )
    
    # Basic simulation parameters
    parser.add_argument(
        '--num_players', 
        type=int, 
        default=2, 
        choices=range(2, 11),
        metavar='N',
        help='Number of players at the table (2-10, default: 2)'
    )
    
    parser.add_argument(
        '--max_rounds', 
        type=int, 
        default=10,
        metavar='N',
        help='Number of hands to play (default: 10)'
    )
    
    parser.add_argument(
        '--initial_stack', 
        type=int, 
        default=1000,
        metavar='N',
        help='Starting stack size for players (default: 1000)'
    )
    
    parser.add_argument(
        '--agent_style',
        choices=['normal', 'aggressive', 'tight', 'loose'],
        default='normal',
        help='PokerMind agent playing style (default: normal)'
    )
    
    # Logging and output control
    parser.add_argument(
        '--no-log',
        action='store_true',
        help='Disable detailed logging for faster runs'
    )
    
    parser.add_argument(
        '--log_level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    # Operating modes
    parser.add_argument(
        '--mode',
        choices=['simulation', 'evaluation', 'profile'],
        default='simulation',
        help='Operating mode (default: simulation)'
    )
    
    # Evaluation-specific options
    parser.add_argument(
        '--eval_opponent',
        choices=['calling_station', 'tight_aggressive', 'loose_aggressive', 'random'],
        default='calling_station',
        help='Opponent type for evaluation mode (default: calling_station)'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (optional)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='PokerMind v1.0 (Phase 5 Final)'
    )
    
    return parser.parse_args()


def print_banner():
    """Print the PokerMind banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                      🃏 POKERMIND v1.0 🃏                     ║
║                Advanced AI Poker Agent                        ║
║                     Phase 5 Final                             ║
╠═══════════════════════════════════════════════════════════════╣
║  Dual-Process Cognitive Architecture                          ║
║  • System 1: GTO Core + Hand Strength + Heuristics          ║
║  • System 2: Confidence-Weighted Synthesis                   ║
║  • Phase 5: Production-Ready with CLI & Evaluation           ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    """Main function to run PokerMind."""
    # Parse arguments
    args = parse_arguments()
    
    # Print banner (unless suppressed)
    if not args.no_log:
        print_banner()
    
    # Setup logging
    setup_logging(args.no_log, args.log_level)
    logger = logging.getLogger(__name__)
    
    # Log startup information
    logger.info(f"PokerMind starting in {args.mode} mode")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Route to appropriate mode
        if args.mode == 'evaluation':
            result = run_evaluation(args)
        elif args.mode == 'profile':
            result = run_performance_profile(args)
        else:  # simulation
            result = run_simulation(args)
        
        # Save output if requested
        if args.output:
            output_path = Path(args.output)
            import json
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"📁 Results saved to: {output_path}")
        
        logger.info("PokerMind completed successfully")
        
    except KeyboardInterrupt:
        print("\n⏹️ Simulation interrupted by user")
        logger.info("Simulation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if not args.no_log:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()