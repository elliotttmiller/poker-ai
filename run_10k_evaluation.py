#!/usr/bin/env python3
"""
Run 10,000-hand evaluation as specified in the directive.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from evaluation.run_evaluation import PokerEvaluator


def main():
    """Run the 10,000-hand evaluation."""
    print("🎯 Running 10,000-hand evaluation as specified in Phase 5 directive")
    print("=" * 60)
    
    evaluator = PokerEvaluator()
    
    # Run the full evaluation
    stats = evaluator.run_evaluation(num_hands=10000, opponent_style="calling_station")
    
    print(f"\n🏆 Final Results:")
    print(f"   Hands Played: {stats.hands_played:,}")
    print(f"   Win Rate: {stats.win_rate:.1%}")
    print(f"   BB/100: {stats.bb_per_100:+.2f}")
    print(f"   Confidence Interval: {stats.confidence_interval[0]:.1%} - {stats.confidence_interval[1]:.1%}")
    print(f"   Average Pot Size: {stats.avg_pot_size:.1f} chips")
    print(f"   Showdown Rate: {(stats.showdowns/stats.hands_played):.1%}")
    print(f"   Total Profit: {stats.total_profit:+,} chips")
    
    return stats


if __name__ == "__main__":
    main()